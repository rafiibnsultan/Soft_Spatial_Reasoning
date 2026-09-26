"""Qwen3-VL training forward with soft-thinking embeddings.

Mirrors the model's own forward, but replaces the response-token embeddings with the
Gumbel-reparameterised soft blend rebuilt from the rollout records:

    inputs_embeds = sum_k softmax(gumbels / tau)_k * E[topk_ids_k]   (think steps)
    inputs_embeds = masked_scatter(image features)
    position_ids  = model.compute_3d_position_ids(...)
    hidden        = language_model(inputs_embeds, position_ids, deepstack features)
    logits        = lm_head(hidden)
    log_probs     = gumbel reparam (think) | cross-entropy (answer)

`gumbel_temperature` is either a scalar or a per-step [B, T, 1] tensor produced by the
AdaptSoft controller.
"""
from __future__ import annotations

import torch

from .model_utils import unwrap_model
from .soft_logprobs import gumbel_reparam_logprobs


def build_mm_token_type_ids(input_ids: torch.Tensor, config) -> torch.Tensor:
    """Per-token type ids (0=text, 1=image, 2=video), required by compute_3d_position_ids."""
    mm = torch.zeros_like(input_ids, dtype=torch.int32)
    mm[input_ids == config.image_token_id] = 1
    video_id = getattr(config, "video_token_id", None)
    if video_id is not None:
        mm[input_ids == video_id] = 2
    return mm


def soft_forward(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rollout_topk_ids: torch.Tensor,
    rollout_topk_gumbels: torch.Tensor,
    gumbel_temperature,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    return_logits: bool = False,
    embed_dtype: torch.dtype | None = None,
):
    """Return per-token log-probs [B, T] (gumbel for think tokens, CE for answer/prompt).

    The caller applies the response mask; prompt-position log-probs are meaningless and ignored.
    """
    base = unwrap_model(model)
    core = base.model
    core.rope_deltas = None
    embed_module = core.get_input_embeddings()
    config = base.config


    _V = getattr(embed_module, "num_embeddings", None) or embed_module.weight.shape[0]
    _mn, _mx = int(rollout_topk_ids.min()), int(rollout_topk_ids.max())
    assert 0 <= _mn and _mx < _V, (
        f"[soft] rollout_topk_ids out of vocab range [0,{_V}): min={_mn} max={_mx} "
        f"(input_ids range [{int(input_ids.min())},{int(input_ids.max())}])"
    )


    finished_mask = (rollout_topk_ids[..., 1:] == 0).all(dim=-1)
    masked = rollout_topk_gumbels.clone()
    tail = masked[..., 1:]


    tail[finished_mask] = -1.0e4
    masked = torch.cat([masked[..., :1], tail], dim=-1)
    weights = torch.softmax(masked / gumbel_temperature, dim=-1)
    topk_emb = embed_module(rollout_topk_ids)
    inputs_embeds = (weights.unsqueeze(-1).to(topk_emb.dtype) * topk_emb).sum(dim=-2)


    if embed_dtype is not None and inputs_embeds.dtype != embed_dtype:
        inputs_embeds = inputs_embeds.to(embed_dtype)


    visual_pos_masks = None
    deepstack_visual_embeds = None
    if pixel_values is not None:
        img_out = core.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        image_embeds = torch.cat(img_out.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        deepstack_visual_embeds = img_out.deepstack_features

        image_mask, _ = core.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]


        _nvt = int(visual_pos_masks.sum())
        assert image_embeds.shape[0] == _nvt, (
            f"[soft] vision count mismatch: image_embeds {image_embeds.shape[0]} != image tokens {_nvt} "
            f"(image_token_id={config.image_token_id}, input_ids max={int(input_ids.max())})"
        )
        for _l, _de in enumerate(deepstack_visual_embeds or []):
            assert _de.shape[0] == _nvt, f"[soft] deepstack[{_l}] rows {_de.shape[0]} != vis tokens {_nvt}"


    mm_token_type_ids = build_mm_token_type_ids(input_ids, config)
    position_ids = core.compute_3d_position_ids(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
    )


    outputs = core.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        use_cache=False,
    )
    hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]


    labels_bt = torch.roll(input_ids, shifts=-1, dims=1)
    ids_bt = torch.roll(rollout_topk_ids, shifts=-1, dims=1)
    gum_bt = torch.roll(rollout_topk_gumbels, shifts=-1, dims=1)
    K = rollout_topk_ids.size(-1)
    B, T = input_ids.shape

    if attention_mask is not None:
        flat_valid = attention_mask.reshape(-1).bool()
        hidden_v = hidden_states.reshape(-1, hidden_states.size(-1))[flat_valid]
        logits_v = base.lm_head(hidden_v)
        logp_v = gumbel_reparam_logprobs(
            logits=logits_v,
            rollout_topk_ids=ids_bt.reshape(-1, K)[flat_valid],
            rollout_topk_gumbels=gum_bt.reshape(-1, K)[flat_valid],
            labels=labels_bt.reshape(-1)[flat_valid],
        )


        log_probs = logp_v.new_zeros(B * T)
        log_probs[flat_valid] = logp_v
        log_probs = log_probs.view(B, T)


        align_parts = None
        if torch.is_grad_enabled() and logits_v.requires_grad:
            _idx_v = ids_bt.reshape(-1, K)[flat_valid]
            align_parts = {
                "hidden": hidden_v,
                "logits": logits_v,
                "target": labels_bt.reshape(-1)[flat_valid],
                "index": _idx_v,


                "think": ~(_idx_v[:, 1:] == 0).all(dim=-1),

                "seq": (torch.arange(B, device=input_ids.device).repeat_interleave(T))[flat_valid],
            }


        if return_logits:


            return log_probs, logits_v, flat_valid, align_parts
        return log_probs


    logits = base.lm_head(hidden_states)
    log_probs = gumbel_reparam_logprobs(
        logits=logits, rollout_topk_ids=ids_bt, rollout_topk_gumbels=gum_bt, labels=labels_bt,
    )
    if return_logits:

        return log_probs, logits, None, None
    return log_probs
