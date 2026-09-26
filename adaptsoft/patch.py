"""AdaptSoft integration with verl's FSDP engine.

Importing this module patches `FSDPEngineWithLMHead` so that

  * the training forward rebuilds the soft embeddings from the rollout records and recomputes the
    per-step temperature from the controller (`soft_forward_step`),
  * the controller is attached to the model, trained by its own optimizer, synchronised across
    ranks, checkpointed, and pushed to the rollout engine on weight sync,
  * the controller's gradient comes from the per-step alignment objective only.

Environment variables:
    ADAPTSOFT_LR          controller learning rate (default 1e-3)
    ADAPTSOFT_H_MEAN      mean used to whiten the normalised top-k entropy
    ADAPTSOFT_H_STD       standard deviation used to whiten it
    ADAPTSOFT_INIT_FROM   path to a controller state dict to load instead of the checkpoint's
"""
from __future__ import annotations

import os

import torch

import verl.utils.torch_functional as verl_F
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_device_name
from verl.utils.model import extract_multi_modal_inputs
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

from .controller import AdaptSoftController
from .grad_align import per_step_alignment, val_grad_state
from .model_utils import unwrap_model
from .soft_forward import soft_forward

LR = float(os.environ.get("ADAPTSOFT_LR", "1e-3"))
H_MEAN = float(os.environ.get("ADAPTSOFT_H_MEAN", "0.0"))
H_STD = float(os.environ.get("ADAPTSOFT_H_STD", "1.0"))
INIT_FROM = os.environ.get("ADAPTSOFT_INIT_FROM", "")

PROJ_DIM = 8
TAU_BASE = 0.5
TAU_DELTA = 0.4
B_INIT = 0.0
CONTROLLER_SEED = 20260824

ALIGN_VAL_FRAC = 0.25
ALIGN_CHUNK = 512
ALIGN_TARGET = 1e-3
ALIGN_SCALE_CAP = 1e6

_ORIG_OPT_STEP = FSDPEngineWithLMHead.optimizer_step
_ORIG_OPT_ZERO = FSDPEngineWithLMHead.optimizer_zero_grad
_ORIG_FORWARD_STEP = FSDPEngineWithLMHead.forward_step
_ORIG_GET_PER_TENSOR = FSDPEngineWithLMHead.get_per_tensor_param
_ORIG_SAVE_CKPT = FSDPEngineWithLMHead.save_checkpoint
_ORIG_LOAD_CKPT = FSDPEngineWithLMHead.load_checkpoint
_MODEL_FWD_PATCHED = {"done": False}


# --------------------------------------------------------------------------------------- controller


def _ensure_controller(engine):
    """Attach the controller to the model and, for the actor, give it its own optimizer.

    The controller is attached with object.__setattr__ so it is not part of
    `engine.module.parameters()`: it stays a replicated fp32 module outside FSDP, and verl's
    optimizer and gradient clipping never see it. All ranks initialise it identically from a fixed
    seed; the patched optimizer step all-reduces its gradients so they stay identical.
    """
    if getattr(engine, "_controller_ready", False):
        return
    base = unwrap_model(engine.module)
    hidden_size = int(getattr(base.config, "hidden_size", 0)) or int(base.lm_head.in_features)
    if getattr(base, "adaptsoft_controller", None) is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(CONTROLLER_SEED)
            controller = AdaptSoftController(
                hidden_size=hidden_size, proj_dim=PROJ_DIM, h_mean=H_MEAN, h_std=H_STD,
                tau_base=TAU_BASE, tau_delta=TAU_DELTA, b_init=B_INIT,
            )
        object.__setattr__(base, "adaptsoft_controller", controller.to(get_device_id()).float())
    engine._controller = base.adaptsoft_controller

    if INIT_FROM and not getattr(engine, "_controller_loaded_external", False):
        engine._controller_loaded_external = True
        sd = torch.load(os.path.expanduser(INIT_FROM), map_location="cpu")
        dev = next(engine._controller.parameters()).device
        engine._controller.load_state_dict({k: v.to(dev) for k, v in sd.items()})

    try:
        engine._controller_dp_group = engine.get_data_parallel_group()
    except Exception:
        engine._controller_dp_group = None

    if getattr(engine, "optimizer", None) is not None:
        engine._controller_optimizer = torch.optim.AdamW(
            [{"params": list(engine._controller.parameters()), "lr": LR}]
        )
    else:
        engine._controller_optimizer = None
    engine._controller_ready = True


def _build_hook(orig_build):
    def _patched(self, *args, **kwargs):
        out = orig_build(self, *args, **kwargs)
        try:
            _ensure_controller(self)
        except Exception:
            pass
        return out
    return _patched


# ------------------------------------------------------------------------------------ model forward


def _patch_model_forward(fsdp_module):
    """Run the soft forward through the model's own forward so FSDP unshards its parameters."""
    if _MODEL_FWD_PATCHED["done"]:
        return
    cls = type(unwrap_model(fsdp_module))
    _orig = cls.forward

    def _soft_or_orig(self, *args, rollout_topk_ids=None, rollout_topk_gumbels=None,
                      gumbel_temperature=0.1, soft_embed_dtype=None, **kw):
        if rollout_topk_ids is None:
            return _orig(self, *args, **kw)
        logp, logits_v, valid_mask, align_parts = soft_forward(
            self, kw.get("input_ids"), kw.get("attention_mask"),
            rollout_topk_ids, rollout_topk_gumbels, gumbel_temperature,
            pixel_values=kw.get("pixel_values"), image_grid_thw=kw.get("image_grid_thw"),
            return_logits=True, embed_dtype=soft_embed_dtype,
        )
        # A dict (not a namespace): FSDP2 pytree-flattens the output to register backward hooks.
        return {"log_probs": logp, "logits_valid": logits_v, "valid_mask": valid_mask,
                "align_parts": align_parts}

    cls.forward = _soft_or_orig
    _MODEL_FWD_PATCHED["done"] = True


def _is_soft(micro_batch) -> bool:
    try:
        return "rollout_topk_ids" in micro_batch.keys()
    except Exception:
        return False


def _to_nested_jagged(padded_bt, cu_seqlens):
    lens = cu_seqlens.diff()
    flat = torch.cat([padded_bt[b, : int(lens[b])] for b in range(lens.shape[0])])
    return torch.nested.nested_tensor_from_jagged(flat, cu_seqlens)


# ------------------------------------------------------------------------------------ forward step


def soft_forward_step(self, micro_batch, loss_function, forward_only):
    _ensure_controller(self)
    if not _is_soft(micro_batch):
        return _ORIG_FORWARD_STEP(self, micro_batch, loss_function, forward_only)

    micro_batch = micro_batch.to(get_device_id())
    device_name = get_device_name()
    calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)

    ids_nested = micro_batch["input_ids"]
    cu = ids_nested.offsets()
    lens = cu.diff()
    B, Tmax = lens.shape[0], int(lens.max().item())
    pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)

    input_ids = torch.nested.to_padded_tensor(ids_nested, padding=pad_token_id, output_size=(B, Tmax))
    attn = (torch.arange(Tmax, device=input_ids.device)[None, :] < lens[:, None]).long()

    # Records are response-aligned [B, response_length, K]; scatter them into the response region of
    # the full sequence and leave a one-hot spine everywhere else.
    resp_ids = micro_batch["rollout_topk_ids"]
    resp_gum = micro_batch["rollout_topk_gumbels"]
    K = resp_ids.size(-1)
    topk_ids = torch.zeros(B, Tmax, K, dtype=torch.long, device=input_ids.device)
    topk_ids[..., 0] = input_ids
    topk_gum = torch.zeros(B, Tmax, K, dtype=resp_gum.dtype, device=input_ids.device)
    rec_rows = (resp_ids.abs().sum(-1) != 0) | (resp_gum.abs().sum(-1) != 0)

    resp_tau = micro_batch["rollout_topk_tau"]
    resp_u = micro_batch["rollout_controller_u"]
    resp_h = micro_batch["rollout_controller_h"]
    topk_tau = torch.ones(B, Tmax, 1, dtype=torch.float32, device=input_ids.device)
    topk_u = torch.zeros(B, Tmax, 1, dtype=torch.float32, device=input_ids.device)
    topk_h = torch.zeros(B, Tmax, 1, dtype=torch.float32, device=input_ids.device)
    topk_z = torch.zeros(B, Tmax, PROJ_DIM, dtype=torch.float32, device=input_ids.device)
    topk_rec = torch.zeros(B, Tmax, 1, dtype=torch.bool, device=input_ids.device)
    # The projection rides the spare carrier slots: z[:n_u] in rollout_controller_u[1:], the rest in
    # rollout_controller_h after the entropy slot.
    n_u = min(PROJ_DIM, resp_u.shape[-1] - 1)

    for i in range(B):
        Li = int(lens[i])
        ri = int(rec_rows[i].sum())
        if ri <= 0:
            continue
        topk_ids[i, Li - ri: Li] = resp_ids[i, :ri]
        topk_gum[i, Li - ri: Li] = resp_gum[i, :ri]
        topk_tau[i, Li - ri: Li, 0] = resp_tau[i, :ri, 0]
        topk_rec[i, Li - ri: Li, 0] = True
        topk_u[i, Li - ri: Li, 0] = resp_u[i, :ri, 0]
        topk_h[i, Li - ri: Li, :] = resp_h[i, :ri, :1]
        if n_u > 0:
            topk_z[i, Li - ri: Li, :n_u] = resp_u[i, :ri, 1: 1 + n_u]
        if PROJ_DIM > n_u:
            rest = PROJ_DIM - n_u
            topk_z[i, Li - ri: Li, n_u:] = resp_h[i, :ri, 1: 1 + rest]

    if not getattr(FSDPEngineWithLMHead, "_record_alignment_checked", False):
        FSDPEngineWithLMHead._record_alignment_checked = True
        hit = tot = 0
        for i in range(B):
            Li, ri = int(lens[i]), int(rec_rows[i].sum())
            if ri > 0:
                hit += int((resp_ids[i, :ri, 0] == input_ids[i, Li - ri: Li]).sum())
                tot += ri
        if tot and hit / tot < 0.99:
            raise RuntimeError(
                f"rollout records are misaligned with the response positions ({hit}/{tot} match)")

    mmi = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
    pixel_values = mmi.get("pixel_values")
    image_grid_thw = mmi.get("image_grid_thw")

    _patch_model_forward(self.module)
    from contextlib import nullcontext
    adt = getattr(self, "_autocast_dtype", torch.bfloat16)
    autocast = nullcontext() if adt == torch.float32 else torch.autocast(device_type=device_name, dtype=adt)

    # Rebuild the temperature from the recorded controller inputs. The value equals the recorded
    # temperature, so the reconstruction and the importance ratio are unchanged, while the gradient
    # reaches the controller parameters.
    controller = self._controller
    mu = controller.mu(topk_z.float(), topk_h.float())
    u_eff = topk_u.float() + (mu - mu.detach())
    tau_new = controller.tau_from_u(u_eff).to(topk_tau.dtype)
    tau_blend = torch.where(topk_rec, tau_new, topk_tau)

    with autocast:
        out = self.module(
            input_ids=input_ids, attention_mask=attn,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            rollout_topk_ids=topk_ids, rollout_topk_gumbels=topk_gum,
            gumbel_temperature=tau_blend,
            use_cache=False,
            soft_embed_dtype=adt,
        )
        log_probs_bt = out["log_probs"]
        logits_valid, valid_mask = out["logits_valid"], out["valid_mask"]

        model_output = {"log_probs": _to_nested_jagged(log_probs_bt, cu)}
        if calculate_entropy:
            with torch.no_grad():
                ent_v = verl_F.entropy_from_logits_with_chunking(logits_valid.detach())
            ent_bt = torch.zeros(log_probs_bt.numel(), dtype=ent_v.dtype, device=ent_v.device)
            if valid_mask is not None:
                ent_bt[valid_mask] = ent_v
            else:
                ent_bt = ent_v
            ent_bt = ent_bt.view(*log_probs_bt.shape)
            model_output["entropy"] = _to_nested_jagged(ent_bt, cu)

        if loss_function is not None:
            loss, metrics = loss_function(
                model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
            )
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss, metrics = torch.tensor(1.0, device=device_name), {}

    align_parts = out.get("align_parts")
    if align_parts is not None and (not torch.is_grad_enabled() or not align_parts["logits"].requires_grad):
        align_parts = None
    if align_parts is not None:
        _alignment_update(self, loss, align_parts, tau_blend, topk_ids)

    return loss, {"model_output": model_output, "loss": loss.detach().item(), "metrics": metrics}


# --------------------------------------------------------------------------------------- alignment


def _alignment_update(self, loss, align_parts, tau_blend, topk_ids):
    """Accumulate the controller's alignment gradient for this micro-batch.

    The logit gradient comes from the training loss itself and is detached, so the controller's
    gradient flows through the hidden states only. `torch.autograd.grad` is used throughout, so no
    gradient is written into the language-model parameters.
    """
    delta = torch.autograd.grad(loss, align_parts["logits"], retain_graph=True)[0].detach()

    seq, think = align_parts["seq"], align_parts["think"]
    B = int(seq.max()) + 1
    n_ref = max(1, min(B - 1, int(round(B * ALIGN_VAL_FRAC))))
    is_ref = (seq >= (B - n_ref)) & think
    is_train = (~(seq >= (B - n_ref))) & think

    # Every rank must agree on whether to run: the backward below triggers FSDP parameter
    # all-gathers, and a rank that skips would desynchronise the collectives.
    have = 1.0 if (int(is_train.sum()) and int(is_ref.sum())) else 0.0
    dp = getattr(self, "_controller_dp_group", None)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        hv = torch.tensor([have], device=align_parts["hidden"].device)
        torch.distributed.all_reduce(hv, op=torch.distributed.ReduceOp.MIN, group=dp)
        have = float(hv.item())
    if have <= 0.0:
        return

    state = val_grad_state(
        align_parts["logits"][is_ref].detach(), align_parts["hidden"][is_ref].detach(),
        align_parts["target"][is_ref], index_ids=align_parts["index"][is_ref], delta=delta[is_ref])

    # Share the reference factors so every rank aligns against the same G_ref. Rows are padded to
    # the global maximum; padded rows have a zero residual and contribute nothing.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ws = torch.distributed.get_world_size(dp)
        if ws > 1:
            m = torch.tensor([state["r"].shape[0]], device=state["r"].device)
            torch.distributed.all_reduce(m, op=torch.distributed.ReduceOp.MAX, group=dp)
            mc = int(m.item())
            shared = {}
            for key in ("r", "ids", "h"):
                v = state[key]
                if v.shape[0] < mc:
                    v = torch.cat([v, torch.zeros((mc - v.shape[0],) + tuple(v.shape[1:]),
                                                  dtype=v.dtype, device=v.device)], dim=0)
                bufs = [torch.empty_like(v) for _ in range(ws)]
                torch.distributed.all_gather(bufs, v.contiguous(), group=dp)
                shared[key] = torch.cat(bufs, dim=0)
            state = shared

    a = per_step_alignment(
        align_parts["logits"][is_train], align_parts["hidden"][is_train],
        align_parts["target"][is_train], state, index_ids=align_parts["index"][is_train],
        chunk=ALIGN_CHUNK, delta=delta[is_train])

    # a_t inherits the magnitude of two gradients, so the loss is rescaled by a running estimate of
    # its own spread. The scale is a detached scalar, identical on every rank.
    rms = float(a.detach().pow(2).mean().sqrt())
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rt = torch.tensor([rms], device=align_parts["hidden"].device)
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM, group=dp)
        rms = float(rt.item()) / torch.distributed.get_world_size(dp)
    ema = getattr(self, "_align_rms_ema", None)
    ema = rms if ema is None else (0.99 * ema + 0.01 * rms)
    self._align_rms_ema = ema
    scale = min(ALIGN_SCALE_CAP, ALIGN_TARGET / (ema + 1e-30))
    align_loss = -a.mean() * scale

    params = [p for p in self._controller.parameters() if p.requires_grad]
    grads = None
    if torch.is_tensor(tau_blend) and tau_blend.requires_grad:
        g_tau = torch.autograd.grad(align_loss, [tau_blend], retain_graph=True, allow_unused=True)[0]
        if g_tau is not None:
            # Centre within each trajectory's think positions, so the controller is trained on where
            # softness goes rather than on its overall level.
            m = (topk_ids[..., 1:] != 0).any(dim=-1, keepdim=True).to(g_tau.dtype)
            n = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            centred = ((g_tau - (g_tau * m).sum(dim=1, keepdim=True) / n) * m).detach()
            grads = torch.autograd.grad([tau_blend], params, grad_outputs=[centred],
                                        retain_graph=True, allow_unused=True)
    if grads is None:
        grads = torch.autograd.grad(align_loss, params, retain_graph=True, allow_unused=True)

    store = getattr(self, "_align_grad_store", None)
    if store is None:
        store = {}
        self._align_grad_store = store
    for p, g in zip(params, grads):
        if g is None:
            continue
        p.grad = g.clone() if p.grad is None else (p.grad + g)
        key = id(p)
        store[key] = g.clone() if key not in store else (store[key] + g)


# --------------------------------------------------------------------------------- optimizer hooks


def _optimizer_step(self):
    """verl's optimizer step plus the controller's own step."""
    copt = getattr(self, "_controller_optimizer", None)
    if copt is not None:
        params = list(self._controller.parameters())
        dp = getattr(self, "_controller_dp_group", None)

        # Replace the accumulated gradients with the alignment-only stash BEFORE synchronising, so
        # the policy loss contribution that the temperature path carries is discarded and every rank
        # synchronises the same quantity.
        stash = getattr(self, "_align_grad_store", None) or {}
        for p in params:
            g = stash.get(id(p))
            p.grad = g.clone() if g is not None else None

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ws = torch.distributed.get_world_size(dp)
            if ws > 1:
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                for p in params:
                    torch.distributed.all_reduce(p.grad, group=dp)
                    p.grad /= ws
                for p in params:
                    if not p.grad.any():
                        p.grad = None

        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
        clip = getattr(self.optimizer_config, "clip_grad", None) or 1.0
        torch.nn.utils.clip_grad_norm_(params, max_norm=clip)

    grad_norm = _ORIG_OPT_STEP(self)
    if copt is not None:
        copt.step()
    return grad_norm


def _optimizer_zero_grad(self):
    _ORIG_OPT_ZERO(self)
    copt = getattr(self, "_controller_optimizer", None)
    if copt is not None:
        copt.zero_grad()
    self._align_grad_store = {}


# -------------------------------------------------------------------- weight sync and checkpoints


def _get_per_tensor_param(self, *args, **kwargs):
    """Chain the controller's parameters onto verl's weight-sync generator.

    The controller is not in the model's state dict, so without this the rollout engine would keep
    its initial controller while the trained one drifts away from it.
    """
    per_tensor_param, peft_cfg = _ORIG_GET_PER_TENSOR(self, *args, **kwargs)
    controller = getattr(self, "_controller", None)
    if controller is None:
        return per_tensor_param, peft_cfg
    dev = get_device_id()

    def _chain():
        for nt in per_tensor_param:
            yield nt
        for name, p in controller.named_parameters():
            yield (f"adaptsoft.{name}", p.detach().to(dev).float())

    return _chain(), peft_cfg


def _save_checkpoint(self, local_path, *args, **kwargs):
    _ORIG_SAVE_CKPT(self, local_path, *args, **kwargs)
    controller = getattr(self, "_controller", None)
    if controller is None:
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        torch.save({k: v.detach().cpu() for k, v in controller.state_dict().items()},
                   os.path.join(local_path, "adaptsoft_controller.pt"))


def _load_checkpoint(self, local_path, *args, **kwargs):
    _ORIG_LOAD_CKPT(self, local_path, *args, **kwargs)
    _ensure_controller(self)
    controller = getattr(self, "_controller", None)
    if controller is None or INIT_FROM:
        return
    path = os.path.join(local_path, "adaptsoft_controller.pt")
    if os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        dev = next(controller.parameters()).device
        controller.load_state_dict({k: v.to(dev) for k, v in sd.items()})


def apply():
    if getattr(FSDPEngineWithLMHead, "_adaptsoft_patched", False):
        return
    FSDPEngineWithLMHead.forward_step = soft_forward_step
    FSDPEngineWithLMHead._build_model_optimizer = _build_hook(
        FSDPEngineWithLMHead._build_model_optimizer)
    FSDPEngineWithLMHead.optimizer_step = _optimizer_step
    FSDPEngineWithLMHead.optimizer_zero_grad = _optimizer_zero_grad
    FSDPEngineWithLMHead.get_per_tensor_param = _get_per_tensor_param
    FSDPEngineWithLMHead.save_checkpoint = _save_checkpoint
    FSDPEngineWithLMHead.load_checkpoint = _load_checkpoint
    FSDPEngineWithLMHead._adaptsoft_patched = True


apply()
