"""Per-token log-probabilities for soft-thinking rollouts.

For each generated position:
  * answer tokens (one-hot record, discrete decoding after </think>): cross-entropy log-prob of
    the realised label.
  * think tokens (Gumbel-perturbed top-k record): the reparameterised log-density of the stored
    rollout noise under the current policy's top-k distribution, so the gradient reaches the
    logits through the residual (rollout noise - log p_theta).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def per_token_ce_logprobs(logits: torch.Tensor, labels: torch.Tensor, inplace_backward: bool = True) -> torch.Tensor:
    """Per-token log p(label): log_softmax and gather."""
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def _prep(logits, rollout_topk_ids, rollout_topk_gumbels, labels):
    """Flatten leading dims; return per-token CE log-prob and the current-policy normalised
    top-k log-probs from a single log_softmax over the vocabulary.
    """
    batch_dim = logits.shape[:-1]
    v = logits.shape[-1]
    k = rollout_topk_ids.shape[-1]
    logits = logits.reshape(-1, v)
    labels = labels.reshape(-1)
    rollout_topk_ids = rollout_topk_ids.reshape(-1, k)
    rollout_topk_gumbels = rollout_topk_gumbels.reshape(-1, k)
    logp_full = torch.log_softmax(logits, dim=-1)
    ce_logp = logp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    topk_logp_raw = logp_full.gather(-1, rollout_topk_ids)
    del logp_full
    topk_probs = torch.softmax(topk_logp_raw, dim=-1)
    ids_finish = (rollout_topk_ids[:, 1:] == 0).all(-1)
    return batch_dim, ce_logp, rollout_topk_ids, rollout_topk_gumbels, topk_probs, ids_finish


def gumbel_reparam_logprobs(logits, rollout_topk_ids, rollout_topk_gumbels, labels, inplace_backward=True):
    """Gumbel reparameterised log-density for think tokens, cross-entropy for answer tokens.

    The rollout stored g = log p_rollout + noise. Under the current policy the residual
    (g - log p_theta) is ~Gumbel(0,1); its standard-Gumbel log-density is the per-token
    soft log-prob, differentiable w.r.t. the current logits.
    """
    batch_dim, out_answer, topk_ids, topk_gumbels, topk_probs, ids_finish = _prep(
        logits, rollout_topk_ids, rollout_topk_gumbels, labels
    )
    topk_logp = (topk_probs + 1e-6).log()
    reparam = (topk_gumbels - topk_logp).clamp(-1.5, 3)
    out_gumbel = -reparam - (-reparam).exp()
    mask = (topk_logp > -3).float()
    out_gumbel = (out_gumbel * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
    return torch.where(ids_finish, out_answer, out_gumbel).view(*batch_dim)
