"""Per-step gradient alignment.

Restricted to the output matrix W, the gradient contributed by a token with logit gradient
delta_t and final hidden state h_t is the outer product delta_t (x) h_t. Its alignment with a
reference gradient G_ref = sum_s delta_s (x) h_s is therefore

    a_t = < delta_t (x) h_t , G_ref >  =  sum_s <delta_t, delta_s> <h_t, h_s>

G_ref is kept as the factors (residuals, ids, hidden) rather than a dense [V, D] matrix, and the
residuals are restricted to the recorded top-k plus the gold id, so each term contracts over k+1
vocab entries.
"""
from __future__ import annotations

import torch


def _residual(logits, target_ids, index_ids=None, delta=None):
    """Logit gradient per position, restricted to `index_ids` plus the gold id.

    logits      [N, V]
    target_ids  [N]
    index_ids   [N, k] vocab ids to keep; None keeps the full vocab
    delta       [N, V] dL/d(logits) from the training loss; None falls back to (p - y)
    returns     [N, k+1] residuals, [N, k+1] the ids they correspond to
    """
    if index_ids is None:
        if delta is not None:
            ids = torch.arange(delta.shape[-1], device=delta.device).expand(delta.shape[0], -1)
            return delta.float(), ids
        p = torch.softmax(logits.float(), dim=-1)
        r = p.clone()
        r.scatter_add_(1, target_ids[:, None], -torch.ones_like(target_ids, dtype=r.dtype)[:, None])
        ids = torch.arange(logits.shape[-1], device=logits.device).expand(logits.shape[0], -1)
        return r, ids

    ids = torch.cat([index_ids, target_ids[:, None]], dim=1)          # [N, k+1]
    if delta is not None:
        r = torch.gather(delta.float(), 1, ids)
        dup = (ids == target_ids[:, None]).cumsum(dim=1) > 1
        return r.masked_fill(dup, 0.0), ids
    p = torch.softmax(logits.float(), dim=-1)
    r = torch.gather(p, 1, ids)
    r = r - (ids == target_ids[:, None]).to(r.dtype)
    dup = (ids == target_ids[:, None]).cumsum(dim=1) > 1
    return r.masked_fill(dup, 0.0), ids


def val_grad_state(logits, hidden, target_ids, index_ids=None, delta=None):
    """Factors of the reference gradient G_ref = sum_s delta_s (x) h_s, detached."""
    r, ids = _residual(logits, target_ids, index_ids, delta)
    return {"r": r.detach(), "ids": ids.detach(), "h": hidden.detach().float()}


def per_step_alignment(logits, hidden, target_ids, val_state, index_ids=None, chunk=2048, delta=None):
    """a_t for every position, without materialising G_ref.

    `delta` is detached, so the gradient of a_t with respect to the controller flows through
    hidden only.

    returns a_t [N] (float32)
    """
    r_t, ids_t = _residual(logits, target_ids, index_ids, delta)        # [N, k+1]
    r_s, ids_s, h_s = val_state["r"], val_state["ids"], val_state["h"]  # [M, k+1], [M, k+1], [M, D]
    h_t = hidden.float()

    out = []
    for i in range(0, h_t.shape[0], chunk):
        ht = h_t[i:i + chunk]                                          # [c, D]
        rt, it = r_t[i:i + chunk], ids_t[i:i + chunk]                   # [c, k+1]
        hh = ht @ h_s.t()                                              # [c, M]
        match = (it[:, None, :, None] == ids_s[None, :, None, :])       # [c, M, k+1, k+1]
        rr = (rt[:, None, :, None] * r_s[None, :, None, :] * match).sum(dim=(-1, -2))   # [c, M]
        out.append((rr * hh).sum(dim=1))                                # [c]
    return torch.cat(out, dim=0)
