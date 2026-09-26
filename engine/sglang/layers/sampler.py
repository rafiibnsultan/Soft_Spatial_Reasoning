import logging
import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.utils.hash import murmur_hash32
from sglang.srt.layers.utils.logprob import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.common import (
    crash_on_warnings,
    get_bool_env_var,
    is_cuda,
    is_musa,
    is_npu,
)

if is_cuda():
    from flashinfer.sampling import (
        min_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs,
    )
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
    )

if is_musa():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


if is_npu():
    import torch_npu

logger = logging.getLogger(__name__)

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")
_CUSTOM_SAMPLER_FACTORIES: Dict[str, Callable[[], "Sampler"]] = {}
_BUILT_IN_SAMPLING_BACKENDS = {"flashinfer", "pytorch", "ascend"}


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_nan_detection = get_global_server_args().enable_nan_detection
        self.tp_sync_group = get_tp_group().device_group
        if is_dp_attention_enabled():
            self.tp_sync_group = get_attention_tp_group().device_group

        self.rl_on_policy_target = get_global_server_args().rl_on_policy_target
        # In RL on-policy mode, deterministic inference is automatically enabled.
        self.enable_deterministic = (
            get_global_server_args().enable_deterministic_inference
        )
        # In RL on-policy mode, we use log_softmax to compute logprobs to match the trainer.
        self.use_log_softmax_logprob = self.rl_on_policy_target is not None
        self.use_ascend_backend = get_global_server_args().sampling_backend == "ascend"

        # --- soft-thinking (SofT-GRPO patch A): global flags from server_args ---
        _sa = get_global_server_args()
        self.enable_soft_thinking = bool(getattr(_sa, "enable_soft_thinking", False))
        self.soft_gumbel_temperature = float(getattr(_sa, "gumbel_softmax_temperature", 0.1) or 0.1)
        self.soft_noise_factor = float(getattr(_sa, "soft_noise_factor", 1.0) or 1.0)
        self.soft_top_k = int(getattr(_sa, "soft_top_k", 5) or 5)

    def _preprocess_logits(
        self, logits: torch.Tensor, sampling_info: SamplingBatchInfo
    ) -> torch.Tensor:
        """Apply custom logit processors and handle NaN detection."""
        # Apply the custom logit processors if registered in the sampling info
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(logits, sampling_info)

        # Detect and handle NaN values in logits
        if self.use_nan_detection and torch.any(torch.isnan(logits)):
            logger.warning("Detected errors during sampling! NaN in the logits.")
            logits = torch.where(
                torch.isnan(logits), torch.full_like(logits, -1e5), logits
            )
            if crash_on_warnings():
                raise ValueError("Detected errors during sampling! NaN in the logits.")

        return logits

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            token_ids_logprobs: Per-sequence list of specific token IDs to retrieve
                logprobs for. Each element is a list of token IDs (or None) for one
                sequence in the batch. This is used in speculative decoding.
            positions: The positions of the tokens in the sequence. Used for deterministic sampling
                to get the unique seed for each position.
        """
        logits = logits_output.next_token_logits

        # Preprocess logits (custom processors and NaN handling)
        logits = self._preprocess_logits(logits, sampling_info)

        # SofT-GRPO patch A: intercept ALL sampling modes with Gumbel top-K soft sampling.
        # Records (soft_topk_indices, soft_topk_gumbels) on logits_output for the trainer;
        # model_runner.forward_extend feeds the weighted soft embedding on the next step.
        if self.enable_soft_thinking:
            probs = torch.softmax(logits / sampling_info.temperatures, dim=-1)
            batch_next_token_ids = self._soft_thinking_sample(probs, logits_output, sampling_info)
            if return_logprob:
                logprobs = torch.log_softmax(logits, dim=-1)
                self._attach_logprobs_to_output(
                    logits_output, logprobs, top_logprobs_nums,
                    token_ids_logprobs, sampling_info, batch_next_token_ids,
                )
            self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)
            return batch_next_token_ids

        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            batch_next_token_ids = torch.argmax(logits, -1)
            if return_logprob:
                original_logprobs = logprobs = torch.nn.functional.log_softmax(
                    logits, dim=-1
                )
        else:
            simple_sampling_case = (
                not sampling_info.need_top_p_sampling
                and not sampling_info.need_top_k_sampling
                and not sampling_info.need_min_p_sampling
            )

            # If requested, cache original logprobs before temperature scaling.
            if return_logprob and SGLANG_RETURN_ORIGINAL_LOGPROB:
                original_logprobs = torch.log_softmax(logits, dim=-1)

            # In RL on-policy mode, we use log_softmax to compute logprobs to match the trainer.
            logprobs_via_logsoftmax_kernel = None
            if self.rl_on_policy_target is not None:
                # TODO: use more inplace ops to save memory
                logits_div_temperature = (
                    logits.bfloat16().div(sampling_info.temperatures).bfloat16()
                )
                logprobs_via_logsoftmax_kernel = torch.log_softmax(
                    logits_div_temperature, dim=-1
                )
                del logits_div_temperature

            if self.use_ascend_backend:
                # Ascend backend: sample from logits directly.
                batch_next_token_ids, logprobs = self._forward_ascend_backend(
                    logits, sampling_info, simple_sampling_case, return_logprob
                )
            elif (
                self.use_log_softmax_logprob
                and self.enable_deterministic
                and simple_sampling_case
            ):
                # RL on-policy path: sample from logprobs to match the trainer.
                batch_next_token_ids = self._sample_from_logprobs(
                    logprobs_via_logsoftmax_kernel,
                    sampling_info,
                    positions,
                )
                if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:
                    logprobs = logprobs_via_logsoftmax_kernel
            else:
                # Standard path: do softmax and sample from probs.
                logits.div_(sampling_info.temperatures)

                # In-place op to save memory
                logits[:] = torch.softmax(logits, dim=-1)
                probs = logits

                batch_next_token_ids = self._sample_from_probs(
                    probs, sampling_info, positions, simple_sampling_case
                )
                if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:
                    logprobs = (
                        logprobs_via_logsoftmax_kernel
                        if logprobs_via_logsoftmax_kernel is not None
                        else torch.log(probs)
                    )
                del probs

        # Attach logprobs to logits_output (in-place modification)
        if return_logprob:
            if SGLANG_RETURN_ORIGINAL_LOGPROB:
                logprobs = original_logprobs
            self._attach_logprobs_to_output(
                logits_output,
                logprobs,
                top_logprobs_nums,
                token_ids_logprobs,
                sampling_info,
                batch_next_token_ids,
            )

        self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)

        return batch_next_token_ids

    def _soft_thinking_sample(self, probs: torch.Tensor, logits_output, sampling_info=None) -> torch.Tensor:
        """SofT-GRPO patch A — Gumbel-perturbed top-K soft sampling (mirrors
        utils/gumbel_rollout.gumbel_softmax_sample).

        Perturb the top-K log-probs with clamped Gumbel(0,1) noise, re-softmax at tau, sort desc.
        Records (soft_topk_indices, soft_topk_gumbels) on logits_output so the trainer can rebuild the
        exact rollout weights = softmax(gumbels/tau) @ embed(indices); tp_worker/model_runner feed
        that weighted embedding as the next decode input (v1). Returns the realized top-1 'spine'
        token id (used to detect </think> and to decode/score the discrete answer).

        v1 mode split: rows whose request has already emitted </think> (soft_thinking_modes[i] is
        False, set by tp_worker.soft_attach_feed) are sampled DISCRETELY — plain full-vocab
        multinomial at temperature — and recorded as one-hot ([tok, 0, ..], gumbels 0). One-hot is
        the finished-token convention the trainer already keys on: CE log-prob instead of gumbel,
        and the feed side embeds the spine directly. Rows with no mode info default to soft
        (v0-compatible: all-soft when the feature isn't wired).
        """
        K = self.soft_top_k
        topk_probs, topk_indices = torch.topk(probs, k=K, dim=-1)          # [B, K]
        # --- ENTROPY DIAGNOSTIC (ENTROPY_PROBE=1): is per-token uncertainty spiky, and at what window? ---
        # Dumps RAW per-token rows (not batch means): full-vocab entropy vs renormalized top-{5,20,50}
        # entropy, all normalized to [0,1], + how much prob mass the top-5 actually captures. Answers
        # whether the gate's current top-5 signal (line below) is blind to real forking-token uncertainty.
        import os as _os
        if _os.environ.get("ENTROPY_PROBE", "0") == "1":
            _ep_n = getattr(self, "_ent_probe_n", 0)
            if _ep_n < int(_os.environ.get("ENTROPY_PROBE_MAX", "2000")):
                try:
                    with torch.no_grad():
                        _lp = torch.log(probs + 1e-12)
                        _h_full = (-(probs * _lp).sum(-1)) / math.log(probs.shape[-1])   # [B] full-vocab, norm
                        def _htop(k):
                            k = min(k, probs.shape[-1])
                            _p = torch.topk(probs, k, dim=-1)[0]
                            _p = _p / _p.sum(-1, keepdim=True)
                            return (-(_p * torch.log(_p + 1e-12)).sum(-1)) / math.log(k)  # renorm top-k, norm
                        _h5, _h20, _h50 = _htop(5), _htop(20), _htop(50)
                        _m5 = torch.topk(probs, min(5, probs.shape[-1]), dim=-1)[0].sum(-1)   # mass in top-5
                        # Also dump the RAW top-5 probabilities. Entropy alone does not determine the
                        # distribution shape, and the shape is what decides how much leverage tau has at
                        # that step (see the leverage analysis) -- with only h we have to ASSUME a shape.
                        _p5 = torch.topk(probs, min(5, probs.shape[-1]), dim=-1)[0]
                        _csv = _os.environ.get("ENTROPY_PROBE_CSV") or _os.path.expanduser("~/data/entropy_probe.csv")
                        _new = not _os.path.exists(_csv)
                        with open(_csv, "a") as _f:
                            if _new:
                                _f.write("call,row,h_full,h_top5,h_top20,h_top50,mass_top5,"
                                         "p1,p2,p3,p4,p5\n")
                            for _r in range(probs.shape[0]):
                                _f.write("%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%s\n" % (
                                    _ep_n, _r, _h_full[_r].item(), _h5[_r].item(),
                                    _h20[_r].item(), _h50[_r].item(), _m5[_r].item(),
                                    ",".join("%.6f" % v for v in _p5[_r].tolist())))
                        self._ent_probe_n = _ep_n + 1
                except Exception:
                    pass
        _topk_raw = topk_probs                     # pre-renormalization: mass_top5 + margin features
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        topk_logits = torch.log(topk_probs + 1e-6)
        gumbels = (-torch.empty_like(topk_logits).exponential_().log()).clamp(-1.5, 3)  # ~Gumbel(0,1)
        topk_gumbels = topk_logits + self.soft_noise_factor * gumbels
        # Phase-D: adaptive per-token temperature from the entropy gate (fallback = fixed scalar).
        gate = getattr(self, "gate", None)
        if gate is not None:
            h_hat = (-(topk_probs * topk_logits).sum(-1, keepdim=True)) / math.log(K)   # [B,1] in [0,1]
            import os as _os
            # STEP-1 CONTROLLER UPGRADE (GATE_FEATS>1): Ĥ_top5 alone is a weak control signal — the probe
            # above exists precisely because top-5 entropy can be blind to real forking-token uncertainty.
            # Give f_φ a FEATURE VECTOR instead. Slot 0 stays Ĥ_top5, so GATE_FEATS=1 is the old gate exactly.
            #   0 Ĥ_top5 | 1 Ĥ_full | 2 Ĥ_top20 | 3 mass_top5 | 4 margin(p1−p2)
            # All ride the EXISTING [B,K] soft_gate_h slot (K=5), so no new pipeline plumbing is needed.
            _nf = int(_os.environ.get("GATE_FEATS", "1") or 1)
            if _nf > 1:
                assert _nf <= K, f"GATE_FEATS={_nf} exceeds soft_top_k={K} (the carrier width)"
                with torch.no_grad():
                    _V = probs.shape[-1]
                    _lpv = torch.log(probs + 1e-12)
                    _h_full = (-(probs * _lpv).sum(-1, keepdim=True)) / math.log(_V)
                    _k20 = min(20, _V)
                    _p20 = torch.topk(probs, _k20, dim=-1)[0]
                    _p20 = _p20 / _p20.sum(-1, keepdim=True)
                    _h20 = (-(_p20 * torch.log(_p20 + 1e-12)).sum(-1, keepdim=True)) / math.log(_k20)
                    _mass = _topk_raw.sum(-1, keepdim=True)
                    _marg = (_topk_raw[:, 0:1] - _topk_raw[:, 1:2]) if K >= 2 else torch.zeros_like(_mass)
                    h_hat = torch.cat([h_hat, _h_full, _h20, _mass, _marg], dim=-1)[:, :_nf].contiguous()
            _is_mlp = hasattr(gate, "mu_logsigma")                                      # MLPEntropyGate (f_φ)
            _gate_args = (h_hat,)
            if _is_mlp:
                # f_φ needs the last-token hidden state h_t. sglang exposes it as logits_output.hidden_states
                # when enable_return_hidden_states is on (forced when GATE_MLP). It must be row-aligned with
                # the rows we're sampling (one h_t per token). HARD-FAIL loudly if it's missing/misaligned so a
                # smoke run tells us immediately, instead of silently producing wrong τ.
                _ht = getattr(logits_output, "hidden_states", None)
                if _ht is None:
                    raise RuntimeError("[soft] MLP gate: logits_output.hidden_states is None — hidden-state "
                                       "capture not active (enable_return_hidden_states / CUDA-graph capture).")
                if _ht.shape[0] != h_hat.shape[0]:
                    raise RuntimeError(f"[soft] MLP gate: hidden_states rows {_ht.shape[0]} != sampled rows "
                                       f"{h_hat.shape[0]} — misaligned capture (likely FULL over all tokens).")
                _ht = _ht.to(h_hat.dtype)
                # proj_dim>0: feed the MLP the PROJECTED summary and record it, so the same input can be
                # rebuilt at train time and the dense gradient can reach fc1/fc2 (not just ent_a/ent_b).
                _zt = gate.project(_ht) if getattr(gate, "proj_dim", 0) > 0 else None
                _gate_args = ((_zt, h_hat) if _zt is not None else (_ht, h_hat))
                _dn = getattr(self, "_gate_ht_log_n", 0) + 1                            # one-time shape probe
                self._gate_ht_log_n = _dn
                if _dn <= 3:
                    try:
                        _gp = _os.environ.get("GATE_PROBE_LOG") or _os.path.expanduser("~/data/gate_probe.log")
                        with open(_gp, "a") as _f:
                            _f.write("[soft] MLP gate h_t OK: hidden_states=%s (rows match logits %d), "
                                     "|h_t|mean=%.3f\n" % (tuple(_ht.shape), h_hat.shape[0],
                                                           float(_ht.float().norm(dim=-1).mean())))
                    except Exception:
                        pass
            # INFERENCE vs TRAINING: GATE_DETERMINISTIC=1 → act on the policy MEAN (no exploration noise) —
            # still adaptive, just no jitter. Unset → sample (training).
            if _os.environ.get("GATE_DETERMINISTIC", "0") == "1":
                gate_u, tau = gate.infer(*_gate_args)                                   # deterministic
                _keff = None
                if getattr(gate, "k_ctrl", False):
                    _keff = gate.mu_logsigma_k(*_gate_args)[2]                          # [B,1] in (1,K+1)
            else:
                gate_u, tau = gate.sample(*_gate_args)                                  # stochastic (train)
                if getattr(gate, "k_ctrl", False):
                    _keff = gate.mu_logsigma_k(*_gate_args)[2]
            # ALLOCATION OVERRIDE (diagnostic, eval only).  tau_t(alpha) = c + alpha*(tau_t - c).
            # Comparing v4/v5/v6 across separate training runs confounds "how much the controller
            # varies softness" with "which LVLM those weights are", because each arm is a different
            # training run. This holds the WEIGHTS fixed and changes only whether the learned
            # allocation is applied:
            #   alpha=1  the learned allocation (default, unchanged behaviour)
            #   alpha=0  the SAME model with a uniform tau = c, allocation removed
            # alpha=0 beating alpha=1 means the controller is spending softness on the wrong steps;
            # equal means allocation is causally inert and any deficit came from the training
            # trajectory instead. Note alpha=0 is NOT the fixed-tau baseline -- that is a separately
            # trained model -- which is exactly why both comparisons are needed.
            _alpha = _os.environ.get("GATE_TAU_ALPHA", "")
            if _alpha != "":
                _c = float(_os.environ.get("GATE_TAU_CENTER",
                                           _os.environ.get("GATE_TAU_BASE", "0.5")))
                tau = _c + float(_alpha) * (tau - _c)
            _tln = getattr(self, "_gate_tau_log_n", 0) + 1
            self._gate_tau_log_n = _tln
            if _tln % 25 == 1:   # periodic trace (was one-shot): characterizes the τ distribution per run
                try:
                    _det = _os.environ.get("GATE_DETERMINISTIC", "0") == "1"
                    _gplog = _os.environ.get("GATE_PROBE_LOG") or _os.path.expanduser("~/data/gate_probe.log")
                    with open(_gplog, "a") as _f:
                        _f.write("[soft] gate τ LIVE at sampler (%s): τ mean=%.3f min=%.3f max=%.3f | Ĥ mean=%.3f\n"
                                 % ("DETERMINISTIC/mean" if _det else "stochastic/sample",
                                    tau.float().mean(), tau.float().min(), tau.float().max(), h_hat.float().mean()))
                except Exception:
                    pass
        else:
            gate_u, tau = None, self.soft_gumbel_temperature
            # SOFT_K=<float in (1, K+1]>: a GLOBAL FIXED soft candidate count -- the k analogue of the
            # fixed GUMBEL_TEMP tau, with no gate and nothing learned. k_eff=1 collapses the blend to the
            # top-1 token (discrete), k_eff=K+1 is the full soft mixture. This exists so k can be SWEPT
            # the way tau was swept: tau turned out to be inert (0.1 vs 0.5 moved neither correctness nor
            # group diversity) because even tau=0.5 is only ~1.09 effective candidates -- the whole usable
            # tau range stays near-discrete. k sets the blend width directly, independent of how peaked
            # the logits are, so it can reach the regime tau cannot. Unset => full soft blend, unchanged.
            _sk = _os.environ.get("SOFT_K")
            if _sk:
                _keff = torch.full((topk_gumbels.shape[0], 1), float(_sk),
                                   device=topk_gumbels.device, dtype=torch.float32)
        soft_probs = (topk_gumbels / tau).softmax(-1)          # tau: scalar OR [B,1] broadcasts over K
        _, sorted_idx = torch.sort(soft_probs, dim=-1, descending=True)
        topk_indices = torch.gather(topk_indices, 1, sorted_idx)
        topk_gumbels = torch.gather(topk_gumbels, 1, sorted_idx)

        modes = getattr(sampling_info, "soft_thinking_modes", None) if sampling_info is not None else None
        if modes is not None and not all(modes):
            disc = ~torch.tensor(modes, dtype=torch.bool, device=probs.device)
            disc_tokens = torch.multinomial(probs[disc], num_samples=1).squeeze(-1)
            topk_indices[disc] = 0
            topk_indices[disc, 0] = disc_tokens
            topk_gumbels[disc] = 0.0
            # gumbels[0]=1 marks the row as a real record even when the sampled token id is 0
            # (an all-zero row is the trainer's padding convention). Harmless to the math: the
            # finished-mask softmax over the single non-inf slot yields weight 1 regardless.
            topk_gumbels[disc, 0] = 1.0
        # Record for the trainer via sglang's generic per-request customized_info pipeline:
        # maybe_collect_customized_info accumulates customized_info[k][i] onto req.customized_info,
        # which is packed into the output (BatchTokenIDOutput.customized_info) and returned to verl.
        # Each is [B, K]; the collector indexes [i] -> per-token [K], appended per decode step.
        ci = logits_output.customized_info if logits_output.customized_info is not None else {}
        ci["soft_topk_indices"] = topk_indices
        ci["soft_topk_gumbels"] = topk_gumbels
        if gate is not None:
            # τ and u are per-token (constant across K); broadcast to [B,K] for the [B,K] collector.
            # τ -> exact per-step embedding rebuild; u -> the gate's REINFORCE log-prob at train time.
            ci["soft_topk_tau"] = tau.expand(-1, K).contiguous()
            _ke_ci = locals().get("_keff", None)
            if _ke_ci is not None:                    # step-3: per-step soft candidate count for the feed
                ci["soft_topk_keff"] = _ke_ci.expand(-1, K).contiguous()
            _zt_rec = locals().get("_zt", None)
            if _zt_rec is None:
                ci["soft_gate_u"] = gate_u.expand(-1, K).contiguous()
            else:
                # carrier packing: soft_gate_u slot 0 = u, slots 1..K-1 = first (K-1) projected dims.
                # The remaining dims ride soft_gate_h's slots after the feature block (see below).
                _nu = min(_zt_rec.shape[-1], K - 1)
                _ub = torch.zeros(gate_u.shape[0], K, dtype=gate_u.dtype, device=gate_u.device)
                _ub[:, 0:1] = gate_u
                if _nu > 0:
                    _ub[:, 1:1 + _nu] = _zt_rec[:, :_nu].to(_ub.dtype)
                ci["soft_gate_u"] = _ub
            _nf_c = h_hat.shape[-1]
            if _nf_c == 1 and _zt_rec is None:
                ci["soft_gate_h"] = h_hat.expand(-1, K).contiguous()   # Ĥ (gate state) for the train-time log-prob
            else:                                        # pack features, then any remaining projected dims
                _fb = torch.zeros(h_hat.shape[0], K, dtype=h_hat.dtype, device=h_hat.device)
                _fb[:, :_nf_c] = h_hat
                _off = _nf_c
                if _zt_rec is not None and _zt_rec.shape[-1] > K - 1:
                    _rest = _zt_rec[:, K - 1:]
                    _fb[:, _off:_off + _rest.shape[-1]] = _rest.to(_fb.dtype)
                    _off += _rest.shape[-1]
                _ke = locals().get("_keff", None)
                if _ke is not None:                      # k_eff rides the last free slot
                    assert _off < K, (f"carrier overflow: n_feats={_nf_c} + leftover proj dims + k_eff "
                                      f"exceeds K={K}. Use GATE_PROJ_DIM={2 * K - 2 - _nf_c} with k-control.")
                    _fb[:, _off:_off + 1] = _ke.to(_fb.dtype)
                    _off += 1
                ci["soft_gate_h"] = _fb
        else:
            # fixed-k sweep (SOFT_K, no gate): k_eff still has to reach the feed, so record it on its own.
            _ke_fx = locals().get("_keff", None)
            if _ke_fx is not None:
                ci["soft_topk_keff"] = _ke_fx.expand(-1, K).contiguous()
        logits_output.customized_info = ci
        # (also expose for v1's in-rollout soft-embed feedback; harmless in the v0 discrete-spine path)
        logits_output.soft_topk_indices = topk_indices
        logits_output.soft_topk_gumbels = topk_gumbels
        if gate is not None:
            logits_output.soft_topk_tau = tau.expand(-1, K).contiguous()   # feed side uses per-step τ
            _ke_f = locals().get("_keff", None)
            if _ke_f is not None:
                logits_output.soft_topk_keff = _ke_f.expand(-1, K).contiguous()   # feed applies the k-mask
        else:
            _ke_fx2 = locals().get("_keff", None)
            if _ke_fx2 is not None:                                              # fixed-k sweep (SOFT_K)
                logits_output.soft_topk_keff = _ke_fx2.expand(-1, K).contiguous()
        return topk_indices[:, 0].to(torch.long)

    def _sample_from_probs(
        self,
        probs: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        positions: torch.Tensor,
        simple_sampling_case: bool,
    ) -> torch.Tensor:
        """Sample from probability distribution (after softmax).

        Used for standard sampling with flashinfer/pytorch backends.
        Handles both simple (direct multinomial) and complex (top-k/top-p/min-p) cases.
        """
        if simple_sampling_case:
            batch_next_token_ids = sampling_from_probs_torch(
                probs,
                sampling_seed=sampling_info.sampling_seed,
                positions=positions,
            )
        else:
            backend = get_global_server_args().sampling_backend
            if backend == "flashinfer":
                assert (
                    sampling_info.sampling_seed is None
                ), "Sampling seed is not supported for flashinfer backend"
                if sampling_info.need_min_p_sampling:
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                    batch_next_token_ids = min_p_sampling_from_probs(
                        probs, sampling_info.min_ps
                    )
                else:
                    batch_next_token_ids = top_k_top_p_sampling_from_probs(
                        probs.contiguous(),
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        filter_apply_order="joint",
                        check_nan=self.use_nan_detection,
                    )
            elif backend == "pytorch":
                # A slower fallback implementation with torch native operations.
                batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                    probs,
                    sampling_info.top_ks,
                    sampling_info.top_ps,
                    sampling_info.min_ps,
                    sampling_info.need_min_p_sampling,
                    sampling_info.sampling_seed,
                    positions,
                )
            else:
                raise ValueError(f"Invalid sampling backend: {backend}")
        return batch_next_token_ids

    def _sample_from_logprobs(
        self,
        logprobs: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from log-probabilities using the Gumbel trick.

        Used for deterministic sampling with simple cases (no top-k/top-p/min-p).
        Requires sampling_seed to be set in sampling_info.
        """
        assert (
            sampling_info.sampling_seed is not None
        ), "sampling_seed is required for sampling from logprobs"
        sampled_index = multinomial_with_seed(
            logprobs, sampling_info.sampling_seed, positions
        )
        return sampled_index.view(-1).to(torch.int32)

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        simple_sampling_case: bool,
    ) -> torch.Tensor:
        """Sample from temperature-scaled logits without softmax.

        Used for the Ascend NPU backend which handles softmax internally.
        """
        if simple_sampling_case:
            probs = torch.softmax(logits, dim=-1)
            batch_next_token_ids = torch.multinomial(probs, num_samples=1).view(-1)
            return batch_next_token_ids.to(torch.int32)
        else:
            assert (
                self.use_ascend_backend
            ), "Only ascend backend supports sampling from logits"
            batch_next_token_ids = top_k_top_p_min_p_sampling_from_logits_ascend(
                logits,
                sampling_info.top_ks,
                sampling_info.top_ps,
                sampling_info.min_ps,
                sampling_info.need_min_p_sampling,
            )
            return batch_next_token_ids.to(torch.int32)

    def _forward_ascend_backend(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        simple_sampling_case: bool,
        return_logprob: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Handle the full Ascend backend sampling path.

        Ascend backend has fused kernels that handle softmax internally,
        so we sample directly from temperature-scaled logits.

        Returns:
            A tuple of (batch_next_token_ids, logprobs). logprobs is None
            when return_logprob is False or SGLANG_RETURN_ORIGINAL_LOGPROB is set.
        """
        logits.div_(sampling_info.temperatures)
        batch_next_token_ids = self._sample_from_logits(
            logits, sampling_info, simple_sampling_case
        )
        logprobs = None
        if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:
            logprobs = torch.log_softmax(logits, dim=-1)
        return batch_next_token_ids, logprobs

    def _attach_logprobs_to_output(
        self,
        logits_output: LogitsProcessorOutput,
        logprobs: torch.Tensor,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        sampling_info: SamplingBatchInfo,
        batch_next_token_ids: torch.Tensor,
    ):
        # clamp to avoid -inf values
        logprobs.clamp_(min=torch.finfo(logprobs.dtype).min)

        # Attach logprobs to logits_output (in-place modification)
        if any(x > 0 for x in top_logprobs_nums):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums, no_copy_to_cpu=True)

        if any(x is not None for x in token_ids_logprobs):
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(
                logprobs, token_ids_logprobs, no_copy_to_cpu=True
            )

        logits_output.next_token_logprobs = logprobs[
            torch.arange(len(batch_next_token_ids), device=sampling_info.device),
            batch_next_token_ids,
        ]

    def _sync_token_ids_across_tp(
        self, batch_next_token_ids: torch.Tensor, sampling_info: SamplingBatchInfo
    ):
        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
    ) -> None:
        """
        Compute logprobs for requested token IDs without performing sampling.

        Optimized for prefill-only scoring requests that need token probabilities
        but don't require next token generation.
        """

        if logits_output.next_token_logits is None:
            logger.warning("No logits available for logprob computation")
            return

        # Check if any requests actually need logprobs computation
        needs_token_ids_logprobs = any(
            token_ids is not None and len(token_ids) > 0
            for token_ids in token_ids_logprobs
        )
        needs_top_logprobs = any(x > 0 for x in top_logprobs_nums)

        if not (needs_token_ids_logprobs or needs_top_logprobs):
            return

        # Preprocess logits (custom processors and NaN handling)
        logits = self._preprocess_logits(logits_output.next_token_logits, sampling_info)

        # Compute logprobs
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Handle top logprobs if requested
        if needs_top_logprobs:
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums, no_copy_to_cpu=True)

        # Handle token_ids logprobs if requested
        if needs_token_ids_logprobs:
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs_batch_optimized(logprobs, token_ids_logprobs)


def register_sampler_backend(backend: str, factory: Callable[[], "Sampler"]) -> None:
    """Register a custom sampler factory for a backend string."""

    if not backend:
        raise ValueError("backend must be a non-empty string")

    from sglang.srt.server_args import SAMPLING_BACKEND_CHOICES

    if backend in _CUSTOM_SAMPLER_FACTORIES:
        logger.warning("Overriding existing sampler factory for backend '%s'", backend)
    SAMPLING_BACKEND_CHOICES.add(backend)
    _CUSTOM_SAMPLER_FACTORIES[backend] = factory


def create_sampler(backend: Optional[str] = None) -> "Sampler":
    """Create a sampler honoring custom backend registrations."""

    server_args = get_global_server_args()
    backend = backend or (server_args.sampling_backend if server_args else None)

    if backend in _CUSTOM_SAMPLER_FACTORIES:
        sampler = _CUSTOM_SAMPLER_FACTORIES[backend]()
        if not isinstance(sampler, Sampler):
            raise TypeError(
                f"Custom sampler factory for backend '{backend}' must return a Sampler"
            )
        return sampler

    if backend is None or backend in _BUILT_IN_SAMPLING_BACKENDS:
        return Sampler()

    raise ValueError(
        f"Unknown sampling backend '{backend}'. Register it via register_sampler_backend()."
    )


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
    sampling_seed: Optional[torch.Tensor],
    positions: torch.Tensor,
):
    """
    A top-k, top-p and min-p sampling implementation with native pytorch operations.
    When sampling_seed is not None, deterministic inference will be enabled, it will sample
    with the sampling_seed of each request.
    """
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        # TODO: probs_sort should be re-normalized for the use of multinomial_with_seed
        assert (
            sampling_seed is None
        ), "With sampling seed, multinomial_with_seed will provide wrong results"
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    if sampling_seed is None:
        sampled_index = torch.multinomial(probs_sort, num_samples=1)
    else:
        # NOTE: when using top-k/top-p/min-p sampling, we need to modify probs before we
        # apply log to get logprobs. Therefore, we cannot use log_softmax directly.
        # For now, we use log to the modified probs to get logprobs, but for numerical
        # stability, we'd better come up with a solution to use log_softmax.
        logprobs = probs_sort.to(torch.float64)  # Using float64 for numerical stability
        del probs_sort
        logprobs.log_()
        sampled_index = multinomial_with_seed(logprobs, sampling_seed, positions)

    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def top_k_top_p_min_p_sampling_from_logits_ascend(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
):
    """A top-k, top-p and min-p sampling implementation for ascend npu with torch_npu interface.

    Takes temperature-scaled logits as input (softmax is applied internally).
    """
    # torch_npu.npu_top_k_top_p requires top_k value range in [1, 1024]
    if hasattr(torch_npu, "npu_top_k_top_p") and torch.all(
        (top_ks <= 1024) & (top_ks >= 1)
    ):
        logits_top_k_top_p = torch_npu.npu_top_k_top_p(logits, top_ps, top_ks)
        probs_top_k_top_p = logits_top_k_top_p.softmax(dim=-1)

        if need_min_p_sampling:
            min_p_thresholds = probs_top_k_top_p.max(dim=-1) * min_ps
            min_p_mask = probs_top_k_top_p < min_p_thresholds.view(-1, 1)
            probs_top_k_top_p.masked_fill_(min_p_mask, 0.0)

        batch_next_token_ids = torch.multinomial(probs_top_k_top_p, num_samples=1)
    else:
        probs = torch.softmax(logits, dim=-1)
        probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

        # when top_k is -1 (in which sglang turns it to TOP_K_ALL), make it explicitly equal to logit's size
        topk_all_mask = top_ks == TOP_K_ALL
        top_ks.masked_fill_(topk_all_mask, probs.shape[1])
        top_k_mask = torch.arange(0, probs.shape[-1], device=probs.device).view(
            1, -1
        ) >= top_ks.view(-1, 1)
        probs_sort.masked_fill_(top_k_mask, 0.0)

        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = probs_sum - probs_sort > top_ps.view(-1, 1)
        probs_sort.masked_fill_(top_p_mask, 0.0)

        if need_min_p_sampling:
            min_p_thresholds = probs_sort[:, 0] * min_ps
            min_p_mask = probs_sort < min_p_thresholds.view(-1, 1)
            probs_sort.masked_fill_(min_p_mask, 0.0)

        sampled_index = torch.multinomial(probs_sort, num_samples=1)
        probs_idx = probs_idx.to(torch.int32)
        batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index)

    return batch_next_token_ids.view(-1)


@torch.compile(dynamic=True)
def multinomial_with_seed(
    logprobs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """
    Samples n elements from an input tensor `inputs` of shape (n, m) using
    a unique random seed for each row. This is a deterministic batched alternative to
    `torch.multinomial`.

    Args:
        inputs: A float tensor of shape (n, m) representing n categorical
                distributions with m categories each. The values are treated
                as weights and do not need to sum to 1.
        seed:   An integer tensor of shape (n,) containing the random seed
                for each corresponding row in `inputs`.
        positions: The positions of the tokens in the sequence. Used for deterministic sampling
                to get the unique seed for each position.

    Returns:
        A tensor of shape (n,) where the i-th element is an index sampled
        from the distribution in `inputs[i]` using `seed[i]`.
    """
    n, m = logprobs.shape
    seed = seed.to(torch.uint64)
    col_indices = torch.arange(m, device=logprobs.device)
    hashed = murmur_hash32(seed, positions, col_indices)

    # NOTE (sehoon): it is critical to keep gumbel noise calculation in float64 to avoid numerical instability.
    # keeping logprobs in float64 is less critical, but we found it's still safer to keep it in float64.
    x = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max

    # x is a uniform sample in [0, 1]. get gumbel noise from it.
    # which is equivalent to -log(-log(x))
    # keep everything in in-place operations to avoid unnecessary memory allocations.
    x.log_().clamp_(min=torch.finfo(x.dtype).min).neg_()  # -log(x)
    x.log_().neg_()  # -log(-log(x)) == gumbel noise

    # add gumbel noise to logprobs
    x.add_(logprobs.to(torch.float64))

    return torch.argmax(x, dim=1, keepdim=True)


def sampling_from_probs_torch(
    probs: torch.Tensor,
    sampling_seed: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
):
    """A sampling implementation with native pytorch operations, without
    top-k, top-p, or min-p filtering.

    Note: For deterministic sampling from logprobs, use Sampler._sample_from_logprobs instead.
    """
    if sampling_seed is None:
        sampled_index = torch.multinomial(probs, num_samples=1)
    else:
        # Deterministic sampling: convert probs to logprobs and use gumbel trick
        sampled_index = multinomial_with_seed(
            torch.log(probs), sampling_seed, positions
        )
    batch_next_token_ids = sampled_index.view(-1).to(torch.int32)
    return batch_next_token_ids


def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def get_token_ids_logprobs_batch_optimized(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[List[int]],
) -> Tuple[List, List]:
    """
    Vectorized batch processing for token ID logprobs extraction.

    Uses a single GPU kernel call for the entire batch instead of multiple
    separate calls, significantly improving performance for large batches.

    Args:
        logprobs: Log probabilities tensor [batch_size, vocab_size]
        token_ids_logprobs: List of token IDs to extract logprobs for

    Example:
        # Input: batch_size=3, vocab_size=5
        logprobs = torch.tensor([
            [-1.2, -2.1, -0.8, -3.0, -1.5],  # batch 0
            [-0.5, -1.8, -2.2, -1.1, -2.7],  # batch 1
            [-2.0, -0.9, -1.4, -2.8, -1.6],  # batch 2
        ])
        token_ids_logprobs = [[1, 3], [2], [0, 2, 4]]

        # Output:
        # values = [tensor([-2.1, -3.0]), tensor([-2.2]), tensor([-2.0, -1.4, -1.6])]
        # indices = [[1, 3], [2], [0, 2, 4]]
    """
    batch_size = len(token_ids_logprobs)
    device = logprobs.device

    # Step 1: Calculate lengths for each request, treating None as empty list
    # Example: [[1, 3], [2], [0, 2, 4]] -> token_lengths = tensor([2, 1, 3])
    token_lengths = torch.tensor(
        [len(token_ids or []) for token_ids in token_ids_logprobs], device=device
    )
    total_tokens = int(token_lengths.sum().item())  # 2 + 1 + 3 = 6

    # Handle edge case where no tokens are requested
    if total_tokens == 0:
        return [logprobs.new_empty(0) for _ in token_ids_logprobs], [
            [] for _ in token_ids_logprobs
        ]

    # Step 2: Build flattened indices using torch operations
    # Example: row_indices = [0, 0, 1, 2, 2, 2] (batch indices repeated by their lengths)
    row_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device), token_lengths
    )
    # Example: col_indices = [1, 3, 2, 0, 2, 4] (flattened token IDs from all requests)
    col_indices = torch.tensor(
        [
            token_id
            for token_ids in token_ids_logprobs
            for token_id in (token_ids or [])
        ],
        device=device,
        dtype=torch.long,
    )

    # Step 3: Single vectorized gather operation
    # Example: logprobs[row_indices, col_indices] -> [-2.1, -3.0, -2.2, -2.0, -1.4, -1.6]
    gathered_logprobs = logprobs[row_indices, col_indices]

    # Step 4: Split results back per request using torch operations
    # Example: split tensor [6] into chunks of sizes [2, 1, 3] -> [tensor(2), tensor(1), tensor(3)]
    split_logprobs = torch.split_with_sizes(
        gathered_logprobs, token_lengths.tolist(), dim=0
    )

    # Step 5: Format output to match expected return structure
    # Example: Convert split tensors back to list format with proper empty handling
    # i=0: [1,3] -> append split_logprobs[0] and [1,3]
    # i=1: [2] -> append split_logprobs[1] and [2]
    # i=2: [0,2,4] -> append split_logprobs[2] and [0,2,4]
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []

    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None and len(token_ids) > 0:
            output_token_ids_logprobs_val.append(split_logprobs[i])
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append(logprobs.new_empty(0))
            output_token_ids_logprobs_idx.append([])

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx


def apply_custom_logit_processor(
    logits: torch.Tensor,
    sampling_batch_info: SamplingBatchInfo,
    num_tokens_in_batch: int = 1,
):
    """Apply custom logit processors to the logits.
    This function will modify the logits in-place.
    num_tokens_in_batch is needed to support spec decoding, where each batch can contain multiple
    tokens. By default, we assume each batch contains only 1 token.
    """

    assert logits.shape[0] == len(sampling_batch_info) * num_tokens_in_batch, (
        f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "
        f"sampling_batch_info ({len(sampling_batch_info)}) x num_tokens_in_batch "
        f"({num_tokens_in_batch})"
    )

    for _, (
        processor,
        batch_mask,
    ) in sampling_batch_info.custom_logit_processor.items():
        # Get the batch indices that need to be processed
        batch_indices = batch_mask.nonzero(as_tuple=True)[0]

        assert batch_mask.shape[0] == len(sampling_batch_info), (
            f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )
        batch_mask = torch.repeat_interleave(batch_mask, num_tokens_in_batch)

        # Apply the processor to the logits
        logits[batch_mask] = processor(
            logits[batch_mask],
            [sampling_batch_info.custom_params[i] for i in batch_indices],
        )

        logger.debug(
            f"Custom logit processor {processor.__class__.__name__} is applied."
        )
