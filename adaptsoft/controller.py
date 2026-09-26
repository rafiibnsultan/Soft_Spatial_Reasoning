"""AdaptSoft controller f_phi.

At each reasoning step the controller reads a fixed random projection of the layer-normalised
hidden state together with the normalised top-k entropy, and emits a temperature

    u_t   = f_phi([P LN(h_t) || Hhat_t])
    tau_t = tau_base + tau_delta * tanh(u_t)

The projection is applied at rollout (`project`) so the controller input can be recorded and the
temperature recomputed at update time from recorded tensors alone.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

PROJECTION_SEED = 20260901


class AdaptSoftController(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        proj_dim: int = 8,
        mid: int = 256,
        h_mean: float = 0.0,
        h_std: float = 1.0,
        tau_base: float = 0.5,
        tau_delta: float = 0.4,
        b_init: float = 0.0,
        log_sigma_init: float = math.log(0.5),
    ):
        super().__init__()
        assert proj_dim > 0 and tau_delta > 0
        self.proj_dim = int(proj_dim)
        self.h_mean = float(h_mean)
        self.h_std = float(h_std) if abs(float(h_std)) > 1e-8 else 1.0
        self.tau_base = float(tau_base)
        self.tau_delta = float(tau_delta)

        self.register_buffer("f_mean", torch.tensor([self.h_mean], dtype=torch.float32), persistent=False)
        self.register_buffer("f_std", torch.tensor([self.h_std], dtype=torch.float32), persistent=False)

        self.ln = nn.LayerNorm(hidden_size)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(PROJECTION_SEED)
            _P = torch.randn(self.proj_dim, hidden_size) / math.sqrt(hidden_size)
        self.register_buffer("hproj", _P, persistent=False)

        self.fc1 = nn.Linear(self.proj_dim + 1, mid)
        self.act = nn.GELU()
        # The head emits (mu, log_sigma); the deterministic controller reads mu.
        self.fc2 = nn.Linear(mid, 2)
        nn.init.normal_(self.fc2.weight, std=0.02)
        with torch.no_grad():
            self.fc2.bias.copy_(torch.tensor([0.0, float(log_sigma_init)]))

        # Additive entropy term, the only part of mu computable from Hhat alone.
        self.ent_a = nn.Parameter(torch.tensor(0.0))
        self.ent_b = nn.Parameter(torch.tensor(float(b_init)))

    def _whiten(self, h_hat: torch.Tensor) -> torch.Tensor:
        return (h_hat - self.f_mean.to(h_hat.dtype)) / self.f_std.to(h_hat.dtype)

    @torch.no_grad()
    def project(self, h: torch.Tensor) -> torch.Tensor:
        """[..., D] hidden state -> [..., proj_dim]. LayerNorm first, then the fixed projection."""
        h = h.detach().to(self.ln.weight.dtype)
        return torch.nn.functional.linear(self.ln(h), self.hproj.to(h.dtype))

    def entropy_prior(self, h_hat: torch.Tensor) -> torch.Tensor:
        z = self._whiten(h_hat.to(self.ent_a.dtype))
        return self.ent_a * z[..., 0:1] + self.ent_b

    def mu(self, z: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
        """Controller output from the projected hidden state `z` and the entropy `h_hat`."""
        z = z.detach().to(self.fc1.weight.dtype)
        h_hat = h_hat.to(self.fc1.weight.dtype)
        x = torch.cat([z, self._whiten(h_hat)], dim=-1)
        out = self.fc2(self.act(self.fc1(x)))
        return out[..., 0:1] + self.entropy_prior(h_hat)

    def tau_from_u(self, u: torch.Tensor) -> torch.Tensor:
        return self.tau_base + self.tau_delta * torch.tanh(u)

    @torch.no_grad()
    def infer(self, z: torch.Tensor, h_hat: torch.Tensor):
        """Rollout path: returns (u, tau) with no sampling."""
        u = self.mu(z, h_hat)
        return u, self.tau_from_u(u)

    def extra_repr(self) -> str:
        return (f"tau={self.tau_base}+{self.tau_delta}*tanh(u), proj_dim={self.proj_dim}, "
                f"h_mean={self.h_mean}, h_std={self.h_std}")
