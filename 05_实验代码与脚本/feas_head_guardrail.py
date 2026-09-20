"""G2 wrapper: feasibility-head guardrail (output-endogenous upper bound).

Wraps an UltraLSNTForecaster, reusing its encoder + blocks to extract the
hidden representation, then adds a lightweight FeasibilityHead that predicts
the per-sample 90th quantile upper bound y_up. The guardrail penalty uses
min(P_rated, y_up) as the upper envelope on ramp windows instead of the
lagged wind-proxy envelope, bypassing proxy lag structurally.

The head is trained with a pinball (quantile) loss with target quantile q=0.9,
so y_up is an ENDOGENOUS, self-calibrated feasible upper bound.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultra_lsnt_timeseries import TemporalEncoder, UltraLSNTBlock


class FeasibilityHead(nn.Module):
    def __init__(self, hidden_dim: int, pred_len: int):
        super().__init__()
        dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, pred_len),
            nn.Softplus(),  # y_up >= 0
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (batch, hidden_dim) -> y_up: (batch, pred_len)
        return self.net(h)


class FeasHeadUltra(nn.Module):
    """Ultra-LSNT + FeasibilityHead; forward returns (pred, aux, y_up)."""

    def __init__(self, base: nn.Module, input_dim: int, hidden_dim: int, seq_len: int, pred_len: int):
        super().__init__()
        self.encoder = TemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len)
        self.blocks = base.blocks  # reuse the base's forward helper is avoided; keep encoder+blocks
        self.decoder = base.decoder
        self.feas_head = FeasibilityHead(hidden_dim, pred_len)
        # copy encoder weights if base encoder has same shape
        if hasattr(base, "encoder"):
            try:
                self.encoder.load_state_dict(base.encoder.state_dict(), strict=False)
                # use base's blocks directly instead of a fresh copy
                self.blocks = base.blocks
                self.decoder = base.decoder
                self.encoder = base.encoder
            except Exception:
                pass

    def forward(self, x, temperature=1.0):
        x = self.encoder(x)
        total_aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x, stats = block(x, temperature)
            total_aux = total_aux + stats["aux_loss"]
        pred = self.decoder(x)  # (batch, pred_len) for output_dim=1
        if pred.dim() == 3 and pred.size(-1) == 1:
            pred = pred.squeeze(-1)
        y_up = self.feas_head(x)  # (batch, pred_len)
        return pred, total_aux, y_up


def feas_head_guardrail_loss(
    pred_kw: torch.Tensor,
    y_up_kw: torch.Tensor,
    p_rated_kw: float,
    *,
    ramp_mask: torch.Tensor | None = None,
    q: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Guardrail loss using endogenous upper bound min(P_rated, y_up) on ramps.

    pred_kw: (B, H) forecasts in kW;  y_up_kw: (B, H) endogenous upper bound.
    ramp_mask: (B,) in {0,1}; where 1, use min(P_rated, y_up) as the ceiling.

    Returns (guard_loss, pinball_loss).
    """
    pred_pu = pred_kw / max(p_rated_kw, 1e-6)
    y_up_pu = y_up_kw / max(p_rated_kw, 1e-6)

    # endogenous upper envelope: on ramps use min(P_rated, y_up); elsewhere P_rated
    ceiling = torch.ones_like(pred_pu)
    if ramp_mask is not None:
        m = ramp_mask.float().clamp(0.0, 1.0).unsqueeze(1)
        ceiling = m * torch.minimum(ceiling, y_up_pu) + (1.0 - m) * ceiling

    over = F.relu(pred_pu - ceiling)
    hard = F.relu(pred_pu - 1.0)  # P_rated is always a hard cap
    guard_loss = (over ** 2).mean() + (hard ** 2).mean()

    # quantile/pinball supervision for the feasibility head
    err = pred_pu - y_up_pu
    pinball = torch.where(err > 0, q * err, (q - 1.0) * err).mean()
    return guard_loss, pinball
