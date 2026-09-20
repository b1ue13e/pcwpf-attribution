"""DACE guardrail: Decision-aware Conformal Envelope.

Replaces the single-point heuristic proxy (v^proxy with 11 hand-set coefficients)
and the fixed-lambda2 squared physics penalty with:

  1. conformal wind interval [v_lo, v_hi] built from training one-step
     proxy residuals (symmetric, coverage 1-alpha);
  2. symmetric envelope penalty: BOTH over-forecast (above upper envelope)
     and under-forecast (below lower envelope) are penalized, with
     decision-aware asymmetric weights rho_up / rho_dn;
  3. Lagrangian dual ascent on lambda2: no fixed hyperparameter, the
     multiplier tracks a moving-average of the current violation.

This module is self-contained and does NOT import or modify the legacy
guardrail_proxy_utils / physics_loss path, so it can be A/B-tested before
the irreversible removal of the old FG-MoE code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Conformal interval
# ---------------------------------------------------------------------------
def conformal_half_width(
    residuals: np.ndarray,
    alpha: float = 0.1,
    clip_min: float = 0.0,
    clip_max: float = 6.0,
) -> float:
    """Symmetric conformal half-width from one-step proxy residuals.

    residuals: array of (v_proxy_{t+1} - v_true_{t+1}) on the training split.
    Coverage guarantee (marginal): P[|v_proxy - v_true| <= w] >= 1 - alpha.
    """
    r = np.abs(np.asarray(residuals, dtype=np.float64))
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float(clip_min)
    q = float(np.quantile(r, 1.0 - alpha))
    return float(np.clip(q, clip_min, clip_max))


def build_interval(
    proxy: torch.Tensor,
    half_width: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    v_lo = proxy - half_width
    v_hi = proxy + half_width
    return v_lo, v_hi


# ---------------------------------------------------------------------------
# Power envelope on an interval (vectorized, no single-point collapse)
# ---------------------------------------------------------------------------
def power_envelope_pu_interval(
    v_lo: torch.Tensor,
    v_hi: torch.Tensor,
    v_in: float,
    v_rated: float,
    v_out: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (P_env from v_hi = conservative upper bound,
                P_env from v_lo = permissive lower bound)."""
    denom = max(v_rated - v_in, 1e-6)

    def env(v: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(v)
        mid = (v >= v_in) & (v < v_rated)
        hi = (v >= v_rated) & (v <= v_out)
        out = torch.where(mid, torch.clamp((v - v_in) / denom, 0.0, 1.0) ** 3, out)
        out = torch.where(hi, torch.ones_like(out), out)
        return out

    return env(v_hi), env(v_lo)


# ---------------------------------------------------------------------------
# DACE config
# ---------------------------------------------------------------------------
@dataclass
class DACESpec:
    p_rated_kw: float = 1.0
    v_in_mps: float = 3.0
    v_rated_mps: float = 12.0
    v_out_mps: float = 25.0

    alpha: float = 0.1                   # conformal coverage
    rho_up: float = 1.0                  # decision-aware weight for over-forecast
    rho_dn: float = 1.0                  # decision-aware weight for under-forecast

    lambda_init: float = 0.03            # initial multiplier (replaces fixed lambda2)
    dual_lr: float = 1e-3                # Lagrangian dual ascent step
    lambda_floor: float = 1e-4
    lambda_cap: float = 5.0
    violation_ema_beta: float = 0.9

    half_width_mps: float = 1.0          # fallback if no training residuals

    # Reliability gate: down-weight the soft envelope penalty in high-ramp
    # windows where the causal wind proxy is least trustworthy.
    gate_mode: str = "none"              # "none" | "ramp"
    gate_gamma: float = 1.0              # single universal constant, 1/(m/s per step)


# ---------------------------------------------------------------------------
# Reliability gate (causal, deployable, single universal constant)
# ---------------------------------------------------------------------------
def compute_ramp_gate(
    wind_history: torch.Tensor,
    gamma: float = 1.0,
    lookback: int = 4,
    hard_slope: float = 0.0,
) -> torch.Tensor:
    """Per-sample gate weight w_t in [0,1] from the recent wind-speed slope.

    soft:  w_t = 1 / (1 + gamma * |slope_t|)
    hard:  if |slope_t| >= hard_slope, w_t = 0 (fully let go), else w_t = 1.
    """
    if wind_history.dim() == 1:
        wind_history = wind_history.unsqueeze(0)
    if wind_history.size(1) < 2:
        return torch.ones(wind_history.size(0), device=wind_history.device, dtype=wind_history.dtype)
    d = wind_history[:, 1:] - wind_history[:, :-1]
    slope = d[:, -lookback:].mean(dim=1).abs()
    if hard_slope > 0:
        return (slope < float(hard_slope)).float()
    return 1.0 / (1.0 + float(gamma) * slope)


# ---------------------------------------------------------------------------
# Loss with Lagrangian dual (stateful multiplier, EMA violation)
# ---------------------------------------------------------------------------
def dace_penalty(
    pred_kw: torch.Tensor,
    proxy_kw: torch.Tensor,
    spec: DACESpec,
    half_width: float,
    lambda_val: torch.Tensor,
    gate_weight: Optional[torch.Tensor] = None,
    rated_cap: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (loss, violation) where loss = lambda_val * asymmetric env penalty.

    pred_kw: normalized-energy predictions in kW (batch, H)
    proxy_kw: deployable wind proxy in m/s (batch, H), same shape as pred_kw
    lambda_val: scalar tensor, the current Lagrangian multiplier
    gate_weight: optional (B,) or (B,H) tensor in [0,1]; down-weights the soft
        envelope terms (over/under) in unreliable windows. Hard terms
        (pred > P_rated, pred < 0) are never gated.
    rated_cap: optional (B,) bool/float mask; where True, the upper envelope
        env_hi is replaced by P_rated (=1.0 pu). This is G1: fix the causal
        proxy's systematic UNDERESTIMATE on ramps by capping the envelope at
        rated power instead of trusting the lagged proxy envelope.
    """
    v_lo, v_hi = build_interval(proxy_kw, half_width)
    env_hi, env_lo = power_envelope_pu_interval(
        v_lo, v_hi, spec.v_in_mps, spec.v_rated_mps, spec.v_out_mps
    )
    if rated_cap is not None:
        cap = rated_cap.float().clamp(0.0, 1.0)
        if cap.dim() == 1:
            cap = cap.unsqueeze(1)
        # where ramping, raise upper envelope to 1.0 (rated) instead of lagged proxy env
        env_hi = env_hi + cap * torch.clamp(1.0 - env_hi, min=0.0)
    pred_pu = pred_kw / max(spec.p_rated_kw, 1e-6)

    over = torch.relu(pred_pu - env_hi)
    under = torch.relu(env_lo - pred_pu)
    hard = torch.relu(pred_pu - 1.0) + torch.relu(-pred_pu)

    if gate_weight is not None:
        if gate_weight.dim() == 1:
            gate_weight = gate_weight.unsqueeze(1)
        over = over * gate_weight
        under = under * gate_weight

    # decision-aware asymmetric weighting
    asym = spec.rho_up * (over ** 2) + spec.rho_dn * (under ** 2) + hard ** 2
    loss = torch.mean(asym)

    # violation signal for dual update: fraction of (batch*horizon) cells outside envelope
    viol = torch.mean(((over + under) > 0).float().reshape(-1)).detach()
    return loss, viol


def lagrange_dual_step(
    spec: DACESpec,
    lambda_val: torch.Tensor,
    violation_ema: torch.Tensor,
    violation: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dual ascent: lambda <- clip(lambda + dual_lr * violation_ema, floor, cap)."""
    beta = spec.violation_ema_beta
    new_ema = beta * violation_ema + (1.0 - beta) * violation
    new_lambda = lambda_val + spec.dual_lr * new_ema
    new_lambda = torch.clamp(new_lambda, spec.lambda_floor, spec.lambda_cap)
    return new_lambda, new_ema


# ---------------------------------------------------------------------------
# Self-test (no external deps beyond torch/numpy)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    spec = DACESpec(p_rated_kw=120000.0, v_in_mps=3.0, v_rated_mps=12.0, v_out_mps=25.0)
    # proxy over the operating range
    proxy = torch.linspace(0.0, 30.0, 24).unsqueeze(0)  # (1,24) m/s
    preds = torch.full_like(proxy, 90000.0)  # flat 90MW predictions
    half = 1.5
    lam = torch.tensor(0.03)
    viol_ema = torch.tensor(0.0)
    loss, viol = dace_penalty(preds, proxy, spec, half, lam)
    lam2, viol_ema2 = lagrange_dual_step(spec, lam, viol_ema, viol)
    print(f"self-test: loss={loss.item():.4f} viol={viol.item():.4f} "
          f"lambda {lam.item():.4f}->{lam2.item():.4f} ema={viol_ema2.item():.4f}")
    # check conformal half width
    residuals = np.random.randn(1000) * 2.0
    w = conformal_half_width(residuals, alpha=0.1)
    print(f"conformal half-width (alpha=0.1): {w:.3f}  (expect ~ 2*1.645=3.29)")

    # reliability gate correctness tests
    wind_calm = torch.tensor([[10.0, 10.1, 10.0, 10.1, 10.0, 10.1]])
    wind_ramp = torch.tensor([[9.0, 10.0, 11.0, 12.0, 13.0, 14.0]])
    w_calm = compute_ramp_gate(wind_calm)
    w_ramp = compute_ramp_gate(wind_ramp)
    assert w_calm.shape == (1,)
    assert float(w_calm) > 0.9, f"calm gate should be near 1, got {w_calm}"
    assert float(w_ramp) < 0.6, f"ramp gate should be small, got {w_ramp}"

    # gated penalty: high-ramp window should have strictly lower soft-envelope
    # loss than ungated for the same above-envelope prediction.
    proxy = torch.tensor([[11.0, 11.0, 11.0, 11.0]])
    preds = torch.tensor([[1.2, 1.2, 1.2, 1.2]]) * spec.p_rated_kw  # above rated (hard term dominates)
    # choose below-rated violation where soft env term dominates
    proxy_below = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    preds_below = torch.tensor([[0.9, 0.9, 0.9, 0.9]]) * spec.p_rated_kw  # above env(5 m/s) but below rated
    lam = torch.tensor(1.0)
    loss_ungated, viol_ungated = dace_penalty(preds_below, proxy_below, spec, 0.0, lam)
    loss_gated, viol_gated = dace_penalty(preds_below, proxy_below, spec, 0.0, lam, gate_weight=w_ramp)
    assert float(loss_gated) < float(loss_ungated), f"gated loss should be lower: {loss_gated} vs {loss_ungated}"
    print(f"gate test OK: calm w={float(w_calm):.3f}, ramp w={float(w_ramp):.3f}, "
          f"gated loss {float(loss_gated):.4f} < ungated {float(loss_ungated):.4f}")
