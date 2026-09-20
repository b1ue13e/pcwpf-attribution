"""PhysGATE: calibration-free physics-derived feasibility guardrail.

Route A design document: 08_报告与说明文档/路线A_PhysGATE_设计方案.md

Core idea (single change vs. the fitted FG-MoE proxy):
    v^up_{t+h} = v_t + q * sigma_delta * h^{eta},   eta = 0.5 * (1 + rho_hat)
    v^lo_{t+h} = v_t - q * sigma_delta * h^{eta}

- sigma_delta: robust std of wind-speed increments over a causal SCADA window
  (MAD-winsorized, flatline/stuck-sensor runs removed).
- rho_hat: lag-1 autocorrelation of the same increments, clipped to [0, 1].
- eta = 0.5 * (1 + rho_hat): interpolates the two physical limits
  (white-noise increments -> h^0.5 growth; persistent ramp -> h^1.0 growth).
- q = 2.0: the only universal design constant (one-sided Gaussian safety factor,
  1 - Phi(q) ~= 2.3%). No site-level calibration anywhere.

The guardrail penalizes
    max(0, -y)^2                                  (non-negativity)
  + max(0, y - 1)^2                              (rated power)
  + max(0, y - P_env(v_up))^2                    (upper statistical band)
  + 1{v_up < v_in or v_lo > v_out} * y^2         (whole band inside cut region)
where y is the forecast normalized by P_rated and P_env is the nameplate
power curve. Everything except y is computed from deployment-available SCADA
history and is detached from the autograd graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Universal design constants (NOT site parameters)
# ---------------------------------------------------------------------------
DEFAULT_Q = 2.0
DEFAULT_WINDOW = 96            # causal window in SCADA steps (24 h @ 15 min)
DEFAULT_MAX_FLAT_RUN = 4       # consecutive equal values = stuck/flatline
DEFAULT_MIN_VALID = 8          # fewer valid increments -> fall back to rho=0
DEFAULT_MAD_WINSOR = 3.0       # winsorize increments at median +/- 3*MAD
DEFAULT_MAX_WIND_MPS = 60.0    # hard physical clip for the band


# ---------------------------------------------------------------------------
# Turbine envelope (nameplate power curve)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TurbineEnvelope:
    """Nameplate constants. Taken from turbine documentation, never fitted."""

    p_rated: float      # rated power, any unit (forecast is divided by it)
    v_in: float         # cut-in wind speed
    v_rated: float      # rated wind speed
    v_out: float        # cut-out wind speed

    def validate(self) -> "TurbineEnvelope":
        if not (0.0 < self.v_in < self.v_rated < self.v_out):
            raise ValueError(
                f"Invalid envelope constants: need 0 < v_in < v_rated < v_out, "
                f"got v_in={self.v_in}, v_rated={self.v_rated}, v_out={self.v_out}"
            )
        if self.p_rated <= 0:
            raise ValueError(f"p_rated must be positive, got {self.p_rated}")
        return self


def power_envelope_pu(
    wind_speed: torch.Tensor,
    theta: TurbineEnvelope,
) -> torch.Tensor:
    """Normalized nameplate power curve P_env(v) / P_rated in [0, 1].

    Piecewise: 0 below cut-in; cubic ramp between cut-in and rated;
    1 between rated and cut-out; 0 above cut-out.
    """
    v_in, v_rated, v_out = theta.v_in, theta.v_rated, theta.v_out
    active = (wind_speed >= v_in) & (wind_speed < v_out)
    ratio = ((wind_speed - v_in) / (v_rated - v_in)).clamp(min=0.0, max=1.0)
    env = torch.where(active, ratio.pow(3.0), torch.zeros_like(wind_speed))
    return env


# ---------------------------------------------------------------------------
# Robust turbulence statistics (causal, online-computable, no fitting)
# ---------------------------------------------------------------------------
def _flat_run_mask(d: torch.Tensor, max_flat_run: int) -> torch.Tensor:
    """Mask increments that sit inside a run of >= max_flat_run equal values.

    d: (B, T) increments. Returns bool (B, T); False = part of a long stagnant
    (flatline / stuck-sensor) run and should be removed from statistics.
    Fully vectorized: run length = distance to the next non-zero increment,
    broadcast over the whole run.
    """
    z = d == 0.0                                 # (B, T): zero increment = stagnant
    T = z.size(1)
    # index of the NEXT non-zero increment (run end, exclusive); T if none follows.
    nz = (~z).float()                            # 1 at breaks
    rev_idx = torch.arange(T, 0, -1, device=d.device, dtype=d.dtype)  # T..1
    # next break AFTER or AT t: at position t if z[t]==0, else the first break to the right
    padded = torch.cat([nz, torch.zeros((z.size(0), 1), device=d.device, dtype=d.dtype)], dim=1)  # (B,T+1)
    pos = torch.arange(T + 1, device=d.device, dtype=d.dtype).unsqueeze(0)          # (1,T+1)
    val = torch.where(padded > 0, pos, torch.full_like(pos, float(T + 1)))
    # reverse-cummin: next break position per padded index j = first j'>=j with padded>0
    next_break = val.flip(dims=[1]).cummin(dim=1).values.flip(dims=[1])             # (B,T+1)
    end = next_break[:, :T]                       # (B,T): run end for increments
    run_len = end - torch.arange(T, device=d.device, dtype=d.dtype).unsqueeze(0)
    long_flat = z & (run_len >= float(max_flat_run))
    return ~long_flat


def rolling_turbulence_stats(
    wind_speed: torch.Tensor,
    *,
    window: int = DEFAULT_WINDOW,
    max_flat_run: int = DEFAULT_MAX_FLAT_RUN,
    min_valid: int = DEFAULT_MIN_VALID,
    mad_winsor: float = DEFAULT_MAD_WINSOR,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate (sigma_delta, rho_hat) from the trailing window of a wind series.

    Args:
        wind_speed: (B, L) wind-speed history (de-normalized, physical units).
    Returns:
        sigma_delta: (B,) robust std of winsorized increments.
        rho_hat: (B,) lag-1 autocorrelation of increments, clipped to [0, 1].
    All outputs are detached (treated as constants by the guardrail).
    """
    if wind_speed.dim() == 1:
        wind_speed = wind_speed.unsqueeze(0)
    v = wind_speed[:, -window:].float().detach()
    if v.size(1) < 2:
        return torch.zeros(v.size(0), device=v.device), torch.zeros(v.size(0), device=v.device)

    d = v[:, 1:] - v[:, :-1]                              # (B, T)
    valid = (~torch.isnan(d)) & _flat_run_mask(d, max_flat_run)
    cnt = valid.float().sum(dim=1)

    # NaN-free robust stats: fill invalid slots with +inf so that sort puts all
    # valid increments in the first `cnt` positions; then quantile winsorization
    # needs no nanmedian path (much faster).
    fill = torch.full_like(d, float("inf"))
    d_c = torch.where(valid, torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0), fill)
    d_sorted, _ = torch.sort(d_c, dim=1)                  # (B, T) valid first, then inf
    k_med = ((cnt - 1.0) // 2.0).clamp(min=0.0).long()
    med = d_sorted.gather(1, k_med.unsqueeze(1)).squeeze(1)
    med = torch.nan_to_num(med, nan=0.0, posinf=0.0, neginf=0.0)
    ad = (d_c - med.unsqueeze(1)).abs()
    ad_sorted, _ = torch.sort(ad, dim=1)                  # valid abs-dev first
    mad = ad_sorted.gather(1, k_med.unsqueeze(1)).squeeze(1)
    mad = torch.nan_to_num(mad, nan=0.0, posinf=0.0, neginf=0.0)
    lo = med - float(mad_winsor) * mad
    hi = med + float(mad_winsor) * mad

    d_w = torch.where(valid, torch.nan_to_num(d, nan=0.0), torch.zeros_like(d)).clamp(lo.unsqueeze(1), hi.unsqueeze(1))
    denom = cnt.clamp(min=1.0)
    mean_w = (d_w * valid.float()).sum(dim=1) / denom
    var_w = (((d_w - mean_w.unsqueeze(1)) ** 2) * valid.float()).sum(dim=1) / denom
    sigma = var_w.clamp(min=0.0).sqrt()

    # lag-1 autocorrelation of valid consecutive increments
    prev_valid = torch.cat([torch.zeros_like(valid[:, :1]), valid[:, :-1]], dim=1)
    pair = valid & prev_valid
    d_prev = torch.cat([torch.zeros_like(d[:, :1]), d_w[:, :-1]], dim=1)
    n = pair.float().sum(dim=1)
    denom_n = n.clamp(min=1.0)
    mx = (d_w * pair.float()).sum(dim=1) / denom_n
    my = (d_prev * pair.float()).sum(dim=1) / denom_n
    vx = (((d_w - mx.unsqueeze(1)) ** 2) * pair.float()).sum(dim=1) / denom_n
    vy = (((d_prev - my.unsqueeze(1)) ** 2) * pair.float()).sum(dim=1) / denom_n
    cov = ((d_w - mx.unsqueeze(1)) * (d_prev - my.unsqueeze(1)) * pair.float()).sum(dim=1) / denom_n
    denom_rho = (vx * vy).clamp(min=1e-12).sqrt()
    rho = torch.where(n < float(min_valid), torch.zeros_like(cov), cov / denom_rho)
    rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    return sigma, rho


def rolling_turbulence_stats_np(
    wind_speed: np.ndarray,
    *,
    window: int = DEFAULT_WINDOW,
    max_flat_run: int = DEFAULT_MAX_FLAT_RUN,
    min_valid: int = DEFAULT_MIN_VALID,
    mad_winsor: float = DEFAULT_MAD_WINSOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy mirror of :func:`rolling_turbulence_stats` for offline diagnostics."""
    x = np.asarray(wind_speed, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    x = x[:, -window:]
    B = x.shape[0]
    if x.shape[1] < 2:
        return np.zeros(B), np.zeros(B)
    d = x[:, 1:] - x[:, :-1]
    # flat-run mask: mirror of torch _flat_run_mask
    z = d == 0
    T = d.shape[1]
    mask = np.ones_like(z, dtype=bool)
    for b in range(B):
        t = 0
        while t < T:
            if z[b, t]:
                s = t
                while t < T and z[b, t]:
                    t += 1
                if t - s >= max_flat_run:
                    mask[b, s:t] = False
            else:
                t += 1
    valid = ~np.isnan(d) & mask

    sigma = np.zeros(B)
    rho = np.zeros(B)
    for b in range(B):
        dv = d[b][valid[b]]
        if dv.size < min_valid:
            # fallback: use whatever is available, no rho
            sigma[b] = float(np.std(dv)) if dv.size else 0.0
            continue
        med = np.median(dv)
        mad = np.median(np.abs(dv - med))
        lo, hi = med - mad_winsor * mad, med + mad_winsor * mad
        dw = np.clip(dv, lo, hi)
        sigma[b] = float(np.std(dw))
        vd = valid[b]
        prev = np.concatenate([[False], vd[:-1]])
        pair = vd & prev
        if pair.sum() >= min_valid:
            a = d[b][pair]
            b_prev = np.concatenate([[0.0], np.clip(d[b], lo, hi)[:-1]])[pair]
            va = np.var(a)
            vb = np.var(b_prev)
            if va > 1e-12 and vb > 1e-12:
                rho[b] = float(np.corrcoef(a, b_prev)[0, 1])
    rho = np.clip(np.nan_to_num(rho), 0.0, 1.0)
    return sigma, rho


# ---------------------------------------------------------------------------
# Adaptive band + guardrail loss
# ---------------------------------------------------------------------------
def output_endogenous_sym_loss(
    pred_pu: torch.Tensor,
    k: float = 2.0,
) -> torch.Tensor:
    """Output-endogenous symmetric feasibility (Option X, minimal form).

    Symmetric feasible band from the batch's OWN prediction distribution
    (mean +- k*std), instead of any wind-speed proxy. Penalizes both tails
    equally -> no net downward bias -> no violation migration. Hard constraints
    (pred > P_rated, pred < 0) stay in the caller.
    """
    mu = pred_pu.mean()
    sigma = pred_pu.std().clamp(min=1e-6)
    upper = mu + float(k) * sigma
    lower = mu - float(k) * sigma
    over = F.relu(pred_pu - upper)
    under = F.relu(lower - pred_pu)
    return (over ** 2 + under ** 2).mean()


def proxy_band(
    v_last: torch.Tensor,
    sigma_delta: torch.Tensor,
    rho_hat: torch.Tensor,
    horizon: int,
    *,
    q: float = DEFAULT_Q,
    max_wind_mps: float = DEFAULT_MAX_WIND_MPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-sided adaptive band around the last verified wind state.

    v_last: (B,), sigma_delta: (B,), rho_hat: (B,).
    Returns v_up, v_lo: (B, H).
    """
    eta = 0.5 * (1.0 + rho_hat)                      # (B,) in [0.5, 1.0]
    h = torch.arange(1, int(horizon) + 1, device=v_last.device, dtype=v_last.dtype).unsqueeze(0)
    band = float(q) * sigma_delta.unsqueeze(1) * h.pow(eta.unsqueeze(1))
    v_up = (v_last.unsqueeze(1) + band).clamp(0.0, float(max_wind_mps))
    v_lo = (v_last.unsqueeze(1) - band).clamp(min=0.0)
    return v_up, v_lo


def physgate_loss(
    pred_pu: torch.Tensor,
    v_last: torch.Tensor,
    sigma_delta: torch.Tensor,
    rho_hat: torch.Tensor,
    theta: TurbineEnvelope,
    *,
    q: float = DEFAULT_Q,
    symmetric: bool = False,
    return_terms: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Feasibility loss for normalized predictions y = y_hat / P_rated.

    pred_pu: (B, H). v_last/sigma_delta/rho_hat: (B,). theta: nameplate envelope.
    symmetric=True adds a lower-band envelope term max(0, P_env(v_lo) - pred)²
    so the guardrail is bounded both above and below.
    """
    horizon = pred_pu.size(1)
    v_up, v_lo = proxy_band(v_last, sigma_delta, rho_hat, horizon, q=q)
    env_up = power_envelope_pu(v_up, theta)                  # (B, H)
    env_lo = power_envelope_pu(v_lo, theta) if symmetric else None  # (B, H) or None

    term_nonneg = F.relu(-pred_pu).pow(2.0)
    term_rated = F.relu(pred_pu - 1.0).pow(2.0)
    term_env = F.relu(pred_pu - env_up).pow(2.0)
    term_env_lo = F.relu(env_lo - pred_pu).pow(2.0) if symmetric else torch.zeros_like(term_env)
    cut = ((v_up < theta.v_in) | (v_lo > theta.v_out)).float()
    term_cut = cut * pred_pu.pow(2.0)

    loss = term_nonneg + term_rated + term_env + term_env_lo + term_cut
    if return_terms:
        terms = {
            "nonneg": term_nonneg.mean(),
            "rated": term_rated.mean(),
            "env": term_env.mean(),
            "env_lo": term_env_lo.mean(),
            "cut": term_cut.mean(),
            "v_up": v_up,
            "v_lo": v_lo,
            "env_up": env_up,
        }
        return loss.mean(), terms
    return loss.mean()
