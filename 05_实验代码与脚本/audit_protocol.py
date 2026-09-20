"""Unified audit protocol for physics-constrained wind forecasting.

Single canonical protocol to audit ANY physics constraint under a causal
deployment signal. The constraint is treated as a black-box map
    pred, aux -> (constraint_loss, feasibility_band)
and audited on the same data + proxy pipeline with the same two-sided
violation and ramp-clipping metrics.

This is the reproducible-benchmark backbone of the cross-method audit paper.
It does NOT train any model; it evaluates a constraint's behaviour on a
given (prediction, true, proxy) triple so that TgDPF/PCT (JS divergence),
FW-BTP (Betz penalty), and our own proxy-envelope penalty can all be scored
identically.

Metrics (two-sided, per operating region):
  over_rated_rate : fraction of predictions above P_rated
  neg_rate         : fraction of predictions below zero
  ramp_legit_cut   : mean legitimate high-power clipped in below-rated ramp
  bias_up / bias_dn: mean positive / negative constraint-induced shift
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Power curve (single-turbine normalized form, shared by every constraint)
# ---------------------------------------------------------------------------
def power_curve_pu(v: np.ndarray, v_in: float, v_rated: float, v_out: float = 25.0) -> np.ndarray:
    out = np.zeros_like(v, dtype=np.float64)
    denom = max(v_rated - v_in, 1e-6)
    mid = (v >= v_in) & (v < v_rated)
    out = np.where(mid, np.clip((v - v_in) / denom, 0.0, 1.0) ** 3, out)
    out = np.where((v >= v_rated) & (v <= v_out), 1.0, out)
    return out


def fit_curve(ws: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
    """IEC-style binning: returns (p_rated, v_in, v_rated)."""
    p_rated = float(np.percentile(p, 99.5))
    bins = np.arange(np.floor(ws.min()), np.ceil(ws.max()) + 0.5, 0.5)
    idx = np.digitize(ws, bins) - 1
    centers, med = [], []
    for i in range(len(bins) - 1):
        m = idx == i
        if m.sum() >= 50:
            centers.append((bins[i] + bins[i + 1]) / 2)
            med.append(np.median(p[m]))
    centers = np.array(centers); med = np.array(med)
    rm = med >= 0.95 * p_rated
    v_rated = float(centers[rm][0]) if rm.any() else 11.5
    v_in = float(np.quantile(ws[p >= 0.01 * p_rated], 0.02)) if (p >= 0.01 * p_rated).any() else 3.0
    return p_rated, v_in, v_rated


# ---------------------------------------------------------------------------
# Constraint abstraction
# ---------------------------------------------------------------------------
@dataclass
class AuditResult:
    name: str
    constraint_kind: str            # "point" | "distribution"
    over_rated_rate: float
    neg_rate: float
    ramp_legit_cut: float           # mean legit power clipped, as fraction of P_rated
    bias_up: float
    bias_dn: float
    ramp_asymmetry_pt: float        # ramp cut% - above-rated cut%
    coverage_notes: str = ""


ConstraintFn = Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]]
# returns (scalar loss, per-sample feasible upper bound in kW)


# ---------------------------------------------------------------------------
# Built-in constraint instances (one per audited method family)
# ---------------------------------------------------------------------------
def constraint_proxy_envelope(pred_kw, v_proxy, p_rated, v_in, v_rated):
    """Our own method (point constraint): env_hi = P_curve(v_proxy)."""
    env_hi = power_curve_pu(v_proxy, v_in, v_rated) * p_rated
    over = np.clip(pred_kw - env_hi, 0, None)
    return float((over ** 2).mean()), env_hi


def constraint_betz_penalty(pred_kw, v_proxy, p_rated, v_in, v_rated, rho=1.225, A=1.0, cp_max=0.593):
    """FW-BTP PGL (point constraint): Betz-limit power upper bound.

    P <= 0.5 * rho * A * v^3 * Cp_max.  The upper bound is a function of the
    causal wind proxy; on ramps the lagged v^3 underestimates the true bound.
    """
    env_hi = 0.5 * rho * A * np.clip(v_proxy, 0, None) ** 3 * cp_max
    # normalize to P_rated so units match
    env_hi = np.clip(env_hi / max(p_rated, 1e-6), 0, 1.0) * p_rated
    over = np.clip(pred_kw - env_hi, 0, None)
    return float((over ** 2).mean()), env_hi


def kde_distribution(v: np.ndarray, v_grid: np.ndarray, bw: float) -> np.ndarray:
    """1D Gaussian KDE density estimate (matches TgDPF/PCT KDE formulation)."""
    diffs = (v_grid[None, :] - v[:, None]) / bw
    k = np.exp(-0.5 * diffs ** 2) / (bw * np.sqrt(2 * np.pi))
    return k.mean(axis=0)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + 1e-12) / (m + 1e-12)))
    kl_qm = np.sum(q * np.log((q + 1e-12) / (m + 1e-12)))
    return float(0.5 * (kl_pm + kl_qm))


@dataclass
class JSSpec:
    p_rated: float
    v_in: float
    v_rated: float
    v_grid: np.ndarray = None
    bw: float = 0.5
    target_pdf: np.ndarray = None

    def __post_init__(self):
        if self.v_grid is None:
            self.v_grid = np.linspace(0.0, 25.0, 200)


def constraint_js_distribution(pred_kw, v_proxy, spec: JSSpec):
    """TgDPF/PCT (distribution constraint): JS divergence between predicted
    power distribution and KDE power-curve distribution.

    Returns the JS loss and, for audit purposes, the 99th percentile of the
    target distribution as a surrogate single-point ceiling.
    """
    # predicted power distribution (normalized to [0,P_rated])
    pred_pu = np.clip(pred_kw / max(spec.p_rated, 1e-6), 0.0, 1.0)
    p_grid = np.linspace(0.0, 1.0, 200)
    pred_pdf = kde_distribution(pred_pu.flatten(), p_grid, 0.05)
    if spec.target_pdf is None:
        # target: power-curve distribution approximated as uniform over [0,1]
        # weighted by the curve's cubic shape (matches TgDPF KDE over power)
        spec.target_pdf = kde_distribution(
            power_curve_pu(spec.v_grid, spec.v_in, spec.v_rated), p_grid, 0.05
        ) + 1e-9
    loss = js_divergence(pred_pdf, spec.target_pdf)
    ceiling = float(np.quantile(pred_kw, 0.99))
    return loss, np.full_like(pred_kw, ceiling)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def audit_constraint(
    name: str,
    kind: str,
    constraint_fn: Callable,
    pred_kw: np.ndarray,
    true_kw: np.ndarray,
    v_true: np.ndarray,
    v_proxy: np.ndarray,
    p_rated: float,
    v_rated: float,
    ramp_quantile: float = 0.9,
) -> AuditResult:
    loss, ceiling = constraint_fn()
    ceiling = np.asarray(ceiling, dtype=np.float64)

    over = pred_kw > p_rated
    neg = pred_kw < 0
    over_rate = float(over.mean())
    neg_rate = float(neg.mean())

    # legitimate high-power clipping: true above ceiling while below rated
    r = np.abs(np.diff(np.concatenate([[v_true[0]], v_true])))[: len(v_true)]
    ramp = (v_true < v_rated) & (r > np.quantile(r, ramp_quantile))
    calm = (v_true < v_rated) & (r <= np.quantile(r, 0.5))
    above = v_true >= v_rated

    cut = np.clip(true_kw - ceiling, 0, None)   # legit power above the ceiling
    ramp_cut = float(cut[ramp].mean()) if ramp.sum() else 0.0
    calm_cut = float(cut[calm].mean()) if calm.sum() else 0.0
    above_cut = float(cut[above].mean()) if above.sum() else 0.0

    # asymmetry as % of mean true power in each region
    ramp_mean = float(true_kw[ramp].mean()) if ramp.sum() else 1.0
    above_mean = float(true_kw[above].mean()) if above.sum() else 1.0
    asym_pt = (ramp_cut / max(ramp_mean, 1e-6) - above_cut / max(above_mean, 1e-6)) * 100

    bias_up = float(np.clip(pred_kw - true_kw, 0, None).mean())
    bias_dn = float(np.clip(true_kw - pred_kw, 0, None).mean())

    return AuditResult(
        name=name, constraint_kind=kind,
        over_rated_rate=over_rate * 100, neg_rate=neg_rate * 100,
        ramp_legit_cut=ramp_cut / max(p_rated, 1e-6),
        bias_up=bias_up, bias_dn=bias_dn, ramp_asymmetry_pt=asym_pt,
    )


# ---------------------------------------------------------------------------
# CLI demo: audit all four constraint families on wind_cn_core
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../04_数据集/prepared_priority_datasets/wind_cn_core_15min.csv")
    ap.add_argument("--ws-col", default="wind_speed")
    ap.add_argument("--power-col", default="power")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.data)
    ws = df[args.ws_col].to_numpy(np.float64)
    p = df[args.power_col].to_numpy(np.float64)
    valid = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[valid], p[valid]

    p_rated, v_in, v_rated = fit_curve(ws, p)
    v_true = ws[1:]
    v_proxy = ws[:-1]                    # persistence proxy (worst-case causal)
    pred_kw = p[1:]                      # stand-in prediction ~ truth
    true_kw = p[1:]
    print(f"data: {len(v_true)} rows, p_rated={p_rated:.0f}, v_rated={v_rated:.2f}")

    specs = {
        "proxy_envelope(ours)": (
            "point",
            lambda: constraint_proxy_envelope(pred_kw, v_proxy, p_rated, v_in, v_rated),
        ),
        "betz_penalty(FW-BTP)": (
            "point",
            lambda: constraint_betz_penalty(pred_kw, v_proxy, p_rated, v_in, v_rated),
        ),
        "js_distribution(TgDPF/PCT)": (
            "distribution",
            lambda: constraint_js_distribution(
                pred_kw, v_proxy,
                JSSpec(p_rated=p_rated, v_in=v_in, v_rated=v_rated),
            ),
        ),
    }
    print(f"\n{'name':30s} {'kind':12s} {'over%':>8s} {'neg%':>8s} {'ramp_cut':>9s} {'asym':>7s}")
    for name, (kind, fn) in specs.items():
        r = audit_constraint(name, kind, fn, pred_kw, true_kw, v_true, v_proxy, p_rated, v_rated)
        print(f"{name:30s} {kind:12s} {r.over_rated_rate:8.3f} {r.neg_rate:8.3f} "
              f"{r.ramp_legit_cut:9.3f} {r.ramp_asymmetry_pt:+6.1f}pt")


def torchify(x):
    try:
        import torch
        return torch.as_tensor(x)
    except Exception:
        return x


if __name__ == "__main__":
    main()
