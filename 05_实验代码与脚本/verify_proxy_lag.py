"""Verify the proxy-lag fundamental limit: env error ~= |dp/dv| * r * dt.

Self-contained (no legacy harness). Reads wind_final.csv directly.

Proposition being tested:
  A persistence wind proxy lags true wind by dt (one SCADA step).
  The power-envelope error  e = |P_env(v_proxy) - P_env(v_true)|  scales as
  |dp/dv| * |dv/dt| * dt, so it is LARGEST on high-ramp segments (where r
  is large) and, because dp/dv>0 inside [v_in, v_rated], the envelope is
  systematically UNDERESTIMATED when wind is rising -> clipping kills
  legitimate high power.

Outputs:
  1. optimal proxy lag (cross-correlation)
  2. e vs ramp-rate r scatter summary (binned means) -> monotone?
  3. bias- after clip, split by operating region (below/above rated)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

V_IN, V_RATED, V_OUT = 3.0, 11.5, 25.0   # placeholder; will be overridden by curve fit


def power_curve(v: np.ndarray, v_in=V_IN, v_rated=V_RATED, v_out=V_OUT):
    out = np.zeros_like(v, dtype=np.float64)
    mid = (v >= v_in) & (v < v_rated)
    hi = (v >= v_rated) & (v <= v_out)
    denom = max(v_rated - v_in, 1e-6)
    out = np.where(mid, np.clip((v - v_in) / denom, 0.0, 1.0) ** 3, out)
    out = np.where(hi, 1.0, out)
    return out  # normalized [0,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../04_数据集/prepared_priority_datasets/wind_cn_core_15min.csv")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--ws-col", default="wind_speed")
    ap.add_argument("--power-col", default="power")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if args.max_rows:
        df = df.iloc[: args.max_rows]
    ws = df[args.ws_col].to_numpy(dtype=np.float64)
    p = df[args.power_col].to_numpy(dtype=np.float64)
    # drop physically invalid wind speeds (stuck-sensor -99 etc.)
    valid = (ws >= 0.0) & (ws <= 40.0) & np.isfinite(ws) & np.isfinite(p)
    ws = ws[valid]
    p = p[valid]
    print(f"[data] {args.data}: {len(ws)} valid rows after filtering wind_speed")

    # ---- true wind series; persistence proxy = lag-1 ----
    v_true = ws[1:]
    v_proxy = ws[:-1]                 # persistence: proxy_t+1 = v_t  => lag dt=1 step

    # ---- fit power-curve rated params from data (IEC-style binning) ----
    p_rated = float(np.percentile(p, 99.5))
    # find v_rated: first wind speed where median power >= 0.95*p_rated
    bins = np.arange(np.floor(ws.min()), np.ceil(ws.max()) + 0.5, 0.5)
    idx = np.digitize(ws, bins) - 1
    centers, med = [], []
    for i in range(len(bins) - 1):
        m = idx == i
        if m.sum() >= 50:
            centers.append((bins[i] + bins[i + 1]) / 2)
            med.append(np.median(p[m]))
    centers = np.array(centers)
    med = np.array(med)
    rated_mask = med >= 0.95 * p_rated
    v_rated = float(centers[rated_mask][0]) if rated_mask.any() else 11.5
    v_in = float(np.quantile(ws[p >= 0.01 * p_rated], 0.02)) if (p >= 0.01 * p_rated).any() else 3.0
    print(f"[curve] p_rated={p_rated:.0f} v_in={v_in:.2f} v_rated={v_rated:.2f}")

    # ---- envelope on true and proxy wind ----
    def env(v):
        return power_curve(v, v_in, v_rated, V_OUT) * p_rated  # kW

    e_true = env(v_true)
    e_proxy = env(v_proxy)
    env_err = np.abs(e_true - e_proxy)

    # ---- ramp rate on true wind ----
    r = np.abs(np.diff(ws))          # |dv/dt| per step, aligned to v_true (shift by 1)
    r = r[: len(v_true)]

    # ---- envelope slope dp/dv (numeric) on true wind ----
    dv = 1e-2
    dpdv = np.abs((env(v_true + dv) - env(v_true - dv)) / (2 * dv))

    # ---- theory prediction ----
    theory = dpdv * r * 1.0          # dt = 1 step

    # ---- 1. optimal lag via cross-correlation ----
    n = len(v_true)
    lags = range(-12, 13)
    best_lag, best_corr = 0, -1.0
    for lag in lags:
        if lag >= 0:
            a, b = v_true[lag:], ws[: n - lag] if lag else ws[:n]
        else:
            a, b = v_true[: n + lag], ws[-lag:]
        m = min(len(a), len(b))
        if m < 100:
            continue
        c = np.corrcoef(a[:m], b[:m])[0, 1]
        if c > best_corr:
            best_corr, best_lag = c, lag
    print(f"[lag] best proxy lag = {best_lag} steps, corr={best_corr:.4f} (persistence => 1)")

    # ---- 2. env error vs ramp-rate (binned, monotone?) ----
    print("\n[env error vs ramp rate] (theory slope ~= |dp/dv|*dt)")
    qs = np.quantile(r, [0.0, 0.5, 0.75, 0.9, 0.99, 1.0])
    print("ramp quantiles (m/s per step):", np.round(qs, 3))
    edges = np.quantile(r, [0, 0.5, 0.75, 0.9, 0.99, 1.0])
    for i in range(len(edges) - 1):
        m = (r >= edges[i]) & (r < edges[i + 1]) if i < len(edges) - 2 else (r >= edges[i])
        if m.sum() < 20:
            continue
        print(f"  r in [{edges[i]:.3f},{edges[i+1]:.3f}): n={m.sum():6d} "
              f"mean env_err={env_err[m].mean():8.1f} kW  mean theory={theory[m].mean():8.1f} kW")

    # correlation between env_err and theory
    fine = np.isfinite(theory) & np.isfinite(env_err)
    corr_et = np.corrcoef(theory[fine], env_err[fine])[0, 1]
    print(f"[verify] corr(theory, env_err) = {corr_et:.3f}")

    # ---- 3. clip bias- by operating region ----
    print("\n[clip bias] persistence-proxy envelope, clip predictions to env(proxy)")
    # use true power as a stand-in "prediction before clip" (physically: pred ~ truth)
    pred = p[1:]                                        # aligned with v_true
    env_proxy = env(v_proxy)
    clipped = np.minimum(pred, env_proxy)
    under = pred - clipped                              # >=0, how much legitimate power is cut
    # operating region by true wind
    region_hi = v_true >= v_rated                       # above rated
    region_ramp = (v_true < v_rated) & (r > np.quantile(r, 0.9))
    region_calm = (v_true < v_rated) & (r <= np.quantile(r, 0.5))
    for name, m in [("below-rated high-ramp", region_ramp), ("below-rated calm", region_calm), ("above-rated", region_hi)]:
        print(f"  {name:22s}: n={m.sum():6d}  mean cut={under[m].mean():9.1f} kW  "
              f"total cut={under[m].sum():11.0f} kW  mean true={pred[m].mean():9.1f} kW")


if __name__ == "__main__":
    main()
