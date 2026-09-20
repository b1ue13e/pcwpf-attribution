"""Identification-interval audit — CORRECTED version (2026-09-03).

Fixes reviewer P0 items:
  P0-1  FA interval direction corrected:  {P_rated < c} ⊆ {U* < c} ⊆ {Y < c}
        => FA_lb = P(P_rated < c),  FA_ub = P(Y < c).
  P0-2  NO pseudo-seeds. ONE fixed 80/20 chronological split; uncertainty comes
        from day-level block bootstrap (resample calendar days of the AUDIT set).
  P0-3  P_rated = observed max power (power_max from dataset meta), NOT a 99.5
        percentile, and NO clipping of Y. We report the share of raw Y above the
        observed max (should be 0 by construction, but reported explicitly).
  P0-4  "Betz" ceiling renamed to cubic_proxy (v/v_rated)^3 clipped to [0,1];
        we make NO claim that it implements the Betz law.

Scope: pure data audit, no model training, no U* identification.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES  # noqa: E402

# observed max power per site (from dataset JSON power_max); NOT manufacturer rating
PMAX = {
    "wind_cn": 200.04933,
    "kelmarsh": 12321.450561523438,
    "smarteole": 14325.267666666668,
    "penmanshiel": 28698.034912109415,
    "wind_us": 16.0,
}


def fit_curve_on_train(ws_tr, p_tr):
    """IEC-style v_in/v_rated from the observational slice only (curve fitting is
    legitimate; the RATED POWER is NOT fitted — it comes from dataset power_max)."""
    bins = np.arange(np.floor(ws_tr.min()), np.ceil(ws_tr.max()) + 0.5, 0.5)
    idx = np.digitize(ws_tr, bins) - 1
    centers, med = [], []
    for i in range(len(bins) - 1):
        m = idx == i
        if m.sum() >= 50:
            centers.append((bins[i] + bins[i + 1]) / 2)
            med.append(np.median(p_tr[m]))
    centers = np.array(centers); med = np.array(med)
    p99 = float(np.percentile(p_tr, 99.5))
    rm = med >= 0.95 * p99
    v_rated = float(centers[rm][0]) if rm.any() else 11.5
    v_in = float(np.quantile(ws_tr[p_tr >= 0.01 * p99], 0.02)) if (p_tr >= 0.01 * p99).any() else 3.0
    return v_in, v_rated


def power_curve_pu(v, v_in, v_rated, v_out=25.0):
    out = np.zeros_like(v, dtype=np.float64)
    denom = max(v_rated - v_in, 1e-6)
    mid = (v >= v_in) & (v < v_rated)
    out = np.where(mid, np.clip((v - v_in) / denom, 0.0, 1.0) ** 3, out)
    out = np.where((v >= v_rated) & (v <= v_out), 1.0, out)
    return out


def audit_one(site, horizon, n_boot, seed, seq_len=48):
    df = pd.read_csv(SITES[site])
    ws = df["wind_speed"].to_numpy(np.float64)
    p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    n = len(ws)

    p_rated = PMAX[site]                            # observed max, NOT fitted, NO clip
    i1 = int(n * 0.8)                               # single fixed 80/20 split
    ws_tr, p_tr = ws[:i1], p[:i1]
    ws_te, p_te = ws[i1:], p[i1:]

    v_in, v_rated = fit_curve_on_train(ws_tr, p_tr)
    # report raw share above observed max (should be 0; explicit, per P0-3)
    share_above_pmax = float((p_te > p_rated).mean())

    # per-unit WITHOUT clipping: keep Y as-is, ceiling c as-is (in power units)
    def windows(ws_n, p_abs):
        nn = len(ws_n) - seq_len - horizon + 1
        vl = np.zeros(nn, np.float32)
        Y = np.zeros((nn, horizon), np.float64)
        for i in range(nn):
            vl[i] = ws_n[i + seq_len - 1]
            Y[i] = p_abs[i + seq_len:i + seq_len + horizon]
        return vl, Y

    v_last, Y_abs = windows(ws_te, p_te)
    if len(v_last) == 0:
        return None
    Yf = Y_abs.flatten()
    v_rep = v_last.repeat(horizon)

    # candidate ceilings in ABSOLUTE power units (same as Y)
    c_box = np.full_like(Yf, p_rated)
    c_fitted = power_curve_pu(v_rep, v_in, v_rated) * p_rated
    c_cubic = np.clip(np.clip(v_rep, 0, None) / max(v_rated, 1e-6), 0, None) ** 3 * p_rated

    def bounds(c, idx=None):
        if idx is not None:
            c = c[idx]; y = Yf[idx]
        else:
            y = Yf
        return (float((c < y).mean()),                    # FE_lb
                float((c < p_rated).mean()),               # FE_ub = P(c < P_rated)
                float((p_rated < c).mean()),               # FA_lb = P(P_rated < c)
                float((y < c).mean()))                     # FA_ub = P(Y < c)

    # point estimates + day-level block bootstrap (on the AUDIT set)
    rng = np.random.default_rng(seed)
    # reconstruct day index for the flattened audit samples (approx: 96 obs/day at 15-min)
    n_days = max(len(Yf) // (96 * horizon), 1)
    day_id = np.repeat(np.arange(n_days), len(Yf) // n_days)[:len(Yf)]

    out = []
    for name, c in [("intrinsic_box", c_box), ("fitted_curve_proxy", c_fitted), ("cubic_proxy", c_cubic)]:
        pt = bounds(c)
        boots = []
        for _ in range(n_boot):
            idx = []
            for _ in range(n_days):
                d = rng.integers(0, n_days)
                idx.append(np.where(day_id == d)[0])
            if not idx:
                continue
            idx = np.concatenate(idx)
            boots.append(bounds(c, idx))
        boots = np.array(boots)
        out.append(dict(site=site, horizon=horizon,
                        ceiling=name,
                        FE_lb=pt[0], FE_ub=pt[1], FA_lb=pt[2], FA_ub=pt[3],
                        FE_lb_lo=float(np.percentile(boots[:, 0], 2.5)),
                        FE_lb_hi=float(np.percentile(boots[:, 0], 97.5)),
                        v_in=round(v_in, 3), v_rated=round(v_rated, 3),
                        p_rated=round(float(p_rated), 1),
                        share_above_pmax=round(share_above_pmax, 6)))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="wind_cn,kelmarsh,smarteole,penmanshiel,wind_us")
    ap.add_argument("--horizons", default="6,12,24")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="../tmp/identification_intervals_corrected.csv")
    args = ap.parse_args()
    rows = []
    for site in args.sites.split(","):
        for h in [int(x) for x in args.horizons.split(",")]:
            r = audit_one(site, h, args.n_boot, args.seed)
            if r:
                rows += r
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"saved {len(out)} rows -> {args.out}")
    print("\n=== 识别区间（修正版：FA 界已交换，95% bootstrap CI on FE_lb）===")
    for site in args.sites.split(","):
        sub = out[out.site == site]
        print(f"\n--- {site} (p_rated={sub.p_rated.iloc[0]}) ---")
        for h in [6, 12, 24]:
            for name in ["intrinsic_box", "fitted_curve_proxy", "cubic_proxy"]:
                r = sub[(sub.horizon == h) & (sub.ceiling == name)].iloc[0]
                print(f"  h{h} {name:20s} FE=[{r.FE_lb:.3f},{r.FE_ub:.3f}] (lb CI {r.FE_lb_lo:.3f}-{r.FE_lb_hi:.3f}) "
                      f"FA=[{r.FA_lb:.3f},{r.FA_ub:.3f}]"  )


if __name__ == "__main__":
    main()
