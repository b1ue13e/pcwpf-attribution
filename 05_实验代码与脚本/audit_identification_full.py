"""Unified identification-interval audit: multi-horizon x multi-seed.

Honest scope (partial identification):
  - NO model training; pure data audit of deterministic ceilings.
  - seed = different observational/audit split offset (no resampling of data);
    this quantifies audit-split sensitivity, not model randomness.
  - FE_ub and FA_ub for c <= P_rated: FA_ub = P(P_rated < c) = 0 identically
    (reported as 0.0, a structural property of the identification bounds, not a bug).

For each (site, horizon, seed) and each candidate ceiling c we compute the four
bounding observables:
    FE_lb = P(c < Y)          FE_ub = P(c < P_rated)
    FA_lb = P(Y < c)          FA_ub = P(P_rated < c)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES, fit_curve, power_curve_pu  # noqa: E402

SITE_META = {
    # each site: (split_fraction, name)
    "wind_cn": (0.8, "wind_cn"),
    "kelmarsh": (0.8, "kelmarsh"),
    "smarteole": (0.8, "smarteole"),
    "penmanshiel": (0.8, "penmanshiel"),
    "wind_us": (0.8, "wind_us"),
}


def audit_one(site, horizon, seed, seq_len=48):
    df = pd.read_csv(SITES[site])
    ws = df["wind_speed"].to_numpy(np.float64)
    p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    n = len(ws)

    # seed = split offset (vary the observational/audit boundary by a few %)
    frac = SITE_META[site][0]
    offset = (seed % 5) * 0.02            # 0..4 -> 0%, 2%, 4%, 6%, 8% shift
    frac_eff = min(frac + offset, 0.9)
    i1 = int(n * frac_eff)
    ws_tr, p_tr = ws[:i1], p[:i1]
    ws_te, p_te = ws[i1:], p[i1:]

    p_rated, v_in, v_rated = fit_curve(ws_tr, p_tr)

    def pu(a):
        return np.clip(a / max(p_rated, 1e-6), 0.0, 1.0)

    def windows(ws_n, p_pu):
        nn = len(ws_n) - seq_len - horizon + 1
        if nn < 1:
            return np.zeros(0), np.zeros((0, horizon))
        vl = np.zeros(nn, np.float32)
        Y = np.zeros((nn, horizon), np.float32)
        for i in range(nn):
            vl[i] = ws_n[i + seq_len - 1]
            Y[i] = p_pu[i + seq_len:i + seq_len + horizon]
        return vl, Y

    v_last, Y = windows(ws_te, pu(p_te))
    if len(v_last) == 0:
        return None
    Yf = Y.flatten()
    v_rep = v_last.repeat(horizon)

    # candidate ceilings (per-unit)
    c_box = np.ones_like(Yf)
    c_fitted = power_curve_pu(v_rep, v_in, v_rated)
    c_betz = np.clip(np.clip(v_rep, 0, None) / max(v_rated, 1e-6), 0, None) ** 3
    c_betz = np.clip(c_betz, 0, 1)

    def bounds(c):
        return dict(FE_lb=float((c < Yf).mean()),
                    FE_ub=float((c < 1.0).mean()),
                    # FA interval (corrected direction, 2026-09-03):
                    #   {P_rated < c} ⊆ {U* < c} ⊆ {Y < c}
                    # => FA_lb = P(P_rated < c),  FA_ub = P(Y < c)
                    FA_lb=float((1.0 < c).mean()),
                    FA_ub=float((Yf < c).mean()))

    rows = []
    for name, c in [("intrinsic_box", c_box), ("fitted_curve_proxy", c_fitted), ("betz_curve_proxy", c_betz)]:
        r = bounds(c)
        r.update(site=site, horizon=horizon, seed=seed, ceiling=name,
                 v_in=round(v_in, 3), v_rated=round(v_rated, 3), p_rated=round(float(p_rated), 1))
        rows.append(r)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="wind_cn,kelmarsh,smarteole,penmanshiel,wind_us")
    ap.add_argument("--horizons", default="6,12,24")
    ap.add_argument("--seeds", default="42,123,456,789,2024")
    ap.add_argument("--out", default="../tmp/identification_intervals_full.csv")
    args = ap.parse_args()
    rows = []
    for site in args.sites.split(","):
        for h in [int(x) for x in args.horizons.split(",")]:
            for s in [int(x) for x in args.seeds.split(",")]:
                r = audit_one(site, h, s)
                if r:
                    rows += r
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"saved {len(out)} rows -> {args.out}")
    print("\n=== FE 区间（mean over seed，每 site x horizon x ceiling）===")
    piv = out.pivot_table(index=["site", "horizon", "ceiling"], values=["FE_lb", "FE_ub"], aggfunc="mean").round(4)
    print(piv.to_string())
    print("\n=== FA_lb 区间（mean over seed）===")
    piv2 = out.pivot_table(index=["site", "horizon", "ceiling"], values=["FA_lb"], aggfunc="mean").round(4)
    print(piv2.to_string())


if __name__ == "__main__":
    main()
