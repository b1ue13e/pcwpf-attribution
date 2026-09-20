"""Betz-limit ceiling in the partial-identification interval framework.

Adds the FW-BTP-style Betz ceiling c_B(v) = min(1, 0.5*rho*A*Cp*v^3 / P_rated)
to the identification-interval audit. Note the honest caveat: with A=1 (no rotor
area), the per-unit Betz ceiling is (v/v_rated)^3, identical in shape to the
fitted cubic curve — so in the per-unit interval framework Betz vs fitted curve
differ only through the wind input, not the curve family. We report both.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES, fit_curve, power_curve_pu  # noqa: E402


def intervals(site, horizon, seq_len=48):
    df = pd.read_csv(SITES[site])
    ws = df["wind_speed"].to_numpy(np.float64)
    p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    n = len(ws)
    i1 = int(n * 0.8)
    ws_tr, p_tr = ws[:i1], p[:i1]
    ws_te, p_te = ws[i1:], p[i1:]

    p_rated, v_in, v_rated = fit_curve(ws_tr, p_tr)

    def pu(a):
        return np.clip(a / max(p_rated, 1e-6), 0.0, 1.0)

    def windows(ws_n, p_pu):
        nn = len(ws_n) - seq_len - horizon + 1
        vl = np.zeros(nn, np.float32)
        Y = np.zeros((nn, horizon), np.float32)
        for i in range(nn):
            vl[i] = ws_n[i + seq_len - 1]
            Y[i] = p_pu[i + seq_len:i + seq_len + horizon]
        return vl, Y

    v_last, Y = windows(ws_te, pu(p_te))
    Yf = Y.flatten()
    v_rep = v_last.repeat(horizon)

    # ceilings (per-unit)
    c_box = np.ones_like(Yf)
    c_fitted = power_curve_pu(v_rep, v_in, v_rated)
    # Betz per-unit: normalized by v_rated^3 so rated wind -> 1.0
    c_betz = np.clip(np.clip(v_rep, 0, None) / max(v_rated, 1e-6), 0, None) ** 3
    c_betz = np.clip(c_betz, 0, 1)

    def b(c):
        return (float((c < Yf).mean()), float((c < 1.0).mean()),
                float((Yf < c).mean()), float((1.0 < c).mean()))

    out = []
    for name, c in [("intrinsic_box", c_box), ("fitted_curve_proxy", c_fitted), ("betz_curve_proxy", c_betz)]:
        FE_lb, FE_ub, FA_lb, FA_ub = b(c)
        out.append(dict(site=site, horizon=horizon, ceiling=name,
                        FE_lb=FE_lb, FE_ub=FE_ub, FA_lb=FA_lb, FA_ub=FA_ub))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="wind_cn,kelmarsh,smarteole,penmanshiel,wind_us")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--out", default="betz_intervals.csv")
    args = ap.parse_args()
    rows = []
    for site in args.sites.split(","):
        rows += intervals(site, args.horizon)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"saved {len(out)} rows -> {args.out}")
    print("\n=== FE 识别区间（每站 × 3 ceiling）===")
    piv = out.pivot_table(index=["site", "ceiling"], values=["FE_lb", "FE_ub"])
    print(piv.round(4).to_string())
    print("\n=== FA 识别区间 ===")
    piv2 = out.pivot_table(index=["site", "ceiling"], values=["FA_lb", "FA_ub"])
    print(piv2.round(4).to_string())


if __name__ == "__main__":
    main()
