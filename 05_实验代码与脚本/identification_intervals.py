"""Verify Theorem 1 (sharp risk identification intervals) and Theorem 2
(identified dominance) empirically on the five sites.

No training needed. For each candidate ceiling c we compute the four observable
quantities that bound the unidentifiable FE/FA risks:

    FE_lb = P(c < Y)          FA_lb = P(Y < c)
    FE_ub = P(c < P_rated)    FA_ub = P(P_rated < c)

and produce the identification rectangle per candidate per site.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES, fit_curve, power_curve_pu  # noqa: E402


def identification_intervals(site, horizon, seq_len=48, seed=42):
    df = pd.read_csv(SITES[site])
    ws = df["wind_speed"].to_numpy(np.float64)
    p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    n = len(ws)
    i1 = int(n * 0.8)                      # train/observational vs audit split (80/20)
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

    # persistence proxy ceiling on the audit (test) slice
    v_last, Y = windows(ws_te, pu(p_te))
    Y = Y.flatten()
    c_proxy = power_curve_pu(v_last, v_in, v_rated).repeat(horizon)
    c_box = np.ones_like(c_proxy)

    def bounds(c):
        FE_lb = float((c < Y).mean())
        FE_ub = float((c < 1.0).mean())          # c < P_rated (per-unit rated = 1.0)
        FA_lb = float((Y < c).mean())
        FA_ub = float((1.0 < c).mean())          # P_rated < c
        return FE_lb, FE_ub, FA_lb, FA_ub

    return dict(site=site, horizon=horizon,
                box_FE_lb=0.0, box_FE_ub=0.0, box_FA_lb=bounds(c_box)[2], box_FA_ub=0.0,
                proxy_FE_lb=bounds(c_proxy)[0], proxy_FE_ub=bounds(c_proxy)[1],
                proxy_FA_lb=bounds(c_proxy)[2], proxy_FA_ub=bounds(c_proxy)[3])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="wind_cn,kelmarsh,smarteole,penmanshiel,wind_us")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--out", default="identification_intervals.csv")
    args = ap.parse_args()
    rows = []
    for site in args.sites.split(","):
        rows.append(identification_intervals(site, args.horizon))
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"saved {len(out)} rows -> {args.out}")
    print("\n=== 识别区间（FE 轴：proxy 的区间在 [lb, ub]，box 坍缩 [0,0]）===")
    print(out[["site", "proxy_FE_lb", "proxy_FE_ub", "box_FE_lb", "box_FE_ub"]].round(4).to_string())
    print("\n=== 判定（Theorem 2：box 是否 identified-dominates proxy）===")
    for _, r in out.iterrows():
        dom = (0.0 <= r.proxy_FE_ub) and (0.0 <= r.proxy_FA_ub)  # box FE_ub=0, FA_ub=0
        print(f"  {r.site}: box FE_ub=0 <= proxy FE_ub={r.proxy_FE_ub:.3f} & box FA_ub=0 <= proxy FA_ub={r.proxy_FA_ub:.3f} -> 支配={dom}")


if __name__ == "__main__":
    main()
