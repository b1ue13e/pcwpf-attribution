"""Numerical validation of the strict minimax-regret solution (rev. 2).

From MinimaxRegret_严格求解_rev2.md:
  - per-point oracle optimum is c* = u (latent), regret baseline l*(u)=c_cons*(R-u);
  - per-point minimax-regret optimum over u in [y, R] is  c* = y (realized power),
    with sup-regret = c_FE;
  - box (c=R) has sup-regret = c_FE + c_cons*(R - y)  (max conservatism);
  - a proxy ceiling c has sup-regret = c_FE + c_cons*(c - y) when c >= y,
    and  c_FE + c_cons*(c - y) again for c < y (so excess regret = c_cons*(y-c)^+
    for below-y samples; for above-y samples excess = c_cons*(c-y)).

Because c is sample-wise, we average the per-point sup-regret over the audit set:
    Regret_avg(c) = c_FE + c_cons * E[ (c - y)^+ + (y - c)^+ ]
                   = c_FE + c_cons * E[ |c - y| ]
(one-sided FE branch dominates when c < y; the FA/conservatism branch when c > y).

So the testable ordering is:  Regret(c=y) = c_FE  <=  Regret(proxy)  <  Regret(box),
with Regret(proxy) - c_FE = c_cons * E|proxy - y| (the mean absolute distance to y).
We report this in per-unit (divided by rated power R) so sites are comparable.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES  # noqa: E402

PMAX = {
    "wind_cn": 200.04933,
    "kelmarsh": 12321.450561523438,
    "smarteole": 14325.267666666668,
    "penmanshiel": 28698.034912109415,
    "wind_us": 16.0,
}


def power_curve_pu(v, v_in, v_rated, v_out=25.0):
    out = np.zeros_like(v, dtype=np.float64)
    denom = max(v_rated - v_in, 1e-6)
    mid = (v >= v_in) & (v < v_rated)
    out = np.where(mid, np.clip((v - v_in) / denom, 0.0, 1.0) ** 3, out)
    out = np.where((v >= v_rated) & (v <= v_out), 1.0, out)
    return out


def fit_curve_on_train(ws_tr, p_tr):
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


def regret_table(site, horizon, c_FE=1.0, c_cons=1.0, seq_len=48):
    df = pd.read_csv(SITES[site])
    ws = df["wind_speed"].to_numpy(np.float64)
    p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    n = len(ws)
    R = PMAX[site]
    i1 = int(n * 0.8)
    ws_tr, p_tr = ws[:i1], p[:i1]
    ws_te, p_te = ws[i1:], p[i1:]
    v_in, v_rated = fit_curve_on_train(ws_tr, p_tr)

    nn = len(ws_te) - seq_len - horizon + 1
    if nn < 1:
        return None
    vl = np.zeros(nn, np.float32)
    y = np.zeros((nn, horizon), np.float64)
    for i in range(nn):
        vl[i] = ws_te[i + seq_len - 1]
        y[i] = p_te[i + seq_len:i + seq_len + horizon]
    yf = y.flatten()
    v_rep = vl.repeat(horizon)

    # candidate ceilings in absolute power
    c_y = yf.copy()                                  # oracle minimax optimum
    c_fitted = power_curve_pu(v_rep, v_in, v_rated) * R
    c_cubic = (np.clip(v_rep, 0, None) / max(v_rated, 1e-6)) ** 3 * R
    c_box = np.full_like(yf, R)

    # per-unit regret = c_FE + c_cons * E| c - y | / R
    def regret(c):
        return c_FE + c_cons * float(np.abs(c - yf).mean() / R)

    return dict(site=site, horizon=horizon,
                reg_ceiling_y=regret(c_y),
                reg_fitted=regret(c_fitted),
                reg_cubic=regret(c_cubic),
                reg_box=regret(c_box),
                # excess regret = c_cons * E|proxy - y|/R
                excess_fitted=float(np.abs(c_fitted - yf).mean() / R),
                excess_cubic=float(np.abs(c_cubic - yf).mean() / R),
                excess_box=float(np.abs(c_box - yf).mean() / R))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="wind_cn,kelmarsh,smarteole,penmanshiel,wind_us")
    ap.add_argument("--horizons", default="6,12,24")
    ap.add_argument("--out", default="../tmp/minimax_regret_results.csv")
    args = ap.parse_args()
    rows = []
    for site in args.sites.split(","):
        for h in [int(x) for x in args.horizons.split(",")]:
            r = regret_table(site, h)
            if r:
                rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"saved {len(out)} rows -> {args.out}\n")
    print("=== per-unit regret（越小越优；c=y 基准 = c_FE=1.0）===")
    for site in args.sites.split(","):
        sub = out[out.site == site]
        print(f"\n--- {site} ---")
        for h in [6, 12, 24]:
            r = sub[sub.horizon == h].iloc[0]
            print(f"  h{h}: c=y {r.reg_ceiling_y:.3f} | fitted {r.reg_fitted:.3f} | "
                  f"cubic {r.reg_cubic:.3f} | box {r.reg_box:.3f}")
    print("\n=== 验证排序 c=y <= fitted <= cubic <= box（每 horizon）===")
    order_ok = 0; total = 0
    for _, r in out.iterrows():
        order = (r.reg_ceiling_y <= r.reg_fitted + 1e-9 and
                 r.reg_fitted <= r.reg_cubic + 1e-9 and
                 r.reg_cubic <= r.reg_box + 1e-9)
        total += 1; order_ok += order
    print(f"排序成立 {order_ok}/{total}")


if __name__ == "__main__":
    main()
