"""Minimax-regret ceiling selection along the c_lambda family.

c_lam = lam * P(v_proxy) + (1-lam) * P_rated   (lam in [0,1])
  lam=0  -> intrinsic box (conservative)
  lam=1  -> fitted curve at proxy wind (tight but high FE)

For each lam we compute the two-extreme regret (U*=P_rated and U*=Y), then the
sup-regret = max of the two. The key question: is the minimizer lam* strictly
interior (0 < lam* < 1)? If yes, the risk-bound + decision route carries
information (a non-trivial robust ceiling beats the trivial box).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES, fit_curve, power_curve_pu  # noqa: E402


def regret_curve(site, horizon, seq_len=48, c_FE=1.0, c_FA=1.0, c_cons=1.0):
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

    seq = seq_len
    nn = len(ws_te) - seq - horizon + 1
    vl = np.zeros(nn, np.float32)
    Y = np.zeros((nn, horizon), np.float32)
    for i in range(nn):
        vl[i] = ws_te[i + seq - 1]
        Y[i] = pu(p_te)[i + seq:i + seq + horizon]
    Yf = Y.flatten()
    v_rep = vl.repeat(horizon)
    c_tight = power_curve_pu(v_rep, v_in, v_rated)

    rows = []
    for lam in np.linspace(0.0, 1.0, 21):
        c = lam * c_tight + (1.0 - lam) * 1.0          # c in [0,1]
        FE_lb = float((c < Yf).mean())
        FE_ub = float((c < 1.0).mean())
        FA_lb = float((Yf < c).mean())
        # FA_ub is always 0 since c <= 1 == P_rated
        conservatism = float((1.0 - c).mean())          # expected headroom above proxy curve

        # ---- correct regret baselines ----
        # Under U*=P_rated: the oracle-optimal ceiling is box (c=1): FE=0, cons=0.
        reg_rated = c_FE * FE_ub + c_cons * conservatism
        # Under U*=Y: the oracle-optimal ceiling is c=Y itself: FE=0, FA=0,
        # conservatism = E[1 - Y].  Regret of candidate c = its excess over that.
        cons_oracle_Y = float((1.0 - Yf).mean())
        reg_Y = c_FE * FE_lb + c_FA * FA_lb + c_cons * max(0.0, conservatism - cons_oracle_Y)
        sup_regret = max(reg_rated, reg_Y)
        rows.append(dict(site=site, horizon=horizon, lam=round(lam, 3),
                         FE_lb=FE_lb, FE_ub=FE_ub, FA_lb=FA_lb,
                         conservatism=conservatism, sup_regret=sup_regret))
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="wind_cn,kelmarsh,smarteole,penmanshiel,wind_us")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--c_FE", type=float, default=1.0)
    ap.add_argument("--c_FA", type=float, default=1.0)
    ap.add_argument("--c_cons", type=float, default=1.0)
    ap.add_argument("--out", default="regret_curves.csv")
    args = ap.parse_args()
    dfs = []
    for site in args.sites.split(","):
        dfs.append(regret_curve(site, args.horizon, c_FE=args.c_FE, c_FA=args.c_FA, c_cons=args.c_cons))
    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"saved {len(out)} rows -> {args.out}")
    print("\n=== minimax-regret 最优 lam*（0=box，1=tight，严格内部=有信息）===")
    for site in args.sites.split(","):
        sub = out[out.site == site]
        best = sub.loc[sub.sup_regret.idxmin()]
        print(f"  {site:12s} lam*={best.lam:.2f}  sup_regret={best.sup_regret:.4f}  "
              f"FE_lb={best.FE_lb:.3f} conservatism={best.conservatism:.3f}  "
              f"{'严格内部✅' if 0 < best.lam < 1 else '退化到端点⚠'}")


if __name__ == "__main__":
    main()
