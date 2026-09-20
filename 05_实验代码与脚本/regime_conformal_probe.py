"""Regime-conditional conformal probe.

Verifies that regime-conditional split conformal fixes two defects of the pooled
version:
  1. under-coverage in a single regime dragging the pooled quantile
     (kelmarsh global coverage was 0.849 < 0.9);
  2. overlap/serial-correlation of multi-step targets degrading the finite-sample
     guarantee (addressed by sample-level block scores).

Prints, per regime AND globally, the realized coverage of:
  - pooled conformal (baseline)
  - regime-conditional + block conformal (fix)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES, fit_curve  # noqa: E402
from conformal_upper_bound import ConformalUpperBound, partial_id_metrics  # noqa: E402


def run(site, horizon, seed, alpha=0.1, epochs=15, seq_len=48):
    df = pd.read_csv(SITES[site])
    ws = df["wind_speed"].to_numpy(np.float64)
    p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    n = len(ws)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    ws_tr, p_tr = ws[:i1], p[:i1]
    ws_cal, p_cal = ws[i1:i2], p[i1:i2]
    ws_te, p_te = ws[i2:], p[i2:]

    p_rated, v_in, v_rated = fit_curve(ws_tr, p_tr)
    ws_mean, ws_std = float(ws_tr.mean()), float(ws_tr.std() + 1e-6)

    def pu(a):
        return np.clip(a / max(p_rated, 1e-6), 0.0, 1.0)

    def nrm(a):
        return (a - ws_mean) / ws_std

    def windows(ws_n, p_pu):
        nn = len(ws_n) - seq_len - horizon + 1
        X = np.zeros((nn, seq_len, 2), np.float32)
        Y = np.zeros((nn, horizon), np.float32)
        vr = np.zeros((nn, horizon), np.float32)
        for i in range(nn):
            X[i, :, 0] = ws_n[i:i + seq_len]
            X[i, :, 1] = ws_n[i:i + seq_len]
            Y[i] = p_pu[i + seq_len:i + seq_len + horizon]
            vr[i] = ws_n[i + seq_len:i + seq_len + horizon]
        return X, Y, vr

    Xtr, Ytr, _ = windows(nrm(ws_tr), pu(p_tr))
    Xcal, Ycal, vr_cal = windows(nrm(ws_cal), pu(p_cal))
    Xte, Yte, vr_te = windows(nrm(ws_te), pu(p_te))

    # regime definition on PHYSICAL wind: 0 = calm/other, 1 = high-ramp below-rated
    def regime_id(vr_phys):
        rmean = np.abs(np.diff(vr_phys, axis=1, prepend=vr_phys[:, :1])).mean(axis=1)
        thr = np.quantile(rmean, 0.75)
        is_ramp = (vr_phys.mean(axis=1) < v_rated) & (rmean > thr)
        return is_ramp.astype(np.int64)   # (N,)

    vr_cal_phys = vr_cal * ws_std + ws_mean
    vr_te_phys = vr_te * ws_std + ws_mean
    g_cal = regime_id(vr_cal_phys)
    g_te = regime_id(vr_te_phys)

    c = ConformalUpperBound(epochs=epochs).fit(Xtr, Ytr, seed=seed)

    # pooled baseline
    c.calibrate(Xcal, Ycal, [alpha])
    U_pooled = c.bound(Xte, alpha)
    m_pooled = partial_id_metrics(U_pooled, Yte, None)
    # per-regime coverage of pooled
    cov_pooled_r = {}
    for g in [0, 1]:
        sel = g_te == g
        cov_pooled_r[g] = float(np.clip(1.0 - (U_pooled < Yte).astype(float)[sel].mean(), 0, 1))

    # regime-conditional + block
    c.calibrate_regime(Xcal, Ycal, g_cal, [alpha])
    U_reg = c.bound_regime(Xte, alpha, g_te)
    m_reg = partial_id_metrics(U_reg, Yte, None)
    cov_reg_r = {}
    for g in [0, 1]:
        sel = g_te == g
        cov_reg_r[g] = float(np.clip(1.0 - (U_reg < Yte).astype(float)[sel].mean(), 0, 1))

    return dict(site=site, horizon=horizon, seed=seed, alpha=alpha,
                pooled_cov=m_pooled["coverage"],
                pooled_cov_r0=cov_pooled_r[0], pooled_cov_r1=cov_pooled_r[1],
                reg_cov=m_reg["coverage"],
                reg_cov_r0=cov_reg_r[0], reg_cov_r1=cov_reg_r[1],
                pooled_FER=m_pooled["FER_lb"], reg_FER=m_reg["FER_lb"])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="kelmarsh,wind_cn,smarteole")
    ap.add_argument("--horizons", default="6,12,24")
    ap.add_argument("--seeds", default="42,123")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="regime_conformal.csv")
    args = ap.parse_args()
    rows = []
    for site in args.sites.split(","):
        for h in [int(x) for x in args.horizons.split(",")]:
            for s in [int(x) for x in args.seeds.split(",")]:
                print(f"[{site}] h={h} s={s}", flush=True)
                rows.append(run(site, h, s, epochs=args.epochs))
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n=== saved {len(out)} rows -> {args.out} ===")
    print("=== 全局 coverage（目标 0.9） pooled vs regime ===\n")
    print(out.groupby("site")[["pooled_cov", "reg_cov"]].mean().round(4).to_string())
    print("\n=== 分 regime coverage（r0=calm, r1=high-ramp）===")
    for s in out.site.unique():
        sub = out[out.site == s]
        print(f"{s}: pooled r0={sub.pooled_cov_r0.mean():.3f} r1={sub.pooled_cov_r1.mean():.3f} "
              f"| reg r0={sub.reg_cov_r0.mean():.3f} r1={sub.reg_cov_r1.mean():.3f}")


if __name__ == "__main__":
    main()
