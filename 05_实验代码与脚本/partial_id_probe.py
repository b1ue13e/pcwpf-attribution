"""Minimal go/no-go for the partial-identification mainline.

Answers two questions with data (no new method claim):

  Q1 (Theorem 1 — robust safety): among deterministic ceilings c, which one is
     safe under ALL U* compatible with Y <= U* <= P_rated?  Answer by theorem:
     only c == P_rated.  We VERIFY the empirical counterpart: how much FE does a
     "tighter-than-rated" deterministic curve incur on the realized-power witness?

  Q2 (Theorem 2 — risk-controlled tightening): does the conformal upper bound
     for realized power achieve FER_lb ~ alpha on the observed (witness) side?

Honest scope: no U* identification; only the observed witness side is certified.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES, fit_curve, power_curve_pu  # noqa: E402
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
        vl = np.zeros(nn, np.float32)
        vr = np.zeros((nn, horizon), np.float32)
        for i in range(nn):
            X[i, :, 0] = ws_n[i:i + seq_len]
            X[i, :, 1] = ws_n[i:i + seq_len]
            Y[i] = p_pu[i + seq_len:i + seq_len + horizon]
            vl[i] = ws_n[i + seq_len - 1]
            vr[i] = ws_n[i + seq_len:i + seq_len + horizon]
        return X, Y, vl, vr

    Xtr, Ytr, _, _ = windows(nrm(ws_tr), pu(p_tr))
    Xcal, Ycal, vl_cal, _ = windows(nrm(ws_cal), pu(p_cal))
    Xte, Yte, vl_te, vr_te = windows(nrm(ws_te), pu(p_te))
    vl_phys = vl_te * ws_std + ws_mean
    vr_phys = vr_te * ws_std + ws_mean
    r = np.abs(np.diff(vr_phys, axis=1, prepend=vr_phys[:, :1])).mean(axis=1)
    ramp = ((vr_phys.mean(axis=1) < v_rated) & (r > np.quantile(r, 0.75)))[:, None]
    ramp_mask = np.repeat(ramp, horizon, axis=1)

    # Q1: deterministic ceilings and their observed FE lower bound
    # intrinsic box (robust-safe by Theorem 1)
    box = np.ones_like(Yte)
    m_box = partial_id_metrics(box, Yte, ramp_mask)
    # deterministic "tighter-than-rated" curve (persistence proxy) — NOT robust-safe
    pers = np.repeat(power_curve_pu(vl_phys, v_in, v_rated)[:, None], horizon, axis=1)
    m_pers = partial_id_metrics(pers, Yte, ramp_mask)

    # Q2: conformal upper bound for realized power
    c = ConformalUpperBound(epochs=epochs).fit(Xtr, Ytr, seed=seed)
    c.calibrate(Xcal, Ycal, [alpha])
    Uhat = c.bound(Xte, alpha)
    m_cf = partial_id_metrics(Uhat, Yte, ramp_mask)

    return dict(site=site, horizon=horizon, seed=seed, alpha=alpha,
                box_FER_lb=m_box["FER_lb"], box_LEC_lb=m_box["LEC_lb"],
                persist_FER_lb=m_pers["FER_lb"], persist_LEC_lb=m_pers["LEC_lb"],
                conformal_FER_lb=m_cf["FER_lb"], conformal_LEC_lb=m_cf["LEC_lb"],
                conformal_coverage=m_cf["coverage"])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="wind_cn,kelmarsh,smarteole,penmanshiel,wind_us")
    ap.add_argument("--horizons", default="6,12,24")
    ap.add_argument("--seeds", default="42,123")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="partial_id_probe.csv")
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
    print("mean over sites/horizons/seeds:")
    print(out[["box_FER_lb", "box_LEC_lb", "persist_FER_lb", "persist_LEC_lb",
               "conformal_FER_lb", "conformal_LEC_lb", "conformal_coverage"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
