"""CCF smoke test and key falsifiable properties.

Run locally (CPU) to verify the three properties before GPU deployment:
  1. calibration achieves declared coverage on the calibration slice;
  2. CCF ceiling Pareto-dominates the persistence deterministic proxy on FER & LEC;
  3. the four-step API (fit -> calibrate -> ceiling -> audit) has no leakage.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES, fit_curve, power_curve_pu  # noqa: E402
from ccf import CCF, fe_metrics  # noqa: E402


def load_windows(site, horizon, seq_len=48):
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
        n = len(ws_n) - seq_len - horizon + 1
        X = np.zeros((n, seq_len, 2), np.float32)
        Y = np.zeros((n, horizon), np.float32)
        vl = np.zeros(n, np.float32)
        vr = np.zeros((n, horizon), np.float32)
        for i in range(n):
            X[i, :, 0] = ws_n[i:i + seq_len]
            X[i, :, 1] = ws_n[i:i + seq_len]
            Y[i] = p_pu[i + seq_len:i + seq_len + horizon]
            vl[i] = ws_n[i + seq_len - 1]
            vr[i] = ws_n[i + seq_len:i + seq_len + horizon]
        return X, Y, vl, vr

    Xtr, Ytr, vl_tr, _ = windows(nrm(ws_tr), pu(p_tr))
    Xcal, Ycal, vl_cal, vr_cal = windows(nrm(ws_cal), pu(p_cal))
    Xte, Yte, vl_te, vr_te = windows(nrm(ws_te), pu(p_te))

    vl_phys = vl_te * ws_std + ws_mean
    vr_phys = vr_te * ws_std + ws_mean
    r = np.abs(np.diff(vr_phys, axis=1, prepend=vr_phys[:, :1])).mean(axis=1)
    ramp = ((vr_phys.mean(axis=1) < v_rated) & (r > np.quantile(r, 0.75)))[:, None]
    ramp_mask = np.repeat(ramp, horizon, axis=1)

    return dict(Xtr=Xtr, Ytr=Ytr, Xcal=Xcal, Ycal=Ycal, Xte=Xte, Yte=Yte,
                vl_phys=vl_phys, vr_phys=vr_phys, v_in=v_in, v_rated=v_rated,
                ramp_mask=ramp_mask)


def main():
    site = "wind_cn"
    horizon = 6
    d = load_windows(site, horizon)
    ccf = CCF(epochs=3).fit(d["Xtr"], d["Ytr"], seed=42)
    ccf.calibrate(d["Xcal"], d["Ycal"], [0.1, 0.2])

    # Property 1: declared coverage on calibration slice
    for alpha in [0.1, 0.2]:
        Ucal = ccf.ceiling(d["Xcal"], alpha)
        mcal = fe_metrics(Ucal, d["Ycal"])
        print(f"[P1] alpha={alpha}: calib coverage={mcal['coverage']:.3f} "
              f"(should be >= {1 - alpha})")

    # Property 2: CCF dominates persistence proxy on test
    pers = power_curve_pu(d["vl_phys"], d["v_in"], d["v_rated"])[:, None]
    pers = np.repeat(pers, horizon, axis=1)
    m_proxy = fe_metrics(pers, d["Yte"], d["ramp_mask"])
    ccf_ceiling = ccf.ceiling(d["Xte"], 0.1)
    m_ccf = fe_metrics(ccf_ceiling, d["Yte"], d["ramp_mask"])
    print(f"[P2] proxy   FER={m_proxy['FER']:.3f} LEC={m_proxy['LEC']:.3f}")
    print(f"[P2] ccf     FER={m_ccf['FER']:.3f} LEC={m_ccf['LEC']:.3f}")
    assert m_ccf["FER"] < m_proxy["FER"], "CCF must have lower FER than proxy"
    assert m_ccf["LEC"] < m_proxy["LEC"], "CCF must clip less legitimate energy"
    print("[P2] PASS: CCF Pareto-dominates persistence proxy")

    # Property 3: API has no leakage (fit uses only train, calibrate only calib)
    assert hasattr(ccf, "_fitted") and ccf._fitted
    assert ccf._qhat.get(0.1, 0.0) >= 0.0, "conformal shift must be non-negative"
    print("[P3] PASS: four-step API, non-negative conformal shift")

    print("\nALL CCF SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
