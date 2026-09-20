"""P0-1 proxy-strength ablation: does the ramp-segment lag failure persist for
stronger proxies (linear-trend, EWMA), or is it a persistence-only artifact?

Verdict question: for EVERY proxy, is the below-rated high-ramp clip cut
(as fraction of true power) still >> the above-rated clip cut?

Proxies tested (all causal, one-step-ahead of true wind at time t):
  persistence : v_t = v_{t-1}
  linear      : v_t = 2*v_{t-1} - v_{t-2}   (trend extrapolation)
  ewma        : v_t = a*v_{t-1} + (1-a)*ewma_{t-1},  a tuned for min lag
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def power_curve(v, v_in, v_rated, v_out=25.0):
    out = np.zeros_like(v, dtype=np.float64)
    denom = max(v_rated - v_in, 1e-6)
    mid = (v >= v_in) & (v < v_rated)
    out = np.where(mid, np.clip((v - v_in) / denom, 0.0, 1.0) ** 3, out)
    out = np.where((v >= v_rated) & (v <= v_out), 1.0, out)
    return out


def fit_curve(ws, p):
    p_rated = float(np.percentile(p, 99.5))
    bins = np.arange(np.floor(ws.min()), np.ceil(ws.max()) + 0.5, 0.5)
    idx = np.digitize(ws, bins) - 1
    centers, med = [], []
    for i in range(len(bins) - 1):
        m = idx == i
        if m.sum() >= 50:
            centers.append((bins[i] + bins[i + 1]) / 2)
            med.append(np.median(p[m]))
    centers = np.array(centers); med = np.array(med)
    rm = med >= 0.95 * p_rated
    v_rated = float(centers[rm][0]) if rm.any() else 11.5
    v_in = float(np.quantile(ws[p >= 0.01 * p_rated], 0.02)) if (p >= 0.01 * p_rated).any() else 3.0
    return p_rated, v_in, v_rated


def ewma(ws, a):
    out = np.empty_like(ws)
    out[0] = ws[0]
    for t in range(1, len(ws)):
        out[t] = a * ws[t - 1] + (1 - a) * out[t - 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../04_数据集/prepared_priority_datasets/wind_cn_core_15min.csv")
    ap.add_argument("--ws-col", default="wind_speed")
    ap.add_argument("--power-col", default="power")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    ws = df[args.ws_col].to_numpy(np.float64)
    p = df[args.power_col].to_numpy(np.float64)
    valid = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[valid], p[valid]

    p_rated, v_in, v_rated = fit_curve(ws, p)
    print(f"[curve] p_rated={p_rated:.0f} v_in={v_in:.2f} v_rated={v_rated:.2f}")

    def env(v):
        return power_curve(v, v_in, v_rated) * p_rated

    v_true = ws[1:]
    pred = p[1:]                                    # stand-in prediction ~ truth
    r = np.abs(np.diff(ws))[: len(v_true)]
    hi_ramp = (v_true < v_rated) & (r > np.quantile(r, 0.9))
    calm = (v_true < v_rated) & (r <= np.quantile(r, 0.5))
    above = v_true >= v_rated

    proxies = {
        "persistence": ws[:-1],
        "linear": 2 * ws[1:-1] - ws[:-2],
        "ewma_a0.6": ewma(ws, 0.6)[1:],
        "ewma_a0.3": ewma(ws, 0.3)[1:],
    }
    n_true = len(v_true)
    for name, vp in proxies.items():
        vp = np.asarray(vp, dtype=np.float64)
        if len(vp) != n_true:
            vp = vp[:n_true]
        e_true = env(v_true[:len(vp)])
        e_proxy = env(vp)
        eps = np.abs(e_true - e_proxy)
        r_aligned = np.abs(np.diff(ws))[: len(vp)]
        corr = np.corrcoef(eps[np.isfinite(eps)], (r_aligned[np.isfinite(eps)] + 1e-9))[0, 1] if len(eps) > 10 else float("nan")
        clipped = np.minimum(pred[:len(vp)], e_proxy)
        cut = pred[:len(vp)] - clipped
        hi = hi_ramp[:len(vp)]; cl = calm[:len(vp)]; ab = above[:len(vp)]
        cut_ramp = cut[hi].mean() / max(pred[:len(vp)][hi].mean(), 1e-6) * 100
        cut_calm = cut[cl].mean() / max(pred[:len(vp)][cl].mean(), 1e-6) * 100
        cut_above = cut[ab].mean() / max(pred[:len(vp)][ab].mean(), 1e-6) * 100
        asym = cut_ramp - cut_above
        print(f"[{name:12s}] eps-corr={corr:.3f} | cut%: ramp={cut_ramp:.1f} calm={cut_calm:.1f} "
              f"above={cut_above:.1f} | ramp-above asymmetry={asym:+.1f}pt")


if __name__ == "__main__":
    main()
