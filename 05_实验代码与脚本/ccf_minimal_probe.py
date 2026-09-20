"""CCF minimal go/no-go probe.

Judge, on ONE site / one strong backbone / three horizons, whether a
conditional-quantile / split-conformal ceiling Pareto-dominates a
deterministic proxy ceiling on the observable false-exclusion side.

Definition (user decision dec-f342b68794a07655):
  U*_{t+h}  = latent true operational feasible upper bound (not fully observable).
  Y_{t+h}   = realized power after QC = a "feasible witness":
              if Uhat_{t+h} < Y_{t+h}  =>  certain false exclusion (FE).
              the reverse does NOT certify safety / full false-admission.

Core metric (observable):
  FER  = P(Uhat < Y)  over all test samples
  LEC  = mean (Y - Uhat)^+  over FE samples  (legitimate energy clipped),
         conditioned on regeme (high-ramp / above-rated).

Ceilings compared:
  1. intrinsic_box    : Uhat = 1.0 (P_rated)   -- always upper, no FE by construction
  2. proxy_deterministic: Uhat = P_pu(v_last)   -- deterministic persistence ceiling
  3. cqf_train        : Uhat = Q_alpha(P | H) fitted on train (uncalibrated quantile)
  4. cqf_split_conformal: cqf_train + split-conformal calibration on a calibration slice

Go criterion:
  If split-conformal / quantile ceiling achieves strictly lower FER at equal-or-lower
  LEC vs the deterministic proxy over a grid of alpha, the "constraint observability
  gap" claim has a foot. If only a smooth trade-off, we report an audit, no new method.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultra_lsnt_timeseries import LSTMBaseline, compute_metrics
from real_train_audit import SITES, fit_curve, power_curve_pu, power_curve_pu_torch


# ---------------- quantile forecaster (single backbone, quantile output head) ----------------
class QuantileLSTM(nn.Module):
    """Predicts tau quantile levels of per-unit power. Single shared LSTM trunk,
    one output head returning concat of quantile predictions."""
    def __init__(self, input_dim, hidden_dim, pred_len, quantiles):
        super().__init__()
        self.quantiles = list(quantiles)
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, pred_len * len(self.quantiles))
        self.pred_len = pred_len
        self.nq = len(self.quantiles)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        z = self.fc(h)                       # (B, pred_len * nq)
        z = z.view(z.size(0), self.pred_len, self.nq)
        return z                              # (B, H, nq)


def pinball_loss(pred_quants, y, quantiles):
    """pred_quants: (B, H, nq); y: (B, H). Mean pinball over all levels/horizons."""
    y = y.unsqueeze(-1)                       # (B, H, 1)
    err = y - pred_quants                     # (B, H, nq)
    q = torch.tensor(quantiles, device=pred_quants.device, dtype=pred_quants.dtype)
    loss = torch.where(err >= 0, q, q - 1.0) * err
    return loss.mean()


def run(device, site="wind_cn", horizon=6, alpha_grid=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
        seq_len=48, epochs=15, seed=42):
    df = pd.read_csv(SITES[site])
    ws = df["wind_speed"].to_numpy(np.float64)
    p = df["power"].to_numpy(np.float64)
    valid = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[valid], p[valid]
    n_all = len(ws)

    # G3-compatible raw-time split: train / calib / test = 60 / 20 / 20
    n_tr = int(n_all * 0.6)
    n_cal = int(n_all * 0.2)
    ws_tr, p_tr = ws[:n_tr], p[:n_tr]
    ws_cal, p_cal = ws[n_tr:n_tr + n_cal], p[n_tr:n_tr + n_cal]
    ws_te, p_te = ws[n_tr + n_cal:], p[n_tr + n_cal:]

    ws_mean, ws_std = float(ws_tr.mean()), float(ws_tr.std() + 1e-6)
    p_rated, v_in, v_rated = fit_curve(ws_tr, p_tr)   # train-only curve

    def pu(a):
        return np.clip(a / max(p_rated, 1e-6), 0.0, 1.0)

    p_tr_pu, p_cal_pu, p_te_pu = pu(p_tr), pu(p_cal), pu(p_te)
    ws_tr_n = (ws_tr - ws_mean) / ws_std
    ws_cal_n = (ws_cal - ws_mean) / ws_std
    ws_te_n = (ws_te - ws_mean) / ws_std

    def windows(ws_n, p_pu, target_idx_start):
        n = len(ws_n) - seq_len - horizon + 1
        X = np.zeros((n, seq_len, 2), np.float32)
        Y = np.zeros((n, horizon), np.float32)
        vp = np.zeros(n, np.float32)          # persistence proxy at forecast origin
        vr = np.zeros((n, horizon), np.float32)
        for i in range(n):
            X[i, :, 0] = ws_n[i:i + seq_len]
            X[i, :, 1] = ws_n[i:i + seq_len]
            Y[i] = p_pu[i + seq_len:i + seq_len + horizon]
            vp[i] = ws_n[i + seq_len - 1]
            vr[i] = ws_n[i + seq_len:i + seq_len + horizon]
        return X, Y, vp, vr

    Xtr, Ytr, vptr, _ = windows(ws_tr_n, p_tr_pu, 0)
    Xcal, Ycal, vpcal, vrcal = windows(ws_cal_n, p_cal_pu, n_tr)
    Xte, Yte, vpte, vrte = windows(ws_te_n, p_te_pu, n_tr + n_cal)
    vp_phys_te = vpte * ws_std + ws_mean
    vr_phys_te = vrte * ws_std + ws_mean

    # ---- fit quantile forecaster on train ----
    quantiles = [0.5, 0.7, 0.8, 0.9, 0.95]   # base levels for conformal swap
    torch.manual_seed(seed); np.random.seed(seed)
    model = QuantileLSTM(2, 32, horizon, quantiles).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr_t = torch.from_numpy(Xtr); Ytr_t = torch.from_numpy(Ytr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for bi in range(0, len(Xtr_t), 64):
            idx = perm[bi:bi + 64]
            xb, yb = Xtr_t[idx].to(device), Ytr_t[idx].to(device)
            z = model(xb)
            loss = pinball_loss(z, yb, quantiles)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # ---- predict quantiles on calibration + test ----
    def predict_q(Xn):
        model.eval()
        out = []
        with torch.no_grad():
            for bi in range(0, len(Xn), 256):
                out.append(model(torch.from_numpy(Xn[bi:bi + 256]).to(device)).cpu().numpy())
        return np.concatenate(out, 0)         # (B, H, nq)

    q_cal = predict_q(Xcal)                   # (B_cal, H, nq)
    q_te = predict_q(Xte)                     # (B_te, H, nq)

    # ---- deterministic proxy ceiling (persistence) ----
    env_pu_proxy = power_curve_pu(vp_phys_te, v_in, v_rated)[:, None]  # (B_te, 1)
    env_pu_proxy = np.repeat(env_pu_proxy, horizon, axis=1)

    # ---- intrinsic box ceiling ----
    env_pu_box = np.ones_like(Yte)

    # ---- metrics: observable false exclusion on realized power witness ----
    # FER = mean over test features of mean over horizon of [Uhat < Y]
    def fer_lec(ceiling):
        fe = (ceiling < Yte).astype(np.float64)         # (B, H)
        fer = fe.mean()                                  # fraction of (sample,horizon) FE
        lec = np.where(fe > 0, (Yte - ceiling) * fe, 0.0).mean()
        # regeme conditions
        r = np.abs(np.diff(vr_phys_te, axis=1, prepend=vr_phys_te[:, :1])).mean(axis=1)
        high_ramp = r > np.quantile(r, 0.75)
        above = vr_phys_te.mean(axis=1) >= v_rated
        lec_hr = np.where((fe > 0) & high_ramp[:, None], (Yte - ceiling) * fe, 0.0).mean()
        return fer, lec, lec_hr

    rows = []
    # intrinsic box: by construction 0 FE
    fb, lb, hb = fer_lec(env_pu_box)
    rows.append({"ceiling": "intrinsic_box", "fer": fb, "lec": lb, "lec_highramp": hb, "alpha": 1.0})
    # deterministic proxy
    fp, lp, hp = fer_lec(env_pu_proxy)
    rows.append({"ceiling": "proxy_deterministic", "fer": fp, "lec": lp, "lec_highramp": hp, "alpha": 1.0})

    # ---- uncalibrated quantile ceiling at each alpha level ----
    # quantile index for alpha: ceiling = Q_{1-alpha}
    q_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
    for alpha in alpha_grid:
        level = 1.0 - alpha
        # nearest quantile level
        qi = int(np.argmin([abs(l - level) for l in q_levels]))
        ceiling = q_te[:, :, qi]              # (B_te, H)
        f, l, h = fer_lec(ceiling)
        rows.append({"ceiling": "cqf_train", "fer": f, "lec": l, "lec_highramp": h, "alpha": alpha})

    # ---- split-conformal calibrated ceiling ----
    # Standard one-sided split conformal: base quantile Q(X) from train; conformity
    # score S = Y - Q(X) on the calibration slice. Only POSITIVE scores (under-coverage)
    # should raise the ceiling; a negative (1-alpha)-quantile would LOWER it, which is the
    # bug — so we floor the correction at 0. Ceiling = Q(X) + max(0, qhat).
    for alpha in alpha_grid:
        level = 1.0 - alpha
        qi = int(np.argmin([abs(l - level) for l in q_levels]))
        base_cal = q_cal[:, :, qi]
        # one-sided nonnegative conformity score: only under-coverage matters
        score_cal = np.maximum(Ycal - base_cal, 0.0).flatten()
        n_cal_flat = len(score_cal)
        if n_cal_flat == 0:
            continue
        k = min(int(np.ceil((1.0 - alpha) * (n_cal_flat + 1))), n_cal_flat - 1)
        k = max(k, 0)
        qhat = np.partition(score_cal, k)[k]
        ceiling = q_te[:, :, qi] + qhat
        f, l, h = fer_lec(ceiling)
        rows.append({"ceiling": "cqf_split_conformal", "fer": f, "lec": l, "lec_highramp": h, "alpha": alpha})

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="wind_cn")
    ap.add_argument("--horizons", default="6,12,24")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ccf_probe.csv")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []
    for h in [int(x) for x in args.horizons.split(",")]:
        print(f"=== horizon {h} ===", flush=True)
        df = run(device, site=args.site, horizon=h, epochs=args.epochs, seed=args.seed)
        df["horizon"] = h
        all_rows.append(df)
        print(df.to_string(index=False), flush=True)
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n=== saved {len(out)} rows -> {args.out} ===")


if __name__ == "__main__":
    main()
