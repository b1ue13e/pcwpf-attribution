"""Faithful PCT reproduction v2 — global fixed bins (对齐官方 train_clean.py).

上一版 v1 用 batch-level min/max 动态 bins，偏离官方（官方 create_histograms()
一次性用训练集固定 bins_p/bins_w，target pc 与 predicted pc 共用）。本版对齐官方：

  - bins_p / bins_w 由训练集 power/windspeed min/max 固定构造一次（bin_num=50）
  - target pc = AveragePowerCurve.create_histograms() 一次性算好
  - predicted pc = 每 batch 用同一固定 bins 算
  - loss = mse + js（官方无权重，直接相加）

两个 arm 只改 js 的数学形式，其余全部一致：
  A official   : outer-product joint + F.kl_div(p.log(), q)   (参数反了)
  B corrected  : true 2D joint KDE + 标准 KL a*(log a - log b)

目标：回答"参数顺序错误"对最终 RMSE 的影响，同时保持官方全局固定 bins。
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES, fit_curve, power_curve_pu  # noqa: E402

torch.manual_seed(0); np.random.seed(0)

# ---------------- faithful PCT constraint ports ----------------
def marginal_pdf(values, bins, a, b=1e-10):
    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / a).pow(2))
    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + b
    pdf = pdf / normalization
    return pdf, kernel_values + torch.tensor(1e-10)

def joint_pdf_outer(k1, k2):
    j = torch.matmul(k1.transpose(1, 2), k2)
    return j / torch.sum(j, dim=(1, 2)).view(-1, 1, 1)

def true_joint_pdf(power, windspeed, bins_p, bins_w, h_p, h_w):
    rp = (power - bins_p.unsqueeze(0).unsqueeze(0))
    rw = (windspeed - bins_w.unsqueeze(0).unsqueeze(0))
    rp2 = (rp / h_p).pow(2).unsqueeze(-1)
    rw2 = (rw / h_w).pow(2).unsqueeze(-2)
    k = torch.exp(-0.5 * (rp2 + rw2)) + 1e-10
    pdf = torch.mean(k, dim=1)
    return pdf / (torch.sum(pdf, dim=(1, 2), keepdim=True) + 1e-10)

def js_official(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(p.log(), m, reduction="sum") + F.kl_div(q.log(), m, reduction="sum"))

def js_standard(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: torch.sum(a * (a.log() - b.log()))
    return 0.5 * (kl(p, m) + kl(q, m))

# ---------------- model ----------------
class NetLSTM(nn.Module):
    def __init__(self, dim_in=3, dim_hidden=20, dim_out=1, num_layer=2, dropout=0.0):
        # official NETLSTM uses dropout=0.3 but that's a training detail; keep simple
        super().__init__()
        self.lstm = nn.LSTM(dim_in, dim_hidden, num_layer, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(dim_hidden, dim_out)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1:, :])

# ---------------- data ----------------
SEQ, HORIZON = 96, 24
def build_windows(ws, pw):
    X, Y = [], []
    for i in range(0, len(ws) - SEQ - HORIZON, HORIZON):
        X.append(np.stack([ws[i:i + SEQ], pw[i:i + SEQ], ws[i:i + SEQ] ** 2], axis=1))
        Y.append(pw[i + SEQ:i + SEQ + HORIZON])
    return np.array(X, np.float32), np.array(Y, np.float32)

# ---------------- constraint (faithful global-bins) ----------------
class AveragePowerCurveTorch:
    """Mirror official AveragePowerCurve: fixed global bins + precomputed target pc."""
    def __init__(self, power_tr, wind_tr, bin_num=50):
        self.bin_num = bin_num
        # official: standardize the average tensor; we standardize per-channel like official
        self.p_mean, self.p_std = float(power_tr.mean()), float(power_tr.std() + 1e-8)
        self.w_mean, self.w_std = float(wind_tr.mean()), float(wind_tr.std() + 1e-8)
        pn = (power_tr - self.p_mean) / self.p_std
        wn = (wind_tr - self.w_mean) / self.w_std
        self.bins_p = torch.linspace(float(pn.min()), float(pn.max()), bin_num)
        self.bins_w = torch.linspace(float(wn.min()), float(wn.max()), bin_num)
        self.bw = torch.pow(torch.tensor(float(bin_num)), -1 / 5) * 0.1
        self.h_p = torch.std(pn) * self.bw + 1e-6
        self.h_w = torch.std(wn) * self.bw + 1e-6
        # precompute target pc (official & standard) using training data
        pnt = pn.reshape(1, -1, 1); wnt = wn.reshape(1, -1, 1)
        _, kp = marginal_pdf(pnt, self.bins_p, self.h_p)
        _, kw = marginal_pdf(wnt, self.bins_w, self.h_w)
        self.pc_target_outer = joint_pdf_outer(kp, kw)
        self.pc_target_true = true_joint_pdf(pnt, wnt, self.bins_p, self.bins_w, self.h_p, self.h_w)

    def js(self, pred_power, wind, mode):
        # pred_power: (B,1) normalized pu; wind: (B,1) normalized wind
        p = pred_power.reshape(1, -1, 1)
        w = wind.reshape(1, -1, 1)
        if mode == "official":
            _, kp = marginal_pdf(p, self.bins_p, self.h_p)
            _, kw = marginal_pdf(w, self.bins_w, self.h_w)
            pc_pred = joint_pdf_outer(kp, kw)
            return js_official(pc_pred, self.pc_target_outer)
        else:
            pc_pred = true_joint_pdf(p, w, self.bins_p, self.bins_w, self.h_p, self.h_w)
            return js_standard(pc_pred, self.pc_target_true)

def run_arm(mode, apc, X_tr, Y_tr, X_te, Y_te, EPOCHS=30):
    model = NetLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    B = X_tr.shape[0]
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(B)
        for i in range(0, B, 8):
            idx = perm[i:i + 8]
            x = torch.tensor(X_tr[idx]); y = torch.tensor(Y_tr[idx])
            opt.zero_grad()
            out = model(x).squeeze(-1)               # (b,1)
            mse = F.mse_loss(out, y[:, :1])
            wind_t = torch.tensor(X_tr[idx][:, -1, 0:1])
            js = apc.js(out, wind_t, mode)
            loss = mse + js
            if not torch.isfinite(loss):
                print(f"  [{mode}] ep{ep} NON-FINITE mse={mse.item():.4f} js={js.item():.4f}")
                return float("nan")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
    model.eval()
    with torch.no_grad():
        preds, targs = [], []
        for i in range(0, X_te.shape[0], 64):
            preds.append(model(torch.tensor(X_te[i:i + 64])).squeeze(-1).numpy())
            targs.append(Y_te[i:i + 64][:, :1])
        preds = np.concatenate(preds); targs = np.concatenate(targs)
    return float(np.sqrt(np.mean((preds - targs) ** 2)))


def main():
    import pandas as pd
    site = "kelmarsh"
    df = pd.read_csv(SITES[site])
    ws = df["wind_speed"].to_numpy(np.float64); p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    p_rated, v_in, v_rated = fit_curve(ws[:int(len(ws) * 0.8)], p[:int(len(p) * 0.8)])
    pw = p / max(p_rated, 1e-6)      # power pu
    wnd = ws / 15.0
    X, Y = build_windows(wnd, pw)
    n_tr = int(X.shape[0] * 0.8)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_te, Y_te = X[n_tr:], Y[n_tr:]

    # power/wind for the constraint in per-unit & normalized wind — mirror official standardization
    pw_tr = pw[:len(ws) * 8 // 10]  # align with fit_curve's 80% train slice (roughly)
    # simpler: use the same training slice as windows via original series indices is messy; use X_tr's lag-power channel as target power source
    # Build training power & wind arrays from the raw series matched to windows:
    # (wind channel 0 is wnd, power channel 1 is pw)
    wind_tr = X_tr[:, -1, 0]            # last-step windspeed per window (normalized)
    power_tr = Y_tr[:, 0]               # first horizon-step target power (normalized pu)

    apc = AveragePowerCurveTorch(torch.tensor(power_tr, dtype=torch.float32),
                                 torch.tensor(wind_tr, dtype=torch.float32))

    t0 = time.time(); rmse_a = run_arm("official", apc, X_tr, Y_tr, X_te, Y_te)
    t1 = time.time(); rmse_b = run_arm("corrected", apc, X_tr, Y_tr, X_te, Y_te)
    t2 = time.time()
    print(f"kelmarsh(faithful-v2): official_rmse(pu)={rmse_a:.5f} corrected_rmse(pu)={rmse_b:.5f} "
          f"delta={(rmse_a - rmse_b):+.5f} [{t1-t0:.0f}s/{t2-t1:.0f}s]")


if __name__ == "__main__":
    main()
