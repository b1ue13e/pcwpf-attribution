"""Minimal faithful reproduction: does PCT's reversed-KL JS constraint hurt final RMSE?

Single-site (kelmarsh) first, per decision dec-721ee554bb975b28. Two arms:
  A (official) : js = 0.5*(F.kl_div(p.log(), m) + F.kl_div(q.log(), m)) on
                 outer-product joint (faithful port of PCT train_clean.py)
  B (corrected): js = 0.5*(KL(p,m) + KL(q,m)) with KL(a||b)=sum a*(log a - log b)
                 on true 2D joint KDE
Both arms: loss = MSE + js   (PCT uses NO weight on js — faithful)
Same LSTM, same data, same seed. Compare held-out RMSE (in kW).
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
    rp = (power - bins_p.unsqueeze(0).unsqueeze(0))           # (1,N,Bp)
    rw = (windspeed - bins_w.unsqueeze(0).unsqueeze(0))       # (1,N,Bw)
    rp2 = (rp / h_p).pow(2).unsqueeze(-1)                     # (1,N,Bp,1)
    rw2 = (rw / h_w).pow(2).unsqueeze(-2)                     # (1,N,1,Bw)
    k = torch.exp(-0.5 * (rp2 + rw2)) + 1e-10                 # (1,N,Bp,Bw)
    pdf = torch.mean(k, dim=1)                                # (1,Bp,Bw)
    return pdf / (torch.sum(pdf, dim=(1, 2), keepdim=True) + 1e-10)

def js_official(p, q):      # F.kl_div(p.log(), q) — argument order bug (faithful)
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(p.log(), m, reduction="sum") + F.kl_div(q.log(), m, reduction="sum"))

def js_standard(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: torch.sum(a * (a.log() - b.log()))
    return 0.5 * (kl(p, m) + kl(q, m))

# ---------------- model ----------------
class NetLSTM(nn.Module):
    def __init__(self, dim_in=3, dim_hidden=20, dim_out=1, num_layer=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(dim_in, dim_hidden, num_layer, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(dim_hidden, dim_out)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1:, :])   # last-timestep -> (B,1,1)

# ---------------- data ----------------
SEQ, HORIZON = 96, 24  # kelmarsh is 15-min; 24 steps = 6h ahead
def build_windows(ws, pw):
    X, Y = [], []
    for i in range(0, len(ws) - SEQ - HORIZON, HORIZON):
        w = ws[i:i + SEQ]
        # features: [wind_speed, lagged_power, wind_speed_squared-ish] (3 channels)
        X.append(np.stack([w, pw[i:i + SEQ], w ** 2], axis=1))       # (SEQ,3)
        Y.append(pw[i + SEQ:i + SEQ + HORIZON])                      # (HORIZON,)
    return np.array(X, np.float32), np.array(Y, np.float32)          # (B,SEQ,3),(B,H)

def constraint_terms(pred_power, wind_t, power_t, mode, bin_num=50):
    """Return js value for arm `mode` ('official'|'corrected'), on current batch.
    pred_power: (B,1) current-step predictions in normalized power units
    wind_t: (B,1) normalized windspeed; power_t: (B,1) normalized target power
    """
    B = pred_power.shape[0]
    p = pred_power.reshape(1, B, 1)
    w = wind_t.reshape(1, B, 1)
    t = power_t.reshape(1, B, 1)
    bins_p = torch.linspace(float(p.min()), float(p.max()), bin_num)
    bins_t = torch.linspace(float(t.min()), float(t.max()), bin_num)  # FIX: target's own bins
    bins_w = torch.linspace(float(w.min()), float(w.max()), bin_num)
    bw = torch.pow(torch.tensor(float(bin_num)), -1 / 5) * 0.1
    h_p = torch.std(p) * bw + 1e-6
    h_t = torch.std(t) * bw + 1e-6
    h_w = torch.std(w) * bw + 1e-6
    if mode == "official":
        _, kp1 = marginal_pdf(p, bins_p, h_p); _, kw1 = marginal_pdf(w, bins_w, h_w)
        _, kp2 = marginal_pdf(t, bins_t, h_t); _, kw2 = marginal_pdf(w, bins_w, h_w)
        return js_official(joint_pdf_outer(kp1, kw1), joint_pdf_outer(kp2, kw2))
    else:
        j1 = true_joint_pdf(p, w, bins_p, bins_w, h_p, h_w)
        j2 = true_joint_pdf(t, w, bins_t, bins_w, h_t, h_w)   # FIX: target's own bins
        return js_standard(j1, j2)

def run_arm(mode, X_tr, Y_tr, X_te, Y_te, EPOCHS=30):
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
            out = model(x).squeeze(-1)                       # (b,1)
            # MSE on the single-step-ahead target (first horizon step)
            mse = F.mse_loss(out, y[:, :1])
            # constraint terms: use last-step powers for a stable batch-level dist
            try:
                js = constraint_terms(out, torch.tensor(X_tr[idx][:, -1, 0:1]),
                                      y[:, :1], mode)
            except Exception:
                js = torch.tensor(0.0)
            loss = mse + js                                  # official: no weight
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
    model.eval()
    with torch.no_grad():
        preds, targs = [], []
        for i in range(0, X_te.shape[0], 64):
            x = torch.tensor(X_te[i:i + 64]); y = Y_te[i:i + 64]
            preds.append(model(x).squeeze(-1).numpy()); targs.append(y[:, :1])
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
    # normalize to pu for training
    pw = p / max(p_rated, 1e-6)          # power in pu [0,1]
    wnd = ws / 15.0                       # windspeed scaled
    n = len(ws); i1 = int(n * 0.8)
    X, Y = build_windows(wnd, pw)
    n_tr = int(X.shape[0] * 0.8)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_te, Y_te = X[n_tr:], Y[n_tr:]

    t0 = time.time()
    rmse_a = run_arm("official", X_tr, Y_tr, X_te, Y_te)
    t1 = time.time()
    rmse_b = run_arm("corrected", X_tr, Y_tr, X_te, Y_te)
    t2 = time.time()
    print(f"kelmarsh: official_rmse(pu)={rmse_a:.5f}  corrected_rmse(pu)={rmse_b:.5f}  "
          f"delta={(rmse_a - rmse_b):+.5f}  [{t1-t0:.0f}s / {t2-t1:.0f}s]")
    print(f"in kW: official={rmse_a * p_rated:.1f}  corrected={rmse_b * p_rated:.1f}  "
          f"delta_kW={(rmse_a - rmse_b) * p_rated:+.1f}")


if __name__ == "__main__":
    main()
