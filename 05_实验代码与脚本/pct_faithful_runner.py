"""Faithful PCT runner — structure/protocol-aligned with official train_clean.py.

Aligned dimensions (from official code, verified 2026-09-04):
  TIME_AHEAD = 144 (6*24), WINDOW_SIZE = 720 (5*6*24), STEP = 144
  batch_size = 500, epochs = 100, lr = 1e-2, Adam
  input features = 3 (index [0,2,4] of the 5-col table), target = power (index 3)
  dynamic js_ratio switch (implemented in AveragePowerCurveTorch)
  architecture: BOTH official Transformer (d_model=12) and paper-claimed LSTM

NOT aligned (declared honestly):
  - DATA: our open 5-site 15-min data replaces the private 江苏 10-min data
  - the exact 5-column semantics (we map wind_speed, wind^2/power, power, wp-like
    forecast onto 3 input channels)
This is a STRUCTURE-faithful transfer, not an exact reproduction.
"""
from __future__ import annotations
import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pct_fulltrain_v3_runner import (marginal_pdf, joint_pdf_outer, true_joint_pdf,
                                      js_official, js_standard, SITES, fit_curve)  # noqa: E402

TIME_AHEAD = 6 * 24          # 144
WINDOW_SIZE = 5 * 6 * 24     # 720
STEP = 6 * 24                # 144
BATCH_SIZE = 500
EPOCHS = 100
LR = 1e-2


def build_windows_faithful(ws_pu, pw, forecast_pu):
    """Mirror data_sampler + time_ahead: window 720, step 144.
    FIX (round-2 review): target must be the NEXT point OUTSIDE the window
    (pw[i+WINDOW_SIZE]), NOT pw[i+WINDOW_SIZE-1] (which is already an input
    channel -> leakage). True 1-step forecast, honest horizon=1."""
    X, Y = [], []
    i = 0
    n = len(ws_pu)
    while i + WINDOW_SIZE < n:      # need one point beyond for the target
        seg_p = pw[i:i + WINDOW_SIZE]
        seg_w = ws_pu[i:i + WINDOW_SIZE]
        seg_f = forecast_pu[i:i + WINDOW_SIZE]
        X.append(np.stack([seg_w, seg_p, seg_f], axis=1))     # (720, 3)
        Y.append(pw[i + WINDOW_SIZE])                          # NEXT point (1-step)
        i += STEP
    return np.array(X, np.float32), np.array(Y, np.float32)


class PCTTransformerFaithful(nn.Module):
    """Official transformer dims (d_input=3, d_model=12, q=8,v=8,h=2,N=1)."""
    def __init__(self, d_input=3, d_model=12, d_out=1):
        super().__init__()
        self.proj = nn.Linear(d_input, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2,
                                           dim_feedforward=48, dropout=0.2,
                                           batch_first=True, activation="relu")
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Linear(d_model, d_out)

    def forward(self, x):
        h = self.proj(x)
        h = self.enc(h)
        return self.fc(h[:, -1, :]).unsqueeze(1)


class PCTLSTMFaithful(nn.Module):
    """Paper-claimed LSTM (NETLSTM in repo comments: dim_in=3,hidden=20,2 layers)."""
    def __init__(self, dim_in=3, dim_hidden=20, dim_out=1):
        super().__init__()
        self.lstm = nn.LSTM(dim_in, dim_hidden, 2, batch_first=True, dropout=0.3)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden // 2)
        self.fc3 = nn.Linear(dim_hidden // 2, dim_out)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = F.relu(self.fc2(h))
        return self.fc3(h).unsqueeze(1)


class APC_Faithful:
    """Official AveragePowerCurve with dynamic js_ratio (already in runner module)."""
    def __init__(self, power_tr, wind_tr, bin_num=50, device="cpu"):
        from pct_fulltrain_v3_runner import AveragePowerCurveTorch
        self.inner = AveragePowerCurveTorch(power_tr, wind_tr, bin_num, device)

    def js(self, pred, wind, mode="official"):
        return self.inner.js(pred, wind, mode)


def run(arch, mode, apc, X, Y, Xte, Yte, device, seed, epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    model = (PCTTransformerFaithful() if arch == "transformer" else PCTLSTMFaithful()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(len(X))
        for i in range(0, len(X), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            x = torch.tensor(X[idx], device=device); y = torch.tensor(Y[idx], device=device)
            opt.zero_grad()
            out = model(x).squeeze(-1)
            mse = F.mse_loss(out, y)
            if mode == "unconstrained":
                loss = mse
            else:
                js = apc.js(out.unsqueeze(-1), x[:, -1, 0:1], mode)
                loss = mse + js
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
    model.eval()
    with torch.no_grad():
        preds, targs = [], []
        for i in range(0, len(Xte), 256):
            preds.append(model(torch.tensor(Xte[i:i + 256], device=device)).squeeze(-1).cpu().numpy())
            targs.append(Yte[i:i + 256])
        preds = np.concatenate(preds); targs = np.concatenate(targs)
    return float(np.sqrt(np.mean((preds - targs) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="kelmarsh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--arch", default="transformer", choices=["transformer", "lstm"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    import pandas as pd
    df = pd.read_csv(SITES[args.site])
    ws = df["wind_speed"].to_numpy(np.float64); p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    p_rated, v_in, v_rated = fit_curve(ws[:int(len(ws) * 0.8)], p[:int(len(p) * 0.8)])
    pw = p / max(p_rated, 1e-6); ws_pu = ws / 15.0
    forecast_pu = (ws / 15.0) ** 2  # wp-like proxy channel
    X, Y = build_windows_faithful(ws_pu, pw, forecast_pu)
    n_tr = int(X.shape[0] * 0.8)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]; X_te, Y_te = X[n_tr:], Y[n_tr:]
    apc = APC_Faithful(torch.tensor(Y_tr, dtype=torch.float32, device=device),
                       torch.tensor(X_tr[:, -1, 0], dtype=torch.float32, device=device),
                       device=device)
    # dynamic switch check
    print(f'js_ratio={apc.inner.js_ratio}', flush=True)
    t0 = time.time()
    r_const = run(args.arch, "official", apc, X_tr, Y_tr, X_te, Y_te, device, args.seed, args.epochs)
    r_un = run(args.arch, "unconstrained", apc, X_tr, Y_tr, X_te, Y_te, device, args.seed, args.epochs)
    line = (f"{args.site} {args.arch} seed={args.seed} epochs={args.epochs} "
            f"window={WINDOW_SIZE} horizon=1 | "
            f"constrained={r_const:.5f} unconstrained={r_un:.5f} "
            f"rel={100*(r_const-r_un)/r_un:+.1f}%")
    print(line, flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
