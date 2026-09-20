"""Architecture-consistency rerun: does the oracle-vs-unconstrained conclusion
depend on our NetLSTM variant? Compare THREE model variants, mirroring PCT repo:

  A. ours      : NetLSTM (2-layer LSTM, Linear(20,1))  <- used in prior experiments
  B. paper-lstm: NETLSTM as written in PCT repo (fc2 20->10 + ReLU + fc3 10->1,
                 returns out; 2 layers, dropout 0.3, dim_in=3, hidden=20)
  C. transformer: the model PCT repo ACTUALLY trains (Transformer d_model=12)

Each architecture x {clean, noisy(0.5)} x {oracle-constraint, unconstrained},
kelmarsh, aligned RNG, epochs=60. Question: is 'oracle constraint does not help'
robust across the three architecture variants?
"""
from __future__ import annotations
import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pct_fulltrain_v3_runner import (marginal_pdf, true_joint_pdf, js_standard,
                                      build_windows, SITES, fit_curve)  # noqa: E402


# --- model variants ---------------------------------------------------------
class NetLSTM_Ours(nn.Module):
    def __init__(self, dim_in=3, dim_hidden=20, dim_out=1):
        super().__init__()
        self.lstm = nn.LSTM(dim_in, dim_hidden, 2, batch_first=True)
        self.fc = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1:, :])


class NetLSTM_Paper(nn.Module):
    """PCT repo NETLSTM (commented-out in train_clean.py, but matches paper's LSTM claim)."""
    def __init__(self, dim_in=3, dim_hidden=20, dim_out=1, num_layer=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(dim_in, dim_hidden, num_layer, batch_first=True, dropout=dropout)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden // 2)
        self.fc3 = nn.Linear(dim_hidden // 2, dim_out)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1:, :].contiguous().view(-1, 20)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out.view(x.size(0), 1, -1)


class SimpleTransformer(nn.Module):
    """Minimal encoder-only transformer mirroring PCT repo dims (d_model=12)."""
    def __init__(self, d_input=3, d_model=12, d_out=1, nhead=2):
        super().__init__()
        self.proj = nn.Linear(d_input, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=48, dropout=0.2,
                                           batch_first=True, activation="relu")
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Linear(d_model, d_out)

    def forward(self, x):
        h = self.proj(x)
        h = self.enc(h)
        return self.fc(h[:, -1:, :])


MODELS = {"ours": NetLSTM_Ours, "paper-lstm": NetLSTM_Paper, "transformer": SimpleTransformer}


class APC:
    def __init__(self, power_tr, wind_tr, bin_num=50, device="cuda"):
        self.bin_num = bin_num; self.device = device
        self.pm, self.ps = float(power_tr.mean()), float(power_tr.std() + 1e-8)
        self.wm, self.ws = float(wind_tr.mean()), float(wind_tr.std() + 1e-8)
        pn = (power_tr - self.pm) / self.ps; wn = (wind_tr - self.wm) / self.ws
        self.bp = torch.linspace(float(pn.min()), float(pn.max()), bin_num, device=device)
        self.bw = torch.linspace(float(wn.min()), float(wn.max()), bin_num, device=device)
        self.h_p = torch.std(pn) * torch.pow(torch.tensor(float(bin_num), device=device), -1/5) * 0.1 + 1e-6
        self.h_w = torch.std(wn) * torch.pow(torch.tensor(float(bin_num), device=device), -1/5) * 0.1 + 1e-6
        pnt = pn.reshape(1, -1, 1); wnt = wn.reshape(1, -1, 1)
        self.target_true = true_joint_pdf(pnt, wnt, self.bp, self.bw, self.h_p, self.h_w)

    def js(self, pred, wind):
        p = pred.reshape(1, -1, 1); w = wind.reshape(1, -1, 1)
        return js_standard(true_joint_pdf(p, w, self.bp, self.bw, self.h_p, self.h_w), self.target_true)


def run_arch(arch, use_constraint, apc, X_tr, Y_tr, X_te, Y_te, epochs, device, rng_seed):
    torch.manual_seed(rng_seed); np.random.seed(rng_seed)
    model = MODELS[arch]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    B = X_tr.shape[0]
    for _ in range(epochs):
        model.train(); perm = np.random.permutation(B)
        for i in range(0, B, 8):
            idx = perm[i:i + 8]
            x = torch.tensor(X_tr[idx], device=device); y = torch.tensor(Y_tr[idx], device=device)
            opt.zero_grad(); out = model(x).squeeze(-1)
            mse = F.mse_loss(out, y[:, :1])
            loss = mse + (apc.js(out, torch.tensor(X_tr[idx][:, -1, 0:1], device=device)) if use_constraint else 0.0)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
    model.eval()
    with torch.no_grad():
        preds, targs = [], []
        for i in range(0, X_te.shape[0], 256):
            preds.append(model(torch.tensor(X_te[i:i + 256], device=device)).squeeze(-1).cpu().numpy())
            targs.append(Y_te[i:i + 256][:, :1])
        preds = np.concatenate(preds); targs = np.concatenate(targs)
    return float(np.sqrt(np.mean((preds - targs) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="kelmarsh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    import pandas as pd
    df = pd.read_csv(SITES[args.site])
    ws = df["wind_speed"].to_numpy(np.float64); p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    p_rated, v_in, v_rated = fit_curve(ws[:int(len(ws) * 0.8)], p[:int(len(p) * 0.8)])
    pw = p / max(p_rated, 1e-6); wnd = ws / 15.0
    X, Y = build_windows(wnd, pw)
    n_tr = int(X.shape[0] * 0.8)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]; X_te, Y_te = X[n_tr:], Y[n_tr:]
    # noisy variant (wind channel only)
    rng_n = np.random.default_rng(args.seed)
    X_tr_n = X_tr.copy(); X_tr_n[:, :, 0] += rng_n.normal(0, 0.5, size=X_tr[:, :, 0].shape)
    X_te_n = X_te.copy(); X_te_n[:, :, 0] += rng_n.normal(0, 0.5, size=X_te[:, :, 0].shape)

    apc = APC(torch.tensor(Y_tr[:, 0], dtype=torch.float32, device=device),
              torch.tensor(X_tr[:, -1, 0], dtype=torch.float32, device=device), device=device)

    res = []
    for arch in ["ours", "paper-lstm", "transformer"]:
        for regime, (a, b) in [("clean", (X_tr, X_te)), ("noisy", (X_tr_n, X_te_n))]:
            r_oracle = run_arch(arch, True, apc, a, Y_tr, b, Y_te, args.epochs, device, args.seed)
            r_un = run_arch(arch, False, apc, a, Y_tr, b, Y_te, args.epochs, device, args.seed)
            gain = (r_un - r_oracle) / r_un * 100
            res.append((arch, regime, r_oracle, r_un, gain))
    line = f"{args.site} seed={args.seed} epochs={args.epochs}\n"
    for arch, regime, ro, ru, g in res:
        line += f"  {arch:12s} {regime:5s} oracle={ro:.5f} uncon={ru:.5f} gain={g:+.2f}%\n"
    print(line.rstrip(), flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(line)


if __name__ == "__main__":
    main()
