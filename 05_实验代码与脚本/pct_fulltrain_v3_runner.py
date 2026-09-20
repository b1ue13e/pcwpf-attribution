"""PCT faithful reproduction — parameterized multi-site/multi-seed runner (GPU).

Usage: python pct_fulltrain_v3_runner.py --site kelmarsh --seed 0 --gpu 0 [--epochs 30]
Faithful aligned-to-official global fixed bins (see v2), GPU-accelerated.
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES, fit_curve, power_curve_pu  # noqa: E402


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


class NetLSTM(nn.Module):
    def __init__(self, dim_in=3, dim_hidden=20, dim_out=1, num_layer=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(dim_in, dim_hidden, num_layer, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1:, :])


SEQ, HORIZON = 96, 24


def build_windows(ws, pw):
    X, Y = [], []
    for i in range(0, len(ws) - SEQ - HORIZON, HORIZON):
        X.append(np.stack([ws[i:i + SEQ], pw[i:i + SEQ], ws[i:i + SEQ] ** 2], axis=1))
        Y.append(pw[i + SEQ:i + SEQ + HORIZON])
    return np.array(X, np.float32), np.array(Y, np.float32)


class AveragePowerCurveTorch:
    def __init__(self, power_tr, wind_tr, bin_num=50, device="cuda"):
        self.bin_num = bin_num
        self.device = device
        self.p_mean, self.p_std = float(power_tr.mean()), float(power_tr.std() + 1e-8)
        self.w_mean, self.w_std = float(wind_tr.mean()), float(wind_tr.std() + 1e-8)
        # dynamic js_ratio (official train_clean.py lines ~326-331): disable JS when
        # the target power distribution is degenerate (std tiny / mean outside bins).
        # This runs ONCE at construction on the raw (pre-standardization) power.
        raw_p = power_tr
        bins_p_raw_min = float(raw_p.min()); bins_p_raw_max = float(raw_p.max())
        if float(raw_p.std()) < 1e-4 or float(raw_p.mean()) < bins_p_raw_min \
           or float(raw_p.mean()) > bins_p_raw_max:
            self.js_ratio = 0.0
        else:
            self.js_ratio = 1.0
        pn = (power_tr - self.p_mean) / self.p_std
        wn = (wind_tr - self.w_mean) / self.w_std
        self.bins_p = torch.linspace(float(pn.min()), float(pn.max()), bin_num, device=device)
        self.bins_w = torch.linspace(float(wn.min()), float(wn.max()), bin_num, device=device)
        self.bw = torch.pow(torch.tensor(float(bin_num), device=device), -1 / 5) * 0.1
        self.h_p = torch.std(pn) * self.bw + 1e-6
        self.h_w = torch.std(wn) * self.bw + 1e-6
        pnt = pn.reshape(1, -1, 1); wnt = wn.reshape(1, -1, 1)
        _, kp = marginal_pdf(pnt, self.bins_p, self.h_p)
        _, kw = marginal_pdf(wnt, self.bins_w, self.h_w)
        self.pc_target_outer = joint_pdf_outer(kp, kw)
        self.pc_target_true = true_joint_pdf(pnt, wnt, self.bins_p, self.bins_w, self.h_p, self.h_w)

    def js(self, pred_power, wind, mode):
        # dynamic switch: if disabled, return 0 (loss degrades to pure MSE), mirroring
        # official code's 'if powercurve.js_ratio > 0: loss = loss_local + js else: loss_local'
        if self.js_ratio == 0.0:
            return torch.tensor(0.0, device=self.device)
        p = pred_power.reshape(1, -1, 1)
        w = wind.reshape(1, -1, 1)
        if mode == "official":
            _, kp = marginal_pdf(p, self.bins_p, self.h_p)
            _, kw = marginal_pdf(w, self.bins_w, self.h_w)
            return js_official(joint_pdf_outer(kp, kw), self.pc_target_outer)
        else:
            pc_pred = true_joint_pdf(p, w, self.bins_p, self.bins_w, self.h_p, self.h_w)
            return js_standard(pc_pred, self.pc_target_true)


def run_arm(mode, apc, X_tr, Y_tr, X_te, Y_te, epochs, device="cuda", rng_seed=0):
    # RNG aligned per arm: identical shuffle order across arms, only constraint differs.
    torch.manual_seed(rng_seed); np.random.seed(rng_seed)
    model = NetLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    B = X_tr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(B)
        for i in range(0, B, 8):
            idx = perm[i:i + 8]
            x = torch.tensor(X_tr[idx], device=device)
            y = torch.tensor(Y_tr[idx], device=device)
            opt.zero_grad()
            out = model(x).squeeze(-1)
            mse = F.mse_loss(out, y[:, :1])
            wind_t = torch.tensor(X_tr[idx][:, -1, 0:1], device=device)
            js = apc.js(out, wind_t, mode)
            loss = mse + js
            if not torch.isfinite(loss):
                return float("nan")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    import pandas as pd
    df = pd.read_csv(SITES[args.site])
    ws = df["wind_speed"].to_numpy(np.float64); p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    p_rated, v_in, v_rated = fit_curve(ws[:int(len(ws) * 0.8)], p[:int(len(p) * 0.8)])
    pw = p / max(p_rated, 1e-6)
    wnd = ws / 15.0
    X, Y = build_windows(wnd, pw)
    n_tr = int(X.shape[0] * 0.8)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_te, Y_te = X[n_tr:], Y[n_tr:]

    wind_tr = torch.tensor(X_tr[:, -1, 0], dtype=torch.float32, device=device)
    power_tr = torch.tensor(Y_tr[:, 0], dtype=torch.float32, device=device)
    apc = AveragePowerCurveTorch(power_tr, wind_tr, device=device)

    t0 = time.time()
    rmse_a = run_arm("official", apc, X_tr, Y_tr, X_te, Y_te, args.epochs, device, rng_seed=args.seed)
    t1 = time.time()
    rmse_b = run_arm("corrected", apc, X_tr, Y_tr, X_te, Y_te, args.epochs, device, rng_seed=args.seed)
    t2 = time.time()
    line = (f"{args.site} seed={args.seed} gpu={args.gpu}  "
            f"official={rmse_a:.5f} corrected={rmse_b:.5f} "
            f"delta={(rmse_a - rmse_b):+.5f} ratio={rmse_a / max(rmse_b, 1e-12):.2f}  "
            f"[{t1 - t0:.0f}s/{t2 - t1:.0f}s]")
    print(line, flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
