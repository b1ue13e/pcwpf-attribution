"""PCT sensitivity: epochs and bandwidth-lambda grid (single site, fast).

Grid: epochs in {30, 60, 100} x lambda in {0.07, 0.1, 0.12} on kelmarsh,
both arms (official / corrected). Reports ratio per cell.
Reuses the faithful v3 pipeline (global fixed bins aligned to official).
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pct_fulltrain_v3_runner import (marginal_pdf, joint_pdf_outer, true_joint_pdf,
                                      js_official, js_standard, NetLSTM, build_windows,
                                      SITES, fit_curve, power_curve_pu)  # noqa: E402


class APC:
    def __init__(self, power_tr, wind_tr, lam, bin_num=50, device="cuda"):
        self.p_mean, self.p_std = float(power_tr.mean()), float(power_tr.std() + 1e-8)
        self.w_mean, self.w_std = float(wind_tr.mean()), float(wind_tr.std() + 1e-8)
        pn = (power_tr - self.p_mean) / self.p_std
        wn = (wind_tr - self.w_mean) / self.w_std
        self.bins_p = torch.linspace(float(pn.min()), float(pn.max()), bin_num, device=device)
        self.bins_w = torch.linspace(float(wn.min()), float(wn.max()), bin_num, device=device)
        self.bw = torch.pow(torch.tensor(float(bin_num), device=device), -1 / 5) * lam
        self.h_p = torch.std(pn) * self.bw + 1e-6
        self.h_w = torch.std(wn) * self.bw + 1e-6
        pnt = pn.reshape(1, -1, 1); wnt = wn.reshape(1, -1, 1)
        _, kp = marginal_pdf(pnt, self.bins_p, self.h_p)
        _, kw = marginal_pdf(wnt, self.bins_w, self.h_w)
        self.pc_target_outer = joint_pdf_outer(kp, kw)
        self.pc_target_true = true_joint_pdf(pnt, wnt, self.bins_p, self.bins_w, self.h_p, self.h_w)

    def js(self, pred, wind, mode):
        p = pred.reshape(1, -1, 1); w = wind.reshape(1, -1, 1)
        if mode == "official":
            _, kp = marginal_pdf(p, self.bins_p, self.h_p)
            _, kw = marginal_pdf(w, self.bins_w, self.h_w)
            return js_official(joint_pdf_outer(kp, kw), self.pc_target_outer)
        pc_pred = true_joint_pdf(p, w, self.bins_p, self.bins_w, self.h_p, self.h_w)
        return js_standard(pc_pred, self.pc_target_true)


def run_arm(mode, apc, X_tr, Y_tr, X_te, Y_te, epochs, device):
    model = NetLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    B = X_tr.shape[0]
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(B)
        for i in range(0, B, 8):
            idx = perm[i:i + 8]
            x = torch.tensor(X_tr[idx], device=device); y = torch.tensor(Y_tr[idx], device=device)
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
    ap.add_argument("--epochs", default="30,60,100")
    ap.add_argument("--lambdas", default="0.07,0.1,0.12")
    ap.add_argument("--bin-num", type=int, default=50)
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
    pw = p / max(p_rated, 1e-6); wnd = ws / 15.0
    X, Y = build_windows(wnd, pw)
    n_tr = int(X.shape[0] * 0.8)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_te, Y_te = X[n_tr:], Y[n_tr:]
    wind_tr = torch.tensor(X_tr[:, -1, 0], dtype=torch.float32, device=device)
    power_tr = torch.tensor(Y_tr[:, 0], dtype=torch.float32, device=device)

    epochs_list = [int(e) for e in args.epochs.split(",")]
    lam_list = [float(l) for l in args.lambdas.split(",")]
    out_lines = []
    for lam in lam_list:
        apc = APC(power_tr, wind_tr, lam, args.bin_num, device)
        for ep in epochs_list:
            t0 = time.time()
            ra = run_arm("official", apc, X_tr, Y_tr, X_te, Y_te, ep, device)
            rb = run_arm("corrected", apc, X_tr, Y_tr, X_te, Y_te, ep, device)
            dt = time.time() - t0
            line = (f"{args.site} seed={args.seed} lam={lam} epochs={ep} "
                    f"official={ra:.5f} corrected={rb:.5f} ratio={ra/max(rb,1e-12):.2f} [{dt:.0f}s]")
            print(line, flush=True)
            out_lines.append(line)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
