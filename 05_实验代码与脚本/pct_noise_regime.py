"""PCT claimed scenario: high-noise wind speed. Does the oracle distributional
constraint actually help THERE (as PCT claims), despite failing in clean deployment?

If oracle constraint helps under noisy wind but not clean deployment, the audit
conclusion is 'constraint benefit is regime-dependent' (narrow, defensible).
If it does not help even there, we have a stronger but riskier claim.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pct_fulltrain_v3_runner import (marginal_pdf, true_joint_pdf, js_standard,
                                      NetLSTM, build_windows, SITES, fit_curve)  # noqa: E402


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


def run(use_constraint, apc, X_tr, Y_tr, X_te, Y_te, epochs, device, rng_seed=0):
    torch.manual_seed(rng_seed); np.random.seed(rng_seed)
    model = NetLSTM().to(device)
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
    ap.add_argument("--noise", type=float, default=0.5)  # std of noise added to wind
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    import pandas as pd
    df = pd.read_csv(SITES[args.site])
    ws = df["wind_speed"].to_numpy(np.float64); p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    n0 = int(len(ws) * 0.8)
    p_rated, v_in, v_rated = fit_curve(ws[:n0], p[:n0])
    pw = p / max(p_rated, 1e-6)
    X, Y = build_windows(ws / 15.0, pw)
    n_tr = int(X.shape[0] * 0.8)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]; X_te, Y_te = X[n_tr:], Y[n_tr:]

    # NOISE: add to wind features in TRAIN (both arms get same noisy inputs; test clean)
    rng = np.random.default_rng(args.seed)
    X_tr_noisy = X_tr.copy()
    X_tr_noisy[:, :, 0] = X_tr[:, :, 0] + rng.normal(0, args.noise, size=X_tr[:, :, 0].shape)
    X_te_noisy = X_te.copy()
    X_te_noisy[:, :, 0] = X_te[:, :, 0] + rng.normal(0, args.noise, size=X_te[:, :, 0].shape)

    apc = APC(torch.tensor(Y_tr[:, 0], dtype=torch.float32, device=device),
              torch.tensor(X_tr_noisy[:, -1, 0], dtype=torch.float32, device=device), device=device)
    r_const = run(True, apc, X_tr_noisy, Y_tr, X_te_noisy, Y_te, args.epochs, device, rng_seed=args.seed)
    r_un = run(False, apc, X_tr_noisy, Y_tr, X_te_noisy, Y_te, args.epochs, device, rng_seed=args.seed)
    line = (f"{args.site} seed={args.seed} noise={args.noise} epochs={args.epochs} | "
            f"oracle_constraint={r_const:.5f} unconstrained={r_un:.5f} | "
            f"gain={((r_un-r_const)/r_un*100):+.2f}%")
    print(line, flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
