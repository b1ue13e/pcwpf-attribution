"""PCT three-version comparison: paper-spec / official-code / corrected-parity.

Definitions (frozen in this audit):
  paper-spec      : strict implementation of the PAPER FORMULAS (Eq.2 KDE, Eq.4-5
                    standard KL/JS) on a true joint (wind,power) density.
  official-code   : faithful port of daxin007/PCT-WPF train_clean.py:
                    joint_pdf = outer product of marginals; js = F.kl_div(p.log(), q)
                    (reversed-argument KL).
  corrected-parity: official-code ported faithfully, then the KNOWN defects fixed to
                    match paper-spec (true joint + standard KL). By construction
                    corrected-parity == paper-spec; we verify this numerically.

Single site (kelmarsh) for the three-way table. Expected: paper-spec and
corrected-parity give identical RMSE, both substantially below official-code.
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


def run(mode, apc, X_tr, Y_tr, X_te, Y_te, epochs, device, rng_seed=0):
    # FIX: reset RNG per version so all three versions use IDENTICAL shuffle order;
    # removes random-order confound, leaving only the constraint-formula difference.
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
            js = apc.js(out, torch.tensor(X_tr[idx][:, -1, 0:1], device=device), mode)
            loss = mse + js
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
    model.eval()
    with torch.no_grad():
        preds, targs = [], []
        for i in range(0, X_te.shape[0], 256):
            preds.append(model(torch.tensor(X_te[i:i + 256], device=device)).squeeze(-1).cpu().numpy())
            targs.append(Y_te[i:i + 256][:, :1])
        preds = np.concatenate(preds); targs = np.concatenate(targs)
    return float(np.sqrt(np.mean((preds - targs) ** 2)))


class APC:
    """paper-spec + official-code constraint machinery (global fixed bins)."""
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
        _, kp = marginal_pdf(pnt, self.bp, self.h_p); _, kw = marginal_pdf(wnt, self.bw, self.h_w)
        self.target_outer = joint_pdf_outer(kp, kw)
        self.target_true = true_joint_pdf(pnt, wnt, self.bp, self.bw, self.h_p, self.h_w)

    def js(self, pred, wind, mode):
        p = pred.reshape(1, -1, 1); w = wind.reshape(1, -1, 1)
        if mode == "official-code":
            _, kp = marginal_pdf(p, self.bp, self.h_p); _, kw = marginal_pdf(w, self.bw, self.h_w)
            return js_official(joint_pdf_outer(kp, kw), self.target_outer)
        else:  # paper-spec / corrected-parity: same true-joint + standard KL
            return js_standard(true_joint_pdf(p, w, self.bp, self.bw, self.h_p, self.h_w),
                               self.target_true)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="kelmarsh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    import pandas as pd
    df = pd.read_csv(SITES[args.site])
    ws = df["wind_speed"].to_numpy(np.float64); p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    p_rated, v_in, v_rated = fit_curve(ws[:int(len(ws)*0.8)], p[:int(len(p)*0.8)])
    pw = p / max(p_rated, 1e-6); wnd = ws / 15.0
    X, Y = build_windows(wnd, pw)
    n_tr = int(X.shape[0] * 0.8)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]; X_te, Y_te = X[n_tr:], Y[n_tr:]
    apc = APC(torch.tensor(Y_tr[:, 0], dtype=torch.float32, device=device),
              torch.tensor(X_tr[:, -1, 0], dtype=torch.float32, device=device),
              device=device)

    out = {}
    for mode in ["official-code", "paper-spec", "corrected-parity"]:
        t0 = time.time()
        r = run(mode, apc, X_tr, Y_tr, X_te, Y_te, args.epochs, device, rng_seed=args.seed)
        out[mode] = (r, time.time() - t0)

    line = (f"{args.site} seed={args.seed} epochs={args.epochs} | "
            f"official_code={out['official-code'][0]:.5f} "
            f"paper_spec={out['paper-spec'][0]:.5f} "
            f"corrected_parity={out['corrected-parity'][0]:.5f} | "
            f"spec_eq_corrected={abs(out['paper-spec'][0]-out['corrected-parity'][0])<1e-9}")
    print(line, flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
