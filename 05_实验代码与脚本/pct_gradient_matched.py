"""Gradient-matched comparison: reversed-KL (official) vs standard JS (corrected).

Reviewer point: comparing official vs corrected with a fixed 1:1 loss weight
confounds "wrong divergence" with "different loss magnitude" (reversed KL is
6-11.5x larger than JS). To isolate the DIVERGENCE effect, we scale each
constraint term so its gradient norm contribution matches an unconstrained
baseline's MSE gradient norm, per step.

Variants (kelmarsh, seed=0, epochs=40):
  unconstrained  : MSE only
  reversed-kl    : official reversed-KL, gradient-matched to MSE
  standard-js    : corrected JS, gradient-matched to MSE
  reversed-kl-raw: official reversed-KL with the paper's literal 1:1 weight
                   (for reference, this is the confounded comparison)
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pct_fulltrain_v3_runner import (marginal_pdf, joint_pdf_outer, true_joint_pdf,
                                      js_official, js_standard, NetLSTM,
                                      build_windows, SITES, fit_curve)  # noqa: E402


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
        _, kp = marginal_pdf(pnt, self.bp, self.h_p)
        _, kw = marginal_pdf(wnt, self.bw, self.h_w)
        self.t_outer = joint_pdf_outer(kp, kw)
        self.t_true = true_joint_pdf(pnt, wnt, self.bp, self.bw, self.h_p, self.h_w)

    def constraint(self, pred, wind, kind):
        p = pred.reshape(1, -1, 1); w = wind.reshape(1, -1, 1)
        if kind == "reversed-kl":
            _, kp = marginal_pdf(p, self.bp, self.h_p)
            _, kw = marginal_pdf(w, self.bw, self.h_w)
            return js_official(joint_pdf_outer(kp, kw), self.t_outer)
        elif kind == "standard-js":
            return js_standard(true_joint_pdf(p, w, self.bp, self.bw, self.h_p, self.h_w),
                               self.t_true)
        raise ValueError(kind)


def grads_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False)
    g = [x.view(-1) for x in g if x is not None]
    if not g:
        return torch.tensor(0.0, device=loss.device)
    return torch.cat(g).norm()


def run(kind, apc, X_tr, Y_tr, X_te, Y_te, epochs, device, rng_seed=0, grad_match=True):
    torch.manual_seed(rng_seed); np.random.seed(rng_seed)
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
            if kind == "unconstrained":
                loss = mse
            else:
                c = apc.constraint(out, torch.tensor(X_tr[idx][:, -1, 0:1], device=device), kind)
                if grad_match:
                    # scale constraint to match MSE gradient norm (per-step)
                    gm = grads_norm(mse, list(model.parameters()))
                    gc = grads_norm(c, list(model.parameters()))
                    scale = (gm / (gc + 1e-12)).detach()
                    loss = mse + c * scale
                else:
                    loss = mse + c  # literal 1:1 (confounded)
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
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
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
              torch.tensor(X_tr[:, -1, 0], dtype=torch.float32, device=device), device=device)
    r = {}
    for kind in ["unconstrained", "reversed-kl", "standard-js"]:
        r[kind] = run(kind, apc, X_tr, Y_tr, X_te, Y_te, args.epochs, device, args.seed, grad_match=(kind != "unconstrained"))
    # confounded reference (1:1)
    r["reversed-kl-raw"] = run("reversed-kl", apc, X_tr, Y_tr, X_te, Y_te, args.epochs, device, args.seed, grad_match=False)
    line = (f"{args.site} seed={args.seed} epochs={args.epochs} | "
            f"uncon={r['unconstrained']:.5f} "
            f"rkl_gradmatch={r['reversed-kl']:.5f} "
            f"js_gradmatch={r['standard-js']:.5f} "
            f"rkl_raw1to1={r['reversed-kl-raw']:.5f}")
    print(line, flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
