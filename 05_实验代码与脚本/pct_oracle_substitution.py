"""Prior-source stress test (audit paper Contribution 3; previously mis-named
'oracle-to-deployment substitution' -- withdrawn per reviewer ruling).

Question: does the claimed benefit of the PCT-style distributional constraint
survive when the constraint's "target distribution" is built from different
prior sources (provenance)? None of these are deployment oracles; the paper
correctly builds the target from historical training data and freezes it.

Five prior-source conditions (kelmarsh single site, epochs=60, aligned RNG):

  A. historical-static : target pc from TRAIN-set true power (the paper's own
                         legitimate frozen training prior; NOT an oracle)
  B. lagged-measured   : target pc from LAGGED (past-hour) measured power;
                         causal, observable at deployment
  C. wind-curve-linear : target pc from LAST-OBSERVED wind + manual ideal linear
                         power curve (NOT a real NWP; naive physics prior)
  D. wind-curve-measured: last-observed wind + bin-mean measured power curve
  E. unconstrained     : MSE only (no physics constraint)

Metric: test RMSE (pu). Hypothesis: constraint benefit depends on prior
provenance and distribution stability (not on 'observability').
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pct_fulltrain_v3_runner import (marginal_pdf, true_joint_pdf, js_standard,
                                      NetLSTM, build_windows, SITES, fit_curve)  # noqa: E402


class APC_Source:
    """Distributional constraint with a SWAPPABLE target-distribution source."""
    def __init__(self, bin_num=50, device="cuda"):
        self.bin_num = bin_num; self.device = device
        self.bp = None; self.bw = None
        self.h_p = 1e-3; self.h_w = 1e-3
        self.target_true = None

    def set_target(self, power_arr, wind_arr):
        """power_arr/wind_arr = torch tensors (already normalized pu-ish) for density.
        NaN entries are dropped (used by lagged-measured where the first `lag`
        samples have no valid past)."""
        keep = ~torch.isnan(power_arr) & ~torch.isnan(wind_arr)
        power_arr = power_arr[keep]; wind_arr = wind_arr[keep]
        self.pm, self.ps = float(power_arr.mean()), float(power_arr.std() + 1e-8)
        self.wm, self.ws = float(wind_arr.mean()), float(wind_arr.std() + 1e-8)
        pn = (power_arr - self.pm) / self.ps
        wn = (wind_arr - self.wm) / self.ws
        self.bp = torch.linspace(float(pn.min()), float(pn.max()), self.bin_num, device=self.device)
        self.bw = torch.linspace(float(wn.min()), float(wn.max()), self.bin_num, device=self.device)
        self.h_p = torch.std(pn) * torch.pow(torch.tensor(float(self.bin_num), device=self.device), -1/5) * 0.1 + 1e-6
        self.h_w = torch.std(wn) * torch.pow(torch.tensor(float(self.bin_num), device=self.device), -1/5) * 0.1 + 1e-6
        pnt = pn.reshape(1, -1, 1); wnt = wn.reshape(1, -1, 1)
        self.target_true = true_joint_pdf(pnt, wnt, self.bp, self.bw, self.h_p, self.h_w)

    def js(self, pred, wind):
        p = pred.reshape(1, -1, 1); w = wind.reshape(1, -1, 1)
        return js_standard(true_joint_pdf(p, w, self.bp, self.bw, self.h_p, self.h_w), self.target_true)


def fit_curve_numpy(ws_tr, p_tr):
    """Return (p_rated, v_in, v_rated) from a rough 1D power curve fit (bin means)."""
    bins = np.linspace(np.percentile(ws_tr, 0.5), np.percentile(ws_tr, 99.5), 30)
    idx = np.digitize(ws_tr, bins)
    means = np.array([p_tr[idx == i].mean() if (idx == i).sum() > 5 else np.nan for i in range(1, len(bins))])
    bcs = (bins[:-1] + bins[1:]) / 2
    ok = np.isfinite(means)
    p_rated = float(np.nanmax(means)); v_rated = float(bcs[np.nanargmax(means)])
    v_in = float(bins[0]) if np.isfinite(means).any() else 3.0
    return p_rated, v_in, v_rated


def run_source(cond, apc, X_tr, Y_tr, X_te, Y_te, epochs, device, rng_seed=0):
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
            if cond == "unconstrained":
                loss = mse
            else:
                js = apc.js(out, torch.tensor(X_tr[idx][:, -1, 0:1], device=device))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="kelmarsh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lag", type=int, default=6)  # 1 hour at 10-min resolution
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    import pandas as pd
    df = pd.read_csv(SITES[args.site])
    ws = df["wind_speed"].to_numpy(np.float64); p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    n0 = int(len(ws) * 0.8)
    p_rated, v_in, v_rated = fit_curve_numpy(ws[:n0], p[:n0])
    pw = p / max(p_rated, 1e-6); wnd = ws / 15.0
    X, Y = build_windows(wnd, pw)
    n_tr = int(X.shape[0] * 0.8)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]; X_te, Y_te = X[n_tr:], Y[n_tr:]

    # constraint-source targets (all torch tensors)
    t_pw = torch.tensor(Y_tr[:, 0], dtype=torch.float32, device=device)
    t_wnd = torch.tensor(X_tr[:, -1, 0], dtype=torch.float32, device=device)
    # lagged-measured: previous-hour measured power (causal, available at deployment)
    # FIX v2 (re-reviewer): shift lag_pw[i] = Y_tr[i - lag] (PAST power).
    #   - The ORIGINAL bug assigned lag_pw[i] = Y_tr[i+lag] (FUTURE) -- VOID.
    #   - v1 fix left the first `lag` samples holding CURRENT Y (not past, but not
    #     a valid causal lag either). v2 fix: first `lag` samples have NO valid
    #     past -> mark NaN so they contribute nothing to the constraint.
    lag_pw = Y_tr[:, 0].copy() * np.nan
    lag_pw = Y_tr[:, 0].copy()
    lag_pw[args.lag:] = Y_tr[:-args.lag, 0]
    lag_pw[:args.lag] = np.nan
    lag_pw = torch.tensor(lag_pw, dtype=torch.float32, device=device)
    # wind-curve: LAST-OBSERVED wind (real m/s) -> manual power-curve mapping (NOT a real NWP; honest naming per reviewer)
    # v_in/v_rated in m/s; output pu power in [0,1].
    # TWO variants of the curve model:
    #   wind-curve-linear : ideal linear ramp (naive physics prior, NOT NWP)
    #   wind-curve-measured: bin-mean MEASURED power curve
    wf_real = X_tr[:, -1, 0] * 15.0  # undo /15.0 normalization
    wcl = np.clip((wf_real - v_in) / max(v_rated - v_in, 1e-6), 0, 1)
    # measured curve: bin means of training power over wind
    bins = np.linspace(v_in, v_rated, 20)
    idx = np.digitize(ws[:n0], bins)
    means_pu = []
    for i in range(len(bins) - 1):
        m = (p[:n0][idx == i].mean() / max(p_rated, 1e-6)) if (idx == i).sum() > 10 else np.nan
        means_pu.append(m)
    bcs = (bins[:-1] + bins[1:]) / 2
    okb = np.isfinite(means_pu)
    wcm = np.clip(np.interp(wf_real, bcs[okb], np.array(means_pu)[okb], left=0.0, right=1.0), 0, 1)

    wcl_pw = torch.tensor(wcl, dtype=torch.float32, device=device)
    wcm_pw = torch.tensor(wcm, dtype=torch.float32, device=device)

    apc = APC_Source(device=device)
    results = {}
    for cond, src_p, src_w in [
        ("oracle-train", t_pw, t_wnd),
        ("lagged-measured", lag_pw, t_wnd),
        ("wind-curve-linear", wcl_pw, t_wnd),
        ("wind-curve-measured", wcm_pw, t_wnd),
    ]:
        apc.set_target(src_p, src_w)
        t0 = time.time()
        r = run_source(cond, apc, X_tr, Y_tr, X_te, Y_te, args.epochs, device, rng_seed=args.seed)
        results[cond] = r
    results["unconstrained"] = run_source("unconstrained", apc, X_tr, Y_tr, X_te, Y_te,
                                          args.epochs, device, rng_seed=args.seed)
    line = (f"{args.site} seed={args.seed} epochs={args.epochs} | "
            f"oracle={results['oracle-train']:.5f} "
            f"lagged_measured={results['lagged-measured']:.5f} "
            f"wcl={results['wind-curve-linear']:.5f} "
            f"wcm={results['wind-curve-measured']:.5f} "
            f"unconstrained={results['unconstrained']:.5f}")
    print(line, flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
