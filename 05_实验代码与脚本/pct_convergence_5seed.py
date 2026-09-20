"""5-seed convergence check: does PCT's reversed-KL JS constraint destabilize training?

Official arm uses js_official (F.kl_div(p.log(), q) arg-order bug + outer-product joint);
corrected arm uses js_standard (true 2D joint + correct KL). Both: loss = mse + js.

Reports per-seed final loss + whether training diverged, for both arms on kelmarsh.
GPU-friendly, quick (8 epochs).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES, fit_curve
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import importlib.util

spec = importlib.util.spec_from_file_location("m", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pct_fulltrain_single.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    df = pd.read_csv(SITES["kelmarsh"])
    ws = df["wind_speed"].to_numpy(np.float64); p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    pr, vi, vr = fit_curve(ws[:int(len(ws) * 0.8)], p[:int(len(p) * 0.8)])
    pw = p / max(pr, 1e-6); wnd = ws / 15.0
    X, Y = m.build_windows(wnd, pw)
    n = int(X.shape[0] * 0.8)
    Xtr = torch.tensor(X[:n]).to(DEVICE); Ytr = torch.tensor(Y[:n]).to(DEVICE)

    def train_short(mode, seed, epochs=8):
        torch.manual_seed(seed); np.random.seed(seed)
        model = m.NetLSTM().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        traj = []
        for ep in range(epochs):
            model.train()
            perm = np.random.permutation(n)
            el = []
            for i in range(0, n, 8):
                idx = perm[i:i + 8]
                x = Xtr[idx]; y = Ytr[idx]
                opt.zero_grad()
                out = model(x).squeeze(-1)
                mse = F.mse_loss(out, y[:, :1])
                js = m.constraint_terms(out, x[:, -1, 0:1], y[:, :1], mode)
                loss = mse + js
                if not torch.isfinite(loss):
                    return dict(mode=mode, seed=seed, div_epoch=ep,
                                final=float("nan"), traj=traj)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                el.append(loss.item())
            traj.append(float(np.mean(el)))
        return dict(mode=mode, seed=seed, div_epoch=-1, final=traj[-1], traj=traj)

    print("seed  official_final  corrected_final  official_div corrected_div")
    rows = []
    for seed in [0, 1, 2, 3, 4]:
        a = train_short("official", seed)
        b = train_short("corrected", seed)
        rows.append((a, b))
        print(f"{seed}    {a['final']:>12.3f}    {b['final']:>12.3f}    "
              f"{'ep'+str(a['div_epoch']) if a['div_epoch']>=0 else '-':>11}     "
              f"{'ep'+str(b['div_epoch']) if b['div_epoch']>=0 else '-':>11}")
    # summary
    off_div = sum(1 for a, b in rows if a["div_epoch"] >= 0)
    corr_div = sum(1 for a, b in rows if b["div_epoch"] >= 0)
    off_final = [a["final"] for a, b in rows if a["div_epoch"] < 0]
    corr_final = [b["final"] for a, b in rows if b["div_epoch"] < 0]
    print(f"\nSUMMARY: official diverged in {off_div}/5 seeds, corrected in {corr_div}/5")
    if off_final: print(f"official non-div final loss mean={np.mean(off_final):.3f}")
    if corr_final: print(f"corrected non-div final loss mean={np.mean(corr_final):.3f}")
    import json
    with open("/root/paper1/results/pct_convergence_5seed.json", "w") as f:
        json.dump([{**a, "_": 0} if False else {"official": a, "corrected": b} for a, b in rows], f, indent=2)


if __name__ == "__main__":
    main()
