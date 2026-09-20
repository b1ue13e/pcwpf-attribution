"""PI-TFN loss-level faithful reconstruction (paper-spec) + ablation.

Audits PI-TFN's four pointwise penalties (Eq. 20):
  L_cutin, L_rated, L_nonneg, L_phys (Betz residual with learned Cp estimator)

Variants:
  A. paper-spec   : all 4 penalties, Cp from learned estimator (64->32->1)
  B. no-physics   : MSE only (unconstrained)
  C. no-phi       : 3 boundary penalties only (drop L_phys; isolates Betz-residual term)
  D. fixed-cp     : L_phys uses a FIXED Betz Cp=16/27 (wrong/idealized prior) instead
                    of learned Cp  -> prior-misspecification probe (parallels nwp-linear)

This is a LOSS-LEVEL reconstruction (no full multi-module architecture), matching the
audit level documented in PI-TFN第二族审计方案.md.
"""
from __future__ import annotations
import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pct_fulltrain_v3_runner import build_windows, SITES, fit_curve  # noqa: E402


class BiLSTM_Backbone(nn.Module):
    """Reduced PI-TFN backbone: bidirectional LSTM + physical MLP for Cp."""
    def __init__(self, dim_in=3, hidden=64, dim_out=1):
        super().__init__()
        self.lstm = nn.LSTM(dim_in, hidden, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, dim_out)
        # Cp estimator: physical MLP 64->32->1 (paper spec: Betz-limited)
        # FIX (re-review): sigmoid gives Cp in [0,1]; paper says Betz-limited, so
        # scale by 16/27 -> Cp in [0, 0.593].
        self.cp_est = nn.Sequential(
            nn.Linear(hidden * 2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        cp = self.cp_est(h) * betz_cp_max()  # (B,1) in [0, 16/27] (Betz-limited)
        return self.fc(h), cp


def betz_cp_max():
    return 16.0 / 27.0  # ≈0.593


def pi_tfn_loss(pred, cp, wind_real, p_rated, v_in, v_rated, rho, area,
                variant="paper-spec", lmb=(0.1, 0.1, 0.05, 0.05)):
    """Pointwise penalties per PI-TFN Eq. 20. All tensors shape (B,1).
    pred: predicted power (pu, already /p_rated). wind_real: m/s.
    Returns a dict of losses (MSE added by caller)."""
    B = pred.shape[0]
    # masks
    # FIX (re-review): L_rated mask must be PREDICTED power > P_rated (paper Eq.20:
    # R={i: ŷ_i > P_rated}), NOT wind > v_rated. pred is pu so rated=1.0.
    S = (wind_real < v_in).float()
    R = (pred > 1.0).float()
    N = (pred < 0).float()
    V = ((wind_real >= v_in) & (wind_real <= v_rated)).float()
    l_cutin = (S * F.relu(pred) ** 2).sum() / (S.sum() + 1e-8)
    l_rated = (R * (pred - 1.0) ** 2).sum() / (R.sum() + 1e-8)  # pred pu, rated=1.0
    l_nonneg = (N * pred ** 2).sum() / (N.sum() + 1e-8)
    # P^phys in pu: 0.5*rho*A*cp*v^3 is in W; convert to kW (/1000) then /p_rated(kW)
    # FIX (round-2 review): previous code divided W by p_rated(kW) -> 1000x too large.
    if variant == "fixed-cp":
        cp_used = torch.full_like(cp, betz_cp_max())
    else:
        cp_used = cp
    p_phys_W = 0.5 * rho * area * cp_used * (wind_real ** 3)
    p_phys = (p_phys_W / 1000.0) / p_rated
    l_phys = (V * (pred - p_phys) ** 2).sum() / (V.sum() + 1e-8)
    return {"cutin": l_cutin, "rated": l_rated, "nonneg": l_nonneg, "phys": l_phys}


def run_variant(variant, epochs, X_tr, Y_tr, X_te, Y_te, wind_real_tr, p_rated,
                v_in, v_rated, rho, area, device, rng_seed=0):
    torch.manual_seed(rng_seed); np.random.seed(rng_seed)
    model = BiLSTM_Backbone(dim_in=X_tr.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    B = X_tr.shape[0]
    for _ in range(epochs):
        model.train(); perm = np.random.permutation(B)
        for i in range(0, B, 8):
            idx = perm[i:i + 8]
            x = torch.tensor(X_tr[idx], device=device)
            y = torch.tensor(Y_tr[idx], device=device)
            wr = torch.tensor(wind_real_tr[idx], device=device)
            opt.zero_grad()
            out, cp = model(x)
            mse = F.mse_loss(out, y[:, :1])
            if variant == "no-physics":
                loss = mse
            else:
                d = pi_tfn_loss(out, cp, wr, p_rated, v_in, v_rated, rho, area, variant)
                if variant == "no-phi":
                    loss = mse + 0.1 * d["cutin"] + 0.1 * d["rated"] + 0.05 * d["nonneg"]
                else:
                    loss = (mse + 0.1 * d["cutin"] + 0.1 * d["rated"]
                            + 0.05 * d["nonneg"] + 0.05 * d["phys"])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
    model.eval()
    with torch.no_grad():
        preds, targs = [], []
        for i in range(0, X_te.shape[0], 256):
            out, _ = model(torch.tensor(X_te[i:i + 256], device=device))
            preds.append(out.cpu().numpy()); targs.append(Y_te[i:i + 256][:, :1])
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
    n0 = int(len(ws) * 0.8)
    p_rated, v_in, v_rated = fit_curve(ws[:n0], p[:n0])
    pw = p / max(p_rated, 1e-6)
    # wind pu for model input; wind_real (m/s) for physics penalties
    X, Y = build_windows(ws / 15.0, pw)
    n_tr = int(X.shape[0] * 0.8)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]; X_te, Y_te = X[n_tr:], Y[n_tr:]
    # wind_real per window = last-step wind * 15 (undo normalization)
    wind_real_tr = X_tr[:, -1, 0] * 15.0
    # physics params (PI-TFN paper: rho=1.225, A = rotor area)
    # FIX (re-review): rotor area was previously back-computed from p_rated (data-
    # derived, not physical). Kelmarsh = 6× Senvion MM92 (merged), rotor D=92 m,
    # A_single = pi*(D/2)^2 = 6647.6 m^2; use 6× A_single for the merged series.
    # HONEST CAVEAT: the paper does not state its exact rotor area / aggregation,
    # so A is a physical-reasoned assumption, NOT a paper-spec value.
    rho = 1.225
    area_single = np.pi * (92.0 / 2.0) ** 2  # 6647.6 m^2 (Senvion MM92)
    area = 6.0 * area_single  # merged 6-turbine series
    results = {}
    for v in ["paper-spec", "no-physics", "no-phi", "fixed-cp"]:
        t0 = time.time()
        r = run_variant(v, args.epochs, X_tr, Y_tr, X_te, Y_te, wind_real_tr,
                        p_rated, v_in, v_rated, rho, area, device, rng_seed=args.seed)
        results[v] = r
    line = (f"{args.site} seed={args.seed} epochs={args.epochs} | "
            f"paper_spec={results['paper-spec']:.5f} unconst={results['no-physics']:.5f} "
            f"no_phi={results['no-phi']:.5f} fixed_cp={results['fixed-cp']:.5f}")
    print(line, flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
