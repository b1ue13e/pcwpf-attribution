"""CCF constraint-audit benchmark (Contribution 2).

Audits ANY power-forecast ceiling under a deployment signal, using the
observable false-exclusion side (feasible-witness semantics, see
Contribution 1 theory note). Protocol invariants:

  - raw-time 60/20/20 train/calib/test split BEFORE windowing;
  - scaler / power curve / quantile model fit on TRAIN only;
  - quantile calibration on the CALIBRATION slice only;
  - target timestamps never overlap across the split (embargo H-1);
  - feasible witness Y only certifies the downward direction: Uhat < Y => FE;
    the reverse (Uhat >= Y) does NOT certify safety.

Metrics (all on the test slice):
  FER        = P(Uhat < Y)                false-exclusion rate
  LEC        = E[(Y - Uhat)^+ | FE]       legitimate energy clipped (conditional)
  coverage   = 1 - FER                    observable ceiling coverage
  sharpness  = E[Uhat]                    mean ceiling height (inverse conservatism proxy)
  ramp_fer   = FER restricted to high-ramp below-rated windows

Ceilings compared in one protocol:
  1. intrinsic_box     : Uhat = 1.0 (P_rated)           [deployable, zero FE by construction]
  2. oracle_curve      : fitted curve @ TRUE future wind [audit-only reference, NOT deployable]
  3. persistence_proxy : fitted curve @ v_last           [deployable deterministic proxy]
  4. linear_proxy      : fitted curve @ 2*v_last - v_prev[deployable deterministic proxy]
  5. ccf_quantile      : Q_alpha(U | H), pinball + split-conformal [deployable, calibrated]

Note: we do NOT fabricate a "manufacturer curve" — the project has no independent
manufacturer power-curve data; oracle_curve uses the site's train-fitted curve.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES, fit_curve, power_curve_pu  # noqa: E402


# ---------------------------------------------------------------------------
# Quantile forecaster (shared, minimal)
# ---------------------------------------------------------------------------
class QuantLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, pred_len, quantiles):
        super().__init__()
        self.quantiles = list(quantiles)
        self.pred_len = pred_len
        self.nq = len(quantiles)
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, pred_len * self.nq)

    def forward(self, x):
        out, _ = self.lstm(x)
        z = self.fc(out[:, -1, :])
        return z.view(z.size(0), self.pred_len, self.nq)


def pinball(pred_q, y, qs):
    y = y.unsqueeze(-1)
    err = y - pred_q
    q = torch.tensor(qs, device=pred_q.device, dtype=pred_q.dtype)
    return torch.where(err >= 0, q, q - 1.0).mul(err).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def audit_ceiling(Uhat: np.ndarray, Y: np.ndarray, ramp_mask: np.ndarray) -> dict:
    """Uhat, Y, ramp_mask are (B, H) arrays on the test slice."""
    fe = (Uhat < Y).astype(np.float64)
    n = fe.size
    fer = float(fe.mean())
    lec = float(np.where(fe > 0, (Y - Uhat) * fe, 0.0).mean()) if fe.any() else 0.0
    cov = 1.0 - fer
    sharp = float(Uhat.mean())
    ramp_fe = fe[ramp_mask]
    ramp_fer = float(ramp_fe.mean()) if ramp_fe.size else 0.0
    return {
        "FER": round(fer, 6),
        "LEC": round(lec, 6),
        "coverage": round(cov, 6),
        "sharpness": round(sharp, 6),
        "ramp_FER": round(ramp_fer, 6),
    }


# ---------------------------------------------------------------------------
# Ceiling constructors
# ---------------------------------------------------------------------------
def ceiling_manufacturer(v, v_in, v_rated):
    return power_curve_pu(v, v_in, v_rated)      # per-unit


def ceiling_persistence(v_last, v_in, v_rated):
    return power_curve_pu(v_last, v_in, v_rated)


def ceiling_linear(v_last, v_prev, v_in, v_rated):
    extrap = 2.0 * v_last - v_prev
    return power_curve_pu(extrap, v_in, v_rated)


def run_site(site, horizon, seed, epochs, alpha_grid, seq_len=48):
    from real_train_audit import fit_curve as _fc  # already imported

    df = pd.read_csv(SITES[site])
    ws = df["wind_speed"].to_numpy(np.float64)
    p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    n_all = len(ws)

    # raw-time 60/20/20 split
    i1 = int(n_all * 0.6)
    i2 = int(n_all * 0.8)
    ws_tr, p_tr = ws[:i1], p[:i1]
    ws_cal, p_cal = ws[i1:i2], p[i1:i2]
    ws_te, p_te = ws[i2:], p[i2:]

    # train-only curve + per-unit
    p_rated, v_in, v_rated = _fc(ws_tr, p_tr)
    ws_mean, ws_std = float(ws_tr.mean()), float(ws_tr.std() + 1e-6)

    def pu(a):
        return np.clip(a / max(p_rated, 1e-6), 0.0, 1.0)

    def norm(a):
        return (a - ws_mean) / ws_std

    def windows(ws_n, p_pu):
        n = len(ws_n) - seq_len - horizon + 1
        if n < 1:
            return (np.zeros((0, seq_len, 2)), np.zeros((0, horizon)),
                    np.zeros(0), np.zeros(0), np.zeros((0, horizon)))
        X = np.zeros((n, seq_len, 2), np.float32)
        Y = np.zeros((n, horizon), np.float32)
        v_last = np.zeros(n, np.float32)
        v_prev = np.zeros(n, np.float32)
        vr = np.zeros((n, horizon), np.float32)
        for i in range(n):
            X[i, :, 0] = ws_n[i:i + seq_len]
            X[i, :, 1] = ws_n[i:i + seq_len]
            Y[i] = p_pu[i + seq_len:i + seq_len + horizon]
            v_last[i] = ws_n[i + seq_len - 1]
            v_prev[i] = ws_n[i + seq_len - 2] if seq_len >= 2 else ws_n[i + seq_len - 1]
            vr[i] = ws_n[i + seq_len:i + seq_len + horizon]
        return X, Y, v_last, v_prev, vr

    Xtr, Ytr, vl_tr, vpr_tr, _ = windows(norm(ws_tr), pu(p_tr))
    Xcal, Ycal, vl_cal, vpr_cal, vr_cal = windows(norm(ws_cal), pu(p_cal))
    Xte, Yte, vl_te, vpr_te, vr_te = windows(norm(ws_te), pu(p_te))
    if len(Xte) == 0:
        raise RuntimeError("test split empty")

    # physical wind for ramp mask (test)
    vl_phys = vl_te * ws_std + ws_mean
    vr_phys = vr_te * ws_std + ws_mean
    r = np.abs(np.diff(vr_phys, axis=1, prepend=vr_phys[:, :1])).mean(axis=1)
    ramp_mask = ((vr_phys.mean(axis=1) < v_rated) & (r > np.quantile(r, 0.75)))[:, None]
    ramp_mask = np.repeat(ramp_mask, horizon, axis=1)

    # ---- deterministic ceilings ----
    rows = []
    Y_true = Yte  # per-unit realized power (feasible witness)

    def add(name, Uhat):
        m = audit_ceiling(Uhat, Y_true, ramp_mask)
        m["ceiling"] = name
        m["alpha"] = 1.0
        rows.append(m)

    # real future wind (oracle, audit-only) — the "observable upper bound" reference
    add("intrinsic_box", np.ones_like(Yte))
    # oracle curve: site's fitted curve evaluated at the TRUE future wind (audit-only reference;
    # we do NOT have a separate manufacturer curve, so "manufacturer" is NOT fabricated here).
    oracle_curve = np.repeat(ceiling_manufacturer(vr_phys.mean(axis=1), v_in, v_rated)[:, None], horizon, 1)
    add("oracle_curve", oracle_curve)
    # deployment-available proxies use LAST-seen wind (persistence) or linear extrapolation
    add("persistence_proxy", np.repeat(ceiling_persistence(vl_phys, v_in, v_rated)[:, None], horizon, 1))
    add("linear_proxy", np.repeat(ceiling_linear(vl_phys, vpr_te * ws_std + ws_mean, v_in, v_rated)[:, None], horizon, 1))

    # ---- train quantile model ----
    qs = [0.5, 0.7, 0.8, 0.9, 0.95]
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantLSTM(2, 32, horizon, qs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr_t = torch.from_numpy(Xtr); Ytr_t = torch.from_numpy(Ytr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for bi in range(0, len(Xtr_t), 64):
            xb, yb = Xtr_t[perm[bi:bi + 64]].to(device), Ytr_t[perm[bi:bi + 64]].to(device)
            loss = pinball(model(xb), yb, qs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    def predict(Xn):
        model.eval()
        out = []
        with torch.no_grad():
            for bi in range(0, len(Xn), 256):
                out.append(model(torch.from_numpy(Xn[bi:bi + 256]).to(device)).cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, horizon, len(qs)))

    q_cal = predict(Xcal)
    q_te = predict(Xte)

    # ---- CCF quantile ceiling at each alpha, with split-conformal ----
    for alpha in alpha_grid:
        qi = int(np.argmin([abs(l - (1.0 - alpha)) for l in qs]))
        base_te = q_te[:, :, qi] if len(q_te) else np.zeros_like(Yte)
        # conformal on calibration
        if len(q_cal):
            base_cal = q_cal[:, :, qi]
            score = np.maximum(Ycal - base_cal, 0.0).flatten()
            k = min(int(np.ceil((1.0 - alpha) * (len(score) + 1))), len(score) - 1)
            k = max(k, 0)
            qhat = np.partition(score, k)[k]
        else:
            qhat = 0.0
        Uhat = base_te + qhat
        m = audit_ceiling(Uhat, Y_true, ramp_mask)
        m["ceiling"] = "ccf_quantile"
        m["alpha"] = alpha
        rows.append(m)

    out = pd.DataFrame(rows)
    out["site"] = site
    out["horizon"] = horizon
    out["seed"] = seed
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="wind_cn,kelmarsh,smarteole,penmanshiel,wind_us")
    ap.add_argument("--horizons", default="6,12,24")
    ap.add_argument("--seeds", default="42,123")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="ccf_benchmark.csv")
    args = ap.parse_args()
    rows = []
    for site in args.sites.split(","):
        for h in [int(x) for x in args.horizons.split(",")]:
            for s in [int(x) for x in args.seeds.split(",")]:
                print(f"[{site}] h={h} seed={s}", flush=True)
                rows.append(run_site(site, h, s, args.epochs, [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n=== saved {len(out)} rows -> {args.out} ===")
    piv = out[out.alpha == 1.0].groupby(["site", "ceiling"]).agg(
        FER=("FER", "mean"), LEC=("LEC", "mean")).round(4)
    print("deterministic ceilings (alpha=1.0):")
    print(piv.to_string())


if __name__ == "__main__":
    main()
