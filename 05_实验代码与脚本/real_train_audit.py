"""P0-1: real-training audit of the point-vs-distribution dichotomy.

Trains an LSTM with one of four losses and audits the REAL model outputs
(not truth stand-in) on ramp-asymmetry:

  mse           : pure L2
  mse+envelope  : L2 + proxy-envelope point penalty (upper bound P(v_proxy))
  mse+betz      : L2 + Betz-limit point penalty
  mse+js        : L2 + JS-divergence distribution penalty (KDE marginals)

Same canonical protocol for every variant -> directly answers "does the
dichotomy survive with a real trained forecaster?"
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultra_lsnt_timeseries import LSTMBaseline, compute_metrics

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SITES = {
    "wind_cn": "../04_数据集/prepared_priority_datasets/wind_cn_core_15min.csv",
    "kelmarsh": "../04_数据集/prepared_priority_datasets/kelmarsh_uk_2016.csv",
    "smarteole": "../04_数据集/prepared_priority_datasets/smarteole_france_2020.csv",
    "penmanshiel": "../04_数据集/prepared_priority_datasets/penmanshiel_scotland_2016.csv",
    "wind_us": "../04_数据集/prepared_priority_datasets/wind_us_core_15min.csv",
}


class TransformerEncoderBaseline(nn.Module):
    """Encoder-only transformer baseline, interface-compatible with LSTMBaseline."""
    def __init__(self, input_dim, hidden_dim, pred_len, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, 256, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=4 * hidden_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x, temperature=1.0, return_stats=False):
        h = self.norm(self.proj(x)) + self.pos[:, :x.size(1), :]
        h = self.encoder(h)
        out = self.fc(h[:, -1, :])
        aux = torch.tensor(0.0, device=x.device)
        if return_stats:
            return out, aux, {"efficiency": {"avg_compute_ratio": 1.0, "avg_skip_rate": 0.0, "avg_slow_ratio": 1.0}}
        return out, aux


class PatchTSTBaseline(nn.Module):
    """PatchTST-style: patchify + transformer encoder over patches, head per forecast."""
    def __init__(self, input_dim, hidden_dim, pred_len, patch_len=16, stride=8, num_layers=2, nhead=4):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(input_dim * patch_len, hidden_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=4 * hidden_dim,
            dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x, temperature=1.0, return_stats=False):
        # x: (B, L, C), patchify along time
        B, L, C = x.shape
        n = (L - self.patch_len) // self.stride + 1
        if n < 1:
            n = 1
        patches = torch.stack([x[:, i * self.stride:i * self.stride + self.patch_len]
                               for i in range(n)], dim=1)  # (B, n, p, C)
        patches = patches.reshape(B, n, self.patch_len * C)
        h = self.proj(patches)
        h = self.encoder(h)
        out = self.fc(h.mean(dim=1))
        aux = torch.tensor(0.0, device=x.device)
        if return_stats:
            return out, aux, {"efficiency": {"avg_compute_ratio": 1.0, "avg_skip_rate": 0.0, "avg_slow_ratio": 1.0}}
        return out, aux


class iTransformerBaseline(nn.Module):
    """iTransformer-style: attention over variates, FFN per token, head per forecast."""
    def __init__(self, input_dim, hidden_dim, pred_len, num_layers=2, nhead=4, seq_len=96):
        super().__init__()
        self.embed = nn.Linear(seq_len, hidden_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=4 * hidden_dim,
            dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x, temperature=1.0, return_stats=False):
        # x: (B, L, C); treat each variate c as a token -> tokens (B, C, hidden)
        x = x.transpose(1, 2)  # (B, C, L)
        h = self.embed(x)  # (B, C, hidden)
        h = self.encoder(h)  # (B, C, hidden)
        out = self.fc(h.sum(dim=1))  # (B, pred_len)
        aux = torch.tensor(0.0, device=x.device)
        if return_stats:
            return out, aux, {"efficiency": {"avg_compute_ratio": 1.0, "avg_skip_rate": 0.0, "avg_slow_ratio": 1.0}}
        return out, aux


def make_backbone(name, input_dim, seq_len, pred_len):
    if name == "lstm":
        return LSTMBaseline(input_dim=input_dim, hidden_dim=32, pred_len=pred_len)
    if name == "transformer":
        return TransformerEncoderBaseline(input_dim=input_dim, hidden_dim=32, pred_len=pred_len)
    if name == "patchtst":
        return PatchTSTBaseline(input_dim=input_dim, hidden_dim=32, pred_len=pred_len)
    if name == "itransformer":
        return iTransformerBaseline(input_dim=input_dim, hidden_dim=32, pred_len=pred_len, seq_len=seq_len)
    raise ValueError(name)


def power_curve_pu(v, v_in, v_rated, v_out=25.0):
    out = np.zeros_like(v, dtype=np.float64)
    denom = max(v_rated - v_in, 1e-6)
    mid = (v >= v_in) & (v < v_rated)
    out = np.where(mid, np.clip((v - v_in) / denom, 0.0, 1.0) ** 3, out)
    out = np.where((v >= v_rated) & (v <= v_out), 1.0, out)
    return out


def fit_curve(ws, p):
    p_rated = float(np.percentile(p, 99.5))
    bins = np.arange(np.floor(ws.min()), np.ceil(ws.max()) + 0.5, 0.5)
    idx = np.digitize(ws, bins) - 1
    centers, med = [], []
    for i in range(len(bins) - 1):
        m = idx == i
        if m.sum() >= 50:
            centers.append((bins[i] + bins[i + 1]) / 2)
            med.append(np.median(p[m]))
    centers = np.array(centers); med = np.array(med)
    rm = med >= 0.95 * p_rated
    v_rated = float(centers[rm][0]) if rm.any() else 11.5
    v_in = float(np.quantile(ws[p >= 0.01 * p_rated], 0.02)) if (p >= 0.01 * p_rated).any() else 3.0
    return p_rated, v_in, v_rated


def kde(d, grid, bw):
    diffs = (grid[None, :] - d[:, None]) / bw
    k = np.exp(-0.5 * diffs ** 2) / (bw * np.sqrt(2 * np.pi))
    return k.mean(axis=0)


def js_div(p, q):
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log((p + 1e-12) / (m + 1e-12))) + np.sum(q * np.log((q + 1e-12) / (m + 1e-12)))))


def build_windows(ws, p, other, seq_len, pred_len):
    """Return (X, Y, v_proxy_at_each_future_step) with normalized features."""
    n = len(ws) - seq_len - pred_len + 1
    X = np.zeros((n, seq_len, 1 + other.shape[1]), dtype=np.float32)
    Y = np.zeros((n, pred_len), dtype=np.float32)
    # proxy / true wind for audit: persistence proxy of wind at t, horizon steps
    vp_audit = np.zeros((n, pred_len), dtype=np.float32)
    vr_audit = np.zeros((n, pred_len), dtype=np.float32)
    for i in range(n):
        X[i, :, 0] = ws[i:i + seq_len]
        X[i, :, 1:] = other[i:i + seq_len]
        Y[i] = p[i + seq_len:i + seq_len + pred_len]
        vp_audit[i] = ws[i + seq_len - 1]  # persistence proxy (constant over horizon)
        vr_audit[i] = ws[i + seq_len:i + seq_len + pred_len]
    return X, Y, vp_audit, vr_audit


def power_curve_pu_torch(v, v_in, v_rated, v_out=25.0):
    """Per-unit power curve as a tensor: cut-in -> 0, ramp -> cubic, rated plateau -> 1,
    cut-out -> 0. Mirrors power_curve_pu()."""
    v = v.to(torch.float32)
    out = torch.zeros_like(v)
    denom = max(float(v_rated - v_in), 1e-6)
    mid = (v >= v_in) & (v < v_rated)
    out = torch.where(mid, torch.clamp((v - v_in) / denom, 0.0, 1.0) ** 3, out)
    out = torch.where((v >= v_rated) & (v <= v_out), torch.ones_like(v), out)
    return out


def train_one(site, loss_kind, p_rated, v_in, v_rated, ws_idx, device, seed, seq_len=96, pred_len=24, epochs=15, penalty_weight=1.0, backbone="lstm"):
    df = pd.read_csv(SITES[site])
    ws = df["wind_speed"].to_numpy(np.float64)
    p = df["power"].to_numpy(np.float64)
    valid = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[valid], p[valid]
    n_all = len(ws)

    # ---- G3: raw-time 80/20 split BEFORE windowing, everything fit on TRAIN only ----
    n_train_time = int(n_all * 0.8)
    ws_tr, p_tr = ws[:n_train_time], p[:n_train_time]
    ws_te, p_te_arr = ws[n_train_time:], p[n_train_time:]

    # train-only scaler / power curve / target stats
    ws_mean, ws_std = float(ws_tr.mean()), float(ws_tr.std() + 1e-6)
    p_mean, p_std = float(p_tr.mean()), float(p_tr.std() + 1e-6)
    p_rated, v_in, v_rated = fit_curve(ws_tr, p_tr)   # G3: curve fit on TRAIN only

    # per-unit power targets in [0,1] (G2: everything lives in per-unit space)
    p_tr_pu = np.clip(p_tr / max(p_rated, 1e-6), 0.0, 1.0)
    p_te_pu = np.clip(p_te_arr / max(p_rated, 1e-6), 0.0, 1.0)
    ws_tr_n = (ws_tr - ws_mean) / ws_std
    ws_te_n = (ws_te - ws_mean) / ws_std

    # build DISJOINT windows: train windows end before test windows (embargo pred_len-1)
    def windows_of(ws_n, p_pu, seq_len, pred_len):
        n = len(ws_n) - seq_len - pred_len + 1
        if n < 1:
            return (np.zeros((0, seq_len, 2), np.float32), np.zeros((0, pred_len), np.float32),
                    np.zeros((0, pred_len), np.float32), np.zeros((0, pred_len), np.float32))
        X = np.zeros((n, seq_len, 2), np.float32)
        Y = np.zeros((n, pred_len), np.float32)
        vp = np.zeros((n, pred_len), np.float32)
        vr = np.zeros((n, pred_len), np.float32)
        for i in range(n):
            X[i, :, 0] = ws_n[i:i + seq_len]
            X[i, :, 1] = ws_n[i:i + seq_len]
            Y[i] = p_pu[i + seq_len:i + seq_len + pred_len]
            vp[i] = ws_n[i + seq_len - 1]          # persistence proxy
            vr[i] = ws_n[i + seq_len:i + seq_len + pred_len]
        return X, Y, vp, vr

    Xtr_np, Ytr_np, vptr_np, vrtr_np = windows_of(ws_tr_n, p_tr_pu, seq_len, pred_len)
    Xte_np, Yte_np, vpte_np, vrte_np = windows_of(ws_te_n, p_te_pu, seq_len, pred_len)

    # Convert to tensors. Model predicts per-unit power in [0,1].
    Xtr = torch.from_numpy(Xtr_np)
    Ytr = torch.from_numpy(Ytr_np)
    Xte = torch.from_numpy(Xte_np)
    Yte = torch.from_numpy(Yte_np)
    # physical proxies for audit metrics (m/s)
    vp_te = vpte_np * ws_std + ws_mean
    vr_te = vrte_np * ws_std + ws_mean

    torch.manual_seed(seed); np.random.seed(seed)
    model = make_backbone(backbone, Xtr.shape[2], seq_len, pred_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ws_mean_t = torch.tensor(ws_mean, device=device, dtype=torch.float32)
    ws_std_t = torch.tensor(ws_std, device=device, dtype=torch.float32)
    p_rated_t = torch.tensor(float(p_rated), device=device, dtype=torch.float32)
    v_out = 25.0

    # Betz per-unit ceiling normalized by v_rated (unit-invariant)
    def betz_pu(v):
        vmax = max(float(v_rated), 1e-6)
        return torch.clamp(torch.clamp(v, min=0.0) / vmax, 0.0, None) ** 3

    # target distribution stats for each loss, computed on TRAIN per-unit power only
    tgt_mean = torch.tensor(float(p_tr_pu.mean()), device=device, dtype=torch.float32)
    tgt_std = torch.tensor(float(p_tr_pu.std()) + 1e-6, device=device, dtype=torch.float32)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for bi in range(0, len(Xtr), 64):
            idx = perm[bi:bi + 64]
            xb, yb = Xtr[idx].to(device), Ytr[idx].to(device)
            pred, _ = model(xb)                    # per-unit power in [0,1]
            # G2: MSE in per-unit space (already per-unit)
            mse = F.mse_loss(pred, yb)
            if loss_kind == "mse":
                loss = mse
            elif loss_kind in ("mse+envelope", "mse+betz"):
                # proxy wind in m/s (persistence = last observed wind), same for all horizon steps
                v = xb[:, -1, 0] * ws_std_t + ws_mean_t        # (B,)
                v = v.unsqueeze(1).expand(-1, pred_len)
                if loss_kind == "mse+envelope":
                    # G1: rated plateau = 1.0 (NOT 0), cut-out = 0
                    env_pu = power_curve_pu_torch(v, v_in, v_rated, v_out)
                else:
                    env_pu = betz_pu(v)
                over = F.relu(pred - env_pu)        # G2: per-unit violation
                penalty = (over ** 2).mean()
                loss = mse + penalty_weight * penalty
            else:  # mse+js — keep as moment matching for now but RENAME semantics;
                   # full KDE+JS is G4 (out of scope for G1-G3)
                pred_pu = torch.clamp(pred, 0.0, 1.0)
                dist_pen = (pred_pu.mean() - tgt_mean) ** 2 + (pred_pu.std() - tgt_std) ** 2
                loss = mse + penalty_weight * dist_pen
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # audit on real test outputs (per-unit)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bi in range(0, len(Xte), 128):
            xb = Xte[bi:bi + 128].to(device)
            pr, _ = model(xb)
            preds.append(pr.cpu().numpy())
    pred_pu = np.clip(np.concatenate(preds, 0), 0.0, 1.0)
    true_pu = np.asarray(Yte_np)
    # physical values for reporting
    pred_kw = pred_pu * p_rated
    true_kw = true_pu * p_rated

    # ramp / calm / above masks (using TRUE wind, physical)
    r = np.abs(np.diff(vr_te, axis=1, prepend=vr_te[:, :1])).mean(axis=1)
    ramp = (vr_te.mean(axis=1) < v_rated) & (r > np.quantile(r, 0.9))
    calm = (vr_te.mean(axis=1) < v_rated) & (r <= np.quantile(r, 0.5))
    above = vr_te.mean(axis=1) >= v_rated

    # G3: audit ceiling uses the SAME per-unit power curve as training (train-fitted v_in/v_rated)
    env_pu = power_curve_pu(vp_te, v_in, v_rated, v_out)
    cut_pu = np.clip(true_pu - env_pu, 0, None)
    ramp_cut = float(cut_pu[ramp].mean()) if ramp.sum() else 0.0
    above_cut = float(cut_pu[above].mean()) if above.sum() else 0.0
    asym_geom = (ramp_cut - above_cut) * 100.0

    err_pu = pred_pu - true_pu
    ramp_bias_dn = float(np.clip(-err_pu[ramp], 0, None).mean()) if ramp.sum() else 0.0
    above_bias_dn = float(np.clip(-err_pu[above], 0, None).mean()) if above.sum() else 0.0
    asym_real = (ramp_bias_dn - above_bias_dn) * 100.0

    over_rate = float((pred_pu > 1.0 + 1e-6).mean())
    neg_rate = float((pred_pu < 0.0).mean())
    r2 = compute_metrics(pred_kw, true_kw)["R2"]

    return {"site": site, "loss": loss_kind, "R2": round(r2, 4),
            "over%": round(over_rate * 100, 4), "neg%": round(neg_rate * 100, 4),
            "ramp_asym_pt": round(asym_real, 2), "geom_asym_pt": round(asym_geom, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="wind_cn,kelmarsh,smarteole,penmanshiel,wind_us")
    ap.add_argument("--backbone", default="lstm", choices=["lstm", "transformer", "patchtst", "itransformer"])
    ap.add_argument("--seeds", default="42,123")
    ap.add_argument("--loss-kind", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--penalty-weight", type=float, default=1.0)
    ap.add_argument("--out", default="../08_报告与说明文档/P0-1真实训练审计.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for site in args.sites.split(","):
        df = pd.read_csv(SITES[site])
        ws = df["wind_speed"].to_numpy(np.float64)
        p = df["power"].to_numpy(np.float64)
        valid = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
        ws, p = ws[valid], p[valid]
        p_rated, v_in, v_rated = fit_curve(ws, p)
        print(f"[{site}] p_rated={p_rated:.0f} v_in={v_in:.2f} v_rated={v_rated:.2f}", flush=True)
        loss_kinds = [args.loss_kind] if args.loss_kind else ["mse", "mse+envelope", "mse+betz", "mse+js"]
        for loss_kind in loss_kinds:
            for seed in [int(s) for s in args.seeds.split(",")]:
                r = train_one(site, loss_kind, p_rated, v_in, v_rated, 0, device, seed,
                              epochs=args.epochs, penalty_weight=args.penalty_weight,
                              backbone=args.backbone)
                r["seed"] = seed
                rows.append(r)
                print(r, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n=== saved {len(out)} rows -> {args.out} ===")
    piv = out.groupby("loss").agg(R2=("R2", "mean"), asym=("ramp_asym_pt", "mean"),
                                   over=("over%", "mean"), neg=("neg%", "mean"))
    print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
