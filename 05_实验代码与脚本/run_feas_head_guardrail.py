"""G2 feasibility-head guardrail: self-contained training + eval runner.

Wraps Ultra-LSNT with a FeasibilityHead that learns the per-sample 90th
quantile upper bound y_up. The guardrail uses min(P_rated, y_up) as the
upper envelope on ramp windows instead of the lagged wind proxy.

Training loss = MSE + lambda2 * guard_loss + lambda_q * pinball(y_up).
Eval computes R2/RMSE/over%/neg% on clean and severe slices.

Protocol mirrors run_physgate_cross_site (base checkpoint init, 5-epoch
guardrail fine-tune) so results are comparable with fgmoe/physgate/x_sym.
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

from ultra_lsnt_timeseries import TimeSeriesConfig, compute_metrics

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guardrail_backbones as _gb  # noqa: E402
import run_pi_moe_full_training_evidence as _rpf  # noqa: E402

if not hasattr(_rpf, "build_ultra_model"):
    _rpf.build_ultra_model = _gb.build_ultra_model
from run_pi_moe_full_training_evidence import (  # noqa: E402
    PhysicsConfig, estimate_curve_params, load_main_data, set_seed,
)
from guardrail_backbones import build_ultra_model, model_forward  # noqa: E402
from feas_head_guardrail import FeasHeadUltra, feas_head_guardrail_loss  # noqa: E402


def build_feas_model(ts_cfg, device, input_dim, cfg_path):
    base = build_ultra_model(ts_cfg, device, input_dim=input_dim, checkpoint_path=None, config_path=cfg_path)
    # build fresh wrapper reusing base encoder/blocks/decoder
    from ultra_lsnt_timeseries import LSNTConfig
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfgj = json.load(f)
    hidden = int(cfgj["hidden_dim"])
    model = FeasHeadUltra(base, input_dim, hidden, ts_cfg.seq_len, ts_cfg.pred_len).to(device)
    return model


def compute_ramp_mask(x, scaler, ws_idx, hard_slope=0.5):
    v_hist = x[:, :, ws_idx] * scaler.std[ws_idx] + scaler.mean[ws_idx]
    d = v_hist[:, 1:] - v_hist[:, :-1]
    slope = d[:, -4:].mean(dim=1).abs()
    return (slope >= hard_slope).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../04_数据集/prepared_priority_datasets/wind_cn_core_15min.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lambda2", type=float, default=0.03)
    ap.add_argument("--lambda-q", type=float, default=0.1)
    ap.add_argument("--q", type=float, default=0.9)
    ap.add_argument("--base-checkpoint", default=None)
    ap.add_argument("--backbone-config", default=None)
    ap.add_argument("--out-dir", default="results/g2_feashead")
    ap.add_argument("--summary-csv", default="results/g2_feashead/summary.csv")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    data, feature_names, ts_cfg, train_loader, val_loader, test_loader, scaler = load_main_data(
        args.data, "power", seq_len=96, pred_len=24, batch_size=128, num_workers=0
    )
    phy0 = estimate_curve_params(data, feature_names, target="power", train_ratio=0.7)
    phy = PhysicsConfig(phy0.p_rated_kw, phy0.v_in_mps, phy0.v_rated_mps, phy0.v_out_mps, lambda2=args.lambda2)

    cfg_path = args.backbone_config or "10_模型检查点与权重/checkpoints_ts/main/model_config.json"
    model = build_feas_model(ts_cfg, device, len(feature_names), cfg_path)
    # load base checkpoint if provided (into the wrapped submodules)
    if args.base_checkpoint and os.path.exists(args.base_checkpoint):
        state = torch.load(args.base_checkpoint, map_location=device)
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
            print(f"[load] base checkpoint {args.base_checkpoint} (strict=False)")

    ws_cols = ["wind_speed", "windspeed", "hub_wind_speed"]
    ws_idx = next(i for i, n in enumerate(feature_names) if n in ws_cols)

    target_mean = float(scaler.mean[-1])
    target_std = float(scaler.std[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.005)
    mse = nn.MSELoss()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 20260000)

    for ep in range(args.epochs):
        model.train()
        tr_loss = tr_mse = tr_guard = tr_pin = 0.0
        nb = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            pred, aux, y_up = model(bx, temperature=1.0)
            if pred.dim() == 3 and pred.size(-1) == 1:
                pred = pred.squeeze(-1)
            pred_kw = pred * target_std + target_mean
            y_up_kw = y_up * target_std + target_mean
            ramp_mask = compute_ramp_mask(bx, scaler, ws_idx, hard_slope=0.5)
            guard_l, pinball = feas_head_guardrail_loss(
                pred_kw, y_up_kw, phy.p_rated_kw, ramp_mask=ramp_mask, q=args.q
            )
            mse_l = mse(pred, by)
            loss = mse_l + args.lambda2 * guard_l + args.lambda_q * pinball + 0.005 * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item()); tr_mse += float(mse_l.item())
            tr_guard += float(guard_l.item()); tr_pin += float(pinball.item())
            nb += 1
        print(f"[epoch {ep+1}] loss={tr_loss/max(nb,1):.4f} mse={tr_mse/max(nb,1):.4f} "
              f"guard={tr_guard/max(nb,1):.4f} pinball={tr_pin/max(nb,1):.4f}", flush=True)

    # eval
    model.eval()
    for scen, sigma_cfg in [("clean", 0.0), ("severe", 0.4)]:
        preds, trues = [], []
        n_over = n_neg = n_all = 0
        with torch.no_grad():
            for bi, (bx, by) in enumerate(test_loader):
                bx, by = bx.to(device), by.to(device)
                if sigma_cfg > 0:
                    from run_pi_moe_full_training_evidence import inject_gaussian_protocol
                    bx = inject_gaussian_protocol(bx, sigma_cfg=sigma_cfg, alpha=1.5)
                pred, _, _ = model(bx, temperature=0.1)
                if pred.dim() == 3 and pred.size(-1) == 1:
                    pred = pred.squeeze(-1)
                pred_kw = pred * target_std + target_mean
                true_kw = by * target_std + target_mean
                n_over += int((pred_kw > phy.p_rated_kw).sum().item())
                n_neg += int((pred_kw < 0).sum().item())
                n_all += int(pred_kw.numel())
                preds.append(pred_kw.cpu().numpy()); trues.append(true_kw.cpu().numpy())
        pred_np = np.concatenate(preds, 0); true_np = np.concatenate(trues, 0)
        m = compute_metrics(pred_np, true_np)
        row = {
            "site": "wind_cn", "guardrail": "g2_feashead", "seed": args.seed,
            "scenario": scen, "R2": m["R2"], "RMSE": m["RMSE"], "MAE": m["MAE"],
            "over_rated_rate_pct": 100.0 * n_over / max(n_all, 1),
            "neg_rate_pct": 100.0 * n_neg / max(n_all, 1),
            "lambda2": args.lambda2, "q": args.q, "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        dfr = pd.DataFrame([row])
        dfr.to_csv(args.summary_csv, mode="a", header=not os.path.exists(args.summary_csv), index=False)
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
