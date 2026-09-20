"""Imputation vs guardrail: input-imputation arm (final Direction-1 evidence).

This runner ONLY produces the imputation arm:
  clean-trained carrier -> evaluate on raw and ffill(p) missingness inputs.
  metrics: R2, RMSE, over_rated_rate_pct, neg_rate_pct.

The guardrail arm (physgate/fgmoe on raw input) is ALREADY available from
physgate_clean_summary.csv; the two arms are merged offline.

Aligns Table 9 protocol: ffill = direct missing rate p (0.1 / 0.3).
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

from ultra_lsnt_timeseries import (
    TimeSeriesConfig, TrainConfig, compute_metrics, create_dataloaders, load_csv_data,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guardrail_backbones as _gb  # noqa: E402
import run_pi_moe_full_training_evidence as _rpf  # noqa: E402

if not hasattr(_rpf, "build_ultra_model"):
    _rpf.build_ultra_model = _gb.build_ultra_model
from run_fairness_closure_fullfault_4090 import build_model, forward_model, set_seed, pick_device  # noqa: E402
from run_pi_moe_full_training_evidence import PhysicsConfig, estimate_curve_params  # noqa: E402


def mask_and_ffill(x: torch.Tensor, p: float, generator: torch.Generator) -> torch.Tensor:
    out = x.clone()
    mask = torch.rand(x.shape, generator=generator, device="cpu") < float(p)
    mask = mask.to(x.device)
    out[mask] = float("nan")
    for t in range(1, out.size(1)):
        missing = torch.isnan(out[:, t, :])
        out[:, t, :] = torch.where(missing, out[:, t - 1, :], out[:, t, :])
    out[torch.isnan(out)] = 0.0
    return out


def evaluate(model_name, model, test_loader, scaler, device, *, p, seed, phy):
    model.eval()
    t_mean = float(scaler.mean[-1])
    t_std = float(scaler.std[-1])
    over_n = neg_n = all_n = 0
    preds, trues = [], []
    with torch.no_grad():
        for bi, (bx, by) in enumerate(test_loader):
            bx, by = bx.to(device), by.to(device)
            if p > 0:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(seed * 1000000 + bi)
                bx = mask_and_ffill(bx, p, gen)
            pred = forward_model(model_name, model, bx)
            pred_kw = pred * t_std + t_mean
            true_kw = by * t_std + t_mean
            over_n += int((pred_kw > phy.p_rated_kw).sum().item())
            neg_n += int((pred_kw < 0).sum().item())
            all_n += int(pred_kw.numel())
            preds.append(pred_kw.cpu().numpy())
            trues.append(true_kw.cpu().numpy())
    m = compute_metrics(np.concatenate(preds, 0), np.concatenate(trues, 0))
    m["over_rated_rate_pct"] = 100.0 * over_n / max(all_n, 1)
    m["neg_rate_pct"] = 100.0 * neg_n / max(all_n, 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["wind_cn", "wind_us", "smarteole"])
    ap.add_argument("--carrier", required=True, choices=["dlinear", "ultra"])
    ap.add_argument("--data", default=None, help="override data path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/impute_vs_guardrail")
    ap.add_argument("--summary-csv", default="results/impute_vs_guardrail/summary.csv")
    ap.add_argument("--backbone-config", default=None)
    ap.add_argument("--base-checkpoint", default=None)
    ap.add_argument("--train-local", action="store_true", help="Train carrier from scratch on the local site (default: load checkpoint)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device("cuda")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    sites = {
        "wind_cn": "04_数据集/wind_final.csv",
        "wind_us": "04_数据集/prepared_priority_datasets/wind_us_core_15min.csv",
        "smarteole": "04_数据集/prepared_priority_datasets/smarteole_france_2020.csv",
    }
    data_path = args.data or sites[args.site]

    data, feature_names, ts_cfg, train_loader, val_loader, test_loader, scaler = (
        _rpf.load_main_data(data_path, "power", seq_len=96, pred_len=24, batch_size=128, num_workers=0)
    )
    phy0 = estimate_curve_params(data, feature_names, target="power", train_ratio=0.7)
    phy = PhysicsConfig(phy0.p_rated_kw, phy0.v_in_mps, phy0.v_rated_mps, phy0.v_out_mps, lambda2=0.03)

    cfg = args.backbone_config or "10_模型检查点与权重/checkpoints_ts/main/model_config.json"
    ckpt = args.base_checkpoint

    if args.carrier == "dlinear" and args.train_local:
        # train DLinear from scratch on the LOCAL site (avoids cross-site checkpoint mismatch)
        from run_dlinear import DLinear
        model = DLinear(ts_cfg.seq_len, ts_cfg.pred_len, len(feature_names)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        mse = nn.MSELoss()
        model.train()
        for ep in range(10):
            ep_loss = 0.0
            n = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad(set_to_none=True)
                out = model(bx)
                loss = mse(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.item())
                n += 1
            print(f"[train-local dlinear] ep {ep+1}/10 loss {ep_loss/max(n,1):.6f}", flush=True)
        model.eval()
    else:
        model = build_model(args.carrier, len(feature_names), ts_cfg, device, base_config=cfg, base_checkpoint=ckpt)

    # clean-trained: no retraining needed, model already from checkpoint.
    for p, scen in [(0.0, "raw"), (0.1, "ffill_p0.1"), (0.3, "ffill_p0.3")]:
        m = evaluate(args.carrier, model, test_loader, scaler, device, p=p, seed=args.seed, phy=phy)
        row = {
            "site": args.site, "carrier": args.carrier, "train_mode": "clean",
            "eval_input": scen, "guardrail": "none", "seed": args.seed,
            "R2": m["R2"], "RMSE": m["RMSE"], "MAE": m["MAE"],
            "over_rated_rate_pct": m["over_rated_rate_pct"],
            "neg_rate_pct": m["neg_rate_pct"],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        dfr = pd.DataFrame([row])
        dfr.to_csv(args.summary_csv, mode="a", header=not os.path.exists(args.summary_csv), index=False)
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
