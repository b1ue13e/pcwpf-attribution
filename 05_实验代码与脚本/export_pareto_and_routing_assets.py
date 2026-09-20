#!/usr/bin/env python3
"""
Export two assets for paper figures:
1) Lite accuracy rows for Pareto plot (Latency vs RMSE).
2) Per-timestep routing weights/top-k for Ultra-LSNT heatmap plotting.

Outputs:
  results/supplementary_evidence/lite_accuracy_for_pareto.csv
  results/supplementary_evidence/routing_timestep_matrix_ultra_block1.csv
  results/supplementary_evidence/routing_timestep_topk_ultra_block1.csv
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ultra_lsnt_lite import UltraLSNTLiteForecaster
from ultra_lsnt_timeseries import (
    LSNTConfig,
    TimeSeriesConfig,
    TrainConfig,
    UltraLSNTForecaster,
    compute_metrics,
    create_dataloaders,
    load_csv_data,
)


OUT_DIR = Path("results/supplementary_evidence")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders(batch_size: int = 256):
    data, _ = load_csv_data("wind_final.csv", "power")
    ts_cfg = TimeSeriesConfig(seq_len=96, pred_len=24, target="power")
    tr_cfg = TrainConfig(batch_size=batch_size, epochs=3, num_workers=0)
    train_loader, val_loader, test_loader, scaler = create_dataloaders(data, ts_cfg, tr_cfg)
    return data, ts_cfg, tr_cfg, train_loader, val_loader, test_loader, scaler


def inverse_scale(arr: np.ndarray, scaler) -> np.ndarray:
    tgt_mean = float(scaler.mean[-1])
    tgt_std = float(scaler.std[-1])
    return arr * tgt_std + tgt_mean


def train_eval_lite(
    model_name: str,
    model_cfg: Dict,
    train_loader,
    test_loader,
    scaler,
    device: torch.device,
    epochs: int = 3,
) -> Dict:
    model = UltraLSNTLiteForecaster(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, aux_loss = model(bx)
            loss = criterion(pred, by) + 0.01 * aux_loss
            loss.backward()
            optimizer.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            pred, _ = model(bx)
            preds.append(pred.cpu().numpy())
            trues.append(by.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    preds_inv = inverse_scale(preds, scaler)
    trues_inv = inverse_scale(trues, scaler)
    m = compute_metrics(preds_inv, trues_inv)

    total_params = sum(p.numel() for p in model.parameters())
    return {
        "model": model_name,
        "R2": float(m["R2"]),
        "RMSE": float(m["RMSE"]),
        "MAE": float(m["MAE"]),
        "MAPE": float(m["MAPE"]),
        "MSE": float(m["MSE"]),
        "total_parameters": int(total_params),
        "seq_len": int(model_cfg["seq_len"]),
        "pred_len": int(model_cfg["pred_len"]),
        "hidden_dim": int(model_cfg["hidden_dim"]),
        "num_blocks": int(model_cfg["num_blocks"]),
        "num_experts": int(model_cfg["num_experts"]),
        "top_k": int(model_cfg["top_k"]),
        "epochs": int(epochs),
        "inverse_scaling": "train-fitted z-score (target column)",
        "dataset": "wind_final.csv (test split from create_dataloaders)",
    }


def load_ultra_main_for_routing(ts_cfg: TimeSeriesConfig, device: torch.device) -> UltraLSNTForecaster:
    cfg_path = Path("checkpoints_ts/main/model_config.json")
    ckpt_path = Path("checkpoints_ts/main/best_model.pth")

    cfgj = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = LSNTConfig(
        input_dim=cfgj["input_dim"],
        hidden_dim=cfgj["hidden_dim"],
        output_dim=1,
        num_blocks=cfgj["num_blocks"],
        num_experts=cfgj["num_experts"],
        top_k=cfgj["top_k"],
        dropout=cfgj.get("dropout", 0.1),
        router_z_loss_coef=cfgj.get("router_z_loss_coef", 0.01),
        router_aux_loss_coef=cfgj.get("router_aux_loss_coef", 0.01),
        router_jitter_noise=cfgj.get("router_jitter_noise", 0.1),
        skip_threshold=cfgj.get("skip_threshold", 0.5),
        dual_mode_expansion=cfgj.get("dual_mode_expansion", 4),
        probabilistic_mode="none",
        heteroscedastic_moe=False,
    )
    model = UltraLSNTForecaster(cfg, ts_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model


def export_routing_per_timestep(
    model: UltraLSNTForecaster,
    test_loader,
    scaler,
    device: torch.device,
    block_index: int = 0,
) -> None:
    model.eval()
    probs_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []
    true_all: List[np.ndarray] = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            pred, _, stats = model(bx, temperature=0.1, return_stats=True)
            block_stats = stats["block_stats"][block_index]
            raw_probs = block_stats["router"]["raw_probs"].cpu().numpy()  # [B, E]

            probs_all.append(raw_probs)
            pred_all.append(pred.cpu().numpy())
            true_all.append(by.numpy())

    probs = np.concatenate(probs_all, axis=0)
    preds = np.concatenate(pred_all, axis=0)
    trues = np.concatenate(true_all, axis=0)

    preds_inv = inverse_scale(preds, scaler)
    trues_inv = inverse_scale(trues, scaler)

    n, m = probs.shape
    cols = [f"p_e{i+1}" for i in range(m)]
    mat = pd.DataFrame(probs, columns=cols)
    mat.insert(0, "t", np.arange(n, dtype=int))
    mat["y_true_h1_kw"] = trues_inv[:, 0]
    mat["y_pred_h1_kw"] = preds_inv[:, 0]
    mat["abs_err_h1_kw"] = np.abs(mat["y_pred_h1_kw"] - mat["y_true_h1_kw"])
    mat_path = OUT_DIR / "routing_timestep_matrix_ultra_block1.csv"
    mat.to_csv(mat_path, index=False)

    # Top-k export (k=2 for main model).
    top_order = np.argsort(-probs, axis=1)
    top1 = top_order[:, 0]
    top2 = top_order[:, 1] if m > 1 else top_order[:, 0]
    top_df = pd.DataFrame(
        {
            "t": np.arange(n, dtype=int),
            "expert_id_top1": top1 + 1,
            "prob_top1": probs[np.arange(n), top1],
            "expert_id_top2": top2 + 1,
            "prob_top2": probs[np.arange(n), top2],
            "y_true_h1_kw": trues_inv[:, 0],
            "y_pred_h1_kw": preds_inv[:, 0],
            "abs_err_h1_kw": np.abs(preds_inv[:, 0] - trues_inv[:, 0]),
        }
    )
    top_path = OUT_DIR / "routing_timestep_topk_ultra_block1.csv"
    top_df.to_csv(top_path, index=False)

    print(f"Saved routing matrix: {mat_path}")
    print(f"Saved routing top-k: {top_path}")
    print(f"Routing rows: {n}, experts: {m}")


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    _, ts_cfg, _, train_loader, _, test_loader, scaler = build_loaders(batch_size=256)

    # 1) Lite accuracy rows for Pareto.
    lite_cfg = {
        "input_dim": 8,
        "hidden_dim": 128,
        "seq_len": 96,
        "pred_len": 24,
        "num_blocks": 2,
        "num_experts": 4,
        "top_k": 2,
        "scales": [1, 2, 4],
    }
    lite_opt_cfg = {
        "input_dim": 8,
        "hidden_dim": 96,
        "seq_len": 96,
        "pred_len": 24,
        "num_blocks": 1,
        "num_experts": 2,
        "top_k": 1,
        "scales": [1, 2],
    }

    print("Training/evaluating Ultra-LSNT-Lite ...")
    row_lite = train_eval_lite("Ultra-LSNT-Lite", lite_cfg, train_loader, test_loader, scaler, device, epochs=3)
    print("Training/evaluating Ultra-LSNT-Lite-Optimized ...")
    row_lite_opt = train_eval_lite(
        "Ultra-LSNT-Lite-Optimized", lite_opt_cfg, train_loader, test_loader, scaler, device, epochs=3
    )

    # Full Ultra row from archived full-test metrics (same inverse-scaling protocol).
    ultra_metrics_path = OUT_DIR / "ultra_metrics_main_fulltest.json"
    ultra_row = {}
    if ultra_metrics_path.exists():
        payload = json.loads(ultra_metrics_path.read_text(encoding="utf-8"))
        mt = payload["metrics"]
        ultra_row = {
            "model": "Ultra-LSNT",
            "R2": float(mt["R2"]),
            "RMSE": float(mt["RMSE"]),
            "MAE": float(mt["MAE"]),
            "MAPE": float(mt["MAPE"]),
            "MSE": float(mt["MSE"]),
            "total_parameters": np.nan,
            "seq_len": 96,
            "pred_len": 24,
            "hidden_dim": 128,
            "num_blocks": 3,
            "num_experts": 4,
            "top_k": 2,
            "epochs": np.nan,
            "inverse_scaling": "train-fitted z-score (target column)",
            "dataset": "wind_final.csv (test split from create_dataloaders)",
        }

    rows = [row_lite, row_lite_opt]
    if ultra_row:
        rows.insert(0, ultra_row)
    lite_df = pd.DataFrame(rows)
    lite_out = OUT_DIR / "lite_accuracy_for_pareto.csv"
    lite_df.to_csv(lite_out, index=False)
    print(f"Saved lite accuracy table: {lite_out}")
    print(lite_df[["model", "R2", "RMSE", "MAE"]].to_string(index=False))

    # 2) Per-timestep routing exports.
    print("Exporting per-timestep routing probabilities ...")
    ultra_model = load_ultra_main_for_routing(ts_cfg, device)
    export_routing_per_timestep(ultra_model, test_loader, scaler, device, block_index=0)


if __name__ == "__main__":
    main()
