import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from ultra_lsnt_timeseries import (
    TimeSeriesConfig,
    TrainConfig,
    compute_metrics,
    create_dataloaders,
    load_csv_data,
)


OUT_DIR = os.path.join("results", "supplementary_evidence")
GAUSS_CSV = os.path.join("results", "robust_gaussian_windcn.csv")
TWO_POINT_CSV = os.path.join("results", "supplementary_evidence", "robust_two_point_windcn.csv")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_persistence(
    test_loader: torch.utils.data.DataLoader,
    scaler,
    sigma_cfg: float,
    alpha: float = 1.5,
) -> dict:
    preds_all = []
    trues_all = []

    for bx, by in test_loader:
        if sigma_cfg > 0:
            # Match the paper's robustness protocol: per-sample std in normalized space.
            std = bx.std(dim=1, keepdim=True)
            noise = torch.randn_like(bx) * (alpha * sigma_cfg) * std
            bx = bx + noise

        # Persistence baseline: repeat the last observed target across the full horizon.
        last_target = bx[:, -1, -1].unsqueeze(1)
        pred = last_target.repeat(1, by.shape[1])

        preds_all.append(pred.numpy())
        trues_all.append(by.numpy())

    preds = np.concatenate(preds_all, axis=0)
    trues = np.concatenate(trues_all, axis=0)

    target_mean = scaler.mean[-1]
    target_std = scaler.std[-1]
    preds = preds * target_std + target_mean
    trues = trues * target_std + target_mean

    metrics = compute_metrics(preds, trues)
    return {
        "R2": float(metrics["R2"]),
        "RMSE": float(metrics["RMSE"]),
        "MAE": float(metrics["MAE"]),
    }


def merge_csv(base_path: str, new_rows: pd.DataFrame, key_cols: list[str], sort_cols: list[str]) -> None:
    if os.path.exists(base_path):
        base_df = pd.read_csv(base_path)
        merged = pd.concat([base_df, new_rows], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_cols, keep="last")
    else:
        merged = new_rows.copy()
    merged = merged.sort_values(sort_cols).reset_index(drop=True)
    merged.to_csv(base_path, index=False)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(42)

    data, _ = load_csv_data("wind_final.csv", "power")
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target="power")
    train_config = TrainConfig(batch_size=128, epochs=1, num_workers=0)
    _, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)

    sigma_cfg_grid = [0.0, 0.1, 0.2, 0.3, 0.4]
    rows = []
    for sigma_cfg in sigma_cfg_grid:
        m = evaluate_persistence(test_loader, scaler, sigma_cfg=sigma_cfg, alpha=1.5)
        rows.append(
            {
                "sigma": sigma_cfg,
                "model": "Persistence",
                "R2": m["R2"],
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
            }
        )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "persistence_gaussian_windcn.csv")
    df.to_csv(out_csv, index=False)

    clean = float(df.loc[df["sigma"] == 0.0, "R2"].iloc[0])
    noisy_02 = float(df.loc[df["sigma"] == 0.2, "R2"].iloc[0])
    noisy_04 = float(df.loc[df["sigma"] == 0.4, "R2"].iloc[0])
    two_point = pd.DataFrame(
        [
            {
                "model": "Persistence",
                "r2_sigma_0_0": clean,
                "r2_sigma_0_2": noisy_02,
                "abs_drop_0_0_to_0_2": clean - noisy_02,
                "rel_drop_pct_0_0_to_0_2": 100.0 * (clean - noisy_02) / max(clean, 1e-12),
                "r2_sigma_0_4": noisy_04,
            }
        ]
    )
    two_point_out = os.path.join(OUT_DIR, "persistence_two_point_windcn.csv")
    two_point.to_csv(two_point_out, index=False)

    # Also merge into the existing paper-facing CSVs.
    merge_csv(
        GAUSS_CSV,
        df[["sigma", "model", "R2"]],
        key_cols=["sigma", "model"],
        sort_cols=["model", "sigma"],
    )
    merge_csv(
        TWO_POINT_CSV,
        two_point,
        key_cols=["model"],
        sort_cols=["model"],
    )

    meta = {
        "script": "run_persistence_windcn.py",
        "dataset": "wind_final.csv",
        "target": "power",
        "split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
        "seq_len": 96,
        "pred_len": 24,
        "inverse_scaling": True,
        "noise": {
            "type": "gaussian",
            "sigma_cfg_grid": sigma_cfg_grid,
            "sigma_eff_formula": "sigma_eff = alpha * sigma_cfg",
            "alpha": 1.5,
            "injection": "randn * (alpha*sigma_cfg) * std(x) in normalized space",
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    meta_out = os.path.join(OUT_DIR, "persistence_gaussian_windcn.meta.json")
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(df.to_string(index=False))
    print(f"saved: {out_csv}")
    print(f"saved: {two_point_out}")
    print(f"merged: {GAUSS_CSV}")
    print(f"merged: {TWO_POINT_CSV}")
    print(f"saved: {meta_out}")


if __name__ == "__main__":
    main()
