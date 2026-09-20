#!/usr/bin/env python3
"""
Minimal matched robustness controls v2:
- Models: DLinear, iTransformer, TimeMixer
- Seeds: multiple seeds (default: 42,2025)
- Training modes: clean_baseline vs drift_augmented
- Perturbation: drift only (this revision)
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_dlinear import DLinear
from ultra_lsnt_timeseries import (
    TimeSeriesConfig,
    TrainConfig,
    compute_metrics,
    create_dataloaders,
    load_csv_data,
)


OUT_DIR = Path("results") / "supplementary_evidence"
OUT_CSV = OUT_DIR / "fairness_minimal_aug_controls_drift_2seed.csv"
OUT_SUMMARY_CSV = OUT_DIR / "fairness_minimal_aug_controls_drift_2seed.summary.csv"
OUT_META = OUT_DIR / "fairness_minimal_aug_controls_drift_2seed.meta.json"


class TimeMixer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        self.input_dim = config["input_dim"]
        self.d_model = config.get("d_model", 128)
        self.scales = config.get("scales", [1, 2, 4])

        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.multiscale_convs = nn.ModuleList()
        for scale in self.scales:
            kernel_size = max(3, scale * 2)
            padding = kernel_size // 2
            conv = nn.Sequential(
                nn.Conv1d(self.d_model, self.d_model, kernel_size=kernel_size, padding=padding),
                nn.GELU(),
                nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            )
            self.multiscale_convs.append(conv)

        self.scale_fusion = nn.Sequential(
            nn.Linear(self.d_model * len(self.scales), self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.frequency_decomp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.d_model * 2),
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=config.get("nhead", 4),
            dropout=config.get("dropout", 0.1),
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.pred_len),
        )
        self.skip = nn.Linear(self.seq_len, self.pred_len) if self.seq_len != self.pred_len else None

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.input_proj(x_enc)
        x_t = x.transpose(1, 2)

        multiscale_features = []
        for conv in self.multiscale_convs:
            scale_feat = conv(x_t)
            if scale_feat.size(-1) != self.seq_len:
                diff = scale_feat.size(-1) - self.seq_len
                if diff > 0:
                    start = diff // 2
                    scale_feat = scale_feat[:, :, start : start + self.seq_len]
                else:
                    padding = -diff
                    scale_feat = F.pad(scale_feat, (padding // 2, padding - padding // 2))
            multiscale_features.append(scale_feat.transpose(1, 2))

        fused = torch.cat(multiscale_features, dim=-1)
        fused = self.scale_fusion(fused)
        freq_components = self.frequency_decomp(fused)
        low_freq, high_freq = torch.chunk(freq_components, 2, dim=-1)
        trend = low_freq.mean(dim=1, keepdim=True)
        seasonal = high_freq
        attn_out, _ = self.temporal_attn(seasonal, seasonal, seasonal)
        seasonal = self.attn_norm(seasonal + attn_out)
        combined = seasonal + trend.expand_as(seasonal)
        aggregated = combined.mean(dim=1)
        output = self.output_proj(aggregated)

        if self.skip is not None:
            skip_out = self.skip(x_enc[:, :, -1])
            output = output + skip_out * 0.1
        return output.unsqueeze(-1), torch.tensor(0.0, device=x_enc.device)


class iTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        self.input_dim = config["input_dim"]
        self.d_model = config.get("d_model", 128)

        self.var_embedding = nn.Linear(self.seq_len, self.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.input_dim, self.d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.get("nhead", 4),
            dim_feedforward=self.d_model * 4,
            dropout=config.get("dropout", 0.1),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.get("num_layers", 2))
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model * self.input_dim, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.pred_len),
        )
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x_inverted = x_enc.transpose(1, 2)
        var_emb = self.var_embedding(x_inverted)
        var_emb = self.norm(var_emb + self.pos_embedding)
        encoded = self.encoder(var_emb)
        output = self.decoder(encoded.reshape(encoded.size(0), -1))
        return output.unsqueeze(-1), torch.tensor(0.0, device=x_enc.device)


@dataclass
class RunConfig:
    models: str = "dlinear,itransformer,timemixer"
    seeds: str = "42,2025"
    finetune_epochs: int = 1
    train_perturb: str = "drift"
    train_severity_max: float = 0.4
    test_perturb: str = "drift"
    test_grid: str = "0.0,0.4"
    alpha: float = 1.5
    batch_size: int = 128
    lr: float = 1e-4
    device: str = "auto"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_models(raw: str) -> List[str]:
    allowed = {"dlinear", "itransformer", "timemixer"}
    models = [m.strip().lower() for m in raw.split(",") if m.strip()]
    if not models:
        raise ValueError("No model provided in --models")
    unknown = [m for m in models if m not in allowed]
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    return models


def parse_seeds(raw: str) -> List[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("No valid seeds parsed from --seeds")
    return out


def parse_test_grid(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No value provided in --test-grid")
    return vals


def pick_device(flag: str) -> torch.device:
    if flag == "cuda":
        return torch.device("cuda")
    if flag == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inject_drift_protocol(x: torch.Tensor, severity_cfg: float, alpha: float = 1.5) -> torch.Tensor:
    """
    x'_{t,d} = x_{t,d} + (t/(L-1)) * (alpha * severity_cfg * std_d)
    std_d is per-sample, per-channel std over the sequence axis.
    """
    if severity_cfg <= 0.0:
        return x
    bsz, seq_len, _ = x.shape
    if seq_len <= 1:
        return x
    std_d = x.std(dim=1, keepdim=True)  # [B,1,D]
    ramp = torch.linspace(0.0, 1.0, steps=seq_len, device=x.device, dtype=x.dtype).view(1, seq_len, 1)
    drift = ramp * (alpha * severity_cfg) * std_d
    return x + drift


def apply_perturb(x: torch.Tensor, perturb: str, severity_cfg: float, alpha: float = 1.5) -> torch.Tensor:
    if perturb == "none":
        return x
    if perturb == "drift":
        return inject_drift_protocol(x, severity_cfg=severity_cfg, alpha=alpha)
    raise ValueError(f"Unsupported perturbation: {perturb}")


def build_loaders(batch_size: int):
    data, _ = load_csv_data("wind_final.csv", "power")
    if data is None:
        raise RuntimeError("Failed to load wind_final.csv")
    ts_cfg = TimeSeriesConfig(seq_len=96, pred_len=24, target="power")
    tr_cfg = TrainConfig(batch_size=batch_size, epochs=1, num_workers=0)
    train_loader, val_loader, test_loader, scaler = create_dataloaders(data, ts_cfg, tr_cfg)
    return data, ts_cfg, train_loader, val_loader, test_loader, scaler


def _load_state(path: Path, device: torch.device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def build_model(model_name: str, input_dim: int, device: torch.device) -> nn.Module:
    if model_name == "dlinear":
        ckpt = torch.load("dlinear_weights/dlinear_trained.pth", map_location=device)
        cfg = ckpt["config"]
        model = DLinear(cfg["seq_len"], cfg["pred_len"], cfg["input_dim"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return model

    if model_name == "itransformer":
        model = iTransformer(
            {
                "seq_len": 96,
                "pred_len": 24,
                "input_dim": input_dim,
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.1,
            }
        ).to(device)
        model.load_state_dict(_load_state(Path("checkpoints_ts/iTransformer_best.pth"), device), strict=False)
        return model

    if model_name == "timemixer":
        model = TimeMixer(
            {
                "seq_len": 96,
                "pred_len": 24,
                "input_dim": input_dim,
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.1,
                "scales": [1, 2, 4],
            }
        ).to(device)
        model.load_state_dict(_load_state(Path("checkpoints_ts/TimeMixer_best.pth"), device), strict=False)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def forward_model(model_name: str, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if model_name == "dlinear":
        return model(x)
    pred, _ = model(x)
    return pred.squeeze(-1)


def evaluate_model(
    model_name: str,
    model: nn.Module,
    test_loader,
    scaler,
    device: torch.device,
    test_perturb: str,
    severity_cfg: float,
    alpha: float,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            by = by.to(device)
            bx_mod = apply_perturb(bx, perturb=test_perturb, severity_cfg=severity_cfg, alpha=alpha)
            pred = forward_model(model_name, model, bx_mod)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(by.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    target_mean = float(scaler.mean[-1])
    target_std = float(scaler.std[-1])
    preds = preds * target_std + target_mean
    trues = trues * target_std + target_mean
    m = compute_metrics(preds, trues)
    return {"R2": float(m["R2"]), "RMSE": float(m["RMSE"]), "MAE": float(m["MAE"])}


def finetune_with_perturb(
    model_name: str,
    model: nn.Module,
    train_loader,
    device: torch.device,
    train_perturb: str,
    train_severity_max: float,
    epochs: int,
    lr: float,
    alpha: float,
) -> nn.Module:
    aug_model = copy.deepcopy(model).to(device)
    aug_model.train()

    optimizer = torch.optim.AdamW(aug_model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            severity_cfg = float(torch.rand(1).item()) * train_severity_max
            bx_mod = apply_perturb(bx, perturb=train_perturb, severity_cfg=severity_cfg, alpha=alpha)

            optimizer.zero_grad()
            pred = forward_model(model_name, aug_model, bx_mod)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aug_model.parameters(), max_norm=1.0)
            optimizer.step()

    aug_model.eval()
    return aug_model


def display_name(model_name: str) -> str:
    return {
        "dlinear": "DLinear",
        "itransformer": "iTransformer",
        "timemixer": "TimeMixer",
    }[model_name]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run minimal matched robustness controls v2 (drift, two-seed).")
    parser.add_argument("--models", type=str, default="dlinear,itransformer,timemixer")
    parser.add_argument("--seeds", type=str, default="42,2025")
    parser.add_argument("--finetune-epochs", type=int, default=1)
    parser.add_argument("--train-perturb", type=str, default="drift", choices=["drift"])
    parser.add_argument("--train-severity-max", type=float, default=0.4)
    parser.add_argument("--test-perturb", type=str, default="drift", choices=["drift"])
    parser.add_argument("--test-grid", type=str, default="0.0,0.4")
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    cfg = RunConfig(
        models=args.models,
        seeds=args.seeds,
        finetune_epochs=args.finetune_epochs,
        train_perturb=args.train_perturb,
        train_severity_max=args.train_severity_max,
        test_perturb=args.test_perturb,
        test_grid=args.test_grid,
        alpha=args.alpha,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    models = parse_models(cfg.models)
    seeds = parse_seeds(cfg.seeds)
    test_grid = parse_test_grid(cfg.test_grid)
    device = pick_device(cfg.device)

    print(f"[info] device={device}")
    print(f"[info] models={models}")
    print(f"[info] seeds={seeds}")
    print(f"[info] test_grid={test_grid}")
    print(f"[info] perturb(train/test)={cfg.train_perturb}/{cfg.test_perturb}")

    data, ts_cfg, train_loader, _, test_loader, scaler = build_loaders(cfg.batch_size)
    input_dim = data.shape[1]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_minimal_fairness_drift_2seed"

    rows: List[Dict] = []
    for seed in seeds:
        print(f"\n[seed] {seed}")
        set_seed(seed)

        for model_name in models:
            disp = display_name(model_name)
            print(f"  [model] {disp}")
            base_model = build_model(model_name, input_dim=input_dim, device=device)

            for sev_cfg in test_grid:
                met = evaluate_model(
                    model_name=model_name,
                    model=base_model,
                    test_loader=test_loader,
                    scaler=scaler,
                    device=device,
                    test_perturb=cfg.test_perturb,
                    severity_cfg=sev_cfg,
                    alpha=cfg.alpha,
                )
                rows.append(
                    {
                        "model": disp,
                        "training_mode": "clean_baseline",
                        "train_perturb": "none",
                        "test_perturb": cfg.test_perturb,
                        "seed": int(seed),
                        "severity_cfg": float(sev_cfg),
                        "severity_eff": float(cfg.alpha * sev_cfg),
                        "R2": met["R2"],
                        "RMSE": met["RMSE"],
                        "MAE": met["MAE"],
                        "run_id": run_id,
                    }
                )
                print(
                    f"    clean_baseline severity_cfg={sev_cfg:.2f} "
                    f"R2={met['R2']:.4f} RMSE={met['RMSE']:.2f} MAE={met['MAE']:.2f}"
                )

            aug_model = finetune_with_perturb(
                model_name=model_name,
                model=base_model,
                train_loader=train_loader,
                device=device,
                train_perturb=cfg.train_perturb,
                train_severity_max=cfg.train_severity_max,
                epochs=cfg.finetune_epochs,
                lr=cfg.lr,
                alpha=cfg.alpha,
            )

            for sev_cfg in test_grid:
                met = evaluate_model(
                    model_name=model_name,
                    model=aug_model,
                    test_loader=test_loader,
                    scaler=scaler,
                    device=device,
                    test_perturb=cfg.test_perturb,
                    severity_cfg=sev_cfg,
                    alpha=cfg.alpha,
                )
                rows.append(
                    {
                        "model": disp,
                        "training_mode": "drift_augmented",
                        "train_perturb": cfg.train_perturb,
                        "test_perturb": cfg.test_perturb,
                        "seed": int(seed),
                        "severity_cfg": float(sev_cfg),
                        "severity_eff": float(cfg.alpha * sev_cfg),
                        "R2": met["R2"],
                        "RMSE": met["RMSE"],
                        "MAE": met["MAE"],
                        "run_id": run_id,
                    }
                )
                print(
                    f"    drift_augmented severity_cfg={sev_cfg:.2f} "
                    f"R2={met['R2']:.4f} RMSE={met['RMSE']:.2f} MAE={met['MAE']:.2f}"
                )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[
        [
            "model",
            "training_mode",
            "train_perturb",
            "test_perturb",
            "seed",
            "severity_cfg",
            "severity_eff",
            "R2",
            "RMSE",
            "MAE",
            "run_id",
        ]
    ]
    df.to_csv(OUT_CSV, index=False)

    summary = (
        df.groupby(["model", "training_mode", "test_perturb", "severity_eff"], as_index=False)
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            n_seed=("seed", "nunique"),
        )
        .sort_values(["model", "training_mode", "severity_eff"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    summary["run_id"] = run_id
    summary = summary[
        [
            "model",
            "training_mode",
            "test_perturb",
            "severity_eff",
            "R2_mean",
            "R2_std",
            "RMSE_mean",
            "RMSE_std",
            "MAE_mean",
            "MAE_std",
            "n_seed",
            "run_id",
        ]
    ]
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    meta = {
        "script": "run_minimal_matched_robustness_controls_v2.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "config": asdict(cfg),
        "dataset": "wind_final.csv",
        "target": "power",
        "split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
        "seq_len": ts_cfg.seq_len,
        "pred_len": ts_cfg.pred_len,
        "drift_definition": "x'_{t,d}=x_{t,d}+(t/(L-1))*(alpha*severity_cfg*std_d), std_d from per-sample per-channel sequence std.",
        "outputs": {
            "csv": str(OUT_CSV),
            "summary_csv": str(OUT_SUMMARY_CSV),
            "meta_json": str(OUT_META),
        },
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[ok] wrote: {OUT_CSV}")
    print(f"[ok] wrote: {OUT_SUMMARY_CSV}")
    print(f"[ok] wrote: {OUT_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
