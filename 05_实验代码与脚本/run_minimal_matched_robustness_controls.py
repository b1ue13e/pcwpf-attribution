#!/usr/bin/env python3
"""
Minimal matched robustness controls for three strong baselines:
- DLinear
- iTransformer
- TimeMixer

Protocol (minimal):
- Start from existing checkpoints.
- Keep clean baseline evaluation.
- Do short fine-tune with Gaussian augmentation only.
- Evaluate at sigma_cfg in a small grid (default: 0.0, 0.4).
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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
OUT_CSV = OUT_DIR / "fairness_minimal_aug_controls.csv"
OUT_META = OUT_DIR / "fairness_minimal_aug_controls.meta.json"


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
    seed: int = 42
    finetune_epochs: int = 1
    sigma_train_max: float = 0.4
    sigma_test_grid: str = "0.0,0.4"
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


def parse_sigma_grid(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No value provided in --sigma-test-grid")
    return vals


def pick_device(flag: str) -> torch.device:
    if flag == "cuda":
        return torch.device("cuda")
    if flag == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inject_gaussian_protocol(x: torch.Tensor, sigma_cfg: float, alpha: float = 1.5) -> torch.Tensor:
    if sigma_cfg <= 0.0:
        return x
    std = x.std(dim=1, keepdim=True)
    noise = torch.randn_like(x) * (alpha * sigma_cfg) * std
    return x + noise


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
    sigma_cfg: float,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            by = by.to(device)
            bx_noisy = inject_gaussian_protocol(bx, sigma_cfg=sigma_cfg, alpha=1.5)
            pred = forward_model(model_name, model, bx_noisy)
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


def finetune_with_gaussian_aug(
    model_name: str,
    model: nn.Module,
    train_loader,
    device: torch.device,
    sigma_train_max: float,
    epochs: int,
    lr: float,
) -> nn.Module:
    aug_model = copy.deepcopy(model).to(device)
    aug_model.train()

    optimizer = torch.optim.AdamW(aug_model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            sigma_cfg = float(torch.rand(1).item()) * sigma_train_max
            bx_noisy = inject_gaussian_protocol(bx, sigma_cfg=sigma_cfg, alpha=1.5)

            optimizer.zero_grad()
            pred = forward_model(model_name, aug_model, bx_noisy)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aug_model.parameters(), max_norm=1.0)
            optimizer.step()

    aug_model.eval()
    return aug_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Run minimal matched robustness controls.")
    parser.add_argument("--models", type=str, default="dlinear,itransformer,timemixer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune-epochs", type=int, default=1)
    parser.add_argument("--sigma-train-max", type=float, default=0.4)
    parser.add_argument("--sigma-test-grid", type=str, default="0.0,0.4")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    cfg = RunConfig(
        models=args.models,
        seed=args.seed,
        finetune_epochs=args.finetune_epochs,
        sigma_train_max=args.sigma_train_max,
        sigma_test_grid=args.sigma_test_grid,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    set_seed(cfg.seed)
    models = parse_models(cfg.models)
    sigma_grid = parse_sigma_grid(cfg.sigma_test_grid)
    device = pick_device(cfg.device)

    print(f"[info] device={device}")
    print(f"[info] models={models}")
    print(f"[info] sigma_grid={sigma_grid}")

    data, ts_cfg, train_loader, _, test_loader, scaler = build_loaders(cfg.batch_size)
    input_dim = data.shape[1]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_minimal_fairness_control"

    rows: List[Dict] = []
    for model_name in models:
        print(f"\n[model] {model_name}")
        base_model = build_model(model_name, input_dim=input_dim, device=device)

        # clean baseline mode
        for sigma_cfg in sigma_grid:
            met = evaluate_model(model_name, base_model, test_loader, scaler, device, sigma_cfg=sigma_cfg)
            rows.append(
                {
                    "model": model_name,
                    "training_mode": "clean_baseline",
                    "seed": cfg.seed,
                    "sigma_cfg": float(sigma_cfg),
                    "sigma_eff": float(1.5 * sigma_cfg),
                    "R2": met["R2"],
                    "RMSE": met["RMSE"],
                    "MAE": met["MAE"],
                    "run_id": run_id,
                }
            )
            print(
                f"  clean_baseline sigma_cfg={sigma_cfg:.2f} "
                f"R2={met['R2']:.4f} RMSE={met['RMSE']:.2f} MAE={met['MAE']:.2f}"
            )

        # gaussian-augmented mode
        aug_model = finetune_with_gaussian_aug(
            model_name,
            base_model,
            train_loader,
            device,
            sigma_train_max=cfg.sigma_train_max,
            epochs=cfg.finetune_epochs,
            lr=cfg.lr,
        )
        for sigma_cfg in sigma_grid:
            met = evaluate_model(model_name, aug_model, test_loader, scaler, device, sigma_cfg=sigma_cfg)
            rows.append(
                {
                    "model": model_name,
                    "training_mode": "gaussian_augmented",
                    "seed": cfg.seed,
                    "sigma_cfg": float(sigma_cfg),
                    "sigma_eff": float(1.5 * sigma_cfg),
                    "R2": met["R2"],
                    "RMSE": met["RMSE"],
                    "MAE": met["MAE"],
                    "run_id": run_id,
                }
            )
            print(
                f"  gaussian_augmented sigma_cfg={sigma_cfg:.2f} "
                f"R2={met['R2']:.4f} RMSE={met['RMSE']:.2f} MAE={met['MAE']:.2f}"
            )

    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[
        ["model", "training_mode", "seed", "sigma_cfg", "sigma_eff", "R2", "RMSE", "MAE", "run_id"]
    ]
    df.to_csv(OUT_CSV, index=False)

    meta = {
        "script": "run_minimal_matched_robustness_controls.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "config": asdict(cfg),
        "dataset": "wind_final.csv",
        "target": "power",
        "split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
        "seq_len": ts_cfg.seq_len,
        "pred_len": ts_cfg.pred_len,
        "outputs": {"csv": str(OUT_CSV), "meta": str(OUT_META)},
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[ok] wrote: {OUT_CSV}")
    print(f"[ok] wrote: {OUT_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
