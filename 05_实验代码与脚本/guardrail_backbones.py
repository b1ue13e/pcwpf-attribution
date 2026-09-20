from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultra_lsnt_timeseries import LSNTConfig, TimeSeriesConfig, UltraLSNTForecaster


BACKBONE_CHOICES: Tuple[str, ...] = ("ultra", "itransformer")

DEFAULT_ULTRA_WINDCN_CONFIG = os.path.join("checkpoints_ts", "main", "model_config.json")
DEFAULT_ULTRA_WINDCN_CHECKPOINT = os.path.join("checkpoints_ts", "main", "best_model.pth")
DEFAULT_ULTRA_WINDUS_CONFIG = os.path.join("checkpoints_ts", "exp_wind_us", "model_config.json")
DEFAULT_ULTRA_WINDUS_CHECKPOINT = os.path.join("checkpoints_ts", "exp_wind_us", "best_model.pth")
DEFAULT_ITRANSFORMER_WINDCN_CHECKPOINT = os.path.join("checkpoints_ts", "iTransformer_best.pth")


def normalize_state_dict(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def partial_load_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, object]:
    model_state = model.state_dict()
    matched: Dict[str, torch.Tensor] = {}
    skipped: Dict[str, Dict[str, List[int] | None]] = {}
    for key, value in state_dict.items():
        model_value = model_state.get(key)
        model_shape = list(model_value.shape) if hasattr(model_value, "shape") else None
        ckpt_shape = list(value.shape) if hasattr(value, "shape") else None
        if model_value is not None and model_shape == ckpt_shape:
            matched[key] = value
        else:
            skipped[key] = {"checkpoint_shape": ckpt_shape, "model_shape": model_shape}

    model_state.update(matched)
    model.load_state_dict(model_state, strict=False)
    return {
        "matched_keys": int(len(matched)),
        "skipped_keys": int(len(skipped)),
        "missing_model_keys": int(sum(1 for key in model_state if key not in matched)),
        "skipped_examples": {k: skipped[k] for k in list(skipped.keys())[:20]},
    }


def infer_dataset_family(data_ref: str | None = None, dataset_name: str | None = None) -> str:
    candidates = [str(data_ref or ""), str(dataset_name or "")]
    merged = " ".join(candidates).replace("\\", "/").lower()
    if any(token in merged for token in ("wind_us", "wind (us)", "uswind")):
        return "wind_us"
    if any(token in merged for token in ("wind_final", "wind_cn", "wind (cn)", "xinjiang")):
        return "wind_cn"
    basis = Path(str(data_ref or dataset_name or "dataset")).stem.strip().lower()
    return basis or "dataset"


def default_ultra_paths(data_ref: str | None = None, dataset_name: str | None = None) -> Tuple[str, str]:
    family = infer_dataset_family(data_ref, dataset_name)
    if family == "wind_us":
        return DEFAULT_ULTRA_WINDUS_CONFIG, DEFAULT_ULTRA_WINDUS_CHECKPOINT
    return DEFAULT_ULTRA_WINDCN_CONFIG, DEFAULT_ULTRA_WINDCN_CHECKPOINT


def default_itransformer_checkpoint(data_ref: str | None = None, dataset_name: str | None = None) -> str | None:
    family = infer_dataset_family(data_ref, dataset_name)
    if family == "wind_cn":
        return DEFAULT_ITRANSFORMER_WINDCN_CHECKPOINT
    return None


def backbone_display_name(backbone: str) -> str:
    return {
        "ultra": "Ultra-LSNT",
        "itransformer": "iTransformer",
    }[backbone]


def clip_variant_name(backbone: str) -> str:
    return f"{backbone_display_name(backbone)} + clip-only (posthoc)"


def raw_variant_name(backbone: str) -> str:
    return f"{backbone_display_name(backbone)} (no guardrail)"


def full_guardrail_name(backbone: str) -> str:
    if backbone == "ultra":
        return "FG-MoE (full-trained)"
    return f"{backbone_display_name(backbone)} + full guardrail"


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


def build_ultra_model(
    ts_cfg: TimeSeriesConfig,
    device: torch.device,
    *,
    input_dim: int,
    checkpoint_path: str | None = None,
    config_path: str | None = None,
    data_ref: str | None = None,
    dataset_name: str | None = None,
) -> UltraLSNTForecaster:
    resolved_config, resolved_checkpoint = default_ultra_paths(data_ref, dataset_name)
    config_path = config_path or resolved_config
    checkpoint_path = checkpoint_path or resolved_checkpoint

    with open(config_path, "r", encoding="utf-8") as f:
        cfgj = json.load(f)
    cfg = LSNTConfig(
        input_dim=int(input_dim),
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
        target_skip_rate=cfgj.get("target_skip_rate", 0.12),
        skip_budget_weight=cfgj.get("skip_budget_weight", 1.0),
        slow_path_penalty_weight=cfgj.get("slow_path_penalty_weight", 0.15),
        easy_alignment_weight=cfgj.get("easy_alignment_weight", 0.2),
        dual_mode_expansion=cfgj.get("dual_mode_expansion", 4),
        probabilistic_mode="none",
        heteroscedastic_moe=False,
    )
    model = UltraLSNTForecaster(cfg, ts_cfg).to(device)
    load_report = {
        "loaded_from": checkpoint_path,
        "config_path": config_path,
        "matched_keys": 0,
        "skipped_keys": 0,
        "missing_model_keys": len(model.state_dict()),
        "skipped_examples": {},
    }
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        load_report = partial_load_state_dict(model, normalize_state_dict(ckpt))
        load_report["loaded_from"] = checkpoint_path
        load_report["config_path"] = config_path
    model._load_report = load_report
    return model


def build_itransformer_model(
    ts_cfg: TimeSeriesConfig,
    device: torch.device,
    *,
    input_dim: int,
    checkpoint_path: str | None = None,
    data_ref: str | None = None,
    dataset_name: str | None = None,
) -> iTransformer:
    checkpoint_path = checkpoint_path or default_itransformer_checkpoint(data_ref, dataset_name)
    model = iTransformer(
        {
            "seq_len": ts_cfg.seq_len,
            "pred_len": ts_cfg.pred_len,
            "input_dim": int(input_dim),
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.1,
        }
    ).to(device)
    load_report = {
        "loaded_from": checkpoint_path,
        "config_path": None,
        "matched_keys": 0,
        "skipped_keys": 0,
        "missing_model_keys": len(model.state_dict()),
        "skipped_examples": {},
    }
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        load_report = partial_load_state_dict(model, normalize_state_dict(ckpt))
        load_report["loaded_from"] = checkpoint_path
        load_report["config_path"] = None
    model._load_report = load_report
    return model


def build_backbone_model(
    backbone: str,
    ts_cfg: TimeSeriesConfig,
    device: torch.device,
    *,
    input_dim: int,
    data_ref: str | None = None,
    dataset_name: str | None = None,
    checkpoint_path: str | None = None,
    config_path: str | None = None,
):
    if backbone == "ultra":
        return build_ultra_model(
            ts_cfg,
            device,
            input_dim=input_dim,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            data_ref=data_ref,
            dataset_name=dataset_name,
        )
    if backbone == "itransformer":
        return build_itransformer_model(
            ts_cfg,
            device,
            input_dim=input_dim,
            checkpoint_path=checkpoint_path,
            data_ref=data_ref,
            dataset_name=dataset_name,
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def model_forward(model, backbone: str, x: torch.Tensor, temperature: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    if backbone == "ultra":
        pred, aux = model(x, temperature=temperature)
        return pred, aux
    if backbone == "itransformer":
        pred, _ = model(x)
        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        aux = torch.zeros((), device=x.device, dtype=pred.dtype)
        return pred, aux
    raise ValueError(f"Unsupported backbone: {backbone}")


def train_base_backbone(
    backbone: str,
    model,
    train_loader,
    val_loader,
    *,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
    aux_weight: float = 0.01,
    max_train_batches: int = 0,
    max_eval_batches: int = 0,
) -> Tuple[torch.nn.Module, List[Dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    history: List[Dict[str, float]] = []

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_aux = 0.0
        train_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred, aux = model_forward(model, backbone, x, temperature=1.0)
            mse = criterion(pred, y)
            loss = mse + aux_weight * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += float(loss.item())
            train_mse += float(mse.item())
            train_aux += float(aux.item())
            train_batches += 1

        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_aux = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                    break
                x = x.to(device)
                y = y.to(device)
                pred, aux = model_forward(model, backbone, x, temperature=0.1)
                mse = criterion(pred, y)
                loss = mse + aux_weight * aux
                val_loss += float(loss.item())
                val_mse += float(mse.item())
                val_aux += float(aux.item())
                val_batches += 1

        row = {
            "epoch": float(ep),
            "train_loss": train_loss / max(train_batches, 1),
            "train_mse": train_mse / max(train_batches, 1),
            "train_aux": train_aux / max(train_batches, 1),
            "val_loss": val_loss / max(val_batches, 1),
            "val_mse": val_mse / max(val_batches, 1),
            "val_aux": val_aux / max(val_batches, 1),
        }
        history.append(row)

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    model._load_report = {
        "loaded_from": None,
        "config_path": None,
        "matched_keys": 0,
        "skipped_keys": 0,
        "missing_model_keys": 0,
        "skipped_examples": {},
        "trained_from_scratch": True,
    }
    return model, history


def save_backbone_checkpoint(model, path: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target)
    return str(target)
