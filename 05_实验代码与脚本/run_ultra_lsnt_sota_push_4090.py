#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from robustness_faults import FAULT_CLEAN, apply_fault_tensor, parse_faults, sample_fault_uniform
from priority_dataset_hub import default_target_for_reference, list_priority_datasets, resolve_dataset_reference
from ultra_lsnt_sota_push import (
    PredictOnlyWrapper,
    SotaPushConfig,
    SotaUltraLSNTForecaster,
    average_state_dicts,
    partial_load_state_dict,
)
from ultra_lsnt_timeseries import StandardScaler, compute_metrics, load_csv_data


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "sota_push_4090"


@dataclass
class DatasetSpec:
    path: str
    target: str
    name: str


@dataclass
class DataBundle:
    spec: DatasetSpec
    resolved_reference: str
    resolved_path: str
    resolved_target: str
    data: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    @property
    def input_dim(self) -> int:
        return int(self.data.shape[1])


class WindowDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int, target_idx: int = -1):
        self.data = data.astype(np.float32, copy=False)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.target_idx = int(target_idx)

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        seq_x = self.data[idx : idx + self.seq_len]
        seq_y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, self.target_idx]
        return torch.from_numpy(seq_x), torch.from_numpy(seq_y)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.backup: Dict[str, torch.Tensor] | None = None

    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            value_cpu = value.detach().cpu()
            if torch.is_floating_point(value_cpu):
                self.shadow[key].mul_(self.decay).add_(value_cpu, alpha=1.0 - self.decay)
            else:
                self.shadow[key] = value_cpu.clone()

    def store(self, model: nn.Module) -> None:
        self.backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module) -> None:
        if self.backup is None:
            return
        model.load_state_dict(self.backup, strict=False)
        self.backup = None


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(flag: str) -> torch.device:
    if flag == "cuda":
        return torch.device("cuda")
    if flag == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_dataset_spec(raw: str, *, default_name: str | None = None) -> DatasetSpec:
    parts = [x.strip() for x in raw.split("::")]
    if len(parts) == 1:
        path = parts[0]
        target = default_target_for_reference(path, "power")
        name = default_name or parts[0]
        return DatasetSpec(path=path, target=target, name=name)
    if len(parts) < 2:
        raise ValueError("Dataset spec must use `path::target::name`, `path::target`, or a dataset alias.")
    path = parts[0]
    target = parts[1] or default_target_for_reference(path, "power")
    name = parts[2] if len(parts) >= 3 and parts[2] else (default_name or Path(path).stem)
    return DatasetSpec(path=path, target=target, name=name)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sanitize_name(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return out.strip("_") or "stage"


def build_data_bundle(
    spec: DatasetSpec,
    *,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    num_workers: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> DataBundle:
    resolved = resolve_dataset_reference(spec.path, spec.target)
    data, feature_names = load_csv_data(resolved["resolved_path"], resolved["target"])
    if data is None:
        raise RuntimeError(f"Failed to load dataset: {spec.path}")

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    scaler = StandardScaler()
    train_data = scaler.fit_transform(data[:train_end])
    val_data = scaler.transform(data[train_end:val_end])
    test_data = scaler.transform(data[val_end:])

    train_ds = WindowDataset(train_data, seq_len=seq_len, pred_len=pred_len, target_idx=-1)
    val_ds = WindowDataset(val_data, seq_len=seq_len, pred_len=pred_len, target_idx=-1)
    test_ds = WindowDataset(test_data, seq_len=seq_len, pred_len=pred_len, target_idx=-1)
    if len(train_ds) <= 0 or len(val_ds) <= 0 or len(test_ds) <= 0:
        raise RuntimeError(f"Dataset `{spec.name}` is too short for seq_len={seq_len}, pred_len={pred_len}.")

    pin_memory = torch.cuda.is_available()
    generator = torch.Generator()
    generator.manual_seed(seed)
    common_loader_kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": pin_memory,
        "persistent_workers": int(num_workers) > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common_loader_kwargs,
    )
    eval_batch = max(1, batch_size * 2)
    val_loader = DataLoader(val_ds, batch_size=eval_batch, shuffle=False, drop_last=False, **common_loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=eval_batch, shuffle=False, drop_last=False, **common_loader_kwargs)
    return DataBundle(
        spec=spec,
        resolved_reference=str(resolved["reference"]),
        resolved_path=str(resolved["resolved_path"]),
        resolved_target=str(resolved["target"]),
        data=data,
        feature_names=feature_names,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def current_temperature(epoch_idx: int, total_epochs: int, start: float, end: float) -> float:
    if total_epochs <= 1:
        return end
    progress = float(epoch_idx) / float(total_epochs - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(end + (start - end) * cosine)


def inverse_scale_target(arr: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return arr * float(scaler.std[-1]) + float(scaler.mean[-1])


def maybe_augment_batch(
    x: torch.Tensor,
    *,
    faults: Sequence[str],
    fault_prob: float,
    max_severity: float,
    alpha: float,
    epoch_idx: int,
    total_epochs: int,
    global_seed: int,
    batch_idx: int,
    composite_prob: float,
) -> Tuple[torch.Tensor, Dict[str, float | str]]:
    rng = np.random.default_rng(global_seed * 1_000_000 + epoch_idx * 10_000 + batch_idx)
    if (not faults) or rng.random() > fault_prob:
        return x, {"fault": FAULT_CLEAN, "severity": 0.0, "fault2": ""}

    curriculum = min(1.0, (epoch_idx + 1) / max(1, total_epochs * 0.65))
    severity = float(rng.uniform(0.05, max(max_severity * curriculum, 0.05)))
    fault = sample_fault_uniform(faults, rng)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(global_seed * 10_000_000 + epoch_idx * 100_000 + batch_idx)
    out = apply_fault_tensor(x, fault=fault, severity=severity, alpha=alpha, generator=gen)

    second_fault = ""
    if len(faults) > 1 and rng.random() < composite_prob:
        candidates = [f for f in faults if f != fault]
        second_fault = candidates[int(rng.integers(0, len(candidates)))]
        second_severity = float(rng.uniform(0.02, max(0.02, severity)))
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(global_seed * 20_000_000 + epoch_idx * 100_000 + batch_idx)
        out = apply_fault_tensor(out, fault=second_fault, severity=second_severity, alpha=alpha, generator=gen2)

    return out, {"fault": fault, "severity": severity, "fault2": second_fault}


def weighted_forecast_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    aux_loss: torch.Tensor,
    huber_beta: float,
    aux_weight: float,
) -> torch.Tensor:
    horizon_weights = torch.linspace(1.05, 0.95, steps=target.size(1), device=target.device, dtype=target.dtype)
    horizon_weights = horizon_weights / horizon_weights.mean()
    mse = ((pred - target) ** 2 * horizon_weights.unsqueeze(0)).mean()
    huber = F.smooth_l1_loss(pred, target, beta=huber_beta, reduction="none")
    huber = (huber * horizon_weights.unsqueeze(0)).mean()
    return 0.45 * mse + 0.55 * huber + aux_weight * aux_loss


def load_state_payload(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def collect_jump_gates(model: nn.Module) -> List[nn.Module]:
    gates: List[nn.Module] = []
    for module in model.modules():
        if hasattr(module, "threshold_logit"):
            gates.append(module)
    return gates


def get_gate_thresholds(model: nn.Module) -> List[float]:
    out: List[float] = []
    for gate in collect_jump_gates(model):
        out.append(float(torch.sigmoid(gate.threshold_logit.detach()).item()))
    return out


def set_gate_thresholds(model: nn.Module, threshold: float) -> None:
    thr = float(np.clip(threshold, 1e-4, 1.0 - 1e-4))
    logit = float(np.log(thr / (1.0 - thr)))
    for gate in collect_jump_gates(model):
        gate.threshold_logit.data.fill_(logit)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler,
    *,
    device: torch.device,
    temperature: float,
    fault: str = FAULT_CLEAN,
    severity: float = 0.0,
    alpha: float = 1.5,
    seed: int = 42,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    efficiency_rows: List[Dict[str, float]] = []
    for batch_idx, (bx, by) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        bx = bx.to(device)
        by = by.to(device)
        if fault != FAULT_CLEAN and severity > 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed * 1_000_000 + batch_idx)
            bx = apply_fault_tensor(bx, fault=fault, severity=severity, alpha=alpha, generator=gen)
        pred, _, stats = model(bx, temperature=temperature, return_stats=True)
        preds.append(pred.cpu().numpy())
        trues.append(by.cpu().numpy())
        efficiency_rows.append(stats.get("efficiency", {}))

    pred_np = inverse_scale_target(np.concatenate(preds, axis=0), scaler)
    true_np = inverse_scale_target(np.concatenate(trues, axis=0), scaler)
    metrics = compute_metrics(pred_np, true_np)
    avg_skip = float(np.mean([row.get("avg_skip_rate", 0.0) for row in efficiency_rows])) if efficiency_rows else 0.0
    avg_slow = float(np.mean([row.get("avg_slow_ratio", 1.0) for row in efficiency_rows])) if efficiency_rows else 1.0
    avg_compute = float(np.mean([row.get("avg_compute_ratio", 1.0) for row in efficiency_rows])) if efficiency_rows else 1.0
    return {
        "R2": float(metrics["R2"]),
        "RMSE": float(metrics["RMSE"]),
        "MAE": float(metrics["MAE"]),
        "MAPE": float(metrics["MAPE"]),
        "avg_skip_rate": avg_skip,
        "avg_slow_ratio": avg_slow,
        "avg_compute_ratio": avg_compute,
    }


@torch.no_grad()
def calibrate_jump_gate_thresholds(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler,
    *,
    device: torch.device,
    temperature: float,
    target_skip_rate: float,
    threshold_grid: Sequence[float],
    min_skip_rate: float,
    score_penalty: float,
    seed: int,
    max_batches: int | None,
) -> Dict[str, object]:
    gates = collect_jump_gates(model)
    if not gates:
        return {"applied": False, "reason": "no_jump_gates"}

    base_thresholds = get_gate_thresholds(model)
    best_metrics = evaluate_loader(
        model,
        loader,
        scaler,
        device=device,
        temperature=temperature,
        fault=FAULT_CLEAN,
        severity=0.0,
        seed=seed,
        max_batches=max_batches,
    )
    best_threshold = float(np.mean(base_thresholds)) if base_thresholds else 0.5
    best_objective = float(best_metrics["R2"]) - float(score_penalty) * abs(float(best_metrics["avg_skip_rate"]) - float(target_skip_rate))
    best_passes = float(best_metrics["avg_skip_rate"]) >= float(min_skip_rate)
    grid_rows: List[Dict[str, float]] = []

    for threshold in threshold_grid:
        set_gate_thresholds(model, threshold)
        metrics = evaluate_loader(
            model,
            loader,
            scaler,
            device=device,
            temperature=temperature,
            fault=FAULT_CLEAN,
            severity=0.0,
            seed=seed,
            max_batches=max_batches,
        )
        objective = float(metrics["R2"]) - float(score_penalty) * abs(float(metrics["avg_skip_rate"]) - float(target_skip_rate))
        passes = float(metrics["avg_skip_rate"]) >= float(min_skip_rate)
        grid_rows.append(
            {
                "threshold": float(threshold),
                "R2": float(metrics["R2"]),
                "RMSE": float(metrics["RMSE"]),
                "avg_skip_rate": float(metrics["avg_skip_rate"]),
                "avg_slow_ratio": float(metrics["avg_slow_ratio"]),
                "objective": objective,
            }
        )
        if passes and ((not best_passes) or objective > best_objective):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_objective = objective
            best_passes = True
        elif (not best_passes) and objective > best_objective:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_objective = objective

    set_gate_thresholds(model, best_threshold)
    return {
        "applied": True,
        "base_thresholds": base_thresholds,
        "selected_threshold": float(best_threshold),
        "selected_metrics": {k: float(v) for k, v in best_metrics.items()},
        "target_skip_rate": float(target_skip_rate),
        "min_skip_rate": float(min_skip_rate),
        "score_penalty": float(score_penalty),
        "grid": grid_rows,
    }


def train_stage(
    *,
    model: SotaUltraLSNTForecaster,
    bundle: DataBundle,
    stage_name: str,
    out_dir: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    temperature_start: float,
    temperature_end: float,
    faults: Sequence[str],
    fault_prob: float,
    max_severity: float,
    alpha: float,
    robust_weight: float,
    val_faults: Sequence[str],
    val_fault_severity: float,
    ema_decay: float,
    checkpoint_topk: int,
    patience: int,
    seed: int,
    use_amp: bool,
    aux_weight: float,
    huber_beta: float,
    composite_prob: float,
    enable_jump_gate_calibration: bool,
    jump_gate_threshold_grid: Sequence[float],
    jump_gate_min_skip: float,
    jump_gate_score_penalty: float,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> Dict[str, object]:
    stage_dir = out_dir / sanitize_name(stage_name)
    ckpt_dir = stage_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=max(lr * 0.05, 1e-6))
    grad_scaler = GradScaler(enabled=use_amp)
    ema = ModelEMA(model, decay=ema_decay)

    history: List[Dict] = []
    topk: List[Tuple[float, Path]] = []
    best_score = -float("inf")
    best_epoch = -1
    bad_epochs = 0

    for epoch_idx in range(epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        total_batches = 0
        temp = current_temperature(epoch_idx, epochs, temperature_start, temperature_end)

        for batch_idx, (bx, by) in enumerate(bundle.train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            bx_aug, _ = maybe_augment_batch(
                bx,
                faults=faults,
                fault_prob=fault_prob,
                max_severity=max_severity,
                alpha=alpha,
                epoch_idx=epoch_idx,
                total_epochs=epochs,
                global_seed=seed,
                batch_idx=batch_idx,
                composite_prob=composite_prob,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                pred, aux_loss = model(bx_aug, temperature=temp)
                loss = weighted_forecast_loss(pred, by, aux_loss=aux_loss, huber_beta=huber_beta, aux_weight=aux_weight)

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            ema.update(model)

            total_loss += float(loss.item())
            total_batches += 1

        scheduler.step()
        ema.store(model)
        ema.copy_to(model)

        clean_metrics = evaluate_loader(
            model,
            bundle.val_loader,
            bundle.scaler,
            device=device,
            temperature=temp,
            fault=FAULT_CLEAN,
            severity=0.0,
            alpha=alpha,
            seed=seed,
            max_batches=max_eval_batches,
        )
        robust_rows = []
        robust_scores = []
        for fault_name in val_faults:
            metrics = evaluate_loader(
                model,
                bundle.val_loader,
                bundle.scaler,
                device=device,
                temperature=temp,
                fault=fault_name,
                severity=val_fault_severity,
                alpha=alpha,
                seed=seed,
                max_batches=max_eval_batches,
            )
            robust_rows.append({"fault": fault_name, **metrics})
            robust_scores.append(metrics["R2"])
        robust_r2 = float(np.mean(robust_scores)) if robust_scores else clean_metrics["R2"]
        stage_score = float(clean_metrics["R2"] + robust_weight * robust_r2)

        ckpt_path = ckpt_dir / f"epoch_{epoch_idx + 1:03d}.pth"
        torch.save(
            {
                "epoch": epoch_idx + 1,
                "stage_name": stage_name,
                "score": stage_score,
                "clean_metrics": clean_metrics,
                "robust_metrics": robust_rows,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "config": model.config.to_dict(),
            },
            ckpt_path,
        )
        topk.append((stage_score, ckpt_path))
        topk.sort(key=lambda item: item[0], reverse=True)
        while len(topk) > checkpoint_topk:
            _, to_remove = topk.pop(-1)
            if to_remove.exists():
                to_remove.unlink()

        if stage_score > best_score:
            best_score = stage_score
            best_epoch = epoch_idx + 1
            bad_epochs = 0
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch": epoch_idx + 1,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "temperature": temp,
                "train_loss": total_loss / max(1, total_batches),
                "clean_R2": clean_metrics["R2"],
                "clean_RMSE": clean_metrics["RMSE"],
                "clean_skip_rate": clean_metrics["avg_skip_rate"],
                "clean_slow_ratio": clean_metrics["avg_slow_ratio"],
                "clean_compute_ratio": clean_metrics["avg_compute_ratio"],
                "robust_R2_mean": robust_r2,
                "score": stage_score,
                "elapsed_sec": time.time() - t0,
            }
        )
        ema.restore(model)
        if bad_epochs >= patience:
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(stage_dir / "history.csv", index=False)

    if not topk:
        clean_metrics = evaluate_loader(
            model,
            bundle.val_loader,
            bundle.scaler,
            device=device,
            temperature=temperature_end,
            fault=FAULT_CLEAN,
            severity=0.0,
            alpha=alpha,
            seed=seed,
            max_batches=max_eval_batches,
        )
        robust_rows = []
        robust_scores = []
        for fault_name in val_faults:
            metrics = evaluate_loader(
                model,
                bundle.val_loader,
                bundle.scaler,
                device=device,
                temperature=temperature_end,
                fault=fault_name,
                severity=val_fault_severity,
                alpha=alpha,
                seed=seed,
                max_batches=max_eval_batches,
            )
            robust_rows.append({"fault": fault_name, **metrics})
            robust_scores.append(metrics["R2"])
        robust_r2 = float(np.mean(robust_scores)) if robust_scores else clean_metrics["R2"]
        stage_score = float(clean_metrics["R2"] + robust_weight * robust_r2)
        ckpt_path = ckpt_dir / "epoch_000.pth"
        torch.save(
            {
                "epoch": 0,
                "stage_name": stage_name,
                "score": stage_score,
                "clean_metrics": clean_metrics,
                "robust_metrics": robust_rows,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "config": model.config.to_dict(),
            },
            ckpt_path,
        )
        best_score = stage_score if not np.isfinite(best_score) else best_score
        best_epoch = 0
        topk.append((stage_score, ckpt_path))

    averaged_paths = [path for _, path in topk]
    averaged_state = average_state_dicts([load_state_payload(path) for path in averaged_paths])
    avg_ckpt_path = stage_dir / "best_avg_model.pth"
    model.load_state_dict(averaged_state, strict=False)

    jump_gate_calibration = {
        "applied": False,
        "reason": "disabled",
    }
    if enable_jump_gate_calibration:
        jump_gate_calibration = calibrate_jump_gate_thresholds(
            model,
            bundle.val_loader,
            bundle.scaler,
            device=device,
            temperature=temperature_end,
            target_skip_rate=float(model.config.target_skip_rate),
            threshold_grid=jump_gate_threshold_grid,
            min_skip_rate=jump_gate_min_skip,
            score_penalty=jump_gate_score_penalty,
            seed=seed,
            max_batches=max_eval_batches,
        )
    torch.save(
        {
            "stage_name": stage_name,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "topk_paths": [str(p) for p in averaged_paths],
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "config": model.config.to_dict(),
            "jump_gate_calibration": jump_gate_calibration,
        },
        avg_ckpt_path,
    )

    test_rows = []
    for fault_name, sev in [
        (FAULT_CLEAN, 0.0),
        ("gaussian", min(max_severity, 0.15)),
        ("gaussian", max_severity),
        ("drift", max_severity),
        ("quantization", max_severity),
        ("missingness", max_severity),
    ]:
        metrics = evaluate_loader(
            model,
            bundle.test_loader,
            bundle.scaler,
            device=device,
            temperature=temperature_end,
            fault=fault_name,
            severity=sev,
            alpha=alpha,
            seed=seed,
            max_batches=max_eval_batches,
        )
        test_rows.append({"fault": fault_name, "severity": sev, **metrics})
    pd.DataFrame(test_rows).to_csv(stage_dir / "test_metrics.csv", index=False)

    save_json(
        stage_dir / "stage_summary.json",
        {
            "stage_name": stage_name,
            "dataset": asdict(bundle.spec),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "num_epochs_run": len(history),
            "clean_test_skip_rate": next((row["avg_skip_rate"] for row in test_rows if row["fault"] == FAULT_CLEAN), 0.0),
            "clean_test_slow_ratio": next((row["avg_slow_ratio"] for row in test_rows if row["fault"] == FAULT_CLEAN), 1.0),
            "jump_gate_calibration": jump_gate_calibration,
            "topk_paths": [str(p) for p in averaged_paths],
        },
    )
    return {
        "stage_dir": stage_dir,
        "checkpoint": avg_ckpt_path,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "history": history_df,
    }


def benchmark_torch_module(
    module: nn.Module,
    sample_input: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    times_ms: List[float] = []
    module.eval()
    with torch.inference_mode():
        for _ in range(max(1, warmup)):
            _ = module(sample_input)
        for _ in range(max(1, iters)):
            t0 = time.perf_counter()
            _ = module(sample_input)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    torch.set_num_threads(old_threads)
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p95_ms": float(np.quantile(arr, 0.95)),
        "std_ms": float(arr.std()),
    }


def compact_error(exc: Exception, limit: int = 2000) -> str:
    text = str(exc).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def save_torchscript_module(module: nn.Module, sample_input: torch.Tensor, out_path: Path) -> str:
    try:
        scripted = torch.jit.script(module)
    except Exception:
        scripted = torch.jit.trace(module, sample_input, strict=False, check_trace=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        torch.jit.save(scripted, f)
    return str(out_path)


def export_onnx_via_temp_path(module: nn.Module, sample_input: torch.Tensor, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="ultra_lsnt_", suffix=".onnx", dir=tempfile.gettempdir())
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.onnx.export(
            module,
            sample_input,
            str(tmp_path),
            input_names=["x"],
            output_names=["pred"],
            dynamic_axes={"x": {0: "batch"}, "pred": {0: "batch"}},
            opset_version=17,
        )
        shutil.copy2(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return str(out_path)


def export_and_benchmark(
    model: SotaUltraLSNTForecaster,
    bundle: DataBundle,
    out_dir: Path,
    *,
    temperature: float,
    warmup: int,
    iters: int,
) -> Dict[str, object]:
    export_dir = out_dir / "edge_ready_bundle"
    export_dir.mkdir(parents=True, exist_ok=True)

    cpu_model = copy.deepcopy(model).cpu().eval()
    wrapper = PredictOnlyWrapper(cpu_model, temperature=temperature).cpu().eval()
    sample_input = next(iter(bundle.test_loader))[0][:1].cpu()

    manifest: Dict[str, object] = {}
    manifest["torch_fp32_cpu_single_thread"] = benchmark_torch_module(wrapper, sample_input, warmup=warmup, iters=iters)

    script_path = export_dir / "model_fp32.ts"
    try:
        manifest["torchscript"] = save_torchscript_module(wrapper, sample_input, script_path)
    except Exception as exc:
        manifest["torchscript_error"] = compact_error(exc)

    try:
        quantized = torch.quantization.quantize_dynamic(copy.deepcopy(wrapper), {nn.Linear}, dtype=torch.qint8)
        manifest["torch_dynamic_int8_cpu_single_thread"] = benchmark_torch_module(
            quantized, sample_input, warmup=warmup, iters=iters
        )
        quant_path = export_dir / "model_dynamic_int8.ts"
        try:
            manifest["torchscript_dynamic_int8"] = save_torchscript_module(quantized, sample_input, quant_path)
        except Exception as exc:
            manifest["torchscript_dynamic_int8_error"] = compact_error(exc)
    except Exception as exc:
        manifest["torch_dynamic_int8_error"] = compact_error(exc)

    onnx_path = export_dir / "model_fp32.onnx"
    try:
        manifest["onnx"] = export_onnx_via_temp_path(wrapper, sample_input, onnx_path)
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            arr = sample_input.numpy()
            for _ in range(max(1, warmup)):
                sess.run(None, {"x": arr})
            times_ms = []
            for _ in range(max(1, iters)):
                t0 = time.perf_counter()
                sess.run(None, {"x": arr})
                times_ms.append((time.perf_counter() - t0) * 1000.0)
            arr_ms = np.asarray(times_ms, dtype=np.float64)
            manifest["onnxruntime_cpu_single_thread"] = {
                "mean_ms": float(arr_ms.mean()),
                "p95_ms": float(np.quantile(arr_ms, 0.95)),
                "std_ms": float(arr_ms.std()),
            }
        except Exception as exc:
            manifest["onnxruntime_error"] = compact_error(exc)
    except Exception as exc:
        manifest["onnx_error"] = compact_error(exc)

    save_json(export_dir / "manifest.json", manifest)
    return manifest


def run_transfer_suite(
    *,
    base_state: Dict[str, torch.Tensor],
    model_cfg: Dict,
    transfer_specs: Sequence[DatasetSpec],
    out_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for idx, spec in enumerate(transfer_specs):
        bundle = build_data_bundle(
            spec,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed + 100 + idx,
        )
        cfg = SotaPushConfig(**{**model_cfg, "input_dim": bundle.input_dim})
        model = SotaUltraLSNTForecaster(cfg).to(device)
        load_report = partial_load_state_dict(model, base_state)

        if args.transfer_epochs > 0:
            result = train_stage(
                model=model,
                bundle=bundle,
                stage_name=f"transfer_{spec.name}",
                out_dir=out_dir,
                device=device,
                epochs=args.transfer_epochs,
                lr=args.transfer_lr,
                weight_decay=args.weight_decay,
                temperature_start=args.temperature_start,
                temperature_end=args.temperature_end,
                faults=parse_faults(args.faults),
                fault_prob=args.fault_prob,
                max_severity=args.max_severity,
                alpha=args.alpha,
                robust_weight=args.robust_weight,
                val_faults=parse_faults(args.val_faults),
                val_fault_severity=args.val_fault_severity,
                ema_decay=args.ema_decay,
                checkpoint_topk=max(1, args.checkpoint_topk),
                patience=max(3, args.transfer_patience),
                seed=args.seed + 100 + idx,
                use_amp=(device.type == "cuda" and not args.disable_amp),
                aux_weight=args.aux_weight,
                huber_beta=args.huber_beta,
                composite_prob=args.composite_prob,
                enable_jump_gate_calibration=not args.disable_jump_gate_calibration,
                jump_gate_threshold_grid=tuple(float(x.strip()) for x in args.jump_gate_threshold_grid.split(",") if x.strip()),
                jump_gate_min_skip=args.jump_gate_min_skip,
                jump_gate_score_penalty=args.jump_gate_score_penalty,
                max_train_batches=args.max_train_batches if args.max_train_batches > 0 else None,
                max_eval_batches=args.max_eval_batches if args.max_eval_batches > 0 else None,
            )
            model.load_state_dict(load_state_payload(Path(result["checkpoint"])), strict=False)

        clean_metrics = evaluate_loader(
            model,
            bundle.test_loader,
            bundle.scaler,
            device=device,
            temperature=args.temperature_end,
            fault=FAULT_CLEAN,
            severity=0.0,
            alpha=args.alpha,
            seed=args.seed + 100 + idx,
            max_batches=args.max_eval_batches if args.max_eval_batches > 0 else None,
        )
        severe_metrics = evaluate_loader(
            model,
            bundle.test_loader,
            bundle.scaler,
            device=device,
            temperature=args.temperature_end,
            fault="gaussian",
            severity=args.max_severity,
            alpha=args.alpha,
            seed=args.seed + 100 + idx,
            max_batches=args.max_eval_batches if args.max_eval_batches > 0 else None,
        )
        rows.append(
            {
                "dataset": spec.name,
                "path": spec.path,
                "target": spec.target,
                "warmstart_matched_keys": load_report["matched"],
                "warmstart_skipped_keys": load_report["skipped"],
                "clean_R2": clean_metrics["R2"],
                "clean_RMSE": clean_metrics["RMSE"],
                "gaussian_severe_R2": severe_metrics["R2"],
                "gaussian_severe_RMSE": severe_metrics["RMSE"],
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_dir / "transfer_summary.csv", index=False)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SOTA push training pipeline for Ultra-LSNT on RTX 4090.")
    p.add_argument("--train-data", type=str, default="wind_final.csv")
    p.add_argument("--target", type=str, default="power")
    p.add_argument("--train-name", type=str, default="Wind_CN")
    p.add_argument("--warmstart", action="append", default=[], help="Dataset spec: path::target::name")
    p.add_argument("--transfer-dataset", action="append", default=[], help="Dataset spec: path::target::name")
    p.add_argument("--list-priority-datasets", action="store_true")

    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--disable-amp", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-eval-batches", type=int, default=0)

    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=128)

    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--router-jitter", type=float, default=0.05)
    p.add_argument("--skip-threshold", type=float, default=0.5)
    p.add_argument("--target-skip-rate", type=float, default=0.12)
    p.add_argument("--skip-budget-weight", type=float, default=1.0)
    p.add_argument("--slow-path-penalty-weight", type=float, default=0.15)
    p.add_argument("--easy-alignment-weight", type=float, default=0.2)
    p.add_argument("--linear-kernel", type=int, default=25)
    p.add_argument("--kernel-sizes", type=str, default="3,7,15")

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--pretrain-epochs", type=int, default=30)
    p.add_argument("--transfer-epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--transfer-patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--pretrain-lr", type=float, default=3e-4)
    p.add_argument("--transfer-lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--checkpoint-topk", type=int, default=3)
    p.add_argument("--temperature-start", type=float, default=1.35)
    p.add_argument("--temperature-end", type=float, default=0.35)

    p.add_argument("--faults", type=str, default="gaussian,drift,quantization,missingness")
    p.add_argument("--fault-prob", type=float, default=0.7)
    p.add_argument("--max-severity", type=float, default=0.4)
    p.add_argument("--composite-prob", type=float, default=0.15)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--val-faults", type=str, default="gaussian,drift")
    p.add_argument("--val-fault-severity", type=float, default=0.3)
    p.add_argument("--robust-weight", type=float, default=0.5)
    p.add_argument("--aux-weight", type=float, default=0.5)
    p.add_argument("--huber-beta", type=float, default=0.25)
    p.add_argument("--disable-jump-gate-calibration", action="store_true")
    p.add_argument("--jump-gate-threshold-grid", type=str, default="0.20,0.22,0.24,0.25,0.26,0.27,0.28,0.30,0.32,0.36,0.42,0.48")
    p.add_argument("--jump-gate-min-skip", type=float, default=0.02)
    p.add_argument("--jump-gate-score-penalty", type=float, default=0.05)

    p.add_argument("--export-edge-ready", action="store_true")
    p.add_argument("--bench-warmup", type=int, default=20)
    p.add_argument("--bench-iters", type=int, default=80)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_priority_datasets:
        print("Available priority dataset aliases:")
        for item in list_priority_datasets():
            print(f" - {item['alias']}: {item['description']} ({item['source_path']})")
        return 0

    set_seed(args.seed)
    device = pick_device(args.device)
    use_amp = device.type == "cuda" and not args.disable_amp
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_target = default_target_for_reference(args.train_data, args.target)
    train_spec = DatasetSpec(path=args.train_data, target=train_target, name=args.train_name)
    warmstart_specs = [parse_dataset_spec(raw) for raw in args.warmstart]
    transfer_specs = [parse_dataset_spec(raw) for raw in args.transfer_dataset]
    fault_list = parse_faults(args.faults)
    val_faults = parse_faults(args.val_faults)
    kernel_sizes = tuple(int(x.strip()) for x in args.kernel_sizes.split(",") if x.strip())
    jump_gate_threshold_grid = tuple(float(x.strip()) for x in args.jump_gate_threshold_grid.split(",") if x.strip())

    save_json(
        out_dir / "run_manifest.json",
        {
            "seed": args.seed,
            "device": str(device),
            "train_spec": asdict(train_spec),
            "train_resolved": resolve_dataset_reference(train_spec.path, train_spec.target),
            "warmstart_specs": [asdict(x) for x in warmstart_specs],
            "warmstart_resolved": [resolve_dataset_reference(x.path, x.target) for x in warmstart_specs],
            "transfer_specs": [asdict(x) for x in transfer_specs],
            "transfer_resolved": [resolve_dataset_reference(x.path, x.target) for x in transfer_specs],
            "args": vars(args),
        },
    )

    last_stage_state: Dict[str, torch.Tensor] | None = None
    last_model_cfg: Dict | None = None
    stage_summaries: List[Dict[str, object]] = []

    for idx, spec in enumerate(warmstart_specs + [train_spec]):
        is_main_stage = idx == len(warmstart_specs)
        bundle = build_data_bundle(
            spec,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed + idx,
        )
        model_cfg = SotaPushConfig(
            input_dim=bundle.input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            num_experts=args.num_experts,
            top_k=args.top_k,
            dropout=args.dropout,
            router_jitter_noise=args.router_jitter,
            skip_threshold=args.skip_threshold,
            target_skip_rate=args.target_skip_rate,
            skip_budget_weight=args.skip_budget_weight,
            slow_path_penalty_weight=args.slow_path_penalty_weight,
            easy_alignment_weight=args.easy_alignment_weight,
            linear_kernel=args.linear_kernel,
            kernel_sizes=kernel_sizes,
        )
        model = SotaUltraLSNTForecaster(model_cfg).to(device)
        load_report = None
        if last_stage_state is not None:
            load_report = partial_load_state_dict(model, last_stage_state)

        result = train_stage(
            model=model,
            bundle=bundle,
            stage_name=("main_" if is_main_stage else "warmstart_") + spec.name,
            out_dir=out_dir,
            device=device,
            epochs=args.epochs if is_main_stage else args.pretrain_epochs,
            lr=args.lr if is_main_stage else args.pretrain_lr,
            weight_decay=args.weight_decay,
            temperature_start=args.temperature_start,
            temperature_end=args.temperature_end,
            faults=fault_list,
            fault_prob=args.fault_prob,
            max_severity=args.max_severity,
            alpha=args.alpha,
            robust_weight=args.robust_weight,
            val_faults=val_faults,
            val_fault_severity=args.val_fault_severity,
            ema_decay=args.ema_decay,
            checkpoint_topk=args.checkpoint_topk,
            patience=args.patience if is_main_stage else max(6, args.patience // 2),
            seed=args.seed + idx,
            use_amp=use_amp,
            aux_weight=args.aux_weight,
            huber_beta=args.huber_beta,
            composite_prob=args.composite_prob,
            enable_jump_gate_calibration=not args.disable_jump_gate_calibration,
            jump_gate_threshold_grid=jump_gate_threshold_grid,
            jump_gate_min_skip=args.jump_gate_min_skip,
            jump_gate_score_penalty=args.jump_gate_score_penalty,
            max_train_batches=args.max_train_batches if args.max_train_batches > 0 else None,
            max_eval_batches=args.max_eval_batches if args.max_eval_batches > 0 else None,
        )
        last_stage_state = load_state_payload(Path(result["checkpoint"]))
        last_model_cfg = model_cfg.to_dict()
        stage_summaries.append(
            {
                "stage_name": ("main_" if is_main_stage else "warmstart_") + spec.name,
                "dataset": asdict(spec),
                "resolved_path": bundle.resolved_path,
                "resolved_target": bundle.resolved_target,
                "checkpoint": str(result["checkpoint"]),
                "best_score": result["best_score"],
                "best_epoch": result["best_epoch"],
                "warmstart_report": load_report,
            }
        )

        if is_main_stage and args.export_edge_ready:
            model.load_state_dict(last_stage_state, strict=False)
            export_and_benchmark(
                model,
                bundle,
                Path(result["stage_dir"]),
                temperature=args.temperature_end,
                warmup=args.bench_warmup,
                iters=args.bench_iters,
            )

    pd.DataFrame(stage_summaries).to_csv(out_dir / "stage_summary.csv", index=False)

    if last_stage_state is not None and last_model_cfg is not None and transfer_specs:
        transfer_df = run_transfer_suite(
            base_state=last_stage_state,
            model_cfg=last_model_cfg,
            transfer_specs=transfer_specs,
            out_dir=out_dir,
            device=device,
            args=args,
        )
        if not transfer_df.empty:
            print(transfer_df.to_string(index=False))

    print(f"[done] outputs saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
