#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import contextlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from priority_dataset_hub import resolve_dataset_reference


TORCH_READY = False
TORCH_IMPORT_ERROR: Exception | None = None
try:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    from run_dlinear import DLinear
    with contextlib.redirect_stdout(open(os.devnull, "w", encoding="utf-8")):
        from run_latest_sota import iTransformer
    from ultra_lsnt_sota_push import SotaPushConfig, SotaUltraLSNTForecaster, partial_load_state_dict

    TORCH_READY = True
except Exception as exc:  # pragma: no cover - exercised only in missing-runtime environments.
    TORCH_IMPORT_ERROR = exc
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]


class StandardScaler:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "StandardScaler":
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before transform().")
        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(true, dtype=np.float64).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(y)
    p = p[mask]
    y = y[mask]
    if len(y) == 0:
        return {"R2": math.nan, "RMSE": math.nan, "MAE": math.nan}
    mse = float(np.mean((p - y) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(p - y)))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum((p - y) ** 2) / denom) if denom > 0 else math.nan
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def load_csv_data(reference: str, target: str = "power") -> Tuple[np.ndarray, List[str]]:
    resolved = resolve_dataset_reference(reference, target)
    path = Path(str(resolved["resolved_path"]))
    target = str(resolved["target"])
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column `{target}` not found in {path}. Available: {list(df.columns)}")
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    drop_cols = [col for col in ["date", "time", "datetime", "timestamp", "时间", "日期", "序列", "id", "index", "seq", "序号"] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    keep_cols = []
    for col in df.columns:
        if col == target:
            keep_cols.append(col)
            continue
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        if float(numeric_col.notna().mean()) >= 0.5:
            df[col] = numeric_col
            keep_cols.append(col)
    df = df[keep_cols]
    df[target] = pd.to_numeric(df[target], errors="coerce")
    high_missing = (df.isna().mean() > 0.3)
    high_missing[target] = False
    if high_missing.any():
        df = df.drop(columns=list(high_missing[high_missing].index))
    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill().fillna(0.0)
    cols = [col for col in df.columns if col != target] + [target]
    df = df[cols]
    return df.to_numpy(dtype=np.float32), cols


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_ROOT = PROJECT_ROOT / "06_实验结果与日志" / "results" / "cross_site_wind"

DEFAULT_DATASETS = [
    "wind_cn_core_15min",
    "wind_us_core_15min",
    "kelmarsh_uk_2016",
    "penmanshiel_scotland_2016",
    "smarteole_france_2020",
]
DEFAULT_MODELS = ["ultra_lsnt", "dlinear", "itransformer"]
DEFAULT_SEEDS = [42, 123, 2025]
DEFAULT_RATIOS = [0.0, 0.01, 0.05, 0.10, 0.50, 1.0]


@dataclass(frozen=True)
class DatasetSpec:
    alias: str
    target: str = "power"


@dataclass
class SplitBundle:
    alias: str
    target: str
    data: np.ndarray
    train_pool_raw: np.ndarray
    test_raw: np.ndarray
    train_pool_scaled: np.ndarray
    test_scaled: np.ndarray
    scaler: StandardScaler
    input_dim: int
    resolved_path: str


class WindowDataset(Dataset):  # type: ignore[misc, valid-type]
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int, target_idx: int = -1):
        self.data = data.astype(np.float32, copy=False)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.target_idx = int(target_idx)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, self.target_idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    if TORCH_READY:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def pick_device(flag: str) -> torch.device:
    if not TORCH_READY:
        raise RuntimeError(f"PyTorch is required for training/evaluation: {TORCH_IMPORT_ERROR}")
    if flag == "cpu":
        return torch.device("cpu")
    if flag == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_csv_list(raw: str, defaults: List[str]) -> List[str]:
    if not raw:
        return defaults
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str, defaults: List[float]) -> List[float]:
    if not raw:
        return defaults
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str, defaults: List[int]) -> List[int]:
    if not raw:
        return defaults
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def load_bundle(alias: str, target: str, train_ratio: float) -> SplitBundle:
    resolved = resolve_dataset_reference(alias, target)
    data, _ = load_csv_data(str(resolved["resolved_path"]), str(resolved["target"]))
    if data is None:
        raise RuntimeError(f"Failed to load dataset {alias}")
    split = int(len(data) * train_ratio)
    if split <= 0 or split >= len(data):
        raise RuntimeError(f"Invalid train/test split for {alias}: n={len(data)} split={split}")
    train_pool_raw = data[:split]
    test_raw = data[split:]
    scaler = StandardScaler()
    train_pool_scaled = scaler.fit_transform(train_pool_raw)
    test_scaled = scaler.transform(test_raw)
    return SplitBundle(
        alias=alias,
        target=str(resolved["target"]),
        data=data,
        train_pool_raw=train_pool_raw,
        test_raw=test_raw,
        train_pool_scaled=train_pool_scaled,
        test_scaled=test_scaled,
        scaler=scaler,
        input_dim=int(data.shape[1]),
        resolved_path=str(resolved["resolved_path"]),
    )


def make_loader(data: np.ndarray, seq_len: int, pred_len: int, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0) -> DataLoader:
    dataset = WindowDataset(data, seq_len=seq_len, pred_len=pred_len, target_idx=-1)
    if len(dataset) <= 0:
        raise RuntimeError(f"Not enough rows for seq_len={seq_len}, pred_len={pred_len}: {len(data)} rows")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def source_train_val_split(train_pool: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    min_rows = seq_len + pred_len + 1
    if len(train_pool) < 2 * min_rows:
        return train_pool, train_pool
    val_rows = max(min_rows, int(len(train_pool) * 0.15))
    if len(train_pool) - val_rows < min_rows:
        return train_pool, train_pool
    return train_pool[:-val_rows], train_pool[-val_rows:]


def calibration_split(train_pool: np.ndarray, ratio: float, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    ratio = float(ratio)
    if ratio <= 0:
        return train_pool[:0], train_pool[:0]
    min_rows = seq_len + pred_len + 1
    rows = int(len(train_pool) * ratio)
    rows = max(rows, min_rows)
    rows = min(rows, len(train_pool))
    subset = train_pool[:rows]
    if rows < min_rows:
        return subset[:0], subset[:0]
    if rows < 2 * min_rows:
        return subset, subset[:0]
    val_rows = max(min_rows, int(rows * 0.15))
    if rows - val_rows < min_rows:
        val_rows = 0
    train_rows = rows - val_rows
    if val_rows <= 0:
        return subset, subset[:0]
    return subset[:train_rows], subset[train_rows:]


def build_model(model_name: str, input_dim: int, args: argparse.Namespace) -> nn.Module:
    model_key = model_name.lower()
    if model_key == "dlinear":
        return DLinear(args.seq_len, args.pred_len, input_dim)
    if model_key == "itransformer":
        return iTransformer(
            {
                "input_dim": input_dim,
                "seq_len": args.seq_len,
                "pred_len": args.pred_len,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            }
        )
    if model_key == "ultra_lsnt":
        cfg = SotaPushConfig(
            input_dim=input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            hidden_dim=args.ultra_hidden_dim,
            num_heads=args.ultra_heads,
            num_blocks=args.ultra_blocks,
            num_experts=args.ultra_experts,
            top_k=args.ultra_top_k,
            dropout=args.dropout,
        )
        return SotaUltraLSNTForecaster(cfg)
    raise ValueError(f"Unsupported model: {model_name}")


def forward_model(model: nn.Module, model_name: str, x: torch.Tensor) -> torch.Tensor:
    model_key = model_name.lower()
    if model_key == "dlinear":
        return model(x)
    if model_key == "itransformer":
        pred, _ = model(x)
        return pred.squeeze(-1)
    if model_key == "ultra_lsnt":
        pred, _ = model(x, temperature=0.35)
        return pred
    raise ValueError(model_name)


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    max_train_batches: int,
    max_eval_batches: int,
) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad = 0
    started = time.time()
    history = []
    for epoch in range(max(0, epochs)):
        model.train()
        train_losses = []
        for batch_idx, (bx, by) in enumerate(train_loader):
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = forward_model(model, model_name, bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx, (bx, by) in enumerate(val_loader):
                if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                    break
                bx = bx.to(device)
                by = by.to(device)
                pred = forward_model(model, model_name, bx)
                val_losses.append(float(criterion(pred, by).item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)) if train_losses else math.nan, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state, strict=False)
    return {"best_val_loss": best_val, "epochs_ran": len(history), "train_seconds": time.time() - started, "history": history}


def evaluate_model(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    scaler: StandardScaler,
    device: torch.device,
    max_eval_batches: int,
) -> Dict[str, float]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_idx, (bx, by) in enumerate(loader):
            if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                break
            bx = bx.to(device)
            pred = forward_model(model, model_name, bx)
            preds.append(pred.detach().cpu().numpy())
            trues.append(by.numpy())
    if not preds:
        return {"R2": math.nan, "RMSE": math.nan, "MAE": math.nan}
    p = np.concatenate(preds, axis=0)
    y = np.concatenate(trues, axis=0)
    p = p * scaler.std[-1] + scaler.mean[-1]
    y = y * scaler.std[-1] + scaler.mean[-1]
    metrics = compute_metrics(p, y)
    return {k: float(v) for k, v in metrics.items() if k in {"R2", "RMSE", "MAE"}}


def save_checkpoint(path: Path, model: nn.Module, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **payload}, path)


def load_checkpoint(path: Path) -> Dict[str, object]:
    return torch.load(path, map_location="cpu")


def append_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")


def existing_keys(path: Path, key_cols: Iterable[str]) -> set[Tuple[str, ...]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return {tuple(str(row[col]) for col in key_cols) for _, row in df.iterrows()}


def aggregate_outputs(out_dir: Path) -> None:
    curve_path = out_dir / "cross_site_recalibration_curve.csv"
    zero_path = out_dir / "cross_site_zero_shot_matrix.csv"
    zero_runs_path = out_dir / "cross_site_zero_shot_runs.csv"
    summary_path = out_dir / "cross_site_summary_for_paper.csv"
    if not curve_path.exists():
        return
    curve = pd.read_csv(curve_path)
    zero = curve[curve["calibration_ratio"].astype(float) == 0.0].copy()
    zero.to_csv(zero_runs_path, index=False, encoding="utf-8-sig")
    zero_summary = (
        zero.groupby(["model", "source_dataset", "target_dataset"], dropna=False)
        .agg(
            mean_R2=("R2", "mean"),
            std_R2=("R2", "std"),
            mean_RMSE=("RMSE", "mean"),
            std_RMSE=("RMSE", "std"),
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            runs=("R2", "count"),
            errors=("error", lambda s: int(s.fillna("").astype(str).str.len().gt(0).sum())),
        )
        .reset_index()
        .sort_values(["model", "source_dataset", "target_dataset"])
    )
    zero_summary.to_csv(zero_path, index=False, encoding="utf-8-sig")
    if curve.empty:
        return
    sort_cols = ["model", "source_dataset", "target_dataset", "seed", "calibration_ratio"]
    curve = curve.sort_values(sort_cols)
    group_cols = ["model", "source_dataset", "target_dataset", "calibration_ratio"]
    summary = (
        curve.groupby(group_cols, dropna=False)
        .agg(
            mean_R2=("R2", "mean"),
            std_R2=("R2", "std"),
            mean_RMSE=("RMSE", "mean"),
            mean_MAE=("MAE", "mean"),
            runs=("R2", "count"),
            errors=("error", lambda s: int(s.fillna("").astype(str).str.len().gt(0).sum())),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")


def run_experiment(args: argparse.Namespace) -> int:
    set_seed(args.seed_base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = parse_csv_list(args.datasets, DEFAULT_DATASETS)
    models = parse_csv_list(args.models, DEFAULT_MODELS)
    seeds = parse_int_list(args.seeds, DEFAULT_SEEDS)
    ratios = parse_float_list(args.calibration_ratios, DEFAULT_RATIOS)

    if args.smoke:
        seeds = seeds[:1]
        ratios = [0.0, 0.01]
        args.epochs = min(args.epochs, 1)
        args.recalibration_epochs = min(args.recalibration_epochs, 1)
        args.max_train_batches = args.max_train_batches or 2
        args.max_eval_batches = args.max_eval_batches or 2

    manifest = {
        "datasets": datasets,
        "models": models,
        "seeds": seeds,
        "calibration_ratios": ratios,
        "args": vars(args),
        "device": args.device if not TORCH_READY else str(pick_device(args.device)),
        "torch_ready": TORCH_READY,
        "torch_import_error": "" if TORCH_READY else str(TORCH_IMPORT_ERROR),
    }
    (out_dir / "cross_site_experiment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    bundles = {alias: load_bundle(alias, "power", args.train_ratio) for alias in datasets}
    if args.check_only:
        rows = []
        for alias, bundle in bundles.items():
            rows.append(
                {
                    "alias": alias,
                    "resolved_path": bundle.resolved_path,
                    "rows": int(len(bundle.data)),
                    "input_dim": int(bundle.input_dim),
                    "train_pool_rows": int(len(bundle.train_pool_scaled)),
                    "test_rows": int(len(bundle.test_scaled)),
                    "train_windows": int(len(WindowDataset(bundle.train_pool_scaled, args.seq_len, args.pred_len))),
                    "test_windows": int(len(WindowDataset(bundle.test_scaled, args.seq_len, args.pred_len))),
                }
            )
        check_path = out_dir / "cross_site_check_only.csv"
        pd.DataFrame(rows).to_csv(check_path, index=False, encoding="utf-8-sig")
        print(f"[check-only] {check_path}")
        return 0
    if not TORCH_READY:
        raise RuntimeError(f"PyTorch is required for training/evaluation: {TORCH_IMPORT_ERROR}")
    device = pick_device(args.device)
    curve_path = out_dir / "cross_site_recalibration_curve.csv"
    done = existing_keys(curve_path, ["model", "source_dataset", "target_dataset", "seed", "calibration_ratio"]) if args.resume else set()
    ckpt_dir = out_dir / "checkpoints"

    for seed in seeds:
        set_seed(seed)
        for model_name in models:
            for source_alias, source_bundle in bundles.items():
                source_ckpt = ckpt_dir / f"{model_name}__{source_alias}__seed{seed}.pt"
                try:
                    if source_ckpt.exists() and args.resume:
                        source_payload = load_checkpoint(source_ckpt)
                        source_state = source_payload["state_dict"]
                        source_train_report = source_payload.get("train_report", {})
                    else:
                        model = build_model(model_name, source_bundle.input_dim, args).to(device)
                        train_data, val_data = source_train_val_split(source_bundle.train_pool_scaled, args.seq_len, args.pred_len)
                        train_loader = make_loader(train_data, args.seq_len, args.pred_len, args.batch_size, True, seed, args.num_workers)
                        val_loader = make_loader(val_data, args.seq_len, args.pred_len, args.batch_size, False, seed, args.num_workers)
                        source_train_report = train_model(
                            model,
                            model_name,
                            train_loader,
                            val_loader,
                            device,
                            epochs=args.epochs,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            patience=args.patience,
                            max_train_batches=args.max_train_batches,
                            max_eval_batches=args.max_eval_batches,
                        )
                        save_checkpoint(
                            source_ckpt,
                            model,
                            {
                                "model": model_name,
                                "source_dataset": source_alias,
                                "seed": seed,
                                "input_dim": source_bundle.input_dim,
                                "train_report": source_train_report,
                            },
                        )
                        source_state = copy.deepcopy(model.state_dict())
                except Exception as exc:
                    for target_alias in bundles:
                        for ratio in ratios:
                            key = (model_name, source_alias, target_alias, str(seed), str(float(ratio)))
                            if key in done:
                                continue
                            error_row = {
                                "model": model_name,
                                "source_dataset": source_alias,
                                "target_dataset": target_alias,
                                "is_within_site": bool(source_alias == target_alias),
                                "is_cross_site": bool(source_alias != target_alias),
                                "seed": seed,
                                "calibration_ratio": float(ratio),
                                "R2": math.nan,
                                "RMSE": math.nan,
                                "MAE": math.nan,
                                "train_seconds": math.nan,
                                "epochs_ran": 0,
                                "source_train_seconds": math.nan,
                                "error": f"source_train_failed: {type(exc).__name__}: {exc}",
                            }
                            append_row(curve_path, error_row)
                            done.add(key)
                    aggregate_outputs(out_dir)
                    continue

                for target_alias, target_bundle in bundles.items():
                    for ratio in ratios:
                        key = (model_name, source_alias, target_alias, str(seed), str(float(ratio)))
                        if key in done:
                            continue
                        row = {
                            "model": model_name,
                            "source_dataset": source_alias,
                            "target_dataset": target_alias,
                            "is_within_site": bool(source_alias == target_alias),
                            "is_cross_site": bool(source_alias != target_alias),
                            "seed": seed,
                            "calibration_ratio": float(ratio),
                            "R2": math.nan,
                            "RMSE": math.nan,
                            "MAE": math.nan,
                            "train_seconds": math.nan,
                            "epochs_ran": 0,
                            "source_train_seconds": source_train_report.get("train_seconds", math.nan) if isinstance(source_train_report, dict) else math.nan,
                            "error": "",
                        }
                        try:
                            target_model = build_model(model_name, target_bundle.input_dim, args).to(device)
                            if model_name == "ultra_lsnt":
                                load_report = partial_load_state_dict(target_model, source_state)
                                row["warmstart_matched_keys"] = load_report["matched"]
                                row["warmstart_skipped_keys"] = load_report["skipped"]
                            else:
                                target_model.load_state_dict(source_state, strict=True)
                                row["warmstart_matched_keys"] = len(source_state)
                                row["warmstart_skipped_keys"] = 0

                            if float(ratio) > 0:
                                cal_train, cal_val = calibration_split(target_bundle.train_pool_scaled, float(ratio), args.seq_len, args.pred_len)
                                train_loader = make_loader(cal_train, args.seq_len, args.pred_len, args.batch_size, True, seed, args.num_workers)
                                val_loader = make_loader(cal_val if len(cal_val) else cal_train, args.seq_len, args.pred_len, args.batch_size, False, seed, args.num_workers)
                                report = train_model(
                                    target_model,
                                    model_name,
                                    train_loader,
                                    val_loader,
                                    device,
                                    epochs=args.recalibration_epochs,
                                    lr=args.recalibration_lr,
                                    weight_decay=args.weight_decay,
                                    patience=args.recalibration_patience,
                                    max_train_batches=args.max_train_batches,
                                    max_eval_batches=args.max_eval_batches,
                                )
                                row["train_seconds"] = report["train_seconds"]
                                row["epochs_ran"] = report["epochs_ran"]

                            test_loader = make_loader(target_bundle.test_scaled, args.seq_len, args.pred_len, args.batch_size, False, seed, args.num_workers)
                            metrics = evaluate_model(target_model, model_name, test_loader, target_bundle.scaler, device, args.max_eval_batches)
                            row.update(metrics)
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        append_row(curve_path, row)
                        done.add(key)
                        print(
                            f"[row] model={model_name} source={source_alias} target={target_alias} "
                            f"seed={seed} ratio={ratio} R2={row['R2']} error={row['error']}"
                        )
                        aggregate_outputs(out_dir)
    aggregate_outputs(out_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-site zero-shot and recalibration experiments.")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", type=str, default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--calibration-ratios", type=str, default=",".join(str(x) for x in DEFAULT_RATIOS))
    parser.add_argument("--out-dir", type=str, default=str(OUT_ROOT))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate dataset loading and split/window sizes without training.")

    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--recalibration-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--recalibration-patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--recalibration-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=42)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ultra-hidden-dim", type=int, default=128)
    parser.add_argument("--ultra-heads", type=int, default=4)
    parser.add_argument("--ultra-blocks", type=int, default=2)
    parser.add_argument("--ultra-experts", type=int, default=4)
    parser.add_argument("--ultra-top-k", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_experiment(parse_args()))
