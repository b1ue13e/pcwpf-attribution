#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from windcn_audit_common import (
    ROOT,
    audit_row,
    fit_zscore,
    inverse_target_scale,
    load_windcn_matrix,
    make_univariate_windows,
    maybe_limit_windows,
    read_split_from_manifest,
    scenario_name,
    utc_run_id,
    write_csv,
)


DEFAULT_OUT_CSV = ROOT / "results" / "supplementary_evidence" / "chronos_family_windcn.csv"
DEFAULT_META_JSON = ROOT / "results" / "supplementary_evidence" / "chronos_family_windcn.meta.json"
DEFAULT_MODEL_ID = "amazon/chronos-bolt-small"
MODEL_LABEL = "Chronos-Bolt-Small"


def parse_sigma_cfgs(raw: str) -> list[float]:
    vals = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise ValueError("sigma-cfgs must not be empty")
    return vals


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Audit Chronos family robustness on Wind (CN) with the audited 80/20 protocol.")
    ap.add_argument("--data", type=str, default="wind_final.csv")
    ap.add_argument("--target", type=str, default="power")
    ap.add_argument("--seq-len", type=int, default=96, help="Standard protocol seq_len (for target alignment)")
    ap.add_argument("--model-seq-len", type=int, default=512, help="Context length for Chronos input (reasonable usage)")
    ap.add_argument("--pred-len", type=int, default=24)
    ap.add_argument("--sigma-cfgs", type=str, default="0.0,0.4") # Only run Clean and Severe 0.60 (0.4*1.5)
    ap.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--max-test-windows", type=int, default=0)
    ap.add_argument("--out-csv", type=str, default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--meta-json", type=str, default=str(DEFAULT_META_JSON))
    ap.add_argument("--smoke", action="store_true")
    return ap


def _load_pipeline(model_id: str, device_map: str):
    try:
        from chronos import BaseChronosPipeline  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Chronos audit requires `chronos-forecasting`. Install it in the target environment."
        ) from exc

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {"device_map": device_map, "torch_dtype": torch_dtype}
    return BaseChronosPipeline.from_pretrained(model_id, **kwargs)


def _as_context_list(batch_contexts: np.ndarray) -> List[torch.Tensor]:
    return [torch.tensor(ctx, dtype=torch.float32) for ctx in batch_contexts]


def _normalize_prediction_output(pred) -> np.ndarray:
    if isinstance(pred, torch.Tensor):
        arr = pred.detach().cpu().numpy()
    elif isinstance(pred, np.ndarray):
        arr = pred
    elif isinstance(pred, dict):
        for key in ("median", "mean", "quantiles", "samples"):
            if key in pred:
                return _normalize_prediction_output(pred[key])
        raise TypeError(f"unsupported dict prediction keys: {list(pred.keys())}")
    elif isinstance(pred, list):
        stacked = []
        for item in pred:
            if hasattr(item, "mean"):
                mean_attr = item.mean
                stacked.append(np.asarray(mean_attr() if callable(mean_attr) else mean_attr, dtype=np.float32))
            elif hasattr(item, "samples"):
                samples = np.asarray(item.samples, dtype=np.float32)
                stacked.append(np.median(samples, axis=0))
            else:
                stacked.append(np.asarray(item, dtype=np.float32))
        arr = np.stack(stacked, axis=0)
    else:
        raise TypeError(f"unsupported prediction type: {type(pred)!r}")

    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 3:
        arr = np.median(arr, axis=1)
    return np.asarray(arr, dtype=np.float32)


def _predict_batch(pipeline, batch_contexts: np.ndarray, pred_len: int, num_samples: int) -> np.ndarray:
    contexts = _as_context_list(batch_contexts)
    predict_fn = getattr(pipeline, "predict", None)
    if predict_fn is None:
        raise RuntimeError("Chronos pipeline does not expose a predict method.")

    attempts = [
        {"inputs": contexts, "prediction_length": pred_len, "limit_prediction_length": False, "num_samples": num_samples},
        {"inputs": contexts, "prediction_length": pred_len, "limit_prediction_length": False}, # num_samples might not be supported by Bolt pipeline?
        {"context": contexts, "prediction_length": pred_len, "num_samples": num_samples},
        {"contexts": contexts, "prediction_length": pred_len, "num_samples": num_samples},
        {"inputs": torch.stack(contexts, dim=0), "prediction_length": pred_len, "limit_prediction_length": False, "num_samples": num_samples},
        {"inputs": torch.stack(contexts, dim=0), "prediction_length": pred_len, "limit_prediction_length": False},
    ]
    last_exc = None
    for kwargs in attempts:
        try:
            return _normalize_prediction_output(predict_fn(**kwargs))
        except TypeError as exc:
            last_exc = exc

    signature = inspect.signature(predict_fn)
    raise RuntimeError(f"unable to call Chronos predict with supported argument layouts: {signature}") from last_exc


def _inject_contexts(x: np.ndarray, sigma_cfg: float) -> np.ndarray:
    if np.isclose(sigma_cfg, 0.0):
        return x
    from windcn_audit_common import inject_gaussian_windows

    return inject_gaussian_windows(x, sigma_cfg)


def main() -> int:
    args = build_parser().parse_args()
    sigma_cfgs = parse_sigma_cfgs(args.sigma_cfgs)
    if args.smoke and args.max_test_windows <= 0:
        args.max_test_windows = 32

    data, feature_names, target_idx = load_windcn_matrix(data_path=args.data, target=args.target)
    # data is (T, C)
    
    # 1. Standard Protocol Alignment
    # We need to know EXACTLY which time steps are predicted in the standard protocol (seq_len=96).
    train_end, test_end = read_split_from_manifest(len(data), dataset_name="wind_cn")
    
    # Standard test set: data[train_end:test_end]
    # Standard sliding window: 
    #   Starts at index 0 of test set.
    #   First window x: test_data[0:96], y: test_data[96:96+24]
    #   Global index of first y start: train_end + 96
    
    # 2. Extended Context Construction
    # We want y to be the same.
    # So for the first sample, y starts at train_end + 96.
    # We want x to end at train_end + 96.
    # We want x to have length `args.model_seq_len` (e.g. 512).
    # So x should range from [train_end + 96 - 512 : train_end + 96].
    
    # We need to extract these windows from the full `data` array (z-scored).
    
    # Z-score normalization (using ONLY training data stats)
    train_data = data[:train_end]
    stats = fit_zscore(train_data.reshape(-1, 1) if train_data.ndim==1 else train_data) # Use full train data for stats
    
    # We need to normalize the RELEVANT part of data.
    # Ideally normalize the whole thing, but strict protocol says fit on train.
    # apply_zscore applies (x - train_mean) / train_std.
    data_scaled = ((data[:, target_idx] - float(stats.mean[target_idx])) / float(stats.std[target_idx])).astype(np.float32)
    
    # Define the range of prediction start indices (global index where y starts)
    # In standard protocol (seq_len=96):
    # test_data = data_scaled[train_end:test_end]
    # max_start = len(test_data) - 96 - 24 + 1
    # Global y_start indices: train_end + 96 + i, for i in 0..max_start-1
    
    test_len = test_end - train_end
    num_windows = test_len - args.seq_len - args.pred_len + 1
    
    if num_windows <= 0:
        raise ValueError("Test set too small for standard protocol")
        
    if args.max_test_windows > 0:
        num_windows = min(num_windows, args.max_test_windows)
        
    print(f"Generating {num_windows} windows with context_len={args.model_seq_len} (Standard={args.seq_len})...")
    
    x_contexts = []
    y_trues = []
    
    for i in range(num_windows):
        # Global index where prediction starts (first step of y)
        pred_start_idx = train_end + args.seq_len + i
        
        # Context range: [pred_start_idx - model_seq_len : pred_start_idx]
        ctx_start = pred_start_idx - args.model_seq_len
        ctx_end = pred_start_idx
        
        # Target range: [pred_start_idx : pred_start_idx + pred_len]
        tgt_end = pred_start_idx + args.pred_len
        
        if ctx_start < 0:
            # Padding if not enough history (unlikely for 512, but possible for very early train)
            # But we are in test set, so train_end is usually large enough.
            # train_end is 80% of 120k ~ 96k. 512 is fine.
            raise ValueError(f"Not enough history for context length {args.model_seq_len} at index {pred_start_idx}")
            
        x_window = data_scaled[ctx_start:ctx_end]
        y_window = data_scaled[ctx_end:tgt_end]
        
        x_contexts.append(x_window)
        y_trues.append(y_window)
        
    x_test = np.stack(x_contexts)
    y_test = np.stack(y_trues)
    
    # Now we have x_test (N, 512) and y_test (N, 24)
    # y_test matches the standard protocol targets exactly.
    # x_test provides extended context.

    pipeline = _load_pipeline(args.model_id, args.device_map)

    run_id = utc_run_id("_chronos_family_windcn")
    rows = []
    y_true_denorm = inverse_target_scale(y_test, stats, target_idx=0) # stats is from train_data fit

    for sigma_cfg in sigma_cfgs:
        # Note: Noise injection should ideally apply to the WHOLE context or just the 'standard' part?
        # Standard protocol injects noise into the input window.
        # Here input is 512. We should inject noise into the whole 512 to be consistent (sensor is noisy).
        x_eval = _inject_contexts(x_test, sigma_cfg)
        
        preds = []
        # Batch prediction
        for start in range(0, len(x_eval), args.batch_size):
            batch = x_eval[start : start + args.batch_size]
            preds.append(_predict_batch(pipeline, batch, pred_len=args.pred_len, num_samples=args.num_samples))
            
        pred_scaled = np.concatenate(preds, axis=0)
        # Chronos output is usually (N, num_samples, H) or (N, H).
        # _normalize_prediction_output handles median.
        
        pred_denorm = inverse_target_scale(pred_scaled, stats, target_idx=0)
        row = audit_row(
            model=MODEL_LABEL,
            scenario=scenario_name(sigma_cfg),
            sigma_cfg=sigma_cfg,
            pred_denorm=pred_denorm,
            true_denorm=y_true_denorm,
            run_id=run_id,
        )
        row["model_label"] = MODEL_LABEL
        row["backend"] = "chronos-forecasting"
        row["model_id"] = args.model_id
        rows.append(row)

    out_csv = write_csv(
        rows,
        args.out_csv,
        columns=[
            "model_label",
            "backend",
            "model_id",
            "model",
            "scenario",
            "sigma_cfg",
            "sigma_eff",
            "R2",
            "RMSE",
            "MAE",
            "residual_gaussian_pvalue",
            "run_id",
        ],
    )

    meta = {
        "run_id": run_id,
        "script": Path(__file__).name,
        "backend": "chronos-forecasting",
        "model_id": args.model_id,
        "model_label": MODEL_LABEL,
        "dataset": args.data,
        "target": args.target,
        "input_mode": "univariate_target_only",
        "split_policy": "chronological_80_20 (Standard Alignment)",
        "standard_seq_len": args.seq_len,
        "model_seq_len": args.model_seq_len,
        "pred_len": args.pred_len,
        "sigma_cfgs": sigma_cfgs,
        "note": "Reasonable Usage: Extended context (512) for TSFM, aligned targets.",
        "output_csv": str(out_csv),
        "smoke": bool(args.smoke),
    }
    meta_path = Path(args.meta_json)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps({"output_csv": str(out_csv), "meta_json": str(meta_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
