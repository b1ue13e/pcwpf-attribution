#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

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


DEFAULT_OUT_CSV = ROOT / "results" / "supplementary_evidence" / "moirai_family_windcn.csv"
DEFAULT_META_JSON = ROOT / "results" / "supplementary_evidence" / "moirai_family_windcn.meta.json"
DEFAULT_MODEL_ID = "Salesforce/moirai-2.0-R-small"
MODEL_LABEL = "Moirai-2.0-R-Small"


def _filter_supported_kwargs(callable_obj, **kwargs):
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    accepted = {}
    for key, value in kwargs.items():
        if key in sig.parameters:
            accepted[key] = value
    return accepted


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
    ap = argparse.ArgumentParser(description="Audit Moirai family robustness on Wind (CN) with the audited 80/20 protocol.")
    ap.add_argument("--data", type=str, default="wind_final.csv")
    ap.add_argument("--target", type=str, default="power")
    ap.add_argument("--seq-len", type=int, default=96, help="Standard protocol seq_len (for target alignment)")
    ap.add_argument("--model-seq-len", type=int, default=512, help="Context length for Moirai input (reasonable usage)")
    ap.add_argument("--pred-len", type=int, default=24)
    ap.add_argument("--sigma-cfgs", type=str, default="0.0,0.1,0.4")
    ap.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--freq", type=str, default="15min")
    ap.add_argument("--max-test-windows", type=int, default=0)
    ap.add_argument("--out-csv", type=str, default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--meta-json", type=str, default=str(DEFAULT_META_JSON))
    ap.add_argument("--smoke", action="store_true")
    return ap


def _build_predictor(model_id: str, pred_len: int, seq_len: int, batch_size: int, num_samples: int):
    try:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module  # type: ignore

        forecast_kwargs = _filter_supported_kwargs(
            Moirai2Forecast,
            module=Moirai2Module.from_pretrained(model_id),
            prediction_length=pred_len,
            context_length=seq_len,
            patch_size="auto",
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        model = Moirai2Forecast(**forecast_kwargs)
        predictor_kwargs = _filter_supported_kwargs(model.create_predictor, batch_size=batch_size)
        return model.create_predictor(**predictor_kwargs), "moirai2"
    except Exception:
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # type: ignore
            # Fallback for Moirai 1.x if 2.0 fails or is different
            # But we default to 2.0
            raise ImportError("Moirai 1.x fallback not fully implemented for this audit script.")
        except ImportError as exc:
            raise RuntimeError(
                "Moirai audit requires `uni2ts` plus GluonTS-compatible dependencies in the target environment."
            ) from exc

def main() -> int:
    args = build_parser().parse_args()
    sigma_cfgs = parse_sigma_cfgs(args.sigma_cfgs)
    if args.smoke and args.max_test_windows <= 0:
        args.max_test_windows = 32

    data, feature_names, target_idx = load_windcn_matrix(data_path=args.data, target=args.target)
    
    # 1. Standard Protocol Alignment
    train_end, test_end = read_split_from_manifest(len(data), dataset_name="wind_cn")
    
    # Z-score normalization (using ONLY training data stats)
    train_data = data[:train_end]
    stats = fit_zscore(train_data.reshape(-1, 1) if train_data.ndim==1 else train_data)
    data_scaled = ((data[:, target_idx] - float(stats.mean[target_idx])) / float(stats.std[target_idx])).astype(np.float32)
    
    # 2. Extended Context Construction (Same logic as Chronos)
    test_len = test_end - train_end
    num_windows = test_len - args.seq_len - args.pred_len + 1
    
    if num_windows <= 0:
        raise ValueError("Test set too small for standard protocol")
        
    if args.max_test_windows > 0:
        num_windows = min(num_windows, args.max_test_windows)
        
    print(f"Generating {num_windows} windows with context_len={args.model_seq_len} (Standard={args.seq_len})...")
    
    x_contexts = [] # For Moirai/GluonTS, usually list of dicts or pandas
    y_trues = []
    
    # GluonTS ListDataset format
    # But uni2ts predictor.predict takes `dataset`
    from gluonts.dataset.common import ListDataset
    
    # We will build input items
    dataset_list = []
    
    for i in range(num_windows):
        pred_start_idx = train_end + args.seq_len + i
        ctx_start = pred_start_idx - args.model_seq_len
        ctx_end = pred_start_idx
        tgt_end = pred_start_idx + args.pred_len
        
        if ctx_start < 0:
             raise ValueError(f"Not enough history for context length {args.model_seq_len} at index {pred_start_idx}")
            
        x_window = data_scaled[ctx_start:ctx_end]
        y_window = data_scaled[ctx_end:tgt_end]
        
        y_trues.append(y_window)
        
        # Moirai input: target (1, context_length) or (context_length,)
        # We need to provide the target field.
        # Note: Moirai expects shape (target_dim, seq_len) if multivariate, or (seq_len,) if univariate?
        # Uni2TS MoiraiForecast uses target_dim=1.
        # Let's check typical usage. Usually (C, T).
        # GluonTS ListDataset checks shape. If 1D, it expects (T,).
        # We have x_window as (512,).
        dataset_list.append({
            "target": x_window, # (512,) 
            "start": "2020-01-01 00:00:00", # Dummy start
            "feat_dynamic_real": None,
            "past_feat_dynamic_real": None
        })
        
    # Convert y_trues to array for metric calc
    y_test = np.stack(y_trues)
    
    # Load Predictor
    predictor, backend_name = _build_predictor(
        args.model_id, 
        args.pred_len, 
        args.model_seq_len, 
        args.batch_size, 
        args.num_samples
    )
    
    run_id = utc_run_id("_moirai_family_windcn")
    rows = []
    y_true_denorm = inverse_target_scale(y_test, stats, target_idx=0)

    for sigma_cfg in sigma_cfgs:
        # Noise Injection
        if np.isclose(sigma_cfg, 0.0):
            # No noise
            current_dataset = ListDataset(dataset_list, freq=args.freq)
        else:
            # Inject noise into x_window inside dataset_list
            from windcn_audit_common import inject_gaussian_windows
            noisy_list = []
            for item in dataset_list:
                # item['target'] is (512,) numpy
                x_clean = item['target']
                # inject_gaussian_windows expects (N, T) or (N, T, 1).
                # We have single sample (512,).
                # Reshape to (1, 512) for injection, then back to (512,)
                x_noisy = inject_gaussian_windows(x_clean.reshape(1, -1), sigma_cfg).flatten() 
                noisy_item = item.copy()
                noisy_item['target'] = x_noisy
                noisy_list.append(noisy_item)
            current_dataset = ListDataset(noisy_list, freq=args.freq)

        forecast_it = predictor.predict(current_dataset)
        
        # Collect predictions
        preds = []
        for forecast in forecast_it:
            # forecast.median is (pred_len,) or (pred_len, target_dim)?
            # GluonTS forecast objects.
            # We want the median.
            # If num_samples > 0, we can compute median from samples.
            if hasattr(forecast, "median"):
                p = forecast.median
            elif hasattr(forecast, "quantile"):
                p = forecast.quantile(0.5)
            else:
                p = np.median(forecast.samples, axis=0)
            
            # p should be (24,)
            if p.ndim > 1:
                p = p.flatten()
            preds.append(p)
            
        pred_scaled = np.stack(preds) # (N, 24)
        
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
        row["backend"] = f"uni2ts-{backend_name}"
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
        "backend": f"uni2ts-{backend_name}",
        "model_id": args.model_id,
        "model_label": MODEL_LABEL,
        "dataset": args.data,
        "target": args.target,
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
