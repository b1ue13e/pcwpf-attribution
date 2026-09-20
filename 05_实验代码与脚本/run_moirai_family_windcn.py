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
    ap.add_argument("--seq-len", type=int, default=96, help="Standard protocol seq_len used to align the prediction target window.")
    ap.add_argument("--model-seq-len", type=int, default=512, help="Extended context length used by the Moirai reference model.")
    ap.add_argument("--pred-len", type=int, default=24)
    ap.add_argument("--sigma-cfgs", type=str, default="0.0,0.1,0.4")
    ap.add_argument("--input-scale", type=str, choices=("zscore", "raw"), default="zscore", help="Model input domain. `raw` keeps physical units; `zscore` preserves the original audit protocol.")
    ap.add_argument("--raw-noise-anchor", type=str, choices=("protocol_equivalent", "raw_unit"), default="protocol_equivalent", help="When --input-scale=raw, inject Gaussian noise either in z-score-equivalent raw units or in literal raw units.")
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
        except ImportError as exc:
            raise RuntimeError(
                "Moirai audit requires `uni2ts` plus GluonTS-compatible dependencies in the target environment."
            ) from exc

        forecast_kwargs = _filter_supported_kwargs(
            MoiraiForecast,
            module=MoiraiModule.from_pretrained(model_id),
            prediction_length=pred_len,
            context_length=seq_len,
            patch_size="auto",
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        model = MoiraiForecast(**forecast_kwargs)
        predictor_kwargs = _filter_supported_kwargs(model.create_predictor, batch_size=batch_size)
        return model.create_predictor(**predictor_kwargs), "moirai"


def _build_dataset(contexts: np.ndarray, freq: str):
    try:
        from gluonts.dataset.common import ListDataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Moirai audit requires `gluonts` in the target environment.") from exc

    start = pd.Period("2022-03-12 16:00", freq=freq)
    records = []
    for idx, context in enumerate(contexts):
        records.append(
            {
                "item_id": f"windcn-{idx}",
                "start": start,
                "target": np.asarray(context, dtype=np.float32),
            }
        )
    return ListDataset(records, freq=freq)


def _extract_forecast_mean(forecasts) -> np.ndarray:
    preds = []
    for forecast in forecasts:
        if hasattr(forecast, "mean"):
            mean_attr = forecast.mean
            preds.append(np.asarray(mean_attr() if callable(mean_attr) else mean_attr, dtype=np.float32))
        elif hasattr(forecast, "samples"):
            preds.append(np.asarray(forecast.samples, dtype=np.float32).mean(axis=0))
        else:
            preds.append(np.asarray(forecast, dtype=np.float32))
    arr = np.stack(preds, axis=0)
    if arr.ndim == 3:
        arr = arr.mean(axis=1)
    return arr.astype(np.float32)


def _inject_contexts(x: np.ndarray, sigma_cfg: float, noise_scale: float = 1.0) -> np.ndarray:
    if np.isclose(sigma_cfg, 0.0):
        return x
    from windcn_audit_common import inject_gaussian_windows

    return inject_gaussian_windows(x, sigma_cfg, noise_scale=noise_scale)


def main() -> int:
    args = build_parser().parse_args()
    sigma_cfgs = parse_sigma_cfgs(args.sigma_cfgs)
    if args.smoke and args.max_test_windows <= 0:
        args.max_test_windows = 32

    data, feature_names, target_idx = load_windcn_matrix(data_path=args.data, target=args.target)
    train_end, test_end = read_split_from_manifest(len(data), dataset_name="wind_cn")

    train_data = data[:train_end]
    stats = fit_zscore(train_data.reshape(-1, 1) if train_data.ndim == 1 else train_data)
    target_mean = float(stats.mean[target_idx])
    target_std = float(stats.std[target_idx])
    if args.input_scale == "raw":
        data_model = data[:, target_idx].astype(np.float32)
        noise_scale = target_std if args.raw_noise_anchor == "protocol_equivalent" else 1.0
    else:
        data_model = ((data[:, target_idx] - target_mean) / target_std).astype(np.float32)
        noise_scale = 1.0

    test_len = test_end - train_end
    num_windows = test_len - args.seq_len - args.pred_len + 1
    if num_windows <= 0:
        raise ValueError("Test set too small for standard protocol")
    if args.max_test_windows > 0:
        num_windows = min(num_windows, args.max_test_windows)

    x_contexts = []
    y_trues = []
    for i in range(num_windows):
        pred_start_idx = train_end + args.seq_len + i
        ctx_start = pred_start_idx - args.model_seq_len
        ctx_end = pred_start_idx
        tgt_end = pred_start_idx + args.pred_len
        if ctx_start < 0:
            raise ValueError(f"Not enough history for context length {args.model_seq_len} at index {pred_start_idx}")
        x_contexts.append(data_model[ctx_start:ctx_end])
        y_trues.append(data_model[ctx_end:tgt_end])

    x_test = np.stack(x_contexts)
    y_test = np.stack(y_trues)

    predictor, backend_name = _build_predictor(
        model_id=args.model_id,
        pred_len=args.pred_len,
        seq_len=args.model_seq_len,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )

    run_id = utc_run_id("_moirai_family_windcn")
    rows = []
    if args.input_scale == "raw":
        y_true_eval = y_test
    else:
        y_true_eval = inverse_target_scale(y_test, stats, target_idx=target_idx)

    for sigma_cfg in sigma_cfgs:
        x_eval = _inject_contexts(x_test, sigma_cfg, noise_scale=noise_scale)
        dataset = _build_dataset(x_eval, args.freq)
        forecasts = list(predictor.predict(dataset))
        pred_scaled = _extract_forecast_mean(forecasts)
        if args.input_scale == "raw":
            pred_eval = pred_scaled
        else:
            pred_eval = inverse_target_scale(pred_scaled, stats, target_idx=target_idx)
        row = audit_row(
            model=MODEL_LABEL,
            scenario=scenario_name(sigma_cfg),
            sigma_cfg=sigma_cfg,
            pred_denorm=pred_eval,
            true_denorm=y_true_eval,
            run_id=run_id,
        )
        row["model_label"] = MODEL_LABEL
        row["backend"] = "uni2ts"
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
            "run_id",
        ],
    )

    meta = {
        "run_id": run_id,
        "script": Path(__file__).name,
        "backend": "uni2ts",
        "backend_version": importlib.metadata.version("uni2ts"),
        "model_id": args.model_id,
        "model_label": MODEL_LABEL,
        "resolved_backend": backend_name,
        "dataset": args.data,
        "target": args.target,
        "input_mode": "univariate_target_only",
        "input_scale": args.input_scale,
        "raw_noise_anchor": args.raw_noise_anchor,
        "noise_units": "zscore" if args.input_scale == "zscore" else ("raw target units (z-score-equivalent)" if args.raw_noise_anchor == "protocol_equivalent" else "raw target units"),
        "split_policy": "chronological_80_20 (standard target alignment + extended context)",
        "standard_seq_len": args.seq_len,
        "model_seq_len": args.model_seq_len,
        "pred_len": args.pred_len,
        "sigma_cfgs": sigma_cfgs,
        "sigma_eff_formula": "sigma_eff = 1.5 * sigma_cfg",
        "point_forecast_source": "forecast mean",
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "freq": args.freq,
        "test_windows": int(len(x_test)),
        "feature_names_reference": feature_names,
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
