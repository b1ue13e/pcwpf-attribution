#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from run_network_constrained_dispatch_ieee24 import load_prediction_array
from wind_dispatch_model import build_dispatch_model, evaluate_cost, get_default_params, parse_solution, solve_dispatch


DEFAULT_OUTPUT_DIR = Path("results") / "ae_dispatch_reserve_ratio_sensitivity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-interface reserve-ratio sensitivity under the copper-plate dispatch screen."
    )
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--reserve-ratio-grid", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--param-mode", type=str, choices=["default", "auto"], default="default")
    parser.add_argument("--w-max", type=float, default=None, help="Optional fixed train-split upper bound in native power units.")
    parser.add_argument(
        "--wind-data-csv",
        type=str,
        default=str(Path("advisor_package_20260311") / "05_data" / "processed" / "wind_final.csv"),
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=str(Path("advisor_package_20260311") / "05_data" / "processed" / "split_manifest_80_20_unified.json"),
    )
    parser.add_argument("--split-dataset-name", type=str, default="wind_cn")
    parser.add_argument("--target-column", type=str, default="power")
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--demand-quantile", type=float, default=0.995)
    parser.add_argument("--gmax-margin", type=float, default=1.20)
    parser.add_argument("--storage-power-ratio", type=float, default=None)
    parser.add_argument("--beta", type=float, default=1.50)
    parser.add_argument("--fuel-cost", type=float, default=0.50)
    parser.add_argument("--startup-cost", type=float, default=5000.0)
    parser.add_argument("--storage-cost", type=float, default=0.05)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--ultra-raw",
        type=str,
        default=str(
            Path("upload_results_ae_operation_20260328_104442")
            / "ae_operation_exports"
            / "windcn_ultra"
            / "seed_42"
            / "prediction_archive_raw_windcn_ultra_seed42.npz"
        ),
    )
    parser.add_argument(
        "--ultra-clip",
        type=str,
        default=str(
            Path("upload_results_ae_operation_20260328_104442")
            / "ae_operation_exports"
            / "windcn_ultra"
            / "seed_42"
            / "prediction_archive_clip_only_windcn_ultra_seed42.npz"
        ),
    )
    parser.add_argument(
        "--ultra-fg",
        type=str,
        default=str(
            Path("upload_results_ae_operation_20260328_104442")
            / "ae_operation_exports"
            / "windcn_ultra"
            / "seed_42"
            / "prediction_archive_full_guardrail_windcn_ultra_seed42.npz"
        ),
    )
    parser.add_argument(
        "--itr-raw",
        type=str,
        default=str(
            Path("upload_results_ae_operation_20260328_104442")
            / "ae_operation_exports"
            / "windcn_itransformer"
            / "seed_42"
            / "prediction_archive_raw_windcn_itransformer_seed42.npz"
        ),
    )
    parser.add_argument(
        "--itr-fg",
        type=str,
        default=str(
            Path("upload_results_ae_operation_20260328_104442")
            / "ae_operation_exports"
            / "windcn_itransformer"
            / "seed_42"
            / "prediction_archive_full_guardrail_windcn_itransformer_seed42.npz"
        ),
    )
    return parser.parse_args()


def parse_grid(raw: str, cast) -> List[float]:
    values = [cast(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("Grid cannot be empty.")
    return values


def resolve_readable_prediction_path(candidates: List[Path]) -> Path:
    errors: List[str] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            _ = load_prediction_array(path)[:1]
            return path
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    raise FileNotFoundError("No readable prediction archive exists:\n" + "\n".join(errors or [str(p) for p in candidates]))


def load_truth_from_archive(path: Path, days: int) -> np.ndarray:
    payload = np.load(path)
    if "ground_truth" not in payload:
        raise KeyError(f"`ground_truth` missing in {path}")
    truth = np.asarray(payload["ground_truth"], dtype=np.float64)
    if truth.ndim != 2 or truth.shape[1] < 24:
        raise ValueError(f"Expected [N, >=24] ground truth in {path}, got {truth.shape}")
    return truth[:days, :24]


def load_models(args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, str]]:
    model_paths = {
        "Ultra-LSNT raw": resolve_readable_prediction_path([Path(args.ultra_raw)]),
        "Ultra-LSNT + clip-only": resolve_readable_prediction_path([Path(args.ultra_clip)]),
        "Ultra-LSNT + FG-MoE": resolve_readable_prediction_path([Path(args.ultra_fg)]),
        "iTransformer raw": resolve_readable_prediction_path([Path(args.itr_raw)]),
        "iTransformer + FG-MoE": resolve_readable_prediction_path([Path(args.itr_fg)]),
    }
    forecasts = {name: load_prediction_array(path)[: args.days] for name, path in model_paths.items()}
    truth = load_truth_from_archive(model_paths["Ultra-LSNT raw"], args.days)
    return forecasts, truth, {k: str(v).replace("\\", "/") for k, v in model_paths.items()}


def infer_wmax(truth: np.ndarray, train_ratio: float, explicit_wmax: float | None) -> Tuple[float, int]:
    if explicit_wmax is not None:
        return float(explicit_wmax), max(1, min(int(len(truth) * float(train_ratio)), len(truth)))
    n_train = max(1, min(int(len(truth) * float(train_ratio)), len(truth)))
    w_max = float(np.nanmax(truth[:n_train]))
    return w_max, n_train


def infer_wmax_from_data(data_csv: Path, split_manifest: Path, dataset_name: str, target_column: str) -> Tuple[float, int]:
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", [])
    row = None
    for item in datasets:
        if str(item.get("name", "")).strip().lower() == str(dataset_name).strip().lower():
            row = item
            break
    if row is None:
        raise KeyError(f"Dataset `{dataset_name}` not found in {split_manifest}")
    train_block = row.get("training_set", {})
    n_train = int(train_block.get("samples", 0))
    if n_train <= 0:
        raise ValueError(f"Invalid training length in {split_manifest} for dataset `{dataset_name}`")

    df = pd.read_csv(data_csv, usecols=[target_column])
    if len(df) < n_train:
        raise ValueError(f"{data_csv} has only {len(df)} rows, but split manifest requires {n_train}")
    w_max = float(pd.to_numeric(df[target_column], errors="coerce").iloc[:n_train].max())
    if not np.isfinite(w_max):
        raise ValueError(f"Unable to compute finite w_max from {data_csv}:{target_column}")
    return w_max, n_train


def apply_fixed_scale_clip(forecast: np.ndarray, w_max: float, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(forecast, dtype=np.float64)
    row_max = np.nanmax(pred, axis=1, keepdims=True)
    scale = np.minimum(1.0, float(w_max) / np.maximum(row_max + float(epsilon), float(epsilon)))
    mapped = np.clip(scale * pred, 0.0, float(w_max))
    return mapped, scale.reshape(-1)


def auto_base_params(
    truth: np.ndarray,
    demand_quantile: float,
    gmax_margin: float,
    beta: float,
    fuel_cost: float,
    startup_cost: float,
    storage_cost: float,
) -> Dict[str, float]:
    vals = np.asarray(truth, dtype=np.float64).reshape(-1)
    demand = float(np.quantile(vals, demand_quantile))
    q995 = float(np.quantile(vals, 0.995))
    q99 = float(np.quantile(vals, 0.99))
    params = get_default_params()
    params["T"] = 24
    params["dt"] = 1
    params["demand"] = demand
    params["g_min"] = 0.0
    params["g_max"] = max(demand * 1.30, q995 * float(gmax_margin))
    params["ramp_max"] = max(params["g_max"] * 0.45, q99 * 0.35)
    params["beta"] = float(beta)
    params["fuel_cost"] = float(fuel_cost)
    params["startup_cost"] = float(startup_cost)
    params["storage_cost"] = float(storage_cost)
    return params


def default_base_params(
    *,
    beta: float,
    fuel_cost: float,
    startup_cost: float,
    storage_cost: float,
    storage_power_ratio: float | None,
) -> Dict[str, float]:
    params = get_default_params()
    params["T"] = 24
    params["dt"] = 1
    params["beta"] = float(beta)
    params["fuel_cost"] = float(fuel_cost)
    params["startup_cost"] = float(startup_cost)
    params["storage_cost"] = float(storage_cost)
    if storage_power_ratio is not None:
        params["P_ch_max"] = float(params["S_max"]) * float(storage_power_ratio)
        params["P_dis_max"] = float(params["S_max"]) * float(storage_power_ratio)
    return params


def compute_wind_curtailment(solution: Dict[str, np.ndarray], params: Dict[str, float], wind_mean: np.ndarray) -> Tuple[float, np.ndarray]:
    g = np.asarray(solution["thermal"], dtype=np.float64)
    e_ch = np.asarray(solution["charge"], dtype=np.float64)
    e_dis = np.asarray(solution["discharge"], dtype=np.float64)
    demand = float(params["demand"])
    curtail = np.zeros_like(wind_mean, dtype=np.float64)
    for t in range(int(params["T"])):
        max_acceptable = demand - (g[t] + e_dis[t] - e_ch[t])
        curtail[t] = max(float(wind_mean[t]) - max_acceptable, 0.0)
    return float(np.sum(curtail)), curtail


def solve_one_day(pred_day: np.ndarray, reserve_ratio: float, params: Dict[str, float]) -> Dict[str, float]:
    wind_mean = np.asarray(pred_day, dtype=np.float64).reshape(-1)
    wind_std = np.maximum(np.abs(wind_mean) * float(reserve_ratio), 1e-6)

    try:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            x_opt = solve_dispatch(params, wind_mean, wind_std)
        success = x_opt is not None
        if x_opt is None:
            _, _, _, x0, _ = build_dispatch_model(params, wind_mean, wind_std)
            x_opt = x0
    except Exception:
        _, _, _, x0, _ = build_dispatch_model(params, wind_mean, wind_std)
        x_opt = x0
        success = False

    solution = parse_solution(x_opt, params)
    cost = evaluate_cost(solution, params, wind_mean, wind_std)
    total_curtailment, _ = compute_wind_curtailment(solution, params, wind_mean)
    storage_throughput = float(np.sum(solution["charge"]) + np.sum(solution["discharge"]))
    reserve_total = float(np.sum(solution["reserve"]))
    total_cost = float(cost["total_cost"])
    reserve_share = float(cost["reserve_cost"] / total_cost) if total_cost > 0 else np.nan
    return {
        "success": float(success),
        "total_cost": total_cost,
        "reserve_cost": float(cost["reserve_cost"]),
        "reserve_cost_share": reserve_share,
        "fuel_cost": float(cost["fuel_cost"]),
        "purchase_cost": float(cost["purchase_cost"]),
        "storage_cost": float(cost["storage_cost"]),
        "startup_cost": float(cost["startup_cost"] + cost["shutdown_cost"]),
        "storage_throughput_kwh": storage_throughput,
        "curtailment_kwh": float(total_curtailment),
        "reserve_total_kwh": reserve_total,
    }


def aggregate_grid_run(
    mapped_forecasts: Dict[str, np.ndarray],
    scales: Dict[str, np.ndarray],
    truth: np.ndarray,
    sources: Dict[str, str],
    reserve_ratio_grid: List[float],
    args: argparse.Namespace,
    w_max: float,
    n_train: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if str(args.param_mode) == "auto":
        base_params = auto_base_params(
            truth=truth,
            demand_quantile=float(args.demand_quantile),
            gmax_margin=float(args.gmax_margin),
            beta=float(args.beta),
            fuel_cost=float(args.fuel_cost),
            startup_cost=float(args.startup_cost),
            storage_cost=float(args.storage_cost),
        )
        if args.storage_power_ratio is not None:
            base_params["P_ch_max"] = float(base_params["S_max"]) * float(args.storage_power_ratio)
            base_params["P_dis_max"] = float(base_params["S_max"]) * float(args.storage_power_ratio)
    else:
        base_params = default_base_params(
            beta=float(args.beta),
            fuel_cost=float(args.fuel_cost),
            startup_cost=float(args.startup_cost),
            storage_cost=float(args.storage_cost),
            storage_power_ratio=args.storage_power_ratio,
        )

    rows: List[Dict[str, float | str]] = []
    for reserve_ratio in reserve_ratio_grid:
        params = dict(base_params)
        params["wind_std_ratio"] = float(reserve_ratio)
        for model, pred in mapped_forecasts.items():
            model_rows = []
            eval_days = min(int(args.days), int(pred.shape[0]), int(truth.shape[0]))
            for day_idx in range(eval_days):
                day_result = solve_one_day(pred[day_idx], reserve_ratio, params)
                day_result["day_idx"] = float(day_idx)
                model_rows.append(day_result)
            frame = pd.DataFrame(model_rows)
            rows.append(
                {
                    "model": model,
                    "reserve_ratio": float(reserve_ratio),
                    "success_rate": float(frame["success"].mean()),
                    "total_cost": float(frame["total_cost"].sum()),
                    "reserve_cost_total": float(frame["reserve_cost"].sum()),
                    "reserve_cost_share": float(frame["reserve_cost"].sum() / max(frame["total_cost"].sum(), 1e-9)),
                    "fuel_cost_total": float(frame["fuel_cost"].sum()),
                    "purchase_cost_total": float(frame["purchase_cost"].sum()),
                    "storage_cost_total": float(frame["storage_cost"].sum()),
                    "startup_cost_total": float(frame["startup_cost"].sum()),
                    "storage_throughput_kwh": float(frame["storage_throughput_kwh"].sum()),
                    "curtailment_kwh": float(frame["curtailment_kwh"].sum()),
                    "reserve_total_kwh": float(frame["reserve_total_kwh"].sum()),
                    "mean_daily_scale": float(np.mean(scales[model][:eval_days])),
                    "min_daily_scale": float(np.min(scales[model][:eval_days])),
                    "model_source": sources[model],
                }
            )

    long_df = pd.DataFrame(rows)
    pair_rows: List[Dict[str, float | str]] = []
    pair_map = [
        ("Ultra-LSNT + clip-only", "Ultra-LSNT raw", "Ultra-clip vs Ultra-raw"),
        ("Ultra-LSNT + FG-MoE", "Ultra-LSNT raw", "Ultra-FG vs Ultra-raw"),
        ("iTransformer + FG-MoE", "iTransformer raw", "iTransformer-FG vs iTransformer-raw"),
    ]
    for reserve_ratio in reserve_ratio_grid:
        sub = long_df[long_df["reserve_ratio"] == float(reserve_ratio)]
        for treated, baseline, label in pair_map:
            t = sub[sub["model"] == treated]
            b = sub[sub["model"] == baseline]
            if t.empty or b.empty:
                continue
            t0 = t.iloc[0]
            b0 = b.iloc[0]
            pair_rows.append(
                {
                    "comparison": label,
                    "reserve_ratio": float(reserve_ratio),
                    "delta_total_cost": float(t0["total_cost"] - b0["total_cost"]),
                    "delta_reserve_cost_share": float(t0["reserve_cost_share"] - b0["reserve_cost_share"]),
                    "delta_storage_throughput_kwh": float(t0["storage_throughput_kwh"] - b0["storage_throughput_kwh"]),
                    "delta_curtailment_kwh": float(t0["curtailment_kwh"] - b0["curtailment_kwh"]),
                    "delta_success_rate": float(t0["success_rate"] - b0["success_rate"]),
                    "delta_reserve_total_kwh": float(t0["reserve_total_kwh"] - b0["reserve_total_kwh"]),
                }
            )
    pair_df = pd.DataFrame(pair_rows)
    meta = {
        "script": "run_ae_dispatch_reserve_ratio_sensitivity.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "days": int(args.days),
        "train_ratio": float(args.train_ratio),
        "train_days": int(n_train),
        "reserve_ratio_grid": reserve_ratio_grid,
        "mapping_rule": "per-day fixed scale+clip with s=min(1, w_max / (max_t forecast_t + eps)); mapped=clip(s*forecast, 0, w_max)",
        "w_max": float(w_max),
        "epsilon": float(args.epsilon),
        "auto_base_params": base_params,
        "model_sources": sources,
    }
    return long_df, pair_df, meta


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reserve_ratio_grid = parse_grid(args.reserve_ratio_grid, float)
    forecasts, truth, sources = load_models(args)
    wmax_source = "archive_truth_fallback"
    if args.w_max is not None:
        w_max, n_train = infer_wmax(truth, args.train_ratio, args.w_max)
        wmax_source = "explicit_argument"
    else:
        try:
            w_max, n_train = infer_wmax_from_data(
                data_csv=Path(args.wind_data_csv),
                split_manifest=Path(args.split_manifest),
                dataset_name=str(args.split_dataset_name),
                target_column=str(args.target_column),
            )
            wmax_source = "wind_final_train_split"
        except Exception:
            w_max, n_train = infer_wmax(truth, args.train_ratio, args.w_max)

    mapped_forecasts: Dict[str, np.ndarray] = {}
    scales: Dict[str, np.ndarray] = {}
    for model, pred in forecasts.items():
        mapped, scale = apply_fixed_scale_clip(pred, w_max=w_max, epsilon=float(args.epsilon))
        mapped_forecasts[model] = mapped
        scales[model] = scale

    long_df, pair_df, meta = aggregate_grid_run(
        mapped_forecasts=mapped_forecasts,
        scales=scales,
        truth=truth,
        sources=sources,
        reserve_ratio_grid=reserve_ratio_grid,
        args=args,
        w_max=w_max,
        n_train=n_train,
    )
    meta["w_max_source"] = wmax_source
    meta["wind_data_csv"] = str(Path(args.wind_data_csv)).replace("\\", "/")
    meta["split_manifest"] = str(Path(args.split_manifest)).replace("\\", "/")
    meta["split_dataset_name"] = str(args.split_dataset_name)
    meta["target_column"] = str(args.target_column)

    long_csv = out_dir / "dispatch_reserve_ratio_sensitivity_long.csv"
    pair_csv = out_dir / "dispatch_reserve_ratio_sensitivity_pairs.csv"
    meta_json = out_dir / "dispatch_reserve_ratio_sensitivity.meta.json"
    long_df.to_csv(long_csv, index=False)
    pair_df.to_csv(pair_csv, index=False)
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote {long_csv}")
    print(f"[ok] wrote {pair_csv}")
    print(f"[ok] wrote {meta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
