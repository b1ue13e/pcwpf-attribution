#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from run_network_constrained_dispatch_ieee24 import load_prediction_array
from wind_dispatch_model import build_dispatch_model, evaluate_cost, get_default_params, parse_solution, solve_dispatch


DEFAULT_OUTPUT_DIR = Path("results") / "ae_dispatch_param_sensitivity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supplementary reserve/storage sensitivity for the AE fast revision package.")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--calibration-days", type=int, default=60)
    parser.add_argument("--sigma-floor-ratio", type=float, default=0.05)
    parser.add_argument("--demand-quantile", type=float, default=0.995)
    parser.add_argument("--gmax-margin", type=float, default=1.20)
    parser.add_argument("--storage-power-ratio", type=float, default=0.50)
    parser.add_argument("--beta", type=float, default=1.50)
    parser.add_argument("--fuel-cost", type=float, default=0.50)
    parser.add_argument("--startup-cost", type=float, default=5000.0)
    parser.add_argument("--storage-cost", type=float, default=0.05)
    parser.add_argument("--reserve-cost-grid", type=str, default="0.1,0.2,0.4")
    parser.add_argument("--storage-capacity-grid", type=str, default="25000,50000,100000")
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


def resolve_existing_path(candidates: List[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No candidate path exists:\n" + "\n".join(str(p) for p in candidates))


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


def parse_grid(raw: str, cast) -> List[float]:
    values = [cast(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("Grid cannot be empty.")
    return values


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
        "iTransformer raw": resolve_readable_prediction_path(
            [
                Path(args.itr_raw),
                Path("ae_uq_holdout_bundle_v5_rolling_bias_aggressive")
                / "ae_uq_holdout_bundle_v5"
                / "ae_operation_exports"
                / "windcn_itransformer"
                / "seed_42"
                / "prediction_archive_raw_windcn_itransformer_seed42.npz",
            ]
        ),
        "iTransformer + FG-MoE": resolve_readable_prediction_path(
            [
                Path(args.itr_fg),
                Path("ae_uq_holdout_bundle_v5_rolling_bias_aggressive")
                / "ae_uq_holdout_bundle_v5"
                / "ae_operation_exports"
                / "windcn_itransformer"
                / "seed_42"
                / "prediction_archive_full_guardrail_windcn_itransformer_seed42.npz",
                Path("upload_results_ae_operation_20260328_104442")
                / "ae_operation_exports"
                / "windcn_itransformer"
                / "seed_42"
                / "prediction_archive_full_guardrail_windcn_itransformer_seed42.npz",
            ]
        ),
        "Ultra-LSNT raw": resolve_readable_prediction_path([Path(args.ultra_raw)]),
        "Ultra-LSNT + FG-MoE": resolve_readable_prediction_path([Path(args.ultra_fg)]),
    }

    forecasts = {name: load_prediction_array(path)[: args.days] for name, path in model_paths.items()}
    truth = load_truth_from_archive(model_paths["Ultra-LSNT raw"], args.days)
    return forecasts, truth, {k: str(v).replace("\\", "/") for k, v in model_paths.items()}


def estimate_sigma_profile(pred: np.ndarray, truth: np.ndarray, calibration_days: int, floor_ratio: float) -> np.ndarray:
    cal_days = max(1, min(int(calibration_days), int(pred.shape[0]), int(truth.shape[0])))
    residual = pred[:cal_days] - truth[:cal_days]
    sigma = np.nanstd(residual, axis=0).astype(np.float64)
    level = np.nanmedian(np.abs(truth[:cal_days]), axis=0).astype(np.float64)
    sigma_floor = np.maximum(level * float(floor_ratio), 1.0)
    sigma = np.maximum(sigma, sigma_floor)
    return sigma


def auto_base_params(truth: np.ndarray, demand_quantile: float, gmax_margin: float, beta: float, fuel_cost: float, startup_cost: float, storage_cost: float) -> Dict[str, float]:
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


def solve_one_day(pred_day: np.ndarray, sigma_profile: np.ndarray, params: Dict[str, float]) -> Dict[str, float]:
    wind_mean = np.asarray(pred_day, dtype=np.float64).reshape(-1)
    wind_std = np.maximum(np.asarray(sigma_profile, dtype=np.float64).reshape(-1), np.abs(wind_mean) * 0.05)

    try:
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
        "reserve_share": reserve_share,
        "fuel_cost": float(cost["fuel_cost"]),
        "purchase_cost": float(cost["purchase_cost"]),
        "storage_cost": float(cost["storage_cost"]),
        "startup_cost": float(cost["startup_cost"] + cost["shutdown_cost"]),
        "storage_throughput_kwh": storage_throughput,
        "curtailment_kwh": float(total_curtailment),
        "reserve_total_kwh": reserve_total,
    }


def aggregate_grid_run(
    forecasts: Dict[str, np.ndarray],
    truth: np.ndarray,
    sources: Dict[str, str],
    reserve_grid: List[float],
    storage_grid: List[float],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    base_params = auto_base_params(
        truth=truth,
        demand_quantile=float(args.demand_quantile),
        gmax_margin=float(args.gmax_margin),
        beta=float(args.beta),
        fuel_cost=float(args.fuel_cost),
        startup_cost=float(args.startup_cost),
        storage_cost=float(args.storage_cost),
    )

    sigma_profiles = {
        model: estimate_sigma_profile(pred, truth, args.calibration_days, args.sigma_floor_ratio)
        for model, pred in forecasts.items()
    }

    rows: List[Dict[str, float | str]] = []
    for reserve_cost in reserve_grid:
        for s_max in storage_grid:
            params = dict(base_params)
            params["reserve_cost"] = float(reserve_cost)
            params["S_max"] = float(s_max)
            params["P_ch_max"] = float(s_max) * float(args.storage_power_ratio)
            params["P_dis_max"] = float(s_max) * float(args.storage_power_ratio)
            for model, pred in forecasts.items():
                model_rows = []
                eval_days = min(int(args.days), int(pred.shape[0]), int(truth.shape[0]))
                for day_idx in range(eval_days):
                    model_rows.append(solve_one_day(pred[day_idx], sigma_profiles[model], params))
                frame = pd.DataFrame(model_rows)
                row = {
                    "model": model,
                    "reserve_cost": float(reserve_cost),
                    "S_max": float(s_max),
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
                    "model_source": sources[model],
                }
                rows.append(row)

    long_df = pd.DataFrame(rows)

    pair_rows: List[Dict[str, float | str]] = []
    pair_map = [
        ("Ultra-LSNT + FG-MoE", "Ultra-LSNT raw", "Ultra-FG vs Ultra-raw"),
        ("iTransformer + FG-MoE", "iTransformer raw", "iTransformer-FG vs iTransformer-raw"),
    ]
    for reserve_cost in reserve_grid:
        for s_max in storage_grid:
            sub = long_df[(long_df["reserve_cost"] == float(reserve_cost)) & (long_df["S_max"] == float(s_max))]
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
                        "reserve_cost": float(reserve_cost),
                        "S_max": float(s_max),
                        "delta_total_cost": float(t0["total_cost"] - b0["total_cost"]),
                        "delta_reserve_cost_share": float(t0["reserve_cost_share"] - b0["reserve_cost_share"]),
                        "delta_storage_throughput_kwh": float(t0["storage_throughput_kwh"] - b0["storage_throughput_kwh"]),
                        "delta_curtailment_kwh": float(t0["curtailment_kwh"] - b0["curtailment_kwh"]),
                        "delta_success_rate": float(t0["success_rate"] - b0["success_rate"]),
                    }
                )
    pair_df = pd.DataFrame(pair_rows)
    meta = {
        "script": "run_ae_dispatch_param_sensitivity.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "days": int(args.days),
        "calibration_days": int(args.calibration_days),
        "reserve_cost_grid": reserve_grid,
        "storage_capacity_grid": storage_grid,
        "auto_base_params": base_params,
        "model_sources": sources,
    }
    return long_df, pair_df, meta


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reserve_grid = parse_grid(args.reserve_cost_grid, float)
    storage_grid = parse_grid(args.storage_capacity_grid, float)
    forecasts, truth, sources = load_models(args)
    long_df, pair_df, meta = aggregate_grid_run(
        forecasts=forecasts,
        truth=truth,
        sources=sources,
        reserve_grid=reserve_grid,
        storage_grid=storage_grid,
        args=args,
    )

    long_csv = out_dir / "dispatch_param_sensitivity_long.csv"
    pair_csv = out_dir / "dispatch_param_sensitivity_pairs.csv"
    meta_json = out_dir / "dispatch_param_sensitivity.meta.json"
    long_df.to_csv(long_csv, index=False)
    pair_df.to_csv(pair_csv, index=False)
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote {long_csv}")
    print(f"[ok] wrote {pair_csv}")
    print(f"[ok] wrote {meta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
