#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pypower.case24_ieee_rts import case24_ieee_rts
from pypower.idx_brch import PF, RATE_A
from pypower.idx_gen import PG
from pypower.ppoption import ppoption

from dispatch_mapping_utils import fit_mapping
from run_network_constrained_dispatch_ieee24 import (
    DEFAULT_LOAD_PROFILE,
    DEFAULT_SRC_WIND_RANGE_KW,
    MODEL_PATHS as DEFAULT_MODEL_PATHS,
    affine_map_to_mw,
    build_hour_case,
    load_forecasts_and_truth,
    load_prediction_array,
    run_dcopf_safe,
)


DEFAULT_OUT_DIR = Path("results") / "supplementary_evidence"
DEFAULT_OUTPUT_PREFIX = "dispatch_closure_mapping_decision"


def parse_methods(raw: str) -> List[str]:
    methods = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not methods:
        raise ValueError("No mapping method was provided.")
    allowed = {"affine", "isotonic", "quantile", "risk_conditional", "risk-conditional", "risk"}
    unknown = [m for m in methods if m not in allowed]
    if unknown:
        raise ValueError(f"Unsupported mapping method(s): {unknown}")
    return methods


def build_output_paths(out_dir: Path, output_prefix: str) -> Dict[str, Path]:
    prefix = str(output_prefix).strip() or DEFAULT_OUTPUT_PREFIX
    return {
        "hourly": out_dir / f"{prefix}_hourly.csv",
        "daily": out_dir / f"{prefix}_daily.csv",
        "aggregate": out_dir / f"{prefix}_aggregate.csv",
        "ranking": out_dir / f"{prefix}_ranking_stability.csv",
        "meta": out_dir / f"{prefix}.meta.json",
    }


def _load_truth_from_npz(path: Path) -> np.ndarray | None:
    if path.suffix.lower() != ".npz" or (not path.exists()):
        return None
    payload = np.load(path)
    if "ground_truth" not in payload:
        return None
    truth = np.asarray(payload["ground_truth"], dtype=np.float64)
    if truth.ndim != 2 or truth.shape[1] < 24:
        raise ValueError(f"Expected [N, >=24] ground_truth in {path}, got {truth.shape}")
    return truth[:, :24]


def _candidate_search_roots() -> List[Path]:
    cwd = Path.cwd()
    roots = [
        cwd,
        cwd / "upload_results_ae_operation_20260328_104442",
        cwd / "_inspect_ae_operation_results_bundle_20260328_123756",
        cwd / "results_export_cfg_lower_corr_20260329_012420",
    ]
    out: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def _archive_search_score(path: Path) -> Tuple[int, int, str]:
    norm = str(path).replace("\\", "/").lower()
    if "upload_results_ae_operation_20260328_104442/ae_operation_exports" in norm and "/smoke/" not in norm:
        pri = 0
    elif "/ae_operation_exports/" in norm and "/smoke/" not in norm:
        pri = 1
    elif "_inspect_ae_operation_results_bundle" in norm and "/ae_operation_exports/" in norm and "/smoke/" not in norm:
        pri = 2
    elif "/smoke/" in norm:
        pri = 8
    else:
        pri = 5
    return (pri, len(norm), norm)


def _infer_archive_variant(raw_path: str) -> str | None:
    lower = str(raw_path).replace("\\", "/").lower()
    for token in ("clip_only", "full_guardrail", "raw"):
        if token in lower:
            return token
    return None


def _infer_archive_seed(raw_path: str) -> str | None:
    match = re.search(r"seed[_-]?(\d+)", str(raw_path).replace("\\", "/").lower())
    if match:
        return match.group(1)
    return None


def _infer_archive_backbone(model_name: str, raw_path: str) -> str | None:
    tokens = f"{model_name} {raw_path}".lower()
    if "itransformer" in tokens:
        return "itransformer"
    if "ultra" in tokens or "fg-moe" in tokens:
        return "ultra"
    return None


def resolve_manifest_prediction_path(raw_path: str, model_name: str) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute() and path.exists():
        return path

    rel = Path.cwd() / path
    if rel.exists():
        return rel

    matches: List[Path] = []
    basename = path.name
    if basename:
        for root in _candidate_search_roots():
            if not root.exists():
                continue
            matches.extend(root.rglob(basename))
    if matches:
        matches = sorted({p.resolve() for p in matches}, key=_archive_search_score)
        return matches[0]

    variant = _infer_archive_variant(raw_path)
    seed = _infer_archive_seed(raw_path)
    backbone = _infer_archive_backbone(model_name, raw_path)
    if variant and seed and backbone:
        fuzzy_pattern = f"prediction_archive_{variant}_*seed{seed}*.npz"
        fuzzy: List[Path] = []
        for root in _candidate_search_roots():
            if not root.exists():
                continue
            for cand in root.rglob(fuzzy_pattern):
                lower_name = cand.name.lower()
                if backbone in lower_name and "smoke" not in str(cand).replace("\\", "/").lower():
                    fuzzy.append(cand)
        if fuzzy:
            fuzzy = sorted({p.resolve() for p in fuzzy}, key=_archive_search_score)
            return fuzzy[0]

    raise FileNotFoundError(f"Manifest prediction file not found: {raw_path}")


def load_model_manifest(path: str) -> List[Dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    items = payload
    if isinstance(payload, dict):
        items = payload.get("models") or payload.get("items") or []
    if not isinstance(items, list) or not items:
        raise ValueError(f"Model manifest is empty or invalid: {path}")

    normalized: List[Dict[str, object]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid manifest row at index {idx}: {item!r}")
        name = str(item.get("name", "")).strip()
        pred_path = str(item.get("path", "")).strip()
        if not name or not pred_path:
            raise ValueError(f"Manifest row {idx} must include non-empty `name` and `path`.")
        normalized.append(
            {
                "name": name,
                "path": pred_path,
                "order": int(item.get("order", idx)),
            }
        )
    normalized.sort(key=lambda row: (int(row["order"]), str(row["name"])))
    return normalized


def load_forecasts_and_truth_from_manifest(days: int, manifest_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, str]]:
    manifest = load_model_manifest(manifest_path)
    model_forecasts: Dict[str, np.ndarray] = {}
    truth: np.ndarray | None = None
    model_sources: Dict[str, str] = {}

    for item in manifest:
        name = str(item["name"])
        path = resolve_manifest_prediction_path(str(item["path"]), name)
        model_forecasts[name] = load_prediction_array(path)
        model_sources[name] = str(path).replace("\\", "/")
        if truth is None:
            truth = _load_truth_from_npz(path)

    if truth is None:
        fallback_truth_path = Path(DEFAULT_MODEL_PATHS["Ultra-LSNT"])
        if fallback_truth_path.exists():
            truth = _load_truth_from_npz(fallback_truth_path)
    if truth is None:
        raise RuntimeError(
            "Unable to resolve ground truth for dispatch closure. "
            "Provide at least one manifest NPZ with a `ground_truth` key."
        )

    max_days = min(min(v.shape[0] for v in model_forecasts.values()), truth.shape[0], int(days))
    for key in list(model_forecasts.keys()):
        model_forecasts[key] = model_forecasts[key][:max_days]
    truth = truth[:max_days]
    return model_forecasts, truth, model_sources


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    m = np.isfinite(y_t) & np.isfinite(y_p)
    if int(np.sum(m)) < 2:
        return float("nan")
    t = y_t[m]
    p = y_p[m]
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _global_hour_idx(day_idx: int, hour: int) -> int:
    return day_idx * 24 + hour


def _resolve_holdout_start_day(n_days: int, calibration_days: int, raw_value: str) -> int:
    if str(raw_value).strip().lower() == "auto":
        start = int(calibration_days)
    else:
        start = int(raw_value)
    start = max(1, min(int(start), int(n_days) - 1))
    return start


def build_shared_risk_reference(
    actual_mw: np.ndarray,
    *,
    risk_q: float,
    ramp_q: float,
    holdout_start_day: int,
) -> Dict[str, np.ndarray | float | int]:
    n_days = int(actual_mw.shape[0])
    flat_actual = np.asarray(actual_mw, dtype=np.float64).reshape(-1)
    day_idx = np.repeat(np.arange(n_days, dtype=int), 24)
    hour_idx = np.tile(np.arange(24, dtype=int), n_days)
    load_factor = np.tile(np.asarray(DEFAULT_LOAD_PROFILE, dtype=np.float64), n_days)
    prev_actual = flat_actual.copy()
    if prev_actual.size > 1:
        prev_actual[1:] = flat_actual[:-1]
    ramp_actual = np.abs(flat_actual - prev_actual)
    calibration_mask = day_idx < int(holdout_start_day)
    holdout_mask = ~calibration_mask

    load_thr = float(np.nanquantile(load_factor[calibration_mask], risk_q))
    ramp_thr = float(np.nanquantile(ramp_actual[calibration_mask], ramp_q))
    shared_high_risk = (load_factor >= load_thr) | (ramp_actual >= ramp_thr)

    return {
        "n_days": int(n_days),
        "day_idx": day_idx,
        "hour_idx": hour_idx,
        "load_factor": load_factor,
        "actual_mw": flat_actual,
        "ramp_actual_mw": ramp_actual,
        "calibration_mask": calibration_mask.astype(bool),
        "holdout_mask": holdout_mask.astype(bool),
        "shared_high_risk": shared_high_risk.astype(bool),
        "load_threshold": float(load_thr),
        "ramp_threshold": float(ramp_thr),
    }


def _finite_values(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _safe_quantile(values: np.ndarray, q: float, fallback: float) -> float:
    vals = _finite_values(values)
    if vals.size == 0:
        return float(fallback)
    return float(np.nanquantile(vals, q))


def _smooth_hourly_profile(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    if int(window) <= 1:
        return arr.copy()
    out = np.zeros_like(arr)
    half = int(window) // 2
    finite_global = arr[np.isfinite(arr)]
    global_fill = float(np.nanmean(finite_global)) if finite_global.size else 0.0
    for idx in range(arr.size):
        lo = max(0, idx - half)
        hi = min(arr.size, idx + half + 1)
        sub = arr[lo:hi]
        sub = sub[np.isfinite(sub)]
        out[idx] = float(np.nanmean(sub)) if sub.size else global_fill
    return out


def _circular_hour_distance(hours: np.ndarray, target_hour: int) -> np.ndarray:
    arr = np.asarray(hours, dtype=int).reshape(-1)
    diff = np.abs(arr - int(target_hour))
    return np.minimum(diff, 24 - diff)


def _build_constant_additive_uq_bundle(
    *,
    forecast_mw: np.ndarray,
    actual_mw: np.ndarray,
    reference: Dict[str, np.ndarray | float | int],
    uq_alpha: float,
    dst_low: float,
    dst_high: float,
) -> Dict[str, object]:
    pred_flat = np.asarray(forecast_mw, dtype=np.float64).reshape(-1)
    actual_flat = np.asarray(actual_mw, dtype=np.float64).reshape(-1)
    calibration_mask = np.asarray(reference["calibration_mask"], dtype=bool)
    holdout_mask = np.asarray(reference["holdout_mask"], dtype=bool)
    shared_high_risk = np.asarray(reference["shared_high_risk"], dtype=bool)
    positive_error = np.maximum(pred_flat - actual_flat, 0.0)
    q = float(min(max(1.0 - float(uq_alpha), 0.5), 0.999))

    def _margin(mask: np.ndarray, fallback: float) -> float:
        values = positive_error[mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float(fallback)
        return float(np.nanquantile(values, q))

    calib_values = positive_error[calibration_mask]
    calib_values = calib_values[np.isfinite(calib_values)]
    global_margin = float(np.nanquantile(calib_values, q)) if calib_values.size else 0.0
    high_margin = _margin(calibration_mask & shared_high_risk, global_margin)
    normal_margin = _margin(calibration_mask & (~shared_high_risk), global_margin)

    margin_flat = np.where(shared_high_risk, high_margin, normal_margin)
    forecast_lcb_flat = np.clip(pred_flat - margin_flat, float(dst_low), float(dst_high))
    uq_covered_flat = (actual_flat >= forecast_lcb_flat).astype(float)

    def _coverage(mask: np.ndarray) -> float:
        sub = uq_covered_flat[mask]
        sub = sub[np.isfinite(sub)]
        if sub.size == 0:
            return float("nan")
        return float(np.mean(sub))

    meta = {
        "uq_alpha": float(uq_alpha),
        "coverage_target": float(1.0 - float(uq_alpha)),
        "margin_mode": "constant_additive",
        "margin_quantile": float(q),
        "global_margin_mw": float(global_margin),
        "normal_margin_mw": float(normal_margin),
        "high_risk_margin_mw": float(high_margin),
        "coverage_calibration": _coverage(calibration_mask),
        "coverage_holdout": _coverage(holdout_mask),
        "coverage_high_risk_holdout": _coverage(holdout_mask & shared_high_risk),
        "coverage_normal_holdout": _coverage(holdout_mask & (~shared_high_risk)),
    }
    return {
        "point_forecast_mw": np.asarray(forecast_mw, dtype=np.float64),
        "forecast_lcb_mw": forecast_lcb_flat.reshape(np.asarray(forecast_mw).shape),
        "uq_margin_mw": margin_flat.reshape(np.asarray(forecast_mw).shape),
        "uq_covered": uq_covered_flat.reshape(np.asarray(forecast_mw).shape),
        "coverage_target": float(1.0 - float(uq_alpha)),
        "meta": meta,
    }


def _build_scaled_horizon_uq_bundle(
    *,
    forecast_mw: np.ndarray,
    actual_mw: np.ndarray,
    reference: Dict[str, np.ndarray | float | int],
    uq_alpha: float,
    dst_low: float,
    dst_high: float,
    score_floor_quantile: float,
    score_floor_min_mw: float,
    horizon_smooth_window: int,
    max_relative_margin: float,
) -> Dict[str, object]:
    pred_flat = np.asarray(forecast_mw, dtype=np.float64).reshape(-1)
    actual_flat = np.asarray(actual_mw, dtype=np.float64).reshape(-1)
    calibration_mask = np.asarray(reference["calibration_mask"], dtype=bool)
    holdout_mask = np.asarray(reference["holdout_mask"], dtype=bool)
    shared_high_risk = np.asarray(reference["shared_high_risk"], dtype=bool)
    hour_idx = np.asarray(reference["hour_idx"], dtype=int)
    positive_error = np.maximum(pred_flat - actual_flat, 0.0)
    q = float(min(max(1.0 - float(uq_alpha), 0.5), 0.999))
    floor_q = float(min(max(float(score_floor_quantile), 0.05), 0.95))
    floor_min = float(max(float(score_floor_min_mw), 0.0))
    rel_cap = float(min(max(float(max_relative_margin), 0.05), 1.50))

    finite_pred_cal = _finite_values(pred_flat[calibration_mask])
    global_floor = float(np.nanquantile(finite_pred_cal, floor_q)) if finite_pred_cal.size else floor_min
    global_floor = float(max(global_floor, floor_min))
    global_scores = positive_error[calibration_mask] / np.maximum(pred_flat[calibration_mask] + global_floor, 1e-6)
    global_q = _safe_quantile(global_scores, q, 0.0)

    regime_specs = {
        "normal": calibration_mask & (~shared_high_risk),
        "high_risk": calibration_mask & shared_high_risk,
    }
    regime_floors: Dict[str, float] = {}
    regime_hour_q: Dict[str, np.ndarray] = {}
    regime_global_q: Dict[str, float] = {}

    for regime_name, regime_mask in regime_specs.items():
        regime_pred = _finite_values(pred_flat[regime_mask])
        regime_floor = float(np.nanquantile(regime_pred, floor_q)) if regime_pred.size else global_floor
        regime_floor = float(max(regime_floor, floor_min))
        regime_scores = positive_error[regime_mask] / np.maximum(pred_flat[regime_mask] + regime_floor, 1e-6)
        regime_q_val = _safe_quantile(regime_scores, q, global_q)
        hourly_q = np.full(24, np.nan, dtype=np.float64)
        for hour in range(24):
            mask = regime_mask & (hour_idx == hour)
            scores = positive_error[mask] / np.maximum(pred_flat[mask] + regime_floor, 1e-6)
            hourly_q[hour] = _safe_quantile(scores, q, regime_q_val)
        hourly_q = _smooth_hourly_profile(hourly_q, int(horizon_smooth_window))
        hourly_q = np.clip(hourly_q, 0.0, rel_cap)
        regime_floors[regime_name] = float(regime_floor)
        regime_hour_q[regime_name] = hourly_q
        regime_global_q[regime_name] = float(regime_q_val)

    # High-risk hours should not receive a looser lower-tail margin than normal hours.
    regime_floors["high_risk"] = float(max(regime_floors["high_risk"], regime_floors["normal"]))
    regime_global_q["high_risk"] = float(max(regime_global_q["high_risk"], regime_global_q["normal"]))
    regime_hour_q["high_risk"] = np.clip(
        np.maximum(regime_hour_q["high_risk"], regime_hour_q["normal"]),
        0.0,
        rel_cap,
    )

    chosen_floor = np.where(shared_high_risk, regime_floors["high_risk"], regime_floors["normal"])
    chosen_q = np.where(shared_high_risk, regime_hour_q["high_risk"][hour_idx], regime_hour_q["normal"][hour_idx])
    margin_flat = np.clip(chosen_q * (pred_flat + chosen_floor), 0.0, float(dst_high) - float(dst_low))
    forecast_lcb_flat = np.clip(pred_flat - margin_flat, float(dst_low), float(dst_high))
    uq_covered_flat = (actual_flat >= forecast_lcb_flat).astype(float)

    def _coverage(mask: np.ndarray) -> float:
        sub = uq_covered_flat[mask]
        sub = sub[np.isfinite(sub)]
        if sub.size == 0:
            return float("nan")
        return float(np.mean(sub))

    meta = {
        "uq_alpha": float(uq_alpha),
        "coverage_target": float(1.0 - float(uq_alpha)),
        "margin_mode": "scaled_horizon",
        "margin_quantile": float(q),
        "score_floor_quantile": float(floor_q),
        "score_floor_min_mw": float(floor_min),
        "horizon_smooth_window": int(horizon_smooth_window),
        "max_relative_margin": float(rel_cap),
        "global_score_quantile": float(global_q),
        "normal_score_quantile": float(regime_global_q["normal"]),
        "high_risk_score_quantile": float(regime_global_q["high_risk"]),
        "normal_floor_mw": float(regime_floors["normal"]),
        "high_risk_floor_mw": float(regime_floors["high_risk"]),
        "mean_margin_mw": float(np.nanmean(margin_flat)) if np.any(np.isfinite(margin_flat)) else np.nan,
        "risk_monotone_enforced": True,
        "coverage_calibration": _coverage(calibration_mask),
        "coverage_holdout": _coverage(holdout_mask),
        "coverage_high_risk_holdout": _coverage(holdout_mask & shared_high_risk),
        "coverage_normal_holdout": _coverage(holdout_mask & (~shared_high_risk)),
    }
    return {
        "point_forecast_mw": np.asarray(forecast_mw, dtype=np.float64),
        "forecast_lcb_mw": forecast_lcb_flat.reshape(np.asarray(forecast_mw).shape),
        "uq_margin_mw": margin_flat.reshape(np.asarray(forecast_mw).shape),
        "uq_covered": uq_covered_flat.reshape(np.asarray(forecast_mw).shape),
        "coverage_target": float(1.0 - float(uq_alpha)),
        "meta": meta,
    }


def _build_rolling_bias_scaled_uq_bundle(
    *,
    forecast_mw: np.ndarray,
    actual_mw: np.ndarray,
    reference: Dict[str, np.ndarray | float | int],
    uq_alpha: float,
    dst_low: float,
    dst_high: float,
    score_floor_quantile: float,
    score_floor_min_mw: float,
    horizon_smooth_window: int,
    max_relative_margin: float,
    rolling_window_days: int,
    min_history_samples: int,
    hour_band_radius: int,
    bias_quantile: float,
    bias_shrink: float,
) -> Dict[str, object]:
    base_bundle = _build_scaled_horizon_uq_bundle(
        forecast_mw=forecast_mw,
        actual_mw=actual_mw,
        reference=reference,
        uq_alpha=uq_alpha,
        dst_low=dst_low,
        dst_high=dst_high,
        score_floor_quantile=score_floor_quantile,
        score_floor_min_mw=score_floor_min_mw,
        horizon_smooth_window=horizon_smooth_window,
        max_relative_margin=max_relative_margin,
    )

    pred_flat = np.asarray(forecast_mw, dtype=np.float64).reshape(-1)
    actual_flat = np.asarray(actual_mw, dtype=np.float64).reshape(-1)
    calibration_mask = np.asarray(reference["calibration_mask"], dtype=bool)
    holdout_mask = np.asarray(reference["holdout_mask"], dtype=bool)
    shared_high_risk = np.asarray(reference["shared_high_risk"], dtype=bool)
    hour_idx = np.asarray(reference["hour_idx"], dtype=int)
    signed_error = pred_flat - actual_flat
    q = float(min(max(1.0 - float(uq_alpha), 0.5), 0.999))
    floor_q = float(min(max(float(score_floor_quantile), 0.05), 0.95))
    floor_min = float(max(float(score_floor_min_mw), 0.0))
    rel_cap = float(min(max(float(max_relative_margin), 0.05), 1.50))
    window_days = max(int(rolling_window_days), 1)
    window_hours = int(window_days) * 24
    min_samples = max(int(min_history_samples), 4)
    hour_radius = max(int(hour_band_radius), 0)
    bias_q = float(min(max(float(bias_quantile), 0.05), 0.95))
    bias_scale = float(min(max(float(bias_shrink), 0.0), 1.50))
    hour_band_lookup = {
        hour: (_circular_hour_distance(hour_idx, hour) <= hour_radius) for hour in range(24)
    }

    margin_flat = np.asarray(base_bundle["uq_margin_mw"], dtype=np.float64).reshape(-1).copy()
    forecast_lcb_flat = np.asarray(base_bundle["forecast_lcb_mw"], dtype=np.float64).reshape(-1).copy()
    uq_covered_flat = np.asarray(base_bundle["uq_covered"], dtype=np.float64).reshape(-1).copy()
    bias_flat = np.full_like(pred_flat, np.nan, dtype=np.float64)
    tail_flat = np.full_like(pred_flat, np.nan, dtype=np.float64)
    history_samples_flat = np.full_like(pred_flat, np.nan, dtype=np.float64)
    selector_code_flat = np.zeros(pred_flat.shape[0], dtype=np.int32)
    monotone_override_flat = np.zeros(pred_flat.shape[0], dtype=np.int32)
    selector_codes = {
        "base_static": 0,
        "regime_hour_band": 1,
        "regime": 2,
        "hour_band": 3,
        "global": 4,
    }
    selector_counts = {name: 0 for name in selector_codes if name != "base_static"}
    monotone_override_count = 0

    def _history_stats(history_idx: np.ndarray, *, target_high: bool, target_hour: int) -> Dict[str, float] | None:
        if history_idx.size == 0:
            return None
        regime_match = shared_high_risk[history_idx] == bool(target_high)
        hour_match = hour_band_lookup[int(target_hour)][history_idx]
        candidate_masks = [
            ("regime_hour_band", regime_match & hour_match),
            ("regime", regime_match),
            ("hour_band", hour_match),
            ("global", np.ones(history_idx.size, dtype=bool)),
        ]
        chosen_name = None
        chosen_idx = None
        fallback_name = "global"
        fallback_idx = history_idx
        fallback_size = history_idx.size
        for name, rel_mask in candidate_masks:
            idx = history_idx[rel_mask]
            if idx.size > fallback_size:
                fallback_name = name
                fallback_idx = idx
                fallback_size = idx.size
            if idx.size >= min_samples:
                chosen_name = name
                chosen_idx = idx
                break
        if chosen_idx is None:
            chosen_name = fallback_name
            chosen_idx = fallback_idx
        if chosen_idx is None or chosen_idx.size == 0:
            return None

        hist_pred = pred_flat[chosen_idx]
        hist_err = signed_error[chosen_idx]
        floor_val = float(max(_safe_quantile(hist_pred, floor_q, floor_min), floor_min))
        bias_raw = _safe_quantile(hist_err, bias_q, 0.0)
        bias_mw = float(max(bias_raw, 0.0) * bias_scale)
        corrected_pos = np.maximum(hist_err - bias_mw, 0.0)
        rel_scores = corrected_pos / np.maximum(hist_pred + floor_val, 1e-6)
        score_q_val = float(np.clip(_safe_quantile(rel_scores, q, 0.0), 0.0, rel_cap))
        return {
            "selector": chosen_name,
            "samples": float(chosen_idx.size),
            "floor_mw": float(floor_val),
            "bias_mw": float(bias_mw),
            "score_q": float(score_q_val),
        }

    holdout_indices = np.flatnonzero(holdout_mask)
    for t in holdout_indices:
        hist_start = max(0, int(t) - int(window_hours))
        history_idx = np.arange(hist_start, int(t), dtype=int)
        if history_idx.size < min_samples:
            history_idx = np.arange(0, int(t), dtype=int)
        if history_idx.size == 0:
            continue

        current_stats = _history_stats(
            history_idx,
            target_high=bool(shared_high_risk[t]),
            target_hour=int(hour_idx[t]),
        )
        if current_stats is None:
            continue

        selected_stats = current_stats
        selected_margin = float(
            current_stats["bias_mw"] + current_stats["score_q"] * (pred_flat[t] + current_stats["floor_mw"])
        )
        if bool(shared_high_risk[t]):
            normal_stats = _history_stats(
                history_idx,
                target_high=False,
                target_hour=int(hour_idx[t]),
            )
            if normal_stats is not None:
                normal_margin = float(
                    normal_stats["bias_mw"] + normal_stats["score_q"] * (pred_flat[t] + normal_stats["floor_mw"])
                )
                if normal_margin > selected_margin:
                    selected_stats = normal_stats
                    selected_margin = normal_margin
                    monotone_override_flat[t] = 1
                    monotone_override_count += 1

        selected_margin = float(np.clip(selected_margin, 0.0, float(dst_high) - float(dst_low)))
        selected_bias = float(np.clip(selected_stats["bias_mw"], 0.0, selected_margin))
        selected_tail = float(max(selected_margin - selected_bias, 0.0))
        forecast_lcb_val = float(np.clip(pred_flat[t] - selected_margin, float(dst_low), float(dst_high)))

        margin_flat[t] = selected_margin
        forecast_lcb_flat[t] = forecast_lcb_val
        uq_covered_flat[t] = float(actual_flat[t] >= forecast_lcb_val)
        bias_flat[t] = selected_bias
        tail_flat[t] = selected_tail
        history_samples_flat[t] = float(selected_stats["samples"])
        selector_name = str(selected_stats["selector"])
        selector_code_flat[t] = int(selector_codes.get(selector_name, 0))
        if selector_name in selector_counts:
            selector_counts[selector_name] += 1

    def _coverage(mask: np.ndarray) -> float:
        sub = uq_covered_flat[mask]
        sub = sub[np.isfinite(sub)]
        if sub.size == 0:
            return float("nan")
        return float(np.mean(sub))

    holdout_selector_counts = {k: int(v) for k, v in selector_counts.items() if int(v) > 0}
    meta = {
        "uq_alpha": float(uq_alpha),
        "coverage_target": float(1.0 - float(uq_alpha)),
        "margin_mode": "rolling_bias_scaled",
        "margin_quantile": float(q),
        "score_floor_quantile": float(floor_q),
        "score_floor_min_mw": float(floor_min),
        "horizon_smooth_window": int(horizon_smooth_window),
        "max_relative_margin": float(rel_cap),
        "rolling_window_days": int(window_days),
        "min_history_samples": int(min_samples),
        "hour_band_radius": int(hour_radius),
        "bias_quantile": float(bias_q),
        "bias_shrink": float(bias_scale),
        "mean_margin_mw": float(np.nanmean(margin_flat)) if np.any(np.isfinite(margin_flat)) else np.nan,
        "mean_bias_correction_mw_holdout": float(np.nanmean(bias_flat[holdout_mask])) if np.any(np.isfinite(bias_flat[holdout_mask])) else np.nan,
        "mean_tail_margin_mw_holdout": float(np.nanmean(tail_flat[holdout_mask])) if np.any(np.isfinite(tail_flat[holdout_mask])) else np.nan,
        "mean_history_samples_holdout": float(np.nanmean(history_samples_flat[holdout_mask])) if np.any(np.isfinite(history_samples_flat[holdout_mask])) else np.nan,
        "selector_counts_holdout": holdout_selector_counts,
        "monotone_override_count_holdout": int(monotone_override_count),
        "coverage_calibration": _coverage(calibration_mask),
        "coverage_holdout": _coverage(holdout_mask),
        "coverage_high_risk_holdout": _coverage(holdout_mask & shared_high_risk),
        "coverage_normal_holdout": _coverage(holdout_mask & (~shared_high_risk)),
    }
    return {
        "point_forecast_mw": np.asarray(forecast_mw, dtype=np.float64),
        "forecast_lcb_mw": forecast_lcb_flat.reshape(np.asarray(forecast_mw).shape),
        "uq_margin_mw": margin_flat.reshape(np.asarray(forecast_mw).shape),
        "uq_covered": uq_covered_flat.reshape(np.asarray(forecast_mw).shape),
        "uq_bias_correction_mw": bias_flat.reshape(np.asarray(forecast_mw).shape),
        "uq_tail_margin_mw": tail_flat.reshape(np.asarray(forecast_mw).shape),
        "uq_history_samples": history_samples_flat.reshape(np.asarray(forecast_mw).shape),
        "uq_selector_code": selector_code_flat.reshape(np.asarray(forecast_mw).shape),
        "uq_monotone_override": monotone_override_flat.reshape(np.asarray(forecast_mw).shape),
        "coverage_target": float(1.0 - float(uq_alpha)),
        "meta": meta,
    }


def build_uq_holdout_bundle(
    *,
    forecast_mw: np.ndarray,
    actual_mw: np.ndarray,
    reference: Dict[str, np.ndarray | float | int],
    uq_alpha: float,
    dst_low: float,
    dst_high: float,
    margin_mode: str,
    score_floor_quantile: float,
    score_floor_min_mw: float,
    horizon_smooth_window: int,
    max_relative_margin: float,
    rolling_window_days: int,
    min_history_samples: int,
    hour_band_radius: int,
    bias_quantile: float,
    bias_shrink: float,
) -> Dict[str, object]:
    if str(margin_mode) == "constant_additive":
        return _build_constant_additive_uq_bundle(
            forecast_mw=forecast_mw,
            actual_mw=actual_mw,
            reference=reference,
            uq_alpha=uq_alpha,
            dst_low=dst_low,
            dst_high=dst_high,
        )
    if str(margin_mode) == "rolling_bias_scaled":
        return _build_rolling_bias_scaled_uq_bundle(
            forecast_mw=forecast_mw,
            actual_mw=actual_mw,
            reference=reference,
            uq_alpha=uq_alpha,
            dst_low=dst_low,
            dst_high=dst_high,
            score_floor_quantile=score_floor_quantile,
            score_floor_min_mw=score_floor_min_mw,
            horizon_smooth_window=horizon_smooth_window,
            max_relative_margin=max_relative_margin,
            rolling_window_days=rolling_window_days,
            min_history_samples=min_history_samples,
            hour_band_radius=hour_band_radius,
            bias_quantile=bias_quantile,
            bias_shrink=bias_shrink,
        )
    return _build_scaled_horizon_uq_bundle(
        forecast_mw=forecast_mw,
        actual_mw=actual_mw,
        reference=reference,
        uq_alpha=uq_alpha,
        dst_low=dst_low,
        dst_high=dst_high,
        score_floor_quantile=score_floor_quantile,
        score_floor_min_mw=score_floor_min_mw,
        horizon_smooth_window=horizon_smooth_window,
        max_relative_margin=max_relative_margin,
    )


def evaluate_hourly(
    *,
    mapping_method: str,
    model_name: str,
    decision_forecast_mw: np.ndarray,
    actual_mw: np.ndarray,
    base_case: dict,
    ppopt: dict,
    args: argparse.Namespace,
    point_forecast_mw: np.ndarray | None = None,
    uq_bundle: Dict[str, object] | None = None,
    holdout_start_day: int | None = None,
) -> pd.DataFrame:
    rows: List[Dict] = []
    n_days = int(decision_forecast_mw.shape[0])
    n_base_gen = int(base_case["gen"].shape[0])
    point_forecast = np.asarray(point_forecast_mw if point_forecast_mw is not None else decision_forecast_mw, dtype=np.float64)
    uq_margin = None
    forecast_lcb = None
    uq_covered = None
    uq_bias = None
    uq_tail = None
    uq_history_samples = None
    uq_selector_code = None
    uq_monotone_override = None
    coverage_target = np.nan
    if uq_bundle is not None:
        uq_margin = np.asarray(uq_bundle["uq_margin_mw"], dtype=np.float64)
        forecast_lcb = np.asarray(uq_bundle["forecast_lcb_mw"], dtype=np.float64)
        uq_covered = np.asarray(uq_bundle["uq_covered"], dtype=np.float64)
        if "uq_bias_correction_mw" in uq_bundle:
            uq_bias = np.asarray(uq_bundle["uq_bias_correction_mw"], dtype=np.float64)
        if "uq_tail_margin_mw" in uq_bundle:
            uq_tail = np.asarray(uq_bundle["uq_tail_margin_mw"], dtype=np.float64)
        if "uq_history_samples" in uq_bundle:
            uq_history_samples = np.asarray(uq_bundle["uq_history_samples"], dtype=np.float64)
        if "uq_selector_code" in uq_bundle:
            uq_selector_code = np.asarray(uq_bundle["uq_selector_code"], dtype=np.float64)
        if "uq_monotone_override" in uq_bundle:
            uq_monotone_override = np.asarray(uq_bundle["uq_monotone_override"], dtype=np.float64)
        coverage_target = float(uq_bundle.get("coverage_target", np.nan))

    for day in range(n_days):
        for hour in range(24):
            f_wind = float(decision_forecast_mw[day, hour])
            a_wind = float(actual_mw[day, hour])
            load_factor = float(DEFAULT_LOAD_PROFILE[hour])
            gidx = _global_hour_idx(day, hour)
            point_f = float(point_forecast[day, hour])
            uq_margin_val = float(uq_margin[day, hour]) if uq_margin is not None else np.nan
            forecast_lcb_val = float(forecast_lcb[day, hour]) if forecast_lcb is not None else np.nan
            uq_covered_val = float(uq_covered[day, hour]) if uq_covered is not None else np.nan
            uq_bias_val = float(uq_bias[day, hour]) if uq_bias is not None else np.nan
            uq_tail_val = float(uq_tail[day, hour]) if uq_tail is not None else np.nan
            uq_history_samples_val = float(uq_history_samples[day, hour]) if uq_history_samples is not None else np.nan
            uq_selector_code_val = float(uq_selector_code[day, hour]) if uq_selector_code is not None else np.nan
            uq_monotone_override_val = float(uq_monotone_override[day, hour]) if uq_monotone_override is not None else np.nan
            holdout_only = int(day >= int(holdout_start_day)) if holdout_start_day is not None else 0

            if gidx == 0:
                prev_f = f_wind
                prev_a = a_wind
            else:
                prev_day = (gidx - 1) // 24
                prev_h = (gidx - 1) % 24
                prev_f = float(decision_forecast_mw[prev_day, prev_h])
                prev_a = float(actual_mw[prev_day, prev_h])

            ramp_actual = abs(a_wind - prev_a)
            ramp_forecast = abs(f_wind - prev_f)
            ramp_error = abs(ramp_forecast - ramp_actual)
            forecast_error = abs(f_wind - a_wind)
            positive_error = max(f_wind - a_wind, 0.0)
            negative_error = max(a_wind - f_wind, 0.0)

            da_case = build_hour_case(
                base_case=base_case,
                wind_avail_mw=f_wind,
                load_scale=args.load_scale,
                hour_factor=load_factor,
                wind_bus=args.wind_bus,
                slack_bus=args.slack_bus,
                slack_cost_linear=args.slack_cost_linear,
                thermal_fix_pg=None,
            )
            da_result = run_dcopf_safe(da_case, ppopt)
            da_success = da_result is not None
            rt_success = False
            curtail = np.nan
            rt_cost = np.nan
            slack = np.nan
            congestion = np.nan

            if da_success:
                thermal_schedule_pg = da_result["gen"][:n_base_gen, PG].copy()
                rt_case = build_hour_case(
                    base_case=base_case,
                    wind_avail_mw=a_wind,
                    load_scale=args.load_scale,
                    hour_factor=load_factor,
                    wind_bus=args.wind_bus,
                    slack_bus=args.slack_bus,
                    slack_cost_linear=args.slack_cost_linear,
                    thermal_fix_pg=thermal_schedule_pg,
                )
                rt_result = run_dcopf_safe(rt_case, ppopt)
                if rt_result is not None:
                    rt_success = True
                    rt_cost = float(rt_result["f"])
                    wind_pg = float(rt_result["gen"][n_base_gen, PG])
                    slack = float(rt_result["gen"][n_base_gen + 1, PG])
                    curtail = float(max(a_wind - wind_pg, 0.0))
                    branch = rt_result["branch"]
                    flow = np.abs(branch[:, PF])
                    rate_a = branch[:, RATE_A]
                    valid = rate_a > 0
                    if np.any(valid):
                        congestion = float(np.any(flow[valid] >= 0.99 * rate_a[valid]))
                    else:
                        congestion = 0.0

            success = da_success and rt_success
            rows.append(
                {
                    "mapping_method": mapping_method,
                    "model": model_name,
                    "day_idx": int(day),
                    "hour": int(hour),
                    "global_hour_idx": int(gidx),
                    "forecast_mw": float(f_wind),
                    "point_forecast_mw": float(point_f),
                    "forecast_lcb_mw": float(forecast_lcb_val) if np.isfinite(forecast_lcb_val) else np.nan,
                    "uq_margin_mw": float(uq_margin_val) if np.isfinite(uq_margin_val) else np.nan,
                    "uq_covered": float(uq_covered_val) if np.isfinite(uq_covered_val) else np.nan,
                    "uq_bias_correction_mw": float(uq_bias_val) if np.isfinite(uq_bias_val) else np.nan,
                    "uq_tail_margin_mw": float(uq_tail_val) if np.isfinite(uq_tail_val) else np.nan,
                    "uq_history_samples": float(uq_history_samples_val) if np.isfinite(uq_history_samples_val) else np.nan,
                    "uq_selector_code": float(uq_selector_code_val) if np.isfinite(uq_selector_code_val) else np.nan,
                    "uq_monotone_override": float(uq_monotone_override_val) if np.isfinite(uq_monotone_override_val) else np.nan,
                    "coverage_target": float(coverage_target) if np.isfinite(coverage_target) else np.nan,
                    "holdout_only": int(holdout_only),
                    "actual_mw": float(a_wind),
                    "forecast_error_mw": float(forecast_error),
                    "positive_error_mw": float(positive_error),
                    "negative_error_mw": float(negative_error),
                    "ramp_actual_mw": float(ramp_actual),
                    "ramp_forecast_mw": float(ramp_forecast),
                    "ramp_error_mw": float(ramp_error),
                    "load_factor": float(load_factor),
                    "da_success": int(da_success),
                    "rt_success": int(rt_success),
                    "success": int(success),
                    "feasibility_fail": int(1 - int(success)),
                    "rt_cost": float(rt_cost) if np.isfinite(rt_cost) else np.nan,
                    "curtailment_mwh": float(curtail) if np.isfinite(curtail) else np.nan,
                    "slack_mwh": float(max(slack, 0.0)) if np.isfinite(slack) else np.nan,
                    "congestion_flag": float(congestion) if np.isfinite(congestion) else np.nan,
                }
            )

        if (day + 1) % 20 == 0 or day + 1 == n_days:
            print(f"[{mapping_method}/{model_name}] processed {day + 1}/{n_days} days")

    return pd.DataFrame(rows)


def add_risk_and_backfire_flags(
    df_hourly: pd.DataFrame,
    risk_q: float,
    ramp_q: float,
    *,
    subset_mode: str = "shared",
) -> pd.DataFrame:
    out = df_hourly.copy()
    out["is_peak_load"] = 0
    out["is_extreme_ramp"] = 0
    out["is_high_risk_shared"] = 0
    out["is_high_risk_model_specific"] = 0
    out["is_high_risk"] = 0
    out["is_backfire_vs_dlinear"] = 0

    for mapping in sorted(out["mapping_method"].unique()):
        dm = out[out["mapping_method"] == mapping].copy()
        base_grid = dm[dm["model"] == "DLinear"]
        if base_grid.empty:
            base_grid = dm.groupby(["day_idx", "hour"], as_index=False).first()

        load_thr = float(np.nanquantile(base_grid["load_factor"], risk_q))
        ramp_thr = float(np.nanquantile(base_grid["ramp_actual_mw"], ramp_q))

        for model in sorted(dm["model"].unique()):
            idx = (out["mapping_method"] == mapping) & (out["model"] == model)
            err_thr = float(np.nanquantile(out.loc[idx, "forecast_error_mw"], risk_q))
            peak_load = out.loc[idx, "load_factor"] >= load_thr
            extreme = out.loc[idx, "ramp_actual_mw"] >= ramp_thr
            high_risk_shared = peak_load | extreme
            high_risk_model_specific = high_risk_shared | (out.loc[idx, "forecast_error_mw"] >= err_thr)
            chosen = high_risk_shared if str(subset_mode) == "shared" else high_risk_model_specific
            out.loc[idx, "is_peak_load"] = peak_load.fillna(False).astype(int)
            out.loc[idx, "is_extreme_ramp"] = extreme.fillna(False).astype(int)
            out.loc[idx, "is_high_risk_shared"] = high_risk_shared.fillna(False).astype(int)
            out.loc[idx, "is_high_risk_model_specific"] = high_risk_model_specific.fillna(False).astype(int)
            out.loc[idx, "is_high_risk"] = chosen.fillna(False).astype(int)

        base = out[(out["mapping_method"] == mapping) & (out["model"] == "DLinear")][
            ["day_idx", "hour", "success", "rt_cost", "curtailment_mwh"]
        ].rename(
            columns={
                "success": "base_success",
                "rt_cost": "base_cost",
                "curtailment_mwh": "base_curt",
            }
        )
        if base.empty:
            continue

        for model in sorted(dm["model"].unique()):
            if model == "DLinear":
                continue
            sub = out[(out["mapping_method"] == mapping) & (out["model"] == model)].merge(
                base,
                on=["day_idx", "hour"],
                how="left",
            )
            cond_fail = (sub["base_success"] == 1) & (sub["success"] == 0)
            cond_worse = (
                (sub["base_success"] == 1)
                & (sub["success"] == 1)
                & (sub["rt_cost"] > sub["base_cost"])
                & (sub["curtailment_mwh"] > sub["base_curt"])
            )
            backfire = (cond_fail | cond_worse).astype(int).to_numpy()
            out.loc[(out["mapping_method"] == mapping) & (out["model"] == model), "is_backfire_vs_dlinear"] = backfire

    return out


def build_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "hours": ("hour", "count"),
        "success_hours": ("success", "sum"),
        "feasibility_fail_hours": ("feasibility_fail", "sum"),
        "rt_cost_total": ("rt_cost", "sum"),
        "curtailment_total_mwh": ("curtailment_mwh", "sum"),
        "slack_total_mwh": ("slack_mwh", "sum"),
        "congestion_hours": ("congestion_flag", "sum"),
        "backfire_hours": ("is_backfire_vs_dlinear", "sum"),
        "peak_load_hours": ("is_peak_load", "sum"),
        "extreme_ramp_hours": ("is_extreme_ramp", "sum"),
        "high_risk_shared_hours": ("is_high_risk_shared", "sum"),
        "high_risk_model_specific_hours": ("is_high_risk_model_specific", "sum"),
        "high_risk_hours": ("is_high_risk", "sum"),
    }
    if "holdout_only" in df_hourly.columns:
        agg_spec["holdout_hours"] = ("holdout_only", "sum")
    if "uq_covered" in df_hourly.columns:
        agg_spec["uq_coverage_hits"] = ("uq_covered", "sum")
    if "uq_margin_mw" in df_hourly.columns:
        agg_spec["uq_margin_mean_mw"] = ("uq_margin_mw", "mean")
    daily = df_hourly.groupby(["mapping_method", "model", "day_idx"], as_index=False).agg(**agg_spec)
    return daily


def _aggregate_subset(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    rows = []
    for (mapping, model), g in df.groupby(["mapping_method", "model"]):
        hours = int(len(g))
        success_rate = float(np.mean(g["success"])) if hours else np.nan
        rows.append(
            {
                "mapping_method": mapping,
                "subset": subset_name,
                "model": model,
                "hours": hours,
                "success_rate": success_rate,
                "feasibility_fail_rate": float(1.0 - success_rate) if np.isfinite(success_rate) else np.nan,
                "total_rt_cost": float(np.nansum(g["rt_cost"])),
                "mean_rt_cost": float(np.nanmean(g["rt_cost"])) if np.any(np.isfinite(g["rt_cost"])) else np.nan,
                "total_curtailment_mwh": float(np.nansum(g["curtailment_mwh"])),
                "mean_curtailment_mwh": float(np.nanmean(g["curtailment_mwh"])) if np.any(np.isfinite(g["curtailment_mwh"])) else np.nan,
                "total_slack_mwh": float(np.nansum(g["slack_mwh"])),
                "mean_slack_mwh": float(np.nanmean(g["slack_mwh"])) if np.any(np.isfinite(g["slack_mwh"])) else np.nan,
                "total_positive_error_mwh": float(np.nansum(g["positive_error_mw"])),
                "mean_positive_error_mw": float(np.nanmean(g["positive_error_mw"])) if np.any(np.isfinite(g["positive_error_mw"])) else np.nan,
                "p95_positive_error_mw": float(np.nanquantile(g["positive_error_mw"], 0.95)) if np.any(np.isfinite(g["positive_error_mw"])) else np.nan,
                "total_negative_error_mwh": float(np.nansum(g["negative_error_mw"])),
                "mean_negative_error_mw": float(np.nanmean(g["negative_error_mw"])) if np.any(np.isfinite(g["negative_error_mw"])) else np.nan,
                "congestion_rate": float(np.nanmean(g["congestion_flag"])) if np.any(np.isfinite(g["congestion_flag"])) else np.nan,
                "backfire_rate_vs_dlinear": float(np.mean(g["is_backfire_vs_dlinear"])) if hours else np.nan,
                "mean_ramp_error_mw": float(np.nanmean(g["ramp_error_mw"])) if np.any(np.isfinite(g["ramp_error_mw"])) else np.nan,
                "p95_ramp_error_mw": float(np.nanquantile(g["ramp_error_mw"], 0.95)) if np.any(np.isfinite(g["ramp_error_mw"])) else np.nan,
                "mean_forecast_error_mw": float(np.nanmean(g["forecast_error_mw"])) if np.any(np.isfinite(g["forecast_error_mw"])) else np.nan,
                "forecast_r2": _safe_r2(g["actual_mw"].to_numpy(), g["forecast_mw"].to_numpy()),
                "coverage_rate": float(np.nanmean(g["uq_covered"])) if ("uq_covered" in g.columns and np.any(np.isfinite(g["uq_covered"]))) else np.nan,
                "coverage_target": float(np.nanmean(g["coverage_target"])) if ("coverage_target" in g.columns and np.any(np.isfinite(g["coverage_target"]))) else np.nan,
                "mean_margin_mw": float(np.nanmean(g["uq_margin_mw"])) if ("uq_margin_mw" in g.columns and np.any(np.isfinite(g["uq_margin_mw"]))) else np.nan,
                "holdout_rate": float(np.nanmean(g["holdout_only"])) if ("holdout_only" in g.columns and np.any(np.isfinite(g["holdout_only"]))) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_aggregate(df_hourly: pd.DataFrame) -> pd.DataFrame:
    frames = [
        _aggregate_subset(df_hourly, "all"),
        _aggregate_subset(df_hourly[df_hourly["is_extreme_ramp"] == 1], "extreme_ramp"),
        _aggregate_subset(df_hourly[df_hourly["is_high_risk_shared"] == 1], "high_risk_shared"),
        _aggregate_subset(df_hourly[df_hourly["is_high_risk_model_specific"] == 1], "high_risk_model_specific"),
        _aggregate_subset(df_hourly[df_hourly["is_high_risk"] == 1], "high_risk"),
    ]
    if "holdout_only" in df_hourly.columns and np.any(df_hourly["holdout_only"].to_numpy(dtype=int) == 1):
        holdout = df_hourly["holdout_only"] == 1
        frames.extend(
            [
                _aggregate_subset(df_hourly[holdout], "all_holdout"),
                _aggregate_subset(df_hourly[holdout & (df_hourly["is_high_risk_shared"] == 1)], "high_risk_shared_holdout"),
                _aggregate_subset(df_hourly[holdout & (df_hourly["is_high_risk_model_specific"] == 1)], "high_risk_model_specific_holdout"),
                _aggregate_subset(df_hourly[holdout & (df_hourly["is_high_risk"] == 1)], "high_risk_holdout"),
            ]
        )
    out = pd.concat(frames, axis=0, ignore_index=True)
    return out


def _spearman_from_rank_vectors(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 2:
        return float("nan")
    am = float(np.mean(a))
    bm = float(np.mean(b))
    num = float(np.sum((a - am) * (b - bm)))
    den = float(np.sqrt(np.sum((a - am) ** 2) * np.sum((b - bm) ** 2)))
    if den <= 1e-12:
        return float("nan")
    return num / den


def build_ranking_stability(df_agg: pd.DataFrame) -> pd.DataFrame:
    all_df = df_agg[df_agg["subset"] == "all"].copy()
    rank_rows = []
    rank_vectors: Dict[str, Dict[str, float]] = {}
    for mapping, g in all_df.groupby("mapping_method"):
        gg = g.sort_values("total_rt_cost", ascending=True).reset_index(drop=True)
        rank_vectors[mapping] = {}
        for i, r in gg.iterrows():
            model = str(r["model"])
            rank = int(i + 1)
            rank_vectors[mapping][model] = rank
            rank_rows.append(
                {
                    "record_type": "rank",
                    "mapping_method": mapping,
                    "model": model,
                    "rank_by_total_rt_cost": rank,
                    "top1_model": str(gg.iloc[0]["model"]),
                    "spearman_rank_corr": np.nan,
                    "top1_consistency": np.nan,
                }
            )

    map_methods = sorted(rank_vectors.keys())
    pair_rows = []
    for i in range(len(map_methods)):
        for j in range(i + 1, len(map_methods)):
            m1 = map_methods[i]
            m2 = map_methods[j]
            models = sorted(set(rank_vectors[m1].keys()).intersection(rank_vectors[m2].keys()))
            r1 = np.asarray([rank_vectors[m1][m] for m in models], dtype=np.float64)
            r2 = np.asarray([rank_vectors[m2][m] for m in models], dtype=np.float64)
            pair_rows.append(
                {
                    "record_type": "pairwise",
                    "mapping_method": f"{m1}__vs__{m2}",
                    "model": "ALL",
                    "rank_by_total_rt_cost": np.nan,
                    "top1_model": np.nan,
                    "spearman_rank_corr": _spearman_from_rank_vectors(r1, r2),
                    "top1_consistency": float(
                        int(
                            min(rank_vectors[m1], key=rank_vectors[m1].get)
                            == min(rank_vectors[m2], key=rank_vectors[m2].get)
                        )
                    ),
                }
            )

    top1_models = [min(v, key=v.get) for v in rank_vectors.values()] if rank_vectors else []
    summary = {
        "record_type": "summary",
        "mapping_method": "ALL",
        "model": "ALL",
        "rank_by_total_rt_cost": np.nan,
        "top1_model": ",".join(top1_models),
        "spearman_rank_corr": float(np.nanmean([r["spearman_rank_corr"] for r in pair_rows])) if pair_rows else np.nan,
        "top1_consistency": float(len(set(top1_models)) == 1) if top1_models else np.nan,
    }
    return pd.DataFrame(rank_rows + pair_rows + [summary])


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch closure with mapping robustness and decision-aware consistency.")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--mapping-methods", type=str, default="affine,isotonic,quantile")
    parser.add_argument("--model-manifest", type=str, default=None, help="Optional JSON manifest of {name, path, order} rows.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--risk-q", type=float, default=0.9)
    parser.add_argument("--ramp-q", type=float, default=0.9)
    parser.add_argument("--risk-subset-mode", type=str, default="shared", choices=["shared", "model_specific"])
    parser.add_argument("--calibration-ratio", type=float, default=0.5)
    parser.add_argument("--decision-mode", type=str, default="point", choices=["point", "uq_holdout"])
    parser.add_argument("--uq-alpha", type=float, default=0.10)
    parser.add_argument("--uq-regime-mode", type=str, default="shared_risk", choices=["shared_risk"])
    parser.add_argument(
        "--uq-margin-mode",
        type=str,
        default="rolling_bias_scaled",
        choices=["rolling_bias_scaled", "scaled_horizon", "constant_additive"],
    )
    parser.add_argument("--uq-score-floor-quantile", type=float, default=0.25)
    parser.add_argument("--uq-score-floor-min-mw", type=float, default=5.0)
    parser.add_argument("--uq-horizon-smooth-window", type=int, default=3)
    parser.add_argument("--uq-max-relative-margin", type=float, default=0.75)
    parser.add_argument("--uq-rolling-window-days", type=int, default=14)
    parser.add_argument("--uq-min-history-samples", type=int, default=6)
    parser.add_argument("--uq-hour-band-radius", type=int, default=1)
    parser.add_argument("--uq-bias-quantile", type=float, default=0.50)
    parser.add_argument("--uq-bias-shrink", type=float, default=0.75)
    parser.add_argument("--holdout-start-day", type=str, default="auto")
    parser.add_argument("--wind-min-mw", type=float, default=0.0)
    parser.add_argument("--wind-max-mw", type=float, default=300.0)
    parser.add_argument("--wind-bus", type=int, default=7)
    parser.add_argument("--slack-bus", type=int, default=13)
    parser.add_argument("--load-scale", type=float, default=0.75)
    parser.add_argument("--slack-cost-linear", type=float, default=2000.0)
    parser.add_argument("--risk-mapper-level-q", type=float, default=0.8)
    parser.add_argument("--risk-mapper-ramp-q", type=float, default=0.85)
    parser.add_argument("--risk-mapper-reserve-q", type=float, default=0.75)
    parser.add_argument("--risk-mapper-reserve-scale", type=float, default=0.5)
    parser.add_argument("--risk-mapper-base", type=str, default="quantile", choices=["affine", "isotonic", "quantile"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    methods = parse_methods(args.mapping_methods)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = build_output_paths(out_dir, args.output_prefix)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_dispatch_closure_mapping_decision"

    if args.model_manifest:
        model_forecasts_raw, truth_raw, model_sources = load_forecasts_and_truth_from_manifest(args.days, args.model_manifest)
    else:
        model_forecasts_raw, truth_raw = load_forecasts_and_truth(args.days)
        model_sources = {k: str(v).replace("\\", "/") for k, v in DEFAULT_MODEL_PATHS.items()}
    n_days = int(truth_raw.shape[0])
    src_low, src_high = DEFAULT_SRC_WIND_RANGE_KW
    truth_mw = affine_map_to_mw(
        truth_raw,
        src_low_kw=src_low,
        src_high_kw=src_high,
        dst_low_mw=args.wind_min_mw,
        dst_high_mw=args.wind_max_mw,
    )

    calib_days = max(1, int(n_days * float(args.calibration_ratio)))
    holdout_start_day = _resolve_holdout_start_day(
        n_days=int(n_days),
        calibration_days=int(calib_days),
        raw_value=str(args.holdout_start_day),
    )
    mapping_params: Dict[str, Dict[str, Dict]] = {}
    mapped_predictions: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in methods}
    for method in methods:
        mapping_params[method] = {}
        for model, pred_raw in model_forecasts_raw.items():
            if method == "affine":
                pred_mw = affine_map_to_mw(
                    pred_raw,
                    src_low_kw=src_low,
                    src_high_kw=src_high,
                    dst_low_mw=args.wind_min_mw,
                    dst_high_mw=args.wind_max_mw,
                )
                mapping_params[method][model] = {
                    "rule": "fixed_affine_clipping",
                    "src_low_kw": float(src_low),
                    "src_high_kw": float(src_high),
                    "dst_low_mw": float(args.wind_min_mw),
                    "dst_high_mw": float(args.wind_max_mw),
                }
            else:
                bundle = fit_mapping(
                    method=method,
                    pred_cal=pred_raw[:calib_days],
                    true_cal=truth_mw[:calib_days],
                    dst_low=args.wind_min_mw,
                    dst_high=args.wind_max_mw,
                    level_quantile=args.risk_mapper_level_q,
                    ramp_quantile=args.risk_mapper_ramp_q,
                    reserve_quantile=args.risk_mapper_reserve_q,
                    reserve_scale=args.risk_mapper_reserve_scale,
                    base_method=args.risk_mapper_base,
                )
                pred_mw = bundle.fn(pred_raw)
                mapping_params[method][model] = bundle.params
            mapped_predictions[method][model] = pred_mw

    uq_reference = build_shared_risk_reference(
        truth_mw,
        risk_q=float(args.risk_q),
        ramp_q=float(args.ramp_q),
        holdout_start_day=int(holdout_start_day),
    )
    uq_meta: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in methods}

    base_case = case24_ieee_rts()
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    hourly_frames = []
    for method in methods:
        for model, pred in mapped_predictions[method].items():
            print(f"[run] method={method}, model={model}")
            point_forecast = np.asarray(pred, dtype=np.float64)
            uq_bundle = None
            decision_forecast = point_forecast
            if str(args.decision_mode) == "uq_holdout":
                uq_bundle = build_uq_holdout_bundle(
                    forecast_mw=point_forecast,
                    actual_mw=truth_mw,
                    reference=uq_reference,
                    uq_alpha=float(args.uq_alpha),
                    dst_low=float(args.wind_min_mw),
                    dst_high=float(args.wind_max_mw),
                    margin_mode=str(args.uq_margin_mode),
                    score_floor_quantile=float(args.uq_score_floor_quantile),
                    score_floor_min_mw=float(args.uq_score_floor_min_mw),
                    horizon_smooth_window=int(args.uq_horizon_smooth_window),
                    max_relative_margin=float(args.uq_max_relative_margin),
                    rolling_window_days=int(args.uq_rolling_window_days),
                    min_history_samples=int(args.uq_min_history_samples),
                    hour_band_radius=int(args.uq_hour_band_radius),
                    bias_quantile=float(args.uq_bias_quantile),
                    bias_shrink=float(args.uq_bias_shrink),
                )
                decision_forecast = np.asarray(uq_bundle["forecast_lcb_mw"], dtype=np.float64)
                uq_meta[method][model] = {
                    k: float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else v
                    for k, v in dict(uq_bundle["meta"]).items()
                }
            dfh = evaluate_hourly(
                mapping_method=method,
                model_name=model,
                decision_forecast_mw=decision_forecast,
                actual_mw=truth_mw,
                base_case=base_case,
                ppopt=ppopt,
                args=args,
                point_forecast_mw=point_forecast,
                uq_bundle=uq_bundle,
                holdout_start_day=int(holdout_start_day),
            )
            hourly_frames.append(dfh)

    hourly = pd.concat(hourly_frames, axis=0, ignore_index=True)
    hourly = add_risk_and_backfire_flags(
        hourly,
        risk_q=float(args.risk_q),
        ramp_q=float(args.ramp_q),
        subset_mode=str(args.risk_subset_mode),
    )
    hourly["run_id"] = run_id
    hourly.to_csv(output_paths["hourly"], index=False)

    daily = build_daily(hourly)
    daily["run_id"] = run_id
    daily.to_csv(output_paths["daily"], index=False)

    agg = build_aggregate(hourly)
    agg["run_id"] = run_id
    agg.to_csv(output_paths["aggregate"], index=False)

    rank = build_ranking_stability(agg)
    rank["run_id"] = run_id
    rank.to_csv(output_paths["ranking"], index=False)

    meta = {
        "script": "run_dispatch_closure_mapping_decision_4090.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "days": n_days,
        "mapping_methods": methods,
        "calibration_days": calib_days,
        "holdout_start_day": int(holdout_start_day),
        "holdout_days": int(n_days - holdout_start_day),
        "decision_mode": str(args.decision_mode),
        "risk_mapper": {
            "level_quantile": float(args.risk_mapper_level_q),
            "ramp_quantile": float(args.risk_mapper_ramp_q),
            "reserve_quantile": float(args.risk_mapper_reserve_q),
            "reserve_scale": float(args.risk_mapper_reserve_scale),
            "base_method": str(args.risk_mapper_base),
        },
        "uq": {
            "enabled": bool(str(args.decision_mode) == "uq_holdout"),
            "uq_alpha": float(args.uq_alpha),
            "coverage_target": float(1.0 - float(args.uq_alpha)),
            "uq_regime_mode": str(args.uq_regime_mode),
            "uq_margin_mode": str(args.uq_margin_mode),
            "uq_score_floor_quantile": float(args.uq_score_floor_quantile),
            "uq_score_floor_min_mw": float(args.uq_score_floor_min_mw),
            "uq_horizon_smooth_window": int(args.uq_horizon_smooth_window),
            "uq_max_relative_margin": float(args.uq_max_relative_margin),
            "uq_rolling_window_days": int(args.uq_rolling_window_days),
            "uq_min_history_samples": int(args.uq_min_history_samples),
            "uq_hour_band_radius": int(args.uq_hour_band_radius),
            "uq_bias_quantile": float(args.uq_bias_quantile),
            "uq_bias_shrink": float(args.uq_bias_shrink),
            "holdout_start_day": int(holdout_start_day),
            "reference": {
                "load_threshold": float(uq_reference["load_threshold"]),
                "ramp_threshold": float(uq_reference["ramp_threshold"]),
            },
            "per_model": uq_meta if str(args.decision_mode) == "uq_holdout" else {},
        },
        "risk_threshold_quantile": float(args.risk_q),
        "ramp_threshold_quantile": float(args.ramp_q),
        "risk_subset_mode": str(args.risk_subset_mode),
        "high_risk_definition": "high_risk alias follows --risk-subset-mode; shared = peak_load(q>=risk_q) OR extreme_ramp(q>=ramp_q); model_specific = shared OR forecast_error_proxy(absolute forecast error >= q_risk)",
        "high_risk_shared_definition": "peak_load(q>=risk_q) OR extreme_ramp(q>=ramp_q)",
        "high_risk_model_specific_definition": "high_risk_shared OR forecast_error_proxy(absolute forecast error >= q_risk)",
        "forecast_error_proxy": "absolute forecast error in MW",
        "network": "IEEE RTS-24 DC-OPF",
        "wind_mapping": {
            "src_range_kw": [float(src_low), float(src_high)],
            "dst_range_mw": [float(args.wind_min_mw), float(args.wind_max_mw)],
        },
        "model_manifest": args.model_manifest,
        "model_sources": model_sources,
        "mapping_params": mapping_params,
        "outputs": {
            "hourly_csv": str(output_paths["hourly"]).replace("\\", "/"),
            "daily_csv": str(output_paths["daily"]).replace("\\", "/"),
            "aggregate_csv": str(output_paths["aggregate"]).replace("\\", "/"),
            "ranking_stability_csv": str(output_paths["ranking"]).replace("\\", "/"),
            "meta_json": str(output_paths["meta"]).replace("\\", "/"),
        },
    }
    output_paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[ok] wrote {output_paths['hourly']}")
    print(f"[ok] wrote {output_paths['daily']}")
    print(f"[ok] wrote {output_paths['aggregate']}")
    print(f"[ok] wrote {output_paths['ranking']}")
    print(f"[ok] wrote {output_paths['meta']}")


if __name__ == "__main__":
    main()
