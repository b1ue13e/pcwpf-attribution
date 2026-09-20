from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch


_WIND_PROXY_CANDIDATES: Sequence[str] = (
    "hub_wind_speed",
    "wind_speed_100m",
    "wind speed at 100m (m/s)",
    "windspeed",
    "wind_speed",
    "wind_speed_70m",
    "wind_speed_50m",
    "wind_speed_30m",
    "wind_speed_10m",
    "round_ws",
)


@dataclass(frozen=True)
class WindProxySpec:
    feature_names: List[str]
    feature_indices: List[int]
    feature_means: List[float]
    feature_stds: List[float]
    pred_len: int
    calibration_mode: str = "auto"
    history_len: int = 8
    max_wind_mps: float = 40.0
    max_step_delta_mps: float = 1.25
    volatility_scale_mps: float = 0.85
    trend_weight_min: float = 0.18
    trend_weight_max: float = 0.72
    damping_denominator: float = 1.25


def add_wind_proxy_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--proxy-calibration-mode",
        type=str,
        default="auto",
        choices=["auto", "fixed"],
        help="Use site-calibrated proxy coefficients or keep the declared fixed coefficients.",
    )
    parser.add_argument("--proxy-history-len", type=int, default=8)
    parser.add_argument("--proxy-max-wind-mps", type=float, default=40.0)
    parser.add_argument("--proxy-max-step-delta-mps", type=float, default=1.25)
    parser.add_argument("--proxy-volatility-scale-mps", type=float, default=0.85)
    parser.add_argument("--proxy-trend-weight-min", type=float, default=0.18)
    parser.add_argument("--proxy-trend-weight-max", type=float, default=0.72)
    parser.add_argument("--proxy-damping-denominator", type=float, default=1.25)
    return parser


def infer_wind_proxy_columns(feature_names: Iterable[str], max_candidates: int = 3) -> List[str]:
    feature_list = list(feature_names)
    selected: List[str] = []
    for candidate in _WIND_PROXY_CANDIDATES:
        if candidate in feature_list and candidate not in selected:
            selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    if not selected:
        raise ValueError(
            f"Could not infer a wind-speed proxy column from features: {feature_list}"
        )
    return selected


def build_wind_proxy_spec(
    feature_names: Sequence[str],
    scaler,
    pred_len: int,
    *,
    calibration_data: np.ndarray | None = None,
    proxy_columns: Sequence[str] | None = None,
    max_candidates: int = 3,
    calibration_mode: str = "auto",
    history_len: int = 8,
    max_wind_mps: float = 40.0,
    max_step_delta_mps: float = 1.25,
    volatility_scale_mps: float = 0.85,
    trend_weight_min: float = 0.18,
    trend_weight_max: float = 0.72,
    damping_denominator: float = 1.25,
) -> WindProxySpec:
    columns = list(proxy_columns) if proxy_columns is not None else infer_wind_proxy_columns(feature_names, max_candidates=max_candidates)
    feature_indices = [int(feature_names.index(col)) for col in columns]
    means = [float(scaler.mean[idx]) for idx in feature_indices]
    stds = [float(scaler.std[idx]) for idx in feature_indices]
    mode = str(calibration_mode).strip().lower()
    if mode not in {"auto", "fixed"}:
        raise ValueError(f"Unsupported proxy calibration mode: {calibration_mode}")
    calibrated_step_delta = float(max_step_delta_mps)
    calibrated_volatility = float(volatility_scale_mps)
    if mode == "auto" and calibration_data is not None and np.asarray(calibration_data).ndim == 2 and np.asarray(calibration_data).shape[0] >= 16:
        cal = np.asarray(calibration_data, dtype=np.float64)
        ws = np.nanmean(cal[:, feature_indices], axis=1)
        ws = np.clip(np.nan_to_num(ws, nan=0.0, posinf=max_wind_mps, neginf=0.0), 0.0, float(max_wind_mps))
        diffs = np.abs(np.diff(ws))
        if diffs.size > 0:
            calibrated_step_delta = float(np.clip(np.nanquantile(diffs, 0.90), 0.80, 4.00))
            calibrated_volatility = float(np.clip(np.nanquantile(diffs, 0.75), 0.30, 3.00))
    trend_min = float(trend_weight_min)
    trend_max = float(trend_weight_max)
    if trend_min <= 0 or trend_max <= 0 or trend_min > trend_max:
        raise ValueError(
            f"Invalid proxy trend-weight bounds: min={trend_weight_min}, max={trend_weight_max}"
        )
    damping = float(damping_denominator)
    if damping <= 0:
        raise ValueError(f"proxy_damping_denominator must be positive, got {damping_denominator}")
    return WindProxySpec(
        feature_names=list(columns),
        feature_indices=feature_indices,
        feature_means=means,
        feature_stds=stds,
        pred_len=int(pred_len),
        calibration_mode=mode,
        history_len=int(history_len),
        max_wind_mps=float(max_wind_mps),
        max_step_delta_mps=calibrated_step_delta,
        volatility_scale_mps=calibrated_volatility,
        trend_weight_min=trend_min,
        trend_weight_max=trend_max,
        damping_denominator=damping,
    )


def _denormalize_signal(x: torch.Tensor, index: int, mean: float, std: float) -> torch.Tensor:
    return x[:, :, index] * float(std) + float(mean)


def build_horizon_wind_proxy(x: torch.Tensor, spec: WindProxySpec) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected x to have shape (batch, seq_len, features); got {tuple(x.shape)}")

    tracks = [
        _denormalize_signal(x, idx, mean, std)
        for idx, mean, std in zip(spec.feature_indices, spec.feature_means, spec.feature_stds)
    ]
    if len(tracks) == 1:
        wind_hist = tracks[0]
    else:
        wind_hist = torch.stack(tracks, dim=0).mean(dim=0)

    wind_hist = torch.nan_to_num(wind_hist, nan=0.0, posinf=spec.max_wind_mps, neginf=0.0)
    wind_hist = wind_hist.clamp(min=0.0, max=float(spec.max_wind_mps))

    history_len = min(int(spec.history_len), int(wind_hist.size(1)))
    recent = wind_hist[:, -history_len:]
    last = recent[:, -1]
    smooth = recent.mean(dim=1)

    if recent.size(1) >= 3:
        diffs = recent[:, 1:] - recent[:, :-1]
        lookback = min(4, int(diffs.size(1)))
        slope = diffs[:, -lookback:].mean(dim=1)
        slope = slope.clamp(min=-float(spec.max_step_delta_mps), max=float(spec.max_step_delta_mps))
        volatility = diffs.abs().mean(dim=1)
        curvature = (
            (diffs[:, 1:] - diffs[:, :-1]).abs().mean(dim=1)
            if diffs.size(1) >= 2
            else torch.zeros_like(last)
        )
    else:
        slope = torch.zeros_like(last)
        volatility = torch.zeros_like(last)
        curvature = torch.zeros_like(last)

    horizon = torch.arange(
        1,
        int(spec.pred_len) + 1,
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(0)
    horizon_scale = torch.pow(horizon, 0.85)
    persistence = last.unsqueeze(1).expand(-1, int(spec.pred_len))
    smooth_path = smooth.unsqueeze(1).expand_as(persistence)
    trend = last.unsqueeze(1) + slope.unsqueeze(1) * horizon_scale
    trend = 0.7 * trend + 0.3 * smooth_path

    trend_weight = volatility / (volatility + float(spec.volatility_scale_mps))
    trend_weight = trend_weight.clamp(min=float(spec.trend_weight_min), max=float(spec.trend_weight_max))
    smooth_weight = 0.12 + 0.18 * (1.0 - trend_weight)
    persistence_weight = (1.0 - trend_weight - smooth_weight).clamp(min=0.10)
    total = persistence_weight + trend_weight + smooth_weight
    persistence_weight = persistence_weight / total
    trend_weight = trend_weight / total
    smooth_weight = smooth_weight / total

    proxy = (
        persistence_weight.unsqueeze(1) * persistence
        + trend_weight.unsqueeze(1) * trend
        + smooth_weight.unsqueeze(1) * smooth_path
    )

    damping = 1.0 / (1.0 + curvature.unsqueeze(1) / float(spec.damping_denominator))
    proxy = persistence + damping * (proxy - persistence)
    return proxy.clamp(min=0.0, max=float(spec.max_wind_mps))


def proxy_spec_to_dict(spec: WindProxySpec) -> dict:
    return {
        "feature_names": list(spec.feature_names),
        "feature_indices": list(spec.feature_indices),
        "pred_len": int(spec.pred_len),
        "calibration_mode": str(spec.calibration_mode),
        "history_len": int(spec.history_len),
        "max_wind_mps": float(spec.max_wind_mps),
        "max_step_delta_mps": float(spec.max_step_delta_mps),
        "volatility_scale_mps": float(spec.volatility_scale_mps),
        "trend_weight_min": float(spec.trend_weight_min),
        "trend_weight_max": float(spec.trend_weight_max),
        "damping_denominator": float(spec.damping_denominator),
    }
