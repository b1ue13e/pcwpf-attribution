from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


MapperFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class MapperBundle:
    method: str
    fn: MapperFn
    params: Dict[str, float | int | str]


def _clean_pairs(pred: np.ndarray, true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    m = np.isfinite(p) & np.isfinite(t)
    if not np.any(m):
        raise ValueError("No valid finite calibration pairs were found.")
    return p[m], t[m]


def _clip(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), low, high)


def identity_mapper_bundle(
    *,
    method: str,
    dst_low: float,
    dst_high: float,
    reason: str,
) -> MapperBundle:
    def mapper(x: np.ndarray) -> np.ndarray:
        return _clip(x, dst_low, dst_high)

    return MapperBundle(
        method=f"{method}_identity_fallback",
        fn=mapper,
        params={
            "fallback_reason": str(reason),
            "dst_low": float(dst_low),
            "dst_high": float(dst_high),
        },
    )


def fit_affine_mapper(
    pred_cal: np.ndarray,
    true_cal: np.ndarray,
    *,
    dst_low: float,
    dst_high: float,
    **_: float | int | str,
) -> MapperBundle:
    p, t = _clean_pairs(pred_cal, true_cal)
    p_var = float(np.var(p))
    if p_var <= 1e-12:
        slope = 0.0
        intercept = float(np.mean(t))
    else:
        slope = float(np.cov(p, t, ddof=0)[0, 1] / p_var)
        intercept = float(np.mean(t) - slope * np.mean(p))

    def mapper(x: np.ndarray) -> np.ndarray:
        y = slope * np.asarray(x, dtype=np.float64) + intercept
        return _clip(y, dst_low, dst_high)

    return MapperBundle(
        method="affine",
        fn=mapper,
        params={
            "slope": slope,
            "intercept": intercept,
            "dst_low": float(dst_low),
            "dst_high": float(dst_high),
        },
    )


def fit_isotonic_mapper(
    pred_cal: np.ndarray,
    true_cal: np.ndarray,
    *,
    dst_low: float,
    dst_high: float,
    **_: float | int | str,
) -> MapperBundle:
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception as exc:
        raise RuntimeError(
            "isotonic mapping requires scikit-learn. Install it before running dispatch closure."
        ) from exc

    p, t = _clean_pairs(pred_cal, true_cal)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p, t)

    def mapper(x: np.ndarray) -> np.ndarray:
        y = ir.predict(np.asarray(x, dtype=np.float64).reshape(-1))
        y = y.reshape(np.asarray(x).shape)
        return _clip(y, dst_low, dst_high)

    return MapperBundle(
        method="isotonic",
        fn=mapper,
        params={
            "n_calibration": int(p.size),
            "x_min": float(np.min(p)),
            "x_max": float(np.max(p)),
            "dst_low": float(dst_low),
            "dst_high": float(dst_high),
        },
    )


def fit_quantile_mapper(
    pred_cal: np.ndarray,
    true_cal: np.ndarray,
    *,
    dst_low: float,
    dst_high: float,
    n_quantiles: int = 1001,
    **_: float | int | str,
) -> MapperBundle:
    p, t = _clean_pairs(pred_cal, true_cal)
    q = np.linspace(0.0, 1.0, int(n_quantiles), dtype=np.float64)
    p_q = np.quantile(p, q)
    t_q = np.quantile(t, q)

    # Guard against repeated bins: enforce weak monotonicity.
    p_q = np.maximum.accumulate(p_q)
    t_q = np.maximum.accumulate(t_q)

    def mapper(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        x_rank = np.interp(x_arr.reshape(-1), p_q, q, left=0.0, right=1.0)
        y = np.interp(x_rank, q, t_q, left=float(t_q[0]), right=float(t_q[-1]))
        y = y.reshape(x_arr.shape)
        return _clip(y, dst_low, dst_high)

    return MapperBundle(
        method="quantile",
        fn=mapper,
        params={
            "n_quantiles": int(n_quantiles),
            "dst_low": float(dst_low),
            "dst_high": float(dst_high),
        },
    )


def _flatten_with_prev(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    prev = np.empty_like(flat)
    if flat.size == 0:
        return flat, prev
    prev[0] = flat[0]
    prev[1:] = flat[:-1]
    return flat, prev


def fit_risk_conditional_mapper(
    pred_cal: np.ndarray,
    true_cal: np.ndarray,
    *,
    dst_low: float,
    dst_high: float,
    level_quantile: float = 0.8,
    ramp_quantile: float = 0.85,
    reserve_quantile: float = 0.75,
    reserve_scale: float = 0.5,
    min_regime_samples: int = 96,
    base_method: str = "quantile",
    n_quantiles: int = 1001,
    **_: float | int | str,
) -> MapperBundle:
    p, t = _clean_pairs(pred_cal, true_cal)
    flat_pred, prev_pred = _flatten_with_prev(p)
    pred_ramp = np.abs(flat_pred - prev_pred)
    level_thr = float(np.nanquantile(flat_pred, level_quantile))
    ramp_thr = float(np.nanquantile(pred_ramp, ramp_quantile))
    high_risk = (flat_pred >= level_thr) | (pred_ramp >= ramp_thr)

    if int(np.sum(high_risk)) < int(min_regime_samples) or int(np.sum(~high_risk)) < int(min_regime_samples):
        fallback = fit_quantile_mapper(
            pred_cal,
            true_cal,
            dst_low=dst_low,
            dst_high=dst_high,
            n_quantiles=n_quantiles,
        )
        fallback.params.update(
            {
                "fallback_reason": "insufficient_regime_samples",
                "level_quantile": float(level_quantile),
                "ramp_quantile": float(ramp_quantile),
            }
        )
        fallback.method = "risk_conditional_fallback"
        return fallback

    fit_kwargs = {"dst_low": dst_low, "dst_high": dst_high, "n_quantiles": n_quantiles}
    if base_method == "affine":
        normal_bundle = fit_affine_mapper(flat_pred[~high_risk], t[~high_risk], **fit_kwargs)
        risk_bundle = fit_affine_mapper(flat_pred[high_risk], t[high_risk], **fit_kwargs)
    elif base_method == "isotonic":
        normal_bundle = fit_isotonic_mapper(flat_pred[~high_risk], t[~high_risk], **fit_kwargs)
        risk_bundle = fit_isotonic_mapper(flat_pred[high_risk], t[high_risk], **fit_kwargs)
    else:
        normal_bundle = fit_quantile_mapper(flat_pred[~high_risk], t[~high_risk], **fit_kwargs)
        risk_bundle = fit_quantile_mapper(flat_pred[high_risk], t[high_risk], **fit_kwargs)

    mapped_risk_cal = risk_bundle.fn(flat_pred[high_risk])
    positive_error = np.maximum(mapped_risk_cal - t[high_risk], 0.0)
    reserve_margin = float(reserve_scale) * float(
        np.nanquantile(positive_error, reserve_quantile)
    ) if positive_error.size else 0.0

    def mapper(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        flat_x, prev_x = _flatten_with_prev(x_arr)
        ramp = np.abs(flat_x - prev_x)
        high_mask = (flat_x >= level_thr) | (ramp >= ramp_thr)
        mapped = np.empty_like(flat_x)
        if np.any(~high_mask):
            mapped[~high_mask] = normal_bundle.fn(flat_x[~high_mask])
        if np.any(high_mask):
            mapped[high_mask] = risk_bundle.fn(flat_x[high_mask]) - reserve_margin
        mapped = mapped.reshape(x_arr.shape)
        return _clip(mapped, dst_low, dst_high)

    return MapperBundle(
        method="risk_conditional",
        fn=mapper,
        params={
            "base_method": str(base_method),
            "level_quantile": float(level_quantile),
            "ramp_quantile": float(ramp_quantile),
            "reserve_quantile": float(reserve_quantile),
            "reserve_scale": float(reserve_scale),
            "level_threshold": level_thr,
            "ramp_threshold": ramp_thr,
            "reserve_margin_mw": reserve_margin,
            "risk_samples": int(np.sum(high_risk)),
            "normal_samples": int(np.sum(~high_risk)),
            "dst_low": float(dst_low),
            "dst_high": float(dst_high),
        },
    )


def fit_mapping(
    method: str,
    pred_cal: np.ndarray,
    true_cal: np.ndarray,
    *,
    dst_low: float,
    dst_high: float,
    **kwargs: float | int | str,
) -> MapperBundle:
    m = method.strip().lower()
    try:
        if m == "affine":
            return fit_affine_mapper(pred_cal, true_cal, dst_low=dst_low, dst_high=dst_high, **kwargs)
        if m == "isotonic":
            return fit_isotonic_mapper(pred_cal, true_cal, dst_low=dst_low, dst_high=dst_high, **kwargs)
        if m == "quantile":
            return fit_quantile_mapper(pred_cal, true_cal, dst_low=dst_low, dst_high=dst_high, **kwargs)
        if m in {"risk_conditional", "risk-conditional", "risk"}:
            return fit_risk_conditional_mapper(pred_cal, true_cal, dst_low=dst_low, dst_high=dst_high, **kwargs)
    except ValueError as exc:
        if "No valid finite calibration pairs were found." in str(exc):
            return identity_mapper_bundle(
                method=m,
                dst_low=dst_low,
                dst_high=dst_high,
                reason="no_valid_finite_calibration_pairs",
            )
        raise
    raise ValueError(f"Unsupported mapping method: {method}")
