#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from windcn_audit_common import (
    ROOT,
    apply_zscore,
    audit_row,
    fit_zscore,
    inverse_target_scale,
    load_windcn_matrix,
    make_multivariate_windows,
    maybe_limit_windows,
    read_split_from_manifest,
    scenario_name,
    utc_run_id,
    write_csv,
)


DEFAULT_OUT_CSV = ROOT / "results" / "supplementary_evidence" / "lightgbm_gaussian_audit_windcn.csv"
DEFAULT_META_JSON = ROOT / "results" / "supplementary_evidence" / "lightgbm_gaussian_audit_windcn.meta.json"


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
    ap = argparse.ArgumentParser(description="Audit LightGBM Gaussian robustness on Wind (CN) under the 80/20 protocol.")
    ap.add_argument("--data", type=str, default="wind_final.csv")
    ap.add_argument("--target", type=str, default="power")
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--pred-len", type=int, default=24)
    ap.add_argument("--sigma-cfgs", type=str, default="0.0,0.1,0.2,0.3,0.4")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--max-train-windows", type=int, default=0, help="Optional cap on training windows for smoke tests.")
    ap.add_argument("--max-test-windows", type=int, default=0, help="Optional cap on test windows for smoke tests.")
    ap.add_argument("--out-csv", type=str, default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--meta-json", type=str, default=str(DEFAULT_META_JSON))
    ap.add_argument("--smoke", action="store_true", help="Run a reduced smoke pass with smaller caps and fewer trees.")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    sigma_cfgs = parse_sigma_cfgs(args.sigma_cfgs)

    if args.smoke:
        if args.max_train_windows <= 0:
            args.max_train_windows = 2048
        if args.max_test_windows <= 0:
            args.max_test_windows = 256
        if args.n_estimators == 500:
            args.n_estimators = 64

    try:
        from lightgbm import LGBMRegressor
        from sklearn.multioutput import MultiOutputRegressor
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM audit requires `lightgbm` and `scikit-learn`. Install them in the target environment."
        ) from exc

    data, feature_names, target_idx = load_windcn_matrix(data_path=args.data, target=args.target)
    train_end, test_end = read_split_from_manifest(len(data), dataset_name="wind_cn")
    train_data = data[:train_end]
    test_data = data[train_end:test_end]

    stats = fit_zscore(train_data)
    train_scaled = apply_zscore(train_data, stats)
    test_scaled = apply_zscore(test_data, stats)

    x_train, y_train = make_multivariate_windows(
        train_scaled,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        target_idx=target_idx,
    )
    x_test, y_test = make_multivariate_windows(
        test_scaled,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        target_idx=target_idx,
    )

    x_train, y_train = maybe_limit_windows(x_train, y_train, max_windows=args.max_train_windows)
    x_test, y_test = maybe_limit_windows(x_test, y_test, max_windows=args.max_test_windows)

    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)

    model = MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            n_jobs=-1,
            random_state=args.random_state,
        )
    )
    model.fit(x_train_flat, y_train)

    run_id = utc_run_id("_lightgbm_gaussian_audit_windcn")
    rows = []
    y_true_denorm = inverse_target_scale(y_test, stats, target_idx=target_idx)
    for sigma_cfg in sigma_cfgs:
        if np.isclose(sigma_cfg, 0.0):
            x_eval = x_test_flat
        else:
            from windcn_audit_common import inject_gaussian_windows

            x_eval = inject_gaussian_windows(x_test, sigma_cfg).reshape(len(x_test), -1)
        pred_scaled = model.predict(x_eval)
        pred_denorm = inverse_target_scale(pred_scaled, stats, target_idx=target_idx)
        rows.append(
            audit_row(
                model="LightGBM",
                scenario=scenario_name(sigma_cfg),
                sigma_cfg=sigma_cfg,
                pred_denorm=pred_denorm,
                true_denorm=y_true_denorm,
                run_id=run_id,
            )
        )

    out_csv = write_csv(
        rows,
        args.out_csv,
        columns=["model", "scenario", "sigma_cfg", "sigma_eff", "R2", "RMSE", "MAE", "residual_gaussian_pvalue", "run_id"],
    )

    meta = {
        "run_id": run_id,
        "script": Path(__file__).name,
        "dataset": args.data,
        "target": args.target,
        "feature_names": feature_names,
        "split_policy": "chronological_80_20",
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "sigma_cfgs": sigma_cfgs,
        "sigma_eff_formula": "sigma_eff = 1.5 * sigma_cfg",
        "normalization": "train-fit z-score; inverse scaling on target before metrics",
        "train_windows": int(len(x_train)),
        "test_windows": int(len(x_test)),
        "model": {
            "wrapper": "MultiOutputRegressor(LGBMRegressor)",
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "random_state": args.random_state,
        },
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
