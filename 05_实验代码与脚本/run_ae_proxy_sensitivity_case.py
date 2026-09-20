#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_RESULTS_ROOT = Path("results") / "ae_proxy_sensitivity"
DEFAULT_SUMMARY_ROOT = Path("results") / "ae_revision_summary"
DEFAULT_DISPATCH_DAYS = 120
DEFAULT_WIND_DATA = "wind_final.csv"
DEFAULT_TARGET = "power"
DEFAULT_DATASET_NAME = "Wind_CN"
DEFAULT_SEED = 42
DEFAULT_BASE_CONFIG = Path("checkpoints_ts") / "main" / "model_config.json"
DEFAULT_BASE_CHECKPOINT = Path("checkpoints_ts") / "main" / "best_model.pth"
DEFAULT_CANONICAL_ITR_RAW = Path("upload_results_ae_operation_20260328_104442") / "ae_operation_exports" / "windcn_itransformer" / "seed_42" / "prediction_archive_raw_windcn_itransformer_seed42.npz"
DEFAULT_CANONICAL_DLINEAR = Path("results") / "supplementary_evidence" / "dlinear_predictions_full.npz"
DEFAULT_CANONICAL_TIMEMIXER = Path("checkpoints_ts") / "TimeMixer_preds.npy"

DEFAULT_TRAIN_KWARGS = {
    "epochs": 8,
    "lr": 2e-5,
    "weight_decay": 0.005,
    "aux_weight": 0.005,
    "lambda2": 0.03,
    "corruption_weight": 0.20,
    "tail_weight": 0.005,
    "ramp_weight": 0.002,
    "consistency_weight": 0.005,
    "tail_quantile": 0.95,
    "corruption_sigma_min": 0.04,
    "corruption_sigma_max": 0.18,
    "structured_fault_prob": 0.35,
    "drift_prob": 0.25,
    "quantization_prob": 0.20,
}

CASE_CONFIGS: Dict[str, Dict[str, float | str]] = {
    "baseline_fixed": {
        "label": "Baseline fixed proxy",
        "proxy_calibration_mode": "fixed",
        "proxy_max_step_delta_mps": 1.25,
        "proxy_volatility_scale_mps": 0.85,
        "proxy_trend_weight_min": 0.18,
        "proxy_trend_weight_max": 0.72,
        "proxy_damping_denominator": 1.25,
    },
    "sv_low": {
        "label": "Low volatility scale",
        "proxy_calibration_mode": "fixed",
        "proxy_max_step_delta_mps": 1.25,
        "proxy_volatility_scale_mps": 0.595,
        "proxy_trend_weight_min": 0.18,
        "proxy_trend_weight_max": 0.72,
        "proxy_damping_denominator": 1.25,
    },
    "sv_high": {
        "label": "High volatility scale",
        "proxy_calibration_mode": "fixed",
        "proxy_max_step_delta_mps": 1.25,
        "proxy_volatility_scale_mps": 1.105,
        "proxy_trend_weight_min": 0.18,
        "proxy_trend_weight_max": 0.72,
        "proxy_damping_denominator": 1.25,
    },
    "delta_low": {
        "label": "Low step delta cap",
        "proxy_calibration_mode": "fixed",
        "proxy_max_step_delta_mps": 0.875,
        "proxy_volatility_scale_mps": 0.85,
        "proxy_trend_weight_min": 0.18,
        "proxy_trend_weight_max": 0.72,
        "proxy_damping_denominator": 1.25,
    },
    "delta_high": {
        "label": "High step delta cap",
        "proxy_calibration_mode": "fixed",
        "proxy_max_step_delta_mps": 1.625,
        "proxy_volatility_scale_mps": 0.85,
        "proxy_trend_weight_min": 0.18,
        "proxy_trend_weight_max": 0.72,
        "proxy_damping_denominator": 1.25,
    },
    "omega_low": {
        "label": "Low trend weight band",
        "proxy_calibration_mode": "fixed",
        "proxy_max_step_delta_mps": 1.25,
        "proxy_volatility_scale_mps": 0.85,
        "proxy_trend_weight_min": 0.126,
        "proxy_trend_weight_max": 0.504,
        "proxy_damping_denominator": 1.25,
    },
    "omega_high": {
        "label": "High trend weight band",
        "proxy_calibration_mode": "fixed",
        "proxy_max_step_delta_mps": 1.25,
        "proxy_volatility_scale_mps": 0.85,
        "proxy_trend_weight_min": 0.234,
        "proxy_trend_weight_max": 0.936,
        "proxy_damping_denominator": 1.25,
    },
}


def sanitize_tag(raw: str) -> str:
    keep = []
    for ch in str(raw).strip().lower():
        if ch.isalnum():
            keep.append(ch)
        else:
            keep.append("_")
    tag = "".join(keep).strip("_")
    while "__" in tag:
        tag = tag.replace("__", "_")
    return tag or "case"


def tagged_output_path(out_dir: Path, stem: str, suffix: str, output_tag: str) -> Path:
    if output_tag == "windcn":
        return out_dir / f"{stem}{suffix}"
    return out_dir / f"{stem}_{output_tag}{suffix}"


def build_case_layout(results_root: Path, case_id: str, seed: int) -> Dict[str, Path]:
    case_root = results_root / case_id / f"seed_{seed}"
    train_dir = case_root / "train"
    export_dir = case_root / "exports"
    dispatch_dir = case_root / "dispatch"
    summary_dir = case_root / "summary"
    meta_dir = case_root / "meta"
    return {
        "case_root": case_root,
        "train_dir": train_dir,
        "export_dir": export_dir,
        "dispatch_dir": dispatch_dir,
        "summary_dir": summary_dir,
        "meta_dir": meta_dir,
    }


def case_output_tag(case_id: str, seed: int) -> str:
    return sanitize_tag(f"windcn_proxy_{case_id}_seed{seed}")


def build_training_outputs(train_dir: Path, output_tag: str) -> Dict[str, Path]:
    return {
        "compare_csv": tagged_output_path(train_dir, "pi_moe_full_vs_ultra_clean_sigma06", ".csv", output_tag),
        "checkpoint": tagged_output_path(train_dir, "pi_moe_full_checkpoint", ".pth", output_tag),
        "meta_json": tagged_output_path(train_dir, "pi_moe_full_training_meta", ".json", output_tag),
        "clip_csv": tagged_output_path(train_dir, "fgmoe_clip_only_control", ".csv", output_tag),
        "clip_meta_json": tagged_output_path(train_dir, "fgmoe_clip_only_control", ".meta.json", output_tag),
    }


def build_export_outputs(export_dir: Path, output_tag: str) -> Dict[str, Path]:
    return {
        "raw": export_dir / f"prediction_archive_raw_{output_tag}.npz",
        "clip_only": export_dir / f"prediction_archive_clip_only_{output_tag}.npz",
        "full_guardrail": export_dir / f"prediction_archive_full_guardrail_{output_tag}.npz",
        "manifest": export_dir / f"prediction_archive_manifest_{output_tag}.json",
    }


def build_dispatch_outputs(dispatch_dir: Path, case_id: str, seed: int) -> Dict[str, Path]:
    prefix = f"dispatch_proxy_{case_id}_seed{seed}"
    return {
        "agg_csv": dispatch_dir / f"{prefix}_aggregate.csv",
        "meta_json": dispatch_dir / f"{prefix}.meta.json",
        "summary_csv": dispatch_dir / f"operation_high_risk_summary_{case_id}_seed{seed}.csv",
        "summary_meta": dispatch_dir / f"operation_high_risk_summary_{case_id}_seed{seed}.meta.json",
        "figure_prefix": dispatch_dir / f"fig_operation_high_risk_{case_id}_seed{seed}",
        "manifest_json": dispatch_dir / f"proxy_case_manifest_{case_id}_seed{seed}.json",
    }


def ensure_dirs(paths: Dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def resolve_existing_path(preferred: List[Path]) -> Path:
    for path in preferred:
        if path.exists():
            return path
    raise FileNotFoundError("No candidate path exists:\n" + "\n".join(str(p) for p in preferred))


def build_proxy_cli(case_cfg: Dict[str, float | str]) -> List[str]:
    return [
        "--proxy-calibration-mode",
        str(case_cfg["proxy_calibration_mode"]),
        "--proxy-max-step-delta-mps",
        str(case_cfg["proxy_max_step_delta_mps"]),
        "--proxy-volatility-scale-mps",
        str(case_cfg["proxy_volatility_scale_mps"]),
        "--proxy-trend-weight-min",
        str(case_cfg["proxy_trend_weight_min"]),
        "--proxy-trend-weight-max",
        str(case_cfg["proxy_trend_weight_max"]),
        "--proxy-damping-denominator",
        str(case_cfg["proxy_damping_denominator"]),
    ]


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def create_manifest(path: Path, models: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"models": models}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_control_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"scenario", "model", "R2", "over_rated_rate_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    return df


def pick_control_row(df: pd.DataFrame, scenario: str, model: str) -> pd.Series:
    sub = df[(df["scenario"].astype(str) == scenario) & (df["model"].astype(str) == model)]
    if sub.empty:
        raise ValueError(f"Unable to find row scenario={scenario}, model={model}")
    return sub.iloc[0]


def pick_dispatch_row(df: pd.DataFrame, model: str) -> pd.Series:
    sub = df[df["model"].astype(str) == model]
    if sub.empty:
        raise ValueError(f"Unable to find dispatch row model={model}")
    return sub.iloc[0]


def write_case_summary(
    *,
    case_id: str,
    case_cfg: Dict[str, float | str],
    case_label: str,
    control_csv: Path,
    dispatch_summary_csv: Path,
    out_csv: Path,
    out_json: Path,
) -> None:
    control = load_control_summary(control_csv)
    dispatch = pd.read_csv(dispatch_summary_csv)

    fg_clean = pick_control_row(control, "clean", "FG-MoE (full-trained)")
    fg_severe = pick_control_row(control, "gaussian_60pct", "FG-MoE (full-trained)")
    raw_clean = pick_control_row(control, "clean", "Ultra-LSNT (no guardrail)")
    clip_clean = pick_control_row(control, "clean", "Ultra-LSNT + clip-only (posthoc)")
    fg_dispatch = pick_dispatch_row(dispatch, "Ultra-LSNT + FG-MoE")

    row = {
        "case_id": case_id,
        "case_label": case_label,
        "proxy_calibration_mode": case_cfg["proxy_calibration_mode"],
        "proxy_max_step_delta_mps": float(case_cfg["proxy_max_step_delta_mps"]),
        "proxy_volatility_scale_mps": float(case_cfg["proxy_volatility_scale_mps"]),
        "proxy_trend_weight_min": float(case_cfg["proxy_trend_weight_min"]),
        "proxy_trend_weight_max": float(case_cfg["proxy_trend_weight_max"]),
        "proxy_damping_denominator": float(case_cfg["proxy_damping_denominator"]),
        "clean_r2_fg": float(fg_clean["R2"]),
        "severe_r2_fg": float(fg_severe["R2"]),
        "clean_over_rated_rate_pct_fg": float(fg_clean["over_rated_rate_pct"]),
        "severe_over_rated_rate_pct_fg": float(fg_severe["over_rated_rate_pct"]),
        "clean_r2_raw": float(raw_clean["R2"]),
        "clean_r2_clip_only": float(clip_clean["R2"]),
        "high_risk_total_slack_mwh": float(fg_dispatch["high_risk_total_slack_mwh"]),
        "reserve_call_proxy_mwh": float(fg_dispatch["reserve_call_proxy_mwh"]),
        "p95_positive_error_mw": float(fg_dispatch["p95_positive_error_mw"]),
        "high_risk_total_rt_cost": float(fg_dispatch["high_risk_total_rt_cost"]),
        "high_risk_total_curtailment_mwh": float(fg_dispatch["high_risk_total_curtailment_mwh"]),
        "high_risk_forecast_r2": float(fg_dispatch["high_risk_forecast_r2"]),
        "source_control_csv": str(control_csv).replace("\\", "/"),
        "source_dispatch_summary_csv": str(dispatch_summary_csv).replace("\\", "/"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one AE proxy-sensitivity case for the fast-revision package.")
    parser.add_argument("--case-id", type=str, required=True, choices=sorted(CASE_CONFIGS))
    parser.add_argument("--phase", type=str, default="all", choices=["prep", "dispatch", "all"])
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--summary-root", type=str, default=str(DEFAULT_SUMMARY_ROOT))
    parser.add_argument("--python-executable", type=str, default=sys.executable or "python")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dispatch-days", type=int, default=DEFAULT_DISPATCH_DAYS)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--dataset", type=str, default=DEFAULT_WIND_DATA)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--base-config", type=str, default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--base-checkpoint", type=str, default=str(DEFAULT_BASE_CHECKPOINT))
    parser.add_argument("--canonical-itr-raw", type=str, default=str(DEFAULT_CANONICAL_ITR_RAW))
    parser.add_argument("--canonical-dlinear", type=str, default=str(DEFAULT_CANONICAL_DLINEAR))
    parser.add_argument("--canonical-timemixer", type=str, default=str(DEFAULT_CANONICAL_TIMEMIXER))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-train", action="store_true")

    for key, value in DEFAULT_TRAIN_KWARGS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    return parser.parse_args()


def maybe_run_prep(args: argparse.Namespace, layout: Dict[str, Path], case_cfg: Dict[str, float | str]) -> Dict[str, Path]:
    output_tag = case_output_tag(args.case_id, args.seed)
    train_outputs = build_training_outputs(layout["train_dir"], output_tag)
    export_outputs = build_export_outputs(layout["export_dir"], output_tag)

    prep_ready = (
        train_outputs["checkpoint"].exists()
        and train_outputs["clip_csv"].exists()
        and export_outputs["raw"].exists()
        and export_outputs["clip_only"].exists()
        and export_outputs["full_guardrail"].exists()
    )
    if args.skip_existing and prep_ready and (not args.force_train):
        print(f"[skip] prep already available for case={args.case_id}")
        return {**train_outputs, **export_outputs}

    proxy_cli = build_proxy_cli(case_cfg)
    train_cmd = [
        args.python_executable,
        "run_pi_moe_full_training_evidence.py",
        "--backbone",
        "ultra",
        "--data",
        args.dataset,
        "--target",
        args.target,
        "--dataset-name",
        args.dataset_name,
        "--base-config",
        args.base_config,
        "--base-checkpoint",
        args.base_checkpoint,
        "--seed",
        str(args.seed),
        "--out-dir",
        str(layout["train_dir"]),
        "--output-tag",
        output_tag,
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--aux-weight",
        str(args.aux_weight),
        "--lambda2",
        str(args.lambda2),
        "--corruption-weight",
        str(args.corruption_weight),
        "--tail-weight",
        str(args.tail_weight),
        "--ramp-weight",
        str(args.ramp_weight),
        "--consistency-weight",
        str(args.consistency_weight),
        "--tail-quantile",
        str(args.tail_quantile),
        "--corruption-sigma-min",
        str(args.corruption_sigma_min),
        "--corruption-sigma-max",
        str(args.corruption_sigma_max),
        "--structured-fault-prob",
        str(args.structured_fault_prob),
        "--drift-prob",
        str(args.drift_prob),
        "--quantization-prob",
        str(args.quantization_prob),
        "--max-train-batches",
        str(args.max_train_batches),
        "--max-eval-batches",
        str(args.max_eval_batches),
    ] + proxy_cli
    if args.force_train:
        train_cmd.append("--force-train")
    run_cmd(train_cmd, cwd=Path.cwd())

    clip_cmd = [
        args.python_executable,
        "run_clip_only_guardrail_control.py",
        "--backbone",
        "ultra",
        "--data",
        args.dataset,
        "--target",
        args.target,
        "--dataset-name",
        args.dataset_name,
        "--base-config",
        args.base_config,
        "--base-checkpoint",
        args.base_checkpoint,
        "--seed",
        str(args.seed),
        "--out-dir",
        str(layout["train_dir"]),
        "--output-tag",
        output_tag,
        "--max-eval-batches",
        str(args.max_eval_batches),
    ] + proxy_cli
    run_cmd(clip_cmd, cwd=Path.cwd())

    export_cmd = [
        args.python_executable,
        "export_guardrail_prediction_archives.py",
        "--dataset",
        args.dataset,
        "--target",
        args.target,
        "--dataset-name",
        args.dataset_name,
        "--backbone",
        "ultra",
        "--variants",
        "raw,clip_only,full_guardrail",
        "--base-config",
        args.base_config,
        "--base-checkpoint",
        args.base_checkpoint,
        "--fg-checkpoint",
        str(train_outputs["checkpoint"]),
        "--seed",
        str(args.seed),
        "--out-dir",
        str(layout["export_dir"]),
        "--output-tag",
        output_tag,
        "--max-eval-batches",
        str(args.max_eval_batches),
    ] + proxy_cli
    run_cmd(export_cmd, cwd=Path.cwd())
    return {**train_outputs, **export_outputs}


def maybe_run_dispatch(
    args: argparse.Namespace,
    layout: Dict[str, Path],
    case_cfg: Dict[str, float | str],
    prep_outputs: Dict[str, Path],
) -> Dict[str, Path]:
    dispatch_outputs = build_dispatch_outputs(layout["dispatch_dir"], args.case_id, args.seed)
    final_case_summary_csv = layout["summary_dir"] / f"proxy_case_summary_{args.case_id}_seed{args.seed}.csv"
    final_case_summary_json = layout["summary_dir"] / f"proxy_case_summary_{args.case_id}_seed{args.seed}.json"
    if args.skip_existing and dispatch_outputs["summary_csv"].exists() and final_case_summary_csv.exists():
        print(f"[skip] dispatch summary already available for case={args.case_id}")
        return {**dispatch_outputs, "case_summary_csv": final_case_summary_csv, "case_summary_json": final_case_summary_json}

    canonical_itr_raw = resolve_existing_path(
        [
            Path(args.canonical_itr_raw),
            Path("ae_uq_holdout_bundle_v5_rolling_bias_aggressive") / "ae_uq_holdout_bundle_v5" / "ae_operation_exports" / "windcn_itransformer" / "seed_42" / "prediction_archive_raw_windcn_itransformer_seed42.npz",
            Path("upload_results_ae_operation_20260328_104442") / "ae_operation_exports" / "windcn_itransformer" / "seed_42" / "prediction_archive_raw_windcn_itransformer_seed42.npz",
        ]
    )
    canonical_dlinear = resolve_existing_path([Path(args.canonical_dlinear)])
    canonical_timemixer = resolve_existing_path([Path(args.canonical_timemixer)])

    manifest_models = [
        {"name": "DLinear", "path": str(canonical_dlinear.resolve()), "order": 0},
        {"name": "TimeMixer", "path": str(canonical_timemixer.resolve()), "order": 1},
        {"name": "iTransformer raw", "path": str(canonical_itr_raw.resolve()), "order": 2},
        {"name": "Ultra-LSNT raw", "path": str(prep_outputs["raw"].resolve()), "order": 3},
        {"name": "Ultra-LSNT + clip-only", "path": str(prep_outputs["clip_only"].resolve()), "order": 4},
        {"name": "Ultra-LSNT + FG-MoE", "path": str(prep_outputs["full_guardrail"].resolve()), "order": 5},
    ]
    create_manifest(dispatch_outputs["manifest_json"], manifest_models)

    dispatch_cmd = [
        args.python_executable,
        "run_dispatch_closure_mapping_decision_4090.py",
        "--days",
        str(args.dispatch_days),
        "--mapping-methods",
        "risk_conditional",
        "--risk-subset-mode",
        "shared",
        "--model-manifest",
        str(dispatch_outputs["manifest_json"]),
        "--out-dir",
        str(layout["dispatch_dir"]),
        "--output-prefix",
        f"dispatch_proxy_{args.case_id}_seed{args.seed}",
    ]
    run_cmd(dispatch_cmd, cwd=Path.cwd())

    summary_cmd = [
        args.python_executable,
        "build_decision_aware_dispatch_evidence.py",
        "--mode",
        "operation",
        "--dispatch-agg-csv",
        str(dispatch_outputs["agg_csv"]),
        "--dispatch-meta-json",
        str(dispatch_outputs["meta_json"]),
        "--focus-mapping",
        "risk_conditional",
        "--focus-subset",
        "high_risk_shared",
        "--out-summary-csv",
        str(dispatch_outputs["summary_csv"]),
        "--out-figure-prefix",
        str(dispatch_outputs["figure_prefix"]),
        "--out-meta",
        str(dispatch_outputs["summary_meta"]),
    ]
    run_cmd(summary_cmd, cwd=Path.cwd())

    write_case_summary(
        case_id=args.case_id,
        case_cfg=case_cfg,
        case_label=str(case_cfg["label"]),
        control_csv=prep_outputs["clip_csv"],
        dispatch_summary_csv=dispatch_outputs["summary_csv"],
        out_csv=final_case_summary_csv,
        out_json=final_case_summary_json,
    )
    return {**dispatch_outputs, "case_summary_csv": final_case_summary_csv, "case_summary_json": final_case_summary_json}


def main() -> int:
    args = parse_args()
    case_cfg = CASE_CONFIGS[args.case_id]
    results_root = Path(args.results_root)
    summary_root = Path(args.summary_root)
    layout = build_case_layout(results_root, args.case_id, args.seed)
    ensure_dirs(layout)
    summary_root.mkdir(parents=True, exist_ok=True)

    prep_outputs: Dict[str, Path] = {**build_training_outputs(layout["train_dir"], case_output_tag(args.case_id, args.seed)), **build_export_outputs(layout["export_dir"], case_output_tag(args.case_id, args.seed))}
    if args.phase in {"prep", "all"}:
        prep_outputs = maybe_run_prep(args, layout, case_cfg)

    if args.phase in {"dispatch", "all"}:
        if not prep_outputs["clip_csv"].exists():
            raise FileNotFoundError(f"Clip/control summary missing before dispatch phase: {prep_outputs['clip_csv']}")
        if not prep_outputs["full_guardrail"].exists():
            raise FileNotFoundError(f"Full-guardrail archive missing before dispatch phase: {prep_outputs['full_guardrail']}")
        maybe_run_dispatch(args, layout, case_cfg, prep_outputs)

    run_meta = {
        "script": "run_ae_proxy_sensitivity_case.py",
        "case_id": args.case_id,
        "case_label": case_cfg["label"],
        "phase": args.phase,
        "seed": int(args.seed),
        "results_root": str(results_root).replace("\\", "/"),
        "summary_root": str(summary_root).replace("\\", "/"),
        "proxy_config": case_cfg,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    run_meta_path = layout["meta_dir"] / f"proxy_case_run_meta_{args.case_id}_seed{args.seed}.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote {run_meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
