#!/usr/bin/env python3
"""
Backbone-aware guardrail mechanism control:
- raw backbone output
- clip-only (posthoc)
- full guardrail rows loaded from run_pi_moe_full_training_evidence.py outputs
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from guardrail_backbones import (
    BACKBONE_CHOICES,
    backbone_display_name,
    build_backbone_model,
    clip_variant_name,
    full_guardrail_name,
    model_forward,
    raw_variant_name,
)
from guardrail_proxy_utils import (
    add_wind_proxy_arguments,
    build_horizon_wind_proxy,
    build_wind_proxy_spec,
    proxy_spec_to_dict,
)
from run_pi_moe_full_training_evidence import (
    DEFAULT_DATA,
    DEFAULT_TARGET,
    build_fgmoe_output_paths,
    default_dataset_name,
    estimate_curve_params,
    infer_output_tag,
    inject_gaussian_protocol,
    load_main_data,
    power_envelope_pu,
    sanitize_output_tag,
    set_seed,
    tagged_output_path,
)
from ultra_lsnt_timeseries import compute_metrics


OUT_DIR = os.path.join("results", "supplementary_evidence")


def build_control_output_paths(out_dir: str, output_tag: str) -> Dict[str, str]:
    return {
        "csv": tagged_output_path(out_dir, "fgmoe_clip_only_control", ".csv", output_tag),
        "meta_json": tagged_output_path(out_dir, "fgmoe_clip_only_control", ".meta.json", output_tag),
    }


def eval_backbone(
    model,
    backbone: str,
    test_loader,
    scaler,
    wind_proxy_spec,
    phy,
    device: torch.device,
    sigma_cfg: float,
    *,
    use_clip_only: bool,
    max_eval_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    target_mean = float(scaler.mean[-1])
    target_std = float(scaler.std[-1])

    all_pred: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    n_neg = 0
    n_over = 0
    n_cut_viol = 0
    n_all = 0

    for batch_idx, (x, y) in enumerate(test_loader):
        if max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break
        x = x.to(device)
        y = y.to(device)
        x_noisy = inject_gaussian_protocol(x, sigma_cfg=sigma_cfg, alpha=1.5)

        with torch.no_grad():
            pred, _ = model_forward(model, backbone, x_noisy, temperature=0.1)

        pred_kw = pred * target_std + target_mean
        true_kw = y * target_std + target_mean
        ws_proxy = build_horizon_wind_proxy(x_noisy, wind_proxy_spec).to(dtype=pred_kw.dtype)

        if use_clip_only:
            pred_kw = pred_kw.clamp(min=0.0, max=float(phy.p_rated_kw))
            ws_env_kw = power_envelope_pu(ws_proxy, phy) * float(phy.p_rated_kw)
            pred_kw = torch.minimum(pred_kw, ws_env_kw)
            cut_mask = (ws_proxy < float(phy.v_in_mps)) | (ws_proxy > float(phy.v_out_mps))
            pred_kw = torch.where(cut_mask, torch.zeros_like(pred_kw), pred_kw)

        n_neg += int((pred_kw < 0).sum().item())
        n_over += int((pred_kw > float(phy.p_rated_kw)).sum().item())
        cut_mask_eval = (ws_proxy < float(phy.v_in_mps)) | (ws_proxy > float(phy.v_out_mps))
        n_cut_viol += int(((pred_kw > 0) & cut_mask_eval).sum().item())
        n_all += int(pred_kw.numel())

        all_pred.append(pred_kw.detach().cpu().numpy())
        all_true.append(true_kw.detach().cpu().numpy())

    pred_np = np.concatenate(all_pred, axis=0)
    true_np = np.concatenate(all_true, axis=0)
    metrics = compute_metrics(pred_np, true_np)
    metrics["neg_rate_pct"] = 100.0 * n_neg / max(n_all, 1)
    metrics["over_rated_rate_pct"] = 100.0 * n_over / max(n_all, 1)
    metrics["cut_region_nonzero_rate_pct"] = 100.0 * n_cut_viol / max(n_all, 1)
    return metrics


def load_full_guardrail_rows(source_csv: str, backbone: str) -> pd.DataFrame:
    if not os.path.exists(source_csv):
        return pd.DataFrame()
    src = pd.read_csv(source_csv)
    if src.empty:
        return pd.DataFrame()

    if "variant" in src.columns:
        out = src[src["variant"] == "full_guardrail"].copy()
    else:
        out = src.copy()
        labels = {"FG-MoE (full-trained)", "PI-MoE (full-trained)", full_guardrail_name(backbone)}
        out = out[out["model"].isin(labels)].copy()

    if "backbone" in out.columns:
        out = out[out["backbone"] == backbone].copy()

    keep_cols = [
        "scenario",
        "sigma_cfg",
        "sigma_eff",
        "backbone",
        "variant",
        "seed",
        "model",
        "R2",
        "RMSE",
        "MAE",
        "neg_rate_pct",
        "over_rated_rate_pct",
        "cut_region_nonzero_rate_pct",
    ]
    for col in ("backbone", "variant", "seed"):
        if col not in out.columns:
            if col == "backbone":
                out[col] = backbone
            elif col == "variant":
                out[col] = "full_guardrail"
            else:
                out[col] = np.nan
    miss = [col for col in keep_cols if col not in out.columns]
    if miss:
        return pd.DataFrame()
    return out[keep_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clip-only vs full guardrail comparison on arbitrary datasets.")
    parser.add_argument("--backbone", type=str, default="ultra", choices=list(BACKBONE_CHOICES))
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Dataset alias or CSV path.")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target column name.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Display name for metadata.")
    parser.add_argument("--output-tag", type=str, default=None, help="Output tag; auto-derived when omitted.")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR, help="Output directory.")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-config", type=str, default=None)
    parser.add_argument("--base-checkpoint", type=str, default=None)
    parser.add_argument("--fg-source-csv", type=str, default=None, help="Override full-guardrail source CSV.")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="Optional cap for smoke runs; 0 means full loader.")
    add_wind_proxy_arguments(parser)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = default_dataset_name(args.data, args.dataset_name)
    output_tag = sanitize_output_tag(args.output_tag) if args.output_tag else infer_output_tag(args.data, dataset_name)
    outputs = build_control_output_paths(args.out_dir, output_tag)
    fg_source_csv = args.fg_source_csv or build_fgmoe_output_paths(args.out_dir, output_tag)["compare_csv"]

    data, feature_names, ts_cfg, _, _, test_loader, scaler = load_main_data(
        args.data,
        args.target,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    phy = estimate_curve_params(data, feature_names, target=args.target, train_ratio=0.7)
    train_cut = max(1, int(len(data) * 0.7))
    calibration_data = data[:train_cut]
    wind_proxy_spec = build_wind_proxy_spec(
        feature_names,
        scaler,
        ts_cfg.pred_len,
        calibration_data=calibration_data,
        calibration_mode=args.proxy_calibration_mode,
        history_len=args.proxy_history_len,
        max_wind_mps=args.proxy_max_wind_mps,
        max_step_delta_mps=args.proxy_max_step_delta_mps,
        volatility_scale_mps=args.proxy_volatility_scale_mps,
        trend_weight_min=args.proxy_trend_weight_min,
        trend_weight_max=args.proxy_trend_weight_max,
        damping_denominator=args.proxy_damping_denominator,
    )
    backbone_model = build_backbone_model(
        args.backbone,
        ts_cfg,
        device,
        input_dim=len(feature_names),
        data_ref=args.data,
        dataset_name=dataset_name,
        checkpoint_path=args.base_checkpoint,
        config_path=args.base_config,
    )

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ_clip_only_guardrail")
    rows: List[Dict] = []
    scenarios = [("clean", 0.0, 0.0), ("gaussian_60pct", 0.4, 0.6)]

    for scenario, sigma_cfg, sigma_eff in scenarios:
        raw_metrics = eval_backbone(
            backbone_model,
            args.backbone,
            test_loader,
            scaler,
            wind_proxy_spec,
            phy,
            device,
            sigma_cfg,
            use_clip_only=False,
            max_eval_batches=args.max_eval_batches,
        )
        rows.append(
            {
                "scenario": scenario,
                "sigma_cfg": sigma_cfg,
                "sigma_eff": sigma_eff,
                "backbone": args.backbone,
                "variant": "raw",
                "seed": args.seed,
                "model": raw_variant_name(args.backbone),
                "R2": raw_metrics["R2"],
                "RMSE": raw_metrics["RMSE"],
                "MAE": raw_metrics["MAE"],
                "neg_rate_pct": raw_metrics["neg_rate_pct"],
                "over_rated_rate_pct": raw_metrics["over_rated_rate_pct"],
                "cut_region_nonzero_rate_pct": raw_metrics["cut_region_nonzero_rate_pct"],
                "run_id": run_id,
            }
        )

        clip_metrics = eval_backbone(
            backbone_model,
            args.backbone,
            test_loader,
            scaler,
            wind_proxy_spec,
            phy,
            device,
            sigma_cfg,
            use_clip_only=True,
            max_eval_batches=args.max_eval_batches,
        )
        rows.append(
            {
                "scenario": scenario,
                "sigma_cfg": sigma_cfg,
                "sigma_eff": sigma_eff,
                "backbone": args.backbone,
                "variant": "clip_only",
                "seed": args.seed,
                "model": clip_variant_name(args.backbone),
                "R2": clip_metrics["R2"],
                "RMSE": clip_metrics["RMSE"],
                "MAE": clip_metrics["MAE"],
                "neg_rate_pct": clip_metrics["neg_rate_pct"],
                "over_rated_rate_pct": clip_metrics["over_rated_rate_pct"],
                "cut_region_nonzero_rate_pct": clip_metrics["cut_region_nonzero_rate_pct"],
                "run_id": run_id,
            }
        )

    fg_rows = load_full_guardrail_rows(fg_source_csv, args.backbone)
    if not fg_rows.empty:
        for _, row in fg_rows.iterrows():
            rows.append(
                {
                    "scenario": str(row["scenario"]),
                    "sigma_cfg": float(row["sigma_cfg"]),
                    "sigma_eff": float(row["sigma_eff"]),
                    "backbone": str(row["backbone"]),
                    "variant": str(row["variant"]),
                    "seed": float(row["seed"]) if not pd.isna(row["seed"]) else args.seed,
                    "model": str(row["model"]),
                    "R2": float(row["R2"]),
                    "RMSE": float(row["RMSE"]),
                    "MAE": float(row["MAE"]),
                    "neg_rate_pct": float(row["neg_rate_pct"]),
                    "over_rated_rate_pct": float(row["over_rated_rate_pct"]),
                    "cut_region_nonzero_rate_pct": float(row["cut_region_nonzero_rate_pct"]),
                    "run_id": run_id,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df = out_df[
        [
            "scenario",
            "sigma_cfg",
            "sigma_eff",
            "backbone",
            "variant",
            "seed",
            "model",
            "R2",
            "RMSE",
            "MAE",
            "neg_rate_pct",
            "over_rated_rate_pct",
            "cut_region_nonzero_rate_pct",
            "run_id",
        ]
    ]
    out_df.to_csv(outputs["csv"], index=False)

    meta = {
        "script": "run_clip_only_guardrail_control.py",
        "dataset": args.data,
        "dataset_name": dataset_name,
        "target": args.target,
        "output_tag": output_tag,
        "backbone": args.backbone,
        "seed": args.seed,
        "protocol": {
            "seq_len": ts_cfg.seq_len,
            "pred_len": ts_cfg.pred_len,
            "scenarios": ["clean", "gaussian_60pct"],
            "gaussian_cfg": 0.4,
            "gaussian_eff": 0.6,
            "alpha": 1.5,
        },
        "models": [
            raw_variant_name(args.backbone),
            clip_variant_name(args.backbone),
            full_guardrail_name(args.backbone),
        ],
        "base_model_label": backbone_display_name(args.backbone),
        "wind_proxy": proxy_spec_to_dict(wind_proxy_spec),
        "base_model_load_report": getattr(backbone_model, "_load_report", {}),
        "fg_source_csv": fg_source_csv,
        "outputs": outputs,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(outputs["meta_json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote {outputs['csv']}")
    print(f"[ok] wrote {outputs['meta_json']}")


if __name__ == "__main__":
    main()
