#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from guardrail_backbones import (
    BACKBONE_CHOICES,
    clip_variant_name,
    full_guardrail_name,
    model_forward,
    normalize_state_dict,
    partial_load_state_dict,
    raw_variant_name,
    build_backbone_model,
)
from guardrail_proxy_utils import (
    add_wind_proxy_arguments,
    build_horizon_wind_proxy,
    build_wind_proxy_spec,
    proxy_spec_to_dict,
)
from run_pi_moe_full_training_evidence import (
    DEFAULT_DATA,
    DEFAULT_DATASET_NAME,
    DEFAULT_TARGET,
    estimate_curve_params,
    infer_output_tag,
    load_main_data,
    power_envelope_pu,
    sanitize_output_tag,
    set_seed,
)


OUT_DIR = Path("results") / "ae_operation_exports"
VARIANT_CHOICES = ("raw", "clip_only", "full_guardrail")


def parse_variants(raw: str) -> List[str]:
    variants = [part.strip().lower() for part in str(raw).split(",") if part.strip()]
    if not variants:
        raise ValueError("No prediction variants were requested.")
    unknown = [variant for variant in variants if variant not in VARIANT_CHOICES]
    if unknown:
        raise ValueError(f"Unsupported prediction variants: {unknown}")
    return variants


def tagged_archive_path(out_dir: Path, variant: str, output_tag: str) -> Path:
    return out_dir / f"prediction_archive_{variant}_{output_tag}.npz"


def tagged_manifest_path(out_dir: Path, output_tag: str) -> Path:
    return out_dir / f"prediction_archive_manifest_{output_tag}.json"


def pick_device(device_flag: str) -> torch.device:
    if device_flag == "cuda":
        return torch.device("cuda")
    if device_flag == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_full_guardrail_model(
    *,
    backbone: str,
    ts_cfg,
    device: torch.device,
    input_dim: int,
    data_ref: str,
    dataset_name: str,
    base_checkpoint: str | None,
    base_config: str | None,
    fg_checkpoint: str,
):
    model = build_backbone_model(
        backbone,
        ts_cfg,
        device,
        input_dim=input_dim,
        data_ref=data_ref,
        dataset_name=dataset_name,
        checkpoint_path=base_checkpoint,
        config_path=base_config,
    )
    if not os.path.exists(fg_checkpoint):
        raise FileNotFoundError(
            f"Full-guardrail checkpoint was not found: {fg_checkpoint}"
        )
    payload = torch.load(fg_checkpoint, map_location=device)
    state_dict = normalize_state_dict(payload)
    load_report = partial_load_state_dict(model, state_dict)
    model._load_report = {
        "loaded_from": fg_checkpoint,
        "config_path": base_config,
        **load_report,
    }
    return model


def apply_clip_projection(pred_kw: torch.Tensor, ws_proxy: torch.Tensor, phy) -> torch.Tensor:
    pred_kw = pred_kw.clamp(min=0.0, max=float(phy.p_rated_kw))
    env_hi = power_envelope_pu(ws_proxy, phy) * float(phy.p_rated_kw)
    pred_kw = torch.minimum(pred_kw, env_hi)
    cut_mask = (ws_proxy < float(phy.v_in_mps)) | (ws_proxy > float(phy.v_out_mps))
    return torch.where(cut_mask, torch.zeros_like(pred_kw), pred_kw)


def export_variant_predictions(
    *,
    variant: str,
    model,
    backbone: str,
    test_loader,
    scaler,
    wind_proxy_spec,
    phy,
    device: torch.device,
    archive_path: Path,
    max_eval_batches: int = 0,
) -> Dict[str, object]:
    model.eval()
    target_mean = float(scaler.mean[-1])
    target_std = float(scaler.std[-1])

    pred_chunks: List[np.ndarray] = []
    true_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                break
            x = x.to(device)
            y = y.to(device)

            pred, _ = model_forward(model, backbone, x, temperature=0.1)
            pred_kw = pred * target_std + target_mean
            true_kw = y * target_std + target_mean

            if variant == "clip_only":
                ws_proxy = build_horizon_wind_proxy(x, wind_proxy_spec).to(dtype=pred_kw.dtype)
                pred_kw = apply_clip_projection(pred_kw, ws_proxy, phy)

            pred_chunks.append(pred_kw.detach().cpu().numpy())
            true_chunks.append(true_kw.detach().cpu().numpy())

    if not pred_chunks:
        raise RuntimeError("No prediction batches were exported; the loader produced no test batches.")

    pred_np = np.concatenate(pred_chunks, axis=0)
    true_np = np.concatenate(true_chunks, axis=0)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive_path, predictions=pred_np, ground_truth=true_np)

    return {
        "variant": variant,
        "archive": str(archive_path).replace("\\", "/"),
        "prediction_shape": list(pred_np.shape),
        "ground_truth_shape": list(true_np.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export raw / clip-only / full-guardrail prediction archives for dispatch experiments.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATA, help="Dataset alias or CSV path.")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target column name.")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME, help="Display name stored in metadata.")
    parser.add_argument("--backbone", type=str, default="ultra", choices=list(BACKBONE_CHOICES))
    parser.add_argument("--variants", type=str, default="raw,clip_only,full_guardrail", help="Comma-separated variants.")
    parser.add_argument("--base-checkpoint", type=str, default=None)
    parser.add_argument("--base-config", type=str, default=None)
    parser.add_argument("--fg-checkpoint", type=str, default=None, help="Required when full_guardrail is requested.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--max-eval-batches", type=int, default=0, help="Optional cap for quick smoke checks.")
    add_wind_proxy_arguments(parser)
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    if "full_guardrail" in variants and not args.fg_checkpoint:
        raise ValueError("--fg-checkpoint is required when full_guardrail is requested.")

    set_seed(args.seed)
    device = pick_device(args.device)
    output_tag = sanitize_output_tag(args.output_tag) if args.output_tag else infer_output_tag(args.dataset, args.dataset_name)
    out_dir = Path(args.out_dir)
    manifest_path = tagged_manifest_path(out_dir, output_tag)

    data, feature_names, ts_cfg, _, _, test_loader, scaler = load_main_data(
        args.dataset,
        args.target,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_cut = max(1, int(len(data) * 0.7))
    calibration_data = data[:train_cut]
    phy = estimate_curve_params(data, feature_names, target=args.target, train_ratio=0.7)
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

    base_model = build_backbone_model(
        args.backbone,
        ts_cfg,
        device,
        input_dim=len(feature_names),
        data_ref=args.dataset,
        dataset_name=args.dataset_name,
        checkpoint_path=args.base_checkpoint,
        config_path=args.base_config,
    )

    variant_reports: List[Dict[str, object]] = []
    for variant in variants:
        archive_path = tagged_archive_path(out_dir, variant, output_tag)
        if variant == "raw":
            report = export_variant_predictions(
                variant=variant,
                model=base_model,
                backbone=args.backbone,
                test_loader=test_loader,
                scaler=scaler,
                wind_proxy_spec=wind_proxy_spec,
                phy=phy,
                device=device,
                archive_path=archive_path,
                max_eval_batches=args.max_eval_batches,
            )
            report["model_name"] = raw_variant_name(args.backbone)
        elif variant == "clip_only":
            report = export_variant_predictions(
                variant=variant,
                model=base_model,
                backbone=args.backbone,
                test_loader=test_loader,
                scaler=scaler,
                wind_proxy_spec=wind_proxy_spec,
                phy=phy,
                device=device,
                archive_path=archive_path,
                max_eval_batches=args.max_eval_batches,
            )
            report["model_name"] = clip_variant_name(args.backbone)
        else:
            fg_model = load_full_guardrail_model(
                backbone=args.backbone,
                ts_cfg=ts_cfg,
                device=device,
                input_dim=len(feature_names),
                data_ref=args.dataset,
                dataset_name=args.dataset_name,
                base_checkpoint=args.base_checkpoint,
                base_config=args.base_config,
                fg_checkpoint=str(args.fg_checkpoint),
            )
            report = export_variant_predictions(
                variant=variant,
                model=fg_model,
                backbone=args.backbone,
                test_loader=test_loader,
                scaler=scaler,
                wind_proxy_spec=wind_proxy_spec,
                phy=phy,
                device=device,
                archive_path=archive_path,
                max_eval_batches=args.max_eval_batches,
            )
            report["model_name"] = full_guardrail_name(args.backbone)
        variant_reports.append(report)

    payload = {
        "script": "export_guardrail_prediction_archives.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "target": args.target,
        "backbone": args.backbone,
        "seed": int(args.seed),
        "output_tag": output_tag,
        "base_checkpoint": args.base_checkpoint,
        "base_config": args.base_config,
        "fg_checkpoint": args.fg_checkpoint,
        "device": str(device),
        "wind_proxy_spec": proxy_spec_to_dict(wind_proxy_spec),
        "curve_params": {
            "p_rated_kw": float(phy.p_rated_kw),
            "v_in_mps": float(phy.v_in_mps),
            "v_rated_mps": float(phy.v_rated_mps),
            "v_out_mps": float(phy.v_out_mps),
            "lambda2": float(phy.lambda2),
        },
        "variants": variant_reports,
        "manifest_path": str(manifest_path).replace("\\", "/"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] wrote {manifest_path}")
    for report in variant_reports:
        print(f"[ok] wrote {report['archive']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
