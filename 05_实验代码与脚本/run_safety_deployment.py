#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch

from guardrail_backbones import build_backbone_model, model_forward
from guardrail_proxy_utils import build_horizon_wind_proxy, build_wind_proxy_spec, proxy_spec_to_dict
from run_pi_moe_full_training_evidence import (
    DEFAULT_DATA,
    DEFAULT_TARGET,
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


DEFAULT_OUTDIR = Path("results") / "deployment_evidence"
DEFAULT_FG_CKPT = Path("results") / "supplementary_evidence" / "pi_moe_full_checkpoint.pth"


def snapshot_env(args: argparse.Namespace) -> Dict[str, str]:
    cpu_info = getattr(__import__("platform"), "machine")()
    return {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "data_path": str(args.data),
        "checkpoint_path": str(args.checkpoint or ""),
        "fg_checkpoint_path": str(args.fg_checkpoint),
        "torch_version": str(torch.__version__),
        "onnxruntime_version": "",
        "cpu_info": cpu_info,
        "threads": str(torch.get_num_threads()),
        "warmup_iters": str(args.warmup),
        "measure_iters": str(args.measure),
        "batch_size": str(args.batch_size),
        "seed": str(args.seed),
        "max_eval_batches": str(args.max_eval_batches),
    }


def hard_fail_if_missing(paths: Iterable[Path], mode: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing and mode == "paper":
        for path in missing:
            print(f"[paper-mode] missing required artifact: {path}", file=sys.stderr)
        raise SystemExit(1)


def build_safety_output_paths(out_dir: Path, output_tag: str) -> Dict[str, Path]:
    out_dir_str = str(out_dir)
    return {
        "table_csv": Path(tagged_output_path(out_dir_str, "table_safety_coupling", ".csv", output_tag)),
        "unified_log_csv": Path(tagged_output_path(out_dir_str, "unified_deployment_logs", ".csv", output_tag)),
        "proxy_json": Path(tagged_output_path(out_dir_str, "safety_proxy_spec", ".json", output_tag)),
    }


def apply_clip_projection(pred_kw: torch.Tensor, ws_proxy: torch.Tensor, phy) -> torch.Tensor:
    out = pred_kw.clamp(min=0.0, max=float(phy.p_rated_kw))
    env_hi = power_envelope_pu(ws_proxy, phy) * float(phy.p_rated_kw)
    out = torch.minimum(out, env_hi)
    cut_mask = (ws_proxy < float(phy.v_in_mps)) | (ws_proxy > float(phy.v_out_mps))
    out = torch.where(cut_mask, torch.zeros_like(out), out)
    return out


def prepare_scenario_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    *,
    sigma_cfg: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    x_noisy = inject_gaussian_protocol(x, sigma_cfg=sigma_cfg, alpha=1.5)
    return x_noisy, y


def measure_latency_ms(fn: Callable[[], None], *, warmup: int, measure: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(measure):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.quantile(np.asarray(times, dtype=np.float64), 0.95))


def evaluate_variant(
    *,
    model_name: str,
    mode: str,
    model,
    test_loader,
    scaler,
    wind_proxy_spec,
    phy,
    device: torch.device,
    sigma_cfg: float,
    warmup: int,
    measure: int,
    max_eval_batches: int,
) -> Dict[str, float]:
    model.eval()
    target_mean = float(scaler.mean[-1])
    target_std = float(scaler.std[-1])

    first_batch = next(iter(test_loader))
    x_latency, _ = prepare_scenario_batch(first_batch, sigma_cfg=sigma_cfg, device=device)

    def latency_step() -> None:
        with torch.no_grad():
            pred_norm, _ = model_forward(model, "ultra", x_latency, temperature=0.1)
            pred_kw = pred_norm * target_std + target_mean
            ws_proxy = build_horizon_wind_proxy(x_latency, wind_proxy_spec).to(dtype=pred_kw.dtype)
            if mode == "clip_only":
                _ = apply_clip_projection(pred_kw, ws_proxy, phy)

    p95_latency_ms = measure_latency_ms(latency_step, warmup=warmup, measure=measure)

    all_pred: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    env_dev_chunks: List[np.ndarray] = []
    n_total = 0
    n_over = 0
    n_cut_nonzero = 0

    for batch_idx, (x, y) in enumerate(test_loader):
        if max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break
        x = x.to(device)
        y = y.to(device)
        x_noisy = inject_gaussian_protocol(x, sigma_cfg=sigma_cfg, alpha=1.5)

        with torch.no_grad():
            pred_norm, _ = model_forward(model, "ultra", x_noisy, temperature=0.1)

        pred_kw = pred_norm * target_std + target_mean
        true_kw = y * target_std + target_mean
        ws_proxy = build_horizon_wind_proxy(x_noisy, wind_proxy_spec).to(dtype=pred_kw.dtype)

        if mode == "clip_only":
            pred_kw = apply_clip_projection(pred_kw, ws_proxy, phy)

        n_total += int(pred_kw.numel())
        n_over += int((pred_kw > float(phy.p_rated_kw)).sum().item())
        cut_mask = (ws_proxy < float(phy.v_in_mps)) | (ws_proxy > float(phy.v_out_mps))
        n_cut_nonzero += int(((pred_kw > 0.0) & cut_mask).sum().item())

        env_hi = power_envelope_pu(ws_proxy, phy) * float(phy.p_rated_kw)
        env_dev = torch.relu(pred_kw - env_hi) + torch.relu(-pred_kw)
        env_dev_chunks.append(env_dev.detach().cpu().numpy().reshape(-1))
        all_pred.append(pred_kw.detach().cpu().numpy())
        all_true.append(true_kw.detach().cpu().numpy())

    pred_np = np.concatenate(all_pred, axis=0)
    true_np = np.concatenate(all_true, axis=0)
    env_np = np.concatenate(env_dev_chunks, axis=0) if env_dev_chunks else np.zeros((0,), dtype=np.float64)
    metrics = compute_metrics(pred_np, true_np)
    return {
        "model_name": model_name,
        "r2": float(metrics["R2"]),
        "rmse": float(metrics["RMSE"]),
        "exceedance_rate": 100.0 * float(n_over) / max(float(n_total), 1.0),
        "cut_region_violations": 100.0 * float(n_cut_nonzero) / max(float(n_total), 1.0),
        "envelope_deviation": float(np.mean(env_np)) if env_np.size else 0.0,
        "p95_latency_ms": p95_latency_ms,
    }


def rows_to_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_unified_log(path: Path, rows: List[Dict[str, object]], env: Dict[str, str]) -> None:
    if not rows:
        return
    enriched = []
    for row in rows:
        item = {
            "timestamp": env["timestamp"],
            "mode": env["mode"],
            "experiment_group": row["experiment_group"],
            "model_name": row["model_name"],
            "variant": row["variant"],
            "runtime_backend": "PyTorch",
            "precision": "FP32",
            "jump_gate": "",
            "corruption_type": row["corruption_type"],
            "corruption_level": row["corruption_level"],
            "seed": env["seed"],
            "warmup_iters": env["warmup_iters"],
            "measure_iters": env["measure_iters"],
            "batch_size": env["batch_size"],
            "p50_latency_ms": "",
            "p95_latency_ms": row["p95_latency_ms"],
            "p99_latency_ms": "",
            "peak_memory_mb": "",
            "r2": row["r2"],
            "rmse": row["rmse"],
            "max_abs_diff": "",
            "mean_abs_diff": "",
            "rmse_diff": "",
            "r2_diff": "",
            "exceedance_rate": row["exceedance_rate"],
            "cut_region_violations": row["cut_region_violations"],
            "envelope_deviation": row["envelope_deviation"],
            "skip_ratio": "",
            "preprocess_p95": "",
            "inference_p95": "",
            "postprocess_p95": "",
            "dispatch_p95": "",
            "total_loop_p95": "",
            "deadline_margin_ms": "",
            "data_path": env["data_path"],
            "checkpoint_path": env["checkpoint_path"],
            "torch_version": env["torch_version"],
            "onnxruntime_version": env["onnxruntime_version"],
            "cpu_info": env["cpu_info"],
            "threads": env["threads"],
        }
        enriched.append(item)

    fieldnames = list(enriched[0].keys())
    existing: List[Dict[str, object]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                group = str(row.get("experiment_group", ""))
                if group.startswith("Layer 1: Internal Safety") or group.startswith("Layer 2: Cross-Model Safety"):
                    continue
                existing.append(row)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(enriched)


def main() -> int:
    parser = argparse.ArgumentParser(description="Safety-deployment coupling audit for Ultra-LSNT guardrail runs.")
    parser.add_argument("--mode", choices=["paper", "smoke"], default="paper")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--fg-checkpoint", type=Path, default=DEFAULT_FG_CKPT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-config", type=str, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=0, help="Optional cap for smoke runs; 0 means full loader.")
    args = parser.parse_args()

    hard_fail_if_missing([args.fg_checkpoint], args.mode)
    args.outdir.mkdir(parents=True, exist_ok=True)
    env = snapshot_env(args)

    print("=" * 60)
    print("PAPER MODE SAFETY-DEPLOYMENT AUDIT")
    print("=" * 60)
    for key, value in env.items():
        print(f"{key:20}: {value}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = default_dataset_name(args.data, args.dataset_name)
    output_tag = sanitize_output_tag(args.output_tag) if args.output_tag else infer_output_tag(args.data, dataset_name)
    output_paths = build_safety_output_paths(args.outdir, output_tag)

    data, feature_names, ts_cfg, _, _, test_loader, scaler = load_main_data(
        args.data,
        args.target,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    phy = estimate_curve_params(data, feature_names, target=args.target, train_ratio=0.7)
    wind_proxy_spec = build_wind_proxy_spec(feature_names, scaler, ts_cfg.pred_len)

    ultra = build_backbone_model(
        "ultra",
        ts_cfg,
        device,
        input_dim=len(feature_names),
        data_ref=args.data,
        dataset_name=dataset_name,
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        config_path=args.base_config,
    )
    fg_moe = build_backbone_model(
        "ultra",
        ts_cfg,
        device,
        input_dim=len(feature_names),
        data_ref=args.data,
        dataset_name=dataset_name,
        checkpoint_path=str(args.fg_checkpoint),
        config_path=args.base_config,
    )

    scenarios = [("Clean", 0.0), ("Severe Gaussian", 0.4)]
    variants = [
        ("Ultra-LSNT Raw Output", "raw", ultra),
        ("Ultra-LSNT + Clip-only", "clip_only", ultra),
        ("Ultra-LSNT + FG-MoE", "full_guardrail", fg_moe),
    ]

    rows: List[Dict[str, object]] = []
    for corruption_type, sigma_cfg in scenarios:
        for model_name, mode, model in variants:
            out = evaluate_variant(
                model_name=model_name,
                mode=mode,
                model=model,
                test_loader=test_loader,
                scaler=scaler,
                wind_proxy_spec=wind_proxy_spec,
                phy=phy,
                device=device,
                sigma_cfg=sigma_cfg,
                warmup=args.warmup,
                measure=args.measure,
                max_eval_batches=args.max_eval_batches,
            )
            row = {
                "experiment_group": "Layer 1: Internal Safety",
                "model_name": model_name,
                "corruption_type": corruption_type,
                "corruption_level": 0.0 if sigma_cfg == 0.0 else 0.6,
                "r2": out["r2"],
                "rmse": out["rmse"],
                "exceedance_rate": out["exceedance_rate"],
                "cut_region_violations": out["cut_region_violations"],
                "envelope_deviation": out["envelope_deviation"],
                "p95_latency_ms": out["p95_latency_ms"],
                "variant": mode,
            }
            rows.append(row)
            print(
                f"[{corruption_type:15}] {model_name:26} "
                f"R2={row['r2']:.4f}, exceed={row['exceedance_rate']:.4f}%, "
                f"cut={row['cut_region_violations']:.4f}%, p95={row['p95_latency_ms']:.2f} ms"
            )

    table_rows = [{key: value for key, value in row.items() if key != "variant"} for row in rows]
    rows_to_csv(output_paths["table_csv"], table_rows)
    append_unified_log(output_paths["unified_log_csv"], rows, env)
    output_paths["proxy_json"].write_text(
        json.dumps(proxy_spec_to_dict(wind_proxy_spec), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ok] wrote {output_paths['table_csv']}")
    print(f"[ok] appended {output_paths['unified_log_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
