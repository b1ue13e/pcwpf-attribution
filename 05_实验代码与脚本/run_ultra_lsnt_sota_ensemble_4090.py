#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from robustness_faults import FAULT_CLEAN, apply_fault_tensor
from run_ultra_lsnt_sota_push_4090 import (
    DatasetSpec,
    build_data_bundle,
    inverse_scale_target,
    load_state_payload,
    pick_device,
    save_json,
)
from ultra_lsnt_sota_push import SotaPushConfig, SotaUltraLSNTForecaster
from ultra_lsnt_timeseries import compute_metrics


ROOT = Path(__file__).resolve().parent


@dataclass
class MemberRun:
    tag: str
    run_dir: Path
    checkpoint: Path
    temperature: float
    config: Dict
    selection_score: float
    model: SotaUltraLSNTForecaster


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Greedy weighted ensemble builder for Ultra-LSNT SOTA search runs.")
    p.add_argument("--search-dir", type=str, default=str(ROOT / "results" / "sota_search_4090"))
    p.add_argument("--leaderboard", type=str, default="")
    p.add_argument("--run-dir", action="append", default=[], help="Optional explicit run directories.")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--ensemble-size", type=int, default=3)
    p.add_argument("--candidate-pool", type=int, default=6)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-eval-batches", type=int, default=0)
    p.add_argument("--selection-robust-fault", type=str, default="gaussian")
    p.add_argument("--selection-robust-severity", type=float, default=0.4)
    p.add_argument("--selection-robust-weight", type=float, default=0.7)
    p.add_argument("--ridge", type=float, default=1e-6)
    return p.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_stage_summary(run_dir: Path) -> Dict[str, str]:
    path = run_dir / "stage_summary.csv"
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("stage_name", "").startswith("main_"):
            return row
    raise RuntimeError(f"Could not find main stage in {path}")


def build_bundle_from_manifest(manifest: Dict, *, batch_size: int, num_workers: int, seed: int):
    args = manifest["args"]
    train_spec = manifest["train_spec"]
    spec = DatasetSpec(path=train_spec["path"], target=train_spec["target"], name=train_spec["name"])
    return build_data_bundle(
        spec,
        seq_len=int(args["seq_len"]),
        pred_len=int(args["pred_len"]),
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=float(args["train_ratio"]),
        val_ratio=float(args["val_ratio"]),
        seed=seed,
    )


def collect_predictions(
    member: MemberRun,
    loader,
    scaler,
    *,
    device: torch.device,
    fault: str,
    severity: float,
    seed: int,
    max_batches: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    member.model.eval()
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    with torch.inference_mode():
        for batch_idx, (bx, by) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            if fault != FAULT_CLEAN and severity > 0:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(seed * 1_000_000 + batch_idx)
                bx = apply_fault_tensor(bx, fault=fault, severity=severity, alpha=1.5, generator=gen)
            pred, _ = member.model(bx, temperature=member.temperature)
            preds.append(pred.detach().cpu().numpy())
            trues.append(by.detach().cpu().numpy())

    pred_np = inverse_scale_target(np.concatenate(preds, axis=0), scaler)
    true_np = inverse_scale_target(np.concatenate(trues, axis=0), scaler)
    return pred_np, true_np


def solve_weights(
    pred_slices: Sequence[np.ndarray],
    true_slices: Sequence[np.ndarray],
    slice_weights: Sequence[float],
    *,
    ridge: float,
) -> np.ndarray:
    mats: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for pred_stack, target, weight in zip(pred_slices, true_slices, slice_weights):
        scale = float(np.sqrt(max(weight, 1e-12)))
        mats.append(pred_stack.reshape(pred_stack.shape[0], -1).T * scale)
        ys.append(target.reshape(-1) * scale)
    x = np.concatenate(mats, axis=0)
    y = np.concatenate(ys, axis=0)
    gram = x.T @ x + ridge * np.eye(x.shape[1], dtype=np.float64)
    rhs = x.T @ y
    weights = np.linalg.solve(gram, rhs)
    weights = np.clip(weights, 0.0, None)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        weights = np.ones(x.shape[1], dtype=np.float64)
    weights = weights / weights.sum()
    return weights.astype(np.float64)


def compute_metric_row(pred: np.ndarray, true: np.ndarray, *, label: str, severity: float) -> Dict[str, float | str]:
    metrics = compute_metrics(pred, true)
    return {
        "fault": label,
        "severity": severity,
        "R2": float(metrics["R2"]),
        "RMSE": float(metrics["RMSE"]),
        "MAE": float(metrics["MAE"]),
        "MAPE": float(metrics["MAPE"]),
    }


def score_rows(clean_row: Dict[str, float | str], robust_row: Dict[str, float | str], robust_weight: float) -> float:
    return float(clean_row["R2"]) + float(robust_weight) * float(robust_row["R2"])


def resolve_candidate_rows(args: argparse.Namespace) -> pd.DataFrame:
    if args.run_dir:
        rows = [{"run_dir": raw, "status": "ok", "selection_score": 0.0} for raw in args.run_dir]
        return pd.DataFrame(rows)

    leaderboard = Path(args.leaderboard) if args.leaderboard else Path(args.search_dir) / "leaderboard.csv"
    if not leaderboard.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard}")
    df = pd.read_csv(leaderboard)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if "selection_score" in df.columns:
        df = df.sort_values("selection_score", ascending=False)
    return df.head(max(1, int(args.candidate_pool))).reset_index(drop=True)


def load_members(rows: pd.DataFrame, device: torch.device) -> Tuple[List[MemberRun], Dict]:
    members: List[MemberRun] = []
    first_manifest: Dict | None = None
    for idx, row in rows.iterrows():
        run_dir = Path(str(row["run_dir"])).resolve()
        manifest = load_json(run_dir / "run_manifest.json")
        stage = read_stage_summary(run_dir)
        checkpoint = Path(stage["checkpoint"])
        payload = torch.load(checkpoint, map_location="cpu")
        cfg = payload["config"]
        model = SotaUltraLSNTForecaster(SotaPushConfig(**cfg)).to(device)
        model.load_state_dict(load_state_payload(checkpoint), strict=False)
        model.eval()
        tag = f"member_{idx+1}"
        if "preset" in row and "seed" in row:
            tag = f"{row['preset']}_seed{row['seed']}"
        members.append(
            MemberRun(
                tag=tag,
                run_dir=run_dir,
                checkpoint=checkpoint,
                temperature=float(manifest["args"].get("temperature_end", 0.35)),
                config=cfg,
                selection_score=float(row.get("selection_score", 0.0)),
                model=model,
            )
        )
        if first_manifest is None:
            first_manifest = manifest
        else:
            a0 = first_manifest["args"]
            a1 = manifest["args"]
            fields = ["seq_len", "pred_len", "train_ratio", "val_ratio"]
            mismatched = [name for name in fields if str(a0[name]) != str(a1[name])]
            if mismatched:
                raise RuntimeError(f"Run {run_dir} is incompatible with ensemble fields: {mismatched}")
            if first_manifest["train_spec"] != manifest["train_spec"]:
                raise RuntimeError("All ensemble members must target the same train_spec.")
    if first_manifest is None:
        raise RuntimeError("No valid members found for ensemble.")
    return members, first_manifest


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.search_dir) / "ensemble_topk"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = resolve_candidate_rows(args)
    members, manifest = load_members(rows, device)
    bundle = build_bundle_from_manifest(
        manifest,
        batch_size=max(1, int(args.batch_size)),
        num_workers=int(args.num_workers),
        seed=int(manifest.get("seed", 42)),
    )
    max_eval_batches = args.max_eval_batches if args.max_eval_batches > 0 else None

    val_cache: Dict[str, Dict[Tuple[str, float], Tuple[np.ndarray, np.ndarray]]] = {}
    test_cache: Dict[str, Dict[Tuple[str, float], Tuple[np.ndarray, np.ndarray]]] = {}
    selection_specs = [
        (FAULT_CLEAN, 0.0, 1.0),
        (args.selection_robust_fault, float(args.selection_robust_severity), float(args.selection_robust_weight)),
    ]
    test_specs = [
        (FAULT_CLEAN, 0.0),
        ("gaussian", 0.15),
        ("gaussian", 0.4),
        ("drift", 0.4),
        ("quantization", 0.4),
        ("missingness", 0.4),
    ]

    for idx, member in enumerate(members):
        member_seed = int(manifest.get("seed", 42)) + idx
        val_cache[member.tag] = {}
        test_cache[member.tag] = {}
        for fault, severity, _ in selection_specs:
            val_cache[member.tag][(fault, severity)] = collect_predictions(
                member,
                bundle.val_loader,
                bundle.scaler,
                device=device,
                fault=fault,
                severity=severity,
                seed=member_seed,
                max_batches=max_eval_batches,
            )
        for fault, severity in test_specs:
            test_cache[member.tag][(fault, severity)] = collect_predictions(
                member,
                bundle.test_loader,
                bundle.scaler,
                device=device,
                fault=fault,
                severity=severity,
                seed=member_seed,
                max_batches=max_eval_batches,
            )

    member_rows: List[Dict[str, object]] = []
    for member in members:
        clean_pred, clean_true = val_cache[member.tag][(FAULT_CLEAN, 0.0)]
        robust_pred, robust_true = val_cache[member.tag][
            (args.selection_robust_fault, float(args.selection_robust_severity))
        ]
        clean_row = compute_metric_row(clean_pred, clean_true, label=FAULT_CLEAN, severity=0.0)
        robust_row = compute_metric_row(
            robust_pred,
            robust_true,
            label=args.selection_robust_fault,
            severity=float(args.selection_robust_severity),
        )
        member_rows.append(
            {
                "member": member.tag,
                "selection_score_input": member.selection_score,
                "val_clean_R2": clean_row["R2"],
                "val_clean_RMSE": clean_row["RMSE"],
                "val_robust_R2": robust_row["R2"],
                "val_robust_RMSE": robust_row["RMSE"],
                "val_joint_score": score_rows(clean_row, robust_row, float(args.selection_robust_weight)),
                "checkpoint": str(member.checkpoint),
            }
        )
    pd.DataFrame(member_rows).to_csv(out_dir / "member_validation_scores.csv", index=False)

    selected_indices: List[int] = []
    selected_weights = np.ones(1, dtype=np.float64)
    best_subset_score = -float("inf")

    candidate_indices = list(range(len(members)))
    target_size = min(max(1, int(args.ensemble_size)), len(candidate_indices))

    while len(selected_indices) < target_size:
        best_trial = None
        for cand_idx in candidate_indices:
            if cand_idx in selected_indices:
                continue
            subset = selected_indices + [cand_idx]
            pred_slices = []
            true_slices = []
            slice_weights = []
            for fault, severity, slice_weight in selection_specs:
                stacked = np.stack([val_cache[members[i].tag][(fault, severity)][0] for i in subset], axis=0)
                _, true_np = val_cache[members[subset[0]].tag][(fault, severity)]
                pred_slices.append(stacked)
                true_slices.append(true_np)
                slice_weights.append(slice_weight)
            weights = solve_weights(pred_slices, true_slices, slice_weights, ridge=float(args.ridge))
            clean_pred = np.tensordot(weights, pred_slices[0], axes=(0, 0))
            robust_pred = np.tensordot(weights, pred_slices[1], axes=(0, 0))
            clean_row = compute_metric_row(clean_pred, true_slices[0], label=FAULT_CLEAN, severity=0.0)
            robust_row = compute_metric_row(
                robust_pred,
                true_slices[1],
                label=args.selection_robust_fault,
                severity=float(args.selection_robust_severity),
            )
            joint_score = score_rows(clean_row, robust_row, float(args.selection_robust_weight))
            if best_trial is None or joint_score > best_trial["score"]:
                best_trial = {
                    "subset": subset,
                    "weights": weights,
                    "score": joint_score,
                    "clean_row": clean_row,
                    "robust_row": robust_row,
                }
        if best_trial is None:
            break
        if best_trial["score"] <= best_subset_score + 1e-8 and selected_indices:
            break
        selected_indices = list(best_trial["subset"])
        selected_weights = np.asarray(best_trial["weights"], dtype=np.float64)
        best_subset_score = float(best_trial["score"])

    selected_members = [members[i] for i in selected_indices]
    if not selected_members:
        raise RuntimeError("Ensemble selection failed to choose any member.")

    val_rows: List[Dict[str, float | str]] = []
    test_rows: List[Dict[str, float | str]] = []
    for fault, severity, _ in selection_specs:
        stacked = np.stack([val_cache[m.tag][(fault, severity)][0] for m in selected_members], axis=0)
        _, true_np = val_cache[selected_members[0].tag][(fault, severity)]
        ens_pred = np.tensordot(selected_weights, stacked, axes=(0, 0))
        val_rows.append(compute_metric_row(ens_pred, true_np, label=fault, severity=severity))

    for fault, severity in test_specs:
        stacked = np.stack([test_cache[m.tag][(fault, severity)][0] for m in selected_members], axis=0)
        _, true_np = test_cache[selected_members[0].tag][(fault, severity)]
        ens_pred = np.tensordot(selected_weights, stacked, axes=(0, 0))
        test_rows.append(compute_metric_row(ens_pred, true_np, label=fault, severity=severity))

    pd.DataFrame(val_rows).to_csv(out_dir / "ensemble_val_metrics.csv", index=False)
    pd.DataFrame(test_rows).to_csv(out_dir / "ensemble_test_metrics.csv", index=False)

    summary = {
        "search_dir": str(Path(args.search_dir).resolve()),
        "device": str(device),
        "ensemble_size_requested": int(args.ensemble_size),
        "ensemble_size_selected": len(selected_members),
        "candidate_pool": int(args.candidate_pool),
        "selection_robust_fault": args.selection_robust_fault,
        "selection_robust_severity": float(args.selection_robust_severity),
        "selection_robust_weight": float(args.selection_robust_weight),
        "selected_members": [
            {
                "member": member.tag,
                "run_dir": str(member.run_dir),
                "checkpoint": str(member.checkpoint),
                "weight": float(weight),
            }
            for member, weight in zip(selected_members, selected_weights)
        ],
        "best_subset_score": float(best_subset_score),
        "val_metrics": val_rows,
        "test_metrics": test_rows,
    }
    save_json(out_dir / "ensemble_summary.json", summary)
    print(f"[done] ensemble outputs saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
