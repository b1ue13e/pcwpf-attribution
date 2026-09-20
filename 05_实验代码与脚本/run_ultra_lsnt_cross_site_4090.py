#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(cmd: list[str]) -> int:
    print("[cmd]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bidirectional cross-site warm-start runner for Ultra-LSNT SOTA push.")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--dataset-a", type=str, default="wind_final.csv")
    p.add_argument("--target-a", type=str, default="power")
    p.add_argument("--name-a", type=str, default="Wind_CN")
    p.add_argument("--dataset-b", type=str, default="us_wind_2012")
    p.add_argument("--target-b", type=str, default="power")
    p.add_argument("--name-b", type=str, default="Wind_US")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--pretrain-epochs", type=int, default=30)
    p.add_argument("--transfer-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--skip-threshold", type=float, default=0.5)
    p.add_argument("--target-skip-rate", type=float, default=0.12)
    p.add_argument("--skip-budget-weight", type=float, default=1.0)
    p.add_argument("--slow-path-penalty-weight", type=float, default=0.15)
    p.add_argument("--easy-alignment-weight", type=float, default=0.2)
    p.add_argument("--disable-jump-gate-calibration", action="store_true")
    p.add_argument("--jump-gate-threshold-grid", type=str, default="0.20,0.22,0.24,0.25,0.26,0.27,0.28,0.30,0.32,0.36,0.42,0.48")
    p.add_argument("--jump-gate-min-skip", type=float, default=0.02)
    p.add_argument("--jump-gate-score-penalty", type=float, default=0.05)
    p.add_argument("--disable-amp", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-eval-batches", type=int, default=0)
    p.add_argument("--base-out-dir", type=str, default=str(ROOT / "results" / "sota_push_cross_site"))
    p.add_argument("--export-edge-ready", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    runner = str(ROOT / "run_ultra_lsnt_sota_push_4090.py")
    base_out = Path(args.base_out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    common = [
        args.python,
        runner,
        "--epochs",
        str(args.epochs),
        "--pretrain-epochs",
        str(args.pretrain_epochs),
        "--transfer-epochs",
        str(args.transfer_epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--skip-threshold",
        str(args.skip_threshold),
        "--target-skip-rate",
        str(args.target_skip_rate),
        "--skip-budget-weight",
        str(args.skip_budget_weight),
        "--slow-path-penalty-weight",
        str(args.slow_path_penalty_weight),
        "--easy-alignment-weight",
        str(args.easy_alignment_weight),
        "--jump-gate-threshold-grid",
        str(args.jump_gate_threshold_grid),
        "--jump-gate-min-skip",
        str(args.jump_gate_min_skip),
        "--jump-gate-score-penalty",
        str(args.jump_gate_score_penalty),
    ]
    if args.export_edge_ready:
        common.append("--export-edge-ready")
    if args.disable_jump_gate_calibration:
        common.append("--disable-jump-gate-calibration")
    if args.disable_amp:
        common.append("--disable-amp")
    if args.max_train_batches > 0:
        common.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_eval_batches > 0:
        common.extend(["--max-eval-batches", str(args.max_eval_batches)])

    cn_from_us = common + [
        "--out-dir",
        str(base_out / f"{args.name_a.lower()}_from_{args.name_b.lower()}"),
        "--train-data",
        args.dataset_a,
        "--target",
        args.target_a,
        "--train-name",
        args.name_a,
        "--warmstart",
        f"{args.dataset_b}::{args.target_b}::{args.name_b}",
        "--transfer-dataset",
        f"{args.dataset_b}::{args.target_b}::{args.name_b}",
    ]
    rc = run_step(cn_from_us)
    if rc != 0:
        return rc

    us_from_cn = common + [
        "--out-dir",
        str(base_out / f"{args.name_b.lower()}_from_{args.name_a.lower()}"),
        "--train-data",
        args.dataset_b,
        "--target",
        args.target_b,
        "--train-name",
        args.name_b,
        "--warmstart",
        f"{args.dataset_a}::{args.target_a}::{args.name_a}",
        "--transfer-dataset",
        f"{args.dataset_a}::{args.target_a}::{args.name_a}",
    ]
    return run_step(us_from_cn)


if __name__ == "__main__":
    raise SystemExit(main())
