#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "run_ultra_lsnt_sota_push_4090.py"
ENSEMBLER = ROOT / "run_ultra_lsnt_sota_ensemble_4090.py"


PRESETS: Dict[str, Dict[str, str]] = {
    "balanced": {
        "hidden_dim": "256",
        "num_heads": "8",
        "num_blocks": "4",
        "num_experts": "8",
        "top_k": "2",
        "dropout": "0.15",
        "lr": "3e-4",
        "fault_prob": "0.7",
        "robust_weight": "0.5",
    },
    "wide": {
        "hidden_dim": "320",
        "num_heads": "8",
        "num_blocks": "4",
        "num_experts": "8",
        "top_k": "2",
        "dropout": "0.12",
        "lr": "2.5e-4",
        "fault_prob": "0.75",
        "robust_weight": "0.55",
    },
    "deep": {
        "hidden_dim": "256",
        "num_heads": "8",
        "num_blocks": "5",
        "num_experts": "10",
        "top_k": "2",
        "dropout": "0.18",
        "lr": "2.5e-4",
        "fault_prob": "0.75",
        "robust_weight": "0.6",
    },
    "aggressive": {
        "hidden_dim": "384",
        "num_heads": "8",
        "num_blocks": "5",
        "num_experts": "10",
        "top_k": "2",
        "dropout": "0.12",
        "lr": "2e-4",
        "fault_prob": "0.8",
        "robust_weight": "0.65",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curated config search for Ultra-LSNT SOTA push on RTX 4090.")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--train-data", type=str, default="wind_final.csv")
    p.add_argument("--target", type=str, default="power")
    p.add_argument("--train-name", type=str, default="Wind_CN")
    p.add_argument("--warmstart", action="append", default=[], help="Dataset spec: path::target::name")
    p.add_argument("--transfer-dataset", action="append", default=[], help="Dataset spec: path::target::name")
    p.add_argument("--presets", type=str, default="balanced,wide,deep")
    p.add_argument("--seeds", type=str, default="42,52")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--pretrain-epochs", type=int, default=30)
    p.add_argument("--transfer-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=str(ROOT / "results" / "sota_search_4090"))
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
    p.add_argument("--export-best-edge-ready", action="store_true")
    p.add_argument("--build-ensemble", action="store_true")
    p.add_argument("--ensemble-size", type=int, default=3)
    p.add_argument("--ensemble-candidate-pool", type=int, default=6)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-eval-batches", type=int, default=0)
    return p.parse_args()


def read_metrics_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_score(metrics_rows: List[Dict[str, str]]) -> Dict[str, float]:
    clean_r2 = float("nan")
    severe_r2 = float("nan")
    clean_rmse = float("nan")
    severe_rmse = float("nan")
    clean_skip = float("nan")
    clean_slow = float("nan")
    for row in metrics_rows:
        fault = row["fault"]
        severity = float(row["severity"])
        if fault == "clean":
            clean_r2 = float(row["R2"])
            clean_rmse = float(row["RMSE"])
            clean_skip = float(row.get("avg_skip_rate", "nan"))
            clean_slow = float(row.get("avg_slow_ratio", "nan"))
        if fault == "gaussian" and abs(severity - 0.4) < 1e-9:
            severe_r2 = float(row["R2"])
            severe_rmse = float(row["RMSE"])
    score = clean_r2 + 0.7 * severe_r2
    return {
        "clean_R2": clean_r2,
        "gaussian_severe_R2": severe_r2,
        "clean_RMSE": clean_rmse,
        "gaussian_severe_RMSE": severe_rmse,
        "clean_skip_rate": clean_skip,
        "clean_slow_ratio": clean_slow,
        "selection_score": score,
    }


def run_one(cmd: List[str], cwd: Path) -> int:
    print("[cmd]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd))
    return int(proc.returncode)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    presets = [name.strip() for name in args.presets.split(",") if name.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    unknown = [name for name in presets if name not in PRESETS]
    if unknown:
        raise SystemExit(f"Unknown preset(s): {unknown}. Available: {sorted(PRESETS)}")

    leaderboard: List[Dict[str, object]] = []
    best_row: Dict[str, object] | None = None
    best_cmd: List[str] | None = None

    for preset_name in presets:
        preset = PRESETS[preset_name]
        for seed in seeds:
            run_name = f"{preset_name}_seed{seed}"
            run_dir = out_dir / run_name
            cmd = [
                args.python,
                str(TRAINER),
                "--train-data",
                args.train_data,
                "--target",
                args.target,
                "--train-name",
                args.train_name,
                "--out-dir",
                str(run_dir),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--pretrain-epochs",
                str(args.pretrain_epochs),
                "--transfer-epochs",
                str(args.transfer_epochs),
                "--num-workers",
                str(args.num_workers),
                "--hidden-dim",
                preset["hidden_dim"],
                "--num-heads",
                preset["num_heads"],
                "--num-blocks",
                preset["num_blocks"],
                "--num-experts",
                preset["num_experts"],
                "--top-k",
                preset["top_k"],
                "--dropout",
                preset["dropout"],
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
                "--lr",
                preset["lr"],
                "--fault-prob",
                preset["fault_prob"],
                "--robust-weight",
                preset["robust_weight"],
            ]
            if args.disable_amp:
                cmd.append("--disable-amp")
            if args.disable_jump_gate_calibration:
                cmd.append("--disable-jump-gate-calibration")
            if args.max_train_batches > 0:
                cmd.extend(["--max-train-batches", str(args.max_train_batches)])
            if args.max_eval_batches > 0:
                cmd.extend(["--max-eval-batches", str(args.max_eval_batches)])
            for raw in args.warmstart:
                cmd.extend(["--warmstart", raw])
            for raw in args.transfer_dataset:
                cmd.extend(["--transfer-dataset", raw])

            rc = run_one(cmd, ROOT)
            if rc != 0:
                leaderboard.append(
                    {
                        "preset": preset_name,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "status": f"failed_{rc}",
                    }
                )
                continue

            metrics_path = run_dir / f"main_{args.train_name}" / "test_metrics.csv"
            rows = read_metrics_csv(metrics_path)
            metrics = extract_score(rows)
            record = {
                "preset": preset_name,
                "seed": seed,
                "run_dir": str(run_dir),
                "status": "ok",
                **preset,
                **metrics,
            }
            leaderboard.append(record)
            if best_row is None or float(record["selection_score"]) > float(best_row["selection_score"]):
                best_row = record
                best_cmd = cmd

            Path(run_dir / "search_record.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    leaderboard_path = out_dir / "leaderboard.csv"
    if leaderboard:
        keys = sorted({k for row in leaderboard for k in row.keys()})
        with leaderboard_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(leaderboard)

    if best_row is not None:
        (out_dir / "best_run.json").write_text(json.dumps(best_row, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.export_best_edge_ready and best_cmd is not None:
            export_cmd = list(best_cmd)
            export_cmd.append("--export-edge-ready")
            rc = run_one(export_cmd, ROOT)
            if rc != 0:
                raise SystemExit(rc)

    if args.build_ensemble:
        ensemble_cmd = [
            args.python,
            str(ENSEMBLER),
            "--search-dir",
            str(out_dir),
            "--ensemble-size",
            str(args.ensemble_size),
            "--candidate-pool",
            str(args.ensemble_candidate_pool),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ]
        if args.max_eval_batches > 0:
            ensemble_cmd.extend(["--max-eval-batches", str(args.max_eval_batches)])
        rc = run_one(ensemble_cmd, ROOT)
        if rc != 0:
            raise SystemExit(rc)

    print(f"[done] search outputs saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
