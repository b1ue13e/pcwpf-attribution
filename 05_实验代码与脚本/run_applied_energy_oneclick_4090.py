#!/usr/bin/env python3
"""
One-click orchestrator for Applied Energy expansion experiments on RTX 4090.

Pipeline (default):
1) run_submission_oneclick_4090.py
2) run_latest_sota.py (PatchTST/TimeMixer/iTransformer)
3) run_minimal_vmd_pso_ceemdan_woa.py
4) run_extended_metaheuristic_baselines_4090.py
5) run_joint_parameter_analysis_4090.py
6) optional external Chronos/Moirai commands
7) aggregate clean/robustness comparison CSVs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "applied_energy_expansion_4090"


@dataclass
class StepResult:
    name: str
    ok: bool
    return_code: int
    elapsed_sec: float
    log_file: str
    note: str = ""


def run_cmd_step(
    name: str,
    cmd: List[str],
    log_dir: Path,
    env_extra: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
    timeout_sec: Optional[int] = None,
) -> StepResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"
    if env_extra:
        env.update(env_extra)

    if dry_run:
        with log_file.open("w", encoding="utf-8") as f:
            f.write("DRY RUN\n")
            f.write(" ".join(cmd) + "\n")
        return StepResult(name=name, ok=True, return_code=0, elapsed_sec=0.0, log_file=str(log_file))

    t0 = time.time()
    with log_file.open("w", encoding="utf-8") as f:
        f.write("[cmd] " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        rc = 0
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
            proc.wait(timeout=timeout_sec)
            rc = int(proc.returncode)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = 124
            msg = f"\n[TIMEOUT] step {name} exceeded {timeout_sec} sec\n"
            print(msg, end="")
            f.write(msg)

    return StepResult(
        name=name,
        ok=(rc == 0),
        return_code=rc,
        elapsed_sec=time.time() - t0,
        log_file=str(log_file),
    )


def run_shell_step(
    name: str,
    shell_cmd: str,
    log_dir: Path,
    dry_run: bool = False,
    timeout_sec: Optional[int] = None,
) -> StepResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"
    if dry_run:
        with log_file.open("w", encoding="utf-8") as f:
            f.write("DRY RUN\n")
            f.write(shell_cmd + "\n")
        return StepResult(name=name, ok=True, return_code=0, elapsed_sec=0.0, log_file=str(log_file))

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    t0 = time.time()
    with log_file.open("w", encoding="utf-8") as f:
        f.write("[shell] " + shell_cmd + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            shell_cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=True,
        )
        rc = 0
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
            proc.wait(timeout=timeout_sec)
            rc = int(proc.returncode)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = 124
            msg = f"\n[TIMEOUT] step {name} exceeded {timeout_sec} sec\n"
            print(msg, end="")
            f.write(msg)
    return StepResult(
        name=name,
        ok=(rc == 0),
        return_code=rc,
        elapsed_sec=time.time() - t0,
        log_file=str(log_file),
    )


def append_or_raise(step: StepResult, results: List[StepResult], continue_on_error: bool) -> None:
    results.append(step)
    if (not step.ok) and (not continue_on_error):
        raise RuntimeError(f"Step failed: {step.name}")


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def aggregate_tables(out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_rows: List[pd.DataFrame] = []
    robust_rows: List[pd.DataFrame] = []

    # Deep SOTA robust csv (sigma==0 is clean)
    deep = read_csv_safe(ROOT / "results" / "supplementary_evidence" / "deep_sota_gaussian_windcn.csv")
    if not deep.empty and {"sigma", "model", "R2", "RMSE", "MAE"}.issubset(deep.columns):
        d0 = deep[deep["sigma"] == 0.0].copy()
        if not d0.empty:
            d0["dataset"] = "wind_cn"
            d0["source"] = "deep_sota_gaussian_windcn.csv"
            clean_rows.append(d0[["dataset", "model", "R2", "RMSE", "MAE", "source"]])
        dr = deep[deep["sigma"] > 0.0].copy()
        if not dr.empty:
            dr["dataset"] = "wind_cn"
            dr["fault_type"] = "gaussian"
            dr["severity"] = dr["sigma"]
            dr["sigma_eff"] = dr["sigma"]
            dr["source"] = "deep_sota_gaussian_windcn.csv"
            robust_rows.append(dr[["dataset", "model", "fault_type", "severity", "sigma_eff", "R2", "RMSE", "MAE", "source"]])

    for p in [
        ROOT / "results" / "supplementary_evidence" / "minimal_vmd_ceemdan_clean_windcn.csv",
        ROOT / "results" / "supplementary_evidence" / "extended_metaheuristic_clean_windcn.csv",
        ROOT / "results" / "supplementary_evidence" / "mamba_clean_windcn.csv",
    ]:
        df = read_csv_safe(p)
        if not df.empty and {"dataset", "model", "R2", "RMSE", "MAE"}.issubset(df.columns):
            df = df.copy()
            df["source"] = p.name
            clean_rows.append(df[["dataset", "model", "R2", "RMSE", "MAE", "source"]])

    for p in [
        ROOT / "results" / "supplementary_evidence" / "minimal_vmd_ceemdan_robustness_windcn.csv",
        ROOT / "results" / "supplementary_evidence" / "extended_metaheuristic_robustness_windcn.csv",
        ROOT / "results" / "supplementary_evidence" / "mamba_robustness_windcn.csv",
        ROOT / "results" / "supplementary_evidence" / "lite_robustness_windcn.csv",
    ]:
        df = read_csv_safe(p)
        if df.empty or "model" not in df.columns:
            continue
        c = df.copy()
        if "dataset" not in c.columns:
            c["dataset"] = "wind_cn"
        if "fault_type" not in c.columns:
            c["fault_type"] = "unknown"
        if "severity" not in c.columns:
            c["severity"] = np.nan
        if "sigma_eff" not in c.columns:
            c["sigma_eff"] = np.nan
        for col in ["R2", "RMSE", "MAE"]:
            if col not in c.columns:
                c[col] = np.nan
        c["source"] = p.name
        robust_rows.append(c[["dataset", "model", "fault_type", "severity", "sigma_eff", "R2", "RMSE", "MAE", "source"]])

    clean_path = out_dir / "expanded_clean_comparison_windcn.csv"
    robust_path = out_dir / "expanded_robustness_comparison_windcn.csv"
    if clean_rows:
        pd.concat(clean_rows, axis=0, ignore_index=True).drop_duplicates().to_csv(clean_path, index=False)
    if robust_rows:
        pd.concat(robust_rows, axis=0, ignore_index=True).drop_duplicates().to_csv(robust_path, index=False)
    return {"clean_table": str(clean_path), "robust_table": str(robust_path)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-click Applied Energy expansion runner on RTX 4090")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default="42,123,2025")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))

    p.add_argument("--skip-base-bundle", action="store_true")
    p.add_argument("--skip-latest-sota", action="store_true")
    p.add_argument("--skip-minimal-hybrid", action="store_true")
    p.add_argument("--skip-extended-hybrid", action="store_true")
    p.add_argument("--skip-joint-analysis", action="store_true")

    p.add_argument("--latest-epochs", type=int, default=80)
    p.add_argument("--latest-batch-size", type=int, default=256)

    p.add_argument("--joint-n-grid", type=str, default="2,4,8")
    p.add_argument("--joint-topk-grid", type=str, default="1,2")
    p.add_argument("--joint-tau-grid", type=str, default="0.3,0.5,0.7,0.9")
    p.add_argument("--joint-lambda2-grid", type=str, default="0.0,0.01,0.1,1.0")

    p.add_argument("--chronos-cmd", type=str, default="")
    p.add_argument("--moirai-cmd", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"applied_energy_expansion_4090_{ts}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps: List[StepResult] = []
    artifacts: Dict[str, str] = {}

    try:
        if not args.skip_base_bundle:
            cmd = [
                args.python,
                "run_submission_oneclick_4090.py",
                "--seed",
                str(args.seed),
                "--seeds",
                args.seeds,
                "--device",
                args.device,
            ]
            if args.continue_on_error:
                cmd.append("--continue-on-error")
            if args.dry_run:
                cmd.append("--dry-run")
            append_or_raise(
                run_cmd_step("phase1_submission_bundle", cmd, log_dir, dry_run=args.dry_run),
                steps,
                args.continue_on_error,
            )

        if not args.skip_latest_sota:
            cmd = [
                args.python,
                "run_latest_sota.py",
                "--model",
                "all",
                "--data",
                "wind_final.csv",
                "--target",
                "power",
                "--epochs",
                str(args.latest_epochs),
                "--batch_size",
                str(args.latest_batch_size),
                "--seed",
                str(args.seed),
                "--gpu_optimized",
            ]
            append_or_raise(run_cmd_step("phase2_latest_sota", cmd, log_dir, dry_run=args.dry_run), steps, args.continue_on_error)

        if not args.skip_minimal_hybrid:
            cmd = [args.python, "run_minimal_vmd_pso_ceemdan_woa.py"]
            append_or_raise(run_cmd_step("phase3_minimal_hybrid", cmd, log_dir, dry_run=args.dry_run), steps, args.continue_on_error)

        if not args.skip_extended_hybrid:
            cmd = [
                args.python,
                "run_extended_metaheuristic_baselines_4090.py",
                "--seed",
                str(args.seed),
                "--device",
                args.device,
            ]
            append_or_raise(run_cmd_step("phase4_extended_hybrid", cmd, log_dir, dry_run=args.dry_run), steps, args.continue_on_error)

        if not args.skip_joint_analysis:
            cmd = [
                args.python,
                "run_joint_parameter_analysis_4090.py",
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--n_grid",
                args.joint_n_grid,
                "--topk_grid",
                args.joint_topk_grid,
                "--tau_grid",
                args.joint_tau_grid,
                "--lambda2_grid",
                args.joint_lambda2_grid,
            ]
            append_or_raise(run_cmd_step("phase5_joint_analysis", cmd, log_dir, dry_run=args.dry_run), steps, args.continue_on_error)

        if args.chronos_cmd.strip():
            append_or_raise(run_shell_step("phase6_chronos_external", args.chronos_cmd, log_dir, dry_run=args.dry_run), steps, args.continue_on_error)
        if args.moirai_cmd.strip():
            append_or_raise(run_shell_step("phase7_moirai_external", args.moirai_cmd, log_dir, dry_run=args.dry_run), steps, args.continue_on_error)

        if not args.dry_run:
            artifacts.update(aggregate_tables(out_dir))

        for p in glob.glob(str(ROOT / "results" / "supplementary_evidence" / "*.csv")):
            name = Path(p).name
            if name.startswith("joint_") or name.startswith("extended_metaheuristic") or name.startswith("minimal_vmd_ceemdan"):
                artifacts[name] = p

    except Exception as e:
        summary_path = out_dir / f"run_summary_{ts}.json"
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "script": "run_applied_energy_oneclick_4090.py",
            "args": vars(args),
            "status": "failed",
            "error": str(e),
            "steps": [step.__dict__ for step in steps],
            "artifacts": artifacts,
            "log_dir": str(log_dir),
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[failed] summary: {summary_path}")
        return 1

    summary_path = out_dir / f"run_summary_{ts}.json"
    latest_summary = out_dir / "run_summary_latest.json"
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "script": "run_applied_energy_oneclick_4090.py",
        "args": vars(args),
        "status": "ok",
        "steps": [step.__dict__ for step in steps],
        "artifacts": artifacts,
        "log_dir": str(log_dir),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with latest_summary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[done] logs: {log_dir}")
    print(f"[done] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

