#!/usr/bin/env python3
"""
One-click experiment runner for table backfill on RTX 4090.

What it does:
1) Runs core Wind(CN) auditable suites used by the paper matrices.
2) Runs cross-domain DLinear vs iTransformer / TimeMixer with GPU->CPU retry.
3) Builds an extended cross-domain transfer matrix for multiple models.
4) Rebuilds main4 matrix evidence CSVs from unified sources.

Usage:
  python run_full_backfill_4090.py
  python run_full_backfill_4090.py --continue-on-error
  python run_full_backfill_4090.py --skip-core --skip-rebuild
"""

from __future__ import annotations

import argparse
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
RESULTS_DIR = ROOT / "results"
EVIDENCE_DIR = RESULTS_DIR / "supplementary_evidence"

DATASET_TO_FIELD = {
    "Wind (US)": "R2_wind_us",
    "Air Quality": "R2_air_quality",
    "GEFCom Load": "R2_gefcom_load",
}

DATASET_TO_ALIAS = {
    "Wind (CN)": "wind_cn",
    "Wind (US)": "wind_us",
    "Air Quality": "air_quality",
    "GEFCom Load": "gefcom",
}


@dataclass
class StepResult:
    name: str
    ok: bool
    return_code: int
    log_file: Path
    elapsed_sec: float


def run_step(
    name: str,
    cmd: List[str],
    log_dir: Path,
    env_extra: Optional[Dict[str, str]] = None,
    timeout_sec: Optional[int] = None,
    dry_run: bool = False,
) -> StepResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"
    merged_env = os.environ.copy()
    merged_env["PYTHONIOENCODING"] = "utf-8"
    merged_env["PYTHONUTF8"] = "1"
    if env_extra:
        merged_env.update(env_extra)

    if dry_run:
        with log_file.open("w", encoding="utf-8") as f:
            f.write("DRY RUN\n")
            f.write(" ".join(cmd) + "\n")
        return StepResult(name=name, ok=True, return_code=0, log_file=log_file, elapsed_sec=0.0)

    t0 = time.time()
    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"[step] {name}\n")
        f.write("[cmd] " + " ".join(cmd) + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

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

    elapsed = time.time() - t0
    return StepResult(name=name, ok=(rc == 0), return_code=rc, log_file=log_file, elapsed_sec=elapsed)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_multidomain_with_retry(
    python_exec: str,
    model2: str,
    epochs: int,
    batch_size: int,
    seed: int,
    log_dir: Path,
    continue_on_error: bool,
    dry_run: bool,
    step_results: Optional[List[StepResult]] = None,
) -> Path:
    prefix = f"multi_domain_baselines_r2_{model2}"
    latest_csv = RESULTS_DIR / f"{prefix}_latest.csv"

    step_gpu = run_step(
        name=f"multidomain_{model2}_gpu",
        cmd=[
            python_exec,
            "run_multi_domain_baselines.py",
            "--model2",
            model2,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--seed",
            str(seed),
            "--device",
            "cuda",
            "--datasets",
            "all",
            "--output-prefix",
            prefix,
        ],
        log_dir=log_dir,
        dry_run=dry_run,
    )
    if step_results is not None:
        step_results.append(step_gpu)
    if (not step_gpu.ok) and (not continue_on_error):
        raise RuntimeError(f"Step failed: {step_gpu.name}")

    if dry_run:
        return latest_csv

    df_main = read_csv_safe(latest_csv)
    metric_col = f"{model2}_R2"
    if df_main.empty or metric_col not in df_main.columns:
        if continue_on_error:
            return latest_csv
        raise RuntimeError(f"Missing expected file/column: {latest_csv} / {metric_col}")

    missing_aliases: List[str] = []
    for _, r in df_main.iterrows():
        ds = str(r.get("dataset", "")).strip()
        if ds in DATASET_TO_ALIAS and pd.isna(r.get(metric_col, np.nan)):
            missing_aliases.append(DATASET_TO_ALIAS[ds])

    if not missing_aliases:
        return latest_csv

    retry_prefix = f"{prefix}_cpu_retry"
    retry_csv = RESULTS_DIR / f"{retry_prefix}_latest.csv"
    step_cpu = run_step(
        name=f"multidomain_{model2}_cpu_retry",
        cmd=[
            python_exec,
            "run_multi_domain_baselines.py",
            "--model2",
            model2,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--seed",
            str(seed),
            "--device",
            "cpu",
            "--datasets",
            ",".join(missing_aliases),
            "--output-prefix",
            retry_prefix,
        ],
        log_dir=log_dir,
        dry_run=dry_run,
    )
    if step_results is not None:
        step_results.append(step_cpu)
    if (not step_cpu.ok) and (not continue_on_error):
        raise RuntimeError(f"Step failed: {step_cpu.name}")

    df_retry = read_csv_safe(retry_csv)
    if df_retry.empty:
        return latest_csv

    update_cols = [c for c in df_retry.columns if c != "dataset"]
    idx_main = {str(d): i for i, d in enumerate(df_main["dataset"].astype(str).tolist())}
    for _, rr in df_retry.iterrows():
        ds = str(rr["dataset"])
        if ds not in idx_main:
            continue
        i = idx_main[ds]
        for c in update_cols:
            v = rr.get(c, np.nan)
            if not pd.isna(v):
                df_main.at[i, c] = v
    df_main.to_csv(latest_csv, index=False)
    return latest_csv


def build_extended_transfer_matrix(
    itr_csv: Path,
    tm_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    # model -> three transfer fields
    scores: Dict[str, Dict[str, float]] = {
        "Ultra-LSNT": {k: np.nan for k in DATASET_TO_FIELD.values()},
        "DLinear": {k: np.nan for k in DATASET_TO_FIELD.values()},
        "iTransformer": {k: np.nan for k in DATASET_TO_FIELD.values()},
        "TimeMixer": {k: np.nan for k in DATASET_TO_FIELD.values()},
    }

    # Ultra-LSNT (existing table_multi_domain.csv format)
    ultra_df = read_csv_safe(ROOT / "table_multi_domain.csv")
    if not ultra_df.empty and {"Dataset", "R2 Score"}.issubset(set(ultra_df.columns)):
        for _, r in ultra_df.iterrows():
            ds = str(r["Dataset"]).strip()
            if ds in DATASET_TO_FIELD:
                scores["Ultra-LSNT"][DATASET_TO_FIELD[ds]] = float(r["R2 Score"])

    # iTransformer + DLinear from itr run
    itr_df = read_csv_safe(itr_csv)
    if not itr_df.empty:
        for _, r in itr_df.iterrows():
            ds = str(r.get("dataset", "")).strip()
            if ds not in DATASET_TO_FIELD:
                continue
            fld = DATASET_TO_FIELD[ds]
            if "DLinear_R2" in r.index and not pd.isna(r["DLinear_R2"]):
                scores["DLinear"][fld] = float(r["DLinear_R2"])
            if "itransformer_R2" in r.index and not pd.isna(r["itransformer_R2"]):
                scores["iTransformer"][fld] = float(r["itransformer_R2"])

    # TimeMixer + DLinear from tm run
    tm_df = read_csv_safe(tm_csv)
    if not tm_df.empty:
        for _, r in tm_df.iterrows():
            ds = str(r.get("dataset", "")).strip()
            if ds not in DATASET_TO_FIELD:
                continue
            fld = DATASET_TO_FIELD[ds]
            if "DLinear_R2" in r.index and not pd.isna(r["DLinear_R2"]) and pd.isna(scores["DLinear"][fld]):
                scores["DLinear"][fld] = float(r["DLinear_R2"])
            if "timemixer_R2" in r.index and not pd.isna(r["timemixer_R2"]):
                scores["TimeMixer"][fld] = float(r["timemixer_R2"])

    rows = []
    for m, v in scores.items():
        rows.append(
            {
                "model": m,
                "R2_wind_us": v["R2_wind_us"],
                "R2_air_quality": v["R2_air_quality"],
                "R2_gefcom_load": v["R2_gefcom_load"],
                "source": "multi_domain_baselines + table_multi_domain",
            }
        )
    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df


def patch_matrix2_spike_from_structured(evidence_dir: Path) -> int:
    """
    Backfill missing R2_spike_06 values in matrix2 from structured robustness CSV.
    This guards against older matrix builders that may miss TimeMixer spike rows.
    """
    m2_path = evidence_dir / "main4_matrix2_perf_robust_transfer.csv"
    struct_path = evidence_dir / "structured_scada_fault_robustness_windcn.csv"
    if (not m2_path.exists()) or (not struct_path.exists()):
        return 0

    m2 = pd.read_csv(m2_path)
    struct = pd.read_csv(struct_path)
    required_m2 = {"model", "R2_spike_06"}
    required_struct = {"fault_type", "severity", "model", "R2"}
    if (not required_m2.issubset(set(m2.columns))) or (not required_struct.issubset(set(struct.columns))):
        return 0

    updated = 0
    candidates = ["DLinear", "iTransformer", "TimeMixer", "Ultra-LSNT"]
    for model_name in candidates:
        mask_m2 = m2["model"].astype(str).str.strip() == model_name
        if not mask_m2.any():
            continue
        cur = m2.loc[mask_m2, "R2_spike_06"].iloc[0]
        if not pd.isna(cur):
            continue

        s = struct.copy()
        sev = pd.to_numeric(s["severity"], errors="coerce")
        mask_s = (
            s["fault_type"].astype(str).str.strip().str.lower() == "spike"
        ) & np.isclose(sev, 0.6, equal_nan=False) & (
            s["model"].astype(str).str.strip() == model_name
        )
        if not mask_s.any():
            continue
        val = pd.to_numeric(s.loc[mask_s, "R2"], errors="coerce").dropna()
        if val.empty:
            continue
        m2.loc[mask_m2, "R2_spike_06"] = float(val.iloc[0])
        updated += 1

    if updated > 0:
        m2.to_csv(m2_path, index=False)
    return updated


def validate_backfill_outputs(evidence_dir: Path) -> List[str]:
    """
    Validate key matrix cells that should be non-empty for paper tables.
    """
    issues: List[str] = []
    m2_path = evidence_dir / "main4_matrix2_perf_robust_transfer.csv"
    if not m2_path.exists():
        return [f"missing file: {m2_path}"]

    m2 = pd.read_csv(m2_path)
    if m2.empty:
        return [f"empty file: {m2_path}"]
    if "model" not in m2.columns:
        return [f"missing column 'model' in {m2_path}"]

    checks = [
        ("TimeMixer", "R2_spike_06"),
        ("DLinear", "R2_wind_us"),
        ("iTransformer", "R2_air_quality"),
        ("TimeMixer", "R2_gefcom_load"),
    ]
    for model_name, col in checks:
        if col not in m2.columns:
            issues.append(f"missing column '{col}' in {m2_path.name}")
            continue
        row = m2[m2["model"].astype(str).str.strip() == model_name]
        if row.empty:
            issues.append(f"missing row model={model_name} in {m2_path.name}")
            continue
        if pd.isna(row.iloc[0][col]):
            issues.append(f"empty cell in {m2_path.name}: model={model_name}, col={col}")

    return issues


def write_summary(path: Path, step_results: List[StepResult], transfer_csv: Path) -> None:
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "steps": [
            {
                "name": s.name,
                "ok": s.ok,
                "return_code": s.return_code,
                "elapsed_sec": s.elapsed_sec,
                "log_file": str(s.log_file.relative_to(ROOT)),
            }
            for s in step_results
        ],
        "transfer_matrix_csv": str(transfer_csv.relative_to(ROOT)) if transfer_csv.exists() else "",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run full backfill pipeline for tables on RTX4090.")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs-md", type=int, default=10, help="Epochs for multi-domain baseline runs.")
    ap.add_argument("--batch-size-md", type=int, default=128)
    ap.add_argument("--mamba-epochs", type=int, default=8)
    ap.add_argument("--skip-core", action="store_true", help="Skip core Wind(CN) suites.")
    ap.add_argument("--skip-multidomain", action="store_true", help="Skip cross-domain baseline runs.")
    ap.add_argument("--skip-rebuild", action="store_true", help="Skip main4 matrix rebuild.")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument(
        "--allow-missing-cells",
        action="store_true",
        help="Do not fail when key table cells are empty after rebuild.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"full_backfill_4090_{ts}"
    step_results: List[StepResult] = []
    transfer_csv = EVIDENCE_DIR / "multi_domain_transfer_matrix.csv"

    def exec_or_raise(step_name: str, cmd: List[str]) -> None:
        res = run_step(step_name, cmd, log_dir, dry_run=args.dry_run)
        step_results.append(res)
        if (not res.ok) and (not args.continue_on_error):
            raise RuntimeError(f"Step failed: {step_name}")

    try:
        if not args.skip_core:
            exec_or_raise(
                "core_mamba",
                [
                    args.python,
                    "run_mamba_auditable_suite.py",
                    "--device",
                    "cuda",
                    "--epochs",
                    str(args.mamba_epochs),
                    "--out_dir",
                    str(EVIDENCE_DIR),
                    "--audit_tag",
                    "mamba_auditable_suite_4090",
                ],
            )
            exec_or_raise(
                "core_arima_svr",
                [
                    args.python,
                    "run_arima_svr_auditable_suite.py",
                    "--device",
                    "cpu",
                    "--out_dir",
                    str(EVIDENCE_DIR),
                    "--audit_tag",
                    "arima_svr_auditable_suite_4090",
                ],
            )
            exec_or_raise(
                "core_ssa_elm",
                [
                    args.python,
                    "run_ssa_elm_auditable_suite.py",
                    "--out_dir",
                    str(EVIDENCE_DIR),
                    "--audit_tag",
                    "ssa_elm_auditable_suite_4090",
                ],
            )
            exec_or_raise("core_vmd_ceemdan", [args.python, "run_minimal_vmd_pso_ceemdan_woa.py"])
            exec_or_raise("core_pi_moe_bundle", [args.python, "run_pi_moe_evidence_bundle.py"])
            exec_or_raise("core_mustfix_bundle", [args.python, "run_mustfix_supplementary_experiments.py"])
            exec_or_raise("core_lite_clean_eval", [args.python, "evaluate_ultra_lsnt_optimized.py"])

        if not args.skip_multidomain:
            itr_csv = run_multidomain_with_retry(
                python_exec=args.python,
                model2="itransformer",
                epochs=args.epochs_md,
                batch_size=args.batch_size_md,
                seed=args.seed,
                log_dir=log_dir,
                continue_on_error=args.continue_on_error,
                dry_run=args.dry_run,
                step_results=step_results,
            )
            tm_csv = run_multidomain_with_retry(
                python_exec=args.python,
                model2="timemixer",
                epochs=args.epochs_md,
                batch_size=args.batch_size_md,
                seed=args.seed,
                log_dir=log_dir,
                continue_on_error=args.continue_on_error,
                dry_run=args.dry_run,
                step_results=step_results,
            )

            if not args.dry_run:
                transfer_df = build_extended_transfer_matrix(itr_csv=itr_csv, tm_csv=tm_csv, output_csv=transfer_csv)
                print("\n[transfer-matrix]")
                print(transfer_df.to_string(index=False))

        if not args.skip_rebuild:
            exec_or_raise(
                "rebuild_main4_matrices",
                [
                    args.python,
                    "build_main4_matrices_from_evidence.py",
                    "--out_dir",
                    str(EVIDENCE_DIR),
                    "--audit_tag",
                    "main4_matrix_build_4090_full_backfill",
                ],
            )
            if not args.dry_run:
                n_patched = patch_matrix2_spike_from_structured(EVIDENCE_DIR)
                if n_patched > 0:
                    print(f"[post-fix] patched matrix2 R2_spike_06 cells from structured robustness: {n_patched}")

                issues = validate_backfill_outputs(EVIDENCE_DIR)
                if issues:
                    msg = " | ".join(issues)
                    if args.allow_missing_cells:
                        print(f"[WARN] quality gate found issues: {msg}")
                    else:
                        raise RuntimeError(f"quality gate failed: {msg}")

    except RuntimeError as e:
        print(f"[ERROR] {e}")
        summary_path = log_dir / "run_summary.json"
        write_summary(summary_path, step_results, transfer_csv)
        print(f"[summary] {summary_path}")
        return 1

    summary_path = log_dir / "run_summary.json"
    write_summary(summary_path, step_results, transfer_csv)
    print(f"\n[done] logs: {log_dir}")
    print(f"[done] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
