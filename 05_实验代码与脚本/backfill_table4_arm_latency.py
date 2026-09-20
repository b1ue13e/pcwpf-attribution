#!/usr/bin/env python3
"""
Backfill ARM latency rows for Table 4 Panel A.

Workflow:
1) Optionally run auditable latency producers (Mamba / VMD-CEEMDAN / ARIMA-SVR / SSA-ELM).
2) Merge produced latencies into latency_fixed.csv.
3) Update Table 4 ARM column in main_PI-MoE_final_AE_ready.tex.

Run in background with --background to detach and log automatically.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_EVIDENCE_DIR = ROOT / "results" / "supplementary_evidence"
DEFAULT_LATENCY_FIXED = ROOT / "latency_fixed.csv"
DEFAULT_TEX = ROOT / "main_PI-MoE_final_AE_ready.tex"
DEFAULT_LOG_DIR = ROOT / "logs_background"

MODEL_ORDER = [
    "Ultra-LSNT(full)",
    "Ultra-LSNT-Lite(orig)",
    "Ultra-LSNT-Lite(opt)",
    "DLinear",
    "TimeMixer",
    "iTransformer",
    "Mamba (auditable)",
    "VMD-PSO-LSTM",
    "CEEMDAN-WOA-GRU",
    "ARIMA (auditable)",
    "SVR (auditable)",
    "SSA-ELM",
]

TABLE4_ROW_MAP = {
    "Ultra-LSNT (full)": "Ultra-LSNT(full)",
    "Ultra-LSNT-Lite (opt.)": "Ultra-LSNT-Lite(opt)",
    "DLinear": "DLinear",
    "Mamba (auditable)": "Mamba (auditable)",
    "Mamba (logged) (bf)": "Mamba (auditable)",
    "VMD-PSO-LSTM": "VMD-PSO-LSTM",
    "VMD-PSO-LSTM (bf)": "VMD-PSO-LSTM",
    "CEEMDAN-WOA-GRU": "CEEMDAN-WOA-GRU",
    "CEEMDAN-WOA-GRU (bf)": "CEEMDAN-WOA-GRU",
    "ARIMA (auditable)": "ARIMA (auditable)",
    "ARIMA (logged) (bf)": "ARIMA (auditable)",
    "SVR (auditable)": "SVR (auditable)",
    "SVR (logged) (bf)": "SVR (auditable)",
    "SSA-ELM": "SSA-ELM",
    "SSA-ELM (bf)": "SSA-ELM",
}


def _to_param_count(v: float) -> float:
    if pd.isna(v):
        return np.nan
    x = float(v)
    if x > 1e5:
        return x
    return x * 1_000_000.0


def _fmt_arm(v: float) -> str:
    if pd.isna(v):
        return "--"
    return f"{float(v):.3f}"


def _run_cmd(cmd: List[str], label: str) -> None:
    print(f"[run] {label}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed: {label} (rc={proc.returncode})")


def _load_base_latency(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    if df.empty or "model" not in df.columns:
        return out
    for _, r in df.iterrows():
        model = str(r["model"]).strip()
        if not model:
            continue
        out[model] = {
            "model": model,
            "mean_ms": float(r["mean_ms"]) if "mean_ms" in r.index and not pd.isna(r["mean_ms"]) else np.nan,
            "p95_ms": float(r["p95_ms"]) if "p95_ms" in r.index and not pd.isna(r["p95_ms"]) else np.nan,
            "std_ms": float(r["std_ms"]) if "std_ms" in r.index and not pd.isna(r["std_ms"]) else np.nan,
            "memory_mb": float(r["memory_mb"]) if "memory_mb" in r.index and not pd.isna(r["memory_mb"]) else np.nan,
            "params": float(r["params"]) if "params" in r.index and not pd.isna(r["params"]) else np.nan,
        }
    return out


def _upsert_row(store: Dict[str, Dict[str, float]], model: str, mean_ms: float, p95_ms: float, std_ms: float, memory_mb: float, params: float) -> None:
    store[model] = {
        "model": model,
        "mean_ms": float(mean_ms) if not pd.isna(mean_ms) else np.nan,
        "p95_ms": float(p95_ms) if not pd.isna(p95_ms) else np.nan,
        "std_ms": float(std_ms) if not pd.isna(std_ms) else np.nan,
        "memory_mb": float(memory_mb) if not pd.isna(memory_mb) else np.nan,
        "params": float(params) if not pd.isna(params) else np.nan,
    }


def _ingest_latency_sources(evidence_dir: Path, store: Dict[str, Dict[str, float]]) -> None:
    mamba_csv = evidence_dir / "mamba_latency_windcn.csv"
    if mamba_csv.exists():
        df = pd.read_csv(mamba_csv)
        if not df.empty:
            r = df.iloc[0]
            _upsert_row(
                store,
                "Mamba (auditable)",
                r.get("latency_mean_ms", np.nan),
                r.get("latency_p95_ms", np.nan),
                r.get("latency_std_ms", np.nan),
                r.get("model_size_mib", np.nan),
                _to_param_count(r.get("active_params_m", np.nan)),
            )

    vmd_csv = evidence_dir / "minimal_vmd_ceemdan_latency_windcn.csv"
    if vmd_csv.exists():
        df = pd.read_csv(vmd_csv)
        for _, r in df.iterrows():
            model = str(r.get("model", "")).strip()
            if model in {"VMD-PSO-LSTM", "CEEMDAN-WOA-GRU"}:
                _upsert_row(
                    store,
                    model,
                    r.get("latency_mean_ms", np.nan),
                    r.get("latency_p95_ms", np.nan),
                    r.get("latency_std_ms", np.nan),
                    r.get("model_size_mib", np.nan),
                    _to_param_count(r.get("active_params_m", np.nan)),
                )

    arima_csv = evidence_dir / "arima_latency_windcn.csv"
    if arima_csv.exists():
        df = pd.read_csv(arima_csv)
        if not df.empty:
            r = df.iloc[0]
            _upsert_row(
                store,
                "ARIMA (auditable)",
                r.get("latency_mean_ms", np.nan),
                r.get("latency_p95_ms", np.nan),
                r.get("latency_std_ms", np.nan),
                r.get("model_size_mib", np.nan),
                _to_param_count(r.get("active_params_m", np.nan)),
            )

    svr_csv = evidence_dir / "svr_latency_windcn.csv"
    if svr_csv.exists():
        df = pd.read_csv(svr_csv)
        if not df.empty:
            r = df.iloc[0]
            _upsert_row(
                store,
                "SVR (auditable)",
                r.get("latency_mean_ms", np.nan),
                r.get("latency_p95_ms", np.nan),
                r.get("latency_std_ms", np.nan),
                r.get("model_size_mib", np.nan),
                _to_param_count(r.get("active_params_m", np.nan)),
            )

    ssa_csv = evidence_dir / "ssa_elm_latency_windcn.csv"
    if ssa_csv.exists():
        df = pd.read_csv(ssa_csv)
        if not df.empty:
            r = df.iloc[0]
            _upsert_row(
                store,
                "SSA-ELM",
                r.get("latency_mean_ms", np.nan),
                r.get("latency_p95_ms", np.nan),
                r.get("latency_std_ms", np.nan),
                r.get("model_size_mib", np.nan),
                _to_param_count(r.get("active_params_m", np.nan)),
            )


def _store_to_df(store: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = list(store.values())
    if not rows:
        return pd.DataFrame(columns=["model", "mean_ms", "p95_ms", "std_ms", "memory_mb", "params"])
    df = pd.DataFrame(rows)
    order_map = {m: i for i, m in enumerate(MODEL_ORDER)}
    df["__ord"] = df["model"].map(lambda x: order_map.get(str(x), 10_000))
    df = df.sort_values(["__ord", "model"]).drop(columns=["__ord"]).reset_index(drop=True)
    return df


def _update_tex_arm_column(tex_path: Path, arm_map: Dict[str, float]) -> None:
    if not tex_path.exists():
        print(f"[skip] TeX not found: {tex_path}")
        return

    lines = tex_path.read_text(encoding="utf-8").splitlines()
    changed = False

    table4_start = None
    table4_end = None
    for i, line in enumerate(lines):
        if table4_start is None and r"\caption{Table 4." in line:
            table4_start = i
            continue
        if table4_start is not None and r"\end{table}" in line:
            table4_end = i
            break

    # Fallback for current manuscript style: locate by table label.
    if table4_start is None or table4_end is None:
        label_idx = None
        for i, line in enumerate(lines):
            if r"\label{tab:master_hardware}" in line:
                label_idx = i
                break
        if label_idx is not None:
            for j in range(label_idx, -1, -1):
                if r"\begin{table}" in lines[j]:
                    table4_start = j
                    break
            if table4_start is not None:
                for j in range(label_idx, len(lines)):
                    if r"\end{table}" in lines[j]:
                        table4_end = j
                        break

    if table4_start is None or table4_end is None:
        print(f"[skip] could not locate Table 4 block in: {tex_path}")
        return

    for i in range(table4_start, table4_end + 1):
        line = lines[i]
        stripped = line.strip()
        for display_name, key in TABLE4_ROW_MAP.items():
            if stripped.startswith(display_name + " &"):
                parts = line.split("&")
                if len(parts) >= 5:
                    parts[2] = f" {_fmt_arm(arm_map.get(key, np.nan))} "
                    lines[i] = "&".join(parts)
                    changed = True
                break

    text = "\n".join(lines)
    old_tail = r"``--'' denotes unavailable counterpart logs in this evidence pack."
    new_tail = r"ARM rows are backfilled by \path{backfill_table4_arm_latency.py} from auditable latency CSVs."
    if old_tail in text:
        text = text.replace(old_tail, new_tail)
        changed = True

    if changed:
        tex_path.write_text(text + "\n", encoding="utf-8")
        print(f"[ok] updated TeX ARM column: {tex_path}")
    else:
        print(f"[skip] no TeX updates applied: {tex_path}")


def _launch_background(log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_log = log_dir / f"table4_arm_{ts}.out.log"
    err_log = log_dir / f"table4_arm_{ts}.err.log"
    pid_file = log_dir / f"table4_arm_{ts}.pid"

    child_argv = [a for a in sys.argv[1:] if a != "--background"]
    cmd = [sys.executable, str(Path(__file__).resolve()), *child_argv]

    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )

    with out_log.open("w", encoding="utf-8") as fout, err_log.open("w", encoding="utf-8") as ferr:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=fout,
            stderr=ferr,
            text=True,
            creationflags=creationflags,
            close_fds=(os.name != "nt"),
        )

    pid_file.write_text(str(proc.pid), encoding="ascii")
    print("Background ARM backfill started.")
    print(f"PID: {proc.pid}")
    print(f"STDOUT: {out_log}")
    print(f"STDERR: {err_log}")
    print(f"PID file: {pid_file}")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill Table 4 ARM latency rows")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--evidence-dir", type=str, default=str(DEFAULT_EVIDENCE_DIR))
    ap.add_argument("--latency-fixed", type=str, default=str(DEFAULT_LATENCY_FIXED))
    ap.add_argument("--tex", type=str, default=str(DEFAULT_TEX))
    ap.add_argument("--skip-runs", action="store_true", help="Skip running producers; only merge existing CSVs")
    ap.add_argument("--skip-tex-update", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.background:
        return _launch_background(Path(args.log_dir))

    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    latency_fixed = Path(args.latency_fixed)

    if not args.skip_runs:
        _run_cmd(
            [args.python, "run_mamba_auditable_suite.py", "--device", "cpu", "--out_dir", str(evidence_dir), "--audit_tag", "arm_latency_backfill_mamba"],
            "mamba_auditable_suite",
        )
        _run_cmd([args.python, "run_minimal_vmd_pso_ceemdan_woa.py"], "minimal_vmd_ceemdan")
        _run_cmd(
            [args.python, "run_arima_svr_auditable_suite.py", "--out_dir", str(evidence_dir), "--audit_tag", "arm_latency_backfill_arima_svr"],
            "arima_svr_auditable_suite",
        )
        _run_cmd(
            [args.python, "run_ssa_elm_auditable_suite.py", "--out_dir", str(evidence_dir), "--audit_tag", "arm_latency_backfill_ssa"],
            "ssa_elm_auditable_suite",
        )

    store = _load_base_latency(latency_fixed)
    _ingest_latency_sources(evidence_dir, store)
    merged_df = _store_to_df(store)

    if args.dry_run:
        print("[dry-run] merged latency table:")
        print(merged_df)
        return 0

    merged_df.to_csv(latency_fixed, index=False)
    print(f"[ok] updated latency fixed CSV: {latency_fixed}")

    audit_csv = evidence_dir / "latency_arm_full_backfill.csv"
    merged_df.to_csv(audit_csv, index=False)
    print(f"[ok] wrote audit latency CSV: {audit_csv}")

    if not args.skip_tex_update:
        arm_map = {str(r["model"]): float(r["mean_ms"]) if not pd.isna(r["mean_ms"]) else np.nan for _, r in merged_df.iterrows()}
        _update_tex_arm_column(Path(args.tex), arm_map)

    print("[done] ARM latency backfill completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
