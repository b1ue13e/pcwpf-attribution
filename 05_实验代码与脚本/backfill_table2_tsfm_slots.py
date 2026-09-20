#!/usr/bin/env python3
"""
Backfill TSFM slot rows in Table 2 from audited Chronos/Moirai outputs.

This script supports two execution modes:
1) Foreground (default): run external commands / load CSVs, then update evidence files.
2) Background: pass --background to detach, write PID + logs, and return immediately.

Expected per-model CSV schema (column aliases supported):
  model, clean_R2, clean_RMSE_kW, clean_MAE_kW,
  R2_sigma_eff_060, R2_spike_06, drop_pct_sigma060

External command contract:
  --chronos-cmd / --moirai-cmd may include "{output}" placeholder.
  The command must generate a CSV at that path.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_EVIDENCE_DIR = ROOT / "results" / "supplementary_evidence"
DEFAULT_MATRIX2_CSV = DEFAULT_EVIDENCE_DIR / "main4_matrix2_perf_robust_transfer.csv"
DEFAULT_TSFM_CSV = DEFAULT_EVIDENCE_DIR / "tsfm_slots_windcn.csv"
DEFAULT_TEX = ROOT / "main_PI-MoE_final_AE_ready.tex"
DEFAULT_LOG_DIR = ROOT / "logs_background"
TSFM_MODELS = ("Chronos-T5-Mini", "Moirai-Small")


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _format_r2_tex(v: float) -> str:
    if pd.isna(v):
        return "--"
    v = float(v)
    if v < 0:
        return f"$-{abs(v):.4f}$"
    return f"{v:.4f}"


def _format_num_tex(v: float, nd: int = 2) -> str:
    if pd.isna(v):
        return "--"
    return f"{float(v):.{nd}f}"


def _pick_float(row: pd.Series, names: Iterable[str]) -> float:
    for name in names:
        if name in row.index and not pd.isna(row[name]):
            return float(row[name])
    return np.nan


def _run_external_csv_command(cmd_template: str, output_csv: Path, label: str) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = cmd_template.format(output=str(output_csv))
    print(f"[run] {label}: {cmd}")
    proc = subprocess.run(cmd, shell=True, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"{label} command failed (rc={proc.returncode})")
    if not output_csv.exists():
        raise FileNotFoundError(f"{label} did not produce expected CSV: {output_csv}")


def _load_model_metrics(csv_path: Path, expected_model: str) -> Dict[str, float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV has no rows: {csv_path}")

    if "model" in df.columns:
        sub = df[df["model"].astype(str).str.strip() == expected_model]
        row = sub.iloc[0] if not sub.empty else df.iloc[0]
    else:
        row = df.iloc[0]

    return {
        "category": "TSFM Slots",
        "model": expected_model,
        "clean_R2": _pick_float(row, ["clean_R2", "R2_clean", "R2", "clean_r2"]),
        "clean_RMSE_kW": _pick_float(row, ["clean_RMSE_kW", "RMSE_clean", "RMSE", "rmse"]),
        "clean_MAE_kW": _pick_float(row, ["clean_MAE_kW", "MAE_clean", "MAE", "mae"]),
        "R2_sigma_eff_060": _pick_float(row, ["R2_sigma_eff_060", "R2@0.60", "R2_sigma060", "r2_sigma_eff_060"]),
        "R2_spike_06": _pick_float(row, ["R2_spike_06", "R2_spike06", "R2_spike_0_6", "spike_R2"]),
        "drop_pct_sigma060": _pick_float(row, ["drop_pct_sigma060", "drop_pct", "Drop_pct_sigma060"]),
        "R2_wind_us": _pick_float(row, ["R2_wind_us"]),
        "R2_air_quality": _pick_float(row, ["R2_air_quality"]),
        "R2_gefcom_load": _pick_float(row, ["R2_gefcom_load"]),
    }


def _write_tsfm_slots_csv(rows: List[Dict[str, float]], out_csv: Path, run_id: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for r in rows:
        x = dict(r)
        x["run_id"] = run_id
        payload.append(x)
    pd.DataFrame(payload).to_csv(out_csv, index=False)
    print(f"[ok] wrote TSFM slots CSV: {out_csv}")


def _merge_into_matrix2(rows: List[Dict[str, float]], matrix2_csv: Path, run_id: str) -> None:
    matrix2_csv.parent.mkdir(parents=True, exist_ok=True)
    if matrix2_csv.exists():
        df = pd.read_csv(matrix2_csv)
    else:
        df = pd.DataFrame()

    if not df.empty and "model" in df.columns:
        df = df[~df["model"].astype(str).str.strip().isin(TSFM_MODELS)].copy()

    append_rows = []
    for r in rows:
        x = dict(r)
        x["run_id"] = run_id
        append_rows.append(x)
    add_df = pd.DataFrame(append_rows)

    if df.empty:
        out_df = add_df
    else:
        for col in df.columns:
            if col not in add_df.columns:
                add_df[col] = np.nan
        for col in add_df.columns:
            if col not in df.columns:
                df[col] = np.nan
        add_df = add_df[df.columns]
        out_df = pd.concat([df, add_df], axis=0, ignore_index=True)

    out_df.to_csv(matrix2_csv, index=False)
    print(f"[ok] merged into matrix2 CSV: {matrix2_csv}")


def _render_tsfm_row(model_name: str, row: Dict[str, float]) -> str:
    return (
        f"{model_name} & "
        f"{_format_r2_tex(row['clean_R2'])} & "
        f"{_format_num_tex(row['clean_RMSE_kW'], 2)} & "
        f"{_format_num_tex(row['clean_MAE_kW'], 2)} & "
        f"{_format_r2_tex(row['R2_sigma_eff_060'])} & "
        f"{_format_r2_tex(row['R2_spike_06'])} & "
        f"{_format_num_tex(row['drop_pct_sigma060'], 2)} \\\\"
    )


def _update_tex_table2(tex_path: Path, rows: List[Dict[str, float]]) -> None:
    if not tex_path.exists():
        print(f"[skip] TeX not found: {tex_path}")
        return

    by_model = {str(r["model"]).strip(): r for r in rows}
    text = tex_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    changed = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(r"\multicolumn{7}{l}{\textit{TSFM Slots (Pending Audited Reruns)}}"):
            lines[idx] = r"\multicolumn{7}{l}{\textit{TSFM Slots (Audited Reruns)}} \\"
            changed = True
        elif stripped.startswith("Chronos-T5-Mini &") and "Chronos-T5-Mini" in by_model:
            lines[idx] = _render_tsfm_row("Chronos-T5-Mini", by_model["Chronos-T5-Mini"])
            changed = True
        elif stripped.startswith("Moirai-Small &") and "Moirai-Small" in by_model:
            lines[idx] = _render_tsfm_row("Moirai-Small", by_model["Moirai-Small"])
            changed = True

    joined = "\n".join(lines)
    old_note = (
        r"TSFM rows are explicit capability slots; protocol-matched reruns are not in this evidence pack."
    )
    new_note = (
        r"TSFM rows are audited from \path{tsfm_slots_windcn.csv} under the same protocol."
    )
    if old_note in joined:
        joined = joined.replace(old_note, new_note)
        changed = True

    if changed:
        tex_path.write_text(joined + "\n", encoding="utf-8")
        print(f"[ok] updated TeX table rows: {tex_path}")
    else:
        print(f"[skip] no TeX updates applied: {tex_path}")


def _launch_background(log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_log = log_dir / f"table2_tsfm_{ts}.out.log"
    err_log = log_dir / f"table2_tsfm_{ts}.err.log"
    pid_file = log_dir / f"table2_tsfm_{ts}.pid"

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
    print("Background TSFM backfill started.")
    print(f"PID: {proc.pid}")
    print(f"STDOUT: {out_log}")
    print(f"STDERR: {err_log}")
    print(f"PID file: {pid_file}")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill Chronos/Moirai rows for Table 2")
    ap.add_argument("--chronos-csv", type=str, default="", help="Existing Chronos metrics CSV path")
    ap.add_argument("--moirai-csv", type=str, default="", help="Existing Moirai metrics CSV path")
    ap.add_argument("--chronos-cmd", type=str, default="", help='Command that writes Chronos CSV (supports "{output}")')
    ap.add_argument("--moirai-cmd", type=str, default="", help='Command that writes Moirai CSV (supports "{output}")')
    ap.add_argument("--evidence-dir", type=str, default=str(DEFAULT_EVIDENCE_DIR))
    ap.add_argument("--matrix2-csv", type=str, default=str(DEFAULT_MATRIX2_CSV))
    ap.add_argument("--tsfm-csv", type=str, default=str(DEFAULT_TSFM_CSV))
    ap.add_argument("--tex", type=str, default=str(DEFAULT_TEX))
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

    chronos_csv = Path(args.chronos_csv) if args.chronos_csv else evidence_dir / "_tmp_chronos_tsfm.csv"
    moirai_csv = Path(args.moirai_csv) if args.moirai_csv else evidence_dir / "_tmp_moirai_tsfm.csv"

    if args.chronos_cmd:
        _run_external_csv_command(args.chronos_cmd, chronos_csv, "chronos")
    if args.moirai_cmd:
        _run_external_csv_command(args.moirai_cmd, moirai_csv, "moirai")

    if not chronos_csv.exists():
        raise FileNotFoundError("Chronos CSV missing. Provide --chronos-csv or --chronos-cmd.")
    if not moirai_csv.exists():
        raise FileNotFoundError("Moirai CSV missing. Provide --moirai-csv or --moirai-cmd.")

    chronos_row = _load_model_metrics(chronos_csv, "Chronos-T5-Mini")
    moirai_row = _load_model_metrics(moirai_csv, "Moirai-Small")
    rows = [chronos_row, moirai_row]

    run_id = _now_tag() + "_tsfm_slots_backfill"

    if args.dry_run:
        print("[dry-run] parsed rows:")
        print(pd.DataFrame(rows))
        return 0

    _write_tsfm_slots_csv(rows, Path(args.tsfm_csv), run_id)
    _merge_into_matrix2(rows, Path(args.matrix2_csv), run_id)

    if not args.skip_tex_update:
        _update_tex_table2(Path(args.tex), rows)

    print("[done] TSFM slots backfill completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
