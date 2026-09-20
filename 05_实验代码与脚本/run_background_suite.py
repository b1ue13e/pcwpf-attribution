#!/usr/bin/env python3
"""
Background experiment suite for paper supplements and Ultra-LSNT-Lite comparison.

Usage:
  python run_background_suite.py
  python run_background_suite.py --tasks lite_suite
  python run_background_suite.py --tasks multi_domain_baselines,dm_extended,paper_digest --continue-on-error
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_TASKS = (
    "multi_domain,multi_domain_baselines,kernel_ablation,noise20,dm_extended,paper_digest,lite_suite"
)


@dataclass
class TaskResult:
    name: str
    ok: bool
    duration_sec: float
    log_file: str
    error: str = ""
    outputs: List[str] = field(default_factory=list)


class SuiteRunner:
    def __init__(self, python_exec: str, log_dir: Path, continue_on_error: bool, dry_run: bool) -> None:
        self.python_exec = python_exec
        self.log_dir = log_dir
        self.continue_on_error = continue_on_error
        self.dry_run = dry_run
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.env = os.environ.copy()
        self.env["PYTHONIOENCODING"] = "utf-8"
        self.env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.summary: List[TaskResult] = []

    def _task_log_path(self, task_name: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.log_dir / f"{task_name}_{ts}.log"

    def run_script(self, task_name: str, script_name: str) -> TaskResult:
        log_path = self._task_log_path(task_name)
        cmd = [self.python_exec, script_name]
        return self._run_command(task_name, cmd, log_path)

    def _run_command(self, task_name: str, cmd: List[str], log_path: Path) -> TaskResult:
        started = time.time()
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"[{datetime.now().isoformat()}] Task: {task_name}\n")
            logf.write(f"Command: {' '.join(cmd)}\n\n")
            logf.flush()

            if self.dry_run:
                logf.write("Dry-run mode: command not executed.\n")
                return TaskResult(
                    name=task_name,
                    ok=True,
                    duration_sec=0.0,
                    log_file=str(log_path.relative_to(ROOT)),
                )

            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    env=self.env,
                )
            except Exception as exc:
                return TaskResult(
                    name=task_name,
                    ok=False,
                    duration_sec=time.time() - started,
                    log_file=str(log_path.relative_to(ROOT)),
                    error=f"Failed to start process: {exc}",
                )

            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
            proc.wait()
            ok = proc.returncode == 0
            err = "" if ok else f"Process exited with code {proc.returncode}"
            return TaskResult(
                name=task_name,
                ok=ok,
                duration_sec=time.time() - started,
                log_file=str(log_path.relative_to(ROOT)),
                error=err,
            )

    def run_dm_extended(self) -> TaskResult:
        task_name = "dm_extended"
        log_path = self._task_log_path(task_name)
        started = time.time()
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"[{datetime.now().isoformat()}] Task: {task_name}\n")
            if self.dry_run:
                logf.write("Dry-run mode: task not executed.\n")
                return TaskResult(
                    name=task_name,
                    ok=True,
                    duration_sec=0.0,
                    log_file=str(log_path.relative_to(ROOT)),
                )

            try:
                from run_dm_test import dm_test_newey_west, get_model_preds
            except Exception as exc:
                return TaskResult(
                    name=task_name,
                    ok=False,
                    duration_sec=time.time() - started,
                    log_file=str(log_path.relative_to(ROOT)),
                    error=f"Import failed: {exc}",
                )

            logf.write("Loading Ultra-LSNT predictions...\n")
            preds_ultra, trues_ultra = get_model_preds()
            if preds_ultra is None or trues_ultra is None:
                return TaskResult(
                    name=task_name,
                    ok=False,
                    duration_sec=time.time() - started,
                    log_file=str(log_path.relative_to(ROOT)),
                    error="Ultra-LSNT predictions unavailable.",
                )

            ultra = np.asarray(preds_ultra).reshape(-1)
            ultra_true = np.asarray(trues_ultra).reshape(-1)
            logf.write(f"Ultra-LSNT samples: {len(ultra)}\n")

            def load_dlinear_preds() -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
                try:
                    import torch
                    from run_dlinear import DLinear
                    from ultra_lsnt_timeseries import TimeSeriesConfig, TrainConfig, create_dataloaders, load_csv_data
                except Exception as exc:
                    return None, None, f"import failed: {exc}"

                weight_path = ROOT / "dlinear_weights" / "dlinear_trained.pth"
                if not weight_path.exists():
                    return None, None, f"missing weights: {weight_path}"

                try:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    data, _ = load_csv_data(str(ROOT / "wind_final.csv"), "power")
                    if data is None:
                        return None, None, "load_csv_data returned None"

                    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target="power")
                    train_config = TrainConfig(batch_size=256, num_workers=0)
                    _, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)

                    model = DLinear(ts_config.seq_len, ts_config.pred_len, data.shape[1]).to(device)
                    ckpt = torch.load(weight_path, map_location=device)
                    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                    model.load_state_dict(state, strict=False)
                    model.eval()

                    preds, trues = [], []
                    with torch.no_grad():
                        for bx, by in test_loader:
                            bx = bx.to(device)
                            pred = model(bx)
                            preds.append(pred.cpu().numpy())
                            trues.append(by.numpy())

                    p = np.concatenate(preds, axis=0) * scaler.std[-1] + scaler.mean[-1]
                    y = np.concatenate(trues, axis=0) * scaler.std[-1] + scaler.mean[-1]
                    return p.reshape(-1), y.reshape(-1), ""
                except Exception as exc:
                    return None, None, f"inference failed: {exc}"

            baselines: Dict[str, Tuple[Path, Path]] = {
                "PatchTST": (ROOT / "checkpoints_ts" / "PatchTST_preds.npy", ROOT / "checkpoints_ts" / "PatchTST_trues.npy"),
                "TimeMixer": (ROOT / "checkpoints_ts" / "TimeMixer_preds.npy", ROOT / "checkpoints_ts" / "TimeMixer_trues.npy"),
                "iTransformer": (
                    ROOT / "checkpoints_ts" / "iTransformer_preds.npy",
                    ROOT / "checkpoints_ts" / "iTransformer_trues.npy",
                ),
            }

            comparisons: List[Dict[str, float]] = []
            d_pred, d_true, d_err = load_dlinear_preds()
            if d_pred is None or d_true is None:
                logf.write(f"[WARN] DLinear baseline unavailable for DM: {d_err}\n")
            else:
                n = min(len(ultra), len(ultra_true), len(d_pred), len(d_true))
                u = ultra[:n]
                y_u = ultra_true[:n]
                b = d_pred[:n]
                y_b = d_true[:n]
                mismatch = float(np.mean(np.abs(y_u - y_b)))
                actual = (y_u + y_b) / 2.0 if mismatch > 1e-6 else y_u
                dm_stat, p_val = dm_test_newey_west(actual, u, b, criterion="MSE")
                comparisons.append(
                    {
                        "baseline": "DLinear",
                        "samples": float(n),
                        "dm_stat": float(dm_stat),
                        "p_value": float(p_val),
                        "true_mismatch_mae": mismatch,
                    }
                )
                logf.write(
                    f"DLinear: n={n}, DM={dm_stat:.6f}, p={p_val:.6e}, true_mismatch_mae={mismatch:.6e}\n"
                )

            for name, (pred_path, true_path) in baselines.items():
                if not pred_path.exists() or not true_path.exists():
                    logf.write(f"[WARN] Missing baseline files for {name}: {pred_path}, {true_path}\n")
                    continue
                bpred = np.load(pred_path).reshape(-1)
                btrue = np.load(true_path).reshape(-1)
                n = min(len(ultra), len(ultra_true), len(bpred), len(btrue))
                if n <= 0:
                    logf.write(f"[WARN] Empty arrays for {name}\n")
                    continue

                u = ultra[:n]
                y_u = ultra_true[:n]
                b = bpred[:n]
                y_b = btrue[:n]

                mismatch = float(np.mean(np.abs(y_u - y_b)))
                actual = (y_u + y_b) / 2.0 if mismatch > 1e-6 else y_u
                dm_stat, p_val = dm_test_newey_west(actual, u, b, criterion="MSE")
                comparisons.append(
                    {
                        "baseline": name,
                        "samples": float(n),
                        "dm_stat": float(dm_stat),
                        "p_value": float(p_val),
                        "true_mismatch_mae": mismatch,
                    }
                )
                logf.write(
                    f"{name}: n={n}, DM={dm_stat:.6f}, p={p_val:.6e}, true_mismatch_mae={mismatch:.6e}\n"
                )

            if not comparisons:
                return TaskResult(
                    name=task_name,
                    ok=False,
                    duration_sec=time.time() - started,
                    log_file=str(log_path.relative_to(ROOT)),
                    error="No DM comparisons were computed.",
                )

            dm_dir = ROOT / "dm_results"
            dm_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = dm_dir / f"dm_extended_{ts}.json"
            csv_path = dm_dir / f"dm_extended_{ts}.csv"
            tex_path = dm_dir / f"dm_extended_{ts}.tex"

            payload = {
                "timestamp": datetime.now().isoformat(),
                "dataset": "wind_final.csv",
                "method": "DM test with Newey-West HAC (from run_dm_test.dm_test_newey_west)",
                "comparisons": comparisons,
            }
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["baseline", "samples", "dm_stat", "p_value", "true_mismatch_mae"]
                )
                writer.writeheader()
                for row in comparisons:
                    writer.writerow(row)

            lines = [
                "\\begin{tabular}{lcc}",
                "\\toprule",
                "Comparison & DM statistic & $p$-value \\\\",
                "\\midrule",
            ]
            for row in comparisons:
                lines.append(
                    f"Ultra-LSNT vs {row['baseline']} & {row['dm_stat']:.2f} & ${row['p_value']:.2e}$ \\\\"
                )
            lines.extend(["\\bottomrule", "\\end{tabular}"])
            tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            outputs = [
                str(json_path.relative_to(ROOT)),
                str(csv_path.relative_to(ROOT)),
                str(tex_path.relative_to(ROOT)),
            ]
            return TaskResult(
                name=task_name,
                ok=True,
                duration_sec=time.time() - started,
                log_file=str(log_path.relative_to(ROOT)),
                outputs=outputs,
            )

    def _latest_file(self, pattern: str) -> Optional[Path]:
        files = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None

    def build_paper_digest(self) -> TaskResult:
        task_name = "paper_digest"
        log_path = self._task_log_path(task_name)
        started = time.time()
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"[{datetime.now().isoformat()}] Task: {task_name}\n")
            if self.dry_run:
                logf.write("Dry-run mode: task not executed.\n")
                return TaskResult(
                    name=task_name,
                    ok=True,
                    duration_sec=0.0,
                    log_file=str(log_path.relative_to(ROOT)),
                )

            out_dir = ROOT / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            md_out = out_dir / f"paper_supplement_digest_{ts}.md"
            json_out = out_dir / f"paper_supplement_digest_{ts}.json"

            digest: Dict[str, object] = {
                "generated_at": datetime.now().isoformat(),
            }

            # 1) Multi-domain two baselines
            md_base = self._latest_file("results/multi_domain_baselines_r2_*.csv")
            if md_base is None:
                latest = ROOT / "results" / "multi_domain_baselines_r2_latest.csv"
                md_base = latest if latest.exists() else None

            multi_rows: List[Dict[str, str]] = []
            if md_base and md_base.exists():
                with md_base.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        multi_rows.append(row)
            digest["multi_domain_baselines_file"] = "" if md_base is None else str(md_base.relative_to(ROOT))
            digest["multi_domain_baselines"] = multi_rows

            # 2) Air Quality kernel ablation k=3/7/15
            ablation_file = ROOT / "universal_ablation_results.csv"
            air_k: Dict[str, float] = {}
            if ablation_file.exists():
                with ablation_file.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("Dataset") == "Air Quality" and row.get("Kernel") in {"3", "7", "15"}:
                            air_k[f"k={row['Kernel']}"] = float(row["R2"])
            digest["air_quality_kernel_ablation"] = air_k

            # 3) DM extra comparisons (prefer DLinear + TimeMixer)
            dm_file = self._latest_file("dm_results/dm_extended_*.csv")
            dm_rows: List[Dict[str, str]] = []
            if dm_file and dm_file.exists():
                with dm_file.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        dm_rows.append(row)
            digest["dm_extended_file"] = "" if dm_file is None else str(dm_file.relative_to(ROOT))
            digest["dm_extended"] = dm_rows

            # 4) Dispatch units/settings from code
            dispatch_info = {
                "avg_reserve_power_unit": "kW",
                "total_curtailment_unit": "kWh",
                "optimization_horizon": "1 day (24 time steps)",
                "dispatch_step": "1 hour",
            }
            cfg_file = ROOT / "wind_dispatch_model.py"
            if cfg_file.exists():
                txt = cfg_file.read_text(encoding="utf-8", errors="ignore")
                m_t = re.search(r"'T':\s*([0-9]+)", txt)
                m_dt = re.search(r"'dt':\s*([0-9]+(?:\.[0-9]+)?)", txt)
                if m_t and m_dt:
                    T = m_t.group(1)
                    dt = m_dt.group(1)
                    dispatch_info["optimization_horizon"] = f"{T} steps"
                    dispatch_info["dispatch_step"] = f"{dt} hour"
            digest["dispatch"] = dispatch_info

            # 5) One-sentence protocol difference
            noise_sentence = (
                "noise_20_comparison is a stitched single-point check at sigma=0.2 "
                "(DLinear from an independent run plus stored Ultra-LSNT/LightGBM values), "
                "while progressive sweep is a unified multi-sigma curve evaluation over sigma in [0, 0.4] "
                "under one consistent sweep protocol."
            )
            digest["noise20_vs_sweep_sentence"] = noise_sentence

            # Write markdown
            lines: List[str] = [
                "# Paper Supplement Digest",
                "",
                f"Generated at: {digest['generated_at']}",
                "",
                "## 1) Multi-domain Baselines (>=2 baselines each dataset)",
                f"Source: `{digest['multi_domain_baselines_file']}`",
                "",
            ]
            if multi_rows:
                cols = ["dataset", "DLinear_R2"]
                # detect second baseline column
                second_col = ""
                for k in multi_rows[0].keys():
                    if k.endswith("_R2") and k != "DLinear_R2":
                        second_col = k
                        break
                if second_col:
                    cols.append(second_col)
                header = "| " + " | ".join(cols) + " |"
                sep = "| " + " | ".join(["---"] * len(cols)) + " |"
                lines.extend([header, sep])
                for row in multi_rows:
                    lines.append("| " + " | ".join(row.get(c, "") for c in cols) + " |")
            else:
                lines.append("No multi-domain baseline file found.")

            lines.extend(
                [
                    "",
                    "## 2) Air Quality Kernel Ablation",
                    f"- k=3: {air_k.get('k=3', 'N/A')}",
                    f"- k=7: {air_k.get('k=7', 'N/A')}",
                    f"- k=15: {air_k.get('k=15', 'N/A')}",
                    "",
                    "## 3) DM Test Extra Comparisons (DM and p-value)",
                    f"Source: `{digest['dm_extended_file']}`",
                ]
            )
            preferred = ["DLinear", "TimeMixer", "iTransformer", "PatchTST"]
            if dm_rows:
                for b in preferred:
                    hit = next((r for r in dm_rows if r.get("baseline") == b), None)
                    if hit is not None:
                        lines.append(
                            f"- Ultra-LSNT vs {b}: DM={hit.get('dm_stat')}, p={hit.get('p_value')}"
                        )
            else:
                lines.append("No DM extended file found.")

            lines.extend(
                [
                    "",
                    "## 4) Dispatch Units and Setup",
                    f"- Avg. reserve power unit: {dispatch_info['avg_reserve_power_unit']}",
                    f"- Total curtailment unit: {dispatch_info['total_curtailment_unit']}",
                    f"- Optimization horizon: {dispatch_info['optimization_horizon']}",
                    f"- Dispatch step: {dispatch_info['dispatch_step']}",
                    "",
                    "## 5) noise_20_comparison vs progressive sweep (one sentence)",
                    f"- {noise_sentence}",
                ]
            )

            md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            json_out.write_text(json.dumps(digest, indent=2, ensure_ascii=False), encoding="utf-8")
            logf.write(f"Saved: {md_out}\nSaved: {json_out}\n")

            return TaskResult(
                name=task_name,
                ok=True,
                duration_sec=time.time() - started,
                log_file=str(log_path.relative_to(ROOT)),
                outputs=[str(md_out.relative_to(ROOT)), str(json_out.relative_to(ROOT))],
            )

    def summarize_lite(self) -> TaskResult:
        task_name = "lite_summary"
        log_path = self._task_log_path(task_name)
        started = time.time()
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"[{datetime.now().isoformat()}] Task: {task_name}\n")
            if self.dry_run:
                logf.write("Dry-run mode: task not executed.\n")
                return TaskResult(
                    name=task_name,
                    ok=True,
                    duration_sec=0.0,
                    log_file=str(log_path.relative_to(ROOT)),
                )

            def read_first_row_csv(path: Path) -> Optional[Dict[str, str]]:
                if not path.exists():
                    return None
                with path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        return row
                return None

            time_row = read_first_row_csv(ROOT / "timemixer_efficiency_results.csv")
            lite_row = read_first_row_csv(ROOT / "ultra_lsnt_lite_efficiency_results.csv")
            lite_opt_row = read_first_row_csv(ROOT / "ultra_lsnt_lite_optimized_results.csv")
            lite_opt_perf = read_first_row_csv(ROOT / "ultra_lsnt_lite_optimized_performance.csv")

            full_r2 = None
            full_rmse = None
            full_json = ROOT / "results" / "metrics_corrected.json"
            if full_json.exists():
                data = json.loads(full_json.read_text(encoding="utf-8"))
                if "Ultra-LSNT" in data:
                    full_r2 = data["Ultra-LSNT"].get("r2")
                    full_rmse = data["Ultra-LSNT"].get("rmse")

            timemixer_r2 = None
            tm_json = ROOT / "checkpoints_ts" / "TimeMixer_latest_results.json"
            if tm_json.exists():
                data = json.loads(tm_json.read_text(encoding="utf-8"))
                timemixer_r2 = data.get("R2")

            summary_rows: List[Dict[str, str]] = []
            if time_row:
                summary_rows.append(
                    {
                        "model": "TimeMixer",
                        "params": time_row.get("total_parameters", ""),
                        "size_mb": time_row.get("model_size_mb", ""),
                        "avg_infer_ms": time_row.get("avg_inference_time_ms", ""),
                        "r2": "" if timemixer_r2 is None else f"{timemixer_r2:.6f}",
                    }
                )
            if lite_row:
                summary_rows.append(
                    {
                        "model": "Ultra-LSNT-Lite",
                        "params": lite_row.get("total_parameters", ""),
                        "size_mb": lite_row.get("model_size_mb", ""),
                        "avg_infer_ms": lite_row.get("avg_inference_time_ms", ""),
                        "r2": "",
                    }
                )
            if lite_opt_row:
                summary_rows.append(
                    {
                        "model": "Ultra-LSNT-Lite-Optimized",
                        "params": lite_opt_row.get("total_parameters", ""),
                        "size_mb": lite_opt_row.get("model_size_mb", ""),
                        "avg_infer_ms": lite_opt_row.get("avg_inference_time_ms", ""),
                        "r2": "",
                    }
                )
            if lite_opt_perf and summary_rows:
                for row in summary_rows:
                    if row["model"] == "Ultra-LSNT-Lite-Optimized":
                        row["r2"] = lite_opt_perf.get("R2", "")
                        break
            if full_r2 is not None or full_rmse is not None:
                summary_rows.append(
                    {
                        "model": "Ultra-LSNT",
                        "params": "",
                        "size_mb": "",
                        "avg_infer_ms": "",
                        "r2": "" if full_r2 is None else f"{full_r2:.6f}",
                    }
                )

            if not summary_rows:
                return TaskResult(
                    name=task_name,
                    ok=False,
                    duration_sec=time.time() - started,
                    log_file=str(log_path.relative_to(ROOT)),
                    error="No Lite comparison artifacts found.",
                )

            out_dir = ROOT / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_out = out_dir / f"ultra_lsnt_lite_suite_comparison_{ts}.csv"
            md_out = out_dir / f"ultra_lsnt_lite_suite_comparison_{ts}.md"

            with csv_out.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["model", "params", "size_mb", "avg_infer_ms", "r2"]
                )
                writer.writeheader()
                for row in summary_rows:
                    writer.writerow(row)

            lines = [
                "# Ultra-LSNT-Lite Comparison Summary",
                "",
                f"Generated at: {datetime.now().isoformat()}",
                "",
                "| Model | Params | Size (MB) | Avg Inference (ms) | R2 |",
                "|---|---:|---:|---:|---:|",
            ]
            for row in summary_rows:
                lines.append(
                    f"| {row['model']} | {row['params']} | {row['size_mb']} | {row['avg_infer_ms']} | {row['r2']} |"
                )

            if full_rmse is not None:
                lines.extend(["", f"Ultra-LSNT full model RMSE (reference): `{full_rmse:.6f}`"])

            md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logf.write(f"Saved: {csv_out}\nSaved: {md_out}\n")

            return TaskResult(
                name=task_name,
                ok=True,
                duration_sec=time.time() - started,
                log_file=str(log_path.relative_to(ROOT)),
                outputs=[str(csv_out.relative_to(ROOT)), str(md_out.relative_to(ROOT))],
            )

    def run_task(self, task: str) -> TaskResult:
        if task == "multi_domain":
            return self.run_script(task, "evaluate_multi_domain.py")
        if task == "multi_domain_baselines":
            log_path = self._task_log_path(task)
            model2 = os.environ.get("MD_BASELINE2", "timemixer").strip().lower()
            epochs = os.environ.get("MD_BASELINE_EPOCHS", "").strip()
            batch_size = os.environ.get("MD_BASELINE_BATCH", "").strip()
            cmd = [self.python_exec, "run_multi_domain_baselines.py", "--model2", model2]
            if epochs:
                cmd.extend(["--epochs", epochs])
            if batch_size:
                cmd.extend(["--batch-size", batch_size])
            return self._run_command(task, cmd, log_path)
        if task == "kernel_ablation":
            return self.run_script(task, "run_universal_ablation.py")
        if task == "noise20":
            return self.run_script(task, "test_dlinear_noise_robustness.py")
        if task == "dm_extended":
            return self.run_dm_extended()
        if task == "paper_digest":
            return self.build_paper_digest()
        if task == "lite_suite":
            result_list = [
                self.run_script("lite_timemixer_eff", "test_timemixer_efficiency.py"),
                self.run_script("lite_base_eff", "test_ultra_lsnt_lite_efficiency.py"),
                self.run_script("lite_opt_eff", "test_ultra_lsnt_lite_optimized.py"),
                self.run_script("lite_opt_perf", "evaluate_ultra_lsnt_optimized.py"),
                self.summarize_lite(),
            ]
            ok = all(x.ok for x in result_list)
            duration = sum(x.duration_sec for x in result_list)
            outputs: List[str] = []
            for x in result_list:
                outputs.extend(x.outputs)
            errors = "; ".join(x.error for x in result_list if x.error)
            return TaskResult(
                name=task,
                ok=ok,
                duration_sec=duration,
                log_file=result_list[-1].log_file if result_list else "",
                error=errors,
                outputs=outputs,
            )
        return TaskResult(
            name=task,
            ok=False,
            duration_sec=0.0,
            log_file="",
            error=f"Unknown task: {task}",
        )

    def run(self, tasks: List[str]) -> int:
        exit_code = 0
        for task in tasks:
            result = self.run_task(task)
            self.summary.append(result)
            status = "OK" if result.ok else "FAIL"
            print(f"[{status}] {task} ({result.duration_sec:.1f}s)")
            if result.log_file:
                print(f"  log: {result.log_file}")
            if result.outputs:
                print(f"  outputs: {', '.join(result.outputs)}")
            if result.error:
                print(f"  error: {result.error}")
            if not result.ok:
                exit_code = 1
                if not self.continue_on_error:
                    break
        self._write_summary_file()
        return exit_code

    def _write_summary_file(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.log_dir / f"suite_summary_{ts}.md"
        lines = [
            "# Background Suite Summary",
            "",
            f"Generated at: {datetime.now().isoformat()}",
            "",
            "| Task | Status | Duration (s) | Log | Error |",
            "|---|---|---:|---|---|",
        ]
        for s in self.summary:
            lines.append(
                f"| {s.name} | {'OK' if s.ok else 'FAIL'} | {s.duration_sec:.1f} | {s.log_file} | {s.error} |"
            )
        lines.append("")
        lines.append("## Output Files")
        for s in self.summary:
            for out in s.outputs:
                lines.append(f"- `{out}`")
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Summary written to: {summary_path.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected experiments in a single background-friendly pipeline.")
    parser.add_argument(
        "--tasks",
        type=str,
        default=DEFAULT_TASKS,
        help=(
            "Comma-separated tasks. Available: "
            "multi_domain,multi_domain_baselines,kernel_ablation,noise20,dm_extended,paper_digest,lite_suite"
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs_background",
        help="Directory for per-task logs and suite summary.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used for subprocess tasks.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining tasks if one task fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and create logs without executing scripts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    runner = SuiteRunner(
        python_exec=args.python,
        log_dir=ROOT / args.log_dir,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
    )
    return runner.run(tasks)


if __name__ == "__main__":
    raise SystemExit(main())
