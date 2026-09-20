#!/usr/bin/env python3
"""
Build auditable core-task view (wind_cn + electricity) from background logs.

Output:
  - experiment_logs/core_wind_electricity_audit_backfill.csv
  - experiment_logs/core_wind_electricity_audit_backfill.md

Design:
  1) Inspect log files for COA-BiLSTM / BWO-SVR / BWO-CNN.
  2) Classify task status from raw log text.
  3) Parse metrics from logs when available.
  4) Backfill missing metrics from metaheuristic_metrics_all_datasets.csv.
"""

from __future__ import annotations

import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


CORE_DATASETS = ["wind_cn", "electricity"]
CORE_MODELS = ["COA-BiLSTM", "BWO-SVR", "BWO-CNN"]
SEEDS = [42, 123, 456]

MODEL_LOG_PATTERNS = {
    "COA-BiLSTM": [
        "experiment_logs/coa_{dataset}_{seed}.log",
        "experiment_logs/coa_{dataset}_{seed}.log.bak.*",
        "optimized_logs/coa_bilstm_{dataset}_{seed}_*.log",
    ],
    "BWO-SVR": [
        "experiment_logs/bwo_svr_{dataset}_{seed}.log",
        "experiment_logs/bwo_svr_{dataset}_{seed}.log.bak.*",
    ],
    "BWO-CNN": [
        "experiment_logs/bwo_cnn_{dataset}_{seed}.log",
        "experiment_logs/bwo_cnn_{dataset}_{seed}.log.bak.*",
    ],
}

ERROR_KEYS = [
    "traceback",
    "nameerror",
    "importerror",
    "error in objective function",
    "could not convert string to float",
    "exit code",
    "退出码",
]

TERMINATED_KEYS = ["terminated", "killed", "退出码: -9", "退出码: 143"]


@dataclass
class Metrics:
    r2: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None
    convergence_iter: Optional[int] = None
    source: str = "none"


def _read_text(path: str) -> str:
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            continue
    return ""


def _find_logs(dataset: str, model: str, seed: int) -> List[str]:
    paths: List[str] = []
    patterns = MODEL_LOG_PATTERNS[model]
    for pattern in patterns:
        resolved = pattern.format(dataset=dataset, seed=seed)
        paths.extend(glob.glob(resolved))
    paths = sorted(set(paths), key=lambda p: os.path.getmtime(p), reverse=True)
    return paths


def _parse_metrics_from_text(text: str) -> Metrics:
    # Accept formats like "R2: 0.78", "R2=0.78", "R²: 0.78"
    num = r"([-+]?\d+(?:\.\d+)?)"
    r2_match = re.search(rf"R2|R²", text, flags=re.IGNORECASE)
    r2_val = None
    rmse_val = None
    mae_val = None
    mape_val = None
    conv_val = None

    if r2_match:
        m = re.search(rf"(?:R2|R²)\s*[:=]\s*{num}", text, flags=re.IGNORECASE)
        if m:
            r2_val = float(m.group(1))

    m = re.search(rf"RMSE\s*[:=]\s*{num}", text, flags=re.IGNORECASE)
    if m:
        rmse_val = float(m.group(1))

    m = re.search(rf"MAE\s*[:=]\s*{num}", text, flags=re.IGNORECASE)
    if m:
        mae_val = float(m.group(1))

    m = re.search(rf"MAPE\s*[:=]\s*{num}", text, flags=re.IGNORECASE)
    if m:
        mape_val = float(m.group(1))

    m = re.search(r"(convergence[_\s-]*iter|收敛迭代)\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        conv_val = int(m.group(2))

    return Metrics(
        r2=r2_val,
        rmse=rmse_val,
        mae=mae_val,
        mape=mape_val,
        convergence_iter=conv_val,
        source="log_parsed" if any(v is not None for v in [r2_val, rmse_val, mae_val, mape_val, conv_val]) else "none",
    )


def _classify_status(text: str, has_metric: bool) -> str:
    low = text.lower()
    if has_metric:
        return "log_has_metrics"
    if any(k in low for k in TERMINATED_KEYS):
        return "terminated"
    if any(k in low for k in ERROR_KEYS):
        return "failed"
    if "dataset:" in low or "starting" in low or "experiment" in low:
        return "started_no_metrics"
    return "unknown"


def _load_csv_metrics(path: str) -> Dict[Tuple[str, str, int], Metrics]:
    out: Dict[Tuple[str, str, int], Metrics] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (row["dataset"], row["model"], int(row["seed"]))
            except Exception:
                continue
            out[key] = Metrics(
                r2=_to_float(row.get("R2")),
                rmse=_to_float(row.get("RMSE")),
                mae=_to_float(row.get("MAE")),
                mape=_to_float(row.get("MAPE")),
                convergence_iter=_to_int(row.get("convergence_iter")),
                source="metaheuristic_metrics_all_datasets.csv",
            )
    return out


def _to_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _to_int(v: Optional[str]) -> Optional[int]:
    fv = _to_float(v)
    if fv is None:
        return None
    return int(round(fv))


def main() -> None:
    csv_metrics = _load_csv_metrics("metaheuristic_metrics_all_datasets.csv")
    rows: List[Dict[str, object]] = []

    for dataset in CORE_DATASETS:
        for model in CORE_MODELS:
            for seed in SEEDS:
                logs = _find_logs(dataset, model, seed)
                chosen_log = logs[0] if logs else ""

                parsed = Metrics()
                status = "no_log"
                note = "no log file matched"

                if chosen_log:
                    text = _read_text(chosen_log)
                    parsed = _parse_metrics_from_text(text)
                    status = _classify_status(text, parsed.source == "log_parsed")
                    note = "latest matched log"

                key = (dataset, model, seed)
                fallback = csv_metrics.get(key)

                final = parsed
                if parsed.source != "log_parsed" and fallback:
                    final = fallback
                    final.source = "csv_backfill_from_metaheuristic_metrics"

                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "task_status": status,
                        "evidence_log": chosen_log.replace("\\", "/") if chosen_log else "",
                        "evidence_note": note,
                        "R2": final.r2,
                        "RMSE": final.rmse,
                        "MAE": final.mae,
                        "MAPE": final.mape,
                        "convergence_iter": final.convergence_iter,
                        "metric_source": final.source,
                    }
                )

    os.makedirs("experiment_logs", exist_ok=True)
    out_csv = "experiment_logs/core_wind_electricity_audit_backfill.csv"
    out_md = "experiment_logs/core_wind_electricity_audit_backfill.md"

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "seed",
                "task_status",
                "evidence_log",
                "evidence_note",
                "R2",
                "RMSE",
                "MAE",
                "MAPE",
                "convergence_iter",
                "metric_source",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    status_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    for row in rows:
        status_counts[row["task_status"]] = status_counts.get(row["task_status"], 0) + 1
        source_counts[row["metric_source"]] = source_counts.get(row["metric_source"], 0) + 1

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Core Audit Backfill (wind_cn + electricity)\n\n")
        f.write(f"- Total tasks: {len(rows)}\n")
        f.write("- Datasets: wind_cn, electricity\n")
        f.write("- Models: COA-BiLSTM, BWO-SVR, BWO-CNN\n")
        f.write("- Seeds: 42, 123, 456\n\n")
        f.write("## Task Status Counts\n")
        for k in sorted(status_counts):
            f.write(f"- {k}: {status_counts[k]}\n")
        f.write("\n## Metric Source Counts\n")
        for k in sorted(source_counts):
            f.write(f"- {k}: {source_counts[k]}\n")
        f.write(f"\nDetailed rows: `{out_csv}`\n")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
