#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_guardrail_controls(results_root: Path) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(results_root.rglob("fgmoe_clip_only_control*.csv")):
        if csv_path.name.endswith(".meta.json"):
            continue
        meta_path = csv_path.with_suffix(".meta.json")
        meta = _load_json(meta_path)
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["result_kind"] = "guardrail_control"
        df["source_file"] = str(csv_path.as_posix())
        df["dataset"] = meta.get("dataset", "")
        df["dataset_name"] = meta.get("dataset_name", "")
        df["output_tag"] = meta.get("output_tag", "")
        if "backbone" not in df.columns:
            df["backbone"] = meta.get("backbone", "")
        frames.append(df)
    return frames


def collect_guardrail_training(results_root: Path) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(results_root.rglob("pi_moe_full_vs_ultra_clean_sigma06*.csv")):
        meta_name = csv_path.name.replace("pi_moe_full_vs_ultra_clean_sigma06", "pi_moe_full_training_meta").replace(".csv", ".json")
        meta_path = csv_path.with_name(meta_name)
        meta = _load_json(meta_path)
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["result_kind"] = "guardrail_training"
        df["source_file"] = str(csv_path.as_posix())
        df["dataset"] = meta.get("dataset", "")
        df["dataset_name"] = meta.get("dataset_name", "")
        df["output_tag"] = meta.get("output_tag", "")
        if "backbone" not in df.columns:
            df["backbone"] = meta.get("backbone", "")
        frames.append(df)
    return frames


def collect_safety_tables(deployment_root: Path) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(deployment_root.rglob("table_safety_coupling*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["result_kind"] = "safety_deployment"
        df["source_file"] = str(csv_path.as_posix())
        frames.append(df)
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate missing-AE guardrail and deployment outputs.")
    parser.add_argument("--results-root", type=Path, default=Path("results") / "missing_ae_guardrails")
    parser.add_argument("--deployment-root", type=Path, default=Path("results") / "deployment_evidence_missing_ae")
    parser.add_argument("--out-dir", type=Path, default=Path("results") / "missing_ae_guardrails")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    frames.extend(collect_guardrail_controls(args.results_root))
    frames.extend(collect_guardrail_training(args.results_root))
    frames.extend(collect_safety_tables(args.deployment_root))

    if frames:
        summary_df = pd.concat(frames, ignore_index=True, sort=False)
    else:
        summary_df = pd.DataFrame()

    csv_path = args.out_dir / "ae_missing_experiments_summary.csv"
    json_path = args.out_dir / "ae_missing_experiments_summary.json"
    summary_df.to_csv(csv_path, index=False)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "results_root": str(args.results_root),
        "deployment_root": str(args.deployment_root),
        "row_count": int(len(summary_df)),
        "guardrail_runs": int((summary_df.get("result_kind") == "guardrail_control").sum()) if "result_kind" in summary_df else 0,
        "training_rows": int((summary_df.get("result_kind") == "guardrail_training").sum()) if "result_kind" in summary_df else 0,
        "deployment_rows": int((summary_df.get("result_kind") == "safety_deployment").sum()) if "result_kind" in summary_df else 0,
        "source_files": sorted(summary_df["source_file"].dropna().unique().tolist()) if "source_file" in summary_df else [],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
