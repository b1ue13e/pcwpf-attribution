#!/usr/bin/env python3
"""
Build a normalized dispatch-mapping sensitivity evidence pack from archived
RTS-24 DC-OPF aggregate outputs (no new rerun).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


REQUIRED_COLUMNS = [
    "model",
    "total_rt_cost",
    "total_curtailment_mwh",
    "opf_success_ratio",
    "cost_reduction_vs_dlinear_pct",
    "curtailment_reduction_vs_dlinear_pct",
]

MODEL_ORDER = ["DLinear", "iTransformer", "TimeMixer", "Ultra-LSNT"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build dispatch mapping sensitivity CSV/JSON from archived wmax aggregate files."
    )
    ap.add_argument(
        "--in260",
        type=str,
        default="results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax260_aggregate.csv",
    )
    ap.add_argument(
        "--in280",
        type=str,
        default="results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax280_aggregate.csv",
    )
    ap.add_argument(
        "--in300",
        type=str,
        default="",
        help="Optional archived aggregate CSV for wmax300.",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="results/supplementary_evidence/dispatch_mapping_sensitivity_wmax260_280.csv",
    )
    ap.add_argument(
        "--out_meta",
        type=str,
        default="results/supplementary_evidence/dispatch_mapping_sensitivity_wmax260_280.meta.json",
    )
    return ap.parse_args()


def _load_and_validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def _build_rows(df: pd.DataFrame, mapping_case: str, wind_max_mw: int, source_file: str, run_id: str) -> List[Dict]:
    rows: List[Dict] = []
    for model in MODEL_ORDER:
        hit = df[df["model"] == model]
        if hit.empty:
            raise ValueError(f"Model '{model}' missing in {source_file}")
        r = hit.iloc[0]
        rows.append(
            {
                "mapping_case": mapping_case,
                "wind_max_mw": int(wind_max_mw),
                "model": model,
                "total_rt_cost": float(r["total_rt_cost"]),
                "total_curtailment_mwh": float(r["total_curtailment_mwh"]),
                "opf_success_ratio": float(r["opf_success_ratio"]),
                "cost_reduction_vs_dlinear_pct": float(r["cost_reduction_vs_dlinear_pct"]),
                "curtailment_reduction_vs_dlinear_pct": float(r["curtailment_reduction_vs_dlinear_pct"]),
                "source_file": source_file,
                "run_id": run_id,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    in260 = Path(args.in260)
    in280 = Path(args.in280)
    in300 = Path(args.in300) if args.in300 else None
    out_csv = Path(args.out_csv)
    out_meta = Path(args.out_meta)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    df260 = _load_and_validate(in260)
    df280 = _load_and_validate(in280)
    df300 = _load_and_validate(in300) if in300 is not None else None

    ts_utc = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_dispatch_mapping_sensitivity_archival"

    rows = []
    rows.extend(_build_rows(df260, "wmax260", 260, str(in260).replace("\\", "/"), run_id))
    rows.extend(_build_rows(df280, "wmax280", 280, str(in280).replace("\\", "/"), run_id))
    if df300 is not None:
        rows.extend(_build_rows(df300, "wmax300", 300, str(in300).replace("\\", "/"), run_id))

    out_df = pd.DataFrame(
        rows,
        columns=[
            "mapping_case",
            "wind_max_mw",
            "model",
            "total_rt_cost",
            "total_curtailment_mwh",
            "opf_success_ratio",
            "cost_reduction_vs_dlinear_pct",
            "curtailment_reduction_vs_dlinear_pct",
            "source_file",
            "run_id",
        ],
    )
    out_df.to_csv(out_csv, index=False)

    source_files = [str(in260).replace("\\", "/"), str(in280).replace("\\", "/")]
    cases = [
        {"mapping_case": "wmax260", "wind_max_mw": 260},
        {"mapping_case": "wmax280", "wind_max_mw": 280},
    ]
    if in300 is not None:
        source_files.append(str(in300).replace("\\", "/"))
        cases.append({"mapping_case": "wmax300", "wind_max_mw": 300})

    meta = {
        "timestamp_utc": ts_utc,
        "source_files": source_files,
        "mapping_definition": {
            "cases": cases,
            "protocol_note": "Derived from archived RTS-24 DC-OPF aggregate outputs under fixed affine+clipping mapped dispatch protocol.",
        },
        "models": MODEL_ORDER,
        "notes": [
            "archival sensitivity evidence; no new rerun in this revision",
            "Rows preserve archived aggregate values and model order for reproducible supplementary reporting.",
        ],
        "outputs": {"csv": str(out_csv).replace("\\", "/"), "meta_json": str(out_meta).replace("\\", "/")},
        "run_id": run_id,
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_meta}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
