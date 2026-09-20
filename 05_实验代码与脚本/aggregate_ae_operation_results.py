#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_dispatch_rows(dispatch_root: Path, focus_mapping: str, focus_subset: str) -> List[Dict]:
    rows: List[Dict] = []
    for agg_path in sorted(dispatch_root.rglob("*_aggregate.csv")):
        parent = agg_path.parent
        if parent.name != str(focus_mapping):
            continue
        meta_candidates = list(parent.glob("*.meta.json"))
        meta = _load_json(meta_candidates[0]) if meta_candidates else {}

        try:
            seed = int(parent.parent.name.replace("seed_", ""))
        except Exception:
            seed = -1

        df = pd.read_csv(agg_path)
        sub = df[
            (df["mapping_method"].astype(str) == str(focus_mapping))
            & (df["subset"].astype(str) == str(focus_subset))
        ].copy()
        if sub.empty:
            continue
        reserve_map = meta.get("mapping_params", {}).get(str(focus_mapping), {})
        for _, row in sub.iterrows():
            model = str(row["model"])
            params = reserve_map.get(model, {}) if isinstance(reserve_map, dict) else {}
            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "mapping_method": str(row["mapping_method"]),
                    "subset": str(row["subset"]),
                    "success_rate": float(row["success_rate"]),
                    "feasibility_fail_rate": float(row["feasibility_fail_rate"]),
                    "backfire_rate_vs_dlinear": float(row["backfire_rate_vs_dlinear"]),
                    "total_slack_mwh": float(row["total_slack_mwh"]),
                    "total_positive_error_mwh": float(row.get("total_positive_error_mwh", np.nan)),
                    "p95_positive_error_mw": float(row.get("p95_positive_error_mw", np.nan)),
                    "total_curtailment_mwh": float(row["total_curtailment_mwh"]),
                    "total_rt_cost": float(row["total_rt_cost"]),
                    "forecast_r2": float(row["forecast_r2"]),
                    "mean_forecast_error_mw": float(row["mean_forecast_error_mw"]),
                    "reserve_margin_mw": float(params.get("reserve_margin_mw", np.nan)),
                    "source_file": str(agg_path).replace("\\", "/"),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate AE operation-aware dispatch outputs across seeds.")
    parser.add_argument("--dispatch-root", type=Path, default=Path("results") / "ae_operation_dispatch")
    parser.add_argument("--focus-mapping", type=str, default="risk_conditional")
    parser.add_argument("--focus-subset", type=str, default="high_risk_shared")
    parser.add_argument("--out-dir", type=Path, default=Path("results") / "ae_operation_summary")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = _collect_dispatch_rows(args.dispatch_root, args.focus_mapping, args.focus_subset)
    raw_df = pd.DataFrame(raw_rows)

    raw_csv = args.out_dir / "operation_seed_stability_raw.csv"
    mean_csv = args.out_dir / "operation_seed_stability_mean_std.csv"
    meta_json = args.out_dir / "operation_seed_stability.meta.json"

    if raw_df.empty:
        raw_df.to_csv(raw_csv, index=False)
        mean_df = pd.DataFrame()
        mean_df.to_csv(mean_csv, index=False)
    else:
        raw_df = raw_df.sort_values(["model", "seed"]).reset_index(drop=True)
        raw_df.to_csv(raw_csv, index=False)
        mean_df = (
            raw_df.groupby("model", as_index=False)
            .agg(
                success_rate_mean=("success_rate", "mean"),
                success_rate_std=("success_rate", "std"),
                feasibility_fail_rate_mean=("feasibility_fail_rate", "mean"),
                feasibility_fail_rate_std=("feasibility_fail_rate", "std"),
                backfire_rate_mean=("backfire_rate_vs_dlinear", "mean"),
                backfire_rate_std=("backfire_rate_vs_dlinear", "std"),
                reserve_margin_mean_mw=("reserve_margin_mw", "mean"),
                reserve_margin_std_mw=("reserve_margin_mw", "std"),
                total_slack_mean_mwh=("total_slack_mwh", "mean"),
                total_slack_std_mwh=("total_slack_mwh", "std"),
                total_positive_error_mean_mwh=("total_positive_error_mwh", "mean"),
                total_positive_error_std_mwh=("total_positive_error_mwh", "std"),
                p95_positive_error_mean_mw=("p95_positive_error_mw", "mean"),
                p95_positive_error_std_mw=("p95_positive_error_mw", "std"),
                total_cost_mean=("total_rt_cost", "mean"),
                total_cost_std=("total_rt_cost", "std"),
            )
        )
        mean_df.to_csv(mean_csv, index=False)

    meta = {
        "script": "aggregate_ae_operation_results.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dispatch_root": str(args.dispatch_root).replace("\\", "/"),
        "focus_mapping": args.focus_mapping,
        "focus_subset": args.focus_subset,
        "row_count": int(len(raw_df)),
        "outputs": {
            "raw_csv": str(raw_csv).replace("\\", "/"),
            "mean_std_csv": str(mean_csv).replace("\\", "/"),
            "meta_json": str(meta_json).replace("\\", "/"),
        },
        "source_files": sorted(raw_df["source_file"].dropna().unique().tolist()) if "source_file" in raw_df else [],
    }
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] wrote {raw_csv}")
    print(f"[ok] wrote {mean_csv}")
    print(f"[ok] wrote {meta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
