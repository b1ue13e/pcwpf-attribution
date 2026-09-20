#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_RESULTS_ROOT = Path("results") / "ae_proxy_sensitivity"
DEFAULT_SUMMARY_ROOT = Path("results") / "ae_revision_summary"
DEFAULT_CANONICAL_CONTROL = Path("results") / "supplementary_evidence" / "fgmoe_clip_only_control_windcn.csv"
DEFAULT_CANONICAL_HIGH_RISK = Path("results_export_cfg_lower_corr_20260329_012420") / "stage1_summary_cfg_lower_corr" / "operation_high_risk_summary_seed42.csv"
CASE_ORDER = [
    "baseline_fixed",
    "sv_low",
    "sv_high",
    "delta_low",
    "delta_high",
    "omega_low",
    "omega_high",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate AE proxy sensitivity case outputs into one revision summary.")
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--summary-root", type=str, default=str(DEFAULT_SUMMARY_ROOT))
    parser.add_argument("--canonical-control-csv", type=str, default=str(DEFAULT_CANONICAL_CONTROL))
    parser.add_argument("--canonical-high-risk-csv", type=str, default=str(DEFAULT_CANONICAL_HIGH_RISK))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pick_control_row(df: pd.DataFrame, scenario: str, model: str) -> pd.Series:
    sub = df[(df["scenario"].astype(str) == scenario) & (df["model"].astype(str) == model)]
    if sub.empty:
        raise ValueError(f"Unable to find scenario={scenario}, model={model}")
    return sub.iloc[0]


def pick_dispatch_row(df: pd.DataFrame, model: str) -> pd.Series:
    sub = df[df["model"].astype(str) == model]
    if sub.empty:
        raise ValueError(f"Unable to find dispatch model={model}")
    return sub.iloc[0]


def load_canonical_baseline(control_csv: Path, high_risk_csv: Path) -> Dict[str, float]:
    control = pd.read_csv(control_csv)
    dispatch = pd.read_csv(high_risk_csv)
    fg_clean = pick_control_row(control, "clean", "FG-MoE (full-trained)")
    fg_severe = pick_control_row(control, "gaussian_60pct", "FG-MoE (full-trained)")
    high_risk = pick_dispatch_row(dispatch, "Ultra-LSNT + FG-MoE")
    return {
        "canonical_clean_r2_fg": float(fg_clean["R2"]),
        "canonical_severe_r2_fg": float(fg_severe["R2"]),
        "canonical_clean_over_rated_rate_pct_fg": float(fg_clean["over_rated_rate_pct"]),
        "canonical_severe_over_rated_rate_pct_fg": float(fg_severe["over_rated_rate_pct"]),
        "canonical_high_risk_total_slack_mwh": float(high_risk["high_risk_total_slack_mwh"]),
        "canonical_reserve_call_proxy_mwh": float(high_risk["reserve_call_proxy_mwh"]),
        "canonical_p95_positive_error_mw": float(high_risk["p95_positive_error_mw"]),
        "canonical_high_risk_total_rt_cost": float(high_risk["high_risk_total_rt_cost"]),
    }


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    summary_root = Path(args.summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)

    rows: List[pd.DataFrame] = []
    missing_cases: List[str] = []
    for case_id in CASE_ORDER:
        case_csv = results_root / case_id / f"seed_{args.seed}" / "summary" / f"proxy_case_summary_{case_id}_seed{args.seed}.csv"
        if not case_csv.exists():
            missing_cases.append(case_id)
            continue
        df = pd.read_csv(case_csv)
        df["case_order"] = CASE_ORDER.index(case_id)
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No proxy sensitivity case summaries found under {results_root}")

    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined = combined.sort_values(["case_order", "case_id"]).reset_index(drop=True)
    canonical = load_canonical_baseline(Path(args.canonical_control_csv), Path(args.canonical_high_risk_csv))
    for key, value in canonical.items():
        combined[key] = value

    baseline_mask = combined["case_id"].astype(str) == "baseline_fixed"
    combined["baseline_vs_canonical_clean_r2_fg"] = pd.NA
    combined["baseline_vs_canonical_severe_r2_fg"] = pd.NA
    combined["baseline_vs_canonical_clean_over_rate_pct_fg"] = pd.NA
    combined["baseline_vs_canonical_severe_over_rate_pct_fg"] = pd.NA
    combined["baseline_vs_canonical_high_risk_total_slack_mwh"] = pd.NA
    combined["baseline_vs_canonical_reserve_call_proxy_mwh"] = pd.NA
    combined["baseline_vs_canonical_p95_positive_error_mw"] = pd.NA
    combined["baseline_vs_canonical_high_risk_total_rt_cost"] = pd.NA
    if baseline_mask.any():
        combined.loc[baseline_mask, "baseline_vs_canonical_clean_r2_fg"] = (
            combined.loc[baseline_mask, "clean_r2_fg"] - combined.loc[baseline_mask, "canonical_clean_r2_fg"]
        )
        combined.loc[baseline_mask, "baseline_vs_canonical_severe_r2_fg"] = (
            combined.loc[baseline_mask, "severe_r2_fg"] - combined.loc[baseline_mask, "canonical_severe_r2_fg"]
        )
        combined.loc[baseline_mask, "baseline_vs_canonical_clean_over_rate_pct_fg"] = (
            combined.loc[baseline_mask, "clean_over_rated_rate_pct_fg"] - combined.loc[baseline_mask, "canonical_clean_over_rated_rate_pct_fg"]
        )
        combined.loc[baseline_mask, "baseline_vs_canonical_severe_over_rate_pct_fg"] = (
            combined.loc[baseline_mask, "severe_over_rated_rate_pct_fg"] - combined.loc[baseline_mask, "canonical_severe_over_rated_rate_pct_fg"]
        )
        combined.loc[baseline_mask, "baseline_vs_canonical_high_risk_total_slack_mwh"] = (
            combined.loc[baseline_mask, "high_risk_total_slack_mwh"] - combined.loc[baseline_mask, "canonical_high_risk_total_slack_mwh"]
        )
        combined.loc[baseline_mask, "baseline_vs_canonical_reserve_call_proxy_mwh"] = (
            combined.loc[baseline_mask, "reserve_call_proxy_mwh"] - combined.loc[baseline_mask, "canonical_reserve_call_proxy_mwh"]
        )
        combined.loc[baseline_mask, "baseline_vs_canonical_p95_positive_error_mw"] = (
            combined.loc[baseline_mask, "p95_positive_error_mw"] - combined.loc[baseline_mask, "canonical_p95_positive_error_mw"]
        )
        combined.loc[baseline_mask, "baseline_vs_canonical_high_risk_total_rt_cost"] = (
            combined.loc[baseline_mask, "high_risk_total_rt_cost"] - combined.loc[baseline_mask, "canonical_high_risk_total_rt_cost"]
        )

    out_csv = summary_root / "ae_proxy_sensitivity_summary.csv"
    out_json = summary_root / "ae_proxy_sensitivity_summary.meta.json"
    combined.to_csv(out_csv, index=False)

    meta = {
        "script": "aggregate_ae_proxy_sensitivity_results.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root).replace("\\", "/"),
        "summary_root": str(summary_root).replace("\\", "/"),
        "canonical_control_csv": str(Path(args.canonical_control_csv)).replace("\\", "/"),
        "canonical_high_risk_csv": str(Path(args.canonical_high_risk_csv)).replace("\\", "/"),
        "missing_cases": missing_cases,
        "case_order": CASE_ORDER,
        "rows": int(len(combined)),
    }
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_json}")
    if missing_cases:
        print("[warn] missing cases:", ", ".join(missing_cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
