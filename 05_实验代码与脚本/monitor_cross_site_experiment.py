#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUT = Path(__file__).resolve().parent.parent / "06_实验结果与日志" / "results" / "cross_site_wind_full_cuda"
TOTAL_EXPECTED = 3 * 5 * 5 * 3 * 6


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize cross-site wind experiment progress.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--expected", type=int, default=TOTAL_EXPECTED)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    curve_path = out_dir / "cross_site_recalibration_curve.csv"
    if not curve_path.exists():
        print(f"missing: {curve_path}")
        return 1

    curve = pd.read_csv(curve_path)
    rows = len(curve)
    errors = int(curve.get("error", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).sum())
    expected = int(args.expected)
    print(f"rows={rows}/{expected} ({rows / expected:.1%}) errors={errors}")
    if rows:
        last = curve.iloc[-1]
        print(
            "last="
            f"model={last.get('model')} "
            f"seed={last.get('seed')} "
            f"source={last.get('source_dataset')} "
            f"target={last.get('target_dataset')} "
            f"ratio={last.get('calibration_ratio')} "
            f"R2={last.get('R2')} "
            f"error={last.get('error')}"
        )

    group_cols = ["model", "seed", "source_dataset"]
    present_cols = [col for col in group_cols if col in curve.columns]
    if present_cols:
        grouped = curve.groupby(present_cols, dropna=False).size().reset_index(name="rows")
        print("\nby model/seed/source:")
        print(grouped.to_string(index=False))

    zero_path = out_dir / "cross_site_zero_shot_matrix.csv"
    summary_path = out_dir / "cross_site_summary_for_paper.csv"
    print(f"\nzero_matrix={zero_path.exists()} summary={summary_path.exists()} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
