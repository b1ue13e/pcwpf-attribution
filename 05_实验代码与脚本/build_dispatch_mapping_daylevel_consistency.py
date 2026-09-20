#!/usr/bin/env python3
"""
Build day-level dispatch mapping consistency evidence from archived RTS-24
DC-OPF daily outputs (no new rerun).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


MODEL_BASE = "DLinear"
MODEL_COMPARE = ["iTransformer", "TimeMixer", "Ultra-LSNT"]
MODEL_ALL = [MODEL_BASE] + MODEL_COMPARE
METRICS = ["rt_cost_total", "curtailment_mwh_total"]

REQUIRED_COLUMNS = ["model", "day_idx"] + METRICS


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build dispatch mapping day-level consistency CSV/JSON from archived daily files."
    )
    ap.add_argument(
        "--in260",
        type=str,
        default="results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax260_daily.csv",
    )
    ap.add_argument(
        "--in280",
        type=str,
        default="results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax280_daily.csv",
    )
    ap.add_argument(
        "--in300",
        type=str,
        default="results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax300_daily.csv",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="results/supplementary_evidence/dispatch_mapping_daylevel_consistency_wmax260_280_300.csv",
    )
    ap.add_argument(
        "--out_meta",
        type=str,
        default="results/supplementary_evidence/dispatch_mapping_daylevel_consistency_wmax260_280_300.meta.json",
    )
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _norm(path: Path) -> str:
    return str(path).replace("\\", "/")


def _load_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    missing_models = [m for m in MODEL_ALL if m not in set(df["model"].astype(str))]
    if missing_models:
        raise ValueError(f"Missing models in {path}: {missing_models}")
    return df.copy()


def _pivot_case(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    piv = df.pivot_table(index="day_idx", columns="model", values=metric, aggfunc="mean")
    piv = piv.dropna(subset=MODEL_ALL)
    piv = piv.sort_index()
    if piv.empty:
        raise ValueError(f"No shared day_idx intersection for metric {metric}")
    return piv


def _bootstrap_daywin(delta: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> Dict[str, float]:
    n = int(delta.shape[0])
    if n <= 0:
        raise ValueError("Empty delta array for bootstrap.")
    wins = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        wins.append(float((delta[idx] < 0.0).mean()))
    lo, hi = np.quantile(np.asarray(wins), [0.025, 0.975])
    return {"low": float(lo), "high": float(hi)}


def _build_case_rows(
    df: pd.DataFrame,
    mapping_case: str,
    wind_max_mw: int,
    source_daily_file: str,
    n_bootstrap: int,
    rng: np.random.Generator,
    run_id: str,
) -> List[Dict]:
    rows: List[Dict] = []
    for metric in METRICS:
        piv = _pivot_case(df, metric)
        base = piv[MODEL_BASE].to_numpy(dtype=float)
        n_days = int(piv.shape[0])
        for model in MODEL_COMPARE:
            comp = piv[model].to_numpy(dtype=float)
            delta = comp - base
            ci = _bootstrap_daywin(delta, n_bootstrap=n_bootstrap, rng=rng)
            rows.append(
                {
                    "mapping_case": mapping_case,
                    "wind_max_mw": int(wind_max_mw),
                    "model": model,
                    "metric": metric,
                    "n_days": n_days,
                    "day_win_ratio_vs_dlinear": float((delta < 0.0).mean()),
                    "day_win_ci95_low": ci["low"],
                    "day_win_ci95_high": ci["high"],
                    "delta_mean": float(np.mean(delta)),
                    "delta_median": float(np.median(delta)),
                    "delta_p25": float(np.quantile(delta, 0.25)),
                    "delta_p75": float(np.quantile(delta, 0.75)),
                    "source_daily_file": source_daily_file,
                    "run_id": run_id,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    p260 = Path(args.in260)
    p280 = Path(args.in280)
    p300 = Path(args.in300)
    out_csv = Path(args.out_csv)
    out_meta = Path(args.out_meta)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    df260 = _load_daily(p260)
    df280 = _load_daily(p280)
    df300 = _load_daily(p300)

    rng = np.random.default_rng(args.seed)
    ts_utc = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_dispatch_mapping_daylevel_consistency_archival"

    rows: List[Dict] = []
    rows.extend(
        _build_case_rows(
            df260,
            mapping_case="wmax260",
            wind_max_mw=260,
            source_daily_file=_norm(p260),
            n_bootstrap=int(args.bootstrap),
            rng=rng,
            run_id=run_id,
        )
    )
    rows.extend(
        _build_case_rows(
            df280,
            mapping_case="wmax280",
            wind_max_mw=280,
            source_daily_file=_norm(p280),
            n_bootstrap=int(args.bootstrap),
            rng=rng,
            run_id=run_id,
        )
    )
    rows.extend(
        _build_case_rows(
            df300,
            mapping_case="wmax300",
            wind_max_mw=300,
            source_daily_file=_norm(p300),
            n_bootstrap=int(args.bootstrap),
            rng=rng,
            run_id=run_id,
        )
    )

    out_df = pd.DataFrame(
        rows,
        columns=[
            "mapping_case",
            "wind_max_mw",
            "model",
            "metric",
            "n_days",
            "day_win_ratio_vs_dlinear",
            "day_win_ci95_low",
            "day_win_ci95_high",
            "delta_mean",
            "delta_median",
            "delta_p25",
            "delta_p75",
            "source_daily_file",
            "run_id",
        ],
    )
    out_df.to_csv(out_csv, index=False)

    meta = {
        "timestamp_utc": ts_utc,
        "source_files": [_norm(p260), _norm(p280), _norm(p300)],
        "base_model": MODEL_BASE,
        "models": MODEL_COMPARE,
        "metrics": METRICS,
        "bootstrap_n": int(args.bootstrap),
        "seed": int(args.seed),
        "notes": [
            "archival daily consistency evidence; no new rerun in this revision",
            "delta_day = metric_model - metric_dlinear (negative indicates model better than DLinear).",
            "day_win_ratio_vs_dlinear is computed on shared day_idx intersection within each mapping case.",
        ],
        "outputs": {"csv": _norm(out_csv), "meta_json": _norm(out_meta)},
        "run_id": run_id,
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_meta}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
