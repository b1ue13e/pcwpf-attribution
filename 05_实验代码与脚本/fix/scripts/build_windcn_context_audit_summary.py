#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional
import json

import numpy as np
import pandas as pd

from windcn_audit_common import ROOT, rel_rmse_increase, utc_run_id


DEFAULT_EVIDENCE_DIR = ROOT / "results" / "supplementary_evidence"
DEFAULT_OUT_CSV = DEFAULT_EVIDENCE_DIR / "windcn_context_audit_summary.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def pick_row(df: pd.DataFrame, **conds):
    if df.empty:
        return None
    mask = np.ones(len(df), dtype=bool)
    for key, value in conds.items():
        if key not in df.columns:
            return None
        mask &= df[key] == value
    sub = df[mask]
    return None if sub.empty else sub.iloc[0]


def find_sigma_row(df: pd.DataFrame, sigma_eff: float, model_col: str, model_name: str):
    if df.empty or model_col not in df.columns:
        return None
    sub = df[df[model_col].astype(str).str.strip() == model_name].copy()
    if sub.empty:
        return None
    if "sigma_eff" in sub.columns:
        sub = sub[np.isclose(sub["sigma_eff"].astype(float), sigma_eff)]
    elif "sigma" in sub.columns:
        sub = sub[np.isclose(sub["sigma"].astype(float), sigma_eff / 1.5)]
    else:
        return None
    return None if sub.empty else sub.iloc[0]


def add_long_gaussian_row(
    row: Dict[str, object],
    df: pd.DataFrame,
    model_name: str,
    model_col: str = "model",
) -> Dict[str, object]:
    clean = None
    if not df.empty and model_col in df.columns:
        clean_sub = df[df[model_col].astype(str).str.strip() == model_name]
        if not clean_sub.empty:
            if "scenario" in clean_sub.columns:
                clean_sub = clean_sub[clean_sub["scenario"].astype(str) == "clean"]
            elif "sigma_eff" in clean_sub.columns:
                clean_sub = clean_sub[np.isclose(clean_sub["sigma_eff"].astype(float), 0.0)]
            elif "sigma" in clean_sub.columns:
                clean_sub = clean_sub[np.isclose(clean_sub["sigma"].astype(float), 0.0)]
            clean = None if clean_sub.empty else clean_sub.iloc[0]

    row_015 = find_sigma_row(df, 0.15, model_col=model_col, model_name=model_name)
    row_060 = find_sigma_row(df, 0.60, model_col=model_col, model_name=model_name)

    if clean is not None:
        row["clean_R2"] = float(clean["R2"])
        row["clean_RMSE_kW"] = float(clean["RMSE"])
        row["clean_MAE_kW"] = float(clean["MAE"])
        row["source_clean"] = str(clean.get("run_id", "")) or str(clean.get("source_file", ""))
    if row_015 is not None:
        row["R2_sigma_eff_015"] = float(row_015["R2"])
        row["RMSE_sigma_eff_015_kW"] = float(row_015["RMSE"])
        row["MAE_sigma_eff_015_kW"] = float(row_015["MAE"])
        row["source_015"] = str(row_015.get("run_id", "")) or str(row_015.get("source_file", ""))
    if row_060 is not None:
        row["R2_sigma_eff_060"] = float(row_060["R2"])
        row["RMSE_sigma_eff_060_kW"] = float(row_060["RMSE"])
        row["MAE_sigma_eff_060_kW"] = float(row_060["MAE"])
        row["source_060"] = str(row_060.get("run_id", "")) or str(row_060.get("source_file", ""))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a wide context-audit summary for Wind (CN) main-table use.")
    ap.add_argument("--evidence-dir", type=str, default=str(DEFAULT_EVIDENCE_DIR))
    ap.add_argument("--out-csv", type=str, default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--allow-legacy-tsfm-slots", action="store_true")
    args = ap.parse_args()

    evidence_dir = Path(args.evidence_dir)
    run_id = utc_run_id("_windcn_context_audit_summary")

    core2 = read_csv(evidence_dir / "core2_clean.csv")
    deep = read_csv(evidence_dir / "deep_sota_gaussian_windcn.csv")
    persistence = read_csv(evidence_dir / "persistence_gaussian_windcn.csv")
    robust_r2 = read_csv(ROOT / "results" / "robust_gaussian_windcn.csv")
    lightgbm_audit = read_csv(evidence_dir / "lightgbm_gaussian_audit_windcn.csv")
    chronos_audit = read_csv(evidence_dir / "chronos_family_windcn.csv")
    moirai_audit = read_csv(evidence_dir / "moirai_family_windcn.csv")
    legacy_tsfm = read_csv(evidence_dir / "tsfm_slots_windcn.csv") if args.allow_legacy_tsfm_slots else pd.DataFrame()

    # Load runtime/memory stats from JSONs (if available in current dir)
    stats_map = {}
    for key, fname in [("LightGBM", "lightgbm_stats.json"), 
                       ("Chronos-Bolt-Small", "chronos_stats.json"), 
                       ("Moirai-2.0-R-Small", "moirai_stats.json")]:
        try:
            if Path(fname).exists():
                with open(fname, "r") as f:
                    s = json.load(f)
                    stats_map[key] = {
                        "runtime_seconds": s.get("runtime_seconds", np.nan),
                        "peak_gpu_memory_mb": s.get("peak_gpu_memory_mb", np.nan)
                    }
        except Exception:
            pass

    models = [
        ("General DL", "DLinear"),
        ("General DL", "TimeMixer"),
        ("General DL", "iTransformer"),
        ("Edge-Native", "Ultra-LSNT"),
        ("Context Non-Deep", "LightGBM"),
        ("Context Non-Deep", "Persistence"),
        ("Cloud Upper Bound", "Chronos-Bolt-Small"),
        ("Cloud Upper Bound", "Moirai-2.0-R-Small"),
    ]

    rows = []
    for category, model in models:
        row: Dict[str, object] = {
            "category": category,
            "model": model,
            "clean_R2": np.nan,
            "clean_RMSE_kW": np.nan,
            "clean_MAE_kW": np.nan,
            "R2_sigma_eff_015": np.nan,
            "RMSE_sigma_eff_015_kW": np.nan,
            "MAE_sigma_eff_015_kW": np.nan,
            "R2_sigma_eff_060": np.nan,
            "RMSE_sigma_eff_060_kW": np.nan,
            "MAE_sigma_eff_060_kW": np.nan,
            "rel_rmse_increase_pct_060": np.nan,
            "runtime_seconds": np.nan,
            "peak_gpu_memory_mb": np.nan,
            "source_clean": "",
            "source_015": "",
            "source_060": "",
            "run_id": run_id,
        }

        if model in stats_map:
            row["runtime_seconds"] = stats_map[model]["runtime_seconds"]
            row["peak_gpu_memory_mb"] = stats_map[model]["peak_gpu_memory_mb"]

        clean_row = pick_row(core2, model=model)
        if clean_row is None:
            clean_row = pick_row(core2, model=f"{model} (Reliability, tau=0.5)")
        if clean_row is not None:
            row["clean_R2"] = float(clean_row["R2"])
            row["clean_RMSE_kW"] = float(clean_row["RMSE_kW"])
            row["clean_MAE_kW"] = float(clean_row["MAE_kW"])
            row["source_clean"] = str(clean_row.get("source_file", ""))

        if model in {"TimeMixer", "iTransformer"}:
            clean = pick_row(deep, model=model, sigma=0.0)
            sigma015 = pick_row(deep, model=model, sigma=0.1)
            sigma060 = pick_row(deep, model=model, sigma=0.4)
            if clean is not None:
                row["clean_R2"] = float(clean["R2"])
                row["clean_RMSE_kW"] = float(clean["RMSE"])
                row["clean_MAE_kW"] = float(clean["MAE"])
                row["source_clean"] = "deep_sota_gaussian_windcn.csv"
            if sigma015 is not None:
                row["R2_sigma_eff_015"] = float(sigma015["R2"])
                row["RMSE_sigma_eff_015_kW"] = float(sigma015["RMSE"])
                row["MAE_sigma_eff_015_kW"] = float(sigma015["MAE"])
                row["source_015"] = "deep_sota_gaussian_windcn.csv"
            if sigma060 is not None:
                row["R2_sigma_eff_060"] = float(sigma060["R2"])
                row["RMSE_sigma_eff_060_kW"] = float(sigma060["RMSE"])
                row["MAE_sigma_eff_060_kW"] = float(sigma060["MAE"])
                row["source_060"] = "deep_sota_gaussian_windcn.csv"

        if model == "Persistence":
            row = add_long_gaussian_row(row, persistence, model_name="Persistence")
        elif model == "LightGBM":
            if not lightgbm_audit.empty:
                row = add_long_gaussian_row(row, lightgbm_audit, model_name="LightGBM")
            else:
                clean = pick_row(robust_r2, model="LightGBM", sigma=0.0)
                sigma015 = pick_row(robust_r2, model="LightGBM", sigma=0.1)
                sigma060 = pick_row(robust_r2, model="LightGBM", sigma=0.4)
                if clean is not None:
                    row["clean_R2"] = float(clean["R2"])
                if sigma015 is not None:
                    row["R2_sigma_eff_015"] = float(sigma015["R2"])
                if sigma060 is not None:
                    row["R2_sigma_eff_060"] = float(sigma060["R2"])
        elif model == "Chronos-Bolt-Small":
            if not chronos_audit.empty:
                row = add_long_gaussian_row(row, chronos_audit, model_name=model, model_col="model_label")
            elif not legacy_tsfm.empty:
                legacy = pick_row(legacy_tsfm, model="Chronos-T5-Mini")
                if legacy is not None:
                    row["clean_R2"] = float(legacy["clean_R2"])
                    row["clean_RMSE_kW"] = float(legacy["clean_RMSE_kW"])
                    row["clean_MAE_kW"] = float(legacy["clean_MAE_kW"])
                    row["R2_sigma_eff_060"] = float(legacy["R2_sigma_eff_060"])
                    row["source_clean"] = "tsfm_slots_windcn.csv (legacy)"
                    row["source_060"] = "tsfm_slots_windcn.csv (legacy)"
        elif model == "Moirai-2.0-R-Small":
            if not moirai_audit.empty:
                row = add_long_gaussian_row(row, moirai_audit, model_name=model, model_col="model_label")
            elif not legacy_tsfm.empty:
                legacy = pick_row(legacy_tsfm, model="Moirai-Small")
                if legacy is not None:
                    row["clean_R2"] = float(legacy["clean_R2"])
                    row["clean_RMSE_kW"] = float(legacy["clean_RMSE_kW"])
                    row["clean_MAE_kW"] = float(legacy["clean_MAE_kW"])
                    row["R2_sigma_eff_060"] = float(legacy["R2_sigma_eff_060"])
                    row["source_clean"] = "tsfm_slots_windcn.csv (legacy)"
                    row["source_060"] = "tsfm_slots_windcn.csv (legacy)"
        else:
            # Current local bundle only stores R2 for DLinear / Ultra-LSNT at gaussian slices.
            clean = pick_row(robust_r2, model=model, sigma=0.0)
            sigma015 = pick_row(robust_r2, model=model, sigma=0.1)
            sigma060 = pick_row(robust_r2, model=model, sigma=0.4)
            if clean is not None:
                row["clean_R2"] = float(clean["R2"])
            if sigma015 is not None:
                row["R2_sigma_eff_015"] = float(sigma015["R2"])
            if sigma060 is not None:
                row["R2_sigma_eff_060"] = float(sigma060["R2"])

        row["rel_rmse_increase_pct_060"] = rel_rmse_increase(
            float(row["clean_RMSE_kW"]) if pd.notna(row["clean_RMSE_kW"]) else np.nan,
            float(row["RMSE_sigma_eff_060_kW"]) if pd.notna(row["RMSE_sigma_eff_060_kW"]) else np.nan,
        )
        rows.append(row)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    
    # Also write JSON as requested
    out_json = out_path.with_suffix(".json")
    out_df.to_json(out_json, orient="records", indent=2)
    
    print(out_df.to_string(index=False))
    print(f"[ok] wrote {out_path} and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
