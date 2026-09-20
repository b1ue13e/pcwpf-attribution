#!/usr/bin/env python3
"""
Build one-row TSFM slot CSV from prediction arrays.

Output columns:
  model, clean_R2, clean_RMSE_kW, clean_MAE_kW,
  R2_sigma_eff_060, R2_spike_06, drop_pct_sigma060
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_array(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
    elif p.suffix.lower() == ".npz":
        z = np.load(p)
        if len(z.files) == 0:
            raise ValueError(f"Empty npz: {p}")
        # Prefer common names first.
        for k in ("pred", "prediction", "predictions", "y_pred", "y_true", "true", "ground_truth"):
            if k in z.files:
                arr = z[k]
                break
        else:
            arr = z[z.files[0]]
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if df.empty:
            raise ValueError(f"Empty csv: {p}")
        # Use first numeric column.
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError(f"No numeric column in csv: {p}")
        arr = df[num_cols[0]].to_numpy()
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true.size != y_pred.size:
        n = min(y_true.size, y_pred.size)
        y_true = y_true[:n]
        y_pred = y_pred[:n]
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build TSFM slot CSV from prediction arrays")
    ap.add_argument("--model", type=str, required=True, help="Model name, e.g., Chronos-T5-Mini")
    ap.add_argument("--true", type=str, required=True, help="Ground-truth array file (.npy/.npz/.csv)")
    ap.add_argument("--pred-clean", type=str, required=True, help="Clean prediction array")
    ap.add_argument("--pred-gaussian", type=str, required=True, help="Gaussian sigma_eff=0.60 prediction array")
    ap.add_argument("--pred-spike", type=str, required=True, help="Spike severity=0.6 prediction array")
    ap.add_argument("--out", type=str, required=True, help="Output csv path")
    args = ap.parse_args()

    y_true = _load_array(args.true)
    p_clean = _load_array(args.pred_clean)
    p_gauss = _load_array(args.pred_gaussian)
    p_spike = _load_array(args.pred_spike)

    m_clean = _metrics(y_true, p_clean)
    m_gauss = _metrics(y_true, p_gauss)
    m_spike = _metrics(y_true, p_spike)

    drop = 100.0 * (m_clean["R2"] - m_gauss["R2"]) / (abs(m_clean["R2"]) + 1e-12)

    row = {
        "model": args.model,
        "clean_R2": m_clean["R2"],
        "clean_RMSE_kW": m_clean["RMSE"],
        "clean_MAE_kW": m_clean["MAE"],
        "R2_sigma_eff_060": m_gauss["R2"],
        "R2_spike_06": m_spike["R2"],
        "drop_pct_sigma060": drop,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
