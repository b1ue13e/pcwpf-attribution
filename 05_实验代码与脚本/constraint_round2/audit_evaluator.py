"""
Audit Evaluator for Wind Power Forecasting:
Decoupled Decision Risk (Pinball Loss) and Market Settlement Diagnostic.

Key Design:
1. Primary Metric: Asymmetric Decision Risk (Pinball Loss / Asymmetric Dispatch Cost)
   L_q(y, y_hat) = max(q * (y - y_hat), (1 - q) * (y_hat - y))
2. Secondary / Diagnostic Metric: UK Elexon BSC Imbalance Cashflow & Shortfall
   Cashflow = (Actual - Bid) * SystemPrice
   [AUDIT NOTICE] Imbalance cashflow alone is NOT net economic profit. It omits day-ahead contract
   revenue. Degenerate baselines like Zero-Bid (bidding 0) produce artificially high positive cashflow
   (£220k-240k/turb-yr) because shortfall is zero and all generation is liquidated at imbalance price.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def pinball_loss(y_true, y_pred, q):
    """
    Compute asymmetric linear pinball loss (dispatch risk).
    q in (0, 1): ratio representing cost of shortfall vs surplus.
    """
    diff = y_true - y_pred
    return np.maximum(q * diff, (q - 1.0) * diff).mean()

def evaluate_predictions(y_true, y_pred, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
    """
    Evaluate point forecast and asymmetric decision risk across quantiles.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    bias = np.mean(y_pred - y_true)
    
    pinball_scores = {f"pinball_q{int(q*100):02d}": pinball_loss(y_true, y_pred, q) for q in quantiles}
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        **pinball_scores
    }

def settlement_diagnostic(y_true, y_bid, prices_30m, turbine_count, capacity_kw):
    """
    Compute UK Elexon BSC 30-minute settlement cashflow component for diagnosis.
    Note: y_true and y_bid are 30-minute aggregate energy in pu.
    """
    # Shortfall: Bid > Actual -> must buy back shortfall
    shortfall_mwh = np.maximum(0.0, y_bid - y_true) * capacity_kw / 1000.0 * 0.5
    # Net imbalance: (Actual - Bid) * price
    imbalance_mwh = (y_true - y_bid) * capacity_kw / 1000.0 * 0.5
    
    shortfall_gbp = shortfall_mwh * prices_30m
    imbalance_cashflow_gbp = imbalance_mwh * prices_30m
    
    # Annualized per turbine
    annual_factor = 365.25 * 48 / len(y_true)
    shortfall_per_turb_yr = (shortfall_gbp.sum() * annual_factor) / turbine_count
    cashflow_per_turb_yr = (imbalance_cashflow_gbp.sum() * annual_factor) / turbine_count
    
    return {
        "shortfall_gbp_per_turb_yr": float(shortfall_per_turb_yr),
        "net_imbalance_cashflow_gbp_per_turb_yr": float(cashflow_per_turb_yr)
    }

def print_audit_banner():
    print("=" * 80)
    print(" [AUDIT NOTICE] DECISION EVALUATION VS SETTLEMENT DIAGNOSTIC")
    print(" 1. Primary Metric: Non-negative Asymmetric Decision Risk (Pinball Loss).")
    print(" 2. Diagnostic Metric: Balancing Net Cashflow Component (GBP).")
    print("    WARNING: Imbalance cashflow alone DOES NOT measure forecast economic value.")
    print("    It excludes forward/day-ahead power purchase agreements (PPA) and contracts.")
    print("    A degenerate Zero-Bid strategy (bidding 0) yields artificially large positive")
    print("    cashflow because all production is cashed out at system imbalance prices.")
    print("=" * 80)

if __name__ == "__main__":
    print_audit_banner()
    print("Self-test: verifying pinball loss and zero-bid properties...")
    y = np.array([0.2, 0.5, 0.8, 1.0])
    y_zero = np.zeros_like(y)
    y_perf = y.copy()
    
    res_zero = evaluate_predictions(y, y_zero)
    res_perf = evaluate_predictions(y, y_perf)
    
    print(f"Perfect forecast pinball_q50: {res_perf['pinball_q50']:.6f}")
    print(f"Zero-bid forecast pinball_q50: {res_zero['pinball_q50']:.6f}")
    assert res_perf["pinball_q50"] == 0.0
    assert res_zero["pinball_q50"] > 0.0
    print("PASS: Audit evaluator self-test successful.")
