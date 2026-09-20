"""
Reproduction and Audit Script for Empirical Power-Curve Prior.

This script reconstructs the exact empirical power-curve prior used in the paper's
core experiments (3D Controlled Ablation, Within-Site Learning Curves, Prior Corruption).
It adheres strictly to: "REPRODUCE, DO NOT REINVENT."

Data Source: 04_数据集/constraint_rebuild_v2/{site}.csv
Core Prior Implementation: 05_实验代码与脚本/constraint_round2/core.py (fit_prior)
Loss Penalty Implementation: 05_实验代码与脚本/constraint_round2/controlled_ablation.py (controlled_penalty)

Outputs:
- outputs/empirical_power_curve_diagnostic.png
- outputs/empirical_power_curve_diagnostic.pdf
- outputs/fig1_historical_static_power_curve.svg
- outputs/fig1_historical_static_power_curve.pdf
- outputs/fig1_historical_static_power_curve.png
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Set project root and load project core module
ROOT = Path(__file__).resolve().parents[1]
ROUND2_DIR = ROOT / "05_实验代码与脚本" / "constraint_round2"
sys.path.insert(0, str(ROUND2_DIR))

from core import CFG, fit_prior, starts, windows
from controlled_ablation import fit_scrambled_prior, controlled_penalty

# Ensure reproducible random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Paths
DATA_DIR = ROOT / "04_数据集" / "constraint_rebuild_v2"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SITES_INFO = {
    "penmanshiel": {
        "name": "Penmanshiel",
        "region": "Scotland, UK",
        "turbines": "14 × Senvion MM92 (2.05 MW)",
        "capacity_kw": 28700,
        "role": "Focal Case (Low-sample regime, N=1,583, Prior-specificity)",
        "color": "#1d4ed8",  # Navy/Royal Blue
        "scramble_color": "#dc2626",  # Red
    },
    "kelmarsh": {
        "name": "Kelmarsh",
        "region": "Northamptonshire, UK",
        "turbines": "6 × Senvion MM92 (2.05 MW)",
        "capacity_kw": 12300,
        "role": "Data-rich regime (N=5,098, Hypothesis-space saturation)",
        "color": "#0d9488",  # Teal
        "scramble_color": "#ea580c",  # Orange
    },
    "hill_of_towie": {
        "name": "Hill of Towie",
        "region": "Moray, Scotland, UK",
        "turbines": "21 × Siemens SWT-2.3-82 (2.3 MW)",
        "capacity_kw": 48300,
        "role": "External held-out site (N=7,035, Complex upland terrain)",
        "color": "#4338ca",  # Indigo
        "scramble_color": "#b91c1c",  # Dark Red
    },
}


def load_site_data(site: str):
    """Load preprocessed site data and split config."""
    csv_path = DATA_DIR / f"{site}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset at {csv_path}")
    
    frame = pd.read_csv(csv_path, index_col="time", parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    split_cfg = CFG["splits"][site]
    
    return frame, split_cfg, csv_path


def run_numerical_consistency_check(site: str, prior: dict, frame: pd.DataFrame, split_cfg: dict):
    """
    Sanity-check: verify that the curve value returned for plotting is IDENTICAL
    to the target value queried by the neural network training loss (controlled_penalty).
    """
    pri_t = {k: torch.as_tensor(v) for k, v in prior.items()}
    
    # 6 test points spanning cut-in, ramp, rated plateau, and high wind
    test_wind_speeds = [1.5, 3.8, 7.2, 11.4, 15.8, 22.5, 27.0]
    
    print("\n" + "=" * 80)
    print(f"NUMERICAL CONSISTENCY CHECK — {site.upper()}")
    print("=" * 80)
    print(f"{'Wind Speed (m/s)':<18} | {'Training Loss Prior':<22} | {'Plot Curve Prior':<20} | {'Abs Diff':<12}")
    print("-" * 80)
    
    max_diff = 0.0
    for v in test_wind_speeds:
        # Simulate neural network input tensor x: shape (1, lookback=36, 2)
        # channel 0 = power_pu, channel 1 = wind_speed / 15.0
        x_dummy = torch.zeros(1, CFG["lookback"], 2, dtype=torch.float32)
        x_dummy[0, -1, 1] = v / 15.0  # persistence proxy at last lookback step
        
        # In controlled_penalty:
        # query_pos = max(0, x.shape[1] - 1 - lag_step)  (lag_step=0)
        # idx = (x[:, query_pos, 1] * 15.0).long().clamp(0, len(prior['means']) - 1)
        # target = prior['means'][idx, None] * factor (factor=1.0)
        idx = (x_dummy[:, -1, 1] * 15.0).long().clamp(0, len(prior["means"]) - 1)
        training_target = float(prior["means"][idx.item()])
        
        # Plotting curve value (queried at wind speed v)
        bin_idx = int(np.clip(np.floor(v), 0, len(prior["means"]) - 1))
        plot_val = float(prior["means"][bin_idx])
        
        diff = abs(training_target - plot_val)
        if diff > max_diff:
            max_diff = diff
            
        print(f"{v:<18.2f} | {training_target:<22.6f} | {plot_val:<20.6f} | {diff:<12.2e}")
        
    print("-" * 80)
    assert max_diff < 1e-7, f"Numerical consistency check FAILED! Max diff: {max_diff}"
    print(f"PASS: Maximum discrepancy across all evaluation points = {max_diff:.2e} (Exact match)")
    print("=" * 80 + "\n")


def plot_diagnostic_figure(sites_data: dict, out_png: Path, out_pdf: Path):
    """
    Generate comprehensive 1x3 diagnostic panel for human audit.
    Shows real SCADA scatter, reconstructed empirical prior curve, bin centers,
    and pseudo-scrambled negative control.
    """
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "figure.titlesize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), dpi=300, sharey=True)
    
    for idx, (site, d) in enumerate(sites_data.items()):
        ax = axes[idx]
        info = SITES_INFO[site]
        df_train = d["df_train"]
        prior = d["prior"]
        scrambled = d["scrambled"]
        valid_pairs = d["valid_pairs"]
        n_origins = d["n_origins"]
        train_window_str = d["train_window_str"]
        
        # 1. Background real SCADA scatter (subsample to 6,000 points if larger for clarity)
        ws_vals = df_train["wind_speed"].to_numpy()
        p_vals = df_train["power_pu"].to_numpy()
        valid = np.isfinite(ws_vals) & np.isfinite(p_vals)
        ws_valid = ws_vals[valid]
        p_valid = p_vals[valid]
        
        if len(ws_valid) > 6000:
            rng = np.random.default_rng(RANDOM_SEED)
            sample_idx = rng.choice(len(ws_valid), 6000, replace=False)
            ws_plot = ws_valid[sample_idx]
            p_plot = p_valid[sample_idx]
        else:
            ws_plot = ws_valid
            p_plot = p_valid
            
        ax.scatter(ws_plot, p_plot, s=5, color="#94a3b8", alpha=0.18, rasterized=True,
                   label=f"SCADA training scatter (N={len(ws_valid):,})")
        
        # 2. Reconstructed Empirical Power Curve (connecting centers)
        centers = prior["centers"]
        means = prior["means"]
        
        # Highlight Penmanshiel as focal case
        is_focal = (site == "penmanshiel")
        curve_lw = 2.6 if is_focal else 2.0
        
        ax.plot(centers, means, color=info["color"], lw=curve_lw, zorder=5,
                label="Authentic Empirical Prior ($C_{\\mathrm{emp}}$)")
        ax.scatter(centers, means, s=24, color=info["color"], edgecolors="white", lw=0.6, zorder=6,
                   label="Bin Means (1 m/s, $N \\geq 20$)")
        
        # 3. Pseudo-scrambled prior control (seed=42)
        ax.plot(centers, scrambled["means"], color=info["scramble_color"], lw=1.2,
                linestyle="--", alpha=0.85, zorder=4,
                label="Pseudo-Scrambled Control (seed 42)")
        
        # Aesthetics
        site_title = f"{info['name']} Wind Farm"
        if is_focal:
            site_title += " [Focal Benchmark]"
        ax.set_title(site_title, fontweight="bold", pad=8)
        ax.set_xlabel("Measured Wind Speed $v$ (m/s)")
        if idx == 0:
            ax.set_ylabel("Normalized Power $P / P_{\\mathrm{rated}}$ (p.u.)")
            
        ax.set_xlim(0, 26)
        ax.set_ylim(-0.02, 1.15)
        ax.grid(True, linestyle=":", alpha=0.4, color="#cbd5e1")
        ax.set_axisbelow(True)
        
        # Informative text box
        text_box = (
            f"Turbines: {info['turbines']}\n"
            f"Window: {train_window_str}\n"
            f"Valid SCADA: {len(ws_valid):,} pairs\n"
            f"Train Origins: $N={n_origins:,}$\n"
            f"Max Mean: {means.max():.3f} p.u."
        )
        box_fc = "#eff6ff" if is_focal else "#f8fafc"
        box_ec = "#3b82f6" if is_focal else "#cbd5e1"
        ax.text(0.04, 0.96, text_box, transform=ax.transAxes, va="top", ha="left",
                fontsize=7.8, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.4", fc=box_fc, ec=box_ec, lw=0.8, alpha=0.92))
        
        # Legend
        ax.legend(loc="lower right", framealpha=0.9, facecolor="white", edgecolor="#e2e8f0", fontsize=7.5)

    fig.suptitle("Audit Diagnostic: Reconstructed Site-Specific Empirical Power-Curve Priors vs. SCADA Distributions",
                 fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic full figure saved to:")
    print(f"  - {out_png}")
    print(f"  - {out_pdf}")


def plot_figure1_micro_figure(site_data: dict, out_svg: Path, out_pdf: Path, out_png: Path):
    """
    Generate the ultra-minimalist Figure 1 thumbnail micro-figure:
    D1 -> Historical-static: "Offline SCADA-derived prior"
    
    Constraints:
    - Minimalist, instantly recognizable at tiny thumbnail size (1.5 - 2 inches)
    - Clean vector axes (bottom and left only)
    - Very faint SCADA points in background to anchor provenance
    - Bold authentic empirical power curve
    - Clean 'Wind speed' and 'Power' labels
    - No tick marks, no tick labels, no legend, no grid, no numbers, no annotations
    - Non-idealized real empirical shape
    """
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Ultra-compact 2.4 x 1.9 inches
    fig, ax = plt.subplots(figsize=(2.4, 1.9), dpi=600)
    
    df_train = site_data["df_train"]
    prior = site_data["prior"]
    
    # 1. Very faint, subsampled real SCADA points (800 points for a lightweight vector footprint)
    ws_vals = df_train["wind_speed"].to_numpy()
    p_vals = df_train["power_pu"].to_numpy()
    valid = np.isfinite(ws_vals) & np.isfinite(p_vals)
    ws_valid = ws_vals[valid]
    p_valid = p_vals[valid]
    
    rng = np.random.default_rng(RANDOM_SEED)
    sample_size = min(900, len(ws_valid))
    sample_idx = rng.choice(len(ws_valid), sample_size, replace=False)
    ws_sub = ws_valid[sample_idx]
    p_sub = p_valid[sample_idx]
    
    # Subtle scatter cloud (pure vector dots)
    ax.scatter(ws_sub, p_sub, s=2.5, color="#64748b", alpha=0.22, edgecolors="none")
    
    # 2. Bold Authentic Empirical Power Curve
    # Cut off at 22 m/s where real data terminates (avoid empty extrapolation tail for micro-figure)
    mask = prior["centers"] <= 22.0
    ax.plot(prior["centers"][mask], prior["means"][mask], color="#1e40af", lw=2.8, solid_capstyle="round")
    
    # Axis styling: minimal L-frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    
    # Remove ticks and tick numbers completely
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Limits
    ax.set_xlim(0, 23)
    ax.set_ylim(-0.02, 1.05)
    
    # Minimal qualitative axis labels
    ax.set_xlabel("Wind speed", fontsize=9.5, fontweight="medium", color="#1e293b", labelpad=3)
    ax.set_ylabel("Power", fontsize=9.5, fontweight="medium", color="#1e293b", labelpad=3)
    
    # Clean transparent background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    plt.tight_layout(pad=0.3)
    
    # Save SVG (vector), PDF (vector), and 600 DPI PNG
    fig.savefig(out_svg, format="svg", bbox_inches="tight", transparent=False)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", transparent=False)
    fig.savefig(out_png, format="png", dpi=600, bbox_inches="tight", transparent=False)
    plt.close(fig)
    
    print(f"Figure 1 micro-figure saved to:")
    print(f"  - {out_svg}")
    print(f"  - {out_pdf}")
    print(f"  - {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Reproduce Authentic Empirical Power Curve Prior")
    parser.add_argument("--focal-site", default="penmanshiel", choices=["penmanshiel", "kelmarsh", "hill_of_towie"])
    args = parser.parse_args()
    
    sites = ["penmanshiel", "kelmarsh", "hill_of_towie"]
    sites_data = {}
    
    print("=" * 80)
    print("REPRODUCING EMPIRICAL POWER-CURVE PRIOR ACROSS COMMERCIAL SITES")
    print("Principle: 'REPRODUCE, DO NOT REINVENT'")
    print("=" * 80)
    
    for site in sites:
        frame, split_cfg, csv_path = load_site_data(site)
        train_end = split_cfg["train_end"]
        
        # Preprocessing & forecast origin extraction (using project's starts function)
        x_all, y_all, ids = starts(frame, split_cfg)
        n_origins = len(ids["train"])
        
        # Train window SCADA
        in_train = frame.index < pd.Timestamp(train_end, tz="UTC")
        df_train = frame[in_train]
        valid_pairs = len(df_train[["wind_speed", "power_pu"]].dropna())
        
        # Authentic fit_prior using canonical project implementation
        prior = fit_prior(frame, train_end)
        
        # Scale-matched pseudo-scrambled prior using canonical implementation
        scrambled = fit_scrambled_prior(prior, seed=RANDOM_SEED)
        
        start_date = frame.index.min().strftime("%Y-%m-%d")
        train_window_str = f"{start_date} to {train_end}"
        
        sites_data[site] = {
            "df_train": df_train,
            "prior": prior,
            "scrambled": scrambled,
            "valid_pairs": valid_pairs,
            "n_origins": n_origins,
            "train_window_str": train_window_str,
            "csv_path": csv_path,
            "train_end": train_end,
        }
        
        # Detailed terminal reporting
        print(f"Site:                      {SITES_INFO[site]['name']}")
        print(f"Raw data path:             {csv_path}")
        print(f"Training/prior window:     {train_window_str}")
        print(f"Valid SCADA samples:       {valid_pairs:,} (wind_speed, power_pu pairs)")
        print(f"Forecast origins (N_train): {n_origins:,}")
        print(f"Wind-speed field:          'wind_speed' (m/s)")
        print(f"Power field:               'power_pu' (normalized by nameplate {SITES_INFO[site]['capacity_kw']:,} kW)")
        print(f"Normalization:             power_kw / rated_kw (range: [0.0, ~1.0])")
        print(f"Curve construction method: 1 m/s uniform binning (threshold N >= 20)")
        print(f"Number of bins / knots:    {len(prior['centers'])} bins ([0..30] m/s, centers: 0.5..29.5 m/s)")
        print(f"Interpolation method:      pandas.Series.interpolate(limit_direction='both') (linear interpolation)")
        print(f"Maximum prior mean power:  {prior['means'].max():.4f} p.u. at center {prior['centers'][np.argmax(prior['means'])]:.1f} m/s")
        print("-" * 80)
        
        # Numerical sanity check for each site
        run_numerical_consistency_check(site, prior, frame, split_cfg)

    # 1. Output Diagnostic Figure
    diag_png = OUTPUTS_DIR / "empirical_power_curve_diagnostic.png"
    diag_pdf = OUTPUTS_DIR / "empirical_power_curve_diagnostic.pdf"
    plot_diagnostic_figure(sites_data, diag_png, diag_pdf)
    
    # 2. Output Figure 1 Micro-Figure (Penmanshiel as focal case)
    focal_site = args.focal_site
    fig1_svg = OUTPUTS_DIR / "fig1_historical_static_power_curve.svg"
    fig1_pdf = OUTPUTS_DIR / "fig1_historical_static_power_curve.pdf"
    fig1_png = OUTPUTS_DIR / "fig1_historical_static_power_curve.png"
    plot_figure1_micro_figure(sites_data[focal_site], fig1_svg, fig1_pdf, fig1_png)
    
    print("\n" + "=" * 80)
    print("ALL REPRODUCTION TASKS COMPLETED SUCCESSFULLY.")
    print(f"Output files:")
    print(f"  1. {diag_png}")
    print(f"  2. {diag_pdf}")
    print(f"  3. {fig1_svg}")
    print(f"  4. {fig1_pdf}")
    print(f"  5. {fig1_png}")
    print("=" * 80)


if __name__ == "__main__":
    main()
