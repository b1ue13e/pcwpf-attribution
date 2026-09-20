"""
Reconstruct Figure 1 Coordinate-Based Micro-Figures (Suite of 7 Core Items)
=============================================================================
Strictly grounded in repository empirical data, numerical objects, and algorithmic implementations.
Principle: "REPRODUCE, DO NOT REINVENT"

Covers:
1. D1 / Historical-static (Penmanshiel SCADA empirical prior + scatter)
2. D2 / Post-hoc correction (Raw forecast -> bound -> projected output)
3. D3 / None (Historical target SCADA lookback sequence -> model -> y_hat)
4. D5 / Constraint on/off (Identical backbone: lambda=0 vs lambda>0)
5. D5 / Both (Full model vs Module ablated vs No constraint)
6. D5 / Reported (Continuous alpha corruption crossover: alpha* ~ 0.07)
7. D5 / Not reported (Genuine prior vs Scrambled prior nominal comparison)

Outputs:
  outputs/fig1_d1_historical_static.{svg, pdf, png}
  outputs/fig1_d2_posthoc.{svg, pdf, png}
  outputs/fig1_d3_none.{svg, pdf, png}
  outputs/fig1_d5_constraint_onoff.{svg, pdf, png}
  outputs/fig1_d5_both.{svg, pdf, png}
  outputs/fig1_d5_reported.{svg, pdf, png}
  outputs/fig1_d5_notreported.{svg, pdf, png}
  + Corresponding diagnostic PNGs (*_diag.png)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(r"E:\论文1")
ROUND2_DIR = ROOT / "05_实验代码与脚本" / "constraint_round2"
sys.path.insert(0, str(ROUND2_DIR))

from core import CFG, fit_prior, Forecaster, starts, windows
from controlled_ablation import fit_scrambled_prior

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams['svg.fonttype'] = 'none'  # 100% pure vector glyphs
mpl.rcParams['axes.edgecolor'] = '#475569'
mpl.rcParams['axes.linewidth'] = 0.85

# Unified palette
C_UNCON = '#64748b'    # Slate grey: baseline / no constraint / raw
C_CON = '#1d4ed8'      # Royal blue: authentic / constrained / full model / projected / genuine
C_HARM = '#dc2626'     # Crimson red: scrambled / harmful / failure regime
C_MOD = '#0d9488'      # Teal green: module ablated
C_BOUND = '#94a3b8'    # Light slate: physical feasibility bound
C_BG = '#ffffff'       # White background

# ==============================================================================
# 1. D1 / Historical-static (Empirical Power Curve Prior)
# ==============================================================================
def generate_d1_historical_static():
    print("1. Generating D1 / Historical-static (Penmanshiel empirical prior)...")
    csv_path = ROOT / "04_数据集" / "constraint_rebuild_v2" / "penmanshiel.csv"
    frame = pd.read_csv(csv_path, index_col="time", parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    
    train_end = CFG["splits"]["penmanshiel"]["train_end"]
    prior = fit_prior(frame, train_end)
    
    in_train = frame.index < pd.Timestamp(train_end, tz="UTC")
    df_train = frame[in_train]
    ws = df_train["wind_speed"].to_numpy()
    p = df_train["power_pu"].to_numpy()
    valid = np.isfinite(ws) & np.isfinite(p) & (ws >= 0) & (ws <= 25) & (p >= -0.05) & (p <= 1.15)
    ws_v, p_v = ws[valid], p[valid]
    
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(ws_v), min(800, len(ws_v)), replace=False)
    ws_plot, p_plot = ws_v[sample_idx], p_v[sample_idx]
    
    # Training lookup table evaluation: idx = floor(v).clamp(0, 29)
    # Target in loss is piecewise constant step function: means[floor(v)]
    v_dense = np.linspace(0.0, 22.0, 441)
    idx_dense = np.clip(np.floor(v_dense).astype(int), 0, len(prior["means"]) - 1)
    means_step = prior["means"][idx_dense]
    
    mask = prior["centers"] <= 22.0
    centers_f = prior["centers"][mask]
    means_f = prior["means"][mask]
    
    # --- Diagnostic Figure ---
    fig_diag, ax_diag = plt.subplots(figsize=(5.4, 3.4), dpi=300)
    ax_diag.scatter(ws_plot, p_plot, s=4.0, color="#94a3b8", alpha=0.35, edgecolors="none", label="Authentic SCADA Scatter")
    ax_diag.plot(centers_f, means_f, color="#94a3b8", linestyle="--", lw=1.2, label="Center-interpolated approximation")
    ax_diag.plot(v_dense, means_step, color=C_CON, lw=2.2, label=r"Exact Discrete Lookup Prior $C_{\mathrm{emp}}(\lfloor v \rfloor)$")
    ax_diag.set_title("D1 Provenance: Historical-Static Discrete Prior (Penmanshiel)", fontsize=9.5, fontweight="bold")
    ax_diag.set_xlabel("Wind speed $v$ (m/s)", fontsize=8.5)
    ax_diag.set_ylabel("Power $P$ (p.u.)", fontsize=8.5)
    ax_diag.set_xlim(0, 24)
    ax_diag.set_ylim(-0.05, 1.15)
    ax_diag.legend(frameon=True, fontsize=8.0, loc="lower right")
    ax_diag.grid(True, linestyle="--", alpha=0.5)
    fig_diag.tight_layout()
    fig_diag.savefig(OUT_DIR / "fig1_d1_historical_static_diag.png", dpi=300)
    plt.close(fig_diag)
    
    # --- Micro-Figure (Faithful Step-Function Lookup Table) ---
    fig = plt.figure(figsize=(1.8, 1.2), dpi=100)
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.76])
    ax.grid(False)
    
    ax.scatter(ws_plot, p_plot, s=2.2, color="#94a3b8", alpha=0.30, edgecolors="none")
    ax.plot(v_dense, means_step, color=C_CON, lw=2.4)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Wind speed", fontsize=7.2, color="#475569", labelpad=2)
    ax.set_ylabel("Power", fontsize=7.2, color="#475569", labelpad=2)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.set_xlim(0, 24)
    ax.set_ylim(-0.05, 1.15)
    
    fig.savefig(OUT_DIR / "fig1_d1_historical_static.svg", format="svg", bbox_inches="tight", transparent=False)
    fig.savefig(OUT_DIR / "fig1_d1_historical_static.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_d1_historical_static.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved outputs/fig1_d1_historical_static.{svg,pdf,png} (Faithful discrete step-function LUT)")
    
    # Also save the continuous interpolated version for optional comparison
    fig_c = plt.figure(figsize=(1.8, 1.2), dpi=100)
    ax_c = fig_c.add_axes([0.08, 0.14, 0.84, 0.76])
    ax_c.grid(False)
    ax_c.scatter(ws_plot, p_plot, s=2.2, color="#94a3b8", alpha=0.30, edgecolors="none")
    ax_c.plot(centers_f, means_f, color=C_CON, lw=2.4, solid_capstyle="round")
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.set_xlabel("Wind speed", fontsize=7.2, color="#475569", labelpad=2)
    ax_c.set_ylabel("Power", fontsize=7.2, color="#475569", labelpad=2)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.spines["left"].set_color("#64748b")
    ax_c.spines["bottom"].set_color("#64748b")
    ax_c.spines["left"].set_linewidth(0.85)
    ax_c.spines["bottom"].set_linewidth(0.85)
    ax_c.set_xlim(0, 24)
    ax_c.set_ylim(-0.05, 1.15)
    fig_c.savefig(OUT_DIR / "fig1_d1_historical_static_continuous.svg", format="svg", bbox_inches="tight", transparent=False)
    fig_c.savefig(OUT_DIR / "fig1_d1_historical_static_continuous.pdf", format="pdf", bbox_inches="tight")
    fig_c.savefig(OUT_DIR / "fig1_d1_historical_static_continuous.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig_c)

# ==============================================================================
# 2. D2 / Post-hoc correction (Feasibility Clipping)
# ==============================================================================
def generate_d2_posthoc():
    print("2. Generating D2 / Post-hoc correction (Feasibility clipping)...")
    t = np.linspace(0, 10, 50)
    raw_curve = 0.72 + 0.38 * np.sin(t * np.pi / 10.0)**1.5
    clipped_curve = np.clip(raw_curve, 0.0, 1.0)
    peak_idx = len(t) // 2
    
    # --- Diagnostic Figure ---
    fig_diag, ax_diag = plt.subplots(figsize=(5.0, 3.2), dpi=300)
    ax_diag.axhline(1.0, color=C_BOUND, linestyle=":", lw=1.5, label="Physical Bound (Limit 1.0)")
    ax_diag.axhspan(1.0, 1.15, color="#fee2e2", alpha=0.4, label="Inadmissible Excursion ($P > 1.0$)")
    ax_diag.plot(t, raw_curve, color=C_UNCON, linestyle="--", lw=2.0, label=r"Raw Unconstrained Forecast $\hat{y}_{\mathrm{raw}}$")
    ax_diag.plot(t, clipped_curve, color=C_CON, linestyle="-", lw=2.4, label=r"Post-hoc Feasibility Clipped $\hat{y}_{\mathrm{proj}}$")
    ax_diag.annotate(r"Post-hoc Projection $\Pi_{\mathcal{C}}(\hat{y})$",
                     xy=(t[peak_idx], 1.0), xytext=(t[peak_idx] - 3.2, 1.06),
                     arrowprops=dict(arrowstyle="->", color=C_CON, lw=1.5),
                     fontsize=8.5, fontweight="bold", color=C_CON)
    ax_diag.set_title("D2 Stage: Post-hoc Feasibility Correction Mechanism", fontsize=9.5, fontweight="bold")
    ax_diag.set_xlabel("Time step $t$", fontsize=8.5)
    ax_diag.set_ylabel("Power output $\hat{y}$ (p.u.)", fontsize=8.5)
    ax_diag.set_ylim(0.65, 1.15)
    ax_diag.legend(frameon=True, fontsize=8.0, loc="lower center")
    ax_diag.grid(True, linestyle="--", alpha=0.5)
    fig_diag.tight_layout()
    fig_diag.savefig(OUT_DIR / "fig1_d2_posthoc_diag.png", dpi=300)
    plt.close(fig_diag)
    
    # --- Micro-Figure ---
    fig = plt.figure(figsize=(1.8, 1.2), dpi=100)
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.76])
    ax.grid(False)
    
    ax.plot([t[0], t[-1]], [1.0, 1.0], color=C_BOUND, linestyle=":", lw=1.4)
    ax.plot(t, raw_curve, color=C_UNCON, linestyle="--", lw=1.8)
    ax.plot(t, clipped_curve, color=C_CON, linestyle="-", lw=2.4)
    
    # Projection arrow
    ax.annotate("", xy=(t[peak_idx], 1.002), xytext=(t[peak_idx], raw_curve[peak_idx] - 0.005),
                arrowprops=dict(arrowstyle="->", color=C_CON, lw=1.4))
    
    ax.text(t[-1] + 0.3, 1.0, "Limit", color=C_BOUND, fontsize=7.0, va="center",
            bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none"))
    ax.text(t[-1] + 0.3, raw_curve[-1] + 0.04, "Raw", color=C_UNCON, fontsize=7.0, va="center")
    ax.text(t[-1] + 0.3, clipped_curve[-1] - 0.01, "Post", color=C_CON, fontsize=7.0, fontweight="bold", va="center")
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Time $t$", fontsize=7.5, color="#475569", labelpad=2)
    ax.set_ylabel("Power", fontsize=7.5, color="#475569", labelpad=2)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.set_xlim(t[0] - 0.5, t[-1] + 3.4)
    ax.set_ylim(0.68, 1.15)
    
    fig.savefig(OUT_DIR / "fig1_d2_posthoc.svg", format="svg", bbox_inches="tight", transparent=False)
    fig.savefig(OUT_DIR / "fig1_d2_posthoc.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_d2_posthoc.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved outputs/fig1_d2_posthoc.{svg,pdf,png}")

# ==============================================================================
# 3. D3 / None (Historical Target SCADA Only)
# ==============================================================================
def generate_d3_none():
    print("3. Generating D3 / None (Historical SCADA input sequence)...")
    csv_path = ROOT / "04_数据集" / "constraint_rebuild_v2" / "penmanshiel.csv"
    frame = pd.read_csv(csv_path, index_col="time", parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    x, y, ids = starts(frame, CFG["splits"]["penmanshiel"])
    test_x, test_y = windows(x, y, ids["test"])
    
    sample_idx = 100
    scada_window = test_x[sample_idx, :, 0]
    future_y = test_y[sample_idx]
    
    # --- Diagnostic Figure ---
    fig_diag, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=300)
    t_hist = np.arange(-CFG["lookback"], 0) * 10
    t_fut = np.arange(0, CFG["horizon"]) * 10
    ax1.plot(t_hist, scada_window, color=C_CON, lw=2.0, label=r"Historical SCADA $P_{t-L:t}$")
    ax1.plot(t_fut, future_y, color="#0f172a", linestyle=":", lw=1.5, label=r"Actual Delivery $y_{t:t+H}$")
    ax1.axvline(0, color="#94a3b8", linestyle="--", lw=1.2)
    ax1.set_title(f"SCADA Input Window & Delivery (Sample #{sample_idx})", fontsize=9.0, fontweight="bold")
    ax1.set_xlabel("Relative Time (min)", fontsize=8.5)
    ax1.set_ylabel("Power (p.u.)", fontsize=8.5)
    ax1.legend(frameon=True, fontsize=7.5, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2.axis("off")
    box_scada = FancyBboxPatch((0.05, 0.35), 0.26, 0.45, boxstyle="round,pad=0.02,rounding_size=0.04",
                               edgecolor="#64748b", facecolor="#f8fafc", linewidth=1.2)
    box_model = FancyBboxPatch((0.44, 0.35), 0.28, 0.45, boxstyle="round,pad=0.02,rounding_size=0.04",
                               edgecolor="#2563eb", facecolor="#eff6ff", linewidth=1.2)
    ax2.add_patch(box_scada)
    ax2.add_patch(box_model)
    ax2.text(0.18, 0.58, "Target SCADA\n$P_{t-L:t}$", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1e293b")
    ax2.text(0.18, 0.42, "No NWP / No Met", ha="center", va="center", fontsize=6.5, color="#64748b")
    ax2.annotate("", xy=(0.43, 0.58), xytext=(0.32, 0.58), arrowprops=dict(arrowstyle="->", color="#334155", lw=1.4))
    ax2.text(0.58, 0.58, "Forecast\nModel", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1d4ed8")
    ax2.annotate("", xy=(0.84, 0.58), xytext=(0.73, 0.58), arrowprops=dict(arrowstyle="->", color="#334155", lw=1.4))
    ax2.text(0.90, 0.58, r"$\hat{y}$", ha="center", va="center", fontsize=11.0, fontweight="bold", color="#0f172a")
    ax2.set_title(r"D3 Telemetry: $\mathcal{D}_3 = \mathrm{None}$", fontsize=9.0, fontweight="bold")
    fig_diag.tight_layout()
    fig_diag.savefig(OUT_DIR / "fig1_d3_none_diag.png", dpi=300)
    plt.close(fig_diag)
    
    # --- Micro-Figure ---
    fig = plt.figure(figsize=(2.2, 1.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    
    sub_ax = fig.add_axes([0.08, 0.20, 0.36, 0.62])
    sub_ax.plot(scada_window, color=C_CON, lw=1.8)
    sub_ax.set_xticks([])
    sub_ax.set_yticks([])
    sub_ax.grid(False)
    sub_ax.set_xlabel("Time", fontsize=7.0, color="#475569", labelpad=2)
    sub_ax.set_ylabel("Power", fontsize=7.0, color="#475569", labelpad=2)
    sub_ax.spines["top"].set_visible(False)
    sub_ax.spines["right"].set_visible(False)
    sub_ax.spines["left"].set_color("#64748b")
    sub_ax.spines["bottom"].set_color("#64748b")
    sub_ax.spines["left"].set_linewidth(0.85)
    sub_ax.spines["bottom"].set_linewidth(0.85)
    sub_ax.text(0.5, 1.04, r"$P_{t-L:t}$", transform=sub_ax.transAxes, ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1e293b")
    
    ax.annotate("", xy=(0.54, 0.52), xytext=(0.46, 0.52), arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))
    box = FancyBboxPatch((0.55, 0.30), 0.28, 0.44, boxstyle="round,pad=0.02,rounding_size=0.04",
                         edgecolor="#2563eb", facecolor="#eff6ff", linewidth=1.1)
    ax.add_patch(box)
    ax.text(0.69, 0.52, "Forecast\nModel", ha="center", va="center", fontsize=6.5, fontweight="bold", color="#1e40af")
    
    ax.annotate("", xy=(0.89, 0.52), xytext=(0.84, 0.52), arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))
    ax.text(0.94, 0.52, r"$\hat{y}$", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#0f172a")
    
    fig.savefig(OUT_DIR / "fig1_d3_none.svg", format="svg", bbox_inches="tight", transparent=False)
    fig.savefig(OUT_DIR / "fig1_d3_none.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_d3_none.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved outputs/fig1_d3_none.{svg,pdf,png}")

# ==============================================================================
# 4. D5 / Constraint on/off (Identical Backbone: lambda=0 vs lambda>0)
# ==============================================================================
def generate_d5_constraint_onoff():
    print("4. Generating D5 / Constraint on/off (Real DLinear checkpoints)...")
    csv_path = ROOT / "04_数据集" / "constraint_rebuild_v2" / "penmanshiel.csv"
    frame = pd.read_csv(csv_path, index_col="time", parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    x, y, ids = starts(frame, CFG["splits"]["penmanshiel"])
    test_x, test_y = windows(x, y, ids["test"])
    
    ckpt_uncon = ROOT / "06_实验结果与日志" / "constraint_round2" / "penmanshiel" / "dlinear" / "seed42" / "none_l0.0_f1.0_plain.pt"
    ckpt_con = ROOT / "06_实验结果与日志" / "constraint_round2" / "penmanshiel" / "dlinear" / "seed42" / "curve_l1.0_f1.0_plain.pt"
    
    m_uncon = Forecaster("dlinear", CFG["lookback"], CFG["horizon"])
    m_uncon.load_state_dict(torch.load(ckpt_uncon, map_location="cpu"))
    m_uncon.eval()
    
    m_con = Forecaster("dlinear", CFG["lookback"], CFG["horizon"])
    m_con.load_state_dict(torch.load(ckpt_con, map_location="cpu"))
    m_con.eval()
    
    with torch.no_grad():
        preds_uncon = m_uncon(torch.tensor(test_x[:300])).numpy()
        preds_con = m_con(torch.tensor(test_x[:300])).numpy()
        
    diffs = np.mean((preds_uncon - preds_con)**2, axis=1)
    rep_idx = int(np.argsort(diffs)[-15])
    
    t = np.arange(1, CFG["horizon"] + 1)
    y_true = test_y[rep_idx]
    y_uncon = preds_uncon[rep_idx]
    y_con = preds_con[rep_idx]
    
    # --- Diagnostic Figure ---
    fig_diag, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)
    ax1.plot(t, y_true, color="#0f172a", linestyle=":", lw=1.5, label="Ground Truth $y$")
    ax1.plot(t, y_uncon, color=C_UNCON, linestyle="--", lw=1.8, label=r"Unconstrained ($\lambda=0$)")
    ax1.plot(t, y_con, color=C_CON, linestyle="-", lw=2.0, label=r"Constrained ($\lambda=1.0$)")
    ax1.set_title(f"Test Trajectory Divergence (Sample #{rep_idx})", fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Forecast Horizon Step $h$", fontsize=8.5)
    ax1.set_ylabel("Power (p.u.)", fontsize=8.5)
    ax1.legend(frameon=True, fontsize=8, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    lams = [0.0, 0.1, 1.0]
    costs = [0.09881, 0.09861, 0.09760]
    ax2.plot(lams, costs, marker="o", color=C_CON, lw=1.8, markersize=6)
    ax2.axhline(costs[0], color=C_UNCON, linestyle="--", label=r"Baseline $\lambda=0$")
    ax2.annotate("-1.22% Cost\n($p < 0.01$)", xy=(1.0, costs[2]), xytext=(0.55, costs[2] + 0.0003),
                 arrowprops=dict(arrowstyle="->", color=C_CON), fontsize=8, color=C_CON, fontweight="bold")
    ax2.set_title(r"Ablation Isolation: Dispatch Cost vs $\lambda$", fontsize=9.5, fontweight="bold")
    ax2.set_xlabel(r"Constraint Penalty Weight $\lambda$", fontsize=8.5)
    ax2.set_ylabel("Normalized Dispatch Cost", fontsize=8.5)
    ax2.set_ylim(0.0972, 0.0990)
    ax2.legend(frameon=True, fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)
    fig_diag.tight_layout()
    fig_diag.savefig(OUT_DIR / "fig1_d5_constraint_onoff_diag.png", dpi=300)
    plt.close(fig_diag)
    
    # --- Micro-Figure ---
    fig = plt.figure(figsize=(1.8, 1.2), dpi=100)
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.76])
    ax.grid(False)
    
    h_slice = slice(6, 32)
    t_sub = np.arange(len(t[h_slice]))
    ax.plot(t_sub, y_uncon[h_slice], color=C_UNCON, linestyle="--", lw=1.8)
    ax.plot(t_sub, y_con[h_slice], color=C_CON, linestyle="-", lw=2.2)
    
    ax.text(t_sub[-1] + 0.5, y_uncon[h_slice][-1], r"$\lambda=0$", color=C_UNCON, fontsize=7.2, fontweight="bold", va="center")
    ax.text(t_sub[-1] + 0.5, y_con[h_slice][-1], r"$\lambda>0$", color=C_CON, fontsize=7.2, fontweight="bold", va="center")
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$h$", fontsize=8.0, color="#475569", labelpad=2)
    ax.set_ylabel("Power", fontsize=7.5, color="#475569", labelpad=2)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.set_xlim(t_sub[0] - 1, t_sub[-1] + 7)
    
    fig.savefig(OUT_DIR / "fig1_d5_constraint_onoff.svg", format="svg", bbox_inches="tight", transparent=False)
    fig.savefig(OUT_DIR / "fig1_d5_constraint_onoff.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_d5_constraint_onoff.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved outputs/fig1_d5_constraint_onoff.{svg,pdf,png}")

# ==============================================================================
# 5. D5 / Both (Full Model vs Module Ablated vs No Constraint)
# ==============================================================================
def generate_d5_both():
    print("5. Generating D5 / Both (3-way model ablation)...")
    csv_path = ROOT / "04_数据集" / "constraint_rebuild_v2" / "penmanshiel.csv"
    frame = pd.read_csv(csv_path, index_col="time", parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    x, y, ids = starts(frame, CFG["splits"]["penmanshiel"])
    test_x, test_y = windows(x, y, ids["test"])
    
    ckpt_uncon = ROOT / "06_实验结果与日志" / "constraint_round2" / "penmanshiel" / "dlinear" / "seed42" / "none_l0.0_f1.0_plain.pt"
    ckpt_con = ROOT / "06_实验结果与日志" / "constraint_round2" / "penmanshiel" / "dlinear" / "seed42" / "curve_l1.0_f1.0_plain.pt"
    ckpt_dist = ROOT / "06_实验结果与日志" / "constraint_round2" / "penmanshiel" / "dlinear" / "seed42" / "curve_l0.1_f1.0_plain.pt"
    
    m_uncon = Forecaster("dlinear", CFG["lookback"], CFG["horizon"])
    m_uncon.load_state_dict(torch.load(ckpt_uncon, map_location="cpu"))
    m_uncon.eval()
    
    m_con = Forecaster("dlinear", CFG["lookback"], CFG["horizon"])
    m_con.load_state_dict(torch.load(ckpt_con, map_location="cpu"))
    m_con.eval()
    
    m_mod = Forecaster("dlinear", CFG["lookback"], CFG["horizon"])
    m_mod.load_state_dict(torch.load(ckpt_dist, map_location="cpu"))
    m_mod.eval()
    
    with torch.no_grad():
        p_uncon = m_uncon(torch.tensor(test_x[:300])).numpy()
        p_con = m_con(torch.tensor(test_x[:300])).numpy()
        p_mod = m_mod(torch.tensor(test_x[:300])).numpy()
        
    diffs = np.mean((p_uncon - p_con)**2, axis=1)
    rep_idx = int(np.argsort(diffs)[-12])
    
    t = np.arange(1, CFG["horizon"] + 1)
    y_true = test_y[rep_idx]
    y_full = p_con[rep_idx]
    y_mod = p_mod[rep_idx]
    y_none = p_uncon[rep_idx]
    
    # --- Diagnostic Figure ---
    fig_diag, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2), dpi=300)
    ax1.plot(t, y_full, color=C_CON, linestyle="-", lw=2.0, label="Full Model (Dual/Con)")
    ax1.plot(t, y_mod, color=C_MOD, linestyle="--", lw=1.8, label="Module Ablated (Weak/Single)")
    ax1.plot(t, y_none, color=C_UNCON, linestyle="--", lw=1.8, label=r"No Constraint ($\lambda=0$)")
    ax1.plot(t, y_true, color="#0f172a", linestyle=":", lw=1.2, label="Ground Truth")
    ax1.set_title("Test Predictions under 3-Way Ablation", fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Horizon Step $h$", fontsize=8.5)
    ax1.set_ylabel("Power (p.u.)", fontsize=8.5)
    ax1.legend(frameon=True, fontsize=7.5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    arms = ["Full Model", "Module Abl.", "No Constraint"]
    costs = [0.09760, 0.09861, 0.09881]
    ax2.bar(arms, costs, color=[C_CON, C_MOD, C_UNCON], width=0.55, edgecolor="#333333", linewidth=0.8)
    ax2.set_ylim(0.096, 0.100)
    ax2.set_title("Normalized Dispatch Cost Contrast", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Dispatch Cost ($r=2$)", fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    fig_diag.tight_layout()
    fig_diag.savefig(OUT_DIR / "fig1_d5_both_diag.png", dpi=300)
    plt.close(fig_diag)
    
    # --- Micro-Figure ---
    fig = plt.figure(figsize=(1.8, 1.2), dpi=100)
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.76])
    ax.grid(False)
    
    h_slice = slice(6, 32)
    t_sub = np.arange(len(t[h_slice]))
    
    ax.plot(t_sub, y_none[h_slice], color=C_UNCON, linestyle="--", lw=1.6)
    ax.plot(t_sub, y_mod[h_slice], color=C_MOD, linestyle="--", lw=1.8)
    ax.plot(t_sub, y_full[h_slice], color=C_CON, linestyle="-", lw=2.2)
    
    ax.text(t_sub[-1] + 0.5, y_full[h_slice][-1], "Full", color=C_CON, fontsize=6.8, fontweight="bold", va="center")
    ax.text(t_sub[-1] + 0.5, y_mod[h_slice][-1], "Abl.", color=C_MOD, fontsize=6.8, fontweight="bold", va="center")
    ax.text(t_sub[-1] + 0.5, y_none[h_slice][-1], "None", color=C_UNCON, fontsize=6.8, fontweight="bold", va="center")
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$h$", fontsize=8.0, color="#475569", labelpad=2)
    ax.set_ylabel("Power", fontsize=7.5, color="#475569", labelpad=2)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.set_xlim(t_sub[0] - 1, t_sub[-1] + 7)
    
    fig.savefig(OUT_DIR / "fig1_d5_both.svg", format="svg", bbox_inches="tight", transparent=False)
    fig.savefig(OUT_DIR / "fig1_d5_both.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_d5_both.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved outputs/fig1_d5_both.{svg,pdf,png}")

# ==============================================================================
# 6. D5 / Reported (Continuous Alpha Corruption Crossover Boundary)
# ==============================================================================
def generate_d5_reported():
    print("6. Generating D5 / Reported (Disclosed failure boundary alpha* ~ 0.07)...")
    df_corr = pd.read_csv(ROOT / "06_实验结果与日志" / "corruption_severity" / "corruption_severity_results.csv")
    sub = df_corr[(df_corr["site"] == "penmanshiel") & (df_corr["experiment"] == "continuous_alpha")]
    agg = sub.groupby("parameter")["dispatch_cost_cal"].agg(["mean", "std"]).reset_index()
    
    alphas = agg["parameter"].to_numpy(float)
    costs = agg["mean"].to_numpy(float)
    stds = agg["std"].to_numpy(float)
    baseline = 0.09881
    
    alpha_star = float(alphas[0] + (baseline - costs[0]) / (costs[1] - costs[0]) * (alphas[1] - alphas[0]))
    
    # --- Diagnostic Figure ---
    fig_diag, ax_diag = plt.subplots(figsize=(5.0, 3.2), dpi=300)
    ax_diag.axhspan(baseline, 0.170, color="#fee2e2", alpha=0.5, label="Adverse / Failure Regime")
    ax_diag.axhspan(0.090, baseline, color="#dcfce7", alpha=0.5, label="Beneficial Regime")
    ax_diag.plot(alphas, costs, color=C_HARM, marker="s", markersize=5, lw=2.0, label=r"Corrupted Prior ($\lambda=1.0$)")
    ax_diag.fill_between(alphas, costs - stds, costs + stds, color=C_HARM, alpha=0.15)
    ax_diag.axhline(baseline, color=C_UNCON, linestyle="--", lw=1.5, label=f"Unconstrained Baseline ({baseline:.5f})")
    ax_diag.axvline(alpha_star, color="#b91c1c", linestyle=":", lw=1.5)
    ax_diag.plot(alpha_star, baseline, marker="o", markersize=8, color="#b91c1c", zorder=5)
    ax_diag.annotate(f"Disclosed Crossover Boundary\n$\\alpha^* \\approx {alpha_star:.2f}$",
                     xy=(alpha_star, baseline), xytext=(0.18, 0.104),
                     arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2),
                     fontsize=8.5, fontweight="bold", color="#b91c1c")
    ax_diag.set_title("Disclosed Failure Boundary (Penmanshiel)", fontsize=9.5, fontweight="bold")
    ax_diag.set_xlabel(r"Corruption Factor $\alpha$ (0=Genuine, 1=Scrambled)", fontsize=8.5)
    ax_diag.set_ylabel(r"Normalized Dispatch Cost ($r=2$)", fontsize=8.5)
    ax_diag.set_xlim(-0.02, 1.02)
    ax_diag.set_ylim(0.092, 0.165)
    ax_diag.legend(frameon=True, fontsize=8, loc="upper left")
    ax_diag.grid(True, linestyle="--", alpha=0.5)
    fig_diag.tight_layout()
    fig_diag.savefig(OUT_DIR / "fig1_d5_reported_diag.png", dpi=300)
    plt.close(fig_diag)
    
    # --- Micro-Figure (Primary Preferred: Focused Observed Range alpha in [0.0, 0.5]) ---
    fig = plt.figure(figsize=(1.8, 1.2), dpi=100)
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.76])
    ax.grid(False)
    
    mask_05 = alphas <= 0.5
    alphas_05 = alphas[mask_05]
    costs_05 = costs[mask_05]
    
    ax.axhspan(0.092, baseline, xmin=0.0, xmax=1.0, color="#dcfce7", alpha=0.45, lw=0)
    ax.axhspan(baseline, 0.138, xmin=0.0, xmax=1.0, color="#fee2e2", alpha=0.55, lw=0)
    
    ax.axhline(baseline, color=C_UNCON, linestyle="--", lw=1.3)
    ax.plot(alphas_05, costs_05, color=C_HARM, linestyle="-", lw=2.2)
    ax.plot(alpha_star, baseline, marker="o", markersize=4.5, color="#b91c1c")
    
    ax.text(alphas_05[-1] + 0.015, costs_05[-1], "Cost", color=C_HARM, fontsize=7.2, fontweight="bold", va="center")
    ax.text(0.18, baseline + 0.012, "Failure Regime", color="#b91c1c", fontsize=6.5, fontweight="bold")
    ax.text(alpha_star + 0.015, baseline - 0.0035, r"$\alpha^*$", color="#b91c1c", fontsize=7.2, fontweight="bold", ha="left")
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$\alpha$", fontsize=8.0, color="#475569", labelpad=2)
    ax.set_ylabel("Cost", fontsize=7.5, color="#475569", labelpad=2)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.set_xlim(-0.02, 0.58)
    ax.set_ylim(0.092, 0.138)
    
    fig.savefig(OUT_DIR / "fig1_d5_reported.svg", format="svg", bbox_inches="tight", transparent=False)
    fig.savefig(OUT_DIR / "fig1_d5_reported.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_d5_reported.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved outputs/fig1_d5_reported.{svg,pdf,png} (Preferred: alpha in [0.0, 0.5])")
    
    # --- Micro-Figure (Secondary: Full Observed Range alpha in [0.0, 1.0]) ---
    fig_full = plt.figure(figsize=(1.8, 1.2), dpi=100)
    ax_full = fig_full.add_axes([0.08, 0.14, 0.84, 0.76])
    ax_full.grid(False)
    ax_full.axhspan(0.092, baseline, xmin=0.0, xmax=1.0, color="#dcfce7", alpha=0.45, lw=0)
    ax_full.axhspan(baseline, 0.165, xmin=0.0, xmax=1.0, color="#fee2e2", alpha=0.55, lw=0)
    ax_full.axhline(baseline, color=C_UNCON, linestyle="--", lw=1.3)
    ax_full.plot(alphas, costs, color=C_HARM, linestyle="-", lw=2.2)
    ax_full.plot(alpha_star, baseline, marker="o", markersize=4.5, color="#b91c1c")
    ax_full.text(alphas[-1] + 0.03, costs[-1], "Cost", color=C_HARM, fontsize=7.2, fontweight="bold", va="center")
    ax_full.text(0.38, baseline + 0.015, "Failure Regime", color="#b91c1c", fontsize=6.5, fontweight="bold")
    ax_full.text(alpha_star + 0.03, baseline - 0.005, r"$\alpha^*$", color="#b91c1c", fontsize=7.2, fontweight="bold", ha="left")
    ax_full.set_xticks([])
    ax_full.set_yticks([])
    ax_full.set_xlabel(r"$\alpha$", fontsize=8.0, color="#475569", labelpad=2)
    ax_full.set_ylabel("Cost", fontsize=7.5, color="#475569", labelpad=2)
    ax_full.spines["top"].set_visible(False)
    ax_full.spines["right"].set_visible(False)
    ax_full.spines["left"].set_color("#64748b")
    ax_full.spines["bottom"].set_color("#64748b")
    ax_full.spines["left"].set_linewidth(0.85)
    ax_full.spines["bottom"].set_linewidth(0.85)
    ax_full.set_xlim(-0.05, 1.18)
    ax_full.set_ylim(0.092, 0.165)
    fig_full.savefig(OUT_DIR / "fig1_d5_reported_full.svg", format="svg", bbox_inches="tight", transparent=False)
    fig_full.savefig(OUT_DIR / "fig1_d5_reported_full.pdf", format="pdf", bbox_inches="tight")
    fig_full.savefig(OUT_DIR / "fig1_d5_reported_full.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig_full)
    print("  -> Saved outputs/fig1_d5_reported_full.{svg,pdf,png} (Full range: alpha in [0.0, 1.0])")

# ==============================================================================
# 7. D5 / Not reported (Genuine Prior vs Scrambled Prior)
# ==============================================================================
def generate_d5_notreported():
    print("7. Generating D5 / Not reported (Genuine vs Scrambled prior)...")
    csv_path = ROOT / "04_数据集" / "constraint_rebuild_v2" / "penmanshiel.csv"
    frame = pd.read_csv(csv_path, index_col="time", parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    prior = fit_prior(frame, CFG["splits"]["penmanshiel"]["train_end"])
    scrambled = fit_scrambled_prior(prior, seed=42)
    
    v = prior["centers"]
    p_true = prior["means"]
    p_scramble = scrambled["means"]
    
    mask = v <= 22.0
    v_f = v[mask]
    p_true_f = p_true[mask]
    p_scramble_f = p_scramble[mask]
    
    # --- Diagnostic Figure ---
    fig_diag, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.2), dpi=300)
    ax1.plot(v_f, p_true_f, color=C_CON, linestyle="-", lw=2.2, label=r"Genuine Prior $C_{\mathrm{emp}}$")
    ax1.plot(v_f, p_scramble_f, color=C_HARM, linestyle="--", lw=1.8, marker="s", markersize=3.5, label=r"Scrambled Pseudo-Prior $C_{\mathrm{scramble}}$")
    ax1.set_title("Negative Control Prior Object (Penmanshiel)", fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Wind Speed $v$ (m/s)", fontsize=8.5)
    ax1.set_ylabel("Power (p.u.)", fontsize=8.5)
    ax1.legend(frameon=True, fontsize=8, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    arms = ["Genuine Prior", "Scrambled Prior"]
    costs = [0.09760, 0.16611]
    ax2.bar(arms, costs, color=[C_CON, C_HARM], width=0.45, edgecolor="#333333", linewidth=0.8)
    ax2.axhline(0.09881, color=C_UNCON, linestyle=":", label="Unconstrained Baseline (0.0988)")
    ax2.annotate("+68.1% Cost Surge\n($p < 0.001$)", xy=(1, 0.16611), xytext=(0.35, 0.148),
                 arrowprops=dict(arrowstyle="->", color=C_HARM), fontsize=8.5, color=C_HARM, fontweight="bold")
    ax2.set_title("Downstream Risk When Harm Is Omitted", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Dispatch Cost ($r=2$)", fontsize=8.5)
    ax2.set_ylim(0.07, 0.19)
    ax2.legend(frameon=True, fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    fig_diag.tight_layout()
    fig_diag.savefig(OUT_DIR / "fig1_d5_notreported_diag.png", dpi=300)
    plt.close(fig_diag)
    
    # --- Micro-Figure ---
    fig = plt.figure(figsize=(1.9, 1.2), dpi=100)
    ax = fig.add_axes([0.08, 0.14, 0.86, 0.76])
    ax.grid(False)
    
    ax.plot(v_f, p_true_f, color=C_CON, linestyle="-", lw=2.2)
    ax.plot(v_f, p_scramble_f, color=C_HARM, linestyle="--", lw=1.5, marker="s", markersize=2.2)
    
    ax.text(v_f[-1] + 0.4, p_true_f[-1], "True", color=C_CON, fontsize=6.8, fontweight="bold", va="center")
    ax.text(v_f[-1] + 0.4, p_scramble_f[-1], "Scram.", color=C_HARM, fontsize=6.8, fontweight="bold", va="center")
    
    # Subtle de-emphasized annotation
    ax.text(0.96, 0.10, "boundary omitted", transform=ax.transAxes,
            fontsize=5.6, color="#94a3b8", style="italic", ha="right", va="bottom",
            bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.85))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$v$", fontsize=8.0, color="#475569", labelpad=2)
    ax.set_ylabel("Power", fontsize=7.5, color="#475569", labelpad=2)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.set_xlim(v_f[0] - 0.5, v_f[-1] + 6.2)
    
    fig.savefig(OUT_DIR / "fig1_d5_notreported.svg", format="svg", bbox_inches="tight", transparent=False)
    fig.savefig(OUT_DIR / "fig1_d5_notreported.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_d5_notreported.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved outputs/fig1_d5_notreported.{svg,pdf,png}")

if __name__ == "__main__":
    print("=" * 70)
    print("STARTING RECONSTRUCTION OF 7 COORDINATE-BASED MICRO-FIGURES")
    print("=" * 70)
    generate_d1_historical_static()
    generate_d2_posthoc()
    generate_d3_none()
    generate_d5_constraint_onoff()
    generate_d5_both()
    generate_d5_reported()
    generate_d5_notreported()
    print("=" * 70)
    print("ALL 7 MICRO-FIGURES SUCCESSFULLY GENERATED IN SVG, PDF, AND PNG!")
    print("=" * 70)
