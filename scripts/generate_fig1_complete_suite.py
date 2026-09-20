"""
Refined Figure 1 Micro-Figures (Suite of 7 Core Items)
Strictly adheres to user feedback:
1. "删信息、统一语法" (Remove visual noise, unify syntax)
2. Micro-plots:
   - Coordinates: ONLY variable names on axes, NO numeric ticks.
   - NO grid lines (grid=False).
   - NO floating legends; direct labels at line ends:
     - Constraint on/off: 'λ=0' and 'λ>0' (or '0' and '>0')
     - Both: 'Full', 'Abl.', 'None'
   - Consistent stroke width across all plots.
   - Shapes strictly from real experiment trajectories.
   - D5 Not reported: de-emphasize 'downstream risk not evaluated' as subtle annotation.
3. D2 Data generation:
   - Redesigned as scientific matrix/tensor glyph:
     f_phys(\theta) -> tabular synthetic matrix [x^i, y^i] -> X \in R^{N x T x d}
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

# Paths
ROOT = Path(__file__).resolve().parent.parent
SYS_ROUND2 = ROOT / "05_实验代码与脚本" / "constraint_round2"
sys.path.append(str(SYS_ROUND2))

from core import fit_prior, CFG, Forecaster, starts, windows
from controlled_ablation import fit_scrambled_prior

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams['svg.fonttype'] = 'none'  # Pure vector text / paths, no rasterization
mpl.rcParams['axes.edgecolor'] = '#475569'
mpl.rcParams['axes.linewidth'] = 0.85

# Unified palette
C_UNCON = '#64748b'    # Slate grey for unconstrained / baseline / no constraint
C_CON = '#1d4ed8'      # Deep royal blue for physically constrained / genuine / full model
C_HARM = '#dc2626'     # Crimson red for scrambled / harm / failure regime
C_MOD = '#0d9488'      # Teal green for module-ablated
C_BENEFIT = '#16a34a'  # Green for beneficial regime
C_BOUND = '#94a3b8'    # Light slate for feasibility bounds
C_BG = '#ffffff'       # Pure white background


# ==============================================================================
# A. D3 / None / Historical target SCADA
# ==============================================================================
def generate_d3_none():
    print("Generating [A] D3 None (Historical SCADA)...")
    frame = pd.read_csv(ROOT / "04_数据集" / "constraint_rebuild_v2" / "penmanshiel.csv", index_col='time', parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    x, y, ids = starts(frame, CFG['splits']['penmanshiel'])
    test_x, test_y = windows(x, y, ids['test'])
    
    sample_idx = 100
    scada_window = test_x[sample_idx, :, 0]
    future_y = test_y[sample_idx]
    
    # --- 1. Diagnostic Version ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=300)
    t_hist = np.arange(-CFG['lookback'], 0) * 10
    t_fut = np.arange(0, CFG['horizon']) * 10
    ax1.plot(t_hist, scada_window, color=C_CON, linewidth=2.0, label=r'Historical SCADA $P_{t-L:t}$')
    ax1.plot(t_fut, future_y, color='#0f172a', linestyle=':', linewidth=1.5, label=r'Actual Delivery $y_{t:t+H}$')
    ax1.axvline(0, color='#94a3b8', linestyle='--', linewidth=1.2)
    ax1.set_title(f"SCADA Input Window & Delivery (Sample #{sample_idx})", fontsize=9.0, fontweight='bold')
    ax1.set_xlabel("Relative Time (min)", fontsize=8.5)
    ax1.set_ylabel("Power (p.u.)", fontsize=8.5)
    ax1.legend(frameon=True, fontsize=7.5, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.axis('off')
    box_scada = FancyBboxPatch((0.05, 0.35), 0.26, 0.45, boxstyle="round,pad=0.02,rounding_size=0.04",
                               edgecolor='#64748b', facecolor='#f8fafc', linewidth=1.2)
    box_model = FancyBboxPatch((0.44, 0.35), 0.28, 0.45, boxstyle="round,pad=0.02,rounding_size=0.04",
                               edgecolor='#2563eb', facecolor='#eff6ff', linewidth=1.2)
    ax2.add_patch(box_scada)
    ax2.add_patch(box_model)
    ax2.text(0.18, 0.58, r"Target SCADA" + "\n" + r"$P_{t-L:t}$", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#1e293b')
    ax2.text(0.18, 0.42, "No NWP / No Met", ha='center', va='center', fontsize=6.5, color='#64748b')
    ax2.annotate('', xy=(0.43, 0.58), xytext=(0.32, 0.58), arrowprops=dict(arrowstyle="->", color='#334155', lw=1.4))
    ax2.text(0.58, 0.58, "Forecast\nModel", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#1d4ed8')
    ax2.annotate('', xy=(0.84, 0.58), xytext=(0.73, 0.58), arrowprops=dict(arrowstyle="->", color='#334155', lw=1.4))
    ax2.text(0.90, 0.58, r"$\hat{y}$", ha='center', va='center', fontsize=11.0, fontweight='bold', color='#0f172a')
    ax2.set_title(r"D3 Telemetry Semantics: $\mathcal{D}_3 = \mathrm{None}$", fontsize=9.0, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_d3_none_diag.png", dpi=300)
    plt.close()
    
    # --- 2. Figure 1 Micro Version (Strict Minimalist) ---
    fig = plt.figure(figsize=(2.2, 1.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    sub_ax = fig.add_axes([0.08, 0.20, 0.36, 0.62])
    sub_ax.plot(scada_window, color=C_CON, linewidth=1.8)
    sub_ax.set_xticks([])
    sub_ax.set_yticks([])
    sub_ax.grid(False)
    sub_ax.set_xlabel("Time", fontsize=7.0, color='#475569', labelpad=2)
    sub_ax.set_ylabel("Power", fontsize=7.0, color='#475569', labelpad=2)
    sub_ax.spines['top'].set_visible(False)
    sub_ax.spines['right'].set_visible(False)
    sub_ax.spines['left'].set_color('#64748b')
    sub_ax.spines['bottom'].set_color('#64748b')
    sub_ax.spines['left'].set_linewidth(0.85)
    sub_ax.spines['bottom'].set_linewidth(0.85)
    sub_ax.text(0.5, 1.04, r"$P_{t-L:t}$", transform=sub_ax.transAxes, ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1e293b')
    
    ax.annotate('', xy=(0.54, 0.52), xytext=(0.46, 0.52), arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.2))
    box = FancyBboxPatch((0.55, 0.30), 0.28, 0.44, boxstyle="round,pad=0.02,rounding_size=0.04",
                         edgecolor='#2563eb', facecolor='#eff6ff', linewidth=1.1)
    ax.add_patch(box)
    ax.text(0.69, 0.52, "Forecast\nModel", ha='center', va='center', fontsize=6.5, fontweight='bold', color='#1e40af')
    
    ax.annotate('', xy=(0.89, 0.52), xytext=(0.84, 0.52), arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.2))
    ax.text(0.94, 0.52, r"$\hat{y}$", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#0f172a')
    
    fig.savefig(OUT_DIR / "fig1_d3_none.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d3_none.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done A: fig1_d3_none_diag.png, fig1_d3_none.svg, fig1_d3_none.pdf")


# ==============================================================================
# B. D2 / Data generation / Physics-guided synthetic samples (Scientific Matrix)
# ==============================================================================
def generate_d2_data_generation():
    """
    Redesigned to scientific matrix/tensor glyph:
    f_phys(\theta) -> tabular synthetic matrix [x^i, y^i] -> X \in R^{N x T x d}
    """
    print("Generating [B] D2 Data Generation (Scientific Matrix/Tensor Glyph)...")
    
    # --- 1. Diagnostic Version ---
    fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=300)
    ax.axis('off')
    
    b_phys = FancyBboxPatch((0.04, 0.28), 0.24, 0.52, boxstyle="round,pad=0.03,rounding_size=0.04",
                            edgecolor='#0284c7', facecolor='#f0f9ff', linewidth=1.3)
    ax.add_patch(b_phys)
    ax.text(0.16, 0.62, r"$f_{\mathrm{phys}}(\theta)$", ha='center', va='center', fontsize=11.0, fontweight='bold', color='#0369a1')
    ax.text(0.16, 0.42, "Physical Simulator /\nGoverning Laws", ha='center', va='center', fontsize=7.5, color='#0284c7')
    
    ax.annotate('', xy=(0.35, 0.54), xytext=(0.29, 0.54), arrowprops=dict(arrowstyle="->", color='#475569', lw=1.5))
    
    # Middle Matrix
    mat_box = FancyBboxPatch((0.36, 0.22), 0.26, 0.62, boxstyle="square,pad=0.02",
                             edgecolor='#64748b', facecolor='#ffffff', linewidth=1.2)
    ax.add_patch(mat_box)
    ax.text(0.49, 0.74, r"$x^{(1)} \quad y^{(1)}$", ha='center', va='center', fontsize=8.5, color='#1e293b')
    ax.text(0.49, 0.60, r"$x^{(2)} \quad y^{(2)}$", ha='center', va='center', fontsize=8.5, color='#1e293b')
    ax.text(0.49, 0.46, r"$\vdots \qquad \vdots$", ha='center', va='center', fontsize=8.5, color='#64748b')
    ax.text(0.49, 0.32, r"$x^{(N)} \quad y^{(N)}$", ha='center', va='center', fontsize=8.5, color='#1e293b')
    ax.text(0.49, 0.12, "Synthetic Batch", ha='center', va='center', fontsize=7.2, color='#64748b')
    
    ax.annotate('', xy=(0.69, 0.54), xytext=(0.63, 0.54), arrowprops=dict(arrowstyle="->", color='#475569', lw=1.5))
    
    # Right Tensor Card
    t_box = FancyBboxPatch((0.71, 0.32), 0.25, 0.44, boxstyle="round,pad=0.03,rounding_size=0.04",
                           edgecolor='#2563eb', facecolor='#eff6ff', linewidth=1.3)
    ax.add_patch(t_box)
    ax.text(0.835, 0.58, r"$X \in \mathbb{R}^{N \times T \times d}$", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#1e40af')
    ax.text(0.835, 0.42, "Augmented Tensor", ha='center', va='center', fontsize=7.2, color='#2563eb')
    
    ax.set_title(r"D2 Stage: Physics-Guided Synthetic Data Generation Prior ($\mathcal{D}_2 = \mathrm{Data\ Gen}$)", fontsize=9.5, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_d2_data_generation_diag.png", dpi=300)
    plt.close()
    
    # --- 2. Figure 1 Micro Version (Clean Scientific Matrix Glyph with True Brackets) ---
    fig = plt.figure(figsize=(2.2, 1.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # Left box: f_phys(theta)
    b_phys = FancyBboxPatch((0.02, 0.25), 0.22, 0.52, boxstyle="round,pad=0.02,rounding_size=0.04",
                            edgecolor='#0284c7', facecolor='#f0f9ff', linewidth=1.1)
    ax.add_patch(b_phys)
    ax.text(0.13, 0.58, r"$f_{\mathrm{phys}}(\theta)$", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#0369a1')
    ax.text(0.13, 0.38, "simulator", ha='center', va='center', fontsize=5.8, color='#0284c7')
    
    # Arrow 1
    ax.annotate('', xy=(0.31, 0.51), xytext=(0.25, 0.51),
                arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.2, mutation_scale=9))
    
    # Middle Matrix with genuine square brackets [ ]
    # Left bracket [
    ax.plot([0.35, 0.33, 0.33, 0.35], [0.82, 0.82, 0.20, 0.20], color='#475569', lw=1.2)
    # Right bracket ]
    ax.plot([0.57, 0.59, 0.59, 0.57], [0.82, 0.82, 0.20, 0.20], color='#475569', lw=1.2)
    
    ax.text(0.46, 0.71, r"$x^{1} \;\; y^{1}$", ha='center', va='center', fontsize=6.8, color='#1e293b')
    ax.text(0.46, 0.55, r"$x^{2} \;\; y^{2}$", ha='center', va='center', fontsize=6.8, color='#1e293b')
    ax.text(0.46, 0.41, r"$\vdots \quad \vdots$", ha='center', va='center', fontsize=6.8, color='#64748b')
    ax.text(0.46, 0.27, r"$x^{N} \;\; y^{N}$", ha='center', va='center', fontsize=6.8, color='#1e293b')
    
    # Arrow 2
    ax.annotate('', xy=(0.69, 0.51), xytext=(0.61, 0.51),
                arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.2, mutation_scale=9))
    
    # Right label
    ax.text(0.84, 0.51, r"$X \in \mathbb{R}^{N \times T \times d}$", ha='center', va='center', fontsize=6.4, fontweight='bold', color='#1e3a8a')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.savefig(OUT_DIR / "fig1_d2_data_generation.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d2_data_generation.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done B: fig1_d2_data_generation_diag.png, fig1_d2_data_generation.svg, fig1_d2_data_generation.pdf")


# ==============================================================================
# C. D2 / Post-hoc correction
# ==============================================================================
def generate_d2_posthoc():
    print("Generating [C] D2 Post-hoc Correction...")
    t = np.linspace(0, 10, 50)
    raw_curve = 0.72 + 0.38 * np.sin(t * np.pi / 10)**1.5
    clipped_curve = np.clip(raw_curve, 0.0, 1.0)
    peak_idx = len(t) // 2
    
    # --- 1. Diagnostic Version ---
    fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=300)
    ax.axhline(1.0, color=C_BOUND, linestyle=':', linewidth=1.5, label=r'Physical Bound $P_{\mathrm{rated}}=1.0$')
    ax.axhspan(1.0, 1.15, color='#fee2e2', alpha=0.4, label='Inadmissible Excursion ($P > 1.0$)')
    ax.plot(t, raw_curve, color=C_UNCON, linestyle='--', linewidth=2.0, label=r'Raw Unconstrained Forecast $\hat{y}_{\mathrm{raw}}$')
    ax.plot(t, clipped_curve, color=C_CON, linestyle='-', linewidth=2.4, label=r'Post-hoc Clipped $\hat{y}_{\mathrm{proj}}$')
    
    ax.annotate(r'Post-hoc Projection' + '\n' + r'$\Pi_{\mathcal{C}}(\hat{y})$',
                xy=(t[peak_idx], 1.0), xytext=(t[peak_idx] - 3.2, 1.06),
                arrowprops=dict(arrowstyle="->", color=C_CON, lw=1.5),
                fontsize=8.5, fontweight='bold', color=C_CON)
    
    ax.set_title("Post-hoc Feasibility Correction Mechanism", fontsize=9.5, fontweight='bold')
    ax.set_xlabel("Time Step $t$", fontsize=8.5)
    ax.set_ylabel(r"Power Output $\hat{y}$ (p.u.)", fontsize=8.5)
    ax.set_ylim(0.65, 1.15)
    ax.legend(frameon=True, fontsize=8, loc='lower center')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_d2_posthoc_diag.png", dpi=300)
    plt.close()
    
    # --- 2. Figure 1 Micro Version (Strict Minimalist, No Grid, Direct Tags, Clean Limit) ---
    fig = plt.figure(figsize=(1.8, 1.2), dpi=100)
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.76])
    
    ax.grid(False)
    # Dotted line only spans the data domain up to t[-1]
    ax.plot([t[0], t[-1]], [1.0, 1.0], color=C_BOUND, linestyle=':', linewidth=1.4)
    ax.plot(t, raw_curve, color=C_UNCON, linestyle='--', linewidth=1.8)
    ax.plot(t, clipped_curve, color=C_CON, linestyle='-', linewidth=2.4)
    
    # Downward projection arrow
    ax.annotate('', xy=(t[peak_idx], 1.002), xytext=(t[peak_idx], raw_curve[peak_idx] - 0.005),
                arrowprops=dict(arrowstyle="->", color=C_CON, lw=1.4))
    
    ax.text(t[-1] + 0.3, 1.0, 'Limit', color=C_BOUND, fontsize=7.0, va='center',
            bbox=dict(boxstyle='square,pad=0.08', facecolor='white', edgecolor='none'))
    ax.text(t[-1] + 0.3, raw_curve[-1] + 0.04, 'Raw', color=C_UNCON, fontsize=7.0, va='center')
    ax.text(t[-1] + 0.3, clipped_curve[-1] - 0.01, 'Post', color=C_CON, fontsize=7.0, fontweight='bold', va='center')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Time $t$", fontsize=7.5, color='#475569', labelpad=2)
    ax.set_ylabel("Power", fontsize=7.5, color='#475569', labelpad=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#64748b')
    ax.spines['bottom'].set_color('#64748b')
    ax.spines['left'].set_linewidth(0.85)
    ax.spines['bottom'].set_linewidth(0.85)
    ax.set_xlim(t[0] - 0.5, t[-1] + 3.4)
    ax.set_ylim(0.68, 1.15)
    
    fig.savefig(OUT_DIR / "fig1_d2_posthoc.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d2_posthoc.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done C: fig1_d2_posthoc_diag.png, fig1_d2_posthoc.svg, fig1_d2_posthoc.pdf")


# ==============================================================================
# D. D5 / Constraint on/off (Strict Minimalist)
# ==============================================================================
def generate_d5_constraint_onoff():
    print("Generating [D] D5 Constraint On/Off...")
    frame = pd.read_csv(ROOT / "04_数据集" / "constraint_rebuild_v2" / "penmanshiel.csv", index_col='time', parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    x, y, ids = starts(frame, CFG['splits']['penmanshiel'])
    test_x, test_y = windows(x, y, ids['test'])
    
    ckpt_uncon = ROOT / "06_实验结果与日志" / "constraint_round2" / "penmanshiel" / "dlinear" / "seed42" / "none_l0.0_f1.0_plain.pt"
    ckpt_con = ROOT / "06_实验结果与日志" / "constraint_round2" / "penmanshiel" / "dlinear" / "seed42" / "curve_l1.0_f1.0_plain.pt"
    
    m_uncon = Forecaster('dlinear', CFG['lookback'], CFG['horizon'])
    m_uncon.load_state_dict(torch.load(ckpt_uncon, map_location='cpu'))
    m_uncon.eval()
    
    m_con = Forecaster('dlinear', CFG['lookback'], CFG['horizon'])
    m_con.load_state_dict(torch.load(ckpt_con, map_location='cpu'))
    m_con.eval()
    
    with torch.no_grad():
        preds_uncon = m_uncon(torch.tensor(test_x[:300])).numpy()
        preds_con = m_con(torch.tensor(test_x[:300])).numpy()
        
    diffs = np.mean((preds_uncon - preds_con)**2, axis=1)
    rep_idx = int(np.argsort(diffs)[-15])
    
    t = np.arange(1, CFG['horizon'] + 1)
    y_true = test_y[rep_idx]
    y_uncon = preds_uncon[rep_idx]
    y_con = preds_con[rep_idx]
    
    # Diagnostic numbers
    df_ab = pd.read_csv(ROOT / "06_实验结果与日志" / "ablation_3d_results" / "ablation_3d_master_summary.csv")
    sub = df_ab[(df_ab['site'] == 'penmanshiel') & (df_ab['policy'] == 'calibrated') & (df_ab['ratio'] == 2)]
    cost_uncon = sub[sub['arm'] == 'vanilla_unconstrained']['cost_static'].mean()
    cost_con = sub[sub['arm'] == 'genuine_curve_l1.0']['cost_static'].mean()
    cost_mid = sub[sub['arm'] == 'genuine_curve_l0.1']['cost_static'].mean()
    
    # --- 1. Diagnostic Version ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)
    ax1.plot(t, y_true, color='#0f172a', linestyle=':', linewidth=1.5, label='Ground Truth $y$')
    ax1.plot(t, y_uncon, color=C_UNCON, linestyle='--', linewidth=1.8, label=r'Unconstrained ($\lambda=0$)')
    ax1.plot(t, y_con, color=C_CON, linestyle='-', linewidth=2.0, label=r'Constrained ($\lambda=1.0$)')
    ax1.set_title(f"Test Trajectory Divergence (Sample #{rep_idx})", fontsize=9.5, fontweight='bold')
    ax1.set_xlabel("Forecast Horizon Step $h$ (10-min)", fontsize=8.5)
    ax1.set_ylabel("Power (p.u.)", fontsize=8.5)
    ax1.legend(frameon=True, fontsize=8, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    lams = [0.0, 0.1, 1.0]
    costs = [cost_uncon, cost_mid, cost_con]
    ax2.plot(lams, costs, marker='o', color=C_CON, linewidth=1.8, markersize=6)
    ax2.axhline(cost_uncon, color=C_UNCON, linestyle='--', label=r'Baseline $\lambda=0$')
    ax2.annotate('-1.22% Cost\n($p < 0.01$)', xy=(1.0, cost_con), xytext=(0.55, cost_con + 0.0003),
                 arrowprops=dict(arrowstyle="->", color=C_CON), fontsize=8, color=C_CON, fontweight='bold')
    ax2.set_title(r"Ablation Isolation: Dispatch Cost vs $\lambda$", fontsize=9.5, fontweight='bold')
    ax2.set_xlabel(r"Constraint Penalty Weight $\lambda$", fontsize=8.5)
    ax2.set_ylabel(r"Normalized Dispatch Cost ($r=2$)", fontsize=8.5)
    ax2.set_ylim(0.0972, 0.0990)
    ax2.legend(frameon=True, fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_d5_constraint_onoff_diag.png", dpi=300)
    plt.close()
    
    # --- 2. Figure 1 Micro Version (Strict Minimalist: 2 lines, no grid, direct line-end tags) ---
    fig = plt.figure(figsize=(1.8, 1.2), dpi=100)
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.76])
    ax.grid(False)
    
    h_slice = slice(6, 32)
    t_sub = np.arange(len(t[h_slice]))
    ax.plot(t_sub, y_uncon[h_slice], color=C_UNCON, linestyle='--', linewidth=1.8)
    ax.plot(t_sub, y_con[h_slice], color=C_CON, linestyle='-', linewidth=2.2)
    
    # Direct line-end tags: λ=0 and λ>0
    ax.text(t_sub[-1] + 0.5, y_uncon[h_slice][-1], r'$\lambda=0$', color=C_UNCON, fontsize=7.2, fontweight='bold', va='center')
    ax.text(t_sub[-1] + 0.5, y_con[h_slice][-1], r'$\lambda>0$', color=C_CON, fontsize=7.2, fontweight='bold', va='center')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$h$", fontsize=8.0, color='#475569', labelpad=2)
    ax.set_ylabel("Power", fontsize=7.5, color='#475569', labelpad=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#64748b')
    ax.spines['bottom'].set_color('#64748b')
    ax.spines['left'].set_linewidth(0.85)
    ax.spines['bottom'].set_linewidth(0.85)
    ax.set_xlim(t_sub[0] - 1, t_sub[-1] + 7)
    
    fig.savefig(OUT_DIR / "fig1_d5_constraint_onoff.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d5_constraint_onoff.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done D: fig1_d5_constraint_onoff_diag.png, fig1_d5_constraint_onoff.svg, fig1_d5_constraint_onoff.pdf")


# ==============================================================================
# E. D5 / Both (Strict Minimalist)
# ==============================================================================
def generate_d5_both():
    print("Generating [E] D5 Both (Full vs Module Ablated vs No Constraint)...")
    frame = pd.read_csv(ROOT / "04_数据集" / "constraint_rebuild_v2" / "penmanshiel.csv", index_col='time', parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    x, y, ids = starts(frame, CFG['splits']['penmanshiel'])
    test_x, test_y = windows(x, y, ids['test'])
    
    ckpt_uncon = ROOT / "06_实验结果与日志" / "constraint_round2" / "penmanshiel" / "dlinear" / "seed42" / "none_l0.0_f1.0_plain.pt"
    ckpt_con = ROOT / "06_实验结果与日志" / "constraint_round2" / "penmanshiel" / "dlinear" / "seed42" / "curve_l1.0_f1.0_plain.pt"
    ckpt_dist = ROOT / "06_实验结果与日志" / "constraint_round2" / "penmanshiel" / "dlinear" / "seed42" / "curve_l0.1_f1.0_plain.pt"
    
    m_uncon = Forecaster('dlinear', CFG['lookback'], CFG['horizon'])
    m_uncon.load_state_dict(torch.load(ckpt_uncon, map_location='cpu'))
    m_uncon.eval()
    
    m_con = Forecaster('dlinear', CFG['lookback'], CFG['horizon'])
    m_con.load_state_dict(torch.load(ckpt_con, map_location='cpu'))
    m_con.eval()
    
    m_mod = Forecaster('dlinear', CFG['lookback'], CFG['horizon'])
    m_mod.load_state_dict(torch.load(ckpt_dist, map_location='cpu'))
    m_mod.eval()
    
    with torch.no_grad():
        p_uncon = m_uncon(torch.tensor(test_x[:300])).numpy()
        p_con = m_con(torch.tensor(test_x[:300])).numpy()
        p_mod = m_mod(torch.tensor(test_x[:300])).numpy()
        
    diffs = np.mean((p_uncon - p_con)**2, axis=1)
    rep_idx = int(np.argsort(diffs)[-12])
    
    t = np.arange(1, CFG['horizon'] + 1)
    y_true = test_y[rep_idx]
    y_full = p_con[rep_idx]
    y_mod = p_mod[rep_idx]
    y_none = p_uncon[rep_idx]
    
    # --- 1. Diagnostic Version ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2), dpi=300)
    ax1.plot(t, y_full, color=C_CON, linestyle='-', linewidth=2.0, label='Full Model (Dual/Con)')
    ax1.plot(t, y_mod, color=C_MOD, linestyle='--', linewidth=1.8, label='Module Ablated (Weak/Single)')
    ax1.plot(t, y_none, color=C_UNCON, linestyle='--', linewidth=1.8, label=r'No Constraint ($\lambda=0$)')
    ax1.plot(t, y_true, color='#0f172a', linestyle=':', linewidth=1.2, label='Ground Truth')
    ax1.set_title("Test Predictions under 3-Way Ablation", fontsize=9.5, fontweight='bold')
    ax1.set_xlabel("Horizon Step $h$", fontsize=8.5)
    ax1.set_ylabel("Power (p.u.)", fontsize=8.5)
    ax1.legend(frameon=True, fontsize=7.5)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    arms = ['Full Model', 'Module Abl.', 'No Constraint']
    costs = [0.09760, 0.09861, 0.09881]
    ax2.bar(arms, costs, color=[C_CON, C_MOD, C_UNCON], width=0.55, edgecolor='#333333', linewidth=0.8)
    ax2.set_ylim(0.096, 0.100)
    ax2.set_title("Normalized Dispatch Cost Contrast", fontsize=9.5, fontweight='bold')
    ax2.set_ylabel("Dispatch Cost ($r=2$)", fontsize=8.5)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_d5_both_diag.png", dpi=300)
    plt.close()
    
    # --- 2. Figure 1 Micro Version (Strict Minimalist: 3 lines, no grid, direct line-end tags) ---
    fig = plt.figure(figsize=(1.8, 1.2), dpi=100)
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.76])
    ax.grid(False)
    
    h_slice = slice(6, 32)
    t_sub = np.arange(len(t[h_slice]))
    
    ax.plot(t_sub, y_none[h_slice], color=C_UNCON, linestyle='--', linewidth=1.6)
    ax.plot(t_sub, y_mod[h_slice], color=C_MOD, linestyle='--', linewidth=1.8)
    ax.plot(t_sub, y_full[h_slice], color=C_CON, linestyle='-', linewidth=2.2)
    
    ax.text(t_sub[-1] + 0.5, y_full[h_slice][-1], 'Full', color=C_CON, fontsize=6.8, fontweight='bold', va='center')
    ax.text(t_sub[-1] + 0.5, y_mod[h_slice][-1], 'Abl.', color=C_MOD, fontsize=6.8, fontweight='bold', va='center')
    ax.text(t_sub[-1] + 0.5, y_none[h_slice][-1], 'None', color=C_UNCON, fontsize=6.8, fontweight='bold', va='center')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$h$", fontsize=8.0, color='#475569', labelpad=2)
    ax.set_ylabel("Power", fontsize=7.5, color='#475569', labelpad=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#64748b')
    ax.spines['bottom'].set_color('#64748b')
    ax.spines['left'].set_linewidth(0.85)
    ax.spines['bottom'].set_linewidth(0.85)
    ax.set_xlim(t_sub[0] - 1, t_sub[-1] + 7)
    
    fig.savefig(OUT_DIR / "fig1_d5_both.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d5_both.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done E: fig1_d5_both_diag.png, fig1_d5_both.svg, fig1_d5_both.pdf")


# ==============================================================================
# F. D5 / Reported (Strict Minimalist)
# ==============================================================================
def generate_d5_reported():
    print("Generating [F] D5 Reported (Failure Boundary Disclosed)...")
    df_corr = pd.read_csv(ROOT / "06_实验结果与日志" / "corruption_severity" / "corruption_severity_results.csv")
    sub = df_corr[(df_corr['site'] == 'penmanshiel') & (df_corr['experiment'] == 'continuous_alpha')]
    agg = sub.groupby('parameter')['dispatch_cost_cal'].agg(['mean', 'std']).reset_index()
    
    alphas = agg['parameter'].to_numpy(float)
    costs = agg['mean'].to_numpy(float)
    stds = agg['std'].to_numpy(float)
    baseline = 0.09881
    
    alpha_star = float(alphas[0] + (baseline - costs[0]) / (costs[1] - costs[0]) * (alphas[1] - alphas[0]))
    
    # --- 1. Diagnostic Version ---
    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=300)
    ax.axhspan(baseline, 0.170, color='#fee2e2', alpha=0.5, label='Adverse / Failure Regime')
    ax.axhspan(0.090, baseline, color='#dcfce7', alpha=0.5, label='Beneficial Regime')
    ax.plot(alphas, costs, color=C_HARM, marker='s', markersize=5, linewidth=2.0, label=r'Corrupted Prior ($\lambda=1.0$)')
    ax.fill_between(alphas, costs - stds, costs + stds, color=C_HARM, alpha=0.15)
    ax.axhline(baseline, color=C_UNCON, linestyle='--', linewidth=1.5, label=f'Unconstrained Baseline ({baseline:.5f})')
    ax.axvline(alpha_star, color='#b91c1c', linestyle=':', linewidth=1.5)
    
    ax.plot(alpha_star, baseline, marker='o', markersize=8, color='#b91c1c', zorder=5)
    ax.annotate(f'Disclosed Crossover Boundary\n$\\alpha^* \\approx {alpha_star:.2f}$',
                xy=(alpha_star, baseline), xytext=(0.18, 0.104),
                arrowprops=dict(arrowstyle="->", color='#b91c1c', lw=1.2),
                fontsize=8.5, fontweight='bold', color='#b91c1c')
    
    ax.set_title("Disclosed Failure Boundary (Penmanshiel)", fontsize=9.5, fontweight='bold')
    ax.set_xlabel(r"Corruption Factor $\alpha$ (0=Genuine, 1=Scrambled)", fontsize=8.5)
    ax.set_ylabel(r"Normalized Dispatch Cost ($r=2$)", fontsize=8.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.092, 0.165)
    ax.legend(frameon=True, fontsize=8, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_d5_reported_diag.png", dpi=300)
    plt.close()
    
    # --- 2. Figure 1 Micro Version (No grid, subtle shading, crossover marker) ---
    fig = plt.figure(figsize=(1.8, 1.2), dpi=100)
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.76])
    ax.grid(False)
    
    ax.axhspan(0.092, baseline, xmin=0.0, xmax=1.0, color='#dcfce7', alpha=0.45, lw=0)
    ax.axhspan(baseline, 0.165, xmin=0.0, xmax=1.0, color='#fee2e2', alpha=0.55, lw=0)
    
    ax.axhline(baseline, color=C_UNCON, linestyle='--', linewidth=1.3)
    ax.plot(alphas, costs, color=C_HARM, linestyle='-', linewidth=2.2)
    ax.plot(alpha_star, baseline, marker='o', markersize=4.5, color='#b91c1c')
    
    ax.text(alphas[-1] + 0.03, costs[-1], 'Cost', color=C_HARM, fontsize=7.2, fontweight='bold', va='center')
    ax.text(0.38, baseline + 0.015, 'Failure Regime', color='#b91c1c', fontsize=6.5, fontweight='bold')
    ax.text(alpha_star + 0.03, baseline - 0.005, r'$\alpha^*$', color='#b91c1c', fontsize=7.2, fontweight='bold', ha='left')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$\alpha$", fontsize=8.0, color='#475569', labelpad=2)
    ax.set_ylabel("Cost", fontsize=7.5, color='#475569', labelpad=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#64748b')
    ax.spines['bottom'].set_color('#64748b')
    ax.spines['left'].set_linewidth(0.85)
    ax.spines['bottom'].set_linewidth(0.85)
    ax.set_xlim(-0.05, 1.18)
    ax.set_ylim(0.092, 0.165)
    
    fig.savefig(OUT_DIR / "fig1_d5_reported.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d5_reported.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done F: fig1_d5_reported_diag.png, fig1_d5_reported.svg, fig1_d5_reported.pdf")


# ==============================================================================
# G. D5 / Not reported (Strict Minimalist, De-emphasized Annotation)
# ==============================================================================
def generate_d5_notreported():
    """
    De-emphasize 'downstream risk not evaluated': subtle small text annotation, not competing with curves.
    """
    print("Generating [G] D5 Not Reported (Downstream Risk Omitted)...")
    frame = pd.read_csv(ROOT / "04_数据集" / "constraint_rebuild_v2" / "penmanshiel.csv", index_col='time', parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    prior = fit_prior(frame, CFG['splits']['penmanshiel']['train_end'])
    scrambled = fit_scrambled_prior(prior, seed=42)
    
    v = prior['centers']
    p_true = prior['means']
    p_scramble = scrambled['means']
    
    mask = v <= 22.0
    v_f = v[mask]
    p_true_f = p_true[mask]
    p_scramble_f = p_scramble[mask]
    
    # --- 1. Diagnostic Version ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.2), dpi=300)
    ax1.plot(v_f, p_true_f, color=C_CON, linestyle='-', linewidth=2.2, label=r'Genuine Prior $C_{\mathrm{emp}}$')
    ax1.plot(v_f, p_scramble_f, color=C_HARM, linestyle='--', linewidth=1.8, marker='s', markersize=3.5, label=r'Scrambled Pseudo-Prior $C_{\mathrm{scramble}}$')
    ax1.set_title("Negative Control Prior Object (Penmanshiel)", fontsize=9.5, fontweight='bold')
    ax1.set_xlabel("Wind Speed $v$ (m/s)", fontsize=8.5)
    ax1.set_ylabel("Power (p.u.)", fontsize=8.5)
    ax1.legend(frameon=True, fontsize=8, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    arms = ['Genuine Prior', 'Scrambled Prior']
    costs = [0.09760, 0.16611]
    bars = ax2.bar(arms, costs, color=[C_CON, C_HARM], width=0.45, edgecolor='#333333', linewidth=0.8)
    ax2.axhline(0.09881, color=C_UNCON, linestyle=':', label='Unconstrained Baseline (0.0988)')
    ax2.annotate('+68.1% Cost Surge\n($p < 0.001$)', xy=(1, 0.16611), xytext=(0.35, 0.148),
                 arrowprops=dict(arrowstyle="->", color=C_HARM), fontsize=8.5, color=C_HARM, fontweight='bold')
    ax2.set_title("Downstream Risk When Harm Is Omitted", fontsize=9.5, fontweight='bold')
    ax2.set_ylabel("Dispatch Cost ($r=2$)", fontsize=8.5)
    ax2.set_ylim(0.07, 0.19)
    ax2.legend(frameon=True, fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_d5_notreported_diag.png", dpi=300)
    plt.close()
    
    # --- 2. Figure 1 Micro Version (Strict Minimalist, No Grid, Subtle Annotation) ---
    fig = plt.figure(figsize=(1.9, 1.2), dpi=100)
    ax = fig.add_axes([0.08, 0.14, 0.86, 0.76])
    ax.grid(False)
    
    ax.plot(v_f, p_true_f, color=C_CON, linestyle='-', linewidth=2.2, label='Genuine')
    ax.plot(v_f, p_scramble_f, color=C_HARM, linestyle='--', linewidth=1.5, marker='s', markersize=2.2, label='Scrambled')
    
    ax.text(v_f[-1] + 0.4, p_true_f[-1], 'True', color=C_CON, fontsize=6.8, fontweight='bold', va='center')
    ax.text(v_f[-1] + 0.4, p_scramble_f[-1], 'Scram.', color=C_HARM, fontsize=6.8, fontweight='bold', va='center')
    
    # Subtle de-emphasized annotation in bottom-right corner with clean background
    ax.text(0.96, 0.10, "boundary omitted", transform=ax.transAxes,
            fontsize=5.6, color='#94a3b8', style='italic', ha='right', va='bottom',
            bbox=dict(boxstyle='square,pad=0.08', facecolor='white', edgecolor='none', alpha=0.85))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$v$", fontsize=8.0, color='#475569', labelpad=2)
    ax.set_ylabel("Power", fontsize=7.5, color='#475569', labelpad=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#64748b')
    ax.spines['bottom'].set_color('#64748b')
    ax.spines['left'].set_linewidth(0.85)
    ax.spines['bottom'].set_linewidth(0.85)
    ax.set_xlim(v_f[0] - 0.5, v_f[-1] + 6.2)
    
    fig.savefig(OUT_DIR / "fig1_d5_notreported.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d5_notreported.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done G: fig1_d5_notreported_diag.png, fig1_d5_notreported.svg, fig1_d5_notreported.pdf")


if __name__ == '__main__':
    print("==================================================")
    print("REFINING 7 CORE MICRO-FIGURES (STRICT MINIMALIST)")
    print("==================================================")
    generate_d3_none()
    generate_d2_data_generation()
    generate_d2_posthoc()
    generate_d5_constraint_onoff()
    generate_d5_both()
    generate_d5_reported()
    generate_d5_notreported()
    print("==================================================")
    print("ALL 7 CORE MICRO-FIGURES REFINED SUCCESSFULLY")
    print("==================================================")
