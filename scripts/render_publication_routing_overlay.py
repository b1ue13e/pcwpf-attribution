#!/usr/bin/env python3
"""
Publication-ready rendering of fig_routing_heatmap_overlay_block1:
Continuous 64-Hour Sparse Routing Heatmap & Power Forecast Overlay.

Conforms to Nature / Applied Energy publication visual standards:
- Pure white background, high contrast Blues colormap
- Decoupled Event badges in generous padding above heatmap (zero collision)
- Centered whitespace legend in power overlay
- Dual coordinate clarity: 15-min steps (0-256) and elapsed physical time (0-64h)
- High-resolution vector PDF and 600 DPI PNG export
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

# Color Palette (Nature / Okabe-Ito inspired)
C_DARK = '#0F172A'       # Slate 900
C_SLATE = '#475569'      # Slate 600
C_GRID = '#F1F5F9'       # Slate 100
C_BORDER = '#CBD5E1'     # Slate 300
C_TRUTH = '#0F172A'      # Realized power line
C_PRED = '#0284C7'       # Ultra-LSNT forecast line (Sky 600)
C_EVENT_A = '#0284C7'    # Event A (Wind Jump)
C_EVENT_B = '#D97706'    # Event B (Turbulence Fluctuation)

EXPERT_COLORS = {
    'E1': '#0284C7',
    'E2': '#D97706',
    'E3': '#059669',
    'E4': '#7C3AED',
}


def setup_publication_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8.5,
        'axes.labelsize': 9.5,
        'axes.titlesize': 10.0,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.0,
        'figure.titlesize': 11.0,
        'lines.linewidth': 1.6,
        'axes.linewidth': 0.8,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def load_routing_data() -> pd.DataFrame:
    p1 = Path('03_图表与绘图素材/origin_figures/data/time_resolved_power_routing.csv')
    if p1.exists():
        return pd.read_csv(p1)
    
    p2 = Path('06_实验结果与日志/results/supplementary_evidence/routing_timestep_matrix_ultra_block1.csv')
    if p2.exists():
        raw = pd.read_csv(p2)
        y_true = raw['y_true_h1_kw'].astype(float).to_numpy()
        ramp = np.abs(y_true[4:] - y_true[:-4])
        center = int(np.argmax(ramp) + 4)
        win = 256
        start = max(0, center - win // 2)
        end = min(len(y_true), start + win)
        start = max(0, end - win)
        sub = raw.iloc[start:end].copy().reset_index(drop=True)
        sub.rename(columns={
            'p_e1': 'Expert_1_Weight',
            'p_e2': 'Expert_2_Weight',
            'p_e3': 'Expert_3_Weight',
            'p_e4': 'Expert_4_Weight',
            'y_true_h1_kw': 'Realized_Power_kW',
            'y_pred_h1_kw': 'Predicted_Power_kW',
        }, inplace=True)
        sub['Step'] = np.arange(len(sub))
        return sub
    
    raise FileNotFoundError("Could not locate routing timestep data.")


def render_routing_overlay():
    setup_publication_style()
    df = load_routing_data()

    steps = df['Step'].to_numpy()
    p_e1 = df['Expert_1_Weight'].to_numpy()
    p_e2 = df['Expert_2_Weight'].to_numpy()
    p_e3 = df['Expert_3_Weight'].to_numpy()
    p_e4 = df['Expert_4_Weight'].to_numpy()
    y_true = df['Realized_Power_kW'].to_numpy()
    y_pred = df['Predicted_Power_kW'].to_numpy()

    # Matrix for heatmap: shape (4, N) ordered E1 at bottom, E4 at top
    probs = np.vstack([p_e1, p_e2, p_e3, p_e4])  # row 0=E1, row 3=E4

    # Event windows
    ev_a = (65, 84)
    ev_b = (170, 194)

    fig = plt.figure(figsize=(7.4, 4.4))
    fig.patch.set_facecolor('white')

    # GridSpec with proper spacing
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.24)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[1, 0], sharex=ax_h)

    # ==================== (a) ROUTING HEATMAP ====================
    ax_h.set_facecolor('white')
    im = ax_h.imshow(
        probs,
        aspect='auto',
        origin='lower',
        cmap='Blues',
        vmin=0.0,
        vmax=0.70,
        interpolation='nearest',
        extent=[0, len(steps) - 1, 0.5, 4.5],
    )

    ax_h.set_yticks([1, 2, 3, 4])
    ax_h.set_yticklabels(['E1 (Wind)', 'E2 (Turb)', 'E3 (Temp)', 'E4 (Hum)'],
                         fontsize=8.5, fontweight='bold', color=C_DARK)
    ax_h.set_ylabel('Expert Module', fontweight='bold', color=C_DARK)
    ax_h.tick_params(axis='x', labelbottom=False, bottom=False)

    # Spine styling
    for sp in ax_h.spines.values():
        sp.set_color(C_BORDER)
        sp.set_linewidth(0.8)

    # Event A Highlight Box (dashed boundary)
    ax_h.add_patch(Rectangle(
        (ev_a[0], 0.52), ev_a[1] - ev_a[0], 3.96,
        fill=False, edgecolor=C_EVENT_A, linewidth=1.4, linestyle=(0, (4, 3)), zorder=5
    ))

    # Event B Highlight Box (dashed boundary)
    ax_h.add_patch(Rectangle(
        (ev_b[0], 0.52), ev_b[1] - ev_b[0], 3.96,
        fill=False, edgecolor=C_EVENT_B, linewidth=1.4, linestyle=(0, (4, 3)), zorder=5
    ))

    # Event Callout Badges placed above the heatmap (y=4.92)
    ax_h.text(
        (ev_a[0] + ev_a[1]) / 2, 4.95,
        'Event A: Wind Jump', ha='center', va='bottom',
        fontsize=7.3, fontweight='bold', color=C_EVENT_A,
        bbox=dict(boxstyle='round,pad=0.22', facecolor='#F0F9FF', edgecolor=C_EVENT_A, lw=0.8),
        clip_on=False, zorder=7
    )

    ax_h.text(
        (ev_b[0] + ev_b[1]) / 2, 4.95,
        'Event B: Turbulence Fluctuation', ha='center', va='bottom',
        fontsize=7.3, fontweight='bold', color=C_EVENT_B,
        bbox=dict(boxstyle='round,pad=0.22', facecolor='#FFF7ED', edgecolor=C_EVENT_B, lw=0.8),
        clip_on=False, zorder=7
    )

    ax_h.set_title('(a) Continuous Expert Routing Probabilities (64-Hour Window)',
                   fontweight='bold', color=C_DARK, pad=24)

    # Colorbar aligned to ax_h using divider
    divider = make_axes_locatable(ax_h)
    cax = divider.append_axes("right", size="2.0%", pad=0.12)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Routing Weight', fontsize=8.0, color=C_DARK)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor(C_BORDER)
    cbar.ax.tick_params(labelsize=7.5)

    # ==================== (b) POWER OVERLAY ====================
    ax_l.set_facecolor('white')
    ax_l.set_axisbelow(True)
    ax_l.yaxis.grid(True, linestyle='--', color='#E2E8F0', linewidth=0.7, alpha=0.8)
    ax_l.xaxis.grid(True, linestyle='--', color='#F8FAFC', linewidth=0.7, alpha=0.8)

    # Event Shaded Bands
    ax_l.axvspan(ev_a[0], ev_a[1], color='#E0F2FE', alpha=0.60, zorder=1, label='Event A Window')
    ax_l.axvspan(ev_b[0], ev_b[1], color='#FFEDD5', alpha=0.60, zorder=1, label='Event B Window')

    # Lines: Realized vs Forecast
    ax_l.plot(steps, y_true, color=C_TRUTH, linewidth=1.5, linestyle='-', label='Realized Power', zorder=3)
    ax_l.plot(steps, y_pred, color=C_PRED, linewidth=1.8, linestyle='--', label='Ultra-LSNT Forecast', zorder=4)

    # Spines
    ax_l.spines['top'].set_visible(False)
    ax_l.spines['right'].set_visible(False)
    ax_l.spines['left'].set_color(C_SLATE)
    ax_l.spines['bottom'].set_color(C_SLATE)

    ax_l.set_ylabel('Active Power (kW)', fontweight='bold', color=C_DARK)
    ax_l.set_xlabel('Observation Step Index (15-min intervals, total 64 hours)', fontweight='bold', color=C_DARK)
    ax_l.set_title('(b) Realized vs. 1-Hour-Ahead Forecast Power with Event Regimes',
                   fontweight='bold', color=C_DARK, pad=8)

    ax_l.set_xlim(0, len(steps) - 1)
    ax_l.set_ylim(8000, 78000)
    ax_l.set_xticks(np.arange(0, len(steps) + 1, 32))
    ax_l.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Legend placed in spacious center valley (steps 96 to 160, power is low ~10k-35k kW, zero collision with 66k peak)
    ax_l.legend(
        loc='upper center', bbox_to_anchor=(0.535, 0.98),
        ncol=2, frameon=True, facecolor='white', edgecolor=C_BORDER,
        fontsize=7.0, framealpha=0.95, columnspacing=0.9, handlelength=1.2, handletextpad=0.4, borderpad=0.3
    )

    # Add dummy axes on right to align ax_l width with ax_h (which has a colorbar)
    divider_l = make_axes_locatable(ax_l)
    cax_dummy = divider_l.append_axes("right", size="2.0%", pad=0.12)
    cax_dummy.axis('off')

    # Save to canonical locations
    out_dirs = [
        Path('03_图表与绘图素材'),
        Path('02_论文正文与LaTeX'),
        Path('02_论文正文与LaTeX/latex_current_pack'),
    ]

    for d in out_dirs:
        if d.exists():
            fig.savefig(d / 'fig_routing_heatmap_overlay_block1.pdf', format='pdf', bbox_inches='tight', facecolor='white')
            fig.savefig(d / 'fig_routing_heatmap_overlay_block1.png', format='png', dpi=600, bbox_inches='tight', facecolor='white')

    print("Successfully generated publication-ready fig_routing_heatmap_overlay_block1!")
    plt.close(fig)


if __name__ == '__main__':
    render_routing_overlay()
