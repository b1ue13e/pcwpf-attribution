#!/usr/bin/env python3
"""
Publication-ready rendering of Figure: Noise Injection Protocol Schematic.
Matches Elsevier (Applied Energy) / Nature publication design standards:
- 3-stage pipeline: Clean SCADA Input -> Perturbation Stress Suite -> Benchmark Evaluation
- 4 perturbation modalities with exact mathematical formulas from manuscript:
  1. Gaussian Jitter (turbulent measurement noise)
  2. Random-Walk Drift (sensor calibration bias)
  3. Telemetry Quantization (low-precision discretization)
  4. Dropout / Missingness (channel transmission loss)
- Clean card geometry with colorblind-safe Okabe-Ito accents
- Zero oversized margins, optimized aspect ratio for single/double column LaTeX rendering
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8.5,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
    })


def draw_card(ax, x, y, w, h, bg_color, border_color, border_width=1.0, radius=1.0):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=border_width,
        edgecolor=border_color,
        facecolor=bg_color,
        zorder=2
    )
    ax.add_patch(p)
    return p


def draw_arrow(ax, x1, y1, x2, y2, color='#64748B', lw=1.5):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        zorder=5
    )
    ax.add_patch(a)


def render_noise_protocol():
    setup_style()

    # Proportional dimensions: 11.0 x 4.6 inches
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 46)
    ax.axis('off')

    # Color definitions
    C_DARK = '#0F172A'       # Slate 900
    C_MUTED = '#475569'      # Slate 600
    C_CONTAINER_BG = '#F8FAFC'
    C_CONTAINER_BD = '#94A3B8'
    
    # 4 modality theme colors
    C_GAUSS_BD = '#0284C7'   # Sky 600
    C_GAUSS_BG = '#F0F9FF'   # Sky 50
    C_DRIFT_BD = '#D97706'   # Amber 600
    C_DRIFT_BG = '#FFFBEB'   # Amber 50
    C_QUANT_BD = '#059669'   # Emerald 600
    C_QUANT_BG = '#ECFDF5'   # Emerald 50
    C_MISS_BD = '#DC2626'    # Rose/Red 600
    C_MISS_BG = '#FEF2F2'    # Rose 50

    # ==================== 1. LEFT CARD: CLEAN SCADA INPUT ====================
    draw_card(ax, 2, 2, 21, 42, '#F8FAFC', '#64748B', border_width=1.2, radius=1.2)
    
    ax.text(12.5, 40.5, 'Clean SCADA Input', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=C_DARK)
    ax.text(12.5, 37.8, r'Features $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$', ha='center', va='center',
            fontsize=8.0, color='#0284C7', fontweight='semibold')
    ax.plot([4.0, 21.0], [35.5, 35.5], color='#CBD5E1', lw=0.8, zorder=3)
    
    input_items = [
        r'• Active Wind Power ($P$)',
        r'• Anemometer Speed ($v$)',
        r'• Wind Direction ($\theta$)',
        r'• Temperature / Humidity',
        r'• 15-min sampling interval',
        r'• Lookback: $L = 96$ (24h)',
        r'• Horizon: $H = 24$ (6h)',
    ]
    for i, item in enumerate(input_items):
        ax.text(4.0, 32.5 - i * 3.4, item, ha='left', va='center',
                fontsize=7.8, color=C_MUTED)

    # Connector Arrow 1
    draw_arrow(ax, 23.2, 23, 27.8, 23, color='#475569', lw=1.6)

    # ==================== 2. MIDDLE CONTAINER: PERTURBATION SUITE ====================
    draw_card(ax, 28, 1.5, 54, 43, C_CONTAINER_BG, C_CONTAINER_BD, border_width=1.2, radius=1.5)
    
    # Header Banner
    ax.text(55.0, 41.8, r'Sensor Perturbation Stress Suite',
            ha='center', va='center', fontsize=10.0, fontweight='bold', color=C_DARK)
    ax.text(55.0, 39.2, r'Effective noise ratio $\sigma_{\mathrm{eff}} = \alpha \cdot \sigma_{\mathrm{cfg}} \in [0.00, 0.60]\quad (\alpha = 1.5\ \mathrm{industrial\ scaling})$',
            ha='center', va='center', fontsize=7.6, color=C_MUTED, fontstyle='italic')

    # --- 2.1 GAUSSIAN JITTER (Top-Left) ---
    draw_card(ax, 29.5, 20.5, 24.5, 16.5, C_GAUSS_BG, C_GAUSS_BD, border_width=1.1, radius=1.0)
    ax.text(41.75, 34.5, 'Gaussian Jitter', ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=C_GAUSS_BD)
    ax.text(41.75, 31.0, r'$x_{\mathrm{noisy}} = x + \sigma_{\mathrm{eff}}\,\mathrm{std}(x)\cdot \varepsilon$',
            ha='center', va='center', fontsize=7.8, color=C_DARK)
    ax.text(41.75, 27.8, r'$\varepsilon \sim \mathcal{N}(0, 1)\ \mathrm{i.i.d.}$',
            ha='center', va='center', fontsize=7.5, color=C_MUTED)
    ax.text(41.75, 23.8, 'Turbulent measurement jitter &\nelectromagnetic interference',
            ha='center', va='center', fontsize=7.0, color='#0369A1', style='italic')

    # --- 2.2 RANDOM-WALK DRIFT (Top-Right) ---
    draw_card(ax, 56.0, 20.5, 24.5, 16.5, C_DRIFT_BG, C_DRIFT_BD, border_width=1.1, radius=1.0)
    ax.text(68.25, 34.5, 'Random-Walk Drift', ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=C_DRIFT_BD)
    ax.text(68.25, 31.0, r'$b_t = b_{t-1} + \sigma_{\mathrm{eff}}\,\mathrm{std}(x)\cdot \varepsilon_t$',
            ha='center', va='center', fontsize=7.8, color=C_DARK)
    ax.text(68.25, 27.8, r'$x_t^{\mathrm{drift}} = x_t + b_t,\quad b_1 = 0$',
            ha='center', va='center', fontsize=7.5, color=C_MUTED)
    ax.text(68.25, 23.8, 'Sensor calibration bias, aging &\nambient temperature drift',
            ha='center', va='center', fontsize=7.0, color='#B45309', style='italic')

    # --- 2.3 TELEMETRY QUANTIZATION (Bottom-Left) ---
    draw_card(ax, 29.5, 2.8, 24.5, 16.5, C_QUANT_BG, C_QUANT_BD, border_width=1.1, radius=1.0)
    ax.text(41.75, 16.8, 'Telemetry Quantization', ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=C_QUANT_BD)
    ax.text(41.75, 13.3, r'$\Delta = \sigma_{\mathrm{eff}}\cdot \mathrm{std}(x)$',
            ha='center', va='center', fontsize=7.8, color=C_DARK)
    ax.text(41.75, 10.2, r'$x_t^{\mathrm{quant}} = \Delta \cdot \mathrm{round}(x_t / \Delta)$',
            ha='center', va='center', fontsize=7.8, color=C_MUTED)
    ax.text(41.75, 6.2, 'Low-precision telemetry, edge\nINT8/FP16 truncation loss',
            ha='center', va='center', fontsize=7.0, color='#047857', style='italic')

    # --- 2.4 TELEMETRY DROPOUT (Bottom-Right) ---
    draw_card(ax, 56.0, 2.8, 24.5, 16.5, C_MISS_BG, C_MISS_BD, border_width=1.1, radius=1.0)
    ax.text(68.25, 16.8, 'Telemetry Dropout', ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=C_MISS_BD)
    ax.text(68.25, 13.3, r'$m_t \sim \mathrm{Bernoulli}(1 - p_{\mathrm{eff}})$',
            ha='center', va='center', fontsize=7.8, color=C_DARK)
    ax.text(68.25, 10.2, r'$x_t^{\mathrm{miss}} = \mathrm{LinearInterp}(m_t \odot x_t)$',
            ha='center', va='center', fontsize=7.6, color=C_MUTED)
    ax.text(68.25, 6.2, 'Network packet dropouts &\nintermittent SCADA blackout',
            ha='center', va='center', fontsize=7.0, color='#B91C1C', style='italic')

    # Connector Arrow 2
    draw_arrow(ax, 82.2, 23, 86.8, 23, color='#475569', lw=1.6)

    # ==================== 3. RIGHT CARD: BENCHMARK & EVALUATION ====================
    draw_card(ax, 87, 2, 21, 42, '#F8FAFC', '#1E293B', border_width=1.2, radius=1.2)
    
    ax.text(97.5, 40.5, 'Evaluation Matrix', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=C_DARK)
    ax.text(97.5, 37.8, 'Strict Fair Stress Test', ha='center', va='center',
            fontsize=8.0, color='#0284C7', fontweight='semibold')
    ax.plot([89.0, 106.0], [35.5, 35.5], color='#CBD5E1', lw=0.8, zorder=3)

    eval_items = [
        (r'Benchmarked Models:', True),
        (r'• Ultra-LSNT (Ours)', False),
        (r'• LightGBM (Tree SOTA)', False),
        (r'• DLinear, PatchTST', False),
        (r'• TimeMixer, iTrans.', False),
        ('', False),
        (r'Audited Metrics:', True),
        (r'• Accuracy: $R^2$, RMSE', False),
        (r'• Margin: Gap over tree', False),
        (r'• Stability: Retained $R^2$', False),
    ]
    y_cur = 33.2
    for item, is_header in eval_items:
        if not item:
            y_cur -= 1.0
            continue
        col = C_DARK if is_header else C_MUTED
        size = 8.0 if is_header else 7.5
        weight = 'bold' if is_header else 'normal'
        ax.text(89.0, y_cur, item, ha='left', va='center',
                fontsize=size, fontweight=weight, color=col)
        y_cur -= 2.6

    # Save to all required destinations
    out_dirs = [
        Path('03_图表与绘图素材'),
        Path('03_图表与绘图素材/thumbnails/figure'),
        Path('02_论文正文与LaTeX'),
        Path('02_论文正文与LaTeX/thumbnails/figure'),
        Path('02_论文正文与LaTeX/latex_current_pack/thumbnails/figure'),
    ]

    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / 'noise_protocol_schematic.pdf', format='pdf', bbox_inches='tight', facecolor='white')
        fig.savefig(d / 'noise_protocol_schematic.png', format='png', dpi=600, bbox_inches='tight', facecolor='white')
        fig.savefig(d / 'fig_noise_protocol.pdf', format='pdf', bbox_inches='tight', facecolor='white')
        fig.savefig(d / 'fig_noise_protocol.png', format='png', dpi=600, bbox_inches='tight', facecolor='white')

    print("Successfully generated publication-ready noise protocol schematic!")
    plt.close(fig)


if __name__ == '__main__':
    render_noise_protocol()
