import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def draw_ablation():
    # Nature / Elsevier double column: 10.5 in x 4.2 in, 600 DPI
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)
    
    fig.patch.set_facecolor('white')
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'stix',
    })

    # Colors
    C_BLUE = '#0072B2'
    C_LIGHT_BLUE = '#56B4E9'
    C_MUTED = '#CBD5E1'
    C_ORANGE = '#D55E00'
    C_GRAY = '#64748B'
    C_DARK = '#0F172A'

    # ==================== SUBPLOT A: KERNEL SIZE ====================
    kernel_data = [
        ('k = 3\n(Short)', 0.8505, C_LIGHT_BLUE, 'Turbulence / Jitter'),
        ('k = 7\n(Medium)', 0.8591, C_BLUE, 'Mesoscale Ramps'),
        ('k = 15\n(Long)', 0.8214, C_MUTED, 'Oversmoothing'),
    ]
    
    labels = [d[0] for d in kernel_data]
    scores = [d[1] for d in kernel_data]
    bar_colors = [d[2] for d in kernel_data]
    x = np.arange(len(labels))

    bars = ax1.bar(x, scores, width=0.52, color=bar_colors, edgecolor='#1E293B', linewidth=1.1, zorder=3)
    
    # Grid & Spines
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, linestyle='--', color='#E2E8F0', linewidth=0.8, alpha=0.8)
    ax1.xaxis.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#334155')
    ax1.spines['bottom'].set_color('#334155')

    ax1.set_ylim(0.79, 0.885)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.0, fontweight='bold', color=C_DARK)
    ax1.set_ylabel('Forecasting Accuracy ($R^2$ Score)', fontsize=9.5, fontweight='bold', color=C_DARK)
    ax1.set_title('(a) Temporal Kernel-Size Sensitivity (Wind CN)', fontsize=10.5, fontweight='bold', color=C_DARK, pad=12)

    # Number annotations above bars
    for i, b in enumerate(bars):
        h = b.get_height()
        font_w = 'bold' if i == 1 else 'normal'
        ax1.text(b.get_x() + b.get_width()/2, h + 0.0035, f'{h:.4f}',
                 ha='center', va='bottom', fontsize=8.8, fontweight=font_w, color=C_DARK)

    # Callout badge for optimal k=7 with zero collision
    ax1.annotate('Optimal Inductive Bias\n($k_m = 7$, $R^2 = 0.8591$)',
                 xy=(1.0, 0.866), xytext=(1.0, 0.876),
                 ha='center', va='center', fontsize=8.2, fontweight='bold', color=C_BLUE,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#EFF6FF', edgecolor=C_BLUE, lw=1.1),
                 arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.4, mutation_scale=12))

    # ==================== SUBPLOT B: ROUTER REGULARIZATION LAMBDA ====================
    lambdas = np.array([0.00, 0.01, 0.05, 0.10])
    r2_vals = np.array([0.8767, 0.8715, 0.8832, 0.8731])

    ax2.set_axisbelow(True)
    ax2.yaxis.grid(True, linestyle='--', color='#E2E8F0', linewidth=0.8, alpha=0.8)
    ax2.xaxis.grid(True, linestyle='--', color='#F1F5F9', linewidth=0.8, alpha=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#334155')
    ax2.spines['bottom'].set_color('#334155')

    # Line plot with markers
    ax2.plot(lambdas, r2_vals, color=C_BLUE, linewidth=2.2, marker='o', markersize=6.5,
             markerfacecolor=C_BLUE, markeredgecolor='white', markeredgewidth=1.2,
             zorder=3, label='Validation $R^2$')

    # Highlight optimal peak point
    ax2.scatter([0.05], [0.8832], s=80, color=C_ORANGE, edgecolors='white', linewidth=1.5,
                zorder=4, label=r'Optimal ($\lambda = 0.05$)')

    # Vertical reference line for conservative default
    ax2.axvline(0.01, color=C_GRAY, linestyle=':', linewidth=1.4, zorder=2)

    ax2.set_xlim(-0.01, 0.11)
    ax2.set_ylim(0.865, 0.890)
    ax2.set_xlabel(r'Router Auxiliary Regularization ($\lambda$)', fontsize=9.5, fontweight='bold', color=C_DARK)
    ax2.set_ylabel('Forecasting Accuracy ($R^2$ Score)', fontsize=9.5, fontweight='bold', color=C_DARK)
    ax2.set_title(r'(b) Router Regularization Response Curve', fontsize=10.5, fontweight='bold', color=C_DARK, pad=12)

    # Callout badge for optimal point (above peak, no collision)
    ax2.annotate('Optimal Balance: $R^2 = 0.8832$\n(Mitigates Expert Collapse)',
                 xy=(0.05, 0.8838), xytext=(0.065, 0.8872),
                 ha='center', va='center', fontsize=8.0, fontweight='bold', color=C_ORANGE,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFBEB', edgecolor=C_ORANGE, lw=1.1),
                 arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=1.3, mutation_scale=12))

    # Callout badge for default point (placed cleanly in lower middle whitespace)
    ax2.annotate(r'Conservative Default ($\lambda = 0.01$)' + '\n(Standard Benchmark Setting)',
                 xy=(0.01, 0.8715), xytext=(0.038, 0.869),
                 ha='left', va='center', fontsize=7.8, color='#334155',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8FAFC', edgecolor=C_GRAY, lw=1.0),
                 arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.2, mutation_scale=11))

    ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.0)

    plt.tight_layout(pad=1.5)

    out_dirs = [
        Path('03_图表与绘图素材'),
        Path('02_论文正文与LaTeX'),
        Path('02_论文正文与LaTeX/latex_current_pack'),
    ]

    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / 'fig_ablation.pdf', format='pdf', bbox_inches='tight', facecolor='white')
        fig.savefig(d / 'fig_ablation.png', format='png', dpi=600, bbox_inches='tight', facecolor='white')

    print('Successfully generated publication-ready fig_ablation across all target directories!')
    plt.close(fig)

if __name__ == '__main__':
    draw_ablation()
