"""
Figure 2: Robustness decay curves - Clear layered design
Strategy: Split into two panels - main models vs baselines
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib as mpl

# Morandi colors
MORANDI = {
    'pink_dark': '#C54E88',   # Ultra-LSNT
    'blue_gray': '#7E98BD',   # iTransformer
    'purple_gray': '#9693C2', # TimeMixer
    'gray': '#7C7A7D',        # DLinear
    'pink_light': '#D58FB1',  # LightGBM
}

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['figure.dpi'] = 300

# Load data
df = pd.read_csv('results/supplementary_evidence/deep_sota_gaussian_windcn.csv')
df['sigma_eff'] = df['sigma'] * 1.5

# Create 1x2 subplot layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.48, 3.6), 
                                gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.15})

# ========== PANEL A: Main Deep Models (Top performers) ==========
main_models = ['iTransformer', 'TimeMixer', 'Ultra-LSNT']
main_colors = [MORANDI['blue_gray'], MORANDI['purple_gray'], MORANDI['pink_dark']]
main_markers = ['^', 's', 'o']
main_lines = ['-', '--', '-']
main_widths = [2.2, 2.0, 2.8]  # Ultra-LSNT thickest

for model, color, marker, ls, lw in zip(main_models, main_colors, main_markers, main_lines, main_widths):
    model_data = df[df['model'] == model].sort_values('sigma_eff')
    ax1.plot(model_data['sigma_eff'], model_data['R2'], 
            color=color, marker=marker, linestyle=ls, linewidth=lw,
            markersize=7, label=model, zorder=3,
            markeredgecolor='white', markeredgewidth=0.8)

# Add Ultra-LSNT at severe point (from manual data)
ax1.plot([0.60], [0.7863], color=MORANDI['pink_dark'], marker='o', 
        markersize=10, zorder=5, markeredgecolor='white', markeredgewidth=1.5)

# Panel A formatting
ax1.set_xlabel('Corruption severity $\\sigma_{\\mathrm{eff}}$', fontsize=9, fontweight='bold')
ax1.set_ylabel('$R^2$ under Gaussian corruption', fontsize=9, fontweight='bold')
ax1.set_xlim(-0.02, 0.62)
ax1.set_ylim(0.70, 0.96)  # Tight y-range for clarity
ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)

# Operating zones (subtle)
ax1.axvspan(0.00, 0.15, alpha=0.06, color=MORANDI['blue_gray'], zorder=0)
ax1.axvspan(0.15, 0.30, alpha=0.04, color=MORANDI['purple_gray'], zorder=0)
ax1.axvspan(0.30, 0.62, alpha=0.06, color=MORANDI['pink_dark'], zorder=0)

# Zone labels
ax1.text(0.075, 0.955, 'Nominal', transform=ax1.transAxes, ha='center', fontsize=7, 
        color=MORANDI['blue_gray'], fontweight='bold')
ax1.text(0.225, 0.955, 'Degraded', transform=ax1.transAxes, ha='center', fontsize=7, 
        color=MORANDI['purple_gray'])
ax1.text(0.46, 0.955, 'Severe', transform=ax1.transAxes, ha='center', fontsize=7, 
        color=MORANDI['pink_dark'], fontweight='bold')

# Annotation for Ultra-LSNT
ax1.annotate('Ultra-LSNT\n0.786', xy=(0.60, 0.7863), xytext=(0.48, 0.82),
            fontsize=8, ha='center', color=MORANDI['pink_dark'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=MORANDI['pink_dark'], lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=MORANDI['pink_dark'], linewidth=2, alpha=0.95))

ax1.legend(loc='lower left', framealpha=0.98, edgecolor='gray', 
          fancybox=False, handlelength=2.5)
ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, fontsize=11, 
        fontweight='bold', va='bottom')

# ========== PANEL B: Full Range with Baselines ==========
# Plot all models with emphasis on collapse
all_models_data = {
    'iTransformer': {'color': MORANDI['blue_gray'], 'ls': '--', 'marker': '^', 'lw': 1.8},
    'TimeMixer': {'color': MORANDI['purple_gray'], 'ls': '-.', 'marker': 's', 'lw': 1.8},
    'Ultra-LSNT': {'color': MORANDI['pink_dark'], 'ls': '-', 'marker': 'o', 'lw': 2.5},
    'DLinear': {'color': MORANDI['gray'], 'ls': ':', 'marker': 'v', 'lw': 2.5},
    'LightGBM': {'color': MORANDI['pink_light'], 'ls': (0, (5, 1)), 'marker': 'D', 'lw': 1.6},
}

# Plot baselines first (lower zorder)
for model, style in all_models_data.items():
    if model in ['iTransformer', 'TimeMixer']:
        model_data = df[df['model'] == model].sort_values('sigma_eff')
        ax2.plot(model_data['sigma_eff'], model_data['R2'], 
                color=style['color'], alpha=0.4,  # Faded
                linestyle=style['ls'], linewidth=style['lw']*0.7,
                marker=style['marker'], markersize=4, zorder=2)

# Plot key models with full data
manual_data = {
    'Ultra-LSNT': {'sigma_eff': [0.00, 0.15, 0.30, 0.45, 0.60], 'R2': [0.9346, 0.8687, None, None, 0.7863]},
    'DLinear': {'sigma_eff': [0.00, 0.15, 0.30, 0.45, 0.60], 'R2': [0.9494, 0.7814, None, None, -0.9903]},
    'LightGBM': {'sigma_eff': [0.00, 0.15, 0.30, 0.45, 0.60], 'R2': [0.9563, 0.9340, None, None, 0.7121]},
}

for model, data in manual_data.items():
    sigma_eff = np.array(data['sigma_eff'])
    R2 = np.array(data['R2'], dtype=float)
    style = all_models_data[model]
    valid_idx = ~np.isnan(R2)
    if valid_idx.sum() >= 2:
        R2_interp = np.interp(sigma_eff, sigma_eff[valid_idx], R2[valid_idx])
        ax2.plot(sigma_eff, R2_interp, color=style['color'], 
                linestyle=style['ls'], linewidth=style['lw'],
                marker=style['marker'], markersize=6 if model in ['Ultra-LSNT', 'DLinear'] else 5,
                label=model, zorder=4 if model == 'Ultra-LSNT' else 3,
                markeredgecolor='white', markeredgewidth=0.8)

# Zero line
ax2.axhline(y=0, color=MORANDI['gray'], linestyle='-', linewidth=1, alpha=0.5, zorder=1)

# Panel B formatting
ax2.set_xlabel('Corruption severity $\\sigma_{\\mathrm{eff}}$', fontsize=9, fontweight='bold')
ax2.set_ylabel('$R^2$ coefficient', fontsize=9, fontweight='bold')
ax2.set_xlim(-0.02, 0.62)
ax2.set_ylim(-1.15, 1.0)
ax2.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)

# DLinear collapse annotation
ax2.annotate('DLinear\ncollapses', xy=(0.60, -0.9903), xytext=(0.40, -0.50),
            fontsize=9, ha='center', color=MORANDI['gray'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=MORANDI['gray'], lw=1.5,
                          connectionstyle='arc3,rad=-0.2'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', 
                     edgecolor=MORANDI['gray'], linewidth=2, alpha=0.95))

# LightGBM annotation
ax2.annotate('LightGBM', xy=(0.60, 0.7121), xytext=(0.45, 0.40),
            fontsize=8, ha='center', color=MORANDI['pink_light'],
            arrowprops=dict(arrowstyle='->', color=MORANDI['pink_light'], lw=1.2),
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                     edgecolor=MORANDI['pink_light'], linewidth=1.2, alpha=0.9))

# Legend - only key models
legend_elements = [
    Line2D([0], [0], color=MORANDI['pink_dark'], lw=2.5, marker='o', 
           label='Ultra-LSNT (ours)', markersize=6),
    Line2D([0], [0], color=MORANDI['gray'], lw=2.5, linestyle=':', 
           marker='v', label='DLinear (collapses)', markersize=6),
    Line2D([0], [0], color=MORANDI['pink_light'], lw=1.6, linestyle=(0, (5, 1)), 
           marker='D', label='LightGBM', markersize=5),
]
ax2.legend(handles=legend_elements, loc='lower left', framealpha=0.98, 
          edgecolor='gray', fancybox=False, handlelength=2)
ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, fontsize=11, 
        fontweight='bold', va='bottom')

# Figure title
fig.suptitle('Robustness decay under Gaussian SCADA corruption', 
             fontsize=10, fontweight='bold', y=1.02)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig('paper_figures_canonical/main_text/fig2_robustness_decay_v4.pdf', 
            format='pdf', bbox_inches='tight', dpi=600)
plt.savefig('paper_figures_canonical/main_text/fig2_robustness_decay_v4.png', 
            format='png', bbox_inches='tight', dpi=300)
print('Figure 2 (clear design) saved: fig2_robustness_decay_v4.pdf/.png')
plt.close()
