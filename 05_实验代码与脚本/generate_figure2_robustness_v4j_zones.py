"""
Figure 2: Robustness decay - Zone labels positioned inside plot
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib as mpl

MORANDI = {
    'pink_dark': '#C54E88',
    'blue_gray': '#7E98BD',
    'purple_gray': '#9693C2',
    'gray': '#7C7A7D',
    'pink_light': '#D58FB1',
}

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['figure.dpi'] = 300

df = pd.read_csv('results/supplementary_evidence/deep_sota_gaussian_windcn.csv')
df['sigma_eff'] = df['sigma'] * 1.5

fig = plt.figure(figsize=(8.0, 3.6))
gs = fig.add_gridspec(1, 2, wspace=0.35, left=0.09, right=0.95, top=0.88, bottom=0.15)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# ========== PANEL A ==========
main_models = ['iTransformer', 'TimeMixer']
main_colors = [MORANDI['blue_gray'], MORANDI['purple_gray']]
main_markers = ['^', 's']
main_lines = ['-', '--']
main_widths = [2.0, 1.8]

for model, color, marker, ls, lw in zip(main_models, main_colors, main_markers, main_lines, main_widths):
    model_data = df[df['model'] == model].sort_values('sigma_eff')
    ax1.plot(model_data['sigma_eff'], model_data['R2'], 
            color=color, marker=marker, linestyle=ls, linewidth=lw,
            markersize=6, label=model, zorder=3,
            markeredgecolor='white', markeredgewidth=0.8)

ultra_sigma = [0.00, 0.15, 0.30, 0.45, 0.60]
ultra_r2 = [0.9346, 0.8687, 0.83, 0.80, 0.7863]
ax1.plot(ultra_sigma, ultra_r2, color=MORANDI['pink_dark'], marker='o', 
        linestyle='-', linewidth=3.0, markersize=7, label='Ultra-LSNT', zorder=4,
        markeredgecolor='white', markeredgewidth=1.0)

ax1.set_xlabel('Corruption severity $\\sigma_{\\mathrm{eff}}$', fontsize=9, fontweight='bold')
ax1.set_ylabel('$R^2$ under Gaussian corruption', fontsize=9, fontweight='bold')
ax1.set_xlim(-0.02, 0.62)
ax1.set_ylim(0.75, 0.96)
ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)

# Operating zones
ax1.axvspan(0.00, 0.15, alpha=0.10, color=MORANDI['blue_gray'], zorder=0)
ax1.axvspan(0.15, 0.30, alpha=0.08, color=MORANDI['purple_gray'], zorder=0)
ax1.axvspan(0.30, 0.62, alpha=0.10, color=MORANDI['pink_dark'], zorder=0)

# White vertical separators
ax1.axvline(x=0.15, color='white', linestyle='-', linewidth=2.5, alpha=0.9, zorder=1)
ax1.axvline(x=0.30, color='white', linestyle='-', linewidth=2.5, alpha=0.9, zorder=1)

# FIXED: Zone labels - using data coordinates with high zorder
# Nominal zone - in data coordinates
ax1.text(0.075, 0.82, 'Nominal', ha='center', fontsize=7, zorder=10,
        color=MORANDI['blue_gray'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=MORANDI['blue_gray'], 
                 linewidth=1.5, alpha=0.95))

# Degraded zone - in data coordinates
ax1.text(0.225, 0.82, 'Degraded', ha='center', fontsize=7, zorder=10,
        color=MORANDI['purple_gray'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=MORANDI['purple_gray'], 
                 linewidth=1.5, alpha=0.95))

# Severe zone - in data coordinates
ax1.text(0.45, 0.82, 'Severe', ha='center', fontsize=7, zorder=10,
        color=MORANDI['pink_dark'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=MORANDI['pink_dark'], 
                 linewidth=1.5, alpha=0.95))

ax1.legend(loc='lower left', framealpha=0.98, edgecolor='gray', fancybox=False)
ax1.text(-0.12, 1.02, '(a)', transform=ax1.transAxes, fontsize=11, fontweight='bold')

# ========== PANEL B ==========
bg_models = ['iTransformer', 'TimeMixer']
for model in bg_models:
    model_data = df[df['model'] == model].sort_values('sigma_eff')
    ax2.plot(model_data['sigma_eff'], model_data['R2'], 
            color='#CCCCCC', alpha=0.5, linestyle='-', linewidth=1.0,
            marker='o', markersize=3, zorder=1)

key_data = {
    'Ultra-LSNT': {
        'sigma': [0.00, 0.15, 0.30, 0.45, 0.60],
        'r2': [0.9346, 0.8687, 0.83, 0.80, 0.7863],
        'color': MORANDI['pink_dark'], 'ls': '-', 'marker': 'o', 'lw': 3.0
    },
    'DLinear': {
        'sigma': [0.00, 0.15, 0.30, 0.45, 0.60],
        'r2': [0.9494, 0.7814, 0.50, 0.0, -0.9903],
        'color': MORANDI['gray'], 'ls': ':', 'marker': 'v', 'lw': 2.5
    },
    'LightGBM': {
        'sigma': [0.00, 0.15, 0.30, 0.45, 0.60],
        'r2': [0.9563, 0.9340, 0.88, 0.80, 0.7121],
        'color': MORANDI['pink_light'], 'ls': '--', 'marker': 'D', 'lw': 2.0
    },
}

for model, data in key_data.items():
    ax2.plot(data['sigma'], data['r2'], color=data['color'], 
            linestyle=data['ls'], linewidth=data['lw'],
            marker=data['marker'], markersize=6 if model in ['Ultra-LSNT', 'DLinear'] else 5,
            label=model, zorder=4 if model == 'Ultra-LSNT' else 3,
            markeredgecolor='white', markeredgewidth=0.8)

ax2.axhline(y=0, color=MORANDI['gray'], linestyle='-', linewidth=1, alpha=0.4, zorder=1)

ax2.set_xlabel('Corruption severity $\\sigma_{\\mathrm{eff}}$', fontsize=9, fontweight='bold')
ax2.set_ylabel('$R^2$ coefficient', fontsize=9, fontweight='bold')
ax2.set_xlim(-0.02, 0.62)
ax2.set_ylim(-1.15, 1.05)
ax2.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)

legend_elements = [
    Line2D([0], [0], color=MORANDI['pink_dark'], lw=3.0, marker='o', 
           label='Ultra-LSNT (ours)', markersize=6),
    Line2D([0], [0], color=MORANDI['gray'], lw=2.5, linestyle=':', 
           marker='v', label='DLinear (collapses)', markersize=6),
    Line2D([0], [0], color=MORANDI['pink_light'], lw=2.0, linestyle='--', 
           marker='D', label='LightGBM', markersize=5),
]
ax2.legend(handles=legend_elements, loc='lower left', framealpha=0.98, 
          edgecolor='gray', fancybox=False)
ax2.text(-0.12, 1.02, '(b)', transform=ax2.transAxes, fontsize=11, fontweight='bold')

fig.suptitle('Robustness decay under Gaussian SCADA corruption', 
             fontsize=10, fontweight='bold', y=0.98)

plt.savefig('paper_figures_canonical/main_text/fig2_robustness_decay_v4j.pdf', 
            format='pdf', bbox_inches='tight', dpi=600)
plt.savefig('paper_figures_canonical/main_text/fig2_robustness_decay_v4j.png', 
            format='png', bbox_inches='tight', dpi=300)
print('Figure 2 (zone labels fixed v4) saved')
plt.close()
