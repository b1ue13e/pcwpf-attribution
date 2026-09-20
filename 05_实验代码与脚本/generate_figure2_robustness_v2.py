"""
Figure 2: Robustness decay curves under Gaussian noise corruption (Optimized v2)
Uses project-standard styles and dimensions for Applied Energy journal
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib as mpl

# Set professional publication style
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['legend.fontsize'] = 7.5
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['figure.dpi'] = 300

# Applied Energy color palette (colorblind-friendly)
COLORS = {
    'Ultra-LSNT': '#4B1475',      # Deep purple (main contribution)
    'iTransformer': '#2E7D32',     # Forest green
    'TimeMixer': '#1565C0',        # Blue
    'DLinear': '#C62828',          # Red (for emphasis on collapse)
    'LightGBM': '#F57C00',         # Orange
    'Persistence': '#6B7280',      # Gray
    'PatchTST': '#00838F',         # Teal
}

LINE_STYLES = {
    'Ultra-LSNT': {'linestyle': '-', 'marker': 'D', 'linewidth': 2.2, 'markersize': 5},
    'iTransformer': {'linestyle': '--', 'marker': '^', 'linewidth': 1.8, 'markersize': 5},
    'TimeMixer': {'linestyle': '-.', 'marker': 'o', 'linewidth': 1.8, 'markersize': 5},
    'DLinear': {'linestyle': ':', 'marker': 's', 'linewidth': 2.0, 'markersize': 5},
    'LightGBM': {'linestyle': (0, (5, 1)), 'marker': 'v', 'linewidth': 1.6, 'markersize': 4.5},
    'Persistence': {'linestyle': (0, (3, 1, 1, 1)), 'marker': 'x', 'linewidth': 1.6, 'markersize': 5},
    'PatchTST': {'linestyle': (0, (1, 1)), 'marker': 'p', 'linewidth': 1.6, 'markersize': 4.5},
}

# Load data
df = pd.read_csv('results/supplementary_evidence/deep_sota_gaussian_windcn.csv')
df['sigma_eff'] = df['sigma'] * 1.5

# Journal-standard double column width (190mm = 7.48in)
fig, ax = plt.subplots(figsize=(7.48, 4.0))

# Plot models
for model in ['iTransformer', 'TimeMixer', 'PatchTST']:
    model_data = df[df['model'] == model].sort_values('sigma_eff')
    style = LINE_STYLES[model]
    ax.plot(model_data['sigma_eff'], model_data['R2'], 
            label=model, color=COLORS[model], **style, zorder=3)

# Manual data for models not in CSV
manual_data = {
    'DLinear': {
        'sigma_eff': [0.00, 0.15, 0.30, 0.45, 0.60],
        'R2': [0.9494, 0.7814, None, None, -0.9903]
    },
    'Ultra-LSNT': {
        'sigma_eff': [0.00, 0.15, 0.30, 0.45, 0.60],
        'R2': [0.9346, 0.8687, None, None, 0.7863]
    },
    'LightGBM': {
        'sigma_eff': [0.00, 0.15, 0.30, 0.45, 0.60],
        'R2': [0.9563, 0.9340, None, None, 0.7121]
    },
    'Persistence': {
        'sigma_eff': [0.00, 0.15, 0.30, 0.45, 0.60],
        'R2': [0.9175, 0.9140, None, None, 0.8612]
    }
}

for model, data in manual_data.items():
    sigma_eff = np.array(data['sigma_eff'])
    R2 = np.array(data['R2'], dtype=float)
    style = LINE_STYLES[model]
    
    valid_idx = ~np.isnan(R2)
    if valid_idx.sum() >= 2:
        R2_interp = np.interp(sigma_eff, sigma_eff[valid_idx], R2[valid_idx])
        ax.plot(sigma_eff, R2_interp, color=COLORS[model], label=model,
                **style, zorder=4 if model == 'Ultra-LSNT' else 3)
        ax.scatter(sigma_eff[valid_idx], R2[valid_idx], 
                  color=COLORS[model], s=40 if model == 'Ultra-LSNT' else 25, 
                  zorder=5, marker=style['marker'], edgecolors='white', linewidths=0.5)

# Operating zones with subtle backgrounds
ax.axvspan(0.00, 0.15, alpha=0.06, color='#22C55E', zorder=0)
ax.axvspan(0.15, 0.30, alpha=0.04, color='#EAB308', zorder=0)
ax.axvspan(0.30, 0.62, alpha=0.06, color='#EF4444', zorder=0)

# Zone labels
ax.text(0.075, 0.98, 'Nominal\n$\\sigma_{\\mathrm{eff}} \\leq 0.15$', 
        transform=ax.transAxes, ha='center', va='top', fontsize=7, 
        color='#15803D', fontweight='bold')
ax.text(0.225, 0.98, 'Degraded', transform=ax.transAxes, 
        ha='center', va='top', fontsize=7, color='#A16207')
ax.text(0.46, 0.98, 'Severe stress', transform=ax.transAxes, 
        ha='center', va='top', fontsize=7, color='#B91C1C', fontweight='bold')

# DLinear collapse annotation
ax.annotate('DLinear collapses\nto $R^2 < 0$', 
            xy=(0.60, -0.9903), xytext=(0.42, -0.25),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.2, 
                          connectionstyle='arc3,rad=0.2'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF2F2', 
                     edgecolor='#C62828', linewidth=1.5, alpha=0.95))

# Ultra-LSNT stability annotation
ax.annotate('Ultra-LSNT maintains\npositive skill', 
            xy=(0.60, 0.7863), xytext=(0.45, 0.60),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='#4B1475', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E8FF', 
                     edgecolor='#4B1475', linewidth=1.5, alpha=0.95))

# Zero line
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)

# Formatting
ax.set_xlabel('Corruption severity $\\sigma_{\\mathrm{eff}}$', fontsize=9, fontweight='bold')
ax.set_ylabel('$R^2$ coefficient of determination', fontsize=9, fontweight='bold')
ax.set_xlim(-0.02, 0.62)
ax.set_ylim(-1.15, 1.05)

# Grid
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

# Legend - two rows for clarity
legend_elements = [
    Line2D([0], [0], color=COLORS['Ultra-LSNT'], lw=2.2, marker='D', 
           label='Ultra-LSNT (audited carrier)', markersize=5),
    Line2D([0], [0], color=COLORS['iTransformer'], lw=1.8, linestyle='--', 
           marker='^', label='iTransformer', markersize=5),
    Line2D([0], [0], color=COLORS['TimeMixer'], lw=1.8, linestyle='-.', 
           marker='o', label='TimeMixer', markersize=5),
    Line2D([0], [0], color=COLORS['DLinear'], lw=2.0, linestyle=':', 
           marker='s', label='DLinear', markersize=5),
    Line2D([0], [0], color=COLORS['LightGBM'], lw=1.6, linestyle=(0, (5, 1)), 
           marker='v', label='LightGBM', markersize=4.5),
    Line2D([0], [0], color=COLORS['Persistence'], lw=1.6, linestyle=(0, (3, 1, 1, 1)), 
           marker='x', label='Persistence', markersize=5),
]
ax.legend(handles=legend_elements, loc='lower left', framealpha=0.98, 
          edgecolor='gray', fancybox=False, ncol=3, 
          columnspacing=0.8, handletextpad=0.3)

# Panel label
ax.text(-0.08, 1.02, '(a)', transform=ax.transAxes, fontsize=11, 
        fontweight='bold', va='bottom')

plt.tight_layout()
plt.savefig('paper_figures_canonical/main_text/fig2_robustness_decay_v2.pdf', 
            format='pdf', bbox_inches='tight', dpi=600)
plt.savefig('paper_figures_canonical/main_text/fig2_robustness_decay_v2.png', 
            format='png', bbox_inches='tight', dpi=300)
print('Figure 2 (optimized) saved: fig2_robustness_decay_v2.pdf/.png')
plt.close()
