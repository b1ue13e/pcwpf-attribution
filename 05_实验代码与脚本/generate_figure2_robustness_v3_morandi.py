"""
Figure 2: Robustness decay curves - Morandi color palette from reference image
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib as mpl

# Morandi color palette from reference image
MORANDI = {
    'pink_light': '#D58FB1',      # R213 G143 B177 - ascr#3 light
    'pink_dark': '#C54E88',       # R197 G78 B136 - ascr#3 dark
    'blue_gray': '#7E98BD',       # R126 G152 B189 - ascr#10
    'purple_gray': '#9693C2',     # R150 G147 B194 - male panel
    'gray': '#7C7A7D',            # R124 G122 B125 - gray
    'bg_light': '#F5F0F0',        # Background light
}

# Set style
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['legend.fontsize'] = 7.5
mpl.rcParams['figure.dpi'] = 300

# Assign colors to models
MODEL_COLORS = {
    'Ultra-LSNT': MORANDI['pink_dark'],      # Main: deep pink
    'iTransformer': MORANDI['blue_gray'],     # Blue-gray
    'TimeMixer': MORANDI['purple_gray'],      # Purple-gray
    'DLinear': MORANDI['gray'],               # Gray (collapse emphasis)
    'LightGBM': MORANDI['pink_light'],        # Light pink
    'Persistence': '#9CA3AF',                 # Light gray
    'PatchTST': '#B4A7D6',                    # Light purple
}

LINE_STYLES = {
    'Ultra-LSNT': {'linestyle': '-', 'marker': 'o', 'linewidth': 2.5, 'markersize': 6},
    'iTransformer': {'linestyle': '--', 'marker': '^', 'linewidth': 2.0, 'markersize': 5.5},
    'TimeMixer': {'linestyle': '-.', 'marker': 's', 'linewidth': 2.0, 'markersize': 5.5},
    'DLinear': {'linestyle': ':', 'marker': 'v', 'linewidth': 2.2, 'markersize': 5.5},
    'LightGBM': {'linestyle': (0, (5, 1)), 'marker': 'D', 'linewidth': 1.8, 'markersize': 5},
    'Persistence': {'linestyle': (0, (3, 1, 1, 1)), 'marker': 'x', 'linewidth': 1.8, 'markersize': 5.5},
    'PatchTST': {'linestyle': (0, (1, 1)), 'marker': 'p', 'linewidth': 1.8, 'markersize': 5},
}

# Load data
df = pd.read_csv('results/supplementary_evidence/deep_sota_gaussian_windcn.csv')
df['sigma_eff'] = df['sigma'] * 1.5

# Journal-standard double column width
fig, ax = plt.subplots(figsize=(7.48, 4.0))

# Plot models from CSV
for model in ['iTransformer', 'TimeMixer', 'PatchTST']:
    model_data = df[df['model'] == model].sort_values('sigma_eff')
    style = LINE_STYLES[model]
    ax.plot(model_data['sigma_eff'], model_data['R2'], 
            label=model, color=MODEL_COLORS[model], **style, zorder=3)

# Manual data
manual_data = {
    'DLinear': {'sigma_eff': [0.00, 0.15, 0.30, 0.45, 0.60], 'R2': [0.9494, 0.7814, None, None, -0.9903]},
    'Ultra-LSNT': {'sigma_eff': [0.00, 0.15, 0.30, 0.45, 0.60], 'R2': [0.9346, 0.8687, None, None, 0.7863]},
    'LightGBM': {'sigma_eff': [0.00, 0.15, 0.30, 0.45, 0.60], 'R2': [0.9563, 0.9340, None, None, 0.7121]},
    'Persistence': {'sigma_eff': [0.00, 0.15, 0.30, 0.45, 0.60], 'R2': [0.9175, 0.9140, None, None, 0.8612]}
}

for model, data in manual_data.items():
    sigma_eff = np.array(data['sigma_eff'])
    R2 = np.array(data['R2'], dtype=float)
    style = LINE_STYLES[model]
    valid_idx = ~np.isnan(R2)
    if valid_idx.sum() >= 2:
        R2_interp = np.interp(sigma_eff, sigma_eff[valid_idx], R2[valid_idx])
        ax.plot(sigma_eff, R2_interp, color=MODEL_COLORS[model], label=model,
                **style, zorder=4 if model == 'Ultra-LSNT' else 3)
        ax.scatter(sigma_eff[valid_idx], R2[valid_idx], 
                  color=MODEL_COLORS[model], s=50 if model == 'Ultra-LSNT' else 30, 
                  zorder=5, marker=style['marker'], edgecolors='white', linewidths=0.5)

# Operating zones (Morandi tones)
ax.axvspan(0.00, 0.15, alpha=0.08, color=MORANDI['blue_gray'], zorder=0)
ax.axvspan(0.15, 0.30, alpha=0.05, color=MORANDI['purple_gray'], zorder=0)
ax.axvspan(0.30, 0.62, alpha=0.08, color=MORANDI['pink_light'], zorder=0)

# Zone labels (Morandi colors)
ax.text(0.075, 0.98, 'Nominal\n$\\sigma \\leq 0.15$', 
        transform=ax.transAxes, ha='center', va='top', fontsize=7, 
        color=MORANDI['blue_gray'], fontweight='bold')
ax.text(0.225, 0.98, 'Degraded', transform=ax.transAxes, 
        ha='center', va='top', fontsize=7, color=MORANDI['purple_gray'])
ax.text(0.46, 0.98, 'Severe stress', transform=ax.transAxes, 
        ha='center', va='top', fontsize=7, color=MORANDI['pink_dark'], fontweight='bold')

# Annotations with Morandi colors
ax.annotate('DLinear collapses\nto $R^2 < 0$', 
            xy=(0.60, -0.9903), xytext=(0.42, -0.25),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color=MORANDI['gray'], lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAFAFA', 
                     edgecolor=MORANDI['gray'], linewidth=1.5, alpha=0.95))

ax.annotate('Ultra-LSNT maintains\npositive skill', 
            xy=(0.60, 0.7863), xytext=(0.45, 0.60),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color=MORANDI['pink_dark'], lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDF2F8', 
                     edgecolor=MORANDI['pink_dark'], linewidth=1.5, alpha=0.95))

# Zero line
ax.axhline(y=0, color=MORANDI['gray'], linestyle='-', linewidth=0.8, alpha=0.4, zorder=1)

# Formatting
ax.set_xlabel('Corruption severity $\\sigma_{\\mathrm{eff}}$', fontsize=9, fontweight='bold')
ax.set_ylabel('$R^2$ coefficient of determination', fontsize=9, fontweight='bold')
ax.set_xlim(-0.02, 0.62)
ax.set_ylim(-1.15, 1.05)
ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

# Legend
legend_elements = [
    Line2D([0], [0], color=MODEL_COLORS['Ultra-LSNT'], lw=2.5, marker='o', 
           label='Ultra-LSNT (audited carrier)', markersize=6),
    Line2D([0], [0], color=MODEL_COLORS['iTransformer'], lw=2.0, linestyle='--', 
           marker='^', label='iTransformer', markersize=5.5),
    Line2D([0], [0], color=MODEL_COLORS['TimeMixer'], lw=2.0, linestyle='-.', 
           marker='s', label='TimeMixer', markersize=5.5),
    Line2D([0], [0], color=MODEL_COLORS['DLinear'], lw=2.2, linestyle=':', 
           marker='v', label='DLinear', markersize=5.5),
    Line2D([0], [0], color=MODEL_COLORS['LightGBM'], lw=1.8, linestyle=(0, (5, 1)), 
           marker='D', label='LightGBM', markersize=5),
    Line2D([0], [0], color=MODEL_COLORS['Persistence'], lw=1.8, linestyle=(0, (3, 1, 1, 1)), 
           marker='x', label='Persistence', markersize=5.5),
]
ax.legend(handles=legend_elements, loc='lower left', framealpha=0.98, 
          edgecolor=MORANDI['gray'], fancybox=False, ncol=3, 
          columnspacing=0.8, handletextpad=0.3)

# Panel label
ax.text(-0.08, 1.02, '(a)', transform=ax.transAxes, fontsize=11, 
        fontweight='bold', va='bottom')

plt.tight_layout()
plt.savefig('paper_figures_canonical/main_text/fig2_robustness_decay_v3.pdf', 
            format='pdf', bbox_inches='tight', dpi=600)
plt.savefig('paper_figures_canonical/main_text/fig2_robustness_decay_v3.png', 
            format='png', bbox_inches='tight', dpi=300)
print('Figure 2 (Morandi palette) saved: fig2_robustness_decay_v3.pdf/.png')
plt.close()
