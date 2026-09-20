"""
Figure 3: Dispatch economics comparison
Two-panel grouped bar chart showing Cost and Curtailment from Table 5
"""
import matplotlib.pyplot as plt
import numpy as np

# Set paper style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 150

# Data from Table 5 (Panel B: L1 mapped-protocol outcomes)
# Models ordered by cost (ascending)
models = ['iTransformer', 'TimeMixer', 'Ultra-LSNT', 'DLinear']
costs = [32.16, 34.90, 35.97, 54.47]  # M CNY
curtailments = [15.22, 19.60, 22.09, 55.71]  # M kWh
fallback_rates = [16.7, 21.7, 24.2, 62.5]  # %

# Model colors (matching paper palette)
model_colors = {
    'iTransformer': '#2A9D8F',
    'TimeMixer': '#F4A261', 
    'Ultra-LSNT': '#4B1475',
    'DLinear': '#E63946',
}

colors = [model_colors[m] for m in models]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.8))

x = np.arange(len(models))
width = 0.6

# Panel A: Cost
bars1 = ax1.bar(x, costs, width, color=colors, edgecolor='white', linewidth=1)
ax1.set_ylabel('Total real-time cost (M CNY)', fontsize=9)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15, ha='right')
ax1.set_ylim(0, 65)
ax1.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

# Add value labels on bars
for bar, cost in zip(bars1, costs):
    height = bar.get_height()
    ax1.annotate(f'{cost:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# Highlight best
ax1.annotate('Lowest cost', xy=(0, 32.16), xytext=(0.5, 45),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='#2A9D8F', lw=1),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2A9D8F', alpha=0.9))

# Panel B: Curtailment
bars2 = ax2.bar(x, curtailments, width, color=colors, edgecolor='white', linewidth=1)
ax2.set_ylabel('Curtailment (M kWh)', fontsize=9)
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=15, ha='right')
ax2.set_ylim(0, 70)
ax2.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

# Add value labels on bars
for bar, curt in zip(bars2, curtailments):
    height = bar.get_height()
    ax2.annotate(f'{curt:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# Highlight Ultra-LSNT's position
ax2.annotate('Ultra-LSNT\n(compact carrier)', xy=(2, 22.09), xytext=(2.5, 45),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='#4B1475', lw=1),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#4B1475', alpha=0.9))

# Add panel labels
ax1.text(-0.15, 1.05, '(a)', transform=ax1.transAxes, fontsize=11, fontweight='bold', va='top')
ax2.text(-0.15, 1.05, '(b)', transform=ax2.transAxes, fontsize=11, fontweight='bold', va='top')

# Add figure title
fig.suptitle('Fixed-interface operational readout (120-day rolling test)', 
             fontsize=10, fontweight='bold', y=1.02)

# Add note about interface
fig.text(0.5, -0.02, 'All values under shared mapping-and-clipping interface. DLinear shows high fallback rate (62.5%).',
         ha='center', fontsize=7, style='italic', color='#4B5563')

plt.tight_layout()
plt.savefig('paper_figures_canonical/main_text/fig3_dispatch_economics.pdf', 
            format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper_figures_canonical/main_text/fig3_dispatch_economics.png', 
            format='png', bbox_inches='tight', dpi=300)
print('Figure 3 saved: fig3_dispatch_economics.pdf/.png')
plt.close()
