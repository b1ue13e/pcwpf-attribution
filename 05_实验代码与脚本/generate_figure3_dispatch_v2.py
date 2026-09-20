"""
Figure 3: Dispatch economics comparison (Optimized v2)
Two-panel grouped bar chart with journal-standard styling
"""
import matplotlib.pyplot as plt
import numpy as np
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

# Data from Table 5 (Panel B)
models = ['iTransformer', 'TimeMixer', 'Ultra-LSNT', 'DLinear']
costs = [32.16, 34.90, 35.97, 54.47]
curtailments = [15.22, 19.60, 22.09, 55.71]
fallback_rates = [16.7, 21.7, 24.2, 62.5]

# Color scheme - colorblind friendly, consistent with paper
COLORS = {
    'iTransformer': '#2E7D32',   # Green (best)
    'TimeMixer': '#1565C0',       # Blue
    'Ultra-LSNT': '#4B1475',      # Purple (our method)
    'DLinear': '#C62828',         # Red (worst)
}

colors = [COLORS[m] for m in models]

# Double column width
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.48, 3.2))

x = np.arange(len(models))
width = 0.55

# Panel A: Cost
bars1 = ax1.bar(x, costs, width, color=colors, edgecolor='white', linewidth=1.2, zorder=3)
ax1.set_ylabel('Real-time cost (M CNY)', fontsize=9, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=20, ha='right')
ax1.set_ylim(0, 65)
ax1.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5, zorder=0)
ax1.set_axisbelow(True)

# Value labels on bars
for i, (bar, cost) in enumerate(zip(bars1, costs)):
    height = bar.get_height()
    weight = 'bold' if i == 0 else 'normal'  # Highlight best
    ax1.annotate(f'{cost:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, fontweight=weight)

# Highlight Ultra-LSNT
ax1.annotate('Ultra-LSNT\n(sparse carrier)', 
            xy=(2, 35.97), xytext=(2.5, 50),
            fontsize=7.5, ha='center',
            arrowprops=dict(arrowstyle='->', color='#4B1475', lw=1),
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#F3E8FF', 
                     edgecolor='#4B1475', linewidth=1.2, alpha=0.95))

# Panel B: Curtailment
bars2 = ax2.bar(x, curtailments, width, color=colors, edgecolor='white', linewidth=1.2, zorder=3)
ax2.set_ylabel('Curtailment (M kWh)', fontsize=9, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=20, ha='right')
ax2.set_ylim(0, 70)
ax2.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5, zorder=0)
ax2.set_axisbelow(True)

# Value labels
for i, (bar, curt) in enumerate(zip(bars2, curtailments)):
    height = bar.get_height()
    weight = 'bold' if i == 0 else 'normal'
    ax2.annotate(f'{curt:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, fontweight=weight)

# Highlight DLinear issue
ax2.annotate('DLinear\n(62.5% fallback)', 
            xy=(3, 55.71), xytext=(2.3, 62),
            fontsize=7.5, ha='center',
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1),
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FEF2F2', 
                     edgecolor='#C62828', linewidth=1.2, alpha=0.95))

# Panel labels
ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, fontsize=11, 
        fontweight='bold', va='bottom')
ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, fontsize=11, 
        fontweight='bold', va='bottom')

# Figure title
fig.suptitle('Fixed-interface operational outcomes (120-day rolling test, L1 copper-plate)', 
             fontsize=10, fontweight='bold', y=1.02)

# Footer note
fig.text(0.5, -0.02, 
         'All values under shared mapping-and-clipping interface. '
         'Lower values indicate better operational performance.',
         ha='center', fontsize=7, style='italic', color='#4B5563')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('paper_figures_canonical/main_text/fig3_dispatch_economics_v2.pdf', 
            format='pdf', bbox_inches='tight', dpi=600)
plt.savefig('paper_figures_canonical/main_text/fig3_dispatch_economics_v2.png', 
            format='png', bbox_inches='tight', dpi=300)
print('Figure 3 (optimized) saved: fig3_dispatch_economics_v2.pdf/.png')
plt.close()
