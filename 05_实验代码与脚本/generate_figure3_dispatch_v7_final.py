"""
Figure 3: Dispatch economics - Final version with all labels, no arrows
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

MORANDI = {
    'pink_dark': '#C54E88',
    'blue_gray': '#7E98BD',
    'purple_gray': '#9693C2',
    'gray': '#7C7A7D',
}

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['legend.fontsize'] = 7.5
mpl.rcParams['figure.dpi'] = 300

models = ['iTransformer', 'TimeMixer', 'Ultra-LSNT', 'DLinear']
costs = [32.16, 34.90, 35.97, 54.47]
curtailments = [15.22, 19.60, 22.09, 55.71]

COLORS = {
    'iTransformer': MORANDI['blue_gray'],
    'TimeMixer': MORANDI['purple_gray'],
    'Ultra-LSNT': MORANDI['pink_dark'],
    'DLinear': MORANDI['gray'],
}

colors_list = [COLORS[m] for m in models]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.48, 3.4))

x = np.arange(len(models))
width = 0.55

# Panel A: Cost
bars1 = ax1.bar(x, costs, width, color=colors_list, edgecolor='white', linewidth=1.2, zorder=3)
ax1.set_ylabel('Real-time cost (M CNY)', fontsize=9, fontweight='bold', color='#374151')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=20, ha='right')
ax1.set_ylim(0, 65)
ax1.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.5, zorder=0)
ax1.set_axisbelow(True)

# All value labels for Panel A
for i, (bar, cost) in enumerate(zip(bars1, costs)):
    height = bar.get_height()
    weight = 'bold' if i == 0 else 'normal'
    ax1.annotate(f'{cost:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, fontweight=weight,
                color='#374151')

# Annotation box for Ultra-LSNT without arrow
ax1.annotate('Ultra-LSNT\n(sparse carrier)', 
            xy=(2, 42),  # Position above the bar
            fontsize=7.5, ha='center', color='#374151',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FDF2F8', 
                     edgecolor=MORANDI['pink_dark'], linewidth=1.2, alpha=0.95))

# Panel B: Curtailment
bars2 = ax2.bar(x, curtailments, width, color=colors_list, edgecolor='white', linewidth=1.2, zorder=3)
ax2.set_ylabel('Curtailment (M kWh)', fontsize=9, fontweight='bold', color='#374151')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=20, ha='right')
ax2.set_ylim(0, 70)
ax2.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.5, zorder=0)
ax2.set_axisbelow(True)

# All value labels for Panel B - INCLUDING DLinear (55.7)
for i, (bar, curt) in enumerate(zip(bars2, curtailments)):
    height = bar.get_height()
    weight = 'bold' if i == 0 else 'normal'
    ax2.annotate(f'{curt:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, fontweight=weight,
                color='#374151')

# Annotation box for DLinear without arrow, positioned inside
ax2.annotate('DLinear\n(62.5% fallback)', 
            xy=(2.8, 62),  # Position inside the plot area
            fontsize=7.5, ha='center', color='#374151',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#F3F4F6', 
                     edgecolor=MORANDI['gray'], linewidth=1.2, alpha=0.95))

# Panel labels
ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, fontsize=11, 
        fontweight='bold', va='bottom', color='#374151')
ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, fontsize=11, 
        fontweight='bold', va='bottom', color='#374151')

# Title - raised to avoid overlap
fig.suptitle('Fixed-interface operational outcomes (120-day rolling test, L1 copper-plate)', 
             fontsize=10, fontweight='bold', y=1.06, color='#374151')

# Footer note
fig.text(0.5, -0.02, 
         'All values under shared mapping-and-clipping interface. '
         'Lower values indicate better operational performance.',
         ha='center', fontsize=7, style='italic', color='#6B7280')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig('paper_figures_canonical/main_text/fig3_dispatch_economics_v7.pdf', 
            format='pdf', bbox_inches='tight', dpi=600)
plt.savefig('paper_figures_canonical/main_text/fig3_dispatch_economics_v7.png', 
            format='png', bbox_inches='tight', dpi=300)
print('Figure 3 (final) saved')
plt.close()
