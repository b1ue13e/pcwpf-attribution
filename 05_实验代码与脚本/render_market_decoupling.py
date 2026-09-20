"""
Render Figure 7: Market Settlement Decoupling and Metric Inversion Diagnostic
Saves high-res vector PDF and PNG to 03_图表与绘图素材 and 01_论文投稿材料/active_manuscript/figures.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.family'] = 'sans-serif'

ROOT = Path(__file__).resolve().parents[1]

fig = plt.figure(figsize=(10.5, 3.8), dpi=300)
gs = fig.add_gridspec(1, 2, left=0.08, right=0.96, top=0.88, bottom=0.18, wspace=0.38)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# -------------------------------------------------------------
# Panel (a): Reductio ad Absurdum: The Degenerate Zero-Bid Baseline
# -------------------------------------------------------------
models_a = [r'Zero-Bid' + '\n' + r'($\hat{y} \equiv 0$)', 
            'Persistence\nBaseline', 
            'Unconstrained\nDLinear', 
            'Distribution\nPrior']

cost_a = [0.30473, 0.08992, 0.08924, 0.09503]
cashflow_a = [239.299, 9.120, 8.543, 12.175] # in k£

x = np.arange(len(models_a))
width = 0.36

color_cost = '#2b6cb0'      # Deep blue
color_cash = '#dd6b20'      # Deep orange

ax1_twin = ax1.twinx()

bars_cost = ax1.bar(x - width/2, cost_a, width, label=r'Decision Cost ($r=2$, Lower is Better)', 
                    color=color_cost, edgecolor='#1a365d', zorder=3)
bars_cash = ax1_twin.bar(x + width/2, cashflow_a, width, label='Standalone Cashflow (Higher is Better)', 
                         color=color_cash, edgecolor='#7c2d12', zorder=3)

ax1.set_ylabel(r'Normalized Decision Cost ($r=2$)', fontsize=9.5, fontweight='bold', color=color_cost)
ax1_twin.set_ylabel(r'Balancing Net Cashflow (k£ / (turb$\cdot$yr))', fontsize=9.5, fontweight='bold', color=color_cash)

ax1.set_ylim(0, 0.36)
ax1_twin.set_ylim(0, 280)

ax1.set_xticks(x)
ax1.set_xticklabels(models_a, fontsize=8.5, fontweight='bold')
ax1.set_title(r'(a) Reductio ad Absurdum: The Degenerate Zero-Bid Baseline', fontsize=10, fontweight='bold', pad=10)
ax1.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)

# Value annotations for Panel (a)
for bar, val in zip(bars_cost, cost_a):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.008, f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=color_cost)

for bar, val in zip(bars_cash, cashflow_a):
    ax1_twin.text(bar.get_x() + bar.get_width()/2, val + 5, f'£{val:.1f}k', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=color_cash)

# Callout annotation in Panel (a)
ax1.annotate('Degenerate Metric!\nZero-Bid generates ~28×\ncashflow without forecasting!',
             xy=(x[0] + width/2, 240), xycoords=('data', 'data'),
             xytext=(x[0] + 0.5, 0.23), textcoords=('data', 'data'),
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffaf0', edgecolor='#dd6b20', linewidth=1.2),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.15', color='#dd6b20', lw=1.2),
             fontsize=7.5, fontweight='bold', color='#7c2d12')

# Combine legends for Panel (a)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=7.5, framealpha=0.9)


# -------------------------------------------------------------
# Panel (b): Empirical Metric Inversion on Hill of Towie Wind Farm
# -------------------------------------------------------------
metrics_b = ['Standalone Cashflow\n(£8.5k → £12.2k)',
             'RMSE Point Error\n(0.174 → 0.180)',
             'Decision Risk Cost\n(0.089 → 0.095)']
changes_b = [42.5, 3.5, 6.5] # in %
colors_b = ['#38a169', '#e53e3e', '#e53e3e'] # green for apparent cashflow, red for worsening risk

x2 = np.arange(len(metrics_b))
bars2 = ax2.bar(x2, changes_b, width=0.45, color=colors_b, edgecolor='#1a202c', zorder=3)

ax2.set_ylabel('Relative Change vs. Unconstrained DLinear (%)', fontsize=9.5, fontweight='bold')
ax2.set_ylim(-10, 55)
ax2.set_xticks(x2)
ax2.set_xticklabels(metrics_b, fontsize=8.5, fontweight='bold')
ax2.set_title(r'(b) Metric Inversion on Hill of Towie Wind Farm', fontsize=10, fontweight='bold', pad=10)
ax2.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='-')

for bar, val in zip(bars2, changes_b):
    sign = '+' if val > 0 else ''
    ax2.text(bar.get_x() + bar.get_width()/2, val + 1.5, f'{sign}{val:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1a202c')

# Callout annotation in Panel (b)
ax2.text(0.55, 35, 
         'Surface Cashflow Illusion:\nNegative bias coincidentally aligns\nwith SBP price spikes, masquerading\noperational deterioration (+6.5% Risk)\nas commercial profitability.',
         ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.45', facecolor='#fff5f5', edgecolor='#e53e3e', linewidth=1.2),
         fontsize=7.5, fontweight='bold', color='#742a2a')

# Save to output locations
out_dirs = [
    ROOT / '03_图表与绘图素材',
    ROOT / '01_论文投稿材料' / 'active_manuscript' / 'figures'
]

for out_dir in out_dirs:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / 'fig_market_decoupling.pdf')
    fig.savefig(out_dir / 'fig_market_decoupling.png')

print("Rendered fig_market_decoupling.pdf and png successfully with clean UTF-8 text and zero raw escapes.")
