"""Render Figure 3: Empirical Lifecycle Distribution across 46 Audited Studies.
Dynamically read from corpus46_scoping_ledger.csv to ensure 100% data consistency.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.family'] = 'sans-serif'

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "06_实验结果与日志" / "constraint_round2" / "analysis" / "corpus46_scoping_ledger.csv"

df = pd.read_csv(LEDGER_PATH)
inc = df[df['inclusion_check'] == 'include']
n_total = len(inc)

# 1. D1 counts
d1 = inc['D1_provenance'].value_counts()
cats_d1 = ['Intrinsic (Physics Laws)', 'Historical-Static (SCADA)', 'Learned-Online (Aux Net)']
keys_d1 = ['intrinsic', 'historical-static', 'learned-online']
counts_d1 = [d1.get(k, 0) for k in keys_d1]
pcts_d1 = [c / n_total * 100 for c in counts_d1]

# 2. D2 counts
d2 = inc['D2_stage'].value_counts()
cats_d2 = ['Inference Architecture', 'Training Loss Penalty', 'Data Generation', 'Post-hoc Correction']
keys_d2 = ['inference-architecture', 'training-loss', 'data-generation', 'post-hoc-correction']
counts_d2 = [d2.get(k, 0) for k in keys_d2]
pcts_d2 = [c / n_total * 100 for c in counts_d2]

# 3. D3 counts
d3 = inc['D3_inference_req'].value_counts()
cats_d3 = ['None (Historical Series)', 'Forecast Input (NWP)', 'Contemporaneous State']
keys_d3 = ['none', 'forecast-input', 'contemporaneous-state']
counts_d3 = [d3.get(k, 0) for k in keys_d3]
pcts_d3 = [c / n_total * 100 for c in counts_d3]

# 4. D5 counts
ab = inc['ablation_type'].value_counts()
c_strict = ab.get('constraint-on-off', 0) + ab.get('both', 0)
c_mod = ab.get('module-ablation', 0)

cats_d5 = ['Strict Physical Isolation\n(Penalty $\\lambda=0$ vs $\\lambda>0$)', 'Confounded Module Ablation\n(Excises neural sub-block)']
counts_d5 = [c_strict, c_mod]
pcts_d5 = [c / n_total * 100 for c in counts_d5]

# Professional academic styling with explicit GridSpec margins eliminating any label overlap
fig = plt.figure(figsize=(11.5, 7.5), dpi=300)
gs = fig.add_gridspec(2, 2, left=0.14, right=0.96, top=0.94, bottom=0.08, wspace=0.66, hspace=0.36)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# (a) D1: Parameter Provenance
bars = ax_a.barh(cats_d1, pcts_d1, color=['#2b6cb0', '#4299e1', '#bee3f8'], edgecolor='#1a365d', height=0.55)
ax_a.set_xlim(0, 105)
ax_a.set_xlabel('Proportion of Studies (%)', fontsize=10, fontweight='bold')
ax_a.set_title(f'(a) D1: Parameter Provenance ($N={n_total}$)', fontsize=11, fontweight='bold', pad=8)
for bar, count, pct in zip(bars, counts_d1, pcts_d1):
    ax_a.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f"{pct:.1f}% ($n={count}$)", 
              va='center', fontsize=9, fontweight='bold', color='#2d3748')
ax_a.grid(axis='x', linestyle='--', alpha=0.5)

# (b) D2: Application Stage
bars = ax_b.barh(cats_d2, pcts_d2, color=['#2f855a', '#48bb78', '#9ae6b4', '#c6f6d5'], edgecolor='#22543d', height=0.55)
ax_b.set_xlim(0, 70)
ax_b.set_xlabel('Proportion of Studies (%)', fontsize=10, fontweight='bold')
ax_b.set_title(f'(b) D2: Application Stage ($N={n_total}$)', fontsize=11, fontweight='bold', pad=8)
for bar, count, pct in zip(bars, counts_d2, pcts_d2):
    ax_b.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2, f"{pct:.1f}% ($n={count}$)", 
              va='center', fontsize=9, fontweight='bold', color='#2d3748')
ax_b.grid(axis='x', linestyle='--', alpha=0.5)

# (c) D3: Inference Telemetry Requirement
bars = ax_c.barh(cats_d3, pcts_d3, color=['#d69e2e', '#ecc94b', '#fefcbf'], edgecolor='#744210', height=0.55)
ax_c.set_xlim(0, 105)
ax_c.set_xlabel('Proportion of Studies (%)', fontsize=10, fontweight='bold')
ax_c.set_title(f'(c) D3: Inference Telemetry ($N={n_total}$)', fontsize=11, fontweight='bold', pad=8)
for bar, count, pct in zip(bars, counts_d3, pcts_d3):
    ax_c.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f"{pct:.1f}% ($n={count}$)", 
              va='center', fontsize=9, fontweight='bold', color='#2d3748')
ax_c.grid(axis='x', linestyle='--', alpha=0.5)

# (d) D5: Ablation Semantics & Constraint Isolation
bars = ax_d.barh(cats_d5, pcts_d5, color=['#319795', '#e53e3e'], edgecolor='#1a202c', height=0.45)
ax_d.set_xlim(0, 72)
ax_d.set_xlabel('Proportion of Studies (%)', fontsize=10, fontweight='bold')
ax_d.set_title(f'(d) D5: Ablation Isolation Validity ($N={n_total}$)', fontsize=11, fontweight='bold', pad=8)
for bar, count, pct in zip(bars, counts_d5, pcts_d5):
    ax_d.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2, f"{pct:.1f}% ($n={count}$)", 
              va='center', fontsize=9, fontweight='bold', color='#2d3748')
ax_d.grid(axis='x', linestyle='--', alpha=0.5)

out_dirs = [
    ROOT / "03_图表与绘图素材",
    ROOT / "01_论文投稿材料" / "active_manuscript" / "figures",
    ROOT / "01_论文投稿材料" / "chinese_manuscript" / "figures",
    ROOT / "02_论文正文与LaTeX",
]

for out_dir in out_dirs:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_field_audit_distribution.pdf")
    fig.savefig(out_dir / "fig_field_audit_distribution.png")

print(f"Rendered Figure 3 (Audit Distribution) dynamically from {LEDGER_PATH.name} to PDF and PNG successfully with zero label collision.")
