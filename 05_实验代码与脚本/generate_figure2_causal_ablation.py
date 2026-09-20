# -*- coding: utf-8 -*-
"""
generate_figure2_causal_ablation.py
===================================
Regenerate Figure 2 (fig_causal_ablation.pdf) directly from the canonical
unified 5-seed master summaries:
- 06_实验结果与日志/ablation_3d_results/ablation_3d_key_comparison_table.csv
- 06_实验结果与日志/ablation_3d_results/ablation_3d_master_summary.csv

Panels:
(a) Penmanshiel: Prior Specificity under Sample Restriction (N_train = 1,583)
    - Authentic genuine prior: -1.28% Cost (p < 0.01)
    - Pseudo-scrambled prior: +61.2% decision-cost increase (p < 0.001)
(b) Kelmarsh: Low-lambda Mimicry (N_train = 5,098)
    - Unconstrained DLinear baseline: 0.08804 (updated from 0.08812)
    - Genuine vs Scrambled prior at lambda = 0.01 (no clear separation)
(c) Monotonic Deterioration with Proxy Staleness
    - Lags k in {0, 6, 12, 18, 36} (0h, 1h, 2h, 3h, 6h)
    - Updated unconstrained reference lines:
      HoT: 0.08913, Kelmarsh: 0.08804, Penmanshiel: 0.09852
(d) Decoupling of Symmetric Error and Tail Operational Risk
    - Kelmarsh dual-axis: RMSE vs Decision Risk (r=2)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
matplotlib = mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['axes.linewidth'] = 0.8

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CSV = ROOT / '06_实验结果与日志' / 'ablation_3d_results' / 'ablation_3d_key_comparison_table.csv'
OUT_PDF_ACTIVE = ROOT / '01_论文投稿材料' / 'active_manuscript' / 'figures' / 'fig_causal_ablation.pdf'
OUT_PDF_CHINESE = ROOT / '01_论文投稿材料' / 'chinese_manuscript' / 'figures' / 'fig_causal_ablation.pdf'
TMP_PNG = ROOT / 'tmp_render_fig2.png'

def main():
    print("=" * 75)
    print(" Regenerating Figure 2 directly from canonical 5-seed master summaries")
    print("=" * 75)

    if not CANONICAL_CSV.exists():
        print(f"[-] Missing canonical CSV: {CANONICAL_CSV}")
        sys.exit(1)

    df = pd.read_csv(CANONICAL_CSV)
    print(f"[+] Loaded canonical summary: {len(df)} records")

    # Helper to fetch values
    def get_val(site, arm, col='cost_mean'):
        sub = df[(df.site == site) & (df.arm == arm)]
        if sub.empty:
            raise ValueError(f"Missing {site} - {arm}")
        return float(sub[col].values[0])

    def get_std(site, arm, col='cost_std'):
        sub = df[(df.site == site) & (df.arm == arm)]
        if sub.empty:
            raise ValueError(f"Missing {site} - {arm}")
        return float(sub[col].values[0])

    # Canvas setup matching exact PDF aspect ratio (832.35 x 568.99 pt -> 11.56 x 7.90 in)
    fig = plt.figure(figsize=(11.56, 7.90), dpi=300)
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.95, top=0.95, bottom=0.08, hspace=0.32, wspace=0.26)

    # -------------------------------------------------------------------------
    # Panel (a): Penmanshiel Prior Specificity
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    
    arms_a = [
        'vanilla_unconstrained',
        'genuine_curve_l0.01',
        'genuine_curve_l1.0',
        'pseudo_scrambled_curve_l1.0'
    ]
    labels_a = [
        'Unconstrained\nDLinear',
        'Empirical\nPower Curve\n($\\lambda = 0.01$)',
        'Genuine Prior\n($\\lambda = 1.0$)',
        'Scrambled Prior\n($\\lambda = 1.0$)'
    ]
    colors_a = ['#475569', '#64748b', '#2b5278', '#a83d31']
    
    costs_a = [get_val('penmanshiel', a) for a in arms_a]
    stds_a = [get_std('penmanshiel', a) for a in arms_a]

    # Exactly -1.28% and +61.2%
    pct_genuine = ((costs_a[2] - costs_a[0]) / costs_a[0]) * 100.0
    pct_scrambled = ((costs_a[3] - costs_a[0]) / costs_a[0]) * 100.0
    print(f"[Panel a] Genuine Delta: {pct_genuine:.2f}%, Scrambled Delta: +{pct_scrambled:.1f}%")

    x_a = np.arange(len(arms_a))
    bars_a = ax_a.bar(x_a, costs_a, yerr=stds_a, capsize=4, color=colors_a, edgecolor='#1e293b', linewidth=0.9, width=0.55, zorder=3)
    
    ax_a.set_title(r"$\mathbf{(a)\ Penmanshiel:\ Prior\ Specificity\ under\ Sample\ Restriction}\ (N_{\mathrm{train}} = 1{,}583)$", 
                   fontsize=9.8, fontweight='bold', pad=8)
    ax_a.set_ylabel(r"$\mathbf{Dynamic\ Decision\ Cost\ \mathrm{Cost}_{\mathrm{dyn}}\ (r = 2)}$", fontsize=9.2, fontweight='bold')
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(labels_a, fontsize=8.2)
    ax_a.set_ylim(0.085, 0.178)
    ax_a.grid(axis='y', linestyle='--', color='#e2e8f0', linewidth=0.7, alpha=0.8, zorder=0)

    # Annotation for genuine prior reduction
    ax_a.annotate(f"{pct_genuine:.2f}% Cost\n($p < 0.01$)", 
                  xy=(x_a[2], costs_a[2] + stds_a[2] + 0.001), 
                  xytext=(x_a[2] - 0.15, 0.122),
                  ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#1d4ed8',
                  arrowprops=dict(arrowstyle='->', color='#1d4ed8', lw=1.2, shrinkB=2))

    # Annotation for scrambled prior surge
    ax_a.text(x_a[3], costs_a[3] + stds_a[3] + 0.004, 
              f"+{pct_scrambled:.1f}% decision-cost\nincrease ($p < 0.001$)", 
              ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#b91c1c')

    # -------------------------------------------------------------------------
    # Panel (b): Kelmarsh Low-lambda Mimicry
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])

    # Canonical Kelmarsh unconstrained baseline
    c_uncon_km = get_val('kelmarsh', 'vanilla_unconstrained')
    c_uncon_km_std = get_std('kelmarsh', 'vanilla_unconstrained')
    print(f"[Panel b] Kelmarsh Unconstrained baseline: {c_uncon_km:.5f} +/- {c_uncon_km_std:.5f}")

    lambdas = np.array([0.001, 0.01, 0.1, 1.0])
    
    # Genuine prior curve across lambda
    cost_genuine_km = np.array([
        c_uncon_km,
        get_val('kelmarsh', 'genuine_curve_l0.01'),
        get_val('kelmarsh', 'genuine_curve_l0.1'),
        get_val('kelmarsh', 'genuine_curve_l1.0')
    ])
    std_genuine_km = np.array([
        c_uncon_km_std,
        get_std('kelmarsh', 'genuine_curve_l0.01'),
        get_std('kelmarsh', 'genuine_curve_l0.1'),
        get_std('kelmarsh', 'genuine_curve_l1.0')
    ])

    # Pseudo-scrambled prior curve across lambda
    cost_scrambled_km = np.array([
        c_uncon_km,
        get_val('kelmarsh', 'pseudo_scrambled_curve_l0.01'),
        get_val('kelmarsh', 'pseudo_scrambled_curve_l0.1'),
        get_val('kelmarsh', 'pseudo_scrambled_curve_l1.0')
    ])
    std_scrambled_km = np.array([
        c_uncon_km_std,
        get_std('kelmarsh', 'pseudo_scrambled_curve_l0.01'),
        get_std('kelmarsh', 'pseudo_scrambled_curve_l0.1'),
        get_std('kelmarsh', 'pseudo_scrambled_curve_l1.0')
    ])

    # Horizontal baseline line
    ax_b.axhline(c_uncon_km, color='#94a3b8', linestyle=':', linewidth=1.5, zorder=2,
                 label=f"Unconstrained DLinear ({c_uncon_km:.5f})")

    # Plot lines with markers
    ax_b.plot(lambdas, cost_genuine_km, marker='o', markersize=5.5, color='#1e4b6e', linewidth=1.8, zorder=4,
              label=r"Genuine Prior ($C_{\mathrm{emp}}$)")
    ax_b.fill_between(lambdas, cost_genuine_km - std_genuine_km, cost_genuine_km + std_genuine_km,
                      color='#1e4b6e', alpha=0.15, zorder=3)

    # Plot scrambled line: note that at lambda=1.0 on Kelmarsh, cost is 0.1426, which shoots up
    # In the zoomed view for mimicry, we show the curve rising sharply
    ax_b.plot(lambdas, cost_scrambled_km, marker='s', markersize=5.0, color='#8c2d19', linestyle='--', linewidth=1.8, zorder=4,
              label=r"Pseudo-Scrambled Prior ($C_{\mathrm{scramble}}$)")
    ax_b.fill_between(lambdas, cost_scrambled_km - std_scrambled_km, cost_scrambled_km + std_scrambled_km,
                      color='#8c2d19', alpha=0.12, zorder=3)

    ax_b.set_xscale('log')
    ax_b.set_title(r"$\mathbf{(b)\ Kelmarsh:\ Low\text{-}\lambda\ Mimicry}\ (N_{\mathrm{train}} = 5{,}098)$", 
                   fontsize=9.8, fontweight='bold', pad=8)
    ax_b.set_xlabel(r"$\mathbf{Constraint\ Penalty\ Weight\ \lambda}$", fontsize=9.2, fontweight='bold')
    ax_b.set_ylabel(r"$\mathbf{Dynamic\ Decision\ Cost\ \mathrm{Cost}_{\mathrm{dyn}}\ (r = 2)}$", fontsize=9.2, fontweight='bold')
    ax_b.set_ylim(0.0870, 0.0935)  # Focus on the mimicry operating envelope
    ax_b.grid(True, linestyle='--', color='#e2e8f0', linewidth=0.7, alpha=0.8)
    ax_b.legend(frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=7.6, loc='upper left')

    # Arrow indicating mimicry at lambda=0.01
    ax_b.annotate("No clear separation\n($p = 0.231$)", 
                  xy=(0.01, get_val('kelmarsh', 'genuine_curve_l0.01')), 
                  xytext=(0.0085, 0.0908),
                  ha='center', va='bottom', fontsize=7.8, fontweight='bold', color='#581c87',
                  arrowprops=dict(arrowstyle='->', color='#581c87', lw=1.2, shrinkB=2))

    # -------------------------------------------------------------------------
    # Panel (c): Monotonic Deterioration with Proxy Staleness
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])

    lags_h = np.array([0, 1, 2, 3, 6])
    
    # 1. Hill of Towie
    c_uncon_hot = get_val('hill_of_towie', 'vanilla_unconstrained')
    c_hot_lags = np.array([
        c_uncon_hot,
        get_val('hill_of_towie', 'lag_scan_curve_k6'),
        get_val('hill_of_towie', 'lag_scan_curve_k12'),
        get_val('hill_of_towie', 'lag_scan_curve_k18'),
        get_val('hill_of_towie', 'lag_scan_curve_k36')
    ])
    
    # 2. Kelmarsh
    c_km_lags = np.array([
        c_uncon_km,
        get_val('kelmarsh', 'lag_scan_curve_k6'),
        get_val('kelmarsh', 'lag_scan_curve_k12'),
        get_val('kelmarsh', 'lag_scan_curve_k18'),
        get_val('kelmarsh', 'lag_scan_curve_k36')
    ])
    
    # 3. Penmanshiel
    c_uncon_pen = get_val('penmanshiel', 'vanilla_unconstrained')
    c_pen_lags = np.array([
        c_uncon_pen,
        get_val('penmanshiel', 'lag_scan_curve_k6'),
        get_val('penmanshiel', 'lag_scan_curve_k12'),
        get_val('penmanshiel', 'lag_scan_curve_k18'),
        get_val('penmanshiel', 'lag_scan_curve_k36')
    ])

    print(f"[Panel c] Unconstrained baselines: HoT={c_uncon_hot:.5f}, KM={c_uncon_km:.5f}, Pen={c_uncon_pen:.5f}")

    # Plot curves
    ax_c.plot(lags_h, c_hot_lags, marker='o', markersize=5.5, color='#d95f02', linewidth=1.8, label='Hill of Towie (Independent)', zorder=4)
    ax_c.plot(lags_h, c_km_lags, marker='s', markersize=5.0, color='#1f78b4', linewidth=1.8, label='Kelmarsh (Development)', zorder=4)
    ax_c.plot(lags_h, c_pen_lags, marker='^', markersize=5.5, color='#7570b3', linewidth=1.8, label='Penmanshiel (Sample-Restricted)', zorder=4)

    # Reference horizontal dashed lines
    ax_c.axhline(c_uncon_hot, color='#d95f02', linestyle=':', linewidth=1.3, alpha=0.7, label='HoT Unconstrained')
    ax_c.axhline(c_uncon_km, color='#1f78b4', linestyle=':', linewidth=1.3, alpha=0.7, label='Kelmarsh Unconstrained')
    ax_c.axhline(c_uncon_pen, color='#7570b3', linestyle=':', linewidth=1.3, alpha=0.7, label='Penmanshiel Unconstrained')

    ax_c.set_title(r"$\mathbf{(c)\ Monotonic\ Deterioration\ with\ Proxy\ Staleness}$", fontsize=9.8, fontweight='bold', pad=8)
    ax_c.set_xlabel(r"$\mathbf{Historical\ Wind\text{-}Proxy\ Lag\ in\ Training\ Loss\ (Hours)}$", fontsize=9.2, fontweight='bold')
    ax_c.set_ylabel(r"$\mathbf{Dynamic\ Decision\ Cost\ \mathrm{Cost}_{\mathrm{dyn}}\ (r = 2)}$", fontsize=9.2, fontweight='bold')
    ax_c.set_xticks([0, 1, 2, 3, 6])
    ax_c.set_ylim(0.086, 0.105)
    ax_c.grid(True, linestyle='--', color='#e2e8f0', linewidth=0.7, alpha=0.8)
    ax_c.legend(frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=7.0, loc='upper left', ncol=2)

    # -------------------------------------------------------------------------
    # Panel (d): Decoupling of Symmetric Error and Tail Operational Risk
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])

    models_d = [
        'Unconstrained\nDLinear',
        'Empirical\nPower Curve',
        'Conditional\nDistribution',
        'Info-Symmetric\nDual-Channel'
    ]
    arms_d = [
        'vanilla_unconstrained',
        'genuine_curve_l0.01',
        'genuine_dist_l0.01',
        'info_symmetric_unconstrained'
    ]
    
    # Kelmarsh data
    rmse_d = [get_val('kelmarsh', a, col='rmse_mean') for a in arms_d]
    cost_d = [get_val('kelmarsh', a, col='cost_mean') for a in arms_d]

    x_d = np.arange(len(models_d))
    w_d = 0.35

    color_rmse = '#475569'
    color_cost = '#b91c1c'

    ax_d2 = ax_d.twinx()

    bars_rmse = ax_d.bar(x_d - w_d/2, rmse_d, w_d, color=color_rmse, edgecolor='#1e293b', linewidth=0.8, label='RMSE Point Error', zorder=3)
    bars_cost = ax_d2.bar(x_d + w_d/2, cost_d, w_d, color=color_cost, edgecolor='#7f1d1d', linewidth=0.8, label=r'Decision Risk ($r = 2$)', zorder=3)

    ax_d.set_title(r"$\mathbf{(d)\ Decoupling\ of\ Symmetric\ Error\ and\ Tail\ Operational\ Risk}$", fontsize=9.8, fontweight='bold', pad=8)
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(models_d, fontsize=8.0)
    ax_d.set_ylabel(r"$\mathbf{RMSE\ Point\ Error\ (p.u.)}$", fontsize=9.2, fontweight='bold', color=color_rmse)
    ax_d2.set_ylabel(r"$\mathbf{Decision\ Cost\ \mathrm{Cost}_{\mathrm{dyn}}\ (r = 2)}$", fontsize=9.2, fontweight='bold', color=color_cost)

    ax_d.set_ylim(0.165, 0.187)
    ax_d2.set_ylim(0.085, 0.099)
    ax_d.grid(axis='y', linestyle='--', color='#e2e8f0', linewidth=0.7, alpha=0.8)

    # Combined legend
    lines_d1, labels_d1 = ax_d.get_legend_handles_labels()
    lines_d2, labels_d2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines_d1 + lines_d2, labels_d1 + labels_d2, frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=7.6, loc='upper left')

    # Save outputs
    plt.savefig(OUT_PDF_ACTIVE, bbox_inches='tight', dpi=300)
    plt.savefig(OUT_PDF_CHINESE, bbox_inches='tight', dpi=300)
    plt.savefig(TMP_PNG, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"\n[+] Successfully regenerated Figure 2:")
    print(f"    - Active: {OUT_PDF_ACTIVE}")
    print(f"    - Chinese: {OUT_PDF_CHINESE}")
    print(f"    - Preview PNG: {TMP_PNG}")

if __name__ == '__main__':
    main()
