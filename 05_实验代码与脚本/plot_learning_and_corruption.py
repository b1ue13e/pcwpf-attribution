"""
Render Publication-Quality Figures for:
1. Within-Site Learning Curves (Kelmarsh & Hill of Towie)
2. Prior Corruption & Drift Severity Gradients (Penmanshiel & Kelmarsh)
3. PCT JS Divergence Contrast (Official vs Corrected)

Target format: PDF vector graphics adhering to Cell Patterns / Elsevier single/double column standards.
"""

import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Nature/Elsevier styling
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['grid.color'] = '#EAEAEA'
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.5

OUT_FIG_DIR_EN = Path(r"e:\论文1\01_论文投稿材料\active_manuscript\figures")
OUT_FIG_DIR_CN = Path(r"e:\论文1\01_论文投稿材料\chinese_manuscript\figures")
OUT_FIG_DIR_EN.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR_CN.mkdir(parents=True, exist_ok=True)

LEARNING_CSV = Path(r"e:\论文1\06_实验结果与日志\learning_curve\within_site_learning_curve_results.csv")
CORRUPT_CSV = Path(r"e:\论文1\06_实验结果与日志\corruption_severity\corruption_severity_results.csv")
PCT_CSV = Path(r"e:\论文1\06_实验结果与日志\pct_js_contrast\pct_js_contrast_results.csv")

def plot_learning_curves():
    if not LEARNING_CSV.exists():
        print("Waiting for learning curve data...")
        return
    df = pd.read_csv(LEARNING_CSV)
    df = df[df['policy'] == 'calibrated']
    
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=300)
    
    colors = {
        'unconstrained': '#1f77b4',       # Blue
        'genuine_prior': '#2ca02c',       # Green
        'scrambled_prior': '#d62728'      # Red
    }
    markers = {
        'unconstrained': 'o',
        'genuine_prior': 's',
        'scrambled_prior': '^'
    }
    labels = {
        'unconstrained': 'Unconstrained DLinear (λ=0)',
        'genuine_prior': 'Genuine Physical Prior (λ=1.0)',
        'scrambled_prior': 'Pseudo-Scrambled Control (λ=1.0)'
    }
    
    sites = [('kelmarsh', 'Kelmarsh (Flat Terrain, MM92)'), 
             ('hill_of_towie', 'Hill of Towie (Complex Upland, SWT-2.3)')]
    
    for ax, (site_key, site_name) in zip(axes, sites):
        sub = df[df['site'] == site_key]
        for arm in ['unconstrained', 'genuine_prior', 'scrambled_prior']:
            arm_sub = sub[sub['arm'] == arm]
            agg = arm_sub.groupby('n_train')['dispatch_cost_r2'].agg(['mean', 'std']).reset_index()
            ax.plot(agg['n_train'], agg['mean'], label=labels[arm], color=colors[arm], 
                    marker=markers[arm], markersize=5, linewidth=1.6)
            ax.fill_between(agg['n_train'], agg['mean'] - agg['std'], agg['mean'] + agg['std'], 
                            color=colors[arm], alpha=0.15)
        
        ax.set_title(site_name, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Training Sample Size $N_{\\mathrm{train}}$', fontsize=10)
        ax.set_ylabel('Normalized Dispatch Cost ($r=2$)', fontsize=10)
        ax.grid(True)
        ax.set_xlim(400, 5200)
        if site_key == 'kelmarsh':
            ax.set_ylim(0.080, 0.185)
        else:
            ax.set_ylim(0.085, 0.205)
            
    axes[0].legend(frameon=True, facecolor='white', edgecolor='none', fontsize=8.5, loc='upper right')
    plt.tight_layout()
    
    out_pdf_en = OUT_FIG_DIR_EN / "fig_within_site_learning_curve.pdf"
    out_pdf_cn = OUT_FIG_DIR_CN / "fig_within_site_learning_curve.pdf"
    plt.savefig(out_pdf_en, bbox_inches='tight')
    plt.savefig(out_pdf_cn, bbox_inches='tight')
    plt.close()
    print(f"Saved Learning Curve Plot -> {out_pdf_en}")

def plot_corruption_severity():
    if not CORRUPT_CSV.exists():
        print("Waiting for corruption data...")
        return
    df = pd.read_csv(CORRUPT_CSV)
    
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), dpi=300)
    
    # 1. Continuous Alpha Interpolation on Penmanshiel & Kelmarsh
    ax = axes[0]
    for site, color, mkr, lbl in [('penmanshiel', '#d62728', 's', 'Penmanshiel ($N=1,583$)'),
                                  ('kelmarsh', '#1f77b4', 'o', 'Kelmarsh ($N=5,098$)')]:
        sub = df[(df['site'] == site) & (df['experiment'] == 'continuous_alpha')]
        agg = sub.groupby('parameter')['dispatch_cost_cal'].agg(['mean', 'std']).reset_index()
        ax.plot(agg['parameter'], agg['mean'], label=lbl, color=color, marker=mkr, markersize=5, linewidth=1.6)
        ax.fill_between(agg['parameter'], agg['mean'] - agg['std'], agg['mean'] + agg['std'], color=color, alpha=0.15)
        
    ax.axhline(0.09852, color='#d62728', linestyle=':', label='Penmanshiel Unconstrained Baseline', alpha=0.7)
    ax.axhline(0.08804, color='#1f77b4', linestyle=':', label='Kelmarsh Unconstrained Baseline', alpha=0.7)
    ax.set_title('(a) Continuous Prior Distortion $\\alpha$', fontsize=10.5, fontweight='bold')
    ax.set_xlabel('Corruption Factor $\\alpha$ (0=Genuine, 1=Scrambled)', fontsize=9.5)
    ax.set_ylabel('Normalized Dispatch Cost ($r=2$)', fontsize=9.5)
    ax.grid(True)
    ax.legend(frameon=True, fontsize=7.5, loc='upper left')
    
    # 2. Turbine Aging Derating
    ax = axes[1]
    for site, color, mkr, lbl in [('penmanshiel', '#d62728', 's', 'Penmanshiel'),
                                  ('kelmarsh', '#1f77b4', 'o', 'Kelmarsh')]:
        sub = df[(df['site'] == site) & (df['experiment'] == 'turbine_aging')]
        agg = sub.groupby('parameter')['dispatch_cost_cal'].agg(['mean', 'std']).reset_index()
        ax.plot(agg['parameter'], agg['mean'], label=lbl, color=color, marker=mkr, markersize=5, linewidth=1.6)
        ax.fill_between(agg['parameter'], agg['mean'] - agg['std'], agg['mean'] + agg['std'], color=color, alpha=0.15)
        
    ax.set_title('(b) Turbine Aging Capacity Derating', fontsize=10.5, fontweight='bold')
    ax.set_xlabel('Prior Capacity Derating Ratio (1.0 = Nominal)', fontsize=9.5)
    ax.set_ylabel('Normalized Dispatch Cost ($r=2$)', fontsize=9.5)
    ax.grid(True)
    ax.legend(frameon=True, fontsize=8.0, loc='upper right')
    
    # 3. Anemometer Calibration Bias
    ax = axes[2]
    for site, color, mkr, lbl in [('penmanshiel', '#d62728', 's', 'Penmanshiel'),
                                  ('kelmarsh', '#1f77b4', 'o', 'Kelmarsh')]:
        sub = df[(df['site'] == site) & (df['experiment'] == 'sensor_bias')]
        agg = sub.groupby('parameter')['dispatch_cost_cal'].agg(['mean', 'std']).reset_index()
        ax.plot(agg['parameter'], agg['mean'], label=lbl, color=color, marker=mkr, markersize=5, linewidth=1.6)
        ax.fill_between(agg['parameter'], agg['mean'] - agg['std'], agg['mean'] + agg['std'], color=color, alpha=0.15)
        
    ax.set_title('(c) Anemometer Calibration Bias', fontsize=10.5, fontweight='bold')
    ax.set_xlabel('Wind Measurement Scale Bias $\\beta$', fontsize=9.5)
    ax.set_ylabel('Normalized Dispatch Cost ($r=2$)', fontsize=9.5)
    ax.grid(True)
    ax.legend(frameon=True, fontsize=8.0, loc='upper right')
    
    plt.tight_layout()
    out_pdf_en = OUT_FIG_DIR_EN / "fig_prior_corruption_gradient.pdf"
    out_pdf_cn = OUT_FIG_DIR_CN / "fig_prior_corruption_gradient.pdf"
    plt.savefig(out_pdf_en, bbox_inches='tight')
    plt.savefig(out_pdf_cn, bbox_inches='tight')
    plt.close()
    print(f"Saved Corruption Severity Plot -> {out_pdf_en}")

def plot_pct_contrast():
    if not PCT_CSV.exists():
        print("Waiting for PCT contrast data...")
        return
    df = pd.read_csv(PCT_CSV)
    
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=300)
    
    # Left: Gradient Norm vs Lambda (Penmanshiel & Kelmarsh)
    ax = axes[0]
    sub = df[df['mode'].isin(['official_reverse', 'corrected_standard'])]
    
    for site, mkr in [('kelmarsh', 'o'), ('penmanshiel', 's')]:
        site_sub = sub[sub['site'] == site]
        for mode, color, ls, lbl in [('official_reverse', '#d62728', '--', 'Official Reverse-JS'),
                                     ('corrected_standard', '#2ca02c', '-', 'Corrected Standard-JS')]:
            m_sub = site_sub[site_sub['mode'] == mode]
            agg = m_sub.groupby('lambda')['mean_grad_norm'].agg(['mean', 'std']).reset_index()
            site_label = f"{site.capitalize()}: {lbl}"
            ax.plot(agg['lambda'], agg['mean'], label=site_label, color=color, 
                    linestyle=ls, marker=mkr, markersize=5, linewidth=1.6)
            ax.fill_between(agg['lambda'], agg['mean'] - agg['std'], agg['mean'] + agg['std'], color=color, alpha=0.1)
            
    ax.set_xscale('log')
    ax.set_title('(a) Gradient $L_2$ Norm vs. Regularization $\\lambda$', fontsize=10.5, fontweight='bold')
    ax.set_xlabel('Regularization Weight $\\lambda$', fontsize=9.5)
    ax.set_ylabel('Mean Gradient $L_2$ Norm', fontsize=9.5)
    ax.grid(True)
    ax.legend(frameon=True, fontsize=7.5, loc='upper left')
    
    # Right: Dispatch Cost vs Lambda
    ax = axes[1]
    for site, mkr in [('kelmarsh', 'o'), ('penmanshiel', 's')]:
        site_sub = sub[sub['site'] == site]
        base_sub = df[(df['site'] == site) & (df['mode'] == 'unconstrained_baseline')]
        base_cost = base_sub['dispatch_cost_cal'].mean()
        
        ax.axhline(base_cost, color='#666666', linestyle=':', label=f'{site.capitalize()} Baseline ($\\lambda=0$)', alpha=0.7)
        for mode, color, ls, lbl in [('official_reverse', '#d62728', '--', 'Reverse-JS'),
                                     ('corrected_standard', '#2ca02c', '-', 'Standard-JS')]:
            m_sub = site_sub[site_sub['mode'] == mode]
            agg = m_sub.groupby('lambda')['dispatch_cost_cal'].agg(['mean', 'std']).reset_index()
            site_label = f"{site.capitalize()}: {lbl}"
            ax.plot(agg['lambda'], agg['mean'], label=site_label, color=color, 
                    linestyle=ls, marker=mkr, markersize=5, linewidth=1.6)
            ax.fill_between(agg['lambda'], agg['mean'] - agg['std'], agg['mean'] + agg['std'], color=color, alpha=0.1)
            
    ax.set_xscale('log')
    ax.set_title('(b) Dispatch Cost ($r=2$) vs. Regularization $\\lambda$', fontsize=10.5, fontweight='bold')
    ax.set_xlabel('Regularization Weight $\\lambda$', fontsize=9.5)
    ax.set_ylabel('Normalized Dispatch Cost ($r=2$)', fontsize=9.5)
    ax.grid(True)
    ax.legend(frameon=True, fontsize=7.5, loc='upper left')
    
    plt.tight_layout()
    out_pdf_en = OUT_FIG_DIR_EN / "fig_pct_loss_empirical_contrast.pdf"
    out_pdf_cn = OUT_FIG_DIR_CN / "fig_pct_loss_empirical_contrast.pdf"
    plt.savefig(out_pdf_en, bbox_inches='tight')
    plt.savefig(out_pdf_cn, bbox_inches='tight')
    plt.close()
    print(f"Saved PCT Empirical Contrast Plot -> {out_pdf_en}")

if __name__ == '__main__':
    plot_learning_curves()
    plot_corruption_severity()
    plot_pct_contrast()
