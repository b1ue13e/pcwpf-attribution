from __future__ import annotations

from pathlib import Path
from typing import Iterable

import json
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path('.')
THUMB_DIR = ROOT / 'thumbnails' / 'figure'
DISPATCH_DIR = ROOT / 'dispatch'


PALETTE = {
    # Warm-gray + yellow/beige/blue style (aligned with user-provided reference figures)
    'ours': '#0D67B5',
    'ours_alt': '#8FB1C7',
    'baseline': '#F4B400',
    'baseline2': '#B99567',
    'gray': '#CDBDA4',
    'light_gray': '#E7DFD0',
    'dark': '#2D2D2D',
}

EXPERT_COLORS = ['#1E5DA8', '#4F86C6', '#8FB1C7', '#C8D8E8']


def configure_style() -> None:
    mpl.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'mathtext.fontset': 'stix',
            'axes.unicode_minus': False,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'figure.dpi': 220,
            'savefig.dpi': 1000,
            'axes.linewidth': 0.9,
            'axes.labelsize': 9,
            'axes.titlesize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.facecolor': '#F2F2F2',
            'axes.facecolor': '#F2F2F2',
            'savefig.facecolor': '#F2F2F2',
            'axes.edgecolor': '#9E9E9E',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.65,
            'grid.color': '#D2D2D2',
            'axes.grid': True,
            'grid.linestyle': '--',
        }
    )


def ensure_dirs() -> None:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    DISPATCH_DIR.mkdir(parents=True, exist_ok=True)


def save_pair(fig: plt.Figure, out_base: Path, *, raster_dpi: int = 600) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out_base.with_suffix('.png'), dpi=raster_dpi, bbox_inches='tight')


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str,
    edge: str,
    lw: float = 1.3,
    fontsize: int = 8,
    alpha: float = 1.0,
) -> tuple[float, float]:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle='round,pad=0.02,rounding_size=0.03',
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        alpha=alpha,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, weight='bold', color=PALETTE['dark'])
    return x + w, y + h / 2


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = '#9AA0A6', lw: float = 1.2, alpha: float = 1.0, style: str = '->') -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=lw,
            color=color,
            alpha=alpha,
            connectionstyle='arc3,rad=0.0',
        )
    )


def make_figure_2_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.1))
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 9.0)
    ax.axis('off')

    ax.text(0.4, 8.3, 'Ultra-LSNT Architecture (Edge-Native Sparse MoE)', fontsize=11, weight='bold', color=PALETTE['dark'])

    # Input
    _draw_box(ax, 0.6, 3.8, 1.8, 1.5, 'Input\\n$X\\in\\mathbb{R}^{L\\times C}$', face='#E9EFF4', edge=PALETTE['ours'])

    # Decomposition module
    _draw_box(ax, 2.9, 3.8, 2.0, 1.5, 'Moving Avg\\nDecomposition', face='#EEE9E0', edge='#9AA0A6')
    _arrow(ax, (2.4, 4.55), (2.9, 4.55), color=PALETTE['ours'])

    # Trend/Seasonal
    _draw_box(ax, 5.3, 5.8, 2.2, 1.2, 'Trend Branch', face='#E7DFD0', edge=PALETTE['ours_alt'])
    _draw_box(ax, 5.3, 2.2, 2.2, 1.2, 'Seasonal Branch', face='#ECE4D8', edge=PALETTE['baseline'])
    _arrow(ax, (4.9, 4.8), (5.3, 6.4), color='#9AA0A6')
    _arrow(ax, (4.9, 4.2), (5.3, 2.8), color='#9AA0A6')

    # Multi-scale conv
    _draw_box(ax, 8.0, 4.9, 1.9, 1.1, 'Conv1D\\n$k=3$', face='#D7E3EC', edge=PALETTE['ours'])
    _draw_box(ax, 8.0, 2.5, 1.9, 1.1, 'Conv1D\\n$k=7$', face='#D7E3EC', edge=PALETTE['ours'])
    _arrow(ax, (7.5, 2.8), (8.0, 5.4), color=PALETTE['baseline'])
    _arrow(ax, (7.5, 2.8), (8.0, 3.0), color=PALETTE['baseline'])

    _draw_box(ax, 10.3, 3.55, 1.8, 1.2, 'Fuse + GELU', face='#F2F2F2', edge='#9AA0A6')
    _arrow(ax, (9.9, 5.45), (10.3, 4.15))
    _arrow(ax, (9.9, 3.05), (10.3, 4.15))

    # Router and experts
    router = Circle((12.9, 4.15), 0.58, facecolor='#E8DDCB', edgecolor=PALETTE['baseline2'], linewidth=1.3)
    ax.add_patch(router)
    ax.text(12.9, 4.15, 'Top-K\\nRouter', ha='center', va='center', fontsize=7.5, weight='bold')
    _arrow(ax, (12.1, 4.15), (12.3, 4.15), color=PALETTE['baseline2'])

    expert_y = [6.7, 5.2, 3.1, 1.6]
    active = {0, 2}
    for i, y in enumerate(expert_y):
        is_active = i in active
        face = '#D6E3ED' if is_active else '#EDE7DC'
        edge = PALETTE['ours'] if is_active else '#9CA3AF'
        label = f'Expert {i + 1}\\n' + ('Active' if is_active else 'Inactive')
        _draw_box(ax, 14.2, y, 2.0, 0.9, label, face=face, edge=edge, alpha=1.0 if is_active else 0.72, fontsize=7)
        _arrow(
            ax,
            (13.45, 4.2 + (0.28 * (1.5 - i))),
            (14.2, y + 0.45),
            color=edge,
            lw=1.3 if is_active else 1.0,
            alpha=1.0 if is_active else 0.45,
            style='->' if is_active else '-',
        )

    # Jump gate
    gate = Polygon([[11.1, 6.2], [11.8, 6.8], [12.5, 6.2], [11.8, 5.6]], closed=True, facecolor='#E8DDCB', edgecolor=PALETTE['baseline2'], linewidth=1.2)
    ax.add_patch(gate)
    ax.text(11.8, 6.2, 'Jump\\nGate', ha='center', va='center', fontsize=7.5, weight='bold', color='#7F6A4A')

    _arrow(ax, (10.9, 4.6), (11.3, 5.8), color=PALETTE['baseline2'])
    _arrow(ax, (12.3, 6.2), (16.3, 6.2), color=PALETTE['baseline2'], lw=1.4)
    ax.text(14.2, 6.45, 'Computational Skip', fontsize=7.5, color='#7F6A4A')

    # Merge and output
    merge = Circle((16.5, 4.15), 0.35, facecolor='white', edgecolor='#374151', linewidth=1.1)
    ax.add_patch(merge)
    ax.text(16.5, 4.15, '$\\oplus$', ha='center', va='center', fontsize=10)
    _arrow(ax, (7.5, 6.4), (16.15, 4.35), color=PALETTE['ours_alt'])
    _arrow(ax, (16.2, 6.2), (16.5, 4.52), color='#B91C1C', lw=1.1)
    _arrow(ax, (16.85, 4.15), (17.1, 4.15), color='#374151')
    ax.text(17.12, 4.15, '$\\hat{Y}$', fontsize=10, weight='bold', va='center', color=PALETTE['dark'])

    ax.text(0.7, 0.65, 'N=4 experts, K=2 active (50% sparse compute), with conditional skip for edge efficiency.', fontsize=8, color='#6A6A6A')

    fig.tight_layout(rect=[0.0, 0.0, 0.96, 1.0])
    save_pair(fig, ROOT / 'Figure_2_Final')
    plt.close(fig)


def make_raw_series() -> None:
    df = pd.read_csv('wind_final.csv')
    if 'power' not in df.columns:
        raise ValueError('wind_final.csv must contain a "power" column')

    series = df['power'].astype(float).to_numpy()
    # 7 days at 15-min intervals (96 points/day)
    length = 96 * 7
    start = min(5000, max(0, series.size - length - 1))
    segment = series[start : start + length]
    t = np.arange(segment.size)

    diff = np.abs(np.diff(segment, prepend=segment[0]))
    ramp_thr = np.percentile(diff, 93)
    ramp_idx = np.where(diff >= ramp_thr)[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.8, 5.8), gridspec_kw={'height_ratios': [2.1, 1.0]})

    ax1.plot(t, segment, color=PALETTE['ours'], linewidth=1.3, label='Measured power')
    ax1.plot(t, pd.Series(segment).rolling(8, min_periods=1).mean(), color=PALETTE['baseline'], linewidth=1.2, linestyle='--', label='Rolling mean')
    for idx in ramp_idx[:8]:
        ax1.axvspan(max(0, idx - 1), min(segment.size - 1, idx + 1), color='#DDB892', alpha=0.24)
    ax1.set_ylabel('Power')
    ax1.set_title('Representative Raw Wind Power Segment (Wind CN)')
    ax1.legend(loc='upper right', frameon=False)
    ax1.grid(True, axis='x', alpha=0.2)

    sns.histplot(segment, bins=40, stat='density', color=PALETTE['ours_alt'], edgecolor='white', ax=ax2)
    sns.kdeplot(segment, color=PALETTE['baseline2'], linewidth=1.6, ax=ax2)
    ax2.set_xlabel('Power')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution in the same segment')

    fig.tight_layout()
    save_pair(fig, THUMB_DIR / 'raw_series_wind_cn', raster_dpi=450)
    plt.close(fig)


def make_noise_protocol() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    ax.set_xlim(0.0, 15.2)
    ax.set_ylim(0.2, 8.2)
    ax.axis('off')

    ax.text(0.25, 7.7, 'Robustness perturbation protocol: clean series to dispatch-coupled metrics', fontsize=11.0, weight='bold', color=PALETTE['dark'])

    _draw_box(ax, 0.4, 3.9, 2.1, 1.25, 'Clean\\nseries', face='#D7E3EC', edge=PALETTE['ours'])
    _draw_box(ax, 2.9, 3.9, 2.2, 1.25, 'Perturbation\\nfamily selection', face='#ECE4D8', edge=PALETTE['baseline2'])

    _draw_box(
        ax,
        5.5,
        5.6,
        3.1,
        1.45,
        'continuous perturbations\\nGaussian / drift / quantization',
        face='#E8EEF7',
        edge=PALETTE['ours_alt'],
        fontsize=7,
    )
    _draw_box(
        ax,
        5.5,
        3.55,
        3.1,
        1.45,
        'structured SCADA faults\\nflatline / spike / dropout',
        face='#F2E6D6',
        edge=PALETTE['baseline2'],
        fontsize=7,
    )
    _draw_box(
        ax,
        5.5,
        1.5,
        3.1,
        1.45,
        'missingness + interpolation\\nMCAR mask / ffill / linear',
        face='#E8E8E8',
        edge='#8F8F8F',
        fontsize=7,
    )

    _draw_box(ax, 9.0, 3.85, 2.0, 1.35, 'Corrupted\\nseries', face='#ECE4D8', edge=PALETTE['baseline2'])
    _draw_box(ax, 11.4, 5.6, 2.7, 1.0, 'deep models', face='#E8EEF7', edge=PALETTE['ours'], fontsize=8)
    _draw_box(ax, 11.4, 4.1, 2.7, 1.0, 'feature-based baselines', face='#F2F2F2', edge='#8F8F8F', fontsize=8)
    _draw_box(ax, 11.4, 2.6, 2.7, 1.0, 'dispatch-coupled validation', face='#E6F2E6', edge='#4E8F5A', fontsize=8)
    _draw_box(
        ax,
        11.4,
        1.0,
        2.7,
        1.2,
        'metrics\\n$R^2$, RMSE, MAE, violation, dispatch',
        face='#EDE7DC',
        edge=PALETTE['baseline2'],
        fontsize=7,
    )

    _arrow(ax, (2.55, 4.5), (2.9, 4.5), color=PALETTE['ours'])
    _arrow(ax, (5.1, 4.5), (5.5, 6.2), color=PALETTE['ours_alt'])
    _arrow(ax, (5.1, 4.5), (5.5, 4.25), color=PALETTE['baseline2'])
    _arrow(ax, (5.1, 4.5), (5.5, 2.2), color='#8F8F8F')
    for y in [6.2, 4.25, 2.2]:
        _arrow(ax, (8.6, y), (9.0, 4.5), color='#6B7280')
    for y in [6.1, 4.6, 3.1, 1.6]:
        _arrow(ax, (11.0, 4.5), (11.4, y), color='#6B7280')

    ax.text(0.4, 0.55, 'Protocol branches cover both continuous perturbations and structured telemetry faults under a unified evaluation interface.', fontsize=7.5, color='#6A6A6A')

    fig.tight_layout()
    save_pair(fig, THUMB_DIR / 'noise_protocol_schematic', raster_dpi=450)
    plt.close(fig)


def make_figure_3_performance() -> None:
    data = [
        ('DLinear', 0.9400, 6236.39, 4085.09),
        ('Mamba (S6)', 0.9208, 7165.88, 4875.17),
        ('iTransformer', 0.9012, 8004.05, 6333.29),
        ('TimeMixer', 0.8886, 8498.38, 6661.36),
        ('Ultra-LSNT', 0.8728, 9080.55, 7171.42),
        ('PatchTST', 0.8666, 9298.08, 7342.16),
        ('Informer', 0.8167, 10900.65, 8765.92),
        ('Autoformer', 0.7219, 13424.28, 10646.77),
    ]
    df = pd.DataFrame(data, columns=['Model', 'R2', 'RMSE', 'MAE'])

    def color_for(m: str) -> str:
        if m == 'Ultra-LSNT':
            return PALETTE['ours']
        if m == 'DLinear':
            return PALETTE['baseline']
        return PALETTE['light_gray']

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5))

    for ax, metric, asc, title in [
        (axes[0], 'RMSE', True, '(a) RMSE (lower is better)'),
        (axes[1], 'MAE', True, '(b) MAE (lower is better)'),
        (axes[2], 'R2', False, '(c) $R^2$ (higher is better)'),
    ]:
        order = df.sort_values(metric, ascending=asc)
        y = np.arange(len(order))
        vals = order[metric].to_numpy()
        models = order['Model'].tolist()
        ax.barh(y, vals, color=[color_for(m) for m in models], edgecolor='white', linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(False, axis='y')
        if metric == 'R2':
            ax.set_xlim(0.70, 0.96)
            for yi, v in zip(y, vals):
                ax.text(min(0.958, v + 0.004), yi, f'{v:.3f}', va='center', fontsize=7)
        else:
            for yi, v in zip(y, vals):
                ax.text(v * 1.01, yi, f'{v:,.0f}', va='center', fontsize=7)

    fig.tight_layout()
    save_pair(fig, ROOT / 'Figure_3_Final', raster_dpi=500)
    plt.close(fig)


def make_figure_4_robustness() -> None:
    ultra_short_x = np.array([0.00, 0.05, 0.10, 0.15, 0.20])
    ultra_short_y = np.array([0.8728, 0.8719, 0.8687, 0.8629, 0.8552])

    x = np.array([0.00, 0.10, 0.20, 0.30, 0.40])
    ultra = np.array([0.8728, 0.8623, 0.8327, 0.7860, 0.7246])
    lgbm = np.array([0.9067, 0.8850, 0.8251, 0.7259, 0.6123])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={'width_ratios': [1.0, 1.15]})

    ax = axes[0]
    ax.plot(ultra_short_x, ultra_short_y, color=PALETTE['ours'], marker='o', linewidth=2.0, label='Ultra-LSNT')
    ax.fill_between(ultra_short_x, ultra_short_y - 0.002, ultra_short_y + 0.002, color=PALETTE['ours'], alpha=0.12)
    ax.set_title('(a) Universal noise stress test (0-20%)')
    ax.set_xlabel('Noise level')
    ax.set_ylabel('$R^2$')
    ax.set_ylim(0.848, 0.876)
    ax.legend(frameon=False, loc='lower left')
    rel_drop = (ultra_short_y[0] - ultra_short_y[-1]) / ultra_short_y[0] * 100
    ax.text(0.97, 0.08, f'Relative drop @20%: {rel_drop:.2f}%', transform=ax.transAxes, ha='right', fontsize=8)

    ax = axes[1]
    ax.plot(x, ultra, color=PALETTE['ours'], marker='o', linewidth=2.1, label='Ultra-LSNT')
    ax.plot(x, lgbm, color=PALETTE['baseline'], marker='s', linewidth=2.0, linestyle='--', label='LightGBM')
    ax.axvspan(0.20, 0.40, color='#E8DDCB', alpha=0.45)
    ax.text(0.205, 0.60, 'High-noise SCADA regime', fontsize=7.5, color='#7F6A4A')

    # Crossover annotation (linear interpolation between 0.1 and 0.2)
    d1 = lgbm[1] - ultra[1]
    d2 = lgbm[2] - ultra[2]
    cross_x = 0.10 + (0.20 - 0.10) * (d1 / (d1 - d2))
    cross_y = ultra[1] + (ultra[2] - ultra[1]) * ((cross_x - 0.10) / 0.10)
    ax.scatter([cross_x], [cross_y], s=35, color='#111827', zorder=5)
    ax.annotate('Robustness crossover', xy=(cross_x, cross_y), xytext=(cross_x + 0.03, cross_y + 0.035), arrowprops={'arrowstyle': '->', 'lw': 0.9}, fontsize=7.5)

    ax.set_title('(b) Extended severe-noise setting (0-40%)')
    ax.set_xlabel('Noise level')
    ax.set_ylabel('$R^2$')
    ax.set_ylim(0.58, 0.93)
    ax.legend(frameon=False, loc='upper right')

    fig.tight_layout()
    save_pair(fig, ROOT / 'Figure_4_Final', raster_dpi=520)
    plt.close(fig)


def make_multi_domain() -> None:
    df = pd.DataFrame(
        [
            ('Wind (CN)', 0.8728, 0.0),
            ('Wind (US)', 0.7750, 11.4),
            ('Air Quality', 0.4885, 0.0),
            ('GEFCom Load', 0.7554, 0.0),
        ],
        columns=['Dataset', 'R2', 'SkipRate'],
    )

    x = np.arange(df.shape[0])

    # Reference palette (from user-provided style sample)
    # blues: R213 G225 B239, R157 G194 B213, R90 G148 B185, R42 G99 B152
    # green: R126 G184 B123
    domain_colors = {
        'Wind (CN)': '#2A6398',
        'Wind (US)': '#5A94B9',
        'Air Quality': '#D5E1EF',
        'GEFCom Load': '#9DC2D5',
    }
    skip_color = '#111111'

    fig, ax1 = plt.subplots(figsize=(8.8, 4.7))
    bars = ax1.bar(
        x,
        df['R2'],
        color=[domain_colors.get(d, '#9DC2D5') for d in df['Dataset']],
        edgecolor='white',
        linewidth=1.0,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Dataset'])
    ax1.set_ylim(0.0, 0.95)
    ax1.set_ylabel('$R^2$', color='#2A6398')
    ax1.tick_params(axis='y', labelcolor='#2A6398')
    ax1.set_title('Multi-domain Generalization and Conditional-Skip Behavior')
    ax1.grid(False, axis='x')

    for b in bars:
        h = b.get_height()
        ax1.text(
            b.get_x() + b.get_width() / 2,
            h + 0.015,
            f'{h:.3f}',
            ha='center',
            va='bottom',
            fontsize=8,
            color='#234D78',
        )

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        df['SkipRate'],
        color=skip_color,
        marker='D',
        linewidth=2.0,
        linestyle='--',
        label='Skip rate',
    )
    ax2.set_ylim(0, max(12.5, float(df['SkipRate'].max()) * 1.2 + 1.0))
    ax2.set_ylabel('Skip rate (%)', color=skip_color)
    ax2.tick_params(axis='y', labelcolor=skip_color)

    for i, s in enumerate(df['SkipRate']):
        ax2.text(i + 0.04, s + 0.35, f'{s:.1f}%', fontsize=7.5, color=skip_color)

    fig.tight_layout()
    save_pair(fig, ROOT / 'fig_multi_domain', raster_dpi=500)
    plt.close(fig)


def make_ablation() -> None:
    kernel_df = pd.DataFrame({'Kernel': [7, 3, 15], 'R2': [0.8591, 0.8505, 0.8214]})
    lambda_df = pd.DataFrame({'Lambda': [0.00, 0.01, 0.05, 0.10], 'R2': [0.8767, 0.8715, 0.8832, 0.8731]})

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    ax = axes[0]
    order = kernel_df.sort_values('R2', ascending=False)
    bars = ax.bar(order['Kernel'].astype(str), order['R2'], color=['#E8DDCB', '#CDBDA4', '#8FB1C7'], edgecolor='white')
    ax.set_title('(a) Kernel-size sensitivity (Wind CN)')
    ax.set_xlabel('Kernel size')
    ax.set_ylabel('$R^2$')
    ax.set_ylim(0.80, 0.87)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.0015, f'{h:.4f}', ha='center', fontsize=7.5)

    ax = axes[1]
    ax.plot(lambda_df['Lambda'], lambda_df['R2'], color=PALETTE['ours'], marker='o', linewidth=2.0)
    best_idx = int(lambda_df['R2'].idxmax())
    best_x = float(lambda_df.loc[best_idx, 'Lambda'])
    best_y = float(lambda_df.loc[best_idx, 'R2'])
    ax.scatter([best_x], [best_y], color=PALETTE['baseline2'], s=35, zorder=4)
    ax.annotate('Best', (best_x, best_y), textcoords='offset points', xytext=(4, 7), fontsize=8)
    ax.axvline(0.01, color='#8F8F8F', linestyle=':', linewidth=1.0)
    ax.text(0.0115, 0.8706, 'Default', fontsize=7.3, color='#8F8F8F')
    ax.set_title('(b) Router regularization $\\lambda$')
    ax.set_xlabel('$\\lambda$')
    ax.set_ylabel('$R^2$')
    ax.set_ylim(0.868, 0.886)

    fig.tight_layout()
    save_pair(fig, ROOT / 'fig_ablation', raster_dpi=520)
    plt.close(fig)


def _pareto_front(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier: list[tuple[float, float]] = []
    best_y = -np.inf
    for x, y in sorted_pts:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def make_efficiency() -> None:
    df = pd.DataFrame(
        [
            ('TimeMixer', 0.8886, 2.37, 7.27, 10.91),
            ('Ultra-LSNT-Lite (orig.)', 0.8728, 5.12, 8.94, 13.41),
            ('Ultra-LSNT-Lite (opt.)', 0.8564, 1.33, 6.53, 9.80),
            ('Ultra-LSNT (full)', 0.8728, 131.58, 15.29, np.nan),
        ],
        columns=['Model', 'R2', 'SizeMiB', 'LatencyMs', 'EnergymJ'],
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={'width_ratios': [1.15, 1.0]})

    ax = axes[0]
    scatter_df = df.dropna(subset=['EnergymJ']).copy()
    for _, row in scatter_df.iterrows():
        marker = '*' if 'opt.' in row['Model'] else 'o'
        color = PALETTE['ours'] if 'Ultra-LSNT' in row['Model'] else PALETTE['baseline']
        ax.scatter(row['EnergymJ'], row['R2'], s=70 + row['SizeMiB'] * 35, color=color, marker=marker, alpha=0.85, edgecolor='white', linewidth=0.8)
        ax.text(row['EnergymJ'] + 0.15, row['R2'] + 0.001, row['Model'].replace('Ultra-LSNT-', 'ULSNT-'), fontsize=7)

    frontier = _pareto_front([(float(r['EnergymJ']), float(r['R2'])) for _, r in scatter_df.iterrows()])
    if len(frontier) >= 2:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], color='#374151', linestyle='--', linewidth=1.1, label='Pareto frontier')

    ax.set_xlabel('Energy per inference (mJ)')
    ax.set_ylabel('$R^2$')
    ax.set_title('(a) Accuracy-energy trade-off')
    ax.set_xlim(9.0, 14.2)
    ax.set_ylim(0.852, 0.892)
    ax.legend(frameon=False, loc='lower right')

    ax = axes[1]
    x = np.arange(df.shape[0])
    bars = ax.bar(x, df['LatencyMs'], color=[PALETTE['gray'], PALETTE['ours_alt'], PALETTE['ours'], PALETTE['light_gray']], edgecolor='white', label='Latency (ms)')
    ax.set_ylabel('Latency (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(['TM', 'Lite-orig', 'Lite-opt', 'Full'])
    ax.set_title('(b) Latency and footprint')

    ax2 = ax.twinx()
    ax2.plot(x, df['SizeMiB'], color=PALETTE['baseline2'], marker='D', linewidth=1.8, label='Size (MiB)')
    ax2.set_yscale('log')
    ax2.set_ylabel('Model size (MiB, log)')

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.25, f'{h:.2f}', ha='center', fontsize=7)

    ax.text(0.98, 0.05, 'Lite-opt: 6.53 ms | 9.80 mJ | 1.33 MiB', transform=ax.transAxes, ha='right', fontsize=7.4, color=PALETTE['ours'])

    fig.tight_layout()
    save_pair(fig, ROOT / 'fig_efficiency', raster_dpi=520)
    plt.close(fig)


def make_figure_5_interpretability() -> None:
    df = pd.read_csv('wind_expert_activation.csv')
    required_cols = {'Hour', 'Expert_1', 'Expert_2', 'Expert_3', 'Expert_4'}
    if not required_cols.issubset(df.columns):
        raise ValueError('wind_expert_activation.csv missing required columns')

    df = df.sort_values('Hour')
    hours = df['Hour'].to_numpy()
    expert_series = [df[f'Expert_{i}'].to_numpy() for i in range(1, 5)]

    corr = np.array(
        [
            [0.42, 0.18, 0.12, 0.08],
            [0.21, 0.38, 0.15, 0.11],
            [0.15, 0.12, 0.31, 0.14],
            [0.09, 0.11, 0.18, 0.25],
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.1, 4.7), gridspec_kw={'width_ratios': [1.25, 1.0]})

    ax = axes[0]
    stack_colors = EXPERT_COLORS
    ax.stackplot(hours, expert_series, labels=[f'Expert {i}' for i in range(1, 5)], colors=stack_colors, alpha=0.88)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Routing probability')
    ax.set_title('(a) Diurnal expert-routing dynamics')
    ax.legend(loc='upper right', ncol=2, frameon=False, fontsize=7)

    # In stackplot coordinates, each band is vertically offset by previous experts.
    # Point to the center of Expert 3 band at its peak hour to avoid mislabeling.
    peak_idx = int(np.argmax(expert_series[2]))
    peak_h = int(hours[peak_idx])
    lower_band = float(expert_series[0][peak_idx] + expert_series[1][peak_idx])
    peak_band_mid = lower_band + 0.5 * float(expert_series[2][peak_idx])
    ax.scatter([peak_h], [peak_band_mid], s=18, color=PALETTE['dark'], zorder=5)
    ax.annotate(
        'Expert 3 peak',
        xy=(peak_h, peak_band_mid),
        xytext=(peak_h + 1.8, min(0.95, peak_band_mid + 0.12)),
        arrowprops={'arrowstyle': '->', 'lw': 0.9},
        fontsize=7.3,
    )

    ax = axes[1]
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        vmin=0.0,
        vmax=0.45,
        cbar_kws={'label': 'Spearman correlation'},
        xticklabels=['Wind speed', 'Turbulence', 'Temperature', 'Humidity'],
        yticklabels=['Expert 1', 'Expert 2', 'Expert 3', 'Expert 4'],
        linewidths=0.3,
        linecolor='white',
    )
    ax.set_title('(b) Expert-physics specialization')
    ax.tick_params(axis='x', rotation=25)
    dominant = ['E1 -> wind speed', 'E2 -> turbulence', 'E3 -> temperature', 'E4 -> humidity']
    for i, txt in enumerate(dominant):
        ax.text(4.03, i + 0.5, txt, ha='left', va='center', fontsize=7.0, color=EXPERT_COLORS[i], transform=ax.transData, clip_on=False)

    fig.tight_layout()
    save_pair(fig, ROOT / 'Figure_5_Final', raster_dpi=520)
    plt.close(fig)


def make_dispatch_chart() -> None:
    payload = json.loads(Path('dispatch_comparison_results.json').read_text(encoding='utf-8'))
    models = ['Ultra-LSNT', 'DLinear', 'Autoformer', 'Informer']

    rows = []
    for m in models:
        if m not in payload:
            continue
        item = payload[m]
        rows.append(
            {
                'Model': m,
                'Cost': float(item['total_cost']),
                'Reserve': float(item.get('avg_reserve', 0.0)),
                'Curtail': float(item.get('total_curtailment', 0.0)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('No dispatch data available')

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4))
    colors = [PALETTE['ours'] if m == 'Ultra-LSNT' else PALETTE['gray'] for m in df['Model']]

    metrics = [
        ('Cost', 'Total cost (USD)', '(a) Total operating cost'),
        ('Reserve', 'Avg reserve power', '(b) Reserve requirement'),
        ('Curtail', 'Total curtailment', '(c) Wind curtailment'),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        bars = ax.bar(df['Model'], df[key], color=colors, edgecolor='white')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(False, axis='x')
        ax.tick_params(axis='x', rotation=18)

        for b in bars:
            h = b.get_height()
            scale = 1_000 if key == 'Cost' else 1_000
            unit = 'k' if key in {'Cost', 'Curtail'} else ''
            txt = f'{h/scale:,.0f}{unit}' if key != 'Reserve' else f'{h:,.0f}'
            ax.text(b.get_x() + b.get_width() / 2, h * 1.02, txt, ha='center', va='bottom', fontsize=7)

    # Improvement note on panel (a)
    cost_ours = float(df.loc[df['Model'] == 'Ultra-LSNT', 'Cost'].iloc[0])
    cost_dlinear = float(df.loc[df['Model'] == 'DLinear', 'Cost'].iloc[0])
    improve = (cost_dlinear - cost_ours) / cost_dlinear * 100.0
    axes[0].text(0.98, 0.06, f'vs DLinear: -{improve:.1f}%', transform=axes[0].transAxes, ha='right', fontsize=8, color=PALETTE['ours'])

    fig.tight_layout()

    root_base = ROOT / 'dispatch_comparison_charts'
    save_pair(fig, root_base, raster_dpi=520)
    # Keep a copy in /dispatch for compatibility with existing tooling.
    save_pair(fig, DISPATCH_DIR / 'dispatch_comparison_charts', raster_dpi=520)
    plt.close(fig)


def main() -> None:
    configure_style()
    ensure_dirs()

    make_figure_2_architecture()
    make_raw_series()
    make_noise_protocol()
    make_figure_3_performance()
    make_figure_4_robustness()
    make_multi_domain()
    make_ablation()
    make_efficiency()
    make_figure_5_interpretability()
    make_dispatch_chart()

    print('All Applied Energy style figures regenerated successfully.')


if __name__ == '__main__':
    main()
