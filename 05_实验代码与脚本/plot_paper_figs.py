from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from paper_figure_style import OKABE_ITO, ensure_output_dir
from ae_plot_style import set_ae_style, save_ae


set_ae_style()
OUT_DIR = ensure_output_dir("figure")


def plot_comparison() -> None:
    models = ["LSTM", "MLP", "Transformer", "Ultra-LSNT\n(Ours)"]
    r2_scores = [0.6788, 0.6811, 0.7268, 0.7513]
    colors = ["#C7C7C7", "#AFAFAF", OKABE_ITO["sky_blue"], OKABE_ITO["vermillion"]]

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=300)
    bars = ax.bar(models, r2_scores, color=colors, edgecolor="black", linewidth=1.0, width=0.62)

    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.008,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0.60, 0.80)
    ax.set_ylabel(r"R-Squared ($R^2$)", fontweight="bold")
    ax.set_title("Performance Comparison on GEFCom2012")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    save_ae(fig, "fig_comparison", out_dir=str(OUT_DIR))
    plt.close(fig)


def plot_ablation() -> None:
    strategies = ["Top-1\n(Fast)", "Top-2\n(Ours)", "Top-4\n(Dense)"]
    r2_scores = [0.7402, 0.7513, 0.7450]
    compute_cost = [25, 50, 100]

    fig, ax1 = plt.subplots(figsize=(6.8, 4.6), dpi=300)

    color_r2 = OKABE_ITO["vermillion"]
    ax1.plot(
        strategies,
        r2_scores,
        color=color_r2,
        marker="o",
        linewidth=2.8,
        markersize=8,
        label="Accuracy",
        zorder=3,
    )
    ax1.set_ylabel(r"R-Squared ($R^2$)", color=color_r2, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_r2)
    ax1.set_ylim(0.73, 0.76)

    ax1.annotate(
        "Best Balance",
        xy=(1, 0.7513),
        xytext=(1, 0.7553),
        arrowprops={"facecolor": "black", "shrink": 0.05, "width": 1.0, "headwidth": 8},
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

    ax2 = ax1.twinx()
    color_compute = OKABE_ITO["blue"]
    ax2.bar(
        strategies,
        compute_cost,
        color=color_compute,
        alpha=0.22,
        width=0.40,
        label="Compute Cost",
        zorder=1,
    )
    ax2.set_ylabel("Compute Cost (%)", color=color_compute, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_compute)
    ax2.set_ylim(0, 120)

    ax1.set_title("Ablation Study: Accuracy vs. Efficiency")
    ax1.grid(axis="x", visible=False)

    fig.tight_layout()
    save_ae(fig, "fig_ablation", out_dir=str(OUT_DIR))
    plt.close(fig)


if __name__ == "__main__":
    plot_comparison()
    plot_ablation()
