from __future__ import annotations

import matplotlib.pyplot as plt

from paper_figure_style import OKABE_ITO, ensure_output_dir, save_figure, set_paper_style


set_paper_style()
OUT_DIR = ensure_output_dir("figure")


def plot_victory() -> None:
    noise_levels = [0.00, 0.05, 0.10, 0.15]

    ours_r2 = [0.8728, 0.5267, 0.4657, 0.4035]
    dlinear_r2 = [0.9438, -1.4144, -2.8436, -3.3539]

    fig, ax = plt.subplots(figsize=(6.9, 4.8), dpi=300)
    ax.plot(
        noise_levels,
        ours_r2,
        marker="*",
        markersize=12,
        color=OKABE_ITO["vermillion"],
        linewidth=2.8,
        label="Ultra-LSNT (Ours)",
        zorder=3,
    )
    ax.plot(
        noise_levels,
        dlinear_r2,
        marker="o",
        markersize=7,
        color=OKABE_ITO["blue"],
        linestyle="--",
        linewidth=2.2,
        label="DLinear (Baseline)",
        zorder=2,
    )

    ax.set_xlabel("Impulse Noise Level (Prob)", fontweight="bold")
    ax.set_ylabel(r"R-Squared ($R^2$)", fontweight="bold")
    ax.set_title("Catastrophic Failure of Linear Models under Impulse Noise")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.6, linewidth=1.0)

    y_min = min(min(ours_r2), min(dlinear_r2))
    ax.set_ylim(y_min - 0.25, 1.05)
    ax.grid(axis="x", visible=False)

    ax.text(
        0.06,
        0.62,
        "Ultra-LSNT maintains\npredictive capability",
        fontsize=10,
        color=OKABE_ITO["vermillion"],
        fontweight="bold",
    )
    ax.text(
        0.055,
        y_min + 0.25,
        "DLinear collapses\n($R^2 < 0$)",
        fontsize=10,
        color=OKABE_ITO["blue"],
        fontweight="bold",
    )

    ax.legend(loc="upper right")

    fig.tight_layout()
    save_figure(fig, OUT_DIR / "fig_battle_dlinear_victory")
    plt.close(fig)


if __name__ == "__main__":
    plot_victory()
