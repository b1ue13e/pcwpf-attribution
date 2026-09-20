from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def set_ae_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, s: str) -> None:
    ax.text(0.02, 0.96, s, transform=ax.transAxes, va="top", ha="left", fontweight="bold")


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _interp_crossover(x: np.ndarray, y: np.ndarray) -> float | None:
    # y = r2_ultra - r2_lgbm
    for i in range(len(x) - 1):
        y1, y2 = y[i], y[i + 1]
        if y1 == 0:
            return float(x[i])
        if y1 * y2 < 0:
            t = -y1 / (y2 - y1)
            xc = x[i] + t * (x[i + 1] - x[i])
            return float(xc)
    return None


def plot_scissor_2x2_v2(
    out_pdf: str = "figures_out/fig_robustness_scissor_2x2_v2.pdf",
    out_png: str = "figures_out/fig_robustness_scissor_2x2_v2.png",
) -> tuple[Path, Path]:
    set_ae_style()

    noise = np.array([0.00, 0.10, 0.20, 0.30, 0.40], dtype=float)

    # Values from report table (Gaussian noise, Ultra-LSNT vs LightGBM)
    r2_ultra = np.array([0.8728, 0.8623, 0.8327, 0.7860, 0.7246], dtype=float)
    r2_lgbm = np.array([0.9067, 0.8850, 0.8251, 0.7259, 0.6123], dtype=float)

    if not (noise.shape == r2_ultra.shape == r2_lgbm.shape):
        raise ValueError("noise, r2_ultra, r2_lgbm must have same length")

    r2_clean_ultra = 0.8728
    rmse_clean_ultra = 9080.55
    denom = 1.0 - r2_clean_ultra
    if denom <= 0:
        raise ValueError("Invalid clean-point R^2 for RMSE conversion")
    var_y = (rmse_clean_ultra**2) / denom

    # Clamp to avoid tiny negative values from numeric precision.
    rmse_ultra_eq = np.sqrt(np.maximum(0.0, (1.0 - r2_ultra) * var_y))
    rmse_lgbm_eq = np.sqrt(np.maximum(0.0, (1.0 - r2_lgbm) * var_y))

    delta_r2 = r2_ultra - r2_lgbm
    delta_rmse = rmse_lgbm_eq - rmse_ultra_eq  # positive => Ultra has lower RMSE

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.2))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.10, top=0.90, wspace=0.30, hspace=0.38)
    ax1, ax2, ax3, ax4 = axes.ravel()

    c_ultra = "navy"
    c_lgbm = "0.55"

    x_cross_ref = _interp_crossover(noise, delta_r2)
    if x_cross_ref is None:
        x_cross_ref = 0.20

    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvspan(0.30, 0.40, color="0.93", zorder=0)
        ax.axvline(x_cross_ref, color="0.35", linewidth=0.9)
        ax.set_xticks(noise)
        ax.set_xlim(-0.01, 0.41)
        style_axes(ax)

    # (a) R^2
    ax1.plot(noise, r2_ultra, marker="o", linewidth=1.2, color=c_ultra, label="Ultra-LSNT")
    ax1.plot(
        noise,
        r2_lgbm,
        marker="^",
        markerfacecolor="none",
        linewidth=1.2,
        linestyle="--",
        color=c_lgbm,
        label="LightGBM",
    )
    ax1.set_title("R$^2$ under increasing noise", pad=3)
    ax1.set_ylabel("R$^2$ (higher is better)")
    ax1.set_ylim(0.58, 0.94)

    # Keep labels in blank regions to avoid overlap with curves.
    ax1.text(
        0.03,
        0.08,
        f"crossover ε≈{x_cross_ref:.2f}",
        transform=ax1.transAxes,
        fontsize=6.8,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.9),
    )
    ax1.text(
        0.97,
        0.96,
        "advantage regime\nε ≥ 0.30",
        transform=ax1.transAxes,
        fontsize=6.8,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.9),
    )
    panel_label(ax1, "(a)")

    # (b) equivalent RMSE
    ax2.plot(noise, rmse_ultra_eq, marker="o", linewidth=1.2, color=c_ultra)
    ax2.plot(noise, rmse_lgbm_eq, marker="^", markerfacecolor="none", linewidth=1.2, linestyle="--", color=c_lgbm)
    ax2.set_title("Equivalent RMSE (from R$^2$)", pad=3)
    ax2.set_ylabel("RMSE (lower is better)")
    panel_label(ax2, "(b)")

    # (c) delta R^2
    ax3.axhline(0, color="0.35", linewidth=0.9)
    ax3.plot(noise, delta_r2, marker="o", linewidth=1.2, color="0.15")
    ax3.set_title("ΔR$^2$ (Ultra-LSNT − LightGBM)", pad=3)
    ax3.set_ylabel("ΔR$^2$")
    ax3.set_xlabel("Noise level ε")
    panel_label(ax3, "(c)")

    # (d) delta RMSE
    ax4.axhline(0, color="0.35", linewidth=0.9)
    ax4.plot(noise, delta_rmse, marker="o", linewidth=1.2, color="0.15")
    ax4.set_title("ΔRMSE (LightGBM − Ultra-LSNT)", pad=3)
    ax4.set_ylabel("ΔRMSE")
    ax4.set_xlabel("Noise level ε")
    panel_label(ax4, "(d)")

    ax1.set_xlabel("")
    ax2.set_xlabel("")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))

    pdf_path = Path(out_pdf)
    png_path = Path(out_png)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


if __name__ == "__main__":
    pdf, png = plot_scissor_2x2_v2()
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")
