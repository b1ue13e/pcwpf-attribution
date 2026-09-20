from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ae_plot_style import save_ae, set_ae_style


def _load_r2_curve(data_path: str) -> np.ndarray:
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing robustness file: {data_path}")

    if p.suffix.lower() == ".npy":
        r2 = np.asarray(np.load(p), dtype=float).reshape(-1)
    elif p.suffix.lower() == ".csv":
        import pandas as pd

        df = pd.read_csv(p)
        col = next((c for c in ["r2", "R2", "R^2"] if c in df.columns), None)
        if col is None:
            raise ValueError(f"{data_path} must contain one of: r2, R2, R^2")
        r2 = df[col].to_numpy(dtype=float).reshape(-1)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}. Use .npy or .csv")

    if r2.size != 5:
        raise ValueError(f"Expected 5 points for noise levels [0,5,10,15,20], got {r2.size}")
    return r2


def plot_windcn_degradation_pattern(
    data_path: str = "robustness_scores_ultra_lsnt.npy",
    noise_levels: tuple[float, ...] = (0.00, 0.05, 0.10, 0.15, 0.20),
    out_name: str = "Figure_5_Final",
    out_dir: str = "figures_out",
) -> None:
    """
    Figure 5:
    Visualize the limited and monotonic degradation of Ultra-LSNT under
    progressively stronger Gaussian perturbations on Wind (CN).
    """
    set_ae_style()

    x = np.asarray(noise_levels, dtype=float)
    r2 = _load_r2_curve(data_path)

    if x.size != r2.size:
        raise ValueError(f"noise_levels size ({x.size}) must match curve size ({r2.size})")

    clean = float(r2[0])
    rel_drop_pct = (1.0 - r2 / clean) * 100.0
    total_drop_pct = float(rel_drop_pct[-1])

    fig, ax1 = plt.subplots(figsize=(6.6, 3.0), constrained_layout=True)
    ax2 = ax1.twinx()

    # Main degradation curve (R^2)
    line = ax1.plot(
        x * 100.0,
        r2,
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        color="#0072B2",
        label="Ultra-LSNT ($R^2$)",
    )[0]
    ax1.fill_between(x * 100.0, r2, clean, color="#0072B2", alpha=0.10)

    # Relative drop bars to show degradation magnitude
    bars = ax2.bar(
        x * 100.0,
        rel_drop_pct,
        width=2.3,
        color="#D55E00",
        alpha=0.20,
        label="Relative drop (%)",
    )

    ax1.set_xlabel("Gaussian noise level (%)")
    ax1.set_ylabel("$R^2$")
    ax2.set_ylabel("Relative drop (%)")

    ax1.set_xticks(x * 100.0)
    ax1.set_xticklabels([f"{int(v)}%" for v in x * 100.0])
    ax1.set_xlim(-1.0, 21.0)

    y_margin = 0.0025
    ax1.set_ylim(float(r2.min()) - y_margin, float(r2.max()) + y_margin)
    ax2.set_ylim(0.0, max(2.5, float(rel_drop_pct.max()) * 1.35))

    ax1.grid(alpha=0.25, axis="y")
    ax2.grid(False)

    ax1.set_title("Figure 5: Degradation pattern under progressive perturbations")
    ax1.text(
        0.98,
        0.07,
        f"Total drop @20% noise: {total_drop_pct:.2f}%",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="0.4", boxstyle="round,pad=0.25", alpha=0.9),
    )

    handles = [line, bars]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper right", frameon=False)

    save_ae(fig, out_name, out_dir=out_dir, dpi_mixed=600)
    plt.close(fig)


if __name__ == "__main__":
    plot_windcn_degradation_pattern()
