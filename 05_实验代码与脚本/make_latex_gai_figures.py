from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_figure_style import OKABE_ITO, save_figure, set_paper_style


ROOT = Path(".")


def _load_metrics(path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for model, metrics in data.items():
        out[model] = {k: float(v) for k, v in metrics.items() if k in {"r2", "rmse", "mae"}}
    return out


def make_figure_3(metrics_path: Path = Path("results/metrics_corrected.json")) -> None:
    """
    Figure_3_Final: Overall point-forecast performance comparison on Wind (CN).
    Panels: RMSE / MAE (lower is better) and R^2 (higher is better).
    """
    metrics = _load_metrics(metrics_path)

    models = ["Ultra-LSNT", "DLinear", "Informer", "Transformer", "Autoformer", "LSTM", "MLP"]
    labels = {
        "Ultra-LSNT": "Ultra-LSNT (Ours)",
        "DLinear": "DLinear",
        "Informer": "Informer",
        "Transformer": "Transformer",
        "Autoformer": "Autoformer",
        "LSTM": "LSTM",
        "MLP": "MLP",
    }

    def color_for(model: str) -> str:
        if model == "Ultra-LSNT":
            return OKABE_ITO["vermillion"]
        if model == "DLinear":
            return OKABE_ITO["blue"]
        return "#B0B0B0"

    rmse = {m: metrics[m]["rmse"] for m in models if m in metrics}
    mae = {m: metrics[m]["mae"] for m in models if m in metrics}
    r2 = {m: metrics[m]["r2"] for m in models if m in metrics}

    order_rmse = sorted(rmse, key=rmse.get)
    order_mae = sorted(mae, key=mae.get)
    order_r2 = sorted(r2, key=r2.get, reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.6), dpi=300)

    def barh_panel(ax: plt.Axes, order: list[str], values: dict[str, float], title: str, xlabel: str, fmt: str) -> None:
        y = np.arange(len(order))
        v = [values[m] for m in order]
        ax.barh(
            y,
            v,
            color=[color_for(m) for m in order],
            edgecolor="black",
            linewidth=0.9,
            height=0.70,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([labels[m] for m in order])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.grid(axis="y", visible=False)
        for yi, val in zip(y, v, strict=True):
            ax.text(val * 1.01, yi, fmt.format(val), va="center", ha="left", fontsize=10)

    barh_panel(axes[0], order_rmse, rmse, "(a) RMSE (lower is better)", "RMSE (kW)", "{:,.0f}")
    barh_panel(axes[1], order_mae, mae, "(b) MAE (lower is better)", "MAE (kW)", "{:,.0f}")

    # R^2 panel (avoid value*1.01 scaling near 1.0)
    ax = axes[2]
    y = np.arange(len(order_r2))
    v = [r2[m] for m in order_r2]
    ax.barh(
        y,
        v,
        color=[color_for(m) for m in order_r2],
        edgecolor="black",
        linewidth=0.9,
        height=0.70,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([labels[m] for m in order_r2])
    ax.invert_yaxis()
    ax.set_title("(c) $R^2$ (higher is better)")
    ax.set_xlabel(r"$R^2$", fontweight="bold")
    ax.set_xlim(0.60, 1.00)
    ax.grid(axis="y", visible=False)
    for yi, val in zip(y, v, strict=True):
        ax.text(min(val + 0.008, 0.995), yi, f"{val:.3f}", va="center", ha="left", fontsize=10)

    fig.tight_layout()
    save_figure(fig, ROOT / "Figure_3_Final")
    plt.close(fig)


def make_figure_4() -> None:
    """
    Figure_4_Final: Robustness under Gaussian noise on Wind (CN).
    Uses the values reported in latex改.txt Table (0--20% noise).
    """
    noise = np.array([0.00, 0.05, 0.10, 0.15, 0.20])
    r2 = np.array([0.8728, 0.8719, 0.8687, 0.8629, 0.8552])

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=300)
    ax.plot(
        noise,
        r2,
        color=OKABE_ITO["vermillion"],
        marker="o",
        linewidth=2.6,
        markersize=7,
        label="Ultra-LSNT",
    )
    ax.set_xlabel(r"Noise level ($\sigma$)", fontweight="bold")
    ax.set_ylabel(r"$R^2$", fontweight="bold")
    ax.set_title("Robustness under Gaussian Noise (Wind CN)")
    ax.set_xlim(-0.005, 0.205)
    ax.set_ylim(0.84, 0.88)
    ax.legend(loc="upper right")

    drop_pct = (r2[0] - r2[-1]) / r2[0] * 100.0
    ax.text(
        0.98,
        0.06,
        f"Relative drop @ 20%: {drop_pct:.2f}%",
        fontsize=10,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.92, "pad": 3},
    )

    fig.tight_layout()
    save_figure(fig, ROOT / "Figure_4_Final")
    plt.close(fig)


def make_figure_5(activation_csv: Path = Path("wind_expert_activation.csv")) -> None:
    """
    Figure_5_Final: Interpretability visualization.
    (a) Expert activation heatmap vs hour of day (Wind CN)
    (b) Expert-physics correlation matrix (reported values)
    """
    import pandas as pd
    import seaborn as sns

    df = pd.read_csv(activation_csv)
    # Expect columns: Expert_1..Expert_4, Hour
    experts = ["Expert_1", "Expert_2", "Expert_3", "Expert_4"]
    if not {"Hour", *experts}.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {activation_csv}: {list(df.columns)}")

    heat = df.sort_values("Hour")[["Hour", *experts]].set_index("Hour").T
    heat.index = ["Expert 1", "Expert 2", "Expert 3", "Expert 4"]

    corr = np.array(
        [
            [0.42, 0.18, 0.12, 0.08],
            [0.21, 0.38, 0.15, 0.11],
            [0.15, 0.12, 0.31, 0.14],
            [0.09, 0.11, 0.18, 0.25],
        ],
        dtype=float,
    )
    corr_x = ["Wind speed", "Turbulence", "Temperature", "Humidity"]
    corr_y = ["Expert 1", "Expert 2", "Expert 3", "Expert 4"]

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 4.8), dpi=300, gridspec_kw={"width_ratios": [1.35, 1.0]})

    sns.heatmap(
        heat,
        ax=axes[0],
        cmap="Reds",
        cbar_kws={"label": "Activation probability"},
        linewidths=0.25,
        linecolor="white",
    )
    axes[0].set_title("(a) Diurnal Expert Activation (Wind CN)")
    axes[0].set_xlabel("Hour of day (0–23)", fontweight="bold")
    axes[0].set_ylabel("")

    sns.heatmap(
        corr,
        ax=axes[1],
        cmap="Blues",
        annot=True,
        fmt=".2f",
        vmin=0.0,
        vmax=0.45,
        cbar_kws={"label": "Spearman correlation"},
        linewidths=0.25,
        linecolor="white",
        xticklabels=corr_x,
        yticklabels=corr_y,
    )
    axes[1].set_title("(b) Expert–Physics Correlation")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", labelrotation=30)

    fig.tight_layout()
    save_figure(fig, ROOT / "Figure_5_Final")
    plt.close(fig)


def main() -> None:
    set_paper_style()
    make_figure_3()
    make_figure_4()
    make_figure_5()


if __name__ == "__main__":
    main()
