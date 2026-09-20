from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

from paper_figure_style import OKABE_ITO, ensure_output_dir, save_figure, set_paper_style


set_paper_style()
OUT_DIR = ensure_output_dir("figure")


# Example:
#   正在测试数据集: Wind (US) (wind_us.csv)...
# We want the dataset name ("Wind (US)") and should not stop at the first '('.
_DATASET_RE = re.compile(r"正在测试数据集[:：]\s*(?P<name>.+)\s+\([^()]+\)\.\.\.")
_SCORE_RE = re.compile(r"噪音\s*(?P<pct>\d+)%\s*:\s*R2\s*=\s*(?P<r2>[-0-9.]+)")


def _parse_universal_log(log_path: Path) -> dict[str, dict[float, float]]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    current: str | None = None
    data: dict[str, dict[float, float]] = {}

    for line in text.splitlines():
        m_ds = _DATASET_RE.search(line)
        if m_ds:
            current = m_ds.group("name").strip()
            data.setdefault(current, {})
            continue

        m_sc = _SCORE_RE.search(line)
        if m_sc and current is not None:
            pct = int(m_sc.group("pct"))
            sigma = pct / 100.0
            r2 = float(m_sc.group("r2"))
            data[current][sigma] = r2

    # Keep only datasets with at least 4 points
    return {k: v for k, v in data.items() if len(v) >= 4}


def plot_from_log(
    *,
    log_path: str | Path = "universal_robustness.log",
    order: tuple[str, ...] = ("Wind (CN)", "Wind (US)", "GEFCom Load", "Air Quality"),
) -> None:
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log: {log_path}")

    data = _parse_universal_log(log_path)
    missing = [k for k in order if k not in data]
    if missing:
        # Fallback: plot whatever exists in sorted order.
        order = tuple(sorted(data.keys()))

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), dpi=300, constrained_layout=True)
    axes = axes.flatten()

    for idx, name in enumerate(order[:4]):
        ax = axes[idx]
        series = data[name]
        xs = sorted(series.keys())
        ys = [series[x] for x in xs]

        ax.plot(xs, ys, marker="o", color=OKABE_ITO["blue"], linewidth=2.6)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel(r"Noise Level ($\sigma$)")
        ax.set_ylabel(r"$R^2$")
        ax.grid(axis="x", visible=False)

        # Panel-specific y-limits for readability
        y_min = min(ys)
        y_max = max(ys)
        pad = max(0.03, (y_max - y_min) * 0.25)
        ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

        decay = (ys[0] - ys[-1]) / ys[0] * 100.0 if ys[0] != 0 else 0.0
        ax.text(
            0.05,
            0.08,
            f"Decay: {decay:.1f}%",
            transform=ax.transAxes,
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "black", "linewidth": 0.6},
        )

    # If fewer than 4 datasets, hide unused panels.
    for j in range(len(order[:4]), 4):
        axes[j].axis("off")

    save_figure(fig, OUT_DIR / "fig_universal_robustness")
    plt.close(fig)


if __name__ == "__main__":
    plot_from_log()
