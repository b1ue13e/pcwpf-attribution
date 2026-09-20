from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

from paper_figure_style import OKABE_ITO, ensure_output_dir, save_figure, set_paper_style


set_paper_style()
OUT_DIR = ensure_output_dir("figure")


_LINE_RE = re.compile(
    r"Noise\s+(?P<noise>[0-9.]+)\s*\|\s*Ultra-LSNT:\s*(?P<ultra>[-0-9.]+)\s*vs\s*LightGBM:\s*(?P<lgbm>[-0-9.]+)"
)


def _parse_scores(text: str) -> tuple[list[float], list[float], list[float]]:
    noise: list[float] = []
    ultra: list[float] = []
    lgbm: list[float] = []
    for line in text.splitlines():
        m = _LINE_RE.search(line)
        if not m:
            continue
        noise.append(float(m.group("noise")))
        ultra.append(float(m.group("ultra")))
        lgbm.append(float(m.group("lgbm")))
    return noise, ultra, lgbm


def plot_gbdt_battle(log_path: str | Path = "gbdt_output_fixed.log") -> None:
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Missing {log_path}. Provide a log file containing 'Noise ... Ultra-LSNT ... LightGBM ...'.")

    noise, ultra, lgbm = _parse_scores(log_path.read_text(encoding="utf-8", errors="ignore"))
    if not noise:
        raise RuntimeError(f"No parsable score lines found in {log_path}.")

    fig, ax = plt.subplots(figsize=(6.9, 4.8), dpi=300)
    ax.plot(noise, ultra, marker="o", color=OKABE_ITO["vermillion"], linewidth=2.6, label="Ultra-LSNT (Ours)")
    ax.plot(noise, lgbm, marker="s", color=OKABE_ITO["blue"], linestyle="--", linewidth=2.2, label="LightGBM (Baseline)")

    ax.set_xlabel(r"Gaussian Noise Level ($\sigma$)", fontweight="bold")
    ax.set_ylabel(r"R-Squared ($R^2$)", fontweight="bold")
    ax.set_title("GBDT Battle: Ultra-LSNT vs. LightGBM")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left")

    fig.tight_layout()
    save_figure(fig, OUT_DIR / "fig_gbdt_battle")
    plt.close(fig)


if __name__ == "__main__":
    plot_gbdt_battle()
