from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from paper_figure_style import ensure_output_dir, save_figure, set_paper_style


set_paper_style()
OUT_DIR = ensure_output_dir("figure")


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
    alpha: float = 1.0,
    fontsize: int = 10,
) -> tuple[float, float]:
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
        alpha=alpha,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold")
    return (x + w, y + h / 2)


def _draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    style: str = "->",
    color: str = "black",
    alpha: float = 1.0,
    width: float = 1.6,
) -> None:
    arrow = FancyArrowPatch(
        posA=start,
        posB=end,
        arrowstyle=style,
        color=color,
        alpha=alpha,
        linewidth=width,
        mutation_scale=15,
    )
    ax.add_patch(arrow)


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.6), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(1.1, 4, "Time Series\nInput $X$", ha="center", va="center", fontsize=12)
    start_point = (1.9, 4)

    blue_face = "#E6F3FF"
    blue_edge = "#0055AA"
    _draw_box(ax, 3, 5, 2, 1, "Conv (k=3)", face=blue_face, edge=blue_edge, fontsize=10)
    _draw_box(ax, 3, 2, 2, 1, "Conv (k=7)", face=blue_face, edge=blue_edge, fontsize=10)
    _draw_arrow(ax, start_point, (3, 5.5))
    _draw_arrow(ax, start_point, (3, 2.5))

    _draw_box(ax, 5.5, 3.5, 1.7, 1.0, "Concat\n+\nPosEnc", face=blue_face, edge=blue_edge, fontsize=10)
    _draw_arrow(ax, (5, 5.5), (5.5, 4.2))
    _draw_arrow(ax, (5, 2.5), (5.5, 3.8))

    encoder_out = (7.2, 4)

    router_x, router_y = 8.7, 4
    circle = patches.Circle((router_x, router_y), 0.62, facecolor="#FFF2CC", edgecolor="#D6B656", linewidth=2.0)
    ax.add_patch(circle)
    ax.text(router_x, router_y, "Top-2\nRouter", ha="center", va="center", fontsize=9, fontweight="bold")
    _draw_arrow(ax, encoder_out, (router_x - 0.62, router_y))

    active_face = "#FFCCCC"
    active_edge = "#B85450"
    inactive_face = "#F0F0F0"
    inactive_edge = "#999999"

    _draw_box(ax, 10.2, 6.5, 2.2, 0.8, "Expert 1\n(Active)", face=active_face, edge=active_edge, fontsize=10)
    _draw_arrow(ax, (router_x + 0.45, router_y + 0.45), (10.2, 6.9), color=active_edge, width=2.2)

    _draw_box(ax, 10.2, 5.0, 2.2, 0.8, "Expert 2", face=inactive_face, edge=inactive_edge, alpha=0.55, fontsize=10)
    _draw_arrow(ax, (router_x + 0.62, router_y + 0.22), (10.2, 5.4), style="-", color=inactive_edge, alpha=0.28)

    _draw_box(ax, 10.2, 2.2, 2.2, 0.8, "Expert 3\n(Active)", face=active_face, edge=active_edge, fontsize=10)
    _draw_arrow(ax, (router_x + 0.45, router_y - 0.45), (10.2, 2.6), color=active_edge, width=2.2)

    _draw_box(ax, 10.2, 0.7, 2.2, 0.8, "Expert 4", face=inactive_face, edge=inactive_edge, alpha=0.55, fontsize=10)
    _draw_arrow(ax, (router_x + 0.62, router_y - 0.22), (10.2, 1.1), style="-", color=inactive_edge, alpha=0.28)

    sum_x, sum_y = 13.1, 4
    circle_sum = patches.Circle((sum_x, sum_y), 0.42, facecolor="white", edgecolor="black", linewidth=1.6)
    ax.add_patch(circle_sum)
    ax.text(sum_x, sum_y, r"$\Sigma$", ha="center", va="center", fontsize=14)

    _draw_arrow(ax, (12.4, 6.9), (sum_x, sum_y + 0.4), color=active_edge)
    _draw_arrow(ax, (12.4, 2.6), (sum_x, sum_y - 0.4), color=active_edge)

    _draw_arrow(ax, (sum_x + 0.42, sum_y), (14, 4))
    ax.text(14, 4, "Forecast\n$Y$", ha="left", va="center", fontsize=12, fontweight="bold")

    fig.tight_layout()
    save_figure(fig, OUT_DIR / "fig_architecture")
    plt.close(fig)


if __name__ == "__main__":
    plot_architecture()
