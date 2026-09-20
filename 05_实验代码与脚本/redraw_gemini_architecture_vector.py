from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
OUT_BASE = ROOT / "ultra_lsnt_architecture_redraw_vector"
TEXT_SCALE = 0.48


def configure_style(*, pure_path: bool = False) -> None:
    svg_fonttype = "path" if pure_path else "none"
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Segoe UI", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,  # TrueType embedding for PDF
            "ps.fonttype": 42,
            "svg.fonttype": svg_fonttype,
            "figure.dpi": 300,   # Increased from 260 to 300 for sharper rendering
            "savefig.dpi": 1000,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
        }
    )


def t(ax: plt.Axes, x: float, y: float, text: str, **kwargs) -> None:
    base = {"ha": "center", "va": "center", "color": "#222222", "zorder": 20}
    base.update(kwargs)
    if "fontsize" in base and isinstance(base["fontsize"], (int, float)):
        base["fontsize"] = base["fontsize"] * TEXT_SCALE
    ax.text(x, y, text, **base)


def rbox(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fc: str,
    ec: str,
    lw: float = 1.6,
    rounding: float = 0.14,
    shadow: bool = False,
    z: int = 5,
) -> FancyBboxPatch:
    if shadow:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.08, y - 0.08),
                w,
                h,
                boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                facecolor="#C8CCD1",
                edgecolor="none",
                alpha=0.22,
                zorder=z - 1,
            )
        )
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arr(
    ax: plt.Axes,
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    c: str = "#7A7A7A",
    lw: float = 1.55,
    ls: str = "-",
    ms: float = 18.0,
    z: int = 12,
    alpha: float = 1.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            linestyle=ls,
            color=c,
            alpha=alpha,
            shrinkA=0.0,
            shrinkB=0.0,
            connectionstyle="arc3,rad=0.0",
            zorder=z,
        )
    )


def path_arr(
    ax: plt.Axes,
    pts: list[tuple[float, float]],
    *,
    c: str = "#7A7A7A",
    lw: float = 1.55,
    ls: str = "-",
    ms: float = 18.0,
    alpha: float = 1.0,
    z: int = 11,
) -> None:
    if len(pts) < 2:
        return
    for i in range(len(pts) - 2):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        ax.plot([x0, x1], [y0, y1], color=c, linewidth=lw, linestyle=ls, alpha=alpha, zorder=z)
    arr(ax, pts[-2], pts[-1], c=c, lw=lw, ls=ls, ms=ms, alpha=alpha, z=z + 1)


def draw(
    *,
    out_base: Path = OUT_BASE,
    pure_path: bool = False,
    text_scale: float = TEXT_SCALE,
    fig_width: float = 17.8,
    fig_height: float = 10.0,
    x_min: float = 0.0,
    x_max: float = 26.4,
    y_min: float = 0.0,
    y_max: float = 13.0,
    save_svg: bool = True,
    save_pdf: bool = True,
    save_png: bool = True,
) -> None:
    global TEXT_SCALE
    TEXT_SCALE = text_scale
    configure_style(pure_path=pure_path)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis("off")

    c_line = "#727272"
    c_blue = "#7EA6C7"
    c_blue_fill = "#DCE9F5"
    c_orange = "#D28C44"
    c_orange_fill = "#F5E6D1"
    c_gray_fill = "#EBECEE"
    c_inactive = "#C9CACC"

    # Title
    ax.add_patch(Rectangle((0.40, 11.95), 0.12, 0.84, facecolor="#8D9198", edgecolor="#8D9198", zorder=3))
    t(ax, 0.84, 12.44, "Ultra-LSNT Architecture", ha="left", fontsize=30, fontweight="normal")
    t(ax, 0.84, 12.00, "(Edge-Native Sparse MoE Forecaster)", ha="left", fontsize=20)

    # Input and MA blocks
    in_box = rbox(ax, 0.10, 6.10, 1.95, 1.95, fc="#F0F1F3", ec=c_line, lw=1.7, rounding=0.10, shadow=True, z=6)
    t(ax, 1.08, 7.55, "Input", fontsize=19)
    t(ax, 1.08, 7.05, "X", fontsize=38, fontweight="bold")
    t(ax, 1.08, 6.55, "(L×C)", fontsize=22)

    ma_box = rbox(ax, 2.55, 5.90, 2.05, 2.35, fc="#ECE8E0", ec=c_line, lw=1.7, rounding=0.10, shadow=True, z=6)
    t(ax, 3.58, 7.83, "Moving", fontsize=28)
    t(ax, 3.58, 7.41, "Average", fontsize=28)
    t(ax, 3.58, 6.99, "Decom-", fontsize=28)
    t(ax, 3.58, 6.57, "position", fontsize=28)
    t(ax, 3.58, 6.14, "(MA)", fontsize=30)
    arr(ax, (2.05, 7.08), (2.53, 7.08), c=c_line, lw=1.7, ms=22)

    # Branch split
    split_x, split_y = 5.10, 7.08
    arr(ax, (4.60, split_y), (5.06, split_y), c="#7FA3C3", lw=1.7, ms=22)
    ax.plot([split_x, split_x], [6.80, 9.95], color="#7FA3C3", linewidth=1.7, zorder=8)

    # Trend branch
    t(ax, 7.85, 11.00, "Trend Branch\n(lightweight / low compute)", fontsize=31, linespacing=1.00)
    trend = rbox(ax, 5.85, 9.08, 4.10, 1.35, fc=c_blue_fill, ec=c_blue, lw=1.9, rounding=0.12, shadow=True, z=8)
    t(ax, 7.90, 10.65, r"$\mathbf{X}_{\mathrm{trend}}$", fontsize=43)
    t(ax, 7.90, 10.21, r"-> Tiny MLP / Linear Head", fontsize=30)
    t(ax, 7.90, 9.79, r"-> $\hat{y}_{\mathrm{trend}}$", fontsize=40)
    arr(ax, (split_x, 9.95), (5.83, 9.95), c="#7FA3C3", lw=1.7, ms=22)

    # Seasonal branch
    t(ax, 7.82, 8.54, "Seasonal Branch\n(hard / nonlinear)", fontsize=31, linespacing=1.00)
    season = rbox(ax, 5.85, 2.05, 4.55, 5.85, fc=c_orange_fill, ec=c_orange, lw=1.9, rounding=0.18, shadow=True, z=7)

    ax.add_patch(Rectangle((6.50, 7.15), 2.30, 0.25, facecolor="#D8DBDF", edgecolor=c_line, linewidth=1.3, zorder=10))
    for x in [7.00, 7.50, 8.00]:
        ax.plot([x, x], [7.15, 7.40], color=c_line, linewidth=1.1, zorder=11)
    t(ax, 9.10, 7.30, "...", fontsize=34, fontweight="bold")
    ax.add_patch(Rectangle((9.45, 7.15), 0.55, 0.25, facecolor="#D8DBDF", edgecolor=c_line, linewidth=1.3, zorder=10))

    conv1 = rbox(ax, 6.45, 5.37, 3.40, 1.34, fc="#D9DBDE", ec="#80858C", lw=1.7, rounding=0.16, shadow=True, z=9)
    t(ax, 8.15, 6.02, r"Conv1D ($k_s$=3)", fontsize=34)
    t(ax, 8.15, 5.57, "Short-term", fontsize=36)

    conv2 = rbox(ax, 6.45, 3.00, 3.40, 1.34, fc="#D9DBDE", ec="#80858C", lw=1.7, rounding=0.16, shadow=True, z=9)
    t(ax, 8.15, 3.66, r"Conv1D ($k_m$=7)", fontsize=34)
    t(ax, 8.15, 3.21, "Mid-term", fontsize=36)

    t(ax, 8.15, 2.42, "Multi-Scale\nTemporal\nEmbedding", fontsize=34, linespacing=1.02)
    t(ax, 7.82, 1.12, "Seasonal Branch\n(hard / nonlinear)", fontsize=31, linespacing=1.00)

    # Seasonal input paths
    arr(ax, (5.10, 7.08), (6.45, 7.08), c="#7FA3C3", lw=1.7, ms=22)
    ax.plot([5.10, 5.10], [7.08, 3.74], color=c_orange, linewidth=1.7, zorder=8)
    arr(ax, (5.10, 3.74), (6.45, 3.74), c=c_orange, lw=1.7, ms=22)

    # Fusion point and text
    fusion = (10.90, 6.46)
    ax.add_patch(Circle(fusion, 0.15, facecolor="#D5D6D8", edgecolor=c_line, linewidth=1.6, zorder=13))
    path_arr(ax, [(9.85, 6.03), (10.88, 6.48)], c=c_line, lw=1.6, ms=22)
    path_arr(ax, [(9.85, 3.67), (10.88, 6.48)], c=c_line, lw=1.6, ms=22)
    # Trend down connector into fusion (aligned with outer frame intersection).
    ax.plot([10.90, 10.90], [10.43, 6.78], color=c_line, linewidth=1.6, zorder=8)
    arr(ax, (10.90, 6.78), (10.90, 6.48), c=c_line, lw=1.6, ms=22)
    t(ax, 12.35, 7.20, "Fusion /", fontsize=39)
    t(ax, 12.35, 6.72, "Projection", fontsize=39)
    t(ax, 12.35, 6.10, r"-> $\mathbf{E}$", fontsize=46)

    # Fusion to router line
    arr(ax, (11.05, 6.46), (13.95, 6.46), c=c_line, lw=1.7, ms=22)

    # Outer frame with open switch gap at bottom (shifted left to match reference geometry).
    xL, xR, yB, yT = 10.90, 20.95, 2.05, 9.70
    ax.plot([xL, xL], [yB, yT], color=c_line, linewidth=1.7, zorder=4)
    ax.plot([xL, xR], [yT, yT], color=c_line, linewidth=1.7, zorder=4)
    ax.plot([xR, xR], [yB, yT], color=c_line, linewidth=1.7, zorder=4)
    ax.plot([xL, 18.25], [yB, yB], color=c_line, linewidth=1.7, zorder=4)
    ax.plot([19.18, xR], [yB, yB], color=c_line, linewidth=1.7, zorder=4)

    # Inner MoE frame + title
    rbox(ax, 13.95, 2.95, 7.45, 5.95, fc="#EEF0F2", ec="#8D8F94", lw=1.7, rounding=0.16, shadow=True, z=5)
    t(ax, 17.68, 9.30, "Sparse MoE Encoder", fontsize=38)

    # Router diamond + icon
    router = Polygon(
        [[14.20, 6.46], [14.85, 7.11], [15.50, 6.46], [14.85, 5.81]],
        closed=True,
        facecolor="#D4D7DB",
        edgecolor=c_line,
        linewidth=1.6,
        zorder=14,
    )
    ax.add_patch(router)
    ax.add_patch(Circle((14.74, 6.46), 0.14, facecolor="#777B81", edgecolor=c_line, linewidth=1.0, zorder=15))
    arr(ax, (14.76, 6.46), (15.24, 6.46), c="#656A71", lw=1.2, ms=14, z=16)
    arr(ax, (14.77, 6.47), (15.10, 6.87), c="#656A71", lw=1.2, ms=14, z=16)
    arr(ax, (14.77, 6.45), (15.12, 6.08), c="#656A71", lw=1.2, ms=14, z=16)
    arr(ax, (14.80, 6.47), (15.00, 7.00), c="#656A71", lw=1.1, ms=12, z=16)

    t(ax, 15.02, 5.30, "Router\n(Softmax)\n-> Top-K", fontsize=38, linespacing=1.00)
    t(ax, 16.25, 3.98, "Top-K Experts\nActivated\n(Sparse Routing)", fontsize=24, fontstyle="italic", linespacing=1.00)
    ax.add_patch(
        FancyArrowPatch(
            (15.20, 4.05),
            (15.58, 4.54),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            linestyle=(0, (1.5, 2.0)),
            color="#B6B8BC",
            connectionstyle="arc3,rad=-0.30",
            zorder=13,
        )
    )

    # Experts
    def expert(x: float, y: float, txt: str, active: bool) -> FancyBboxPatch:
        return rbox(
            ax,
            x,
            y,
            1.15,
            0.98,
            fc="#F7EFD9" if active else "#EFF0F2",
            ec=c_orange if active else c_inactive,
            lw=1.8,
            rounding=0.15,
            shadow=False,
            z=10,
        )

    ex = 17.35
    e1 = expert(ex, 7.60, "E1", True)
    e2 = expert(ex, 6.38, "E2", False)
    e3 = expert(ex, 5.18, "E3", True)
    e4 = expert(ex, 3.98, "E4", False)
    en = expert(ex, 2.48, "EN", True)
    t(ax, ex + 0.58, 8.09, "E1", fontsize=35)
    t(ax, ex + 0.58, 6.87, "E2", fontsize=35, color="#B8BABC")
    t(ax, ex + 0.58, 5.67, "E3", fontsize=35)
    t(ax, ex + 0.58, 4.47, "E4", fontsize=35, color="#B8BABC")
    t(ax, ex + 0.58, 3.42, "...", fontsize=38)
    t(ax, ex + 0.58, 2.97, "EN", fontsize=35)

    sum_c = (20.05, 6.46)
    ax.add_patch(Circle(sum_c, 0.44, facecolor="#E4E6E9", edgecolor=c_line, linewidth=1.7, zorder=13))
    t(ax, sum_c[0], sum_c[1], r"$\Sigma$", fontsize=42)

    # Active paths
    arr(ax, (15.47, 6.46), (17.34, 8.09), c=c_orange, lw=1.6, ms=20)
    arr(ax, (15.47, 6.46), (17.34, 5.67), c=c_orange, lw=1.6, ms=20)
    arr(ax, (15.47, 6.46), (17.34, 2.97), c=c_orange, lw=1.6, ms=20)
    path_arr(ax, [(18.50, 8.09), (19.20, 8.09), (19.80, 6.86)], c=c_orange, lw=1.6, ms=20)
    arr(ax, (18.50, 5.67), (19.62, 5.67), c=c_orange, lw=1.6, ms=20)
    path_arr(ax, [(18.50, 2.97), (19.20, 2.97), (19.75, 5.35)], c=c_orange, lw=1.6, ms=20)

    # Inactive dashed paths
    arr(ax, (15.47, 6.46), (17.34, 6.87), c=c_inactive, lw=1.45, ls="--", ms=19, alpha=0.95)
    arr(ax, (15.47, 6.46), (17.34, 4.47), c=c_inactive, lw=1.45, ls="--", ms=19, alpha=0.95)
    path_arr(ax, [(18.50, 6.87), (19.20, 6.87), (19.80, 6.72)], c=c_inactive, lw=1.45, ls="--", ms=19, alpha=0.95)
    path_arr(ax, [(18.50, 4.47), (19.20, 4.47), (19.80, 6.10)], c=c_inactive, lw=1.45, ls="--", ms=19, alpha=0.95)

    # Weighted sum label
    t(ax, 20.78, 5.72, "Weighted", fontsize=37)
    t(ax, 20.78, 5.20, "Sum", fontsize=37)
    t(ax, 20.78, 4.68, r"-> $\mathbf{h}_{\mathrm{t}}$", fontsize=44)

    # Sum to prediction
    arr(ax, (20.49, 6.46), (21.62, 6.46), c=c_line, lw=1.7, ms=22)

    pred = rbox(ax, 21.65, 5.66, 1.95, 2.05, fc="#F0F1F3", ec=c_line, lw=1.7, rounding=0.12, shadow=True, z=9)
    t(ax, 22.62, 7.30, "Prediction", fontsize=35)
    t(ax, 22.62, 6.82, "Head", fontsize=35)
    t(ax, 22.62, 6.31, r"->", fontsize=35)
    t(ax, 22.62, 5.98, r"$\hat{y}_{\mathrm{season}}$", fontsize=44)
    arr(ax, (23.60, 7.08), (24.32, 7.08), c=c_line, lw=1.7, ms=22)

    # Top trend skip line to final plus
    ax.plot([10.40, 24.75], [10.20, 10.20], color=c_line, linewidth=1.7, zorder=7)

    # Add "Skip Path" label for the green bypass line (ensure font is Arial/Helvetica)
    t(ax, 17.5, 10.35, "Skip Path", fontsize=26, color="#2E7D32", fontweight="bold")

    plus_center = (24.75, 7.15)
    ax.add_patch(Circle((24.82, 7.03), 0.48, facecolor="#C9CDD3", edgecolor="none", alpha=0.25, zorder=6))
    ax.add_patch(Circle(plus_center, 0.48, facecolor="#E8EAEC", edgecolor=c_line, linewidth=1.7, zorder=10))
    t(ax, plus_center[0], plus_center[1], "+", fontsize=62)
    arr(ax, (24.75, 10.20), (24.75, 7.63), c=c_line, lw=1.7, ms=22)

    t(ax, 24.75, 6.04, "Final", fontsize=38)
    t(ax, 24.75, 5.47, "Forecast", fontsize=38)
    t(ax, 24.75, 4.87, r"$\hat{y}$", fontsize=52)

    # Jump gate: open switch (no enclosing line over switch).
    ax.add_patch(Circle((18.35, yB), 0.11, facecolor="#F2F2F2", edgecolor=c_line, linewidth=1.6, zorder=14))
    ax.add_patch(Circle((19.15, yB), 0.11, facecolor="#F2F2F2", edgecolor=c_line, linewidth=1.6, zorder=14))
    ax.plot([18.48, 18.90], [2.15, 2.48], color="#46484C", linewidth=1.7, zorder=15)
    t(ax, 18.75, 1.55, "Jump Gate", fontsize=38)
    t(ax, 18.75, 1.05, "(Conditional Skip)", fontsize=32)
    rbox(ax, 19.45, 0.72, 2.25, 0.92, fc="#E4E6E9", ec="none", lw=0.0, rounding=0.18, shadow=False, z=4)
    t(ax, 20.58, 1.18, "Conditional", fontsize=28, fontstyle="italic")
    t(ax, 20.58, 0.78, "Execution", fontsize=28, fontstyle="italic")

    # Right vertical connector from jump line up to the line before prediction head.
    arr(ax, (20.95, yB), (20.95, 6.46), c=c_line, lw=1.7, ms=22)

    # Edge-friendly badge (rounded base + top-right circular lobe for icons).
    bx, by, bw, bh = 21.35, 0.12, 4.20, 1.54
    badge = rbox(ax, bx, by, bw, bh, fc="#E6E8EA", ec="none", lw=0.0, rounding=0.22, shadow=True, z=3)
    # Lobe shadow + body
    ax.add_patch(Circle((25.84 + 0.08, 1.80 - 0.08), 0.62, facecolor="#C8CCD1", edgecolor="none", alpha=0.22, zorder=2))
    ax.add_patch(Circle((25.84, 1.80), 0.62, facecolor="#E6E8EA", edgecolor="none", zorder=3))

    t(ax, 21.55, 1.38, "Edge-Friendly", ha="left", fontsize=40, fontweight="bold")
    t(ax, 21.55, 0.95, "• Reduced Average Compute", ha="left", fontsize=31)
    t(ax, 21.55, 0.50, "• Lower Energy-per-Inference", ha="left", fontsize=31)

    ax.add_patch(Circle((25.62, 1.80), 0.29, facecolor="#B9D69A", edgecolor=c_line, linewidth=1.2, zorder=12))
    ax.plot([25.62, 25.62], [1.62, 1.94], color="#3A663B", linewidth=1.4, zorder=13)
    ax.plot([25.62, 25.52], [1.79, 1.87], color="#3A663B", linewidth=1.2, zorder=13)
    ax.plot([25.62, 25.71], [1.75, 1.87], color="#3A663B", linewidth=1.2, zorder=13)

    rbox(ax, 25.98, 1.52, 0.36, 0.47, fc="#C9E3A3", ec=c_line, lw=1.1, rounding=0.05, z=12)
    ax.add_patch(Rectangle((26.11, 2.00), 0.10, 0.05, facecolor=c_line, edgecolor=c_line, zorder=12))
    bolt = Polygon(
        [[26.09, 1.88], [26.17, 1.88], [26.12, 1.78], [26.19, 1.78], [26.07, 1.63], [26.11, 1.74], [26.05, 1.74]],
        closed=True,
        facecolor="#4C6B2E",
        edgecolor="#4C6B2E",
        linewidth=0.8,
        zorder=13,
    )
    ax.add_patch(bolt)

    fig.subplots_adjust(left=0.01, right=0.995, top=0.99, bottom=0.03)
    if save_svg:
        fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    if save_pdf:
        fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    if save_png:
        fig.savefig(f"{out_base}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redraw Ultra-LSNT architecture as vector graphics.")
    parser.add_argument(
        "--out-base",
        type=str,
        default=str(OUT_BASE),
        help="Output path prefix without extension.",
    )
    parser.add_argument(
        "--pure-path",
        action="store_true",
        help="Convert text to paths in SVG output for path-only editing.",
    )
    parser.add_argument("--text-scale", type=float, default=TEXT_SCALE, help="Global text scale multiplier.")
    parser.add_argument("--fig-width", type=float, default=17.8, help="Figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=10.0, help="Figure height in inches.")
    parser.add_argument("--x-min", type=float, default=0.0, help="Axes x minimum.")
    parser.add_argument("--x-max", type=float, default=26.4, help="Axes x maximum.")
    parser.add_argument("--y-min", type=float, default=0.0, help="Axes y minimum.")
    parser.add_argument("--y-max", type=float, default=13.0, help="Axes y maximum.")
    args = parser.parse_args()

    draw(
        out_base=Path(args.out_base),
        pure_path=args.pure_path,
        text_scale=args.text_scale,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
    )
