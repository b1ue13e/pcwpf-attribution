from __future__ import annotations

import json
import re
import shutil
from io import StringIO
from pathlib import Path
from typing import Any
import zipfile

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd
try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None

from paper_figure_style import (
    ART_DPI,
    FigureBundle,
    MODEL_STYLE,
    OKABE_ITO,
    REFERENCE_FIGURE_PALETTE,
    add_panel_label,
    bundle_manifest_row,
    canonical_dir,
    figure_size,
    render_pdf_preview,
    save_figure_bundle,
    set_paper_style,
    style_axes,
    write_manifest,
)


ROOT = Path(__file__).resolve().parent
SUPP = ROOT / "results" / "supplementary_evidence"
ADVISOR_EVID = ROOT / "advisor_package_20260311" / "03_current_manuscript_evidence"
MAIN_TEX = ROOT / "main_PI-MoE_final_AE_ready.tex"

MANUAL_LOCKED_PDFS: dict[str, dict[str, str]] = {
    "fig5_deployment_pareto": {
        "pdf": "paper_figures_canonical/main_text/fig5_deployment_pareto.pdf",
        "label": "Figure 5",
        "section": "main_text",
    },
    "fig7_lambda2_frontier": {
        "pdf": "paper_figures_canonical/main_text/fig7_lambda2_frontier.pdf",
        "label": "Figure 7",
        "section": "main_text",
    },
    "fig_app_tau_sweep_pareto": {
        "pdf": "paper_figures_canonical/appendix/fig_app_tau_sweep_pareto.pdf",
        "label": "fig_app_tau_sweep_pareto",
        "section": "appendix",
    },
    "fig_app_deploy_robustness": {
        "pdf": "paper_figures_canonical/appendix/fig_app_deploy_robustness.pdf",
        "label": "fig_app_deploy_robustness",
        "section": "appendix",
    },
    "fig_app_efficiency": {
        "pdf": "paper_figures_canonical/appendix/fig_app_efficiency.pdf",
        "label": "fig_app_efficiency",
        "section": "appendix",
    },
    "fig_app_joint_nkt_pareto_3d": {
        "pdf": "paper_figures_canonical/deprecated_figures/fig_app_joint_nkt_pareto_3d.pdf",
        "label": "fig_app_joint_nkt_pareto_3d",
        "section": "appendix",
    },
    "fig_app_interpretability": {
        "pdf": "paper_figures_canonical/appendix/fig_app_interpretability.pdf",
        "label": "fig_app_interpretability",
        "section": "appendix",
    },
    "fig_app_routing_heatmap": {
        "pdf": "paper_figures_canonical/appendix/fig_app_routing_heatmap.pdf",
        "label": "fig_app_routing_heatmap",
        "section": "appendix",
    },
    "fig_app_g2_deployment_pareto_palette": {
        "pdf": "paper_figures_canonical/appendix/fig_app_g2_deployment_pareto_palette.pdf",
        "label": "fig_app_g2_deployment_pareto_palette",
        "section": "appendix",
    },
    "fig_app_g2_deploy_robustness_palette": {
        "pdf": "paper_figures_canonical/appendix/fig_app_g2_deploy_robustness_palette.pdf",
        "label": "fig_app_g2_deploy_robustness_palette",
        "section": "appendix",
    },
    "fig_app_g2_efficiency_palette": {
        "pdf": "paper_figures_canonical/appendix/fig_app_g2_efficiency_palette.pdf",
        "label": "fig_app_g2_efficiency_palette",
        "section": "appendix",
    },
}

GAUSSIAN_FULL_SWEEP = {
    "Ultra-LSNT": [(0.00, 0.8728), (0.15, 0.8687), (0.30, 0.8552), (0.45, 0.7866), (0.60, 0.7263)],
    "iTransformer": [(0.00, 0.9012), (0.15, 0.8993), (0.30, 0.8937), (0.45, 0.8844), (0.60, 0.8717)],
    "TimeMixer": [(0.00, 0.8886), (0.15, 0.8847), (0.30, 0.8738), (0.45, 0.8566), (0.60, 0.8337)],
    "DLinear": [(0.00, 0.9379), (0.15, 0.7814), (0.30, 0.2952), (0.45, -0.4659), (0.60, -1.5790)],
}

DEPLOYMENT_POINTS = [
    {"label": "Full / PyTorch FP32", "family": "Full", "backend": "PyTorch", "precision": "FP32", "latency": 6.69, "memory": 16.70},
    {"label": "Full / PyTorch INT8", "family": "Full", "backend": "PyTorch", "precision": "INT8", "latency": 6.27, "memory": 2.53},
    {"label": "Full / No-Skip", "family": "Full", "backend": "Reference", "precision": "Reference", "latency": 3.43, "memory": 0.39},
    {"label": "Lite / PyTorch FP32", "family": "Lite", "backend": "PyTorch", "precision": "FP32", "latency": 1.60, "memory": 0.07},
    {"label": "Lite / PyTorch INT8", "family": "Lite", "backend": "PyTorch", "precision": "INT8", "latency": 2.19, "memory": 0.00},
    {"label": "Lite / ONNX FP32", "family": "Lite", "backend": "ONNX", "precision": "FP32", "latency": 0.67, "memory": 0.75},
]

TAU_SWEEP_AUDIT = [
    {"tau": 0.3, "tag": "Edge-frugal", "R2": 0.8661, "RMSE": 9314.40, "skip_rate_pct": 33.38, "cpu_latency_ms": 3.621},
    {"tau": 0.5, "tag": "Default", "R2": 0.8728, "RMSE": 9081.66, "skip_rate_pct": 0.00, "cpu_latency_ms": 3.628},
    {"tau": 0.7, "tag": "Conservative", "R2": 0.8728, "RMSE": 9081.66, "skip_rate_pct": 0.00, "cpu_latency_ms": 3.766},
    {"tau": 0.9, "tag": "Strict", "R2": 0.8728, "RMSE": 9081.66, "skip_rate_pct": 0.00, "cpu_latency_ms": 3.665},
]

DM_AUDIT_ROWS = [
    {"baseline": "PatchTST", "dm_stat": -5.90, "p_value": 1.82e-9},
    {"baseline": "TimeMixer", "dm_stat": 3.45, "p_value": 2.83e-4},
    {"baseline": "iTransformer", "dm_stat": 5.12, "p_value": 3.07e-7},
]

EXPERT_COLORS = [
    REFERENCE_FIGURE_PALETTE["rose"],
    REFERENCE_FIGURE_PALETTE["soft_pink"],
    REFERENCE_FIGURE_PALETTE["dusty_blue"],
    REFERENCE_FIGURE_PALETTE["lavender"],
]


def _reference_sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "ae_reference_seq",
        ["#FBF8FA", "#E7D7E3", REFERENCE_FIGURE_PALETTE["lavender"], REFERENCE_FIGURE_PALETTE["dusty_blue"]],
    )


def _reference_diverging_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "ae_reference_div",
        [REFERENCE_FIGURE_PALETTE["dusty_blue"], "#FFFFFF", REFERENCE_FIGURE_PALETTE["rose"]],
    )


def _prefer_path(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    joined = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find any candidate path:\n{joined}")


def _rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str,
    edge: str,
    fontsize: int = 8,
    alpha: float = 1.0,
    weight: str = "bold",
) -> tuple[float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, color="#1F2937")
    return x + w, y + h / 2


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#6B7280",
    lw: float = 1.1,
    style: str = "->",
    alpha: float = 1.0,
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=lw,
            color=color,
            alpha=alpha,
            connectionstyle=connectionstyle,
        )
    )


def _normalize_segment(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _select_segment(arr: np.ndarray, *, length: int, mode: str) -> tuple[np.ndarray, int]:
    arr = np.asarray(arr, dtype=float)
    if arr.size <= length:
        return arr, 0

    s = pd.Series(arr)
    if mode == "volatile":
        score = s.diff().abs().rolling(length, min_periods=length // 2).sum()
    elif mode == "peaky":
        score = s.rolling(length, min_periods=length // 2).max() - s.rolling(length, min_periods=length // 2).median()
    elif mode == "cyclic":
        score = s.rolling(length, min_periods=length // 2).std()
    else:
        score = s.rolling(length, min_periods=length // 2).mean()

    idx = int(score.fillna(-np.inf).to_numpy().argmax())
    start = max(0, min(idx - length + 1, arr.size - length))
    return arr[start : start + length], start


def _load_load_history_series() -> np.ndarray:
    load_path = ROOT / "GEFCom2012年全球能源预测大赛电力负荷预测数据集" / "Load Forecasting" / "Load_history.csv"
    df = pd.read_csv(load_path)
    df = df[df["zone_id"] == 1].copy()
    hour_cols = [f"h{i}" for i in range(1, 25)]
    values = df[hour_cols].replace({",": ""}, regex=True).astype(float).to_numpy().reshape(-1)
    return values


def _stage_manual_locked_pdf(key: str) -> dict[str, Any]:
    meta = MANUAL_LOCKED_PDFS[key]
    pdf_path = ROOT / meta["pdf"]
    if not pdf_path.exists():
        raise FileNotFoundError(f"Manual-locked figure is missing: {pdf_path}")

    out_base = pdf_path.with_suffix("")
    png_path, tif_path = render_pdf_preview(pdf_path, out_base)
    if png_path is None:
        png_path = out_base.with_suffix(".png")
    if tif_path is None:
        tif_path = out_base.with_suffix(".tif")

    width_mm = 0.0
    height_mm = 0.0
    if png_path.exists():
        try:
            from PIL import Image

            with Image.open(png_path) as img:
                width_px, height_px = img.size
            width_mm = width_px / ART_DPI["preview"] * 25.4
            height_mm = height_px / ART_DPI["preview"] * 25.4
        except Exception:
            pass

    return {
        "label": meta["label"],
        "section": meta["section"],
        "source": "manual_locked",
        "base_path": str(out_base),
        "pdf_path": str(pdf_path),
        "png_path": str(png_path),
        "tif_path": str(tif_path),
        "art_kind": "combination",
        "width_mm": round(width_mm, 2),
        "height_mm": round(height_mm, 2),
    }


def _load_data_overview_sources() -> list[dict[str, Any]]:
    wind_cn = pd.read_csv(ROOT / "wind_final.csv")["power"].astype(float).to_numpy()
    wind_us = pd.read_csv(ROOT / "wind_us.csv")["power (MW)"].astype(float).to_numpy()
    air_quality = pd.read_csv(ROOT / "air_quality_ready.csv")["AQI"].astype(float).to_numpy()
    gefcom = _load_load_history_series()

    return [
        {
            "label": "Wind (CN)",
            "series": wind_cn,
            "length": 96 * 7,
            "mode": "volatile",
            "annotation": "Clustered ramps",
            "ann_xy": (0.60, 0.78),
            "ann_text": (0.08, 0.88),
        },
        {
            "label": "Wind (US)",
            "series": wind_us,
            "length": 12 * 24 * 3,
            "mode": "cyclic",
            "annotation": "Persistent gust shelf",
            "ann_xy": (0.58, 0.65),
            "ann_text": (0.08, 0.86),
        },
        {
            "label": "Air Quality",
            "series": air_quality,
            "length": 24 * 10,
            "mode": "peaky",
            "annotation": "Decay spikes",
            "ann_xy": (0.70, 0.76),
            "ann_text": (0.10, 0.88),
        },
        {
            "label": "GEFCom Load",
            "series": gefcom,
            "length": 24 * 14,
            "mode": "cyclic",
            "annotation": "Strong diurnal cycle",
            "ann_xy": (0.52, 0.66),
            "ann_text": (0.08, 0.88),
        },
    ]


def _load_fault_anchor_summary() -> pd.DataFrame:
    path = _prefer_path(
        SUPP / "fairness_closure_fullfault_hard_table.csv",
        ADVISOR_EVID / "table2_robustness" / "fairness_closure_fullfault_hard_table.csv",
    )
    return pd.read_csv(path)


def _load_physics_closure() -> pd.DataFrame:
    path = _prefer_path(
        SUPP / "physics_closure.csv",
        ADVISOR_EVID / "table4_hardware_safety" / "physics_closure.csv",
    )
    return pd.read_csv(path)


def _load_windcn_context_summary() -> pd.DataFrame:
    path = _prefer_path(
        SUPP / "windcn_context_audit_summary.csv",
        ADVISOR_EVID / "table2_robustness" / "windcn_context_audit_summary.csv",
    )
    return pd.read_csv(path)


def _load_main_hardware_joint() -> pd.DataFrame:
    path = _prefer_path(
        SUPP / "main4_matrix3_hardware_safety_joint.csv",
        ADVISOR_EVID / "table4_hardware_safety" / "main4_matrix3_hardware_safety_joint.csv",
    )
    return pd.read_csv(path)


def _load_decision_summary() -> pd.DataFrame:
    path = _prefer_path(
        SUPP / "decision_aware_robustness_summary.csv",
        ADVISOR_EVID / "table5_dispatch" / "decision_aware_robustness_summary.csv",
    )
    return pd.read_csv(path)


def _pdf_page_size_mm(pdf_path: Path) -> tuple[float, float]:
    try:
        import fitz  # type: ignore

        with fitz.open(pdf_path) as doc:
            page = doc.load_page(0)
            return page.rect.width * 25.4 / 72.0, page.rect.height * 25.4 / 72.0
    except Exception:
        pass

    try:
        from PyPDF2 import PdfReader  # type: ignore

        page = PdfReader(str(pdf_path)).pages[0]
        return float(page.mediabox.width) * 25.4 / 72.0, float(page.mediabox.height) * 25.4 / 72.0
    except Exception:
        return 0.0, 0.0


def _stage_external_pdf(source_path: Path, out_base: Path, *, art_kind: str = "line") -> FigureBundle:
    if not source_path.exists():
        raise FileNotFoundError(f"External figure source not found: {source_path}")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_base.with_suffix(".pdf")
    shutil.copy2(source_path, pdf_path)

    png_path, tif_path = render_pdf_preview(pdf_path, out_base)
    if png_path is None or tif_path is None or not png_path.exists() or not tif_path.exists():
        raise RuntimeError(f"Could not render preview assets for staged PDF: {pdf_path}")

    width_mm, height_mm = _pdf_page_size_mm(pdf_path)
    return FigureBundle(
        base_path=out_base,
        pdf_path=pdf_path,
        png_path=png_path,
        tif_path=tif_path,
        art_kind=art_kind,
        width_mm=width_mm,
        height_mm=height_mm,
    )


def _nondominated_minimize_x_maximize_y(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    keep: list[int] = []
    rows = df.to_dict("records")
    for idx, row in enumerate(rows):
        dominated = False
        for jdx, other in enumerate(rows):
            if idx == jdx:
                continue
            if other[x_col] <= row[x_col] and other[y_col] >= row[y_col] and (other[x_col] < row[x_col] or other[y_col] > row[y_col]):
                dominated = True
                break
        if not dominated:
            keep.append(idx)
    return df.iloc[keep].sort_values(x_col).reset_index(drop=True)


def _cost_bubble_size(cost_m_cny: float) -> float:
    return 28.0 + 6.0 * float(cost_m_cny)


def _shift_annotation(ann: Any, dx_points: float, dy_points: float) -> None:
    x, y = ann.get_position()
    ann.set_position((float(x) + dx_points, float(y) + dy_points))


def _relax_annotations(
    ax: plt.Axes,
    annotations: list[Any],
    *,
    anchors: list[tuple[float, float]] | None = None,
    step_points: float = 2.8,
    max_iter: int = 180,
) -> None:
    if not annotations:
        return

    fig = ax.figure
    anchors = anchors or [tuple(ann.xy) for ann in annotations]
    points_per_px = 72.0 / fig.dpi

    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = []
        for ann in annotations:
            bbox_patch = ann.get_bbox_patch()
            if bbox_patch is not None:
                box = bbox_patch.get_window_extent(renderer=renderer)
            else:
                box = ann.get_window_extent(renderer=renderer)
            boxes.append(box.expanded(1.04, 1.14))
        axes_box = ax.get_window_extent(renderer=renderer)
        deltas = [[0.0, 0.0] for _ in annotations]
        moved = False

        for idx in range(len(annotations)):
            for jdx in range(idx + 1, len(annotations)):
                if not boxes[idx].overlaps(boxes[jdx]):
                    continue
                moved = True
                ci_x = boxes[idx].x0 + boxes[idx].width / 2.0
                ci_y = boxes[idx].y0 + boxes[idx].height / 2.0
                cj_x = boxes[jdx].x0 + boxes[jdx].width / 2.0
                cj_y = boxes[jdx].y0 + boxes[jdx].height / 2.0
                vec_x = ci_x - cj_x
                vec_y = ci_y - cj_y
                if abs(vec_x) < 1.0:
                    vec_x = 1.0 if idx % 2 == 0 else -1.0
                if abs(vec_y) < 1.0:
                    vec_y = 1.0 if idx <= jdx else -1.0
                norm = max(np.hypot(vec_x, vec_y), 1.0)
                push_x = step_points * vec_x / norm
                push_y = step_points * vec_y / norm
                deltas[idx][0] += push_x
                deltas[idx][1] += push_y
                deltas[jdx][0] -= push_x
                deltas[jdx][1] -= push_y

        for idx, anchor in enumerate(anchors):
            anchor_x, anchor_y = ax.transData.transform(anchor)
            if boxes[idx].contains(anchor_x, anchor_y):
                moved = True
                box_x = boxes[idx].x0 + boxes[idx].width / 2.0
                box_y = boxes[idx].y0 + boxes[idx].height / 2.0
                vec_x = box_x - anchor_x
                vec_y = box_y - anchor_y
                if abs(vec_x) < 1.0:
                    vec_x = 1.0
                if abs(vec_y) < 1.0:
                    vec_y = 1.0
                norm = max(np.hypot(vec_x, vec_y), 1.0)
                deltas[idx][0] += step_points * vec_x / norm
                deltas[idx][1] += step_points * vec_y / norm

        for idx, box in enumerate(boxes):
            pad_px = 4.0
            if box.x0 < axes_box.x0 + pad_px:
                moved = True
                deltas[idx][0] += (axes_box.x0 + pad_px - box.x0) * points_per_px
            if box.x1 > axes_box.x1 - pad_px:
                moved = True
                deltas[idx][0] -= (box.x1 - (axes_box.x1 - pad_px)) * points_per_px
            if box.y0 < axes_box.y0 + pad_px:
                moved = True
                deltas[idx][1] += (axes_box.y0 + pad_px - box.y0) * points_per_px
            if box.y1 > axes_box.y1 - pad_px:
                moved = True
                deltas[idx][1] -= (box.y1 - (axes_box.y1 - pad_px)) * points_per_px

        if not moved:
            break

        for ann, (dx_points, dy_points) in zip(annotations, deltas, strict=False):
            if abs(dx_points) < 0.05 and abs(dy_points) < 0.05:
                continue
            _shift_annotation(ann, dx_points, dy_points)


def _adjust_texts(
    ax: plt.Axes,
    texts: list[Any],
    target_x: list[float],
    target_y: list[float],
    *,
    arrow_color: str = "#6B7280",
    force_text: tuple[float, float] = (0.2, 0.2),
    force_static: tuple[float, float] = (0.2, 0.2),
    expand: tuple[float, float] = (1.05, 1.05),
) -> None:
    if not texts:
        return
    if adjust_text is not None:
        adjust_text(
            texts,
            target_x=target_x,
            target_y=target_y,
            force_text=force_text,
            force_static=force_static,
            expand=expand,
            ensure_inside_axes=True,
            prevent_crossings=True,
            ax=ax,
            iter_lim=400,
            arrowprops={"arrowstyle": "-", "color": arrow_color, "lw": 0.5},
        )
        return
    _relax_annotations(ax, texts, anchors=list(zip(target_x, target_y, strict=False)))


def _auto_adjust_texts(
    ax: plt.Axes,
    texts: list[Any],
    anchors: list[tuple[float, float]],
    *,
    arrow_color: str = "#6B7280",
    force: float = 0.20,
    expand: tuple[float, float] = (1.08, 1.12),
) -> None:
    if not texts:
        return

    anchor_x = [float(x) for x, _ in anchors]
    anchor_y = [float(y) for _, y in anchors]
    try:
        adjust_text(
            texts,
            x=anchor_x,
            y=anchor_y,
            target_x=anchor_x,
            target_y=anchor_y,
            force_text=(force, force),
            force_static=(force, force),
            force_pull=(0.06, 0.06),
            expand=expand,
            ensure_inside_axes=True,
            prevent_crossings=True,
            ax=ax,
            time_lim=1.5,
            arrowprops={"arrowstyle": "-", "color": arrow_color, "lw": 0.55},
        )
    except Exception:
        _relax_annotations(ax, texts, anchors=anchors, step_points=2.5)


def _deepen_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    for line in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
        line.set_zorder(0)
        line.set_alpha(0.28)
        line.set_color("#D1D5DB")
        line.set_linestyle("--")


def build_figure1_architecture() -> dict[str, Any]:
    out_base = canonical_dir("main_text") / "fig1_architecture_training_path"
    bundle = _stage_external_pdf(ROOT / "architecture.pdf", out_base, art_kind="line")
    return bundle_manifest_row(bundle, label="Figure 1", section="main_text", source="rebuilt")


def build_figure2_workflow() -> dict[str, Any]:
    out_base = canonical_dir("main_text") / "fig2_dispatch_pipeline"
    bundle = _stage_external_pdf(ROOT / "worflow.pdf", out_base, art_kind="line")
    return bundle_manifest_row(bundle, label="Figure 2", section="main_text", source="rebuilt")


def build_figure3_data_overview() -> dict[str, Any]:
    sources = _load_data_overview_sources()
    fig, axes = plt.subplots(2, 2, figsize=figure_size("double", 92), sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, (ax, meta) in enumerate(zip(axes, sources, strict=False)):
        seg, start = _select_segment(meta["series"], length=meta["length"], mode=meta["mode"])
        norm = _normalize_segment(seg)
        x = np.arange(norm.size)

        ax.plot(x, norm, color=OKABE_ITO["blue"], linewidth=1.1)
        for ref in (0.0, 0.5, 1.0):
            ax.axhline(ref, color="#9CA3AF", linewidth=0.45, linestyle=(0, (3, 2)))

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, norm.size - 1)
        ax.set_title(meta["label"], loc="left", fontsize=9, fontweight="bold")
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_ylabel("Scaled value")
        ax.set_xlabel("Relative time")
        style_axes(ax, grid=False, y_grid_only=False)
        add_panel_label(ax, f"({chr(97 + idx)})")

        ann_x = int(meta["ann_xy"][0] * (norm.size - 1))
        ann_y = float(norm[ann_x])
        ax.annotate(
            meta["annotation"],
            xy=(ann_x, ann_y),
            xycoords="data",
            xytext=meta["ann_text"],
            textcoords="axes fraction",
            fontsize=7.2,
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": "#4B5563"},
            bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "boxstyle": "round,pad=0.2"},
        )

        ax.text(0.02, 0.06, f"slice start = {start}", transform=ax.transAxes, fontsize=6.8, color="#6B7280")

    fig.suptitle("Representative raw-series excerpts reveal heterogeneous dynamics across controller-side transfer tasks", fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_base = canonical_dir("main_text") / "fig3_data_small_multiples"
    bundle = save_figure_bundle(fig, out_base, art_kind="line")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="Figure 3", section="main_text", source="rebuilt")


def build_figure4_robustness() -> dict[str, Any]:
    matched = _load_fault_anchor_summary()
    quartet = ["TimeMixer", "iTransformer", "Ultra-LSNT", "DLinear"]
    matched = matched[matched["model"].isin(quartet)].copy()
    matched["training_mode"] = pd.Categorical(matched["training_mode"], categories=["clean_trained", "fault_augmented"], ordered=True)

    panel_defs = [
        ("Gaussian", "gaussian_R2_mean"),
        ("Drift", "drift_R2_mean"),
        ("Quantization", "quantization_R2_mean"),
    ]
    x_pos = {"clean_trained": 0.0, "fault_augmented": 1.0}
    tick_labels = ["Clean-\ntrained", "Fault-\naugmented"]
    fig, axes = plt.subplots(1, 3, figsize=figure_size("double", 62), sharey=False)
    fig4_widths = {
        "TimeMixer": 1.0,
        "iTransformer": 1.1,
        "Ultra-LSNT": 1.25,
        "DLinear": 1.15,
    }
    fig4_marker_sizes = {
        "TimeMixer": 3.2,
        "iTransformer": 3.4,
        "Ultra-LSNT": 3.8,
        "DLinear": 3.4,
    }

    for ax, (title, value_col) in zip(axes, panel_defs, strict=False):
        ax.axvspan(0.5, 1.15, color="#ECE7EF", alpha=0.60, zorder=0)
        ax.text(0.97, 0.06, "Fault-augmented", transform=ax.transAxes, fontsize=6.7, color="#4B5563", ha="right")
        ax.text(0.03, 0.92, "Severe slice", transform=ax.transAxes, fontsize=6.7, color="#6B7280", ha="left", va="top")

        panel_vals: list[float] = []
        for model in quartet:
            sub = matched[matched["model"] == model].sort_values("training_mode")
            xs = [x_pos[str(mode)] for mode in sub["training_mode"]]
            ys = [float(v) for v in sub[value_col]]
            panel_vals.extend(ys)
            style = MODEL_STYLE[model]
            ax.plot(
                xs,
                ys,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=fig4_widths[model],
                markersize=fig4_marker_sizes[model],
                markeredgewidth=0.45,
                label=model,
                zorder=3,
            )

        ymin = min(panel_vals)
        ymax = max(panel_vals)
        pad = max(0.02, (ymax - ymin) * 0.18)
        ax.set_ylim(max(0.15 if title == "Gaussian" else 0.68, ymin - pad), min(1.0, ymax + pad))
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlim(-0.10, 1.10)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Training condition")
        style_axes(ax, grid=True, y_grid_only=True)

    axes[0].set_ylabel("$R^2$ at severe slice")
    axes[1].set_ylabel("$R^2$ at severe slice")
    axes[2].set_ylabel("$R^2$ at severe slice")
    add_panel_label(axes[0], "(a)")
    add_panel_label(axes[1], "(b)")
    add_panel_label(axes[2], "(c)")
    legend_map: dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=False):
            legend_map.setdefault(label, handle)
    fig.legend(list(legend_map.values()), list(legend_map.keys()), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out_base = canonical_dir("main_text") / "fig4_robustness_matched_control"
    bundle = save_figure_bundle(fig, out_base, art_kind="line")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="Figure 4", section="main_text", source="rebuilt")


def _nondominated_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q["latency"] <= p["latency"] and q["memory"] <= p["memory"] and (q["latency"] < p["latency"] or q["memory"] < p["memory"]):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    return sorted(frontier, key=lambda item: item["latency"])


def _load_jump_gate_stage_summary() -> dict[str, Any]:
    local_path = ROOT / "results" / "sota_search_4090_fullrun_windcn" / "aggressive_seed52" / "main_Wind_CN" / "stage_summary.json"
    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))

    zip_path = ROOT / "cloud_4090_paper_upload_pack_20260323_123152.zip"
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("aggressive_seed52/main_Wind_CN/stage_summary.json"):
                return json.loads(zf.read(name).decode("utf-8"))
    raise FileNotFoundError("Jump Gate calibration summary not found in local results or cloud pack zip.")


def _load_turbine_10seed_repeat() -> tuple[pd.DataFrame, pd.DataFrame]:
    zip_path = ROOT / "turbine_v2_10seeds_bundle.zip"
    with zipfile.ZipFile(zip_path) as zf:
        summary_name = next((name for name in zf.namelist() if name.endswith("10seeds_safety_summary.csv")), None)
        if summary_name is None:
            raise FileNotFoundError("10seeds_safety_summary.csv not found in turbine 10-seed bundle.")
        summary = pd.read_csv(StringIO(zf.read(summary_name).decode("utf-8")))

        deployment_logs: list[pd.DataFrame] = []
        for name in zf.namelist():
            if name.endswith(".csv") and "unified_deployment_logs_turbine_v2_" in name:
                deployment_logs.append(pd.read_csv(StringIO(zf.read(name).decode("utf-8"))))
        if not deployment_logs:
            raise FileNotFoundError("Unified deployment logs not found in turbine 10-seed bundle.")

    deployment = pd.concat(deployment_logs, ignore_index=True)
    return summary, deployment


def build_figure5_deployment_pareto() -> dict[str, Any]:
    return _stage_manual_locked_pdf("fig5_deployment_pareto")

    fig, ax = plt.subplots(figsize=figure_size("one_point_five", 84))

    shape_map = {"PyTorch": "o", "ONNX": "s", "Reference": "D"}
    family_color = {
        "Full": "#E63946",
        "Lite": "#F4A6AE",
        "Reference": "#BDBDBD",
    }
    label_map = {
        "Full / PyTorch FP32": "Ultra-Full / FP32",
        "Full / PyTorch INT8": "Ultra-Full / INT8",
        "Full / No-Skip": "No-skip ref",
        "Lite / PyTorch FP32": "Ultra-Lite / PT FP32",
        "Lite / PyTorch INT8": "Ultra-Lite / PT INT8",
        "Lite / ONNX FP32": "Ultra-Lite / ONNX",
    }
    label_offsets = {
        "Full / PyTorch FP32": (10, -6),
        "Full / PyTorch INT8": (-46, 10),
        "Full / No-Skip": (12, -12),
        "Lite / PyTorch FP32": (10, 10),
        "Lite / PyTorch INT8": (12, 10),
        "Lite / ONNX FP32": (10, -14),
    }

    plot_points: list[dict[str, float | str]] = []
    for point in DEPLOYMENT_POINTS:
        plot_point = dict(point)
        plot_point["memory_plot"] = float(point["memory"])
        plot_point["family_key"] = "Reference" if point["backend"] == "Reference" else point["family"]
        plot_points.append(plot_point)

    ax.add_patch(Rectangle((0.45, 0.0), 1.75, 1.0, facecolor="#E63946", edgecolor="#E63946", linewidth=0.8, alpha=0.08, zorder=1))
    ax.text(0.52, 0.82, "Controller-side\nbudget zone", fontsize=6.5, color="#9D1C28", ha="left", va="top", zorder=2)

    texts: list[Any] = []
    target_x: list[float] = []
    target_y: list[float] = []
    point_lookup: dict[str, tuple[float, float]] = {}
    for point in plot_points:
        family_key = str(point["family_key"])
        color = family_color[family_key]
        marker = shape_map[str(point["backend"])]
        fill = color if family_key == "Full" else "white"
        if family_key == "Lite":
            fill = "#FFF5F6"
        if family_key == "Reference":
            fill = "#F3F4F6"
        size = 90 if family_key == "Full" else 78
        x = float(point["latency"])
        y = float(point["memory_plot"])
        point_lookup[str(point["label"])] = (x, y)
        ax.scatter(x, y, marker=marker, s=size, facecolors=fill, edgecolors=color, linewidth=1.5, zorder=4 if family_key != "Reference" else 2)
        text = ax.text(x, y, label_map[str(point["label"])], fontsize=6.2, color="#374151", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.15}, zorder=6)
        texts.append(text)
        target_x.append(x)
        target_y.append(y)

    path_color = "#B56576"
    path_specs = [
        ("Full / PyTorch FP32", "Full / PyTorch INT8", "INT8 quantization"),
        ("Full / PyTorch INT8", "Lite / PyTorch INT8", "carrier simplification"),
        ("Lite / PyTorch INT8", "Lite / ONNX FP32", "runtime export"),
    ]
    for idx, (src_label, dst_label, step_label) in enumerate(path_specs):
        src = point_lookup[src_label]
        dst = point_lookup[dst_label]
        arrow = FancyArrowPatch(
            src,
            dst,
            arrowstyle="-|>",
            mutation_scale=10.0,
            linewidth=1.35,
            color=path_color,
            alpha=0.95,
            zorder=3,
            connectionstyle="arc3,rad=0.00",
        )
        ax.add_patch(arrow)
        mid_x = (src[0] * dst[0]) ** 0.5
        mid_y = 0.5 * (src[1] + dst[1])
        text = ax.text(
            mid_x,
            mid_y,
            step_label,
            fontsize=6.0,
            color=path_color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 0.12},
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.5)
    ax.set_xlabel("P95 latency (ms, log scale)")
    ax.set_ylabel("Peak memory (MiB, symlog scale)")
    ax.set_xlim(0.45, 8.5)
    ax.set_ylim(0.0, 30.0)
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1.0, 2.0, 4.0, 8.0]))
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.01, 0.1, 1.0, 10.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    style_axes(ax, grid=True, y_grid_only=False)
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.7, linestyle=(0, (3, 2)), alpha=0.75, zorder=0)

    _adjust_texts(ax, texts, target_x, target_y, arrow_color="#9CA3AF", force_text=(0.20, 0.20), force_static=(0.20, 0.20), expand=(1.08, 1.10))

    handles = [
        Line2D([0], [0], color=path_color, linewidth=1.35, label="Deployment optimization path"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#E63946", markeredgecolor="#E63946", label="Ultra-Full"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#FFF5F6", markeredgecolor="#F4A6AE", label="Ultra-Lite"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="#F3F4F6", markeredgecolor="#BDBDBD", label="No-skip reference"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.96, edgecolor="#D1D5DB", fontsize=6.4)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.14)

    out_base = canonical_dir("main_text") / "fig5_deployment_pareto"
    bundle = save_figure_bundle(fig, out_base, art_kind="line")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="Figure 5", section="main_text", source="rebuilt")


def build_figure6_controller_safety_chain() -> dict[str, Any]:
    guard = pd.read_csv(SUPP / "fgmoe_clip_only_control_windcn.csv")
    guard = guard[guard["scenario"].isin(["clean", "gaussian_60pct"])].copy()
    guard["scenario_label"] = guard["scenario"].map({"clean": "Clean", "gaussian_60pct": "Severe Gaussian"})
    guard_variant_map = {
        "Ultra-LSNT (no guardrail)": "Raw",
        "Ultra-LSNT + clip-only (posthoc)": "Clip-only",
        "FG-MoE (full-trained)": "FG-MoE",
    }
    guard["variant"] = guard["model"].map(guard_variant_map)
    variant_order = ["Raw", "Clip-only", "FG-MoE"]
    variant_style = {
        "Raw": {"color": REFERENCE_FIGURE_PALETTE["graphite"], "marker": "o", "linestyle": "-"},
        "Clip-only": {"color": REFERENCE_FIGURE_PALETTE["soft_pink"], "marker": "s", "linestyle": "--"},
        "FG-MoE": {"color": REFERENCE_FIGURE_PALETTE["dusty_blue"], "marker": "D", "linestyle": "-"},
    }

    jump = _load_jump_gate_stage_summary()["jump_gate_calibration"]
    jump_grid = pd.DataFrame(jump["grid"]).sort_values("threshold")
    selected_tau = float(jump["selected_threshold"])
    selected_r2 = float(jump["selected_metrics"]["R2"])
    selected_skip = float(jump["selected_metrics"]["avg_skip_rate"]) * 100.0
    selected_compute = float(jump["selected_metrics"]["avg_compute_ratio"]) * 100.0
    target_skip = float(jump["target_skip_rate"]) * 100.0

    turbine_summary, turbine_logs = _load_turbine_10seed_repeat()
    turbine_summary = turbine_summary[turbine_summary["corruption_type"].isin(["Clean", "Severe Gaussian"])].copy()
    turbine_summary["variant"] = turbine_summary["model_name"].map(
        {
            "Ultra-LSNT Raw Output": "Raw",
            "Ultra-LSNT + Clip-only": "Clip-only",
            "Ultra-LSNT + FG-MoE": "FG-MoE",
        }
    )
    turbine_summary["r2_mean"] = turbine_summary["r2_mean"].astype(float)
    turbine_summary["r2_std"] = turbine_summary["r2_std"].astype(float)
    fg_latency = (
        turbine_logs[(turbine_logs["model_name"] == "Ultra-LSNT + FG-MoE") & (turbine_logs["corruption_type"].isin(["Clean", "Severe Gaussian"]))]
        .groupby("corruption_type", as_index=False)["p95_latency_ms"]
        .mean()
    )
    latency_map = {row["corruption_type"]: float(row["p95_latency_ms"]) for _, row in fg_latency.iterrows()}

    fig = plt.figure(figsize=figure_size("double", 88))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.1, 1.15], wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    x_guard = np.arange(2)
    scenario_order = ["Clean", "Severe Gaussian"]
    for variant in variant_order:
        sub = guard[guard["variant"] == variant].set_index("scenario_label").loc[scenario_order].reset_index()
        style = variant_style[variant]
        ax_a.plot(
            x_guard,
            sub["R2"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.0 if variant == "FG-MoE" else 1.8,
            markersize=5.6,
            label=variant,
        )
    ax_a.axhline(0.0, color="#9CA3AF", linewidth=0.8, linestyle=(0, (4, 2)))
    ax_a.set_xticks(x_guard)
    ax_a.set_xticklabels(["Clean", "Severe\nGaussian"])
    ax_a.set_ylim(0.30, 0.90)
    ax_a.set_ylabel("$R^2$")
    ax_a.set_title("Wind(CN) guardrail ordering", fontsize=9, fontweight="bold")
    style_axes(ax_a, grid=True, y_grid_only=True)
    ax_a.text(-0.16, 1.02, "(a)", transform=ax_a.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    ax_a.legend(loc="lower left", fontsize=6.8, ncol=1)
    ax_a.text(
        0.98,
        0.06,
        "Rated-power exceedance\n(clean / severe)\nRaw: 0.116 / 0.066%\nClip-only: 0 / 0\nFG-MoE: 0 / near-zero",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "#E5E7EB", "boxstyle": "round,pad=0.25", "alpha": 0.95},
    )

    ax_b2 = ax_b.twinx()
    ax_b.plot(
        jump_grid["threshold"],
        jump_grid["R2"],
        color=REFERENCE_FIGURE_PALETTE["rose"],
        marker="o",
        linewidth=1.9,
        markersize=4.8,
        label="Clean $R^2$",
    )
    ax_b.scatter([selected_tau], [selected_r2], s=42, color=REFERENCE_FIGURE_PALETTE["rose"], edgecolors="white", linewidths=0.8, zorder=5)
    ax_b.axvline(selected_tau, color=REFERENCE_FIGURE_PALETTE["lavender"], linewidth=1.0, linestyle=(0, (3, 2)))
    ax_b.set_xlabel(r"Jump Gate threshold $\tau$")
    ax_b.set_ylabel("Clean $R^2$")
    ax_b.set_xlim(0.19, 0.505)
    ax_b.set_ylim(jump_grid["R2"].min() - 0.0015, selected_r2 + 0.0015)
    ax_b.set_title("Jump Gate calibration", fontsize=9, fontweight="bold")
    style_axes(ax_b, grid=True, y_grid_only=True)
    ax_b.text(-0.16, 1.02, "(b)", transform=ax_b.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")

    ax_b2.plot(
        jump_grid["threshold"],
        jump_grid["avg_skip_rate"] * 100.0,
        color=REFERENCE_FIGURE_PALETTE["dusty_blue"],
        marker="s",
        linewidth=1.7,
        markersize=4.4,
        linestyle="--",
        label="Skip rate",
    )
    ax_b2.scatter([selected_tau], [selected_skip], s=40, color=REFERENCE_FIGURE_PALETTE["dusty_blue"], edgecolors="white", linewidths=0.8, zorder=5)
    ax_b2.axhline(target_skip, color=REFERENCE_FIGURE_PALETTE["dusty_blue"], linewidth=0.9, linestyle=(0, (2, 2)), alpha=0.9)
    ax_b2.set_ylabel("Skip rate (%)", color=REFERENCE_FIGURE_PALETTE["dusty_blue"])
    ax_b2.tick_params(axis="y", colors=REFERENCE_FIGURE_PALETTE["dusty_blue"])
    ax_b2.set_ylim(8.0, 70.0)
    ax_b.text(
        0.03,
        0.05,
        f"Selected $\\tau$ = {selected_tau:.2f}\nTarget skip = {target_skip:.1f}%\nObserved skip = {selected_skip:.1f}%\nAvg. compute ratio = {selected_compute:.0f}%",
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "#E5E7EB", "boxstyle": "round,pad=0.25", "alpha": 0.96},
    )
    ax_b.legend(
        handles=[
            Line2D([0], [0], color=REFERENCE_FIGURE_PALETTE["rose"], marker="o", linewidth=1.9, label="Clean $R^2$"),
            Line2D([0], [0], color=REFERENCE_FIGURE_PALETTE["dusty_blue"], marker="s", linewidth=1.7, linestyle="--", label="Skip rate"),
        ],
        loc="upper left",
        fontsize=6.6,
    )

    x_turb = np.arange(2)
    for variant in variant_order:
        sub = turbine_summary[turbine_summary["variant"] == variant].set_index("corruption_type").loc[scenario_order].reset_index()
        style = variant_style[variant]
        ax_c.errorbar(
            x_turb,
            sub["r2_mean"],
            yerr=sub["r2_std"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.0 if variant == "FG-MoE" else 1.8,
            markersize=5.2,
            capsize=3.0,
            label=variant,
        )
    ax_c.axhline(0.0, color="#9CA3AF", linewidth=0.8, linestyle=(0, (4, 2)))
    ax_c.set_xticks(x_turb)
    ax_c.set_xticklabels(["Clean", "Severe\nGaussian"])
    ax_c.set_ylim(-1.02, 0.82)
    ax_c.set_ylabel("Mean $R^2$ across 10 seeds")
    ax_c.set_title("Independent turbine repeat", fontsize=9, fontweight="bold")
    style_axes(ax_c, grid=True, y_grid_only=True)
    ax_c.text(-0.16, 1.02, "(c)", transform=ax_c.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    ax_c.text(
        0.98,
        0.95,
        "343 observed faults\n247 wind-speed flatlines\nFG-MoE P95: "
        f"{latency_map.get('Clean', float('nan')):.2f} / {latency_map.get('Severe Gaussian', float('nan')):.2f} ms",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "#E5E7EB", "boxstyle": "round,pad=0.25", "alpha": 0.96},
    )

    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.16, wspace=0.34)
    out_base = canonical_dir("main_text") / "fig6_controller_safety_chain"
    bundle = save_figure_bundle(fig, out_base, art_kind="combination")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="Figure 6", section="main_text", source="rebuilt")


def build_legacy_figure6_dispatch_value_evidence() -> dict[str, Any]:
    decision = _load_decision_summary()
    decision = decision[decision["model"].isin(["iTransformer", "TimeMixer", "Ultra-LSNT", "DLinear"])].copy()
    order = ["iTransformer", "TimeMixer", "Ultra-LSNT", "DLinear"]
    decision["model"] = pd.Categorical(decision["model"], categories=order, ordered=True)
    decision = decision.sort_values("model").reset_index(drop=True)

    dlinear = decision[decision["model"] == "DLinear"].iloc[0]
    decision["cost_saved_vs_dlinear"] = float(dlinear["l1_operating_cost_m_cny"]) - decision["l1_operating_cost_m_cny"]
    decision["fallback_reduction_vs_dlinear"] = float(dlinear["l1_fallback_pct"]) - decision["l1_fallback_pct"]
    short_labels = {"iTransformer": "iTrans", "TimeMixer": "TimeMixer", "Ultra-LSNT": "Ultra", "DLinear": "DLin"}

    fig = plt.figure(figsize=figure_size("double", 86))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.10, 0.90, 0.90], wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    zone = Rectangle((0.80, 0.0), 0.10, 25.0, facecolor="#EEF3FB", edgecolor=REFERENCE_FIGURE_PALETTE["dusty_blue"], linewidth=0.9, alpha=0.65, zorder=0)
    ax_a.add_patch(zone)
    ax_a.annotate(
        "Shared mapped-and-clipped interface\n($R^2>0.80$, fallback < 25%)",
        xy=(0.835, 11.5),
        xytext=(0.43, 61.0),
        textcoords="data",
        fontsize=6.3,
        color=REFERENCE_FIGURE_PALETTE["dusty_blue"],
        arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": REFERENCE_FIGURE_PALETTE["dusty_blue"]},
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "boxstyle": "round,pad=0.2", "alpha": 0.95},
    )
    offset_map = {
        "iTransformer": (10, -18),
        "TimeMixer": (12, 10),
        "Ultra-LSNT": (10, 22),
        "DLinear": (10, -10),
    }
    for _, row in decision.iterrows():
        model = str(row["model"])
        style = MODEL_STYLE[model]
        bubble = _cost_bubble_size(row["l1_operating_cost_m_cny"])
        ax_a.scatter(
            row["r2_at_060"],
            row["l1_fallback_pct"],
            s=bubble,
            color=style["color"],
            edgecolors="white",
            linewidth=0.9,
            zorder=3,
        )
        label = short_labels[model] if model != "DLinear" else "DLin (base)"
        ax_a.annotate(
            f"{label}\n{row['l1_operating_cost_m_cny']:.2f} M CNY",
            (row["r2_at_060"], row["l1_fallback_pct"]),
            xytext=offset_map[model],
            textcoords="offset points",
            fontsize=6.3,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.2},
        )
    ax_a.set_xlabel("$R^2$ at $\\sigma_{eff}=0.60$")
    ax_a.set_ylabel("Mapped fallback (%)")
    ax_a.set_title("Shared-interface screening", fontsize=8.6, fontweight="bold")
    ax_a.set_xlim(0.30, 0.90)
    ax_a.set_ylim(0.0, 70.0)
    style_axes(ax_a, grid=True, y_grid_only=True)
    ax_a.text(0.01, 0.99, "(a)", transform=ax_a.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")
    bubble_levels = [35, 45, 55]
    bubble_handles = [
        ax_a.scatter([], [], s=_cost_bubble_size(level), color="#D1D5DB", edgecolors="#9CA3AF", linewidth=0.8, label=f"{level:.0f} M CNY")
        for level in bubble_levels
    ]
    ax_a.legend(handles=bubble_handles, title="Bubble = L1 cost", loc="upper left", fontsize=6.2, title_fontsize=6.4, frameon=True, framealpha=0.95, edgecolor="#D1D5DB")

    x = np.arange(len(decision))
    cost_colors = [MODEL_STYLE[str(model)]["color"] for model in decision["model"]]
    ax_b.axhline(0.0, color="#9CA3AF", linewidth=0.8, linestyle=(0, (4, 2)))
    ax_b.bar(x, decision["cost_saved_vs_dlinear"], color=cost_colors, edgecolor="white", linewidth=0.8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([short_labels[str(model)] for model in decision["model"]], rotation=0)
    ax_b.set_ylabel("$\\Delta$ L1 cost vs DLin (M CNY)")
    ax_b.set_title("Cost saved relative to DLinear", fontsize=8.6, fontweight="bold")
    style_axes(ax_b, grid=True, y_grid_only=True)
    ax_b.text(0.01, 0.99, "(b)", transform=ax_b.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")
    for xx, yy in zip(x, decision["cost_saved_vs_dlinear"], strict=False):
        ax_b.text(xx, yy + 0.8, f"{yy:.1f}", ha="center", va="bottom", fontsize=6.2, color="#374151")

    ax_c.axhline(0.0, color="#9CA3AF", linewidth=0.8, linestyle=(0, (4, 2)))
    ax_c.bar(x, decision["fallback_reduction_vs_dlinear"], color=cost_colors, edgecolor="white", linewidth=0.8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([short_labels[str(model)] for model in decision["model"]], rotation=0)
    ax_c.set_ylabel("$\\Delta$ fallback vs DLin (ppt)")
    ax_c.set_title("Fallback reduction relative to DLinear", fontsize=8.6, fontweight="bold")
    style_axes(ax_c, grid=True, y_grid_only=True)
    ax_c.text(0.01, 0.99, "(c)", transform=ax_c.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")
    for xx, yy in zip(x, decision["fallback_reduction_vs_dlinear"], strict=False):
        ax_c.text(xx, yy + 1.0, f"{yy:.1f}", ha="center", va="bottom", fontsize=6.2, color="#374151")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.15, wspace=0.34)
    out_base = canonical_dir("main_text") / "fig6_dispatch_value_evidence"
    bundle = save_figure_bundle(fig, out_base, art_kind="combination")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="Figure 6 (legacy dispatch evidence)", section="main_text", source="rebuilt:decision_summary_legacy")


def build_figure7_lambda2_frontier() -> dict[str, Any]:
    return _stage_manual_locked_pdf("fig7_lambda2_frontier")

    joint = pd.read_csv(SUPP / "joint_parameter_full_windcn.csv")
    lambda_levels = [0.0, 0.01, 0.1, 1.0]
    anchor = (
        joint[(joint["N"] == 8) & (joint["K"] == 2) & (joint["tau"] == 0.5) & (joint["lambda2"].isin(lambda_levels))]
        .copy()
        .sort_values("lambda2")
    )
    if anchor.empty or anchor["lambda2"].nunique() != len(lambda_levels):
        raise ValueError("Figure 7 redraw requires N=8, K=2, tau=0.5 anchors at lambda2 = 0, 0.01, 0.1, 1.")

    fig, ax = plt.subplots(figsize=figure_size("one_point_five", 84))
    lambda_colors = {
        0.0: "#D58FB1",
        0.01: "#C54E88",
        0.1: "#7E98BD",
        1.0: "#9693C2",
    }
    line_color = "#7C7A7D"

    ax.plot(anchor["R2"], anchor["over_rated_pct"], color=line_color, linewidth=1.35, zorder=2)

    offset_map = {
        0.0: (-18, 12),
        0.01: (10, 2),
        0.1: (-4, 20),
        1.0: (8, 8),
    }
    for _, row in anchor.iterrows():
        level = float(row["lambda2"])
        ax.scatter(
            row["R2"],
            row["over_rated_pct"],
            s=110,
            color=lambda_colors[level],
            edgecolors="white",
            linewidth=1.0,
            zorder=4,
        )
        label = rf"$\lambda_2={level:g}$"
        dx, dy = offset_map[level]
        ax.annotate(
            label,
            (row["R2"], row["over_rated_pct"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.7,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.18},
            arrowprops={"arrowstyle": "-", "color": "#6B7280", "linewidth": 0.7},
            zorder=5,
        )

    ax.set_xlabel("$R^2$ at $\sigma_{eff}=0.60$")
    ax.set_ylabel("Rated-power exceedance (%)")
    ax.set_xlim(float(anchor["R2"].min()) - 0.08, float(anchor["R2"].max()) + 0.04)
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.5)
    ax.set_ylim(0.0, max(25.0, float(anchor["over_rated_pct"].max()) * 1.25))
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.01, 0.1, 1.0, 10.0]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    style_axes(ax, grid=True, y_grid_only=False)
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.7, linestyle=(0, (3, 2)), alpha=0.75, zorder=0)

    handles = [Line2D([0], [0], color=line_color, linewidth=1.35, label="Retained carrier path")]
    handles.extend(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=lambda_colors[level],
            markeredgecolor="white",
            markersize=7,
            label=rf"$\lambda_2={level:g}$",
        )
        for level in lambda_levels
    )
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.96, edgecolor="#D1D5DB", fontsize=6.2)

    out_base = canonical_dir("main_text") / "fig7_lambda2_frontier"
    bundle = save_figure_bundle(fig, out_base, art_kind="combination")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="Figure 7", section="main_text", source="rebuilt")


def build_graphical_abstract() -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(1328 / ART_DPI["halftone"], 531 / ART_DPI["halftone"]))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panels = [
        (0.03, 0.16, 0.26, 0.68, "#F8F1F4", REFERENCE_FIGURE_PALETTE["soft_pink"], "Corrupted\nSCADA"),
        (0.37, 0.16, 0.26, 0.68, "#EEF3FB", REFERENCE_FIGURE_PALETTE["dusty_blue"], "Edge-native\nSparse MoE"),
        (0.71, 0.16, 0.26, 0.68, "#F3F0F8", REFERENCE_FIGURE_PALETTE["lavender"], "Dispatch-\nscreened value"),
    ]
    for x, y, w, h, face, edge, title in panels:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor=face, edgecolor=edge, linewidth=1.0))
        ax.text(x + 0.02, y + h - 0.05, title, ha="left", va="top", fontsize=7.6, fontweight="bold", linespacing=1.0)

    x = np.linspace(0.06, 0.25, 160)
    y = 0.36 + 0.05 * np.sin(np.linspace(0, 8, x.size))
    y[35] += 0.10
    y[80:90] = np.nan
    y[120] -= 0.08
    ax.plot(x, y, color=REFERENCE_FIGURE_PALETTE["rose"], linewidth=1.8)
    ax.plot([0.11, 0.11], [0.42, 0.57], color=REFERENCE_FIGURE_PALETTE["rose"], linewidth=1.8)
    ax.text(0.06, 0.57, "Missing, drift,\nand noise", fontsize=7.6, color=REFERENCE_FIGURE_PALETTE["rose"], va="top")
    ax.text(0.06, 0.22, "Repair before forecasting.", fontsize=6.6, color="#374151", va="bottom")

    router_box = FancyBboxPatch((0.48, 0.46), 0.085, 0.075, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#F7E8EF", edgecolor=REFERENCE_FIGURE_PALETTE["rose"], linewidth=1.0)
    ax.add_patch(router_box)
    ax.text(0.522, 0.497, "Router\nTop-$K$", ha="center", va="center", fontsize=6.2, fontweight="bold")
    for i, yy in enumerate([0.44, 0.36, 0.28, 0.20]):
        active = i in {1, 2}
        edge = REFERENCE_FIGURE_PALETTE["rose"] if active else "#A8A4A7"
        face = "#FEE2E2" if active else "#F3F4F6"
        ax.add_patch(FancyBboxPatch((0.56, yy), 0.07, 0.05, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=face, edgecolor=edge, linewidth=1.0))
        ax.text(0.595, yy + 0.025, f"E{i + 1}", ha="center", va="center", fontsize=6.6, fontweight="bold")
        _arrow(ax, (0.56, 0.49 - 0.015 * i), (0.56, yy + 0.025), color=edge, lw=1.2 if active else 0.8, alpha=1.0 if active else 0.35)
    gate = Polygon([[0.43, 0.29], [0.465, 0.325], [0.50, 0.29], [0.465, 0.255]], closed=True, facecolor="#F1EDF8", edgecolor=REFERENCE_FIGURE_PALETTE["lavender"], linewidth=1.0)
    ax.add_patch(gate)
    ax.text(0.465, 0.29, "Jump\nGate", ha="center", va="center", fontsize=6.1, fontweight="bold", color=REFERENCE_FIGURE_PALETTE["lavender"])
    shield = FancyBboxPatch((0.43, 0.17), 0.17, 0.045, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="#EEF3FB", edgecolor=REFERENCE_FIGURE_PALETTE["dusty_blue"], linewidth=1.0)
    ax.add_patch(shield)
    ax.text(0.515, 0.192, "FG-MoE envelope", ha="center", va="center", fontsize=6.1, fontweight="bold", color=REFERENCE_FIGURE_PALETTE["dusty_blue"])

    xx = np.linspace(0.74, 0.94, 120)
    truth = 0.34 + 0.09 * np.sin(np.linspace(0, 6.2, xx.size))
    pred = truth + 0.008 * np.sin(np.linspace(0, 18, xx.size))
    ax.plot(xx, truth, color=REFERENCE_FIGURE_PALETTE["graphite"], linewidth=1.2)
    ax.plot(xx, pred, color=REFERENCE_FIGURE_PALETTE["dusty_blue"], linewidth=1.6)
    ax.axhline(0.47, xmin=0.73, xmax=0.96, color=REFERENCE_FIGURE_PALETTE["rose"], linestyle=(0, (4, 2)), linewidth=1.0)
    ax.text(0.74, 0.57, "$\\downarrow$ L1 cost", fontsize=8.2, fontweight="bold", color=REFERENCE_FIGURE_PALETTE["dusty_blue"])
    ax.text(0.74, 0.46, "$\\downarrow$ fallback burden", fontsize=8.2, fontweight="bold", color=REFERENCE_FIGURE_PALETTE["dusty_blue"])
    ax.text(0.74, 0.35, "$\\approx 0$ rated-power exceedance", fontsize=8.0, fontweight="bold", color=REFERENCE_FIGURE_PALETTE["rose"])
    ax.text(0.72, 0.22, "Screened before dispatch.", fontsize=6.6, color="#374151", va="bottom")

    _arrow(ax, (0.31, 0.50), (0.36, 0.50), lw=1.2)
    _arrow(ax, (0.65, 0.50), (0.70, 0.50), lw=1.2)

    out_base = canonical_dir("graphical_abstract") / "graphical_abstract"
    bundle = save_figure_bundle(fig, out_base, art_kind="combination")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="Graphical Abstract", section="graphical_abstract", source="rebuilt")


def build_appendix_tau_sweep() -> dict[str, Any]:
    return _stage_manual_locked_pdf("fig_app_tau_sweep_pareto")

    df = pd.DataFrame(TAU_SWEEP_AUDIT)
    fig, ax = plt.subplots(figsize=figure_size("one_point_five", 76))

    tau_colors = {
        0.3: REFERENCE_FIGURE_PALETTE["soft_pink"],
        0.5: REFERENCE_FIGURE_PALETTE["rose"],
        0.7: REFERENCE_FIGURE_PALETTE["dusty_blue"],
        0.9: REFERENCE_FIGURE_PALETTE["lavender"],
    }
    frontier = _nondominated_minimize_x_maximize_y(df, "cpu_latency_ms", "R2")
    ax.plot(frontier["cpu_latency_ms"], frontier["R2"], color=REFERENCE_FIGURE_PALETTE["graphite"], linewidth=1.6, linestyle=(0, (4, 2)), zorder=2)
    for _, row in df.iterrows():
        color = tau_colors[float(row["tau"])]
        is_frontier = bool(((frontier["tau"] == row["tau"]) & (frontier["cpu_latency_ms"] == row["cpu_latency_ms"])).any())
        ax.scatter(
            row["cpu_latency_ms"],
            row["R2"],
            s=72,
            color=color if is_frontier else "white",
            edgecolors=color,
            linewidth=1.0,
            alpha=1.0 if is_frontier else 0.95,
            zorder=3,
        )
        ax.annotate(
            f"$\\tau={row['tau']:g}$",
            (row["cpu_latency_ms"], row["R2"]),
            xytext=(6, 6 if row["tau"] != 0.9 else -14),
            textcoords="offset points",
            fontsize=6.8,
            color=color,
        )

    edge_row = df[df["tau"] == 0.3].iloc[0]
    default_row = df[df["tau"] == 0.5].iloc[0]
    ax.annotate(
        f"skip {edge_row['skip_rate_pct']:.1f}%",
        (edge_row["cpu_latency_ms"], edge_row["R2"]),
        xytext=(-46, -16),
        textcoords="offset points",
        fontsize=6.8,
        color=REFERENCE_FIGURE_PALETTE["soft_pink"],
        arrowprops={"arrowstyle": "-", "linewidth": 0.7, "color": REFERENCE_FIGURE_PALETTE["soft_pink"]},
    )
    ax.annotate(
        "default reliability anchor",
        (default_row["cpu_latency_ms"], default_row["R2"]),
        xytext=(8, 18),
        textcoords="offset points",
        fontsize=6.7,
        color=REFERENCE_FIGURE_PALETTE["rose"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.1},
    )
    dominated = df[~df["tau"].isin(frontier["tau"])].sort_values("cpu_latency_ms")
    if not dominated.empty:
        ref_xy = dominated.iloc[-1]
        ax.annotate(
            "same $R^2$, slower than\n$\\tau=0.5$ reference",
            (ref_xy["cpu_latency_ms"], ref_xy["R2"]),
            xytext=(16, -32),
            textcoords="offset points",
            fontsize=6.5,
            color="#4B5563",
            arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": "#6B7280"},
            bbox={"facecolor": "white", "edgecolor": "#E5E7EB", "boxstyle": "round,pad=0.2", "alpha": 0.95},
        )

    ax.set_title("Tau sweep with non-dominated operating points highlighted", fontsize=9, fontweight="bold")
    ax.set_xlabel("CPU batch-1 latency (ms)")
    ax.set_ylabel("$R^2$")
    ax.set_xlim(3.58, 3.80)
    ax.set_ylim(0.864, 0.875)
    style_axes(ax, grid=True, y_grid_only=False)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=tau_colors[level], markeredgecolor="white", label=f"$\\tau={level:g}$")
        for level in [0.3, 0.5, 0.7, 0.9]
    ]
    handles.append(Line2D([0], [0], color=REFERENCE_FIGURE_PALETTE["graphite"], linewidth=1.6, linestyle=(0, (4, 2)), label="Non-dominated path"))
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95, edgecolor="#D1D5DB")

    out_base = canonical_dir("appendix") / "fig_app_tau_sweep_pareto"
    bundle = save_figure_bundle(fig, out_base, art_kind="line")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="fig_app_tau_sweep_pareto", section="appendix", source="rebuilt:tau_audit_table")


def build_appendix_deploy_robustness() -> dict[str, Any]:
    return _stage_manual_locked_pdf("fig_app_deploy_robustness")

    hardware = _load_main_hardware_joint()
    hardware = hardware[hardware["panel"] == "A_hardware"].copy()
    hardware["latency_plot"] = hardware["p95_ms"].fillna(hardware["latency_ms"])
    point_meta = {
        "Ultra-LSNT(full)": {"color": "#E63946", "marker": "o", "label": "Ultra-Full", "size": 96, "z": 5},
        "Ultra-LSNT-Lite(opt)": {"color": "#F4A6AE", "marker": "o", "label": "Ultra-Lite", "size": 92, "z": 5},
        "ARIMA (auditable)": {"color": "#457B9D", "marker": "D", "label": "ARIMA", "size": 82, "z": 4},
        "SSA-ELM": {"color": "#7B8EA8", "marker": "P", "label": "SSA-ELM", "size": 76, "z": 3},
    }
    muted_edge = "#A8A8A8"

    fig, ax = plt.subplots(figsize=figure_size("one_point_five", 84))
    ax.axvspan(0.03, 2.0, facecolor="#E63946", edgecolor="#E63946", alpha=0.08, zorder=1)
    ax.axvline(2.0, color="#C97B84", linewidth=1.0, linestyle=(0, (3, 2)))
    ax.text(0.05, 0.95, "Controller-side budget\nLatency < 2 ms", transform=ax.transAxes, fontsize=6.6, color="#9D1C28", va="top")

    texts: list[Any] = []
    target_x: list[float] = []
    target_y: list[float] = []
    for _, row in hardware.iterrows():
        item = row["item"]
        if item in point_meta:
            meta = point_meta[item]
            face = meta["color"] if item == "Ultra-LSNT(full)" else "white"
            if item == "Ultra-LSNT-Lite(opt)":
                face = "#FFF5F6"
            ax.scatter(
                row["latency_plot"],
                row["R2_sigma_eff_060"],
                s=meta["size"],
                marker=meta["marker"],
                facecolors=face,
                edgecolors=meta["color"],
                linewidth=1.5,
                zorder=meta["z"],
            )
            texts.append(
                ax.text(
                    row["latency_plot"],
                    row["R2_sigma_eff_060"],
                    meta["label"],
                    fontsize=6.2,
                    color="#374151",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.15},
                    zorder=6,
                )
            )
            target_x.append(float(row["latency_plot"]))
            target_y.append(float(row["R2_sigma_eff_060"]))
        else:
            ax.scatter(
                row["latency_plot"],
                row["R2_sigma_eff_060"],
                s=42,
                marker="o",
                facecolors="white",
                edgecolors=muted_edge,
                linewidth=0.9,
                alpha=0.95,
                zorder=1,
            )

    ax.axhline(0.0, color="#9CA3AF", linewidth=0.8, linestyle=(0, (4, 2)))
    ax.set_xscale("log")
    ax.set_xlabel("Batch-1 latency (ms, log scale)")
    ax.set_ylabel("$R^2$ at $\\sigma_{eff}=0.60$")
    ax.set_title("Severe robustness vs latency", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0.03, 16.0)
    ax.set_ylim(0.30, 0.95)
    style_axes(ax, grid=True, y_grid_only=True)
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.7, linestyle=(0, (3, 2)), alpha=0.75, zorder=0)
    _adjust_texts(ax, texts, target_x, target_y, arrow_color="#6B7280", force_text=(0.20, 0.20), force_static=(0.20, 0.20), expand=(1.08, 1.10))

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#E63946", markeredgecolor="#E63946", label="Ultra-Full"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#FFF5F6", markeredgecolor="#F4A6AE", label="Ultra-Lite"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor="#457B9D", label="Highlighted rival"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=muted_edge, label="Other baselines"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.96, edgecolor="#D1D5DB", fontsize=6.3)

    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.98)
    out_base = canonical_dir("appendix") / "fig_app_deploy_robustness"
    bundle = save_figure_bundle(fig, out_base, art_kind="line")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="fig_app_deploy_robustness", section="appendix", source="rebuilt:single_panel_robustness")


def build_appendix_efficiency() -> dict[str, Any]:
    return _stage_manual_locked_pdf("fig_app_efficiency")

    hardware = _load_main_hardware_joint()
    hardware = hardware[hardware["panel"] == "A_hardware"].copy()
    hardware["latency_plot"] = hardware["p95_ms"].fillna(hardware["latency_ms"])
    point_meta = {
        "Ultra-LSNT(full)": {"color": "#E63946", "marker": "o", "label": "Ultra-Full", "size": 96, "z": 5},
        "Ultra-LSNT-Lite(opt)": {"color": "#F4A6AE", "marker": "o", "label": "Ultra-Lite", "size": 92, "z": 5},
        "DLinear": {"color": "#457B9D", "marker": "s", "label": "DLinear", "size": 84, "z": 4},
        "SSA-ELM": {"color": "#7B8EA8", "marker": "P", "label": "SSA-ELM", "size": 76, "z": 3},
    }
    muted_edge = "#A8A8A8"

    fig, ax = plt.subplots(figsize=figure_size("one_point_five", 84))
    ax.axvspan(0.03, 2.0, facecolor="#E63946", edgecolor="#E63946", alpha=0.08, zorder=1)
    ax.axvline(2.0, color="#C97B84", linewidth=1.0, linestyle=(0, (3, 2)))
    ax.text(0.05, 0.95, "Controller-side budget\nLatency < 2 ms", transform=ax.transAxes, fontsize=6.6, color="#9D1C28", va="top")

    texts: list[Any] = []
    target_x: list[float] = []
    target_y: list[float] = []
    ultra_path_x: list[float] = []
    ultra_path_y: list[float] = []
    for _, row in hardware.iterrows():
        item = row["item"]
        if item in point_meta:
            meta = point_meta[item]
            face = meta["color"] if item == "Ultra-LSNT(full)" else "white"
            if item == "Ultra-LSNT-Lite(opt)":
                face = "#FFF5F6"
            ax.scatter(
                row["latency_plot"],
                row["clean_R2"],
                s=meta["size"],
                marker=meta["marker"],
                facecolors=face,
                edgecolors=meta["color"],
                linewidth=1.5,
                zorder=meta["z"],
            )
            texts.append(
                ax.text(
                    row["latency_plot"],
                    row["clean_R2"],
                    meta["label"],
                    fontsize=6.2,
                    color="#374151",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.15},
                    zorder=6,
                )
            )
            target_x.append(float(row["latency_plot"]))
            target_y.append(float(row["clean_R2"]))
            if item.startswith("Ultra-LSNT"):
                ultra_path_x.append(float(row["latency_plot"]))
                ultra_path_y.append(float(row["clean_R2"]))
        else:
            ax.scatter(
                row["latency_plot"],
                row["clean_R2"],
                s=42,
                marker="o",
                facecolors="white",
                edgecolors=muted_edge,
                linewidth=0.9,
                alpha=0.95,
                zorder=1,
            )

    if ultra_path_x:
        order = np.argsort(np.array(ultra_path_x))
        ax.plot(
            np.array(ultra_path_x)[order],
            np.array(ultra_path_y)[order],
            color="#B56576",
            linewidth=1.35,
            linestyle="-",
            zorder=3,
            label="Ultra scaling path",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Batch-1 latency (ms, log scale)")
    ax.set_ylabel("Clean-slice $R^2$")
    ax.set_title("Clean accuracy vs latency", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0.03, 16.0)
    ax.set_ylim(0.68, 0.98)
    style_axes(ax, grid=True, y_grid_only=True)
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.7, linestyle=(0, (3, 2)), alpha=0.75, zorder=0)
    _adjust_texts(ax, texts, target_x, target_y, arrow_color="#6B7280", force_text=(0.20, 0.20), force_static=(0.20, 0.20), expand=(1.08, 1.10))

    legend_handles = [
        Line2D([0], [0], color="#B56576", linewidth=1.35, linestyle="-", label="Ultra scaling path"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#E63946", markeredgecolor="#E63946", label="Ultra-Full"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#FFF5F6", markeredgecolor="#F4A6AE", label="Ultra-Lite"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="white", markeredgecolor="#457B9D", label="Core budget rival"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=muted_edge, label="Other baselines"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.96, edgecolor="#D1D5DB", fontsize=6.3)

    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.98)
    out_base = canonical_dir("appendix") / "fig_app_efficiency"
    bundle = save_figure_bundle(fig, out_base, art_kind="line")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="fig_app_efficiency", section="appendix", source="rebuilt:single_panel_clean_efficiency")


def build_appendix_joint_nkt_map() -> dict[str, Any]:
    return _stage_manual_locked_pdf("fig_app_joint_nkt_pareto_3d")

    joint = pd.read_csv(SUPP / "joint_structural_nkt_windcn.csv")
    joint = joint[(joint["lambda2"] == 0.0) & (joint["sigma_eff"] == 0.6)].copy()
    pareto = pd.read_csv(SUPP / "joint_structural_pareto_windcn.csv")
    pareto = pareto[(pareto["lambda2"] == 0.0) & (pareto["sigma_eff"] == 0.6)].copy()

    grid = joint.pivot_table(index=["N", "K"], columns="tau", values="R2")
    row_labels = [f"N={n}, K={k}" for n, k in grid.index]
    col_labels = [f"$\\tau={tau:g}$" for tau in grid.columns]
    tau_colors = {
        0.3: REFERENCE_FIGURE_PALETTE["soft_pink"],
        0.5: REFERENCE_FIGURE_PALETTE["rose"],
        0.7: REFERENCE_FIGURE_PALETTE["dusty_blue"],
        0.9: REFERENCE_FIGURE_PALETTE["lavender"],
    }

    fig = plt.figure(figsize=figure_size("double", 86))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.9, 1.1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    for i, _ in enumerate(row_labels):
        for j, tau in enumerate(grid.columns):
            ax_a.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=tau_colors[float(tau)],
                    edgecolor="white",
                    linewidth=1.1,
                    alpha=0.26,
                )
            )
            value = grid.iloc[i, j]
            ax_a.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6.6, color="#1F2937")
    ax_a.set_xticks(np.arange(len(col_labels)))
    ax_a.set_xticklabels(col_labels)
    ax_a.set_yticks(np.arange(len(row_labels)))
    ax_a.set_yticklabels(row_labels)
    ax_a.set_xlim(-0.5, len(col_labels) - 0.5)
    ax_a.set_ylim(len(row_labels) - 0.5, -0.5)
    ax_a.set_title("$R^2$ grid with $\\tau$-coded columns", fontsize=8.8, fontweight="bold")
    ax_a.set_facecolor("#F9FAFB")
    style_axes(ax_a, grid=False, y_grid_only=False, hide_top_right=False)
    add_panel_label(ax_a, "(a)")

    marker_map = {1: "o", 2: "s", 4: "^"}
    for _, row in joint.iterrows():
        ax_b.scatter(
            row["latency_ms"],
            row["R2"],
            s=78,
            color=tau_colors[float(row["tau"])],
            marker=marker_map[int(row["K"])],
            edgecolors="white",
            linewidth=0.7,
            alpha=0.92,
        )
    pareto = pareto.sort_values("latency_ms")
    ax_b.scatter(pareto["latency_ms"], pareto["R2"], s=110, facecolors="none", edgecolors=REFERENCE_FIGURE_PALETTE["rose"], linewidth=1.3, zorder=5)
    for _, row in pareto.iterrows():
        ax_b.annotate(
            f"N={int(row['N'])}, K={int(row['K'])}",
            (row["latency_ms"], row["R2"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=6.4,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.15},
        )
    ax_b.text(float(pareto["latency_ms"].min()) + 0.06, float(pareto["R2"].max()) + 0.00035, "Discrete non-dominated settings", fontsize=6.8, color=REFERENCE_FIGURE_PALETTE["rose"])

    ax_b.set_xlabel("Batch-1 latency (ms)")
    ax_b.set_ylabel("$R^2$ at $\\sigma_{eff}=0.60$")
    ax_b.set_title("Latency-accuracy geometry", fontsize=8.8, fontweight="bold")
    ax_b.set_xlim(joint["latency_ms"].min() - 0.15, joint["latency_ms"].max() + 0.20)
    ax_b.set_ylim(joint["R2"].min() - 0.001, joint["R2"].max() + 0.0012)
    style_axes(ax_b, grid=True, y_grid_only=False)
    add_panel_label(ax_b, "(b)")

    tau_handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color, markeredgecolor="white", label=f"$\\tau={tau:g}$") for tau, color in tau_colors.items()]
    k_handles = [Line2D([0], [0], marker=marker_map[k], linestyle="none", markerfacecolor="#D1D5DB", markeredgecolor="#4B5563", label=f"$K={k}$") for k in sorted(marker_map)]
    tau_handles.append(Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=REFERENCE_FIGURE_PALETTE["rose"], markersize=8, label="Non-dominated point"))
    ax_b.legend(handles=tau_handles + k_handles, loc="lower left", frameon=True, framealpha=0.95, edgecolor="#D1D5DB")

    fig.tight_layout()
    out_base = canonical_dir("appendix") / "fig_app_joint_nkt_pareto_3d"
    bundle = save_figure_bundle(fig, out_base, art_kind="combination")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="fig_app_joint_nkt_pareto_3d", section="appendix", source="rebuilt:joint_structural_nkt")


def build_appendix_interpretability() -> dict[str, Any]:
    return _stage_manual_locked_pdf("fig_app_interpretability")

    wind = pd.read_csv(ROOT / "wind_expert_activation.csv")
    air = pd.read_csv(ROOT / "air_quality_expert_activation.csv")
    expert_cols = [col for col in wind.columns if col.startswith("Expert_")]

    fig = plt.figure(figsize=figure_size("double", 78))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 0.8])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    for idx, col in enumerate(expert_cols, start=1):
        ax_a.plot(wind["Hour"], wind[col], color=EXPERT_COLORS[idx - 1], linewidth=1.8, label=f"Expert {idx}")
    ax_a.axhline(0.25, color="#9CA3AF", linewidth=0.9, linestyle=(0, (4, 2)), label="Uniform 25%")
    ax_a.set_xlim(0, 23)
    ax_a.set_ylim(0, 0.40)
    ax_a.set_xlabel("Hour of day")
    ax_a.set_ylabel("Expert share")
    ax_a.set_title("Wind (CN) diurnal expert shares", fontsize=9, fontweight="bold")
    style_axes(ax_a, grid=True, y_grid_only=True)
    add_panel_label(ax_a, "(a)")
    ax_a.legend(loc="upper right", ncol=2)

    wind_mean = wind[expert_cols].mean()
    air_mean = air[expert_cols].mean()
    x = np.arange(2)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(expert_cols))
    for idx, col in enumerate(expert_cols):
        values = np.array([wind_mean[col], air_mean[col]])
        ax_b.bar(x + offsets[idx], values, width=width, color=EXPERT_COLORS[idx], edgecolor="white", linewidth=0.7, label=f"Expert {idx + 1}")
        for xx, yy in zip(x + offsets[idx], values, strict=False):
            ax_b.text(xx, yy + 0.008, f"{yy:.2f}", ha="center", va="bottom", fontsize=5.8, color="#374151")
    ax_b.axhline(0.25, color="#9CA3AF", linewidth=0.9, linestyle=(0, (4, 2)))
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["Wind (CN)", "Air quality"])
    ax_b.set_ylim(0, 0.40)
    ax_b.set_ylabel("Mean expert share")
    ax_b.set_title("Cross-task routing composition", fontsize=9, fontweight="bold")
    style_axes(ax_b, grid=True, y_grid_only=True)
    add_panel_label(ax_b, "(b)")

    fig.tight_layout()
    out_base = canonical_dir("appendix") / "fig_app_interpretability"
    bundle = save_figure_bundle(fig, out_base, art_kind="combination")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="fig_app_interpretability", section="appendix", source="rebuilt:expert_activation_csvs")


def build_appendix_routing_heatmap() -> dict[str, Any]:
    return _stage_manual_locked_pdf("fig_app_routing_heatmap")

    route = pd.read_csv(SUPP / "routing_timestep_matrix_ultra_block1.csv")
    truth = pd.read_csv(SUPP / "y_true_y_pred_ultra_dlinear_full_15min.csv")
    expert_cols = ["p_e1", "p_e2", "p_e3", "p_e4"]

    route_shift = route[expert_cols].max(axis=1).diff().abs().fillna(0.0)
    ramp_score = truth["y_true_kw"].diff().abs().fillna(0.0)
    score = ramp_score.rolling(16, min_periods=4).sum() + 25000.0 * route_shift.rolling(16, min_periods=4).sum()
    center = int(score.to_numpy().argmax())
    window = 256
    start = max(0, min(center - window // 3, len(route) - window))
    stop = start + window

    route_sub = route.iloc[start:stop].reset_index(drop=True)
    truth_sub = truth.iloc[start:stop].reset_index(drop=True)
    ramp = truth_sub["y_true_kw"].diff().abs().fillna(0.0).to_numpy()
    event_candidates = np.argsort(ramp)[::-1]
    event_centers: list[int] = []
    for idx in event_candidates:
        if all(abs(int(idx) - prev) >= 28 for prev in event_centers):
            event_centers.append(int(idx))
        if len(event_centers) == 2:
            break
    event_centers = sorted(event_centers)

    fig = plt.figure(figsize=figure_size("double", 96))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 0.9])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)

    mat = route_sub[expert_cols].to_numpy().T
    im = ax_top.imshow(mat, aspect="auto", cmap=_reference_sequential_cmap(), vmin=0.0, vmax=1.0)
    ax_top.set_yticks(np.arange(4))
    ax_top.set_yticklabels([f"E{i}" for i in range(1, 5)])
    ax_top.set_ylabel("Expert")
    ax_top.set_title("Top-$K$ gating weights over a representative 64-hour controller window", fontsize=9, fontweight="bold")
    for tick, color in zip(ax_top.get_yticklabels(), EXPERT_COLORS, strict=False):
        tick.set_color(color)
        tick.set_fontweight("bold")
    add_panel_label(ax_top, "(a)")

    box_colors = [REFERENCE_FIGURE_PALETTE["dusty_blue"], REFERENCE_FIGURE_PALETTE["rose"]]
    for idx, center_idx in enumerate(event_centers):
        x0 = max(0, center_idx - 10)
        x1 = min(window - 1, center_idx + 10)
        ax_top.add_patch(Rectangle((x0, -0.45), x1 - x0, 3.9, facecolor="none", edgecolor=box_colors[idx], linewidth=1.2, linestyle=(0, (4, 2))))
        ax_bottom.axvspan(x0, x1, facecolor=box_colors[idx], alpha=0.08, edgecolor=box_colors[idx], linewidth=0.8, linestyle=(0, (4, 2)))
        ax_bottom.text((x0 + x1) / 2, truth_sub["y_true_kw"].max() * 1.02, f"Event {chr(65 + idx)}", ha="center", va="bottom", fontsize=6.8, color=box_colors[idx])

    cbar = fig.colorbar(im, ax=ax_top, orientation="horizontal", fraction=0.08, pad=0.10)
    cbar.set_label("Gating weight (per-step, sums to 1)")

    ax_bottom.plot(truth_sub.index, truth_sub["y_true_kw"], color=REFERENCE_FIGURE_PALETTE["graphite"], linewidth=1.4, label="Realized")
    ax_bottom.plot(truth_sub.index, truth_sub["y_pred_ultra_kw"], color=REFERENCE_FIGURE_PALETTE["rose"], linewidth=1.8, label="Ultra-LSNT")
    ax_bottom.set_ylabel("Power (kW)")
    ax_bottom.set_xlabel("Window index (15-min)")
    ax_bottom.set_title("Realized vs routed forecast trajectory", fontsize=9, fontweight="bold")
    style_axes(ax_bottom, grid=True, y_grid_only=True)
    add_panel_label(ax_bottom, "(b)")
    ax_bottom.legend(loc="upper right")
    residual = truth_sub["y_pred_ultra_kw"] - truth_sub["y_true_kw"]
    ax_resid = ax_bottom.inset_axes([0.05, 0.62, 0.34, 0.26])
    ax_resid.axhline(0.0, color="#9CA3AF", linewidth=0.7, linestyle=(0, (4, 2)))
    ax_resid.plot(truth_sub.index, residual, color=REFERENCE_FIGURE_PALETTE["dusty_blue"], linewidth=1.0)
    for idx, center_idx in enumerate(event_centers):
        x0 = max(0, center_idx - 10)
        x1 = min(window - 1, center_idx + 10)
        ax_resid.axvspan(x0, x1, facecolor=box_colors[idx], alpha=0.08, edgecolor=box_colors[idx], linewidth=0.6)
    ax_resid.set_title("Residual", fontsize=6.3, pad=1.5)
    ax_resid.set_xticks([])
    ax_resid.tick_params(labelsize=5.4)
    style_axes(ax_resid, grid=True, y_grid_only=True, hide_top_right=False)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    fig.tight_layout()
    out_base = canonical_dir("appendix") / "fig_app_routing_heatmap"
    bundle = save_figure_bundle(fig, out_base, art_kind="combination")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="fig_app_routing_heatmap", section="appendix", source="rebuilt:routing_timestep_matrix")


def build_appendix_dm_summary() -> dict[str, Any]:
    df = pd.DataFrame(DM_AUDIT_ROWS)
    y_pos = np.arange(len(df))[::-1]

    fig, ax = plt.subplots(figsize=figure_size("one_point_five", 68))
    ax.axvline(0.0, color=REFERENCE_FIGURE_PALETTE["graphite"], linewidth=0.9, linestyle=(0, (4, 2)))
    ax.axvspan(-7.0, 0.0, facecolor="#F7EAF1", alpha=0.55, zorder=0)
    ax.axvspan(0.0, 7.0, facecolor="#EDF2F9", alpha=0.55, zorder=0)
    ax.text(-6.7, len(df) - 0.85, "Ultra-LSNT favored", fontsize=6.6, color=REFERENCE_FIGURE_PALETTE["rose"])
    ax.text(0.20, len(df) - 0.85, "Baseline favored", fontsize=6.6, color=REFERENCE_FIGURE_PALETTE["dusty_blue"])

    for y, (_, row) in zip(y_pos, df.iterrows(), strict=False):
        color = REFERENCE_FIGURE_PALETTE["rose"] if row["dm_stat"] < 0 else REFERENCE_FIGURE_PALETTE["dusty_blue"]
        ax.hlines(y, 0.0, row["dm_stat"], color=color, linewidth=2.2)
        ax.scatter(row["dm_stat"], y, s=42 + 12 * (-np.log10(row["p_value"])), color=color, edgecolors="white", linewidth=0.8, zorder=3)
        ax.annotate(f"$p={row['p_value']:.1e}$", (row["dm_stat"], y), xytext=(8 if row["dm_stat"] >= 0 else -56, 4), textcoords="offset points", fontsize=6.5, color="#374151")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Ultra-LSNT vs {name}" for name in df["baseline"]])
    ax.set_xlabel("DM statistic (negative favors Ultra-LSNT)")
    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_title("Wind (CN) DM audit", fontsize=8.8, fontweight="bold")
    style_axes(ax, grid=True, y_grid_only=False)
    ax.text(
        0.98,
        0.04,
        "3 contrasts; Bonferroni-adjusted $\\alpha=0.0167$.\nNo CI shown because the archived audit stores paired DM statistics and $p$ values only.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.0,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "#E5E7EB", "boxstyle": "round,pad=0.25", "alpha": 0.96},
    )

    out_base = canonical_dir("appendix") / "fig_app_dm_matrix"
    bundle = save_figure_bundle(fig, out_base, art_kind="combination")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="fig_app_dm_matrix", section="appendix", source="rebuilt:dm_table21")


def build_appendix_mapping_ranking_sensitivity() -> dict[str, Any]:
    agg = pd.read_csv(SUPP / "dispatch_mapping_sensitivity_wmax260_280_300.csv")
    agg = agg[agg["model"].isin(["Ultra-LSNT", "iTransformer", "TimeMixer", "DLinear"])].copy()
    case_order = ["wmax260", "wmax280", "wmax300"]
    case_labels = ["260 MW", "280 MW", "300 MW"]
    rank_table = agg.pivot(index="model", columns="mapping_case", values="total_rt_cost")[case_order]
    ranks = rank_table.rank(axis=0, method="dense")

    day_consistency = pd.read_csv(SUPP / "dispatch_mapping_daylevel_consistency_wmax260_280_300.csv")
    day_consistency = day_consistency[(day_consistency["metric"] == "rt_cost_total") & (day_consistency["model"].isin(["Ultra-LSNT", "iTransformer", "TimeMixer"]))].copy()

    fig = plt.figure(figsize=figure_size("double", 84))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.1, 0.9])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    x = np.arange(len(case_order))
    for model in ["iTransformer", "TimeMixer", "Ultra-LSNT", "DLinear"]:
        style = MODEL_STYLE[model]
        vals = ranks.loc[model, case_order].to_numpy()
        ax_a.plot(x, vals, color=style["color"], linestyle=style["linestyle"], marker=style["marker"], linewidth=style["linewidth"], label=model)
        for xx, yy in zip(x, vals, strict=False):
            cost = rank_table.loc[model, case_order[int(xx)]] / 1e6
            ax_a.text(xx, yy - 0.18, f"{cost:.2f}M", ha="center", va="top", fontsize=6.2, color="#4B5563")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(case_labels)
    ax_a.set_ylim(4.3, 0.7)
    ax_a.set_ylabel("Dispatch rank (1 = best)")
    ax_a.set_title("Ranking sensitivity across wind caps", fontsize=8.8, fontweight="bold")
    style_axes(ax_a, grid=True, y_grid_only=True)
    add_panel_label(ax_a, "(a)")
    ax_a.legend(loc="upper center", ncol=4)

    for model in ["iTransformer", "TimeMixer", "Ultra-LSNT"]:
        style = MODEL_STYLE[model]
        sub = day_consistency[day_consistency["model"] == model].copy()
        sub = sub.set_index("mapping_case").loc[case_order].reset_index()
        ax_b.plot(x, sub["day_win_ratio_vs_dlinear"], color=style["color"], linestyle=style["linestyle"], marker=style["marker"], linewidth=style["linewidth"], label=model)
        ax_b.fill_between(x, sub["day_win_ci95_low"], sub["day_win_ci95_high"], color=style["color"], alpha=0.10)
    ax_b.axhline(0.5, color=REFERENCE_FIGURE_PALETTE["graphite"], linewidth=0.8, linestyle=(0, (4, 2)))
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(case_labels)
    ax_b.set_ylabel("Day-win ratio vs DLinear")
    ax_b.set_xlabel("Network screening case")
    ax_b.set_title("Day-level cost consistency", fontsize=8.8, fontweight="bold")
    style_axes(ax_b, grid=True, y_grid_only=True)
    add_panel_label(ax_b, "(b)")

    fig.tight_layout()
    out_base = canonical_dir("appendix") / "fig_app_mapping_ranking_sensitivity"
    bundle = save_figure_bundle(fig, out_base, art_kind="combination")
    plt.close(fig)
    return bundle_manifest_row(bundle, label="fig_app_mapping_ranking_sensitivity", section="appendix", source="rebuilt:dispatch_mapping_sensitivity")


def _build_appendix_g2_palette_panels() -> list[dict[str, Any]]:
    raise RuntimeError("Figure G.2 palette panels are manual-locked. Use the finalized PDFs in paper_figures_canonical/appendix instead of regenerating them.")
    palette = {
        "soft_pink": REFERENCE_FIGURE_PALETTE["soft_pink"],
        "rose": REFERENCE_FIGURE_PALETTE["rose"],
        "dusty_blue": REFERENCE_FIGURE_PALETTE["dusty_blue"],
        "lavender": REFERENCE_FIGURE_PALETTE["lavender"],
        "graphite": REFERENCE_FIGURE_PALETTE["graphite"],
        "zone_fill": "#F7E9EF",
        "lite_fill": "#F7EAF1",
        "neutral_fill": "#F3F4F6",
        "muted_edge": "#A8A4A7",
    }
    manifest_rows: list[dict[str, Any]] = []

    # Panel (a): appendix-only recolor of the deployment path figure.
    fig, ax = plt.subplots(figsize=figure_size("one_point_five", 84))
    shape_map = {"PyTorch": "o", "ONNX": "s", "Reference": "D"}
    family_color = {
        "Full": palette["rose"],
        "Lite": palette["soft_pink"],
        "Reference": palette["graphite"],
    }
    label_map = {
        "Full / PyTorch FP32": "Ultra-Full / FP32",
        "Full / PyTorch INT8": "Ultra-Full / INT8",
        "Full / No-Skip": "No-skip ref",
        "Lite / PyTorch FP32": "Ultra-Lite / PT FP32",
        "Lite / PyTorch INT8": "Ultra-Lite / PT INT8",
        "Lite / ONNX FP32": "Ultra-Lite / ONNX",
    }
    plot_points: list[dict[str, float | str]] = []
    for point in DEPLOYMENT_POINTS:
        plot_point = dict(point)
        plot_point["memory_plot"] = float(point["memory"])
        plot_point["family_key"] = "Reference" if point["backend"] == "Reference" else point["family"]
        plot_points.append(plot_point)

    ax.add_patch(Rectangle((0.45, 0.0), 1.75, 1.0, facecolor=palette["zone_fill"], edgecolor=palette["zone_fill"], linewidth=0.8, alpha=0.55, zorder=1))
    ax.text(0.52, 0.82, "Controller-side\nbudget zone", fontsize=6.5, color=palette["rose"], ha="left", va="top", zorder=2)

    texts: list[Any] = []
    target_x: list[float] = []
    target_y: list[float] = []
    point_lookup: dict[str, tuple[float, float]] = {}
    for point in plot_points:
        family_key = str(point["family_key"])
        color = family_color[family_key]
        marker = shape_map[str(point["backend"])]
        fill = color if family_key == "Full" else "white"
        if family_key == "Lite":
            fill = palette["lite_fill"]
        if family_key == "Reference":
            fill = palette["neutral_fill"]
        size = 90 if family_key == "Full" else 78
        x = float(point["latency"])
        y = float(point["memory_plot"])
        point_lookup[str(point["label"])] = (x, y)
        ax.scatter(x, y, marker=marker, s=size, facecolors=fill, edgecolors=color, linewidth=1.5, zorder=4 if family_key != "Reference" else 2)
        text = ax.text(x, y, label_map[str(point["label"])], fontsize=6.2, color="#374151", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.15}, zorder=6)
        texts.append(text)
        target_x.append(x)
        target_y.append(y)

    path_color = palette["rose"]
    path_specs = [
        ("Full / PyTorch FP32", "Full / PyTorch INT8", "INT8 quantization"),
        ("Full / PyTorch INT8", "Lite / PyTorch INT8", "carrier simplification"),
        ("Lite / PyTorch INT8", "Lite / ONNX FP32", "runtime export"),
    ]
    for src_label, dst_label, step_label in path_specs:
        src = point_lookup[src_label]
        dst = point_lookup[dst_label]
        arrow = FancyArrowPatch(
            src,
            dst,
            arrowstyle="-|>",
            mutation_scale=10.0,
            linewidth=1.35,
            color=path_color,
            alpha=0.95,
            zorder=3,
            connectionstyle="arc3,rad=0.00",
        )
        ax.add_patch(arrow)
        mid_x = (src[0] * dst[0]) ** 0.5
        mid_y = 0.5 * (src[1] + dst[1])
        ax.text(
            mid_x,
            mid_y,
            step_label,
            fontsize=6.0,
            color=path_color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 0.12},
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.5)
    ax.set_xlabel("P95 latency (ms, log scale)")
    ax.set_ylabel("Peak memory (MiB, symlog scale)")
    ax.set_xlim(0.45, 8.5)
    ax.set_ylim(0.0, 30.0)
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1.0, 2.0, 4.0, 8.0]))
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.01, 0.1, 1.0, 10.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    style_axes(ax, grid=True, y_grid_only=False)
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.7, linestyle=(0, (3, 2)), alpha=0.75, zorder=0)
    _adjust_texts(ax, texts, target_x, target_y, arrow_color="#9CA3AF", force_text=(0.20, 0.20), force_static=(0.20, 0.20), expand=(1.08, 1.10))

    handles = [
        Line2D([0], [0], color=path_color, linewidth=1.35, label="Deployment optimization path"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=palette["rose"], markeredgecolor=palette["rose"], label="Ultra-Full"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=palette["lite_fill"], markeredgecolor=palette["soft_pink"], label="Ultra-Lite"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=palette["neutral_fill"], markeredgecolor=palette["graphite"], label="No-skip reference"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.96, edgecolor="#D1D5DB", fontsize=6.4)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.14)
    out_base = canonical_dir("appendix") / "fig_app_g2_deployment_pareto_palette"
    bundle = save_figure_bundle(fig, out_base, art_kind="line")
    plt.close(fig)
    manifest_rows.append(bundle_manifest_row(bundle, label="fig_app_g2_deployment_pareto_palette", section="appendix", source="rebuilt:g2_palette_variant"))

    hardware = _load_main_hardware_joint()
    hardware = hardware[hardware["panel"] == "A_hardware"].copy()
    hardware["latency_plot"] = hardware["p95_ms"].fillna(hardware["latency_ms"])

    # Panel (b): severe robustness vs latency.
    fig, ax = plt.subplots(figsize=figure_size("one_point_five", 84))
    point_meta = {
        "Ultra-LSNT(full)": {"color": palette["rose"], "marker": "o", "label": "Ultra-Full", "size": 96, "z": 5},
        "Ultra-LSNT-Lite(opt)": {"color": palette["soft_pink"], "marker": "o", "label": "Ultra-Lite", "size": 92, "z": 5},
        "ARIMA (auditable)": {"color": palette["dusty_blue"], "marker": "D", "label": "ARIMA", "size": 82, "z": 4},
        "SSA-ELM": {"color": palette["lavender"], "marker": "P", "label": "SSA-ELM", "size": 76, "z": 3},
    }
    ax.axvspan(0.03, 2.0, facecolor=palette["zone_fill"], edgecolor=palette["zone_fill"], alpha=0.55, zorder=1)
    ax.axvline(2.0, color=palette["rose"], linewidth=1.0, linestyle=(0, (3, 2)))
    ax.text(0.05, 0.95, "Controller-side budget\nLatency < 2 ms", transform=ax.transAxes, fontsize=6.6, color=palette["rose"], va="top")
    texts = []
    target_x = []
    target_y = []
    for _, row in hardware.iterrows():
        item = row["item"]
        if item in point_meta:
            meta = point_meta[item]
            face = meta["color"] if item == "Ultra-LSNT(full)" else "white"
            if item == "Ultra-LSNT-Lite(opt)":
                face = palette["lite_fill"]
            ax.scatter(row["latency_plot"], row["R2_sigma_eff_060"], s=meta["size"], marker=meta["marker"], facecolors=face, edgecolors=meta["color"], linewidth=1.5, zorder=meta["z"])
            texts.append(ax.text(row["latency_plot"], row["R2_sigma_eff_060"], meta["label"], fontsize=6.2, color="#374151", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.15}, zorder=6))
            target_x.append(float(row["latency_plot"]))
            target_y.append(float(row["R2_sigma_eff_060"]))
        else:
            ax.scatter(row["latency_plot"], row["R2_sigma_eff_060"], s=42, marker="o", facecolors="white", edgecolors=palette["muted_edge"], linewidth=0.9, alpha=0.95, zorder=1)

    ax.axhline(0.0, color="#9CA3AF", linewidth=0.8, linestyle=(0, (4, 2)))
    ax.set_xscale("log")
    ax.set_xlabel("Batch-1 latency (ms, log scale)")
    ax.set_ylabel("$R^2$ at $\\sigma_{eff}=0.60$")
    ax.set_title("Severe robustness vs latency", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0.03, 16.0)
    ax.set_ylim(0.30, 0.95)
    style_axes(ax, grid=True, y_grid_only=True)
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.7, linestyle=(0, (3, 2)), alpha=0.75, zorder=0)
    _adjust_texts(ax, texts, target_x, target_y, arrow_color="#6B7280", force_text=(0.20, 0.20), force_static=(0.20, 0.20), expand=(1.08, 1.10))
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=palette["rose"], markeredgecolor=palette["rose"], label="Ultra-Full"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=palette["lite_fill"], markeredgecolor=palette["soft_pink"], label="Ultra-Lite"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor=palette["dusty_blue"], label="Highlighted rival"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=palette["muted_edge"], label="Other baselines"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.96, edgecolor="#D1D5DB", fontsize=6.3)
    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.98)
    out_base = canonical_dir("appendix") / "fig_app_g2_deploy_robustness_palette"
    bundle = save_figure_bundle(fig, out_base, art_kind="line")
    plt.close(fig)
    manifest_rows.append(bundle_manifest_row(bundle, label="fig_app_g2_deploy_robustness_palette", section="appendix", source="rebuilt:g2_palette_variant"))

    # Panel (c): clean accuracy vs latency.
    fig, ax = plt.subplots(figsize=figure_size("one_point_five", 84))
    point_meta = {
        "Ultra-LSNT(full)": {"color": palette["rose"], "marker": "o", "label": "Ultra-Full", "size": 96, "z": 5},
        "Ultra-LSNT-Lite(opt)": {"color": palette["soft_pink"], "marker": "o", "label": "Ultra-Lite", "size": 92, "z": 5},
        "DLinear": {"color": palette["dusty_blue"], "marker": "s", "label": "DLinear", "size": 84, "z": 4},
        "SSA-ELM": {"color": palette["lavender"], "marker": "P", "label": "SSA-ELM", "size": 76, "z": 3},
    }
    ax.axvspan(0.03, 2.0, facecolor=palette["zone_fill"], edgecolor=palette["zone_fill"], alpha=0.55, zorder=1)
    ax.axvline(2.0, color=palette["rose"], linewidth=1.0, linestyle=(0, (3, 2)))
    ax.text(0.05, 0.95, "Controller-side budget\nLatency < 2 ms", transform=ax.transAxes, fontsize=6.6, color=palette["rose"], va="top")
    texts = []
    target_x = []
    target_y = []
    ultra_path_x: list[float] = []
    ultra_path_y: list[float] = []
    for _, row in hardware.iterrows():
        item = row["item"]
        if item in point_meta:
            meta = point_meta[item]
            face = meta["color"] if item == "Ultra-LSNT(full)" else "white"
            if item == "Ultra-LSNT-Lite(opt)":
                face = palette["lite_fill"]
            ax.scatter(row["latency_plot"], row["clean_R2"], s=meta["size"], marker=meta["marker"], facecolors=face, edgecolors=meta["color"], linewidth=1.5, zorder=meta["z"])
            texts.append(ax.text(row["latency_plot"], row["clean_R2"], meta["label"], fontsize=6.2, color="#374151", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.15}, zorder=6))
            target_x.append(float(row["latency_plot"]))
            target_y.append(float(row["clean_R2"]))
            if item.startswith("Ultra-LSNT"):
                ultra_path_x.append(float(row["latency_plot"]))
                ultra_path_y.append(float(row["clean_R2"]))
        else:
            ax.scatter(row["latency_plot"], row["clean_R2"], s=42, marker="o", facecolors="white", edgecolors=palette["muted_edge"], linewidth=0.9, alpha=0.95, zorder=1)

    if ultra_path_x:
        order = np.argsort(np.array(ultra_path_x))
        ax.plot(np.array(ultra_path_x)[order], np.array(ultra_path_y)[order], color=palette["rose"], linewidth=1.35, linestyle="-", zorder=3, label="Ultra scaling path")

    ax.set_xscale("log")
    ax.set_xlabel("Batch-1 latency (ms, log scale)")
    ax.set_ylabel("Clean-slice $R^2$")
    ax.set_title("Clean accuracy vs latency", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0.03, 16.0)
    ax.set_ylim(0.68, 0.98)
    style_axes(ax, grid=True, y_grid_only=True)
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.7, linestyle=(0, (3, 2)), alpha=0.75, zorder=0)
    _adjust_texts(ax, texts, target_x, target_y, arrow_color="#6B7280", force_text=(0.20, 0.20), force_static=(0.20, 0.20), expand=(1.08, 1.10))
    legend_handles = [
        Line2D([0], [0], color=palette["rose"], linewidth=1.35, linestyle="-", label="Ultra scaling path"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=palette["rose"], markeredgecolor=palette["rose"], label="Ultra-Full"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=palette["lite_fill"], markeredgecolor=palette["soft_pink"], label="Ultra-Lite"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="white", markeredgecolor=palette["dusty_blue"], label="Core budget rival"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=palette["muted_edge"], label="Other baselines"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.96, edgecolor="#D1D5DB", fontsize=6.3)
    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.98)
    out_base = canonical_dir("appendix") / "fig_app_g2_efficiency_palette"
    bundle = save_figure_bundle(fig, out_base, art_kind="line")
    plt.close(fig)
    manifest_rows.append(bundle_manifest_row(bundle, label="fig_app_g2_efficiency_palette", section="appendix", source="rebuilt:g2_palette_variant"))

    return manifest_rows


def stage_appendix_assets() -> list[dict[str, Any]]:
    # Figure G.2 is manually finalized and must not be regenerated here.
    return [
        _stage_manual_locked_pdf("fig_app_g2_deployment_pareto_palette"),
        _stage_manual_locked_pdf("fig_app_g2_deploy_robustness_palette"),
        _stage_manual_locked_pdf("fig_app_g2_efficiency_palette"),
    ]


def validate_latex_references(tex_path: Path) -> list[dict[str, Any]]:
    pattern = re.compile(r"\\safeincludegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    text = tex_path.read_text(encoding="utf-8")
    refs = pattern.findall(text)
    results: list[dict[str, Any]] = []
    for ref in refs:
        path = (tex_path.parent / ref).resolve()
        results.append({"reference": ref, "exists": path.exists(), "resolved_path": str(path)})
    return results


def build_all() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    set_paper_style()
    manifest_rows = [
        build_figure2_workflow(),
        build_figure3_data_overview(),
        build_figure5_deployment_pareto(),
        build_figure6_controller_safety_chain(),
        build_legacy_figure6_dispatch_value_evidence(),
        build_figure7_lambda2_frontier(),
        build_graphical_abstract(),
        build_appendix_deploy_robustness(),
        build_appendix_efficiency(),
        build_appendix_tau_sweep(),
        build_appendix_interpretability(),
        build_appendix_routing_heatmap(),
        build_appendix_dm_summary(),
        build_appendix_mapping_ranking_sensitivity(),
    ]
    manifest_rows.extend(stage_appendix_assets())
    validation_rows = validate_latex_references(MAIN_TEX)
    write_manifest(manifest_rows, basename="applied_energy_figure_manifest")
    write_manifest(validation_rows, basename="latex_reference_validation")
    return manifest_rows, validation_rows


def main() -> None:
    manifest_rows, validation_rows = build_all()
    missing = [row for row in validation_rows if not row["exists"]]
    print(f"Generated or staged {len(manifest_rows)} figure assets.")
    if missing:
        print(f"Missing {len(missing)} LaTeX figure references:")
        for row in missing:
            print(f" - {row['reference']}")
    else:
        print("All LaTeX figure references resolved.")


if __name__ == "__main__":
    main()
