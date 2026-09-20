from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from cycler import cycler
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt


MM_TO_INCH = 1.0 / 25.4
CANONICAL_ROOT = Path("paper_figures_canonical")
CANONICAL_MAIN = CANONICAL_ROOT / "main_text"
CANONICAL_APPENDIX = CANONICAL_ROOT / "appendix"
CANONICAL_GRAPHICAL_ABSTRACT = CANONICAL_ROOT / "graphical_abstract"
CANONICAL_MANIFEST = CANONICAL_ROOT / "manifests"

FIG_WIDTH_MM = {
    "single": 90.0,
    "one_point_five": 135.0,
    "double": 190.0,
}
FIG_MAX_HEIGHT_MM = 240.0
ART_DPI = {
    "line": 1000,
    "combination": 500,
    "halftone": 300,
    "preview": 300,
}


@dataclass(frozen=True)
class SaveSpec:
    dpi: int = 300
    formats: tuple[str, ...] = ("png", "pdf")
    bbox_inches: str = "tight"
    pad_inches: float = 0.02


@dataclass(frozen=True)
class FigureBundle:
    base_path: Path
    pdf_path: Path
    png_path: Path
    tif_path: Path
    art_kind: str
    width_mm: float
    height_mm: float


REFERENCE_FIGURE_PALETTE = {
    "soft_pink": "#D58FB1",
    "rose": "#C54E88",
    "dusty_blue": "#7E98BD",
    "lavender": "#9693C2",
    "graphite": "#7C7A7D",
    "charcoal": "#2F2A2C",
}

OKABE_ITO = {
    "black": REFERENCE_FIGURE_PALETTE["charcoal"],
    "orange": REFERENCE_FIGURE_PALETTE["soft_pink"],
    "sky_blue": REFERENCE_FIGURE_PALETTE["dusty_blue"],
    "bluish_green": REFERENCE_FIGURE_PALETTE["lavender"],
    "yellow": REFERENCE_FIGURE_PALETTE["soft_pink"],
    "blue": REFERENCE_FIGURE_PALETTE["dusty_blue"],
    "vermillion": REFERENCE_FIGURE_PALETTE["rose"],
    "reddish_purple": REFERENCE_FIGURE_PALETTE["lavender"],
    "gray": REFERENCE_FIGURE_PALETTE["graphite"],
}

MODEL_STYLE = {
    "Ultra-LSNT": {"color": REFERENCE_FIGURE_PALETTE["rose"], "linestyle": "-", "marker": "o", "linewidth": 2.4},
    "FG-MoE": {"color": REFERENCE_FIGURE_PALETTE["lavender"], "linestyle": "-", "marker": "D", "linewidth": 2.0},
    "iTransformer": {"color": REFERENCE_FIGURE_PALETTE["dusty_blue"], "linestyle": "--", "marker": "s", "linewidth": 1.8},
    "TimeMixer": {"color": REFERENCE_FIGURE_PALETTE["soft_pink"], "linestyle": ":", "marker": "^", "linewidth": 1.8},
    "DLinear": {"color": REFERENCE_FIGURE_PALETTE["graphite"], "linestyle": "-.", "marker": "v", "linewidth": 1.8},
    "LightGBM": {"color": REFERENCE_FIGURE_PALETTE["lavender"], "linestyle": "-", "marker": "P", "linewidth": 1.5},
    "Persistence": {"color": REFERENCE_FIGURE_PALETTE["graphite"], "linestyle": (0, (6, 2)), "marker": "X", "linewidth": 1.6},
    "PatchTST": {"color": REFERENCE_FIGURE_PALETTE["soft_pink"], "linestyle": (0, (3, 2)), "marker": "h", "linewidth": 1.6},
    "Chronos-Bolt-Small": {"color": REFERENCE_FIGURE_PALETTE["lavender"], "linestyle": (0, (1, 2)), "marker": "H", "linewidth": 1.6},
    "Moirai-2.0-R-Small": {"color": OKABE_ITO["black"], "linestyle": "-", "marker": "+", "linewidth": 1.6},
}


def _first_available_font(candidates: list[str], fallback: str) -> str:
    for name in candidates:
        try:
            _ = font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            return name
        except Exception:
            continue
    return fallback


def mm_to_inch(mm: float) -> float:
    return mm * MM_TO_INCH


def figure_size(width_key: str, height_mm: float) -> tuple[float, float]:
    width_mm = FIG_WIDTH_MM[width_key]
    clipped_height = min(height_mm, FIG_MAX_HEIGHT_MM)
    return mm_to_inch(width_mm), mm_to_inch(clipped_height)


def canonical_dir(section: str) -> Path:
    if section == "main_text":
        return ensure_output_dir(CANONICAL_MAIN)
    if section == "appendix":
        return ensure_output_dir(CANONICAL_APPENDIX)
    if section == "graphical_abstract":
        return ensure_output_dir(CANONICAL_GRAPHICAL_ABSTRACT)
    raise ValueError(f"Unknown figure section: {section}")


def set_paper_style() -> None:
    """
    Apply an Applied Energy / Elsevier-friendly plotting style.
    """
    sans_font = _first_available_font(
        ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        fallback="DejaVu Sans",
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [sans_font],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": ART_DPI["preview"],
            "savefig.dpi": ART_DPI["line"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titlepad": 6,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(
                color=[
                    REFERENCE_FIGURE_PALETTE["rose"],
                    REFERENCE_FIGURE_PALETTE["dusty_blue"],
                    REFERENCE_FIGURE_PALETTE["soft_pink"],
                    REFERENCE_FIGURE_PALETTE["lavender"],
                    REFERENCE_FIGURE_PALETTE["graphite"],
                    OKABE_ITO["black"],
                ]
            ),
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.25,
            "grid.linewidth": 0.45,
            "legend.frameon": False,
            "legend.fontsize": 7,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "hatch.linewidth": 0.4,
        }
    )


def style_axes(
    ax: plt.Axes,
    *,
    grid: bool = True,
    y_grid_only: bool = True,
    hide_top_right: bool = True,
) -> None:
    ax.set_axisbelow(True)
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y" if y_grid_only else "both", color="#D9D9D9", linewidth=0.45, linestyle="--", alpha=0.7)
    else:
        ax.grid(False)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_figure(
    fig: plt.Figure,
    out_base: str | Path,
    *,
    spec: SaveSpec = SaveSpec(),
    mkdir: bool = True,
) -> list[Path]:
    out_base = Path(out_base)
    if mkdir:
        out_base.parent.mkdir(parents=True, exist_ok=True)

    fig.canvas.draw()

    written: list[Path] = []
    for fmt in spec.formats:
        out_path = out_base.with_suffix(f".{fmt}")
        fig.savefig(
            out_path,
            dpi=spec.dpi if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"} else None,
            bbox_inches=spec.bbox_inches,
            pad_inches=spec.pad_inches,
        )
        written.append(out_path)
    return written


def save_figure_bundle(
    fig: plt.Figure,
    out_base: str | Path,
    *,
    art_kind: str = "line",
    pad_inches: float = 0.02,
) -> FigureBundle:
    if art_kind not in {"line", "combination", "halftone"}:
        raise ValueError(f"Unsupported art kind: {art_kind}")

    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()

    raster_dpi = ART_DPI[art_kind]
    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    tif_path = out_base.with_suffix(".tif")

    with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=pad_inches)
        fig.savefig(png_path, dpi=ART_DPI["preview"], bbox_inches="tight", pad_inches=pad_inches)
        fig.savefig(tif_path, dpi=raster_dpi, bbox_inches="tight", pad_inches=pad_inches)

    width_in, height_in = fig.get_size_inches()
    return FigureBundle(
        base_path=out_base,
        pdf_path=pdf_path,
        png_path=png_path,
        tif_path=tif_path,
        art_kind=art_kind,
        width_mm=width_in / MM_TO_INCH,
        height_mm=height_in / MM_TO_INCH,
    )


def render_pdf_preview(pdf_path: str | Path, out_base: str | Path) -> tuple[Path | None, Path | None]:
    """
    Best-effort renderer for copied PDF assets when only vector originals exist.
    """
    pdf_path = Path(pdf_path)
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fitz  # type: ignore
    except Exception:
        fitz = None

    if fitz is not None:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        mat = fitz.Matrix(ART_DPI["preview"] / 72.0, ART_DPI["preview"] / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_path = out_base.with_suffix(".png")
        pix.save(png_path)

        mat_tif = fitz.Matrix(ART_DPI["combination"] / 72.0, ART_DPI["combination"] / 72.0)
        pix_tif = page.get_pixmap(matrix=mat_tif, alpha=False)
        tif_path = out_base.with_suffix(".tif")
        pix_tif.save(tif_path)
        return png_path, tif_path

    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        return None, None

    png_path = out_base.with_suffix(".png")
    tif_path = out_base.with_suffix(".tif")
    png_stem = str(out_base)

    subprocess.run(
        [
            pdftoppm,
            "-singlefile",
            "-r",
            str(ART_DPI["preview"]),
            "-png",
            str(pdf_path),
            png_stem,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if png_path.exists():
        try:
            from PIL import Image

            with Image.open(png_path) as img:
                img.save(
                    tif_path,
                    format="TIFF",
                    dpi=(ART_DPI["combination"], ART_DPI["combination"]),
                    compression="tiff_lzw",
                )
        except Exception:
            tif_path = None
    else:
        tif_path = None

    return png_path if png_path.exists() else None, tif_path if tif_path and tif_path.exists() else None


def write_manifest(rows: list[dict[str, Any]], *, basename: str = "figure_manifest") -> tuple[Path, Path]:
    CANONICAL_MANIFEST.mkdir(parents=True, exist_ok=True)
    json_path = CANONICAL_MANIFEST / f"{basename}.json"
    csv_path = CANONICAL_MANIFEST / f"{basename}.csv"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    return json_path, csv_path


def bundle_manifest_row(bundle: FigureBundle, *, label: str, section: str, source: str) -> dict[str, Any]:
    return {
        "label": label,
        "section": section,
        "source": source,
        "base_path": str(bundle.base_path),
        "pdf_path": str(bundle.pdf_path),
        "png_path": str(bundle.png_path),
        "tif_path": str(bundle.tif_path),
        "art_kind": bundle.art_kind,
        "width_mm": round(bundle.width_mm, 2),
        "height_mm": round(bundle.height_mm, 2),
    }


def ensure_output_dir(output_dir: str | Path = "figure") -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def close_figures(figs: Iterable[plt.Figure]) -> None:
    for fig in figs:
        plt.close(fig)
