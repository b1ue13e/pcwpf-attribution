from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

def set_ae_style(font: str = "Times New Roman", base_fontsize: int = 8) -> None:
    """
    Elsevier/Applied Energy-friendly defaults:
    - embedded TrueType fonts in PDF (fonttype=42)
    - compact layout, small fonts, consistent line widths
    """
    mpl.rcParams.update({
        "font.family": font,
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,   # embed TrueType
        "ps.fonttype": 42,
    })

def save_ae(fig: plt.Figure, out_prefix: str, out_dir: str = "figures_out",
            dpi_lineart: int = 1000, dpi_mixed: int = 600) -> None:
    """
    Save both vector (PDF) and high-dpi raster (TIFF).
    - PDF for line art / diagrams
    - TIFF for journals that insist on raster or for mixed content
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{out_prefix}.pdf"
    tif_path = out_dir / f"{out_prefix}.tif"

    fig.savefig(pdf_path, bbox_inches="tight")
    # Use higher dpi by default; tune per figure type
    fig.savefig(tif_path, dpi=dpi_mixed, bbox_inches="tight")