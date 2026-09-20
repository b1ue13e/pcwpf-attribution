from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(".")


def _png_to_pdf(png_path: Path, pdf_path: Path) -> None:
    img = Image.open(png_path)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img_rgb = bg
    else:
        img_rgb = img.convert("RGB")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    img_rgb.save(pdf_path, "PDF", resolution=300.0)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_pair(src_base: Path, dst_base: Path) -> None:
    for ext in (".png", ".pdf"):
        src = src_base.with_suffix(ext)
        if src.exists():
            _copy(src, dst_base.with_suffix(ext))


def main() -> None:
    # 1) Generate key figures used by latex改.txt (3/4/5) in ROOT.
    import make_latex_gai_figures as gen

    gen.main()

    # 2) Figure 2 in latex改.txt is architecture; reuse vector exports from figure/.
    _copy_pair(ROOT / "figure" / "fig_architecture", ROOT / "Figure_2_Final")

    # 3) Copy figures referenced directly by latex改.txt from figure/ into ROOT.
    _copy_pair(ROOT / "figure" / "fig_multi_domain", ROOT / "fig_multi_domain")
    _copy_pair(ROOT / "figure" / "fig_ablation", ROOT / "fig_ablation")

    # fig_efficiency only exists as PNG in figure/; copy + make a PDF sibling.
    src_eff = ROOT / "figure" / "fig_efficiency.png"
    dst_eff = ROOT / "fig_efficiency.png"
    if src_eff.exists():
        _copy(src_eff, dst_eff)
        pdf_eff = ROOT / "fig_efficiency.pdf"
        if not pdf_eff.exists():
            _png_to_pdf(dst_eff, pdf_eff)

    # 4) thumbnails/figure/*.pdf assets (copy from figure/ if present)
    _copy(ROOT / "figure" / "raw_series_wind_cn.pdf", ROOT / "thumbnails" / "figure" / "raw_series_wind_cn.pdf")
    _copy(
        ROOT / "figure" / "noise_protocol_schematic.pdf",
        ROOT / "thumbnails" / "figure" / "noise_protocol_schematic.pdf",
    )

    # 5) Dispatch chart path
    src_dispatch = ROOT / "figure" / "dispatch_comparison_charts.png"
    dst_dispatch = ROOT / "dispatch" / "dispatch_comparison_charts.png"
    if src_dispatch.exists():
        _copy(src_dispatch, dst_dispatch)
        pdf_dispatch = ROOT / "dispatch" / "dispatch_comparison_charts.pdf"
        if not pdf_dispatch.exists():
            _png_to_pdf(dst_dispatch, pdf_dispatch)

    # 6) Also place Figure_2/3/4/5 PDFs next to PNGs (already generated), but ensure they exist.
    for base in ("Figure_2_Final", "Figure_3_Final", "Figure_4_Final", "Figure_5_Final"):
        png = ROOT / f"{base}.png"
        pdf = ROOT / f"{base}.pdf"
        if png.exists() and not pdf.exists():
            _png_to_pdf(png, pdf)


if __name__ == "__main__":
    main()

