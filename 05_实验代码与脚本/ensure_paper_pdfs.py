from __future__ import annotations

import re
from pathlib import Path

from PIL import Image


_INCLUDE_RE = re.compile(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}")


def _bases_from_tex(tex_path: Path) -> list[str]:
    text = tex_path.read_text(encoding="utf-8", errors="ignore")
    out: list[str] = []
    for m in _INCLUDE_RE.finditer(text):
        name = m.group(1).strip()
        # Drop any directory and extension; LaTeX searches by base name when extension is omitted.
        p = Path(name)
        base = p.name
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
            base = p.with_suffix("").name
        out.append(base)
    return sorted(set(out))


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


def ensure_pdfs(
    *,
    figure_dir: str | Path = "figure",
    tex_files: tuple[str, ...] = ("ultra_lsnt_paper_complete.tex", "ultra_lsnt_paper.tex"),
) -> None:
    figure_dir = Path(figure_dir)
    bases: set[str] = set()
    for tex in tex_files:
        tex_path = Path(tex)
        if tex_path.exists():
            bases.update(_bases_from_tex(tex_path))

    if not bases:
        return

    for base in sorted(bases):
        pdf_path = figure_dir / f"{base}.pdf"
        if pdf_path.exists():
            continue

        for ext in (".png", ".jpg", ".jpeg"):
            src = figure_dir / f"{base}{ext}"
            if src.exists():
                _png_to_pdf(src, pdf_path)
                break


if __name__ == "__main__":
    ensure_pdfs()
