from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


INCLUDE_RE = re.compile(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}")


def _parse_includes(tex_like_path: Path) -> list[Path]:
    text = tex_like_path.read_text(encoding="utf-8", errors="ignore")
    out: list[Path] = []
    for m in INCLUDE_RE.finditer(text):
        p = Path(m.group(1).strip())
        out.append(p)
    return out


def _is_blank_right_half(png_path: Path, *, white_threshold: int = 245, frac_threshold: float = 0.985) -> bool:
    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    crop = img.crop((w // 2, 0, w, h))
    arr = np.asarray(crop, dtype=np.uint8)  # (H, W, 3)
    if arr.size == 0:
        return True
    white_mask = (arr[..., 0] >= white_threshold) & (arr[..., 1] >= white_threshold) & (arr[..., 2] >= white_threshold)
    frac = float(white_mask.mean())
    return frac >= frac_threshold


def main() -> None:
    src = Path("latex改.txt")
    if not src.exists():
        print("[fail] latex改.txt not found")
        sys.exit(2)

    includes = _parse_includes(src)
    missing: list[str] = []
    missing_pdf_sibling: list[str] = []

    for rel in includes:
        path = Path(rel.as_posix())
        if not path.exists():
            missing.append(str(path))
            continue

        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            pdf = path.with_suffix(".pdf")
            if not pdf.exists():
                missing_pdf_sibling.append(str(pdf))

    if missing:
        print("[fail] Missing includegraphics targets:")
        for m in missing:
            print("  -", m)
        sys.exit(2)

    if missing_pdf_sibling:
        print("[fail] Missing PDF siblings (paper submission):")
        for m in missing_pdf_sibling:
            print("  -", m)
        sys.exit(2)

    # Sanity: ensure key composites are not blank on the right half (previous bug).
    for name in ("Figure_3_Final.png",):
        p = Path(name)
        if p.exists() and _is_blank_right_half(p):
            print(f"[fail] {name} appears blank on the right half (check assembly).")
            sys.exit(2)

    # Basic PDF size sanity (avoid empty/corrupt exports)
    tiny: list[str] = []
    for rel in includes:
        path = Path(rel.as_posix())
        if path.suffix.lower() == ".pdf" and path.exists():
            if path.stat().st_size < 10_000:
                tiny.append(str(path))
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            pdf = path.with_suffix(".pdf")
            if pdf.exists() and pdf.stat().st_size < 10_000:
                tiny.append(str(pdf))
    if tiny:
        print("[fail] Tiny PDFs detected (likely corrupt/empty):")
        for p in tiny:
            print("  -", p)
        sys.exit(2)

    print("[ok] latex改.txt figure assets look consistent.")


if __name__ == "__main__":
    main()
