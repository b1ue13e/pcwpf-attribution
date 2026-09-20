from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from PIL import Image


_INCLUDE_RE = re.compile(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}")


def _bases_from_tex(tex_path: Path) -> list[str]:
    text = tex_path.read_text(encoding="utf-8", errors="ignore")
    out: list[str] = []
    for m in _INCLUDE_RE.finditer(text):
        name = m.group(1).strip()
        p = Path(name)
        base = p.name
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
            base = p.with_suffix("").name
        out.append(base)
    return sorted(set(out))


def _is_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(8)
        return head.startswith(b"%PDF-")
    except Exception:
        return False


def main() -> int:
    proc = subprocess.run([sys.executable, "make_paper_figures.py"], capture_output=True, text=True)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")

    if proc.returncode != 0 or "Traceback" in out:
        print(out)
        return 1

    if "findfont:" in out:
        print(out)
        print("\n[fail] Matplotlib emitted 'findfont' warnings; font embedding may be inconsistent.")
        return 2

    figure_dir = Path("figure")
    tex_files = [Path("ultra_lsnt_paper_complete.tex"), Path("ultra_lsnt_paper.tex")]
    bases: set[str] = set()
    for tex in tex_files:
        if tex.exists():
            bases.update(_bases_from_tex(tex))

    missing_pdf: list[str] = []
    bad_pdf: list[str] = []
    small_png: list[str] = []

    for base in sorted(bases):
        pdf = figure_dir / f"{base}.pdf"
        if not pdf.exists():
            missing_pdf.append(base)
        else:
            if (pdf.stat().st_size < 10_000) or (not _is_pdf(pdf)):
                bad_pdf.append(base)

        png = figure_dir / f"{base}.png"
        if png.exists():
            try:
                with Image.open(png) as im:
                    w, h = im.size
                if w < 1200 or h < 800:
                    small_png.append(f"{base} ({w}x{h})")
            except Exception:
                pass

    if missing_pdf or bad_pdf:
        if missing_pdf:
            print("[fail] Missing PDFs for:", ", ".join(missing_pdf))
        if bad_pdf:
            print("[fail] Corrupt/tiny PDFs for:", ", ".join(bad_pdf))
        return 3

    if small_png:
        print("[warn] Low-resolution PNGs (PDF is preferred for submission):")
        for s in small_png:
            print("  -", s)

    print("[ok] Figures rebuilt; PDFs exist and look structurally valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
