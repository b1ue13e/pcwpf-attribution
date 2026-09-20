import os
import shutil
from pathlib import Path
import cairosvg
import fitz  # PyMuPDF

def main():
    root = Path(r"E:\论文1")
    src_svg = root / "outputs" / "Figure2_PRISMA_redesign.svg"
    assert src_svg.exists(), f"Source SVG does not exist: {src_svg}"

    print(f"Reading source SVG from: {src_svg}")

    # Generate temporary PDF and PNG
    temp_pdf = root / "outputs" / "_temp_fig2.pdf"
    temp_png = root / "outputs" / "_temp_fig2.png"

    print("Converting SVG to vector PDF via cairosvg...")
    cairosvg.svg2pdf(url=str(src_svg), write_to=str(temp_pdf))

    # Normalize PDF header to PDF-1.4 to eliminate pdflatex version mismatch warning
    with open(temp_pdf, "rb") as f:
        pdf_data = f.read()

    if pdf_data.startswith(b"%PDF-1.7\n"):
        pdf_data = b"%PDF-1.4\n" + pdf_data[9:]
        with open(temp_pdf, "wb") as f:
            f.write(pdf_data)
        print(f"Normalized PDF header to PDF-1.4 (exact byte length preserved: {len(pdf_data)} bytes)")
    else:
        print(f"Generated vector PDF: {len(pdf_data)} bytes")

    # Verify normalized PDF integrity
    doc_check = fitz.open(str(temp_pdf))
    assert doc_check.is_pdf and doc_check.page_count == 1
    pdf_format = doc_check.metadata.get("format")
    print(f"Verified PDF format via PyMuPDF: {pdf_format}")
    doc_check.close()

    print("Rendering high-res 600 DPI PNG via PyMuPDF...")
    doc = fitz.open(str(temp_pdf))
    page = doc[0]
    pix = page.get_pixmap(dpi=600)
    pix.save(str(temp_png))
    print(f"Generated 600 DPI PNG: {pix.width}x{pix.height}, {temp_png.stat().st_size} bytes")
    doc.close()

    # Destination mapping
    targets = [
        # 1. Active manuscript (English)
        (root / "01_论文投稿材料" / "active_manuscript" / "figures" / "prisma_flow_diagram.pdf", temp_pdf),
        (root / "01_论文投稿材料" / "active_manuscript" / "figures" / "prisma_flow_diagram.png", temp_png),
        (root / "01_论文投稿材料" / "active_manuscript" / "figures" / "prisma_flow_diagram.svg", src_svg),
        (root / "01_论文投稿材料" / "active_manuscript" / "figures" / "Figure2_PRISMA_vector.pdf", temp_pdf),
        (root / "01_论文投稿材料" / "active_manuscript" / "figures" / "Figure2_PRISMA_vector.svg", src_svg),

        # 2. Chinese manuscript
        (root / "01_论文投稿材料" / "chinese_manuscript" / "figures" / "prisma_flow_diagram.pdf", temp_pdf),
        (root / "01_论文投稿材料" / "chinese_manuscript" / "figures" / "Figure2_PRISMA_vector.pdf", temp_pdf),

        # 3. Outputs
        (root / "outputs" / "Figure2_PRISMA_vector.pdf", temp_pdf),
        (root / "outputs" / "Figure2_PRISMA_vector.png", temp_png),

        # 4. Figures assets repository
        (root / "03_图表与绘图素材" / "Figure2_PRISMA_vector.pdf", temp_pdf),
        (root / "03_图表与绘图素材" / "Figure2_PRISMA_vector.png", temp_png),
        (root / "03_图表与绘图素材" / "Figure2_PRISMA_vector.svg", src_svg),
        (root / "03_图表与绘图素材" / "prisma_flow_diagram.pdf", temp_pdf),
        (root / "03_图表与绘图素材" / "prisma_flow_diagram.png", temp_png),
        (root / "03_图表与绘图素材" / "prisma_flow_diagram.svg", src_svg),
        (root / "03_图表与绘图素材" / "figure2.pdf", temp_pdf),
        (root / "03_图表与绘图素材" / "figure2.png", temp_png),
        (root / "03_图表与绘图素材" / "Figure_2_Final.pdf", temp_pdf),
        (root / "03_图表与绘图素材" / "Figure_2_Final.png", temp_png),

        # 5. 02_论文正文与LaTeX
        (root / "02_论文正文与LaTeX" / "prisma_flow_diagram.pdf", temp_pdf),
        (root / "02_论文正文与LaTeX" / "figure2.pdf", temp_pdf),
        (root / "02_论文正文与LaTeX" / "figure2.png", temp_png),

        # 6. 02_论文正文与LaTeX/latex_current_pack (coordinator mentioned)
        (root / "02_论文正文与LaTeX" / "latex_current_pack" / "figure2.pdf", temp_pdf),
        (root / "02_论文正文与LaTeX" / "latex_current_pack" / "figure2.png", temp_png),
        (root / "02_论文正文与LaTeX" / "latex_current_pack" / "Figure_2_Final.pdf", temp_pdf),
        (root / "02_论文正文与LaTeX" / "latex_current_pack" / "Figure_2_Final.png", temp_png),
    ]

    for dest, src in targets:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"Copied to: {dest.relative_to(root)} ({dest.stat().st_size} bytes)")

    # Clean up temp files
    temp_pdf.unlink()
    temp_png.unlink()
    print("Deployment completed successfully!")

if __name__ == "__main__":
    main()
