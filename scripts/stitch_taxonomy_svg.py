"""
Stitch Figure 1 Constraint Lifecycle Taxonomy SVGs into Master Vector Graphic
=============================================================================
Combines:
  1. D1_parameter_provenance_vector.svg
  2. D2_application_stage_vector.svg
  3. D3_Inference_Telemetry_Requirements_vector.svg
  4. figure_D4_implementation_fidelity.svg
  5. D5_Evaluation_Validity_Harm_Reporting.svg

According to the layout of:
  ChatGPT Image 2026年9月13日 01_00_18.png (1448 x 1086 px)
"""

import os
import re
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import fitz

ROOT = Path(r"E:\论文1")
OUTPUTS = ROOT / "outputs"
SCRATCH = Path(r"C:\Users\lidong\.gemini\antigravity\brain\97432196-e3d0-481a-aa8e-b27440ab53f4\scratch")
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

FILES = {
    'D1': OUTPUTS / "D1_parameter_provenance_vector.svg",
    'D2': OUTPUTS / "D2_application_stage_vector.svg",
    'D3': OUTPUTS / "D3_Inference_Telemetry_Requirements_vector.svg",
    'D4': OUTPUTS / "figure_D4_implementation_fidelity.svg",
    'D5': OUTPUTS / "D5_Evaluation_Validity_Harm_Reporting.svg",
}

# Grid layout parameters (1448 x 1086 canvas)
# Canvas width: 1448, height: 1086
LAYOUT = {
    'D1': {'x': 5, 'y': 78, 'width': 717, 'height': 377},
    'D2': {'x': 726, 'y': 78, 'width': 717, 'height': 377},
    'D3': {'x': 5, 'y': 460, 'width': 717, 'height': 326},
    'D4': {'x': 726, 'y': 460, 'width': 717, 'height': 326},
    'D5': {'x': 5, 'y': 791, 'width': 1438, 'height': 285},
}

def namespace_svg(svg_path: Path, prefix: str):
    """
    Read an SVG, namespace all IDs and CSS classes to prevent collisions,
    and return the cleaned inner XML and its viewBox.
    """
    content = svg_path.read_text(encoding='utf-8')
    
    # Strip xml declaration and doctype if present
    content = re.sub(r'<\?xml[^>]*\?>', '', content)
    content = re.sub(r'<!DOCTYPE[^>]*>', '', content)
    content = content.strip()

    # Polish baseline-shift in D2 and D3 for clean typography without glyph overlap
    if prefix == 'd2':
        content = content.replace(
            '<text class="math" x="58" y="305" text-anchor="middle" font-size="23" font-style="italic">f<tspan baseline-shift="sub" font-size="15">phys</tspan>(θ)</text>',
            '<text class="math" x="58" y="303" text-anchor="middle" font-size="22" font-style="italic">f<tspan dy="4" font-size="14">phys</tspan><tspan dy="-4" font-size="22">(θ)</tspan></text>'
        )
    if prefix == 'd3':
        content = re.sub(
            r'<text x="103" y="155" class="serif math" text-anchor="middle">\s*P<tspan baseline-shift="sub" font-size="12">t[−-]L:t</tspan>\s*</text>',
            '<text x="103" y="155" class="serif math" text-anchor="middle">P<tspan dy="5" font-size="13">t−L:t</tspan></text>',
            content
        )

    tree = ET.fromstring(content)
    viewBox = tree.attrib.get('viewBox', '0 0 100 100')
    orig_w = tree.attrib.get('width', '100')
    orig_h = tree.attrib.get('height', '100')

    # Find all IDs defined in this SVG
    id_map = {}
    for elem in tree.iter():
        if 'id' in elem.attrib:
            old_id = elem.attrib['id']
            new_id = f"{prefix}_{old_id}"
            id_map[old_id] = new_id
            elem.attrib['id'] = new_id

    # Update all references to IDs (url(#id), href="#id", etc.)
    for elem in tree.iter():
        for attr, val in list(elem.attrib.items()):
            for old_id, new_id in id_map.items():
                if f"url(#{old_id})" in val:
                    elem.attrib[attr] = val.replace(f"url(#{old_id})", f"url(#{new_id})")
                elif f"url('#{old_id}')" in val:
                    elem.attrib[attr] = val.replace(f"url('#{old_id}')", f"url('#{new_id}')")
                elif val == f"#{old_id}":
                    elem.attrib[attr] = f"#{new_id}"

    # Also update any url(#...) inside <style> tags
    for elem in tree.iter():
        if elem.tag.endswith('style') and elem.text:
            s_text = elem.text
            for old_id, new_id in id_map.items():
                s_text = s_text.replace(f"url(#{old_id})", f"url(#{new_id})")
            elem.text = s_text

    # Namespace CSS classes
    # Find all classes mentioned in class attributes
    classes = set()
    for elem in tree.iter():
        if 'class' in elem.attrib:
            for c in elem.attrib['class'].split():
                classes.add(c)

    if classes:
        # Update class attributes
        for elem in tree.iter():
            if 'class' in elem.attrib:
                old_classes = elem.attrib['class'].split()
                new_classes = [f"{prefix}_{c}" for c in old_classes]
                elem.attrib['class'] = " ".join(new_classes)

        # Update <style> blocks
        for elem in tree.iter():
            if elem.tag.endswith('style') and elem.text:
                s_text = elem.text
                for c in classes:
                    # Replace .class with .prefix_class (with word boundary)
                    s_text = re.sub(rf'\.{c}\b', f'.{prefix}_{c}', s_text)
                elem.text = s_text

    # Convert back to XML string
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    inner_svg = ET.tostring(tree, encoding='unicode')
    # Clean ns0 namespace prefixes
    inner_svg = re.sub(r'</?ns0:', lambda m: '</' if m.group().startswith('</') else '<', inner_svg)
    inner_svg = re.sub(r'\s*xmlns:ns0="[^"]*"', '', inner_svg)
    
    # Strip the outer <svg...> and </svg>
    start_idx = inner_svg.find('>') + 1
    end_idx = inner_svg.rfind('</svg>')
    inner_body = inner_svg[start_idx:end_idx].strip()

    return {
        'viewBox': viewBox,
        'width': orig_w,
        'height': orig_h,
        'body': inner_body,
    }

def build_master_svg():
    print("1. Processing and namespacing all 5 SVGs...")
    processed = {}
    for name, path in FILES.items():
        processed[name] = namespace_svg(path, prefix=name.lower())
        print(f"  [{name}] viewBox={processed[name]['viewBox']}")

    print("2. Assembling master SVG canvas (1448 x 1086)...")
    
    # Master title and subtitle matching reference image
    title_text = "Constraint Lifecycle Taxonomy"
    subtitle_text = "Five orthogonal dimensions for auditing physics-constrained forecasting studies"

    svg_parts = []
    svg_parts.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    svg_parts.append('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1448" height="1086" viewBox="0 0 1448 1086">')
    
    # Master canvas background
    svg_parts.append('  <!-- Canvas Background -->')
    svg_parts.append('  <rect width="1448" height="1086" fill="#ffffff"/>')

    # Top Header Title & Subtitle
    svg_parts.append('  <!-- Master Taxonomy Header -->')
    svg_parts.append('  <g id="master-header">')
    svg_parts.append(f'    <text x="724" y="38" text-anchor="middle" font-family="\'Tinos\', \'Liberation Serif\', \'DejaVu Serif\', \'Times New Roman\', serif" font-size="39" font-weight="bold" fill="#111111">{title_text}</text>')
    svg_parts.append(f'    <text x="724" y="63" text-anchor="middle" font-family="\'Tinos\', \'Liberation Serif\', \'DejaVu Serif\', \'Times New Roman\', serif" font-size="22" font-style="normal" fill="#111111">{subtitle_text}</text>')
    svg_parts.append('  </g>')

    # Place D1, D2, D3, D4, D5 as nested SVG viewports
    for name in ['D1', 'D2', 'D3', 'D4', 'D5']:
        cfg = LAYOUT[name]
        p = processed[name]
        svg_parts.append(f'\n  <!-- ==================== Panel {name} ==================== -->')
        svg_parts.append(f'  <svg id="panel_{name}" x="{cfg["x"]}" y="{cfg["y"]}" width="{cfg["width"]}" height="{cfg["height"]}" viewBox="{p["viewBox"]}" preserveAspectRatio="none">')
        svg_parts.append(p['body'])
        svg_parts.append('  </svg>')

    svg_parts.append('\n</svg>')

    full_svg = "\n".join(svg_parts)
    
    out_svg = OUTPUTS / "constraint_lifecycle_taxonomy_assembled.svg"
    out_svg.write_text(full_svg, encoding='utf-8')
    print(f"  -> Saved assembled SVG to: {out_svg}")
    
    # Also save to scratch for verification
    scratch_svg = SCRATCH / "constraint_lifecycle_taxonomy_assembled.svg"
    scratch_svg.write_text(full_svg, encoding='utf-8')

    return out_svg

def render_and_verify(svg_file: Path):
    print("\n3. Rendering pixel-perfect PDF via Edge Engine...")
    pdf_out = OUTPUTS / "constraint_lifecycle_taxonomy_assembled.pdf"
    png_out = OUTPUTS / "constraint_lifecycle_taxonomy_assembled.png"
    scratch_png = SCRATCH / "constraint_lifecycle_taxonomy_assembled.png"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
  size: 1448px 1086px;
  margin: 0;
}}
html, body {{
  margin: 0;
  padding: 0;
  width: 1448px;
  height: 1086px;
  overflow: hidden;
  background: white;
}}
svg {{
  width: 1448px;
  height: 1086px;
  display: block;
}}
</style>
</head>
<body>
{svg_file.read_text(encoding='utf-8')}
</body>
</html>"""

    temp_html = OUTPUTS / "temp_assemble_render.html"
    temp_html.write_text(html_content, encoding='utf-8')

    cmd = [
        EDGE_EXE,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_out.resolve()}",
        temp_html.as_uri()
    ]
    subprocess.run(cmd, check=True)
    print(f"  -> Saved vector PDF: {pdf_out}")

    # Render PNG at 150 DPI for fast preview and 300 DPI for publication
    print("4. Rendering PNG preview via PyMuPDF...")
    doc = fitz.open(str(pdf_out))
    page = doc[0]
    pix = page.get_pixmap(dpi=150)
    pix.save(str(scratch_png))
    print(f"  -> Saved scratch PNG preview: {scratch_png} ({pix.width}x{pix.height})")

    pix_pub = page.get_pixmap(dpi=300)
    pix_pub.save(str(png_out))
    print(f"  -> Saved publication 300 DPI PNG: {png_out} ({pix_pub.width}x{pix_pub.height})")

    if temp_html.exists():
        temp_html.unlink()

    # Save aliases and deploy to manuscript folders
    print("\n5. Deploying to manuscript directories...")
    import shutil
    fig_svg = OUTPUTS / "fig_constraint_lifecycle.svg"
    fig_pdf = OUTPUTS / "fig_constraint_lifecycle.pdf"
    fig_png = OUTPUTS / "fig_constraint_lifecycle.png"
    shutil.copyfile(svg_file, fig_svg)
    shutil.copyfile(pdf_out, fig_pdf)
    shutil.copyfile(png_out, fig_png)
    print(f"  -> Saved {fig_svg}")
    print(f"  -> Saved {fig_pdf}")
    print(f"  -> Saved {fig_png}")

    deploy_targets = [
        ROOT / "01_论文投稿材料" / "active_manuscript" / "figures" / "fig_constraint_lifecycle.pdf",
        ROOT / "01_论文投稿材料" / "chinese_manuscript" / "figures" / "fig_constraint_lifecycle.pdf",
        ROOT / "02_论文正文与LaTeX" / "fig_constraint_lifecycle.pdf",
        ROOT / "03_图表与绘图素材" / "fig_constraint_lifecycle.pdf",
        ROOT / "03_图表与绘图素材" / "fig_constraint_lifecycle.png",
    ]
    for target in deploy_targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == '.pdf':
            shutil.copyfile(pdf_out, target)
        elif target.suffix == '.png':
            shutil.copyfile(png_out, target)
        print(f"  -> Deployed to {target}")

if __name__ == '__main__':
    out_svg = build_master_svg()
    render_and_verify(out_svg)
    print("\nMaster SVG assembly completed successfully!")
