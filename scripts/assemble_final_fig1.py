"""
Assemble Master Figure 1 (Constraint Lifecycle Taxonomy) with Authentic Micro-Plots
===================================================================================
Replaces the 7 coordinate-based micro-plots with authentic, publication-grade vector graphics:
  - D1 / Historical-static: Exact discrete lookup table step function & SCADA scatter
  - D2 / Post-hoc correction: Feasibility projection / clipping trajectory
  - D3 / None: Historical SCADA sequence into forecast model -> y_hat
  - D5 / Constraint on/off: Real DLinear checkpoints (lambda=0 vs lambda=1.0)
  - D5 / Both: 3-way model ablation (Full vs Ablated vs None)
  - D5 / Reported: Continuous corruption crossover (alpha* ~ 0.07, alpha in [0.0, 0.5])
  - D5 / Not reported: Genuine vs scrambled empirical prior

Retains 100% of all non-coordinate schematics and typography.
Exports SVG, PDF, and high-resolution PNG.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import fitz  # PyMuPDF

ROOT = Path(r"E:\论文1")
OUTPUTS = ROOT / "outputs"
SCRATCH = Path(r"C:\Users\lidong\.gemini\antigravity\brain\97432196-e3d0-481a-aa8e-b27440ab53f4\scratch")
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

MASTER_SVG_TEMPLATE = OUTPUTS / "fig_constraint_lifecycle_template.svg"
FINAL_SVG = OUTPUTS / "fig_constraint_lifecycle.svg"
ASSEMBLED_SVG = OUTPUTS / "constraint_lifecycle_taxonomy_assembled.svg"
FINAL_PDF = OUTPUTS / "fig_constraint_lifecycle.pdf"
FINAL_PNG = OUTPUTS / "fig_constraint_lifecycle.png"

MICRO_FILES = {
    'd1_hs': OUTPUTS / "fig1_d1_historical_static.svg",
    'd2_ph': OUTPUTS / "fig1_d2_posthoc.svg",
    'd3_none': OUTPUTS / "fig1_d3_none.svg",
    'd5_onoff': OUTPUTS / "fig1_d5_constraint_onoff.svg",
    'd5_both': OUTPUTS / "fig1_d5_both.svg",
    'd5_rep': OUTPUTS / "fig1_d5_reported.svg",
    'd5_notrep': OUTPUTS / "fig1_d5_notreported.svg",
}

def clean_and_namespace_svg(svg_path: Path, prefix: str):
    """Namespaces all IDs and returns viewBox and sanitized inner XML."""
    content = svg_path.read_text(encoding='utf-8')
    content = re.sub(r'<\?xml[^>]*\?>', '', content)
    content = re.sub(r'<!DOCTYPE[^>]*>', '', content).strip()
    
    tree = ET.fromstring(content)
    viewBox = tree.attrib.get('viewBox', '0 0 100 100')
    
    # Map IDs
    id_map = {}
    for elem in tree.iter():
        if 'id' in elem.attrib:
            old_id = elem.attrib['id']
            new_id = f"{prefix}_{old_id}"
            id_map[old_id] = new_id
            elem.attrib['id'] = new_id
            
    # Update references
    for elem in tree.iter():
        for attr, val in list(elem.attrib.items()):
            for old_id, new_id in id_map.items():
                if f"url(#{old_id})" in val:
                    elem.attrib[attr] = val.replace(f"url(#{old_id})", f"url(#{new_id})")
                elif f"url('#{old_id}')" in val:
                    elem.attrib[attr] = val.replace(f"url('#{old_id}')", f"url('#{new_id}')")
                elif val == f"#{old_id}":
                    elem.attrib[attr] = f"#{new_id}"
                    
    for elem in tree.iter():
        if elem.tag.endswith('style') and elem.text:
            s_text = elem.text
            for old_id, new_id in id_map.items():
                s_text = s_text.replace(f"url(#{old_id})", f"url(#{new_id})")
            elem.text = s_text
            
    # Convert to string
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    raw_xml = ET.tostring(tree, encoding='unicode')
    raw_xml = re.sub(r'</?ns0:', lambda m: '</' if m.group().startswith('</') else '<', raw_xml)
    raw_xml = re.sub(r'\s*xmlns:ns0="[^"]*"', '', raw_xml)
    
    start_idx = raw_xml.find('>') + 1
    end_idx = raw_xml.rfind('</svg>')
    inner_body = raw_xml[start_idx:end_idx].strip()
    
    return viewBox, inner_body

def build_assembled_figure():
    print("1. Preparing namespaced micro-plots...")
    micro_data = {}
    for key, path in MICRO_FILES.items():
        vb, inner = clean_and_namespace_svg(path, prefix=key)
        micro_data[key] = {'viewBox': vb, 'inner': inner}
        print(f"  -> {key}: viewBox={vb}")

    # Read base SVG
    print("2. Reading layout base SVG...")
    base_text = MASTER_SVG_TEMPLATE.read_text(encoding='utf-8')
    
    # --------------------------------------------------------------------------
    # Panel D1 Replacement (Historical-static)
    # --------------------------------------------------------------------------
    print("3. Replacing D1 Historical-static micro-plot...")
    d1_pattern = re.compile(
        r'<g stroke="#111" fill="none" stroke-width="1\.3" marker-end="url\(#d1_arrowBlack\)">.*?'
        r'<text x="145" y="160"[^>]*transform="rotate\(-90 145 160\)">Power</text>',
        re.DOTALL
    )
    d1_vb = micro_data['d1_hs']['viewBox']
    d1_inner = micro_data['d1_hs']['inner']
    d1_replacement = (
        f'<!-- Replaced with authentic D1 Historical-static micro-plot -->\n'
        f'    <svg x="135" y="112" width="115" height="142" viewBox="{d1_vb}" preserveAspectRatio="xMidYMid meet">\n'
        f'{d1_inner}\n'
        f'    </svg>'
    )
    base_text, count_d1 = d1_pattern.subn(d1_replacement, base_text)
    print(f"  -> D1 replacement count: {count_d1}")
    assert count_d1 == 1, f"Failed to replace D1 (count={count_d1})"

    # --------------------------------------------------------------------------
    # Panel D2 Replacement (Post-hoc correction)
    # --------------------------------------------------------------------------
    print("4. Replacing D2 Post-hoc correction micro-plot...")
    d2_pattern = re.compile(
        r'<line x1="386" y1="328" x2="517" y2="328".*?'
        r'<text class="d2_serif" x="570" y="312"[^>]*>projected output ŷ.*?</text>',
        re.DOTALL
    )
    d2_vb = micro_data['d2_ph']['viewBox']
    d2_inner = micro_data['d2_ph']['inner']
    d2_replacement = (
        f'<!-- Replaced with authentic D2 Post-hoc correction micro-plot -->\n'
        f'    <svg x="390" y="260" width="250" height="92" viewBox="{d2_vb}" preserveAspectRatio="xMidYMid meet">\n'
        f'{d2_inner}\n'
        f'    </svg>'
    )
    base_text, count_d2 = d2_pattern.subn(d2_replacement, base_text)
    print(f"  -> D2 replacement count: {count_d2}")
    assert count_d2 == 1, f"Failed to replace D2 (count={count_d2})"

    # --------------------------------------------------------------------------
    # Panel D3 Replacement (None)
    # --------------------------------------------------------------------------
    print("5. Replacing D3 None micro-plot...")
    d3_pattern = re.compile(
        r'<text x="103" y="155" class="d3_serif d3_math"[^>]*>.*?t[−-]L:t.*?</text>\s*'
        r'<line x1="61" y1="243" x2="124" y2="243" stroke="#111" stroke-width="1\.1" marker-end="url\(#d3_arrow\)" />.*?'
        r'<text x="219" y="204" class="d3_serif d3_math" text-anchor="middle">ŷ</text>',
        re.DOTALL
    )
    d3_vb = micro_data['d3_none']['viewBox']
    d3_inner = micro_data['d3_none']['inner']
    d3_replacement = (
        f'<!-- Replaced with authentic D3 None micro-plot -->\n'
        f'    <svg x="14" y="130" width="208" height="148" viewBox="{d3_vb}" preserveAspectRatio="xMidYMid meet">\n'
        f'{d3_inner}\n'
        f'    </svg>'
    )
    base_text, count_d3 = d3_pattern.subn(d3_replacement, base_text)
    print(f"  -> D3 replacement count: {count_d3}")
    assert count_d3 == 1, f"Failed to replace D3 (count={count_d3})"

    # --------------------------------------------------------------------------
    # Panel D5 Replacement (Constraint on/off, Both, Reported, Not reported)
    # --------------------------------------------------------------------------
    print("6. Replacing D5 micro-plots...")
    # Card 2: Constraint on/off
    d5_onoff_pattern = re.compile(
        r'<line x1="62" y1="206" x2="230" y2="206".*?'
        r'<text x="107" y="145"[^>]*>constrained \(λ &gt; 0\)</text>',
        re.DOTALL
    )
    d5_onoff_vb = micro_data['d5_onoff']['viewBox']
    d5_onoff_inner = micro_data['d5_onoff']['inner']
    d5_onoff_rep = (
        f'<!-- Replaced with authentic D5 Constraint on/off micro-plot -->\n'
        f'    <svg x="36" y="112" width="216" height="132" viewBox="{d5_onoff_vb}" preserveAspectRatio="xMidYMid meet">\n'
        f'{d5_onoff_inner}\n'
        f'    </svg>'
    )
    base_text, count_d5_1 = d5_onoff_pattern.subn(d5_onoff_rep, base_text)
    print(f"  -> D5 Constraint on/off replacement count: {count_d5_1}")
    assert count_d5_1 == 1, f"Failed to replace D5 on/off (count={count_d5_1})"

    # Card 3: Both
    d5_both_pattern = re.compile(
        r'<line x1="548" y1="208" x2="725" y2="208".*?'
        r'<text x="658" y="169"[^>]*>no constraint</text>',
        re.DOTALL
    )
    d5_both_vb = micro_data['d5_both']['viewBox']
    d5_both_inner = micro_data['d5_both']['inner']
    d5_both_rep = (
        f'<!-- Replaced with authentic D5 Both micro-plot -->\n'
        f'    <svg x="522" y="125" width="216" height="119" viewBox="{d5_both_vb}" preserveAspectRatio="xMidYMid meet">\n'
        f'{d5_both_inner}\n'
        f'    </svg>'
    )
    base_text, count_d5_2 = d5_both_pattern.subn(d5_both_rep, base_text)
    print(f"  -> D5 Both replacement count: {count_d5_2}")
    assert count_d5_2 == 1, f"Failed to replace D5 Both (count={count_d5_2})"

    # Card 4: Reported
    d5_rep_pattern = re.compile(
        r'<line x1="800" y1="211" x2="1008" y2="211".*?'
        r'<rect x="1002\.0" y="119\.0" width="6" height="6" fill="#ef2b2d" />',
        re.DOTALL
    )
    d5_rep_vb = micro_data['d5_rep']['viewBox']
    d5_rep_inner = micro_data['d5_rep']['inner']
    d5_rep_replacement = (
        f'<!-- Replaced with authentic D5 Reported micro-plot -->\n'
        f'    <svg x="792" y="112" width="216" height="132" viewBox="{d5_rep_vb}" preserveAspectRatio="xMidYMid meet">\n'
        f'{d5_rep_inner}\n'
        f'    </svg>'
    )
    base_text, count_d5_3 = d5_rep_pattern.subn(d5_rep_replacement, base_text)
    print(f"  -> D5 Reported replacement count: {count_d5_3}")
    assert count_d5_3 == 1, f"Failed to replace D5 Reported (count={count_d5_3})"

    # Card 5: Not reported
    d5_notrep_pattern = re.compile(
        r'<line x1="1073" y1="210" x2="1275" y2="210".*?'
        r'<text x="1116" y="137"[^>]*>Scrambled prior.*?</text>',
        re.DOTALL
    )
    d5_notrep_vb = micro_data['d5_notrep']['viewBox']
    d5_notrep_inner = micro_data['d5_notrep']['inner']
    d5_notrep_rep = (
        f'<!-- Replaced with authentic D5 Not reported micro-plot -->\n'
        f'    <svg x="1060" y="112" width="216" height="132" viewBox="{d5_notrep_vb}" preserveAspectRatio="xMidYMid meet">\n'
        f'{d5_notrep_inner}\n'
        f'    </svg>'
    )
    base_text, count_d5_4 = d5_notrep_pattern.subn(d5_notrep_rep, base_text)
    print(f"  -> D5 Not reported replacement count: {count_d5_4}")
    assert count_d5_4 == 1, f"Failed to replace D5 Not reported (count={count_d5_4})"

    # Save final assembled SVG
    FINAL_SVG.write_text(base_text, encoding='utf-8')
    ASSEMBLED_SVG.write_text(base_text, encoding='utf-8')
    print(f"7. Successfully saved assembled SVG -> {FINAL_SVG}")

def convert_to_pdf_and_png():
    print("8. Converting assembled SVG to PDF & PNG...")
    temp_html = SCRATCH / "render_fig1_final.html"
    temp_pdf = SCRATCH / "render_fig1_final.pdf"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{
    size: 1448px 1086px;
    margin: 0;
  }}
  body {{
    margin: 0;
    padding: 0;
    background: #ffffff;
  }}
  svg {{
    display: block;
    width: 1448px;
    height: 1086px;
  }}
</style>
</head>
<body>
{FINAL_SVG.read_text(encoding='utf-8')}
</body>
</html>"""
    temp_html.write_text(html_content, encoding='utf-8')
    
    cmd = [
        EDGE_EXE,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={temp_pdf}",
        "--no-pdf-header-footer",
        str(temp_html)
    ]
    subprocess.run(cmd, check=True)
    
    # Crop / sync PDF
    doc = fitz.open(temp_pdf)
    page = doc[0]
    # Render 200 DPI preview PNG
    pix = page.get_pixmap(dpi=200)
    pix.save(FINAL_PNG)
    pix.save(OUTPUTS / "constraint_lifecycle_taxonomy_assembled.png")
    doc.close()
    
    shutil.copy2(temp_pdf, FINAL_PDF)
    shutil.copy2(temp_pdf, OUTPUTS / "constraint_lifecycle_taxonomy_assembled.pdf")
    print(f"  -> Saved {FINAL_PDF}")
    print(f"  -> Saved {FINAL_PNG}")
    
    # Sync to paper manuscript paths
    sync_targets = [
        ROOT / "01_论文投稿材料" / "active_manuscript" / "figures" / "fig_constraint_lifecycle.pdf",
        ROOT / "01_论文投稿材料" / "chinese_manuscript" / "figures" / "fig_constraint_lifecycle.pdf",
        ROOT / "02_论文正文与LaTeX" / "fig_constraint_lifecycle.pdf",
        ROOT / "03_图表与绘图素材" / "fig_constraint_lifecycle.pdf",
        ROOT / "03_图表与绘图素材" / "fig_constraint_lifecycle.png",
    ]
    for target in sync_targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == '.pdf':
            shutil.copy2(FINAL_PDF, target)
        elif target.suffix == '.png':
            shutil.copy2(FINAL_PNG, target)
        print(f"  -> Synced to {target}")

if __name__ == '__main__':
    build_assembled_figure()
    convert_to_pdf_and_png()
    print("ALL DONE SUCCESSFULLY!")
