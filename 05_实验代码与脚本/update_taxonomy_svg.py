"""Update constraint lifecycle taxonomy SVG and render PDF/PNG.

Comprehensive refinements:
1. D1:
   - Break 'Historical-static' and 'Learned-online' headers into two lines.
   - Clean spacing and single label 'Not specified'.
2. D2:
   - Explicit text coordinate layout for Training Loss: L = L_task + lambda L_phys (prevents tspan dy overlap).
   - Explicit text coordinate layout for X in R^{NxTxd} in Data Generation.
   - Remove coordinate tick numbers in Post-hoc correction, enlarge labels.
3. D3:
   - Remove coordinate ticks and labels (0.0, 0.5, 1.0).
   - Clean padding inside Contemporaneous state box.
4. D4:
   - Clean file icon (no micro line clutter).
   - Enlarge formulas and parity descriptions.
5. D5:
   - Symmetric, collision-free subtitles:
     Reported: 'Failure boundary disclosed'
     Not reported: 'No failure boundary documented'
   - Remove all coordinate tick numbers from all mini plots.
   - y-axis line ends cleanly at y=125 in Reported and Not reported.
   - Position rotated y-axis labels at x=760 and x=1046 so they never collide with the axis.
   - Purely schematic failure boundary at x=848 labeled 'failure boundary alpha*'.
   - Enlarge regime, legend, and callout labels.
"""

from __future__ import annotations
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import cairosvg

def update_svg_content(content: str) -> str:
    # -------------------------------------------------------------
    # 1. Panel D1 updates
    # -------------------------------------------------------------
    content = content.replace(
        '.small { font-size: 9.8px; }',
        '.small { font-size: 11px; }'
    )
    content = content.replace(
        '.d1_body { font-size: 10.8px; }',
        '.d1_body { font-size: 11.5px; }'
    )
    content = content.replace(
        '.d1_head { font-size: 12.7px; font-weight: 700; }',
        '.d1_head { font-size: 13.2px; font-weight: 700; }'
    )
    content = content.replace(
        '.d1_title { font-size: 17px; font-weight: 700; }',
        '.d1_title { font-size: 18.5px; font-weight: 700; }'
    )
    content = content.replace(
        '.d1_subtitle { font-size: 13px; font-style: italic; font-weight: 600; }',
        '.d1_subtitle { font-size: 13.5px; font-style: italic; font-weight: 600; }'
    )
    content = content.replace(
        '.d1_math { font-size: 16px; font-style: italic; }',
        '.d1_math { font-size: 17.5px; font-style: italic; }'
    )

    # D1 Column 1 (Intrinsic)
    old_d1_col1 = """  <text x="67.5" y="70" text-anchor="middle" class="d1_serif d1_head">Intrinsic</text>
  <text x="67.5" y="86" text-anchor="middle" class="d1_serif d1_body">Governing laws /</text>
  <text x="67.5" y="100" text-anchor="middle" class="d1_serif d1_body">physical limits</text>"""

    new_d1_col1 = """  <text x="67.5" y="70" text-anchor="middle" class="d1_serif d1_head">Intrinsic</text>
  <text x="67.5" y="87" text-anchor="middle" class="d1_serif d1_body">Governing laws /</text>
  <text x="67.5" y="101" text-anchor="middle" class="d1_serif d1_body">physical limits</text>"""
    content = content.replace(old_d1_col1, new_d1_col1)

    # D1 Column 2 (Historical-static): 2-line header
    old_d1_col2 = """  <text x="192.5" y="70" text-anchor="middle" class="d1_serif d1_head" font-size="12.1">Historical-static</text>
  <text x="192.5" y="86" text-anchor="middle" class="d1_serif d1_body">Offline SCADA-</text>
  <text x="192.5" y="100" text-anchor="middle" class="d1_serif d1_body">derived prior</text>"""

    new_d1_col2 = """  <text x="192.5" y="66" text-anchor="middle" class="d1_serif d1_head">Historical-</text>
  <text x="192.5" y="80" text-anchor="middle" class="d1_serif d1_head">static</text>
  <text x="192.5" y="96" text-anchor="middle" class="d1_serif d1_body">Offline SCADA-</text>
  <text x="192.5" y="108" text-anchor="middle" class="d1_serif d1_body">derived prior</text>"""
    content = content.replace(old_d1_col2, new_d1_col2)

    content = content.replace(
        '<text x="201" y="220" text-anchor="middle" class="d1_serif d1_body">Wind speed</text>',
        '<text x="201" y="222" text-anchor="middle" class="d1_serif" font-size="12.5">Wind speed</text>'
    )
    content = content.replace(
        '<text x="145" y="160" text-anchor="middle" class="d1_serif d1_body" transform="rotate(-90 145 160)">Power</text>',
        '<text x="143" y="160" text-anchor="middle" class="d1_serif" font-size="12.5" transform="rotate(-90 143 160)">Power</text>'
    )

    # D1 Column 3 (Learned-online): 2-line header
    old_d1_col3 = """  <text x="317.5" y="70" text-anchor="middle" class="d1_serif d1_head" font-size="12.3">Learned-online</text>
  <text x="317.5" y="86" text-anchor="middle" class="d1_serif d1_body">Adaptive learned</text>
  <text x="317.5" y="100" text-anchor="middle" class="d1_serif d1_body">parameters</text>"""

    new_d1_col3 = """  <text x="317.5" y="66" text-anchor="middle" class="d1_serif d1_head">Learned-</text>
  <text x="317.5" y="80" text-anchor="middle" class="d1_serif d1_head">online</text>
  <text x="317.5" y="96" text-anchor="middle" class="d1_serif d1_body">Adaptive learned</text>
  <text x="317.5" y="108" text-anchor="middle" class="d1_serif d1_body">parameters</text>"""
    content = content.replace(old_d1_col3, new_d1_col3)

    content = content.replace(
        '<text x="324" y="220" text-anchor="middle" class="d1_serif d1_body">online update</text>',
        '<text x="324" y="222" text-anchor="middle" class="d1_serif" font-size="12.5">online update</text>'
    )

    # D1 Column 4 (Unreported): single clean label
    old_d1_col4 = """  <text x="440.5" y="70" text-anchor="middle" class="d1_serif d1_head">Unreported</text>
  <text x="440.5" y="86" text-anchor="middle" class="d1_serif d1_body">Provenance</text>
  <text x="440.5" y="100" text-anchor="middle" class="d1_serif d1_body">unspecified</text>
  <rect x="396" y="125" width="91" height="69" rx="2" fill="#fff" stroke="#5a7590" stroke-width="1.3" stroke-dasharray="4 3" />
  <text x="441.5" y="155" text-anchor="middle" class="d1_serif d1_body">provenance: —</text>
  <text x="441.5" y="174" text-anchor="middle" class="d1_serif d1_body">not specified</text>"""

    new_d1_col4 = """  <text x="440.5" y="70" text-anchor="middle" class="d1_serif d1_head">Unreported</text>
  <text x="440.5" y="87" text-anchor="middle" class="d1_serif d1_body">Provenance</text>
  <text x="440.5" y="101" text-anchor="middle" class="d1_serif d1_body">unspecified</text>
  <rect x="396" y="125" width="91" height="69" rx="2" fill="#fff" stroke="#5a7590" stroke-width="1.3" stroke-dasharray="4 3" />
  <text x="441.5" y="164" text-anchor="middle" class="d1_serif" font-size="13" font-weight="600" fill="#2d3748">Not specified</text>"""
    content = content.replace(old_d1_col4, new_d1_col4)

    # -------------------------------------------------------------
    # 2. Panel D2 updates
    # -------------------------------------------------------------
    content = content.replace(
        ".d2_math { font-family: FreeSerif, 'STIX Two Math', serif; fill:#111; }",
        ".d2_math { font-family: 'DejaVu Math TeX Gyre', 'FreeSerif', 'DejaVu Serif', 'Cambria Math', serif; fill:#111; }"
    )

    # Training loss formula with explicit coordinates (prevents tspan dy overlap)
    old_d2_formula = '<text class="d2_math" x="171" y="162" text-anchor="middle" font-size="28" font-style="italic">ℒ = ℒ<tspan dy="5" font-size="18">task</tspan><tspan dy="-5"> + λℒ</tspan><tspan dy="5" font-size="18">phys</tspan></text>'
    new_d2_formula = """<g class="d2_math" font-style="italic">
    <text x="76" y="162" font-size="28">ℒ</text>
    <text x="103" y="162" font-size="24" font-style="normal">=</text>
    <text x="126" y="162" font-size="28">ℒ</text>
    <text x="144" y="168" font-size="15">task</text>
    <text x="181" y="162" font-size="24" font-style="normal">+</text>
    <text x="204" y="162" font-size="25">λ</text>
    <text x="223" y="162" font-size="28">ℒ</text>
    <text x="241" y="168" font-size="15">phys</text>
  </g>"""
    content = content.replace(old_d2_formula, new_d2_formula)

    # Data generation text sizes & explicit formula coordinates
    content = content.replace(
        '<text class="d2_serif" x="20" y="246" font-size="13">Physics-guided synthetic samples</text>',
        '<text class="d2_serif" x="20" y="246" font-size="15">Physics-guided synthetic samples</text>'
    )
    content = content.replace(
        '<text class="d2_serif" x="47" y="321" text-anchor="middle" font-size="13">physical</text>\n  <text class="d2_serif" x="47" y="335" text-anchor="middle" font-size="13">simulator</text>',
        '<text class="d2_serif" x="47" y="321" text-anchor="middle" font-size="14">physical</text>\n  <text class="d2_serif" x="47" y="336" text-anchor="middle" font-size="14">simulator</text>'
    )

    old_d2_datagen_math = '<text class="d2_math" x="278" y="259" text-anchor="middle" font-size="13.5" font-style="italic">X ∈ R<tspan dy="-4" font-size="9.5">N×T×d</tspan></text>'
    new_d2_datagen_math = """<g class="d2_math" font-style="italic">
    <text x="240" y="259" font-size="15">X</text>
    <text x="256" y="259" font-size="15" font-style="normal">∈</text>
    <text x="273" y="259" font-size="15" font-weight="bold">R</text>
    <text x="286" y="253" font-size="10.5">N×T×d</text>
  </g>"""
    content = content.replace(old_d2_datagen_math, new_d2_datagen_math)

    # Post-hoc correction: replace coordinate ticks and numeric clutter
    old_d2_ticks = """  <line x1="386" y1="316" x2="517" y2="316" stroke="#111" stroke-width="1.25" />
  <line x1="386" y1="316" x2="386" y2="260" stroke="#111" stroke-width="1.25" />
  <line x1="386" y1="316" x2="386" y2="320" stroke="#111" stroke-width="1" />
  <line x1="412" y1="316" x2="412" y2="320" stroke="#111" stroke-width="1" />
  <line x1="438" y1="316" x2="438" y2="320" stroke="#111" stroke-width="1" />
  <line x1="464" y1="316" x2="464" y2="320" stroke="#111" stroke-width="1" />
  <line x1="490" y1="316" x2="490" y2="320" stroke="#111" stroke-width="1" />
  <line x1="516" y1="316" x2="516" y2="320" stroke="#111" stroke-width="1" />
  <line x1="382" y1="316" x2="386" y2="316" stroke="#111" stroke-width="1" />
  <line x1="382" y1="288" x2="386" y2="288" stroke="#111" stroke-width="1" />
  <line x1="382" y1="260" x2="386" y2="260" stroke="#111" stroke-width="1" />
  <text class="d2_serif" x="386" y="331" text-anchor="middle" font-size="12">0</text>
  <text class="d2_serif" x="412" y="331" text-anchor="middle" font-size="12">2</text>
  <text class="d2_serif" x="438" y="331" text-anchor="middle" font-size="12">4</text>
  <text class="d2_serif" x="464" y="331" text-anchor="middle" font-size="12">6</text>
  <text class="d2_serif" x="490" y="331" text-anchor="middle" font-size="12">8</text>
  <text class="d2_serif" x="516" y="331" text-anchor="middle" font-size="12">10</text>
  <text class="d2_serif" x="378" y="320" text-anchor="end" font-size="12">0.7</text>
  <text class="d2_serif" x="378" y="292" text-anchor="end" font-size="12">0.9</text>
  <text class="d2_serif" x="378" y="264" text-anchor="end" font-size="12">1.1</text>
  <text class="d2_serif" x="451" y="345" text-anchor="middle" font-size="14">Time step <tspan font-style="italic">t</tspan></text>
  <g transform="translate(353,288) rotate(-90)"><text class="d2_serif" text-anchor="middle" font-size="13.5">Power (p.u.)</text></g>"""

    new_d2_ticks = """  <line x1="386" y1="316" x2="517" y2="316" stroke="#111" stroke-width="1.25" marker-end="url(#d2_arrow)" />
  <line x1="386" y1="316" x2="386" y2="258" stroke="#111" stroke-width="1.25" marker-end="url(#d2_arrow)" />
  <text class="d2_serif" x="451" y="336" text-anchor="middle" font-size="15">Time step <tspan font-style="italic">t</tspan></text>
  <g transform="translate(365,288) rotate(-90)"><text class="d2_serif" text-anchor="middle" font-size="15">Power (p.u.)</text></g>"""
    content = content.replace(old_d2_ticks, new_d2_ticks)

    old_d2_legend = """  <line x1="530" y1="267" x2="562" y2="267" stroke="#8da4b7" stroke-width="2.3" stroke-dasharray="7 5" />
  <text class="d2_serif" x="570" y="271" font-size="13">raw forecast ŷ<tspan baseline-shift="sub" font-size="10">raw</tspan></text>
  <line x1="530" y1="287" x2="562" y2="287" stroke="#096ef1" stroke-width="3" />
  <text class="d2_serif" x="570" y="291" font-size="13">physical bound</text>
  <text class="d2_serif" x="570" y="307" font-size="13">projected output ŷ<tspan baseline-shift="sub" font-size="10">proj</tspan></text>"""

    new_d2_legend = """  <line x1="528" y1="267" x2="560" y2="267" stroke="#8da4b7" stroke-width="2.3" stroke-dasharray="7 5" />
  <text class="d2_serif" x="568" y="271" font-size="14">raw forecast ŷ<tspan baseline-shift="sub" font-size="10.5">raw</tspan></text>
  <line x1="528" y1="287" x2="560" y2="287" stroke="#096ef1" stroke-width="3" />
  <text class="d2_serif" x="568" y="291" font-size="14">physical bound</text>
  <text class="d2_serif" x="568" y="308" font-size="14">projected output ŷ<tspan baseline-shift="sub" font-size="10.5">proj</tspan></text>"""
    content = content.replace(old_d2_legend, new_d2_legend)

    # -------------------------------------------------------------
    # 3. Panel D3 updates
    # -------------------------------------------------------------
    content = content.replace(
        '.d3_panelTitle { font-size:17px; font-weight:700; }',
        '.d3_panelTitle { font-size:18.5px; font-weight:700; }'
    )
    content = content.replace(
        '.d3_panelSub { font-size:15px; }',
        '.d3_panelSub { font-size:15px; }'
    )
    content = content.replace(
        '.d3_small { font-size:13px; }',
        '.d3_small { font-size:13.5px; }'
    )

    old_d3_ticks = """<line x1="61" y1="243" x2="124" y2="243" stroke="#111" stroke-width="1.1" marker-end="url(#d3_arrow)" />
<line x1="61" y1="243" x2="61" y2="160" stroke="#111" stroke-width="1.1" marker-end="url(#d3_arrow)" />
<line x1="57" y1="243" x2="62" y2="243" stroke="#111" stroke-width="1" />
<line x1="57" y1="202" x2="62" y2="202" stroke="#111" stroke-width="1" />
<line x1="57" y1="160" x2="62" y2="160" stroke="#111" stroke-width="1" />
<text x="55" y="248" class="d3_serif d3_small" text-anchor="end">0.0</text>
<text x="55" y="207" class="d3_serif d3_small" text-anchor="end">0.5</text>
<text x="55" y="165" class="d3_serif d3_small" text-anchor="end">1.0</text>
<text x="93" y="263" class="d3_serif d3_small" text-anchor="middle">Time</text>
<text x="24" y="206" class="d3_serif d3_small" text-anchor="middle" transform="rotate(-90 24 206)">Power (p.u.)</text>"""

    new_d3_ticks = """<line x1="61" y1="243" x2="124" y2="243" stroke="#111" stroke-width="1.1" marker-end="url(#d3_arrow)" />
<line x1="61" y1="243" x2="61" y2="160" stroke="#111" stroke-width="1.1" marker-end="url(#d3_arrow)" />
<text x="93" y="261" class="d3_serif" font-size="14" text-anchor="middle">Time</text>
<text x="40" y="203" class="d3_serif" font-size="14" text-anchor="middle" transform="rotate(-90 40 203)">Power (p.u.)</text>"""
    content = content.replace(old_d3_ticks, new_d3_ticks)

    old_d3_state_box = """<text x="518.5" y="183" class="d3_serif d3_small" text-anchor="middle">SCADA states</text>
<text x="518.5" y="201" class="d3_serif d3_small" text-anchor="middle">grid signals</text>
<text x="518.5" y="219" class="d3_serif d3_small" text-anchor="middle">turbine states</text>"""

    new_d3_state_box = """<text x="518.5" y="185" class="d3_serif" font-size="13" text-anchor="middle">SCADA states</text>
<text x="518.5" y="202" class="d3_serif" font-size="13" text-anchor="middle">grid signals</text>
<text x="518.5" y="219" class="d3_serif" font-size="13" text-anchor="middle">turbine states</text>"""
    content = content.replace(old_d3_state_box, new_d3_state_box)

    # -------------------------------------------------------------
    # 4. Panel D4 updates
    # -------------------------------------------------------------
    old_d4_public_code = """  <path d="M226 110 h25 l8 8 v36 h-33 z" fill="#f9fcff" stroke="#4d7f9c" stroke-width="1.4" />
  <path d="M251 110 v9 h8" fill="none" stroke="#4d7f9c" stroke-width="1.4" />
  <line x1="232" y1="123" x2="252" y2="123" stroke="#31556d" stroke-width="1.2" />
  <line x1="232" y1="131" x2="252" y2="131" stroke="#31556d" stroke-width="1.2" />
  <line x1="232" y1="139" x2="252" y2="139" stroke="#31556d" stroke-width="1.2" />
  <line x1="232" y1="147" x2="248" y2="147" stroke="#31556d" stroke-width="1.2" />
  <text x="271" y="117" class="d4_serif" font-family="monospace" font-size="13.5">loss.py</text>
  <text x="271" y="134" class="d4_serif" font-family="monospace" font-size="13.5">model.py</text>
  <circle cx="276" cy="141.7" r="1.15" fill="#1d4057" />
  <circle cx="276" cy="147.0" r="1.15" fill="#1d4057" />
  <circle cx="276" cy="152.3" r="1.15" fill="#1d4057" />
  <text x="286" y="157" font-family="'Liberation Mono','DejaVu Sans Mono',monospace" font-size="13.5" fill="#1c73dc">return </text>
  <text x="337" y="157" class="d4_eq" font-size="15" fill="#1c73dc">ℒ</text>
  <text x="350" y="161" class="d4_serif d4_italic" font-size="10.5" fill="#1c73dc">phys</text>"""

    new_d4_public_code = """  <path d="M226 110 h25 l8 8 v36 h-33 z" fill="#f9fcff" stroke="#4d7f9c" stroke-width="1.4" />
  <path d="M251 110 v9 h8" fill="none" stroke="#4d7f9c" stroke-width="1.4" />
  <text x="271" y="121" class="d4_serif" font-family="monospace" font-size="14" font-weight="bold">loss.py</text>
  <text x="271" y="138" class="d4_serif" font-family="monospace" font-size="14" font-weight="bold">model.py</text>
  <text x="271" y="157" font-family="'Liberation Mono','DejaVu Sans Mono',monospace" font-size="14" fill="#1c73dc">return </text>
  <text x="325" y="157" class="d4_eq" font-size="16" fill="#1c73dc">ℒ</text>
  <text x="339" y="161" class="d4_serif d4_italic" font-size="11" fill="#1c73dc">phys</text>"""
    content = content.replace(old_d4_public_code, new_d4_public_code)

    old_d4_notfound = """  <path d="M482 110 h25 l8 8 v36 h-33 z" fill="none" stroke="#4f8dad" stroke-width="1.45" stroke-dasharray="4,3" />
  <path d="M507 110 v9 h8" fill="none" stroke="#4f8dad" stroke-width="1.45" stroke-dasharray="4,3" />
  <line x1="488" y1="124" x2="508" y2="124" stroke="#4f8dad" stroke-width="1.2" stroke-dasharray="4,3" />
  <line x1="488" y1="132" x2="508" y2="132" stroke="#4f8dad" stroke-width="1.2" stroke-dasharray="4,3" />
  <line x1="488" y1="140" x2="508" y2="140" stroke="#4f8dad" stroke-width="1.2" stroke-dasharray="4,3" />
  <line x1="488" y1="148" x2="504" y2="148" stroke="#4f8dad" stroke-width="1.2" stroke-dasharray="4,3" />
  <text x="548" y="129" class="d4_serif" font-size="15">code:</text>
  <text x="548" y="147" class="d4_serif" font-size="15">not available</text>"""

    new_d4_notfound = """  <path d="M482 110 h25 l8 8 v36 h-33 z" fill="none" stroke="#4f8dad" stroke-width="1.45" stroke-dasharray="4,3" />
  <path d="M507 110 v9 h8" fill="none" stroke="#4f8dad" stroke-width="1.45" stroke-dasharray="4,3" />
  <text x="548" y="130" class="d4_serif" font-size="15.5" font-weight="600">code:</text>
  <text x="548" y="149" class="d4_serif" font-size="15.5" font-weight="600">not available</text>"""
    content = content.replace(old_d4_notfound, new_d4_notfound)

    content = content.replace(
        '<text x="78" y="222" text-anchor="middle" class="d4_serif d4_title" font-size="14.5">Published equation</text>',
        '<text x="78" y="222" text-anchor="middle" class="d4_serif d4_title" font-size="15.5">Published equation</text>'
    )
    content = content.replace(
        '<text x="230" y="222" text-anchor="middle" class="d4_serif d4_title" font-size="14.5">Source code</text>',
        '<text x="230" y="222" text-anchor="middle" class="d4_serif d4_title" font-size="15.5">Source code</text>'
    )
    content = content.replace(
        '<text x="391" y="221" text-anchor="middle" class="d4_serif d4_title" font-size="15">Aligned</text>\n  <text x="391" y="235" text-anchor="middle" class="d4_serif" font-size="12.5">Executable implementation</text>\n  <text x="391" y="247" text-anchor="middle" class="d4_serif" font-size="12.5">matches formulation</text>',
        '<text x="391" y="221" text-anchor="middle" class="d4_serif d4_title" font-size="16.5">Aligned</text>\n  <text x="391" y="235" text-anchor="middle" class="d4_serif" font-size="13">Executable implementation</text>\n  <text x="391" y="247" text-anchor="middle" class="d4_serif" font-size="13">matches formulation</text>'
    )
    content = content.replace(
        '<text x="590.5" y="221" text-anchor="middle" class="d4_serif d4_title" font-size="15">Mismatch</text>\n  <text x="590.5" y="235" text-anchor="middle" class="d4_serif" font-size="12.5">Discrepancy between</text>\n  <text x="590.5" y="247" text-anchor="middle" class="d4_serif" font-size="12.5">formulation and code</text>',
        '<text x="590.5" y="221" text-anchor="middle" class="d4_serif d4_title" font-size="16.5">Mismatch</text>\n  <text x="590.5" y="235" text-anchor="middle" class="d4_serif" font-size="13">Discrepancy between</text>\n  <text x="590.5" y="247" text-anchor="middle" class="d4_serif" font-size="13">formulation and code</text>'
    )

    # -------------------------------------------------------------
    # 5. Panel D5 updates
    # -------------------------------------------------------------
    # Sub-headers and concise, symmetric subtitles
    content = content.replace(
        '<text x="131" y="92" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="15" font-weight="bold" font-style="normal" text-anchor="middle" fill="#111">Constraint on/off</text>\n<text x="131" y="108" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">Identical backbone; λ = 0 ↔ λ &gt; 0</text>',
        '<text x="131" y="92" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="16" font-weight="bold" font-style="normal" text-anchor="middle" fill="#111">Constraint on/off</text>\n<text x="131" y="107" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">Identical backbone; λ = 0 ↔ λ &gt; 0</text>'
    )
    content = content.replace(
        '<text x="371" y="92" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="15" font-weight="bold" font-style="normal" text-anchor="middle" fill="#111">Module ablation</text>\n<text x="371" y="108" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">Full model ↔ neural block removed</text>',
        '<text x="371" y="92" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="16" font-weight="bold" font-style="normal" text-anchor="middle" fill="#111">Module ablation</text>\n<text x="371" y="107" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">Full model ↔ neural block removed</text>'
    )
    content = content.replace(
        '<text x="617" y="92" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="15" font-weight="bold" font-style="normal" text-anchor="middle" fill="#111">Both</text>\n<text x="617" y="107" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">Constraint on/off</text>\n<text x="617" y="120" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">and module ablation</text>',
        '<text x="617" y="92" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="16" font-weight="bold" font-style="normal" text-anchor="middle" fill="#111">Both</text>\n<text x="617" y="107" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">Constraint on/off &amp; module ablation</text>'
    )
    content = content.replace(
        '<text x="887" y="92" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="15" font-weight="bold" font-style="normal" text-anchor="middle" fill="#111">Reported</text>\n<text x="887" y="107" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">Degradation or failure boundary disclosed</text>',
        '<text x="887" y="92" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="16" font-weight="bold" font-style="normal" text-anchor="middle" fill="#111">Reported</text>\n<text x="887" y="107" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">Failure boundary disclosed</text>'
    )
    content = content.replace(
        '<text x="1206" y="92" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="15" font-weight="bold" font-style="normal" text-anchor="middle" fill="#111">Not reported</text>\n<text x="1206" y="107" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">No adverse regime documented</text>',
        '<text x="1206" y="92" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="16" font-weight="bold" font-style="normal" text-anchor="middle" fill="#111">Not reported</text>\n<text x="1206" y="107" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">No failure boundary documented</text>'
    )

    # Sub-panel 1 (Constraint on/off): remove ticks & numbers
    old_d5_p1_axes = """<line x1="62" y1="206" x2="230" y2="206" stroke="#111" stroke-width="1" />
<line x1="62" y1="206" x2="62" y2="128" stroke="#111" stroke-width="1" />
<line x1="62" y1="206" x2="62" y2="210" stroke="#111" stroke-width="0.8" />
<text x="62" y="221" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">0</text>
<line x1="118" y1="206" x2="118" y2="210" stroke="#111" stroke-width="0.8" />
<text x="118" y="221" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">10</text>
<line x1="174" y1="206" x2="174" y2="210" stroke="#111" stroke-width="0.8" />
<text x="174" y="221" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">20</text>
<line x1="230" y1="206" x2="230" y2="210" stroke="#111" stroke-width="0.8" />
<text x="230" y="221" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">30</text>
<line x1="58" y1="206" x2="62" y2="206" stroke="#111" stroke-width="0.8" />
<text x="55" y="210" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="end" fill="#111">0.6</text>
<line x1="58" y1="168" x2="62" y2="168" stroke="#111" stroke-width="0.8" />
<text x="55" y="172" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="end" fill="#111">0.8</text>
<line x1="58" y1="130" x2="62" y2="130" stroke="#111" stroke-width="0.8" />
<text x="55" y="134" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="end" fill="#111">1.0</text>
<text x="145" y="239" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="13" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">Forecast horizon h</text>
<text x="28" y="170" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="13" transform="rotate(-90 28 170)" text-anchor="middle">Power (p.u.)</text>"""

    new_d5_p1_axes = """<line x1="62" y1="206" x2="230" y2="206" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
<line x1="62" y1="206" x2="62" y2="124" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
<text x="145" y="226" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="14" font-weight="normal" text-anchor="middle" fill="#111">Forecast horizon h</text>
<text x="42" y="165" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="14" transform="rotate(-90 42 165)" text-anchor="middle">Power (p.u.)</text>"""
    content = content.replace(old_d5_p1_axes, new_d5_p1_axes)

    content = content.replace(
        '<text x="107" y="126" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="normal" font-style="normal" text-anchor="start" fill="#111">no constraint (λ = 0)</text>',
        '<text x="107" y="126" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" font-style="normal" text-anchor="start" fill="#111">no constraint (λ = 0)</text>'
    )
    content = content.replace(
        '<text x="107" y="145" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="normal" font-style="normal" text-anchor="start" fill="#111">constrained (λ &gt; 0)</text>',
        '<text x="107" y="145" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" font-style="normal" text-anchor="start" fill="#111">constrained (λ &gt; 0)</text>'
    )

    # Sub-panel 2 (Module ablation)
    content = content.replace(
        '<text x="258" y="147" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="13" font-weight="bold" font-style="normal" text-anchor="start" fill="#111">Full:</text>',
        '<text x="256" y="147" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="14" font-weight="bold" font-style="normal" text-anchor="start" fill="#111">Full:</text>'
    )
    content = content.replace(
        '<text x="250" y="201" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11.5" font-weight="bold" font-style="normal" text-anchor="start" fill="#111">Ablated:</text>',
        '<text x="248" y="201" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="13.5" font-weight="bold" font-style="normal" text-anchor="start" fill="#111">Ablated:</text>'
    )
    content = content.replace(
        '<text x="350" y="147" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">backbone</text>',
        '<text x="350" y="147" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">backbone</text>'
    )
    content = content.replace(
        '<text x="350" y="201" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">backbone</text>',
        '<text x="350" y="201" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">backbone</text>'
    )
    content = content.replace(
        '<text x="422.5" y="139" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="10" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">constraint</text>\n<text x="422.5" y="152" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="10" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">module</text>',
        '<text x="422.5" y="139" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">constraint</text>\n<text x="422.5" y="153" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">module</text>'
    )

    # Sub-panel 3 (Both): remove ticks & numbers
    old_d5_p3_axes = """<line x1="548" y1="208" x2="725" y2="208" stroke="#111" stroke-width="1" />
<line x1="548" y1="208" x2="548" y2="128" stroke="#111" stroke-width="1" />
<line x1="548" y1="208" x2="548" y2="212" stroke="#111" stroke-width="0.8" />
<text x="548" y="223" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">0</text>
<line x1="607" y1="208" x2="607" y2="212" stroke="#111" stroke-width="0.8" />
<text x="607" y="223" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">10</text>
<line x1="666" y1="208" x2="666" y2="212" stroke="#111" stroke-width="0.8" />
<text x="666" y="223" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">20</text>
<line x1="725" y1="208" x2="725" y2="212" stroke="#111" stroke-width="0.8" />
<text x="725" y="223" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">30</text>
<line x1="544" y1="208" x2="548" y2="208" stroke="#111" stroke-width="0.8" />
<text x="541" y="212" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="end" fill="#111">0.6</text>
<line x1="544" y1="169" x2="548" y2="169" stroke="#111" stroke-width="0.8" />
<text x="541" y="173" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="end" fill="#111">0.8</text>
<line x1="544" y1="130" x2="548" y2="130" stroke="#111" stroke-width="0.8" />
<text x="541" y="134" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="end" fill="#111">1.0</text>
<text x="637" y="240" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="13" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">Forecast horizon h</text>
<text x="514" y="170" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="13" transform="rotate(-90 514 170)" text-anchor="middle">Power (p.u.)</text>"""

    new_d5_p3_axes = """<line x1="548" y1="208" x2="725" y2="208" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
<line x1="548" y1="208" x2="548" y2="124" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
<text x="637" y="226" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="14" font-weight="normal" text-anchor="middle" fill="#111">Forecast horizon h</text>
<text x="528" y="165" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="14" transform="rotate(-90 528 165)" text-anchor="middle">Power (p.u.)</text>"""
    content = content.replace(old_d5_p3_axes, new_d5_p3_axes)

    content = content.replace(
        '<text x="658" y="139" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="start" fill="#111">full model</text>',
        '<text x="658" y="139" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" font-style="normal" text-anchor="start" fill="#111">full model</text>'
    )
    content = content.replace(
        '<text x="658" y="154" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="start" fill="#111">module ablated</text>',
        '<text x="658" y="154" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" font-style="normal" text-anchor="start" fill="#111">module ablated</text>'
    )
    content = content.replace(
        '<text x="658" y="169" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="start" fill="#111">no constraint</text>',
        '<text x="658" y="169" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" font-style="normal" text-anchor="start" fill="#111">no constraint</text>'
    )

    # Sub-panel 4 (Reported):
    # - y-axis from 211 to 125 (no collision with subtitle)
    # - Rotated label Normalized metric (↓) at x=762 (clear of axis)
    # - failure boundary alpha* at x=848
    old_d5_reported_block = """<line x1="800" y1="211" x2="1008" y2="211" stroke="#111" stroke-width="1" />
<line x1="800" y1="211" x2="800" y2="114" stroke="#111" stroke-width="1" />


<line x1="800" y1="211" x2="800" y2="215" stroke="#111" stroke-width="0.8" />
<text x="800" y="226" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" text-anchor="middle" fill="#111">0.0</text>
<line x1="841.6" y1="211" x2="841.6" y2="215" stroke="#111" stroke-width="0.8" />
<text x="841.6" y="226" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" text-anchor="middle" fill="#111">0.2</text>
<line x1="883.2" y1="211" x2="883.2" y2="215" stroke="#111" stroke-width="0.8" />
<text x="883.2" y="226" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" text-anchor="middle" fill="#111">0.4</text>
<line x1="924.8" y1="211" x2="924.8" y2="215" stroke="#111" stroke-width="0.8" />
<text x="924.8" y="226" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" text-anchor="middle" fill="#111">0.6</text>
<line x1="966.4" y1="211" x2="966.4" y2="215" stroke="#111" stroke-width="0.8" />
<text x="966.4" y="226" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" text-anchor="middle" fill="#111">0.8</text>
<line x1="1008" y1="211" x2="1008" y2="215" stroke="#111" stroke-width="0.8" />
<text x="1008" y="226" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" text-anchor="middle" fill="#111">1.0</text>
<text x="904" y="243" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="13" text-anchor="middle" fill="#111">Corruption factor α</text>


<line x1="796" y1="211" x2="800" y2="211" stroke="#111" stroke-width="0.8" />
<text x="793" y="215" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" text-anchor="end" fill="#111">0.8</text>
<line x1="796" y1="190.6" x2="800" y2="190.6" stroke="#111" stroke-width="0.8" />
<text x="793" y="194.5" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" text-anchor="end" fill="#111">1.0</text>
<line x1="796" y1="170.2" x2="800" y2="170.2" stroke="#111" stroke-width="0.8" />
<text x="793" y="174" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" text-anchor="end" fill="#111">1.2</text>
<line x1="796" y1="149.7" x2="800" y2="149.7" stroke="#111" stroke-width="0.8" />
<text x="793" y="153.5" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" text-anchor="end" fill="#111">1.4</text>
<line x1="796" y1="129.3" x2="800" y2="129.3" stroke="#111" stroke-width="0.8" />
<text x="793" y="133" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" text-anchor="end" fill="#111">1.6</text>
<text x="767" y="168" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="13" transform="rotate(-90 767 168)" text-anchor="middle">Normalized metric (↓)</text>



<polygon points="800,211 1008,211 1008,190.6 800,190.6" fill="#eaf7ed" opacity="0.85" />

<polygon points="800,190.6 1008,190.6 1008,114 800,114" fill="#fff1f2" opacity="0.6" />


<line x1="800" y1="190.6" x2="1008" y2="190.6" stroke="#2e7d32" stroke-width="1.1" stroke-dasharray="5 4" />
<text x="1004" y="186" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="10" font-weight="normal" text-anchor="end" fill="#2e7d32">Baseline (no physics, 1.00)</text>
<text x="914" y="204" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#1b5e20">Beneficial regime (below baseline)</text>


<text x="860" y="145" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11.5" font-weight="bold" text-anchor="middle" fill="#c62828">Adverse regime</text>


<line x1="814.3" y1="114" x2="814.3" y2="211" stroke="#d32f2f" stroke-width="1.1" stroke-dasharray="4 3" />
<text x="818" y="123" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11.5" font-style="italic" font-weight="bold" text-anchor="start" fill="#d32f2f">α* ≈ 0.07</text>


<polygon points="800.0,192.1 820.8,188.4 841.6,181.0 862.4,172.6 883.2,160.7 904.0,149.4 924.8,137.5 945.6,129.6 966.4,124.2 987.2,119.4 1008.0,114.0 1008.0,145.0 987.2,148.9 966.4,152.2 945.6,158.9 924.8,166.3 904.0,172.6 883.2,177.6 862.4,184.1 841.6,188.0 820.8,191.2 800.0,192.6" fill="#ffcdd2" opacity="0.45" />


<polyline points="800.0,192.3 820.8,189.8 841.6,184.5 862.4,178.3 883.2,169.2 904.0,161.0 924.8,151.9 945.6,144.3 966.4,138.2 987.2,134.2 1008.0,129.5" fill="none" stroke="#d32f2f" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" />


<rect x="797.5" y="189.8" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="818.3" y="187.3" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="839.1" y="182.0" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="859.9" y="175.8" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="880.7" y="166.7" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="901.5" y="158.5" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="922.3" y="149.4" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="943.1" y="141.8" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="963.9" y="135.7" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="984.7" y="131.7" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="1005.5" y="127.0" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />"""

    new_d5_reported_block = """<line x1="800" y1="211" x2="1008" y2="211" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
<line x1="800" y1="211" x2="800" y2="125" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />

<text x="904" y="230" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="14" text-anchor="middle" fill="#111">Corruption / stress factor α</text>
<text x="760" y="168" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="13.5" transform="rotate(-90 760 168)" text-anchor="middle">Normalized metric (↓)</text>

<!-- Regimes shading -->
<polygon points="800,211 1008,211 1008,190.6 800,190.6" fill="#eaf7ed" opacity="0.85" />
<polygon points="800,190.6 1008,190.6 1008,125 800,125" fill="#fff1f2" opacity="0.6" />

<!-- Baseline -->
<line x1="800" y1="190.6" x2="1008" y2="190.6" stroke="#2e7d32" stroke-width="1.2" stroke-dasharray="5 4" />
<text x="1004" y="186" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="bold" text-anchor="end" fill="#2e7d32">Baseline (no physics)</text>
<text x="940" y="204" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#1b5e20">Beneficial regime (below baseline)</text>
<text x="925" y="145" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="13" font-weight="bold" text-anchor="middle" fill="#c62828">Adverse regime</text>

<!-- Schematic failure boundary at crossover x = 848 -->
<line x1="848" y1="125" x2="848" y2="211" stroke="#d32f2f" stroke-width="1.3" stroke-dasharray="4 3" />
<text x="853" y="125" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="13" font-style="italic" font-weight="bold" text-anchor="start" fill="#d32f2f">failure boundary α*</text>

<!-- Shaded uncertainty band passing through crossover -->
<polygon points="800.0,203.0 824.0,199.0 848.0,190.6 872.0,175.0 896.0,162.0 920.0,149.0 944.0,138.0 968.0,130.0 992.0,125.0 1008.0,122.0 1008.0,140.0 992.0,143.0 968.0,149.0 944.0,156.0 920.0,165.0 896.0,176.0 872.0,186.0 848.0,190.6 824.0,201.0 800.0,205.0" fill="#ffcdd2" opacity="0.45" />

<!-- Red curve starting beneficial (< baseline 190.6), crossing at alpha*, rising to adverse -->
<polyline points="800.0,201.0 824.0,196.5 848.0,190.6 872.0,180.5 896.0,169.0 920.0,157.0 944.0,147.0 968.0,139.5 992.0,133.0 1008.0,129.0" fill="none" stroke="#d32f2f" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round" />

<rect x="797.5" y="198.5" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="821.5" y="194.0" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="845.5" y="188.1" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="869.5" y="178.0" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="893.5" y="166.5" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="917.5" y="154.5" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="941.5" y="144.5" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="965.5" y="137.0" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="989.5" y="130.5" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />
<rect x="1005.5" y="126.5" width="5" height="5" fill="#d32f2f" stroke="#b71c1c" stroke-width="0.6" />"""
    content = content.replace(old_d5_reported_block, new_d5_reported_block)

    # Sub-panel 5 (Not reported):
    # - y-axis line ends at y=125
    # - Rotated Power (p.u.) label at x=1046
    old_d5_p5_axes = """<line x1="1073" y1="210" x2="1275" y2="210" stroke="#111" stroke-width="1" />
<line x1="1073" y1="210" x2="1073" y2="115" stroke="#111" stroke-width="1" />
<line x1="1073" y1="210" x2="1073" y2="214" stroke="#111" stroke-width="0.8" />
<text x="1073" y="225" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">0</text>
<line x1="1123" y1="210" x2="1123" y2="214" stroke="#111" stroke-width="0.8" />
<text x="1123" y="225" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">5</text>
<line x1="1173" y1="210" x2="1173" y2="214" stroke="#111" stroke-width="0.8" />
<text x="1173" y="225" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">10</text>
<line x1="1223" y1="210" x2="1223" y2="214" stroke="#111" stroke-width="0.8" />
<text x="1223" y="225" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">15</text>
<line x1="1275" y1="210" x2="1275" y2="214" stroke="#111" stroke-width="0.8" />
<text x="1275" y="225" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">20</text>
<line x1="1069" y1="210" x2="1073" y2="210" stroke="#111" stroke-width="0.8" />
<text x="1066" y="214" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="end" fill="#111">0.0</text>
<line x1="1069" y1="176" x2="1073" y2="176" stroke="#111" stroke-width="0.8" />
<text x="1066" y="180" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="end" fill="#111">0.4</text>
<line x1="1069" y1="142" x2="1073" y2="142" stroke="#111" stroke-width="0.8" />
<text x="1066" y="146" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="end" fill="#111">0.8</text>
<text x="1174" y="243" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="13" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">Wind speed (m/s)</text>
<text x="1042" y="170" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="13" transform="rotate(-90 1042 170)" text-anchor="middle">Power (p.u.)</text>"""

    new_d5_p5_axes = """<line x1="1073" y1="210" x2="1275" y2="210" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
<line x1="1073" y1="210" x2="1073" y2="125" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
<text x="1174" y="228" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="14" font-weight="normal" text-anchor="middle" fill="#111">Wind speed (m/s)</text>
<text x="1046" y="168" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="14" transform="rotate(-90 1046 168)" text-anchor="middle">Power (p.u.)</text>"""
    content = content.replace(old_d5_p5_axes, new_d5_p5_axes)

    content = content.replace(
        '<text x="1116" y="121" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="start" fill="#111">Genuine prior <tspan font-style="italic">C</tspan><tspan font-size="8.5" dy="3">emp</tspan><tspan dy="-3" font-size="11" /></text>',
        '<text x="1116" y="121" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" font-style="normal" text-anchor="start" fill="#111">Genuine prior <tspan font-style="italic">C</tspan><tspan font-size="9.5" dy="3">emp</tspan><tspan dy="-3" font-size="12.5" /></text>'
    )
    content = content.replace(
        '<text x="1116" y="137" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="start" fill="#111">Scrambled prior <tspan font-style="italic">C</tspan><tspan font-size="8.5" dy="3">scramb</tspan><tspan dy="-3" font-size="11" /></text>',
        '<text x="1116" y="138" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" font-style="normal" text-anchor="start" fill="#111">Scrambled prior <tspan font-style="italic">C</tspan><tspan font-size="9.5" dy="3">scramb</tspan><tspan dy="-3" font-size="12.5" /></text>'
    )

    old_d5_p5_callout = """<rect x="1280" y="115" width="99" height="98" rx="5" fill="#ffd9e1" stroke="none" stroke-width="0" />
<text x="1329" y="132" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">Downstream</text>
<text x="1329" y="146" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">decision-risk</text>
<text x="1329" y="160" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">exposure not</text>
<text x="1329" y="174" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">evaluated</text>
<text x="1329" y="190" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">(failure boundary</text>
<text x="1329" y="203" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11" font-weight="normal" font-style="normal" text-anchor="middle" fill="#111">omitted)</text>"""

    new_d5_p5_callout = """<rect x="1280" y="112" width="102" height="102" rx="5" fill="#ffd9e1" stroke="#f096a6" stroke-width="1.1" />
<text x="1331" y="130" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="13" font-weight="bold" text-anchor="middle" fill="#9c1c38">Downstream</text>
<text x="1331" y="145" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="bold" text-anchor="middle" fill="#9c1c38">decision-risk</text>
<text x="1331" y="160" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" text-anchor="middle" fill="#5a1020">exposure not</text>
<text x="1331" y="175" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="12.5" font-weight="normal" text-anchor="middle" fill="#5a1020">evaluated</text>
<text x="1331" y="193" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11.5" font-style="italic" text-anchor="middle" fill="#7a152a">(failure boundary</text>
<text x="1331" y="206" font-family="Times New Roman, Times, DejaVu Serif, serif" font-size="11.5" font-style="italic" text-anchor="middle" fill="#7a152a">omitted)</text>"""
    content = content.replace(old_d5_p5_callout, new_d5_p5_callout)

    return content

def main():
    root = Path(__file__).resolve().parents[1]
    backup_file = root / "outputs" / "constraint_lifecycle_taxonomy_assembled.svg.bak"
    if not backup_file.exists():
        raise FileNotFoundError(f"Backup file {backup_file} not found!")
    
    print(f"Reading from pristine backup: {backup_file}")
    content = backup_file.read_text(encoding="utf-8")

    updated = update_svg_content(content)

    # Validate XML syntax
    try:
        ET.fromstring(updated)
        print("XML syntax validation PASSED!")
    except ET.ParseError as e:
        print(f"ERROR: XML parsing failed: {e}")
        raise

    svg_source = root / "outputs" / "constraint_lifecycle_taxonomy_assembled.svg"
    svg_source.write_text(updated, encoding="utf-8")
    print("Updated SVG written to outputs/constraint_lifecycle_taxonomy_assembled.svg")

    fig_svg = root / "outputs" / "fig_constraint_lifecycle.svg"
    fig_svg.write_text(updated, encoding="utf-8")
    print("Updated SVG written to outputs/fig_constraint_lifecycle.svg")

    pdf_out = root / "outputs" / "constraint_lifecycle_taxonomy_assembled.pdf"
    png_out = root / "outputs" / "constraint_lifecycle_taxonomy_assembled.png"
    print("Rendering SVG to PDF with cairosvg...")
    cairosvg.svg2pdf(url=str(svg_source), write_to=str(pdf_out))
    print(f"PDF generated: {pdf_out} (size: {pdf_out.stat().st_size} bytes)")

    print("Rendering SVG to PNG with cairosvg...")
    cairosvg.svg2png(url=str(svg_source), write_to=str(png_out), dpi=300)
    print(f"PNG generated: {png_out} (size: {png_out.stat().st_size} bytes)")

    shutil.copy2(pdf_out, root / "outputs" / "fig_constraint_lifecycle.pdf")
    shutil.copy2(png_out, root / "outputs" / "fig_constraint_lifecycle.png")

    targets = [
        root / "01_论文投稿材料" / "active_manuscript" / "figures",
        root / "01_论文投稿材料" / "chinese_manuscript" / "figures",
        root / "03_图表与绘图素材",
        root / "02_论文正文与LaTeX",
    ]

    for t in targets:
        if t.exists():
            dest_pdf = t / "fig_constraint_lifecycle.pdf"
            shutil.copy2(pdf_out, dest_pdf)
            print(f"Synced PDF -> {dest_pdf}")
            dest_svg = t / "fig_constraint_lifecycle.svg"
            shutil.copy2(svg_source, dest_svg)
            print(f"Synced SVG -> {dest_svg}")
            dest_png = t / "fig_constraint_lifecycle.png"
            shutil.copy2(png_out, dest_png)
            print(f"Synced PNG -> {dest_png}")

    print("All sync operations completed successfully!")

if __name__ == "__main__":
    main()
