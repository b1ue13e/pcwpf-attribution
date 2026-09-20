"""Generate camera-ready Figure 1: Five-Dimensional Constraint Lifecycle Framework (Schema v4).

Fully resolves all user review critiques and audit findings:
1. D5:
   - Unified title to 'Evaluation Validity & Harm Reporting'.
   - Subtitle: 'Does the evaluation isolate the physical prior and expose failure regimes?'.
   - Two orthogonal subfields: 'Ablation semantics' and 'Failure / harm reporting'.
   - Section A (Ablation semantics): All mini-plot y-axes unified to 'Normalized metric (↓)'.
   - Section B (Failure / harm reporting):
     - Reported: 'Beneficial regime', 'Adverse regime', schematic 'empirical failure crossover α*'.
     - Not reported: Generic methodological taxonomy ('Nominal region only', 'Failure boundary omitted', 'Downstream failure regimes unexamined'), completely eliminating empirical audit statistics (e.g. 0/46). Mini-plot axes aligned to 'Normalized metric (↓)' and 'Corruption / stress factor α'.
2. D1 (Parameter Provenance):
   - Provenance clutter eliminated: single clean 'Not specified' with 'Calibration omitted'.
   - Arrowhead of y-axis in Historical-static terminates safely at y=120, avoiding 'derived prior' (y=113).
3. D2 (Application Stage):
   - Eliminated CairoSVG tspan dy and middle-anchor bugs in 'Data generation' (simulator f_phys(θ)) and 'Post-hoc correction' (Time step t).
   - Enlarged and padded 3-entry legend in Post-hoc correction.
4. D3 (Inference Telemetry Requirements):
   - Coordinate ticks eliminated across mini-plots and horizon dimension bar.
   - Fixed CairoSVG subscript collision in P_{t-L:t}.
5. D4 (Implementation Fidelity):
   - Streamlined clean 'loss.py' editor window with 'return L_phys'.
   - 'Published equation' header box widened to prevent border collision.
6. Typography:
   - All body text, legends, and math labels calibrated to ensure crisp ~7-9 pt readability in compiled 2-column PDF.
   - Font stacks robust across Windows/Linux ('Times New Roman', 'Segoe UI Symbol', 'Cambria Math', 'DejaVu Serif', serif).
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import cairosvg

def build_perfect_svg() -> str:
    svg_parts = []
    
    # -------------------------------------------------------------
    # Root SVG tag & canvas
    # -------------------------------------------------------------
    svg_parts.append("""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1448" height="1086" viewBox="0 0 1448 1086">
  <!-- Canvas Background -->
  <rect width="1448" height="1086" fill="#ffffff"/>
  <!-- Master Taxonomy Header -->
  <g id="master-header">
    <text x="724" y="38" text-anchor="middle" font-family="'Times New Roman', 'DejaVu Serif', serif" font-size="39" font-weight="bold" fill="#111111">Constraint Lifecycle Taxonomy</text>
    <text x="724" y="63" text-anchor="middle" font-family="'Times New Roman', 'DejaVu Serif', serif" font-size="22" font-style="normal" fill="#2d3748">Five orthogonal dimensions for auditing physics-constrained forecasting studies</text>
  </g>
""")

    # -------------------------------------------------------------
    # Panel D1: Parameter Provenance
    # -------------------------------------------------------------
    svg_parts.append("""  <!-- ==================== Panel D1 ==================== -->
  <svg id="panel_D1" x="5" y="78" width="717" height="377" viewBox="0 0 506 274" preserveAspectRatio="none">
    <defs>
      <style>
        .d1_serif { font-family: 'Times New Roman', 'DejaVu Serif', serif; fill:#111; }
        .d1_body { font-family: 'Times New Roman', 'DejaVu Serif', serif; font-size: 13.5px; fill:#2d3748; }
        .d1_head { font-family: 'Times New Roman', 'DejaVu Serif', serif; font-size: 16px; font-weight: 700; fill:#111; }
        .d1_title { font-family: 'Times New Roman', 'DejaVu Serif', serif; font-size: 19.5px; font-weight: 700; fill:#111; }
        .d1_subtitle { font-family: 'Times New Roman', 'DejaVu Serif', serif; font-size: 14px; font-style: italic; font-weight: 600; fill:#333; }
        .d1_math { font-family: 'Times New Roman', 'Segoe UI Symbol', 'Cambria Math', 'DejaVu Serif', serif; font-size: 19px; font-style: italic; fill:#103b5c; }
      </style>
      <marker id="d1_arrowBlue" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L7,3.5 L0,7 z" fill="#2f88d8" />
      </marker>
      <marker id="d1_arrowBlack" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L7,3.5 L0,7 z" fill="#111" />
      </marker>
      <clipPath id="d1_plotClip">
        <rect x="154" y="118" width="94" height="88" />
      </clipPath>
    </defs>

    <!-- Outer card -->
    <rect x="1" y="1" width="504" height="272" rx="5" ry="5" fill="#f7a9aa" stroke="#111" stroke-width="1.6" />
    <rect x="3.8" y="4" width="498.4" height="265" rx="4" ry="4" fill="#ffd0d0" stroke="#fff" stroke-width="1.2" />

    <!-- Header badge & title -->
    <rect x="7" y="7" width="52" height="35" rx="2" ry="2" fill="#fff0f0" stroke="#454545" stroke-width="1.1" />
    <text x="18" y="34" class="d1_serif" font-size="25" font-weight="700">D1</text>
    <text x="73" y="25" class="d1_title">Parameter Provenance</text>
    <text x="74" y="43" class="d1_subtitle">Where does the constraint originate?</text>

    <!-- 4 Subpanel Boxes -->
    <rect x="6" y="50" width="123" height="213" rx="3" fill="#fbfbfb" stroke="#666" stroke-width="1.2" />
    <rect x="131" y="50" width="123" height="213" rx="3" fill="#fbfbfb" stroke="#666" stroke-width="1.2" />
    <rect x="256" y="50" width="123" height="213" rx="3" fill="#fbfbfb" stroke="#666" stroke-width="1.2" />
    <rect x="381" y="50" width="119" height="213" rx="3" fill="#fbfbfb" stroke="#666" stroke-width="1.2" />

    <!-- Col 1: Intrinsic -->
    <text x="67.5" y="71" text-anchor="middle" class="d1_head">Intrinsic</text>
    <text x="67.5" y="89" text-anchor="middle" class="d1_body">Governing laws /</text>
    <text x="67.5" y="104" text-anchor="middle" class="d1_body">physical limits</text>

    <!-- Flow lines -->
    <g fill="none" stroke="#2f88d8" stroke-width="1.0" marker-end="url(#d1_arrowBlue)">
      <path d="M18 128 C33 128, 43 126, 57 125" />
      <path d="M18 138 C33 139, 44 135, 58 134" />
      <path d="M18 148 C32 148, 43 145, 58 144" />
      <path d="M18 158 C33 158, 43 157, 59 156" />
      <path d="M18 169 C33 169, 43 170, 58 169" />
      <path d="M18 179 C33 179, 43 183, 58 182" />
    </g>
    <g fill="none" stroke="#2f88d8" stroke-width="1.0" marker-end="url(#d1_arrowBlue)">
      <path d="M88 124 C101 124, 110 122, 118 121" />
      <path d="M88 134 C101 134, 110 133, 118 132" />
      <path d="M88 144 C101 144, 110 144, 118 143" />
      <path d="M88 156 C101 156, 110 157, 118 156" />
      <path d="M88 168 C101 168, 110 170, 118 170" />
      <path d="M88 181 C101 181, 110 184, 118 184" />
    </g>

    <!-- Turbine icon -->
    <g transform="translate(75,155)" fill="#103b5c" stroke="#103b5c">
      <ellipse cx="0" cy="0" rx="4.7" ry="2.6" />
      <path d="M-1.6 -1 L-7.5 -30 Q-5.8 -37 -3.3 -34 L2.0 -3 Z" />
      <path d="M1.8 1.0 L28.0 2.7 Q33.7 4.7 28.2 6.5 L3.0 4.0 Z" />
      <path d="M-0.7 2.0 L-12 25 Q-15 31 -10.5 27 L1.8 4.0 Z" />
      <path d="M-2.2 3.0 L-5.8 42 L6.0 42 L2.0 3.0 Z" />
    </g>
    <text x="68" y="238" text-anchor="middle" class="d1_math">Cₚ ≤ 16/27</text>

    <!-- Col 2: Historical-static -->
    <text x="192.5" y="67" text-anchor="middle" class="d1_head">Historical-</text>
    <text x="192.5" y="82" text-anchor="middle" class="d1_head">static</text>
    <text x="192.5" y="99" text-anchor="middle" class="d1_body">Offline SCADA-</text>
    <text x="192.5" y="113" text-anchor="middle" class="d1_body">derived prior</text>

    <!-- Mini scatter plot & curve: Arrow terminates at y=120 to safely clear text at y=113 -->
    <g stroke="#111" fill="none" stroke-width="1.3" marker-end="url(#d1_arrowBlack)">
      <path d="M154 205 L154 120" />
      <path d="M154 205 L248 205" />
    </g>
    <g clip-path="url(#d1_plotClip)">
      <circle cx="195.60" cy="138.55" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="167.12" cy="204.80" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="186.22" cy="173.62" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="160.29" cy="205.00" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="193.00" cy="142.15" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="175.11" cy="196.65" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="181.33" cy="184.33" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="163.91" cy="205.00" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="177.84" cy="194.81" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="179.12" cy="186.74" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="163.52" cy="205.00" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="201.14" cy="126.11" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="210.00" cy="130.44" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="179.08" cy="191.05" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="185.73" cy="165.02" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="174.78" cy="194.27" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="169.91" cy="203.01" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="172.81" cy="199.55" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="166.50" cy="205.00" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <circle cx="228.45" cy="139.56" r="0.75" fill="#9fb0bf" opacity="0.55" />
      <polyline points="154.0,205.0 163.6,205.0 167.4,204.2 171.2,200.9 175.1,195.3 178.9,187.8 182.8,177.6 186.6,166.3 190.4,153.4 194.2,143.3 198.1,137.8 201.9,135.0 205.8,133.6 209.6,134.2 213.4,131.4 217.2,130.4 221.1,135.9 224.9,135.1 228.8,134.6 232.6,136.8 236.4,148.3" fill="none" stroke="#0968e8" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round" />
    </g>
    <text x="201" y="223" text-anchor="middle" class="d1_serif" font-size="14.5">Wind speed</text>
    <text x="142" y="165" text-anchor="middle" class="d1_serif" font-size="14.5" transform="rotate(-90 142 165)">Power</text>

    <!-- Col 3: Learned-online -->
    <text x="317.5" y="67" text-anchor="middle" class="d1_head">Learned-</text>
    <text x="317.5" y="82" text-anchor="middle" class="d1_head">online</text>
    <text x="317.5" y="99" text-anchor="middle" class="d1_body">Adaptive learned</text>
    <text x="317.5" y="113" text-anchor="middle" class="d1_body">parameters</text>

    <text x="264" y="153" class="d1_serif" font-size="13.5" font-style="italic">x<tspan font-size="10" dy="3">t-1</tspan></text>
    <path d="M287 149 L303 149" stroke="#111" stroke-width="1.2" marker-end="url(#d1_arrowBlack)" />
    <rect x="304" y="132" width="40" height="31" fill="#e5f4fb" stroke="#333" stroke-width="1.1" />
    <text x="324" y="153" text-anchor="middle" class="d1_serif" font-size="16" font-style="italic">g<tspan font-size="11" dy="3">ϕ</tspan></text>
    <path d="M344 149 L359 149" stroke="#111" stroke-width="1.2" marker-end="url(#d1_arrowBlack)" />
    <text x="362" y="154" class="d1_serif" font-size="13.5" font-style="italic">θ<tspan font-size="10" dy="3">t</tspan></text>
    <path d="M324 189 L324 165" stroke="#111" stroke-width="1.2" marker-end="url(#d1_arrowBlack)" />
    <text x="324" y="201" text-anchor="middle" class="d1_serif" font-size="13.5" font-style="italic">θ<tspan font-size="10" dy="3">t-1</tspan></text>
    <text x="324" y="223" text-anchor="middle" class="d1_serif" font-size="14.5">online update</text>

    <!-- Col 4: Unreported (Decluttered: single 'Not specified', subtitle 'Calibration omitted') -->
    <text x="440.5" y="71" text-anchor="middle" class="d1_head">Unreported</text>
    <text x="440.5" y="89" text-anchor="middle" class="d1_body">Calibration</text>
    <text x="440.5" y="104" text-anchor="middle" class="d1_body">omitted</text>
    <rect x="396" y="125" width="91" height="69" rx="2" fill="#fff" stroke="#5a7590" stroke-width="1.3" stroke-dasharray="4 3" />
    <text x="441.5" y="164" text-anchor="middle" class="d1_serif" font-size="15" font-weight="600" fill="#2d3748">Not specified</text>
  </svg>
""")

    # -------------------------------------------------------------
    # Panel D2: Application Stage
    # -------------------------------------------------------------
    svg_parts.append("""  <!-- ==================== Panel D2 ==================== -->
  <svg id="panel_D2" x="726" y="78" width="717" height="377" viewBox="0 0 696 373" preserveAspectRatio="none">
    <defs>
      <linearGradient id="d2_headerGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#d8efff" />
        <stop offset="1" stop-color="#c8e9fb" />
      </linearGradient>
      <linearGradient id="d2_formulaGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#e5f5ff" />
        <stop offset="1" stop-color="#d7effd" />
      </linearGradient>
      <marker id="d2_arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L8,4 L0,8 Z" fill="#111" />
      </marker>
      <clipPath id="d2_matrixClip"><rect x="235" y="267" width="86" height="63" /></clipPath>
      <style>
        .d2_serif { font-family: 'Times New Roman', 'DejaVu Serif', serif; fill:#111; }
        .d2_math { font-family: 'Times New Roman', 'Segoe UI Symbol', 'Cambria Math', 'DejaVu Serif', serif; fill:#111; }
        .d2_title { font-weight:700; fill:#111; }
        .d2_panel { fill:#fff; stroke:#3b8ec4; stroke-width:1.35; }
        .blue { stroke:#0878ec; fill:none; }
      </style>
    </defs>

    <!-- Outer card -->
    <rect x="0.8" y="0.8" width="694.4" height="371.4" rx="7.5" fill="#f8fcff" stroke="#0b0b0b" stroke-width="1.6" />
    <rect x="3.7" y="3.7" width="688.6" height="365.6" rx="6.5" fill="none" stroke="#1595ea" stroke-width="2.1" />

    <!-- Header badge & title -->
    <rect x="7" y="7" width="681.5" height="52.5" rx="3" fill="url(#d2_headerGrad)" stroke="#2f8bc2" stroke-width="1.2" />
    <rect x="12" y="9" width="72" height="47" rx="2.3" fill="#eef8ff" stroke="#1e8ed7" stroke-width="1.2" />
    <text class="d2_serif d2_title" x="21" y="45" font-size="36">D2</text>
    <line x1="96" y1="11" x2="96" y2="55" stroke="#5a9ec8" stroke-width="1.15" />
    <text class="d2_serif d2_title" x="105" y="35" font-size="28">Application Stage</text>
    <text class="d2_serif" x="105" y="55" font-size="20" font-style="italic">Where does physics enter the forecasting system?</text>

    <!-- Subpanel 1: Training loss -->
    <rect class="d2_panel" x="7" y="65" width="326" height="137" rx="3.2" />
    <text class="d2_serif d2_title" x="20" y="90" font-size="23">Training loss</text>
    <text class="d2_serif" x="20" y="112" font-size="18">Physics term in objective function</text>
    <rect x="20" y="124" width="302" height="59" rx="4" fill="url(#d2_formulaGrad)" />
    <g class="d2_math" font-style="italic">
      <text x="76" y="162" font-size="29">ℒ</text>
      <text x="103" y="162" font-size="25" font-style="normal">=</text>
      <text x="126" y="162" font-size="29">ℒ</text>
      <text x="144" y="168" font-size="16">task</text>
      <text x="181" y="162" font-size="25" font-style="normal">+</text>
      <text x="204" y="162" font-size="26">λ</text>
      <text x="223" y="162" font-size="29">ℒ</text>
      <text x="241" y="168" font-size="16">phys</text>
    </g>

    <!-- Subpanel 2: Architecture -->
    <rect class="d2_panel" x="336" y="65" width="352" height="137" rx="3.2" />
    <text class="d2_serif d2_title" x="351" y="90" font-size="23">Architecture</text>
    <text class="d2_serif" x="351" y="112" font-size="18">Physics-constrained layer / graph</text>
    <text class="d2_serif" x="351" y="150" font-size="17">feature</text>
    <text class="d2_math" x="406" y="151" font-size="19" font-style="italic">x</text>
    <line x1="423" y1="145" x2="448" y2="145" stroke="#111" stroke-width="1.5" marker-end="url(#d2_arrow)" />
    <rect x="451" y="123" width="103" height="42" fill="#d7edf8" stroke="#2f6f99" stroke-width="1.2" />
    <text class="d2_serif" x="502.5" y="140" text-anchor="middle" font-size="15.5">constrained</text>
    <text class="d2_serif" x="502.5" y="156" text-anchor="middle" font-size="15.5">module</text>
    <line x1="556" y1="145" x2="581" y2="145" stroke="#111" stroke-width="1.5" marker-end="url(#d2_arrow)" />
    <text class="d2_serif" x="586" y="150" font-size="16.5">prediction</text>
    <text class="d2_math" x="662" y="151" font-size="20" font-style="italic">ŷ</text>
    <line x1="502.5" y1="181" x2="502.5" y2="167" stroke="#111" stroke-width="1.45" marker-end="url(#d2_arrow)" />
    <text class="d2_serif" x="502.5" y="196" text-anchor="middle" font-size="15.5">physics / physical prior</text>

    <!-- Subpanel 3: Data generation (Explicit layout without CairoSVG tspan dy collision) -->
    <rect class="d2_panel" x="7" y="206" width="326" height="153" rx="3.2" />
    <text class="d2_serif d2_title" x="20" y="229" font-size="22">Data generation</text>
    <text class="d2_serif" x="20" y="247" font-size="16.5">Physics-guided synthetic samples</text>

    <!-- Simulator box -->
    <rect x="15" y="266" width="70" height="42" fill="#dff1fb" stroke="#2c739e" stroke-width="1.25" />
    <g class="d2_math" font-style="italic">
      <text x="25" y="293" font-size="21">f</text>
      <text x="34" y="299" font-size="12.5" font-style="normal">phys</text>
      <text x="56" y="293" font-size="18">(θ)</text>
    </g>
    <text class="d2_serif" x="50" y="324" text-anchor="middle" font-size="14.5">physical</text>
    <text class="d2_serif" x="50" y="340" text-anchor="middle" font-size="14.5">simulator</text>

    <line x1="88" y1="287" x2="108" y2="287" stroke="#111" stroke-width="1.45" marker-end="url(#d2_arrow)" />

    <text class="d2_math" x="114" y="287" font-size="18" font-style="italic">{</text>
    <text class="d2_math" x="122" y="287" font-size="16" font-style="italic">x</text>
    <text class="d2_math" x="131" y="277" font-size="11" font-style="italic">(i)</text>
    <text class="d2_math" x="142" y="287" font-size="16" font-style="italic">,</text>
    <text class="d2_math" x="148" y="287" font-size="16" font-style="italic">y</text>
    <text class="d2_math" x="157" y="277" font-size="11" font-style="italic">(i)</text>
    <text class="d2_math" x="170" y="287" font-size="18" font-style="italic">}</text>
    <text class="d2_math" x="177" y="276" font-size="11" font-style="italic">N</text>
    <text class="d2_math" x="176" y="296" font-size="10" font-style="italic">i=1</text>

    <line x1="200" y1="287" x2="224" y2="287" stroke="#111" stroke-width="1.45" marker-end="url(#d2_arrow)" />

    <g class="d2_math" font-style="italic">
      <text x="236" y="261" font-size="16">X</text>
      <text x="252" y="261" font-size="16" font-style="normal" font-family="'Segoe UI Symbol', 'Cambria Math', sans-serif">∈</text>
      <text x="269" y="261" font-size="16" font-weight="bold">R</text>
      <text x="282" y="255" font-size="11">N×T×d</text>
    </g>
    <rect x="232" y="266" width="88" height="65" fill="#f4fbff" stroke="#2c85ba" stroke-width="1.25" />
    <g clip-path="url(#d2_matrixClip)">
      <line x1="238" y1="275" x2="314" y2="275" stroke="#8ac7f3" stroke-width="1" />
      <line x1="238" y1="287" x2="314" y2="287" stroke="#8ac7f3" stroke-width="1" />
      <line x1="238" y1="299" x2="314" y2="299" stroke="#8ac7f3" stroke-width="1" />
      <line x1="238" y1="311" x2="314" y2="311" stroke="#8ac7f3" stroke-width="1" />
      <line x1="238" y1="323" x2="314" y2="323" stroke="#8ac7f3" stroke-width="1" />
      <path d="M236 280 C244 274,250 282,257 277 S272 284,281 278 S296 284,314 278" stroke="#146dff" stroke-width="1.8" fill="none" />
      <path d="M236 292 C244 286,251 294,258 289 S272 296,282 290 S297 295,315 290" stroke="#146dff" stroke-width="1.8" fill="none" />
      <path d="M236 304 C245 298,252 306,260 301 S275 308,283 302 S298 308,315 302" stroke="#146dff" stroke-width="1.8" fill="none" />
      <path d="M236 316 C245 310,252 319,260 313 S275 321,283 314 S298 320,315 314" stroke="#146dff" stroke-width="1.8" fill="none" />
    </g>
    <circle cx="276" cy="339" r="1.15" fill="#111" />
    <circle cx="276" cy="344" r="1.15" fill="#111" />
    <circle cx="276" cy="349" r="1.15" fill="#111" />

    <!-- Subpanel 4: Post-hoc correction -->
    <rect class="d2_panel" x="336" y="206" width="352" height="153" rx="3.2" />
    <text class="d2_serif d2_title" x="351" y="229" font-size="22">Post-hoc correction</text>
    <text class="d2_serif" x="351" y="247" font-size="16.5">Projection / clipping after inference</text>

    <!-- Axes without ticks -->
    <line x1="365" y1="316" x2="480" y2="316" stroke="#111" stroke-width="1.25" marker-end="url(#d2_arrow)" />
    <line x1="365" y1="316" x2="365" y2="258" stroke="#111" stroke-width="1.25" marker-end="url(#d2_arrow)" />
    
    <!-- Clean x-axis label without CairoSVG tspan dislocation bug -->
    <text class="d2_serif" x="388" y="336" font-size="16">Time step</text>
    <text class="d2_serif" x="454" y="336" font-size="16" font-style="italic">t</text>
    <g transform="translate(348,288) rotate(-90)"><text class="d2_serif" text-anchor="middle" font-size="15.5">Power (p.u.)</text></g>

    <!-- Physical bound ceiling reference line -->
    <line x1="365" y1="274" x2="480" y2="274" stroke="#718096" stroke-width="1.3" stroke-dasharray="4 3" />

    <!-- Raw curve (dashed) and Clipped curve (solid) -->
    <polyline points="365.0,309.0 371.0,301.6 377.0,294.3 383.0,287.4 389.0,281.0 395.0,275.3 401.0,270.5 407.0,266.6 413.0,263.7 419.0,262.0 425.0,261.4 431.0,262.0 437.0,263.7 443.0,266.6 449.0,270.5 455.0,275.3 461.0,281.0 467.0,287.4 473.0,294.3 479.0,301.6 480.0,302.8" fill="none" stroke="#8da4b7" stroke-width="2.3" stroke-dasharray="7 5" stroke-linejoin="round" stroke-linecap="round" />
    <polyline points="365.0,309.0 371.0,301.6 377.0,294.3 383.0,287.4 389.0,281.0 395.0,275.3 401.0,274.0 407.0,274.0 413.0,274.0 419.0,274.0 425.0,274.0 431.0,274.0 437.0,274.0 443.0,274.0 449.0,274.0 455.0,275.3 461.0,281.0 467.0,287.4 473.0,294.3 479.0,301.6 480.0,302.8" fill="none" stroke="#096ef1" stroke-width="3.0" stroke-linejoin="round" stroke-linecap="round" />

    <!-- Legend (3 distinct entries: raw, bound, projected) - Enlarge font to 15.5px -->
    <line x1="492" y1="267" x2="514" y2="267" stroke="#8da4b7" stroke-width="2.3" stroke-dasharray="6 4" />
    <text class="d2_serif" x="520" y="272" font-size="15.5">raw forecast <tspan class="d2_math" font-style="italic">ŷ</tspan><tspan font-size="11" dy="3">raw</tspan></text>
    <line x1="492" y1="289" x2="514" y2="289" stroke="#718096" stroke-width="1.6" stroke-dasharray="4 3" />
    <text class="d2_serif" x="520" y="293" font-size="15.5">physical bound</text>
    <line x1="492" y1="311" x2="514" y2="311" stroke="#096ef1" stroke-width="3" />
    <text class="d2_serif" x="520" y="315" font-size="15.5">projected output <tspan class="d2_math" font-style="italic">ŷ</tspan><tspan font-size="11" dy="3">proj</tspan></text>
  </svg>
""")

    # -------------------------------------------------------------
    # Panel D3: Inference Telemetry Requirements
    # -------------------------------------------------------------
    svg_parts.append("""  <!-- ==================== Panel D3 ==================== -->
  <svg id="panel_D3" x="5" y="460" width="717" height="326" viewBox="0 0 694 305" preserveAspectRatio="none">
    <defs>
      <marker id="d3_arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L8,4 L0,8 Z" fill="#111" />
      </marker>
      <style>
        .d3_serif { font-family: 'Times New Roman', 'DejaVu Serif', serif; fill:#111; }
        .d3_panelTitle { font-size: 21px; font-weight: 700; fill:#111; }
        .d3_panelSub { font-size: 16.5px; fill:#2d3748; }
        .d3_title { font-size: 23px; font-weight: 700; fill:#111; }
        .d3_subtitle { font-size: 18px; font-style: italic; fill:#2d3748; }
        .d3_math { font-family: 'Times New Roman', 'Segoe UI Symbol', 'Cambria Math', 'DejaVu Serif', serif; font-size: 20px; font-style: italic; fill:#111; }
        .d3_small { font-size: 15.5px; font-weight: 700; fill:#103b5c; }
        .d3_model { fill: #d9eef8; stroke: #4f8fb3; stroke-width: 1.2; }
      </style>
    </defs>

    <!-- Outer card -->
    <rect x="1.5" y="1.5" width="691" height="302" rx="7" fill="#dff7ef" stroke="#085f3d" stroke-width="2" />
    <rect x="5" y="5" width="684" height="295" rx="5" fill="none" stroke="#47ce87" stroke-width="1.6" />
    <line x1="6" y1="59" x2="688" y2="59" stroke="#47ce87" stroke-width="1.6" />

    <!-- Header badge & title -->
    <rect x="8" y="8" width="76" height="48" rx="2" fill="#e9fbf3" stroke="#20b76f" stroke-width="1.3" />
    <text x="46" y="42" class="d3_serif" text-anchor="middle" font-size="36" font-weight="700">D3</text>
    <text x="101" y="31" class="d3_serif d3_title">Inference Telemetry Requirements</text>
    <text x="101" y="53" class="d3_serif d3_subtitle">What external information is required at deployment?</text>

    <!-- Three category boxes -->
    <rect x="8" y="62" width="220" height="226" rx="3" fill="#ffffff" stroke="#3ebd7d" stroke-width="1.3" />
    <rect x="232" y="62" width="228" height="226" rx="3" fill="#ffffff" stroke="#3ebd7d" stroke-width="1.3" />
    <rect x="464" y="62" width="222" height="226" rx="3" fill="#ffffff" stroke="#3ebd7d" stroke-width="1.3" />

    <!-- Subpanel 1: None (Clean axes without coordinate ticks, explicit P_{t-L:t}) -->
    <text x="118" y="88" class="d3_serif d3_panelTitle" text-anchor="middle">None</text>
    <text x="118" y="108" class="d3_serif d3_panelSub" text-anchor="middle">Historical target SCADA</text>
    <g class="d3_math">
      <text x="80" y="153" font-size="20">P</text>
      <text x="93" y="159" font-size="13" font-style="normal">t−L:t</text>
    </g>

    <line x1="61" y1="243" x2="124" y2="243" stroke="#111" stroke-width="1.1" marker-end="url(#d3_arrow)" />
    <line x1="61" y1="243" x2="61" y2="160" stroke="#111" stroke-width="1.1" marker-end="url(#d3_arrow)" />
    <text x="93" y="261" class="d3_serif" font-size="16" text-anchor="middle">Time</text>
    <text x="40" y="203" class="d3_serif" font-size="16" text-anchor="middle" transform="rotate(-90 40 203)">Power (p.u.)</text>

    <polyline points="63.0,208.1 65.9,211.6 68.9,206.4 71.8,210.0 74.8,215.9 77.7,214.3 80.7,221.3 83.6,220.8 86.6,224.4 89.5,224.5 92.4,225.6 95.4,236.3 98.3,241.3 101.3,240.5 104.2,238.9 107.2,237.5 110.1,238.2 113.1,235.8 116.0,235.3" fill="none" stroke="#1478f2" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" />

    <line x1="123" y1="198" x2="135" y2="198" stroke="#111" stroke-width="1.2" marker-end="url(#d3_arrow)" />
    <rect x="137" y="169" width="60" height="54" rx="2" class="d3_model" />
    <text x="167" y="193" class="d3_serif d3_small" text-anchor="middle">forecast</text>
    <text x="167" y="211" class="d3_serif d3_small" text-anchor="middle">model</text>
    <line x1="197" y1="198" x2="211" y2="198" stroke="#111" stroke-width="1.2" marker-end="url(#d3_arrow)" />
    <text x="220" y="204" class="d3_math" text-anchor="middle">ŷ</text>

    <!-- Subpanel 2: Forecast input (Clean dimension indicator without ticks) -->
    <text x="346" y="88" class="d3_serif d3_panelTitle" text-anchor="middle">Forecast input</text>
    <text x="346" y="108" class="d3_serif d3_panelSub" text-anchor="middle">Future NWP</text>

    <g class="d3_math">
      <text x="260" y="146" font-size="19">NWP</text>
      <text x="306" y="153" font-size="12">t+1:t+H</text>
      <text x="361" y="146" font-size="19" font-style="normal" font-family="'Segoe UI Symbol', 'Cambria Math', sans-serif">∈</text>
      <text x="381" y="146" font-size="19" font-style="normal" font-weight="700">R</text>
      <text x="401" y="136" font-size="12">H×d</text>
    </g>

    <text x="257" y="177" class="d3_serif" font-size="15" font-style="italic" text-anchor="middle">v</text>
    <text x="257" y="195" class="d3_serif" font-size="15" font-style="italic" text-anchor="middle">T</text>
    <text x="257" y="213" class="d3_serif" font-size="15" font-style="italic" text-anchor="middle">P</text>
    <g fill="#111"><circle cx="257" cy="224" r="1.2" /><circle cx="257" cy="229" r="1.2" /><circle cx="257" cy="234" r="1.2" /></g>

    <g stroke="#ffffff" stroke-width="1">
      <rect x="272" y="162" width="18" height="16" fill="#57c7d8" /><rect x="290" y="162" width="18" height="16" fill="#49b6ca" /><rect x="308" y="162" width="18" height="16" fill="#d8df67" /><rect x="326" y="162" width="18" height="16" fill="#a9d46b" />
      <rect x="272" y="178" width="18" height="16" fill="#56bdca" /><rect x="290" y="178" width="18" height="16" fill="#80d2c2" /><rect x="308" y="178" width="18" height="16" fill="#e6e87a" /><rect x="326" y="178" width="18" height="16" fill="#62b7d1" />
      <rect x="272" y="194" width="18" height="16" fill="#5ab3c9" /><rect x="290" y="194" width="18" height="16" fill="#d9df67" /><rect x="308" y="194" width="18" height="16" fill="#87cfad" /><rect x="326" y="194" width="18" height="16" fill="#cddf66" />
      <rect x="272" y="210" width="18" height="16" fill="#53c2d4" /><rect x="290" y="210" width="18" height="16" fill="#93d2ab" /><rect x="308" y="210" width="18" height="16" fill="#dce572" /><rect x="326" y="210" width="18" height="16" fill="#80c6b9" />
      <rect x="272" y="226" width="18" height="8" fill="#48afc7" /><rect x="290" y="226" width="18" height="8" fill="#55b7cc" /><rect x="308" y="226" width="18" height="8" fill="#5eb3c7" /><rect x="326" y="226" width="18" height="8" fill="#5bc1ce" />
    </g>
    <rect x="272" y="162" width="72" height="72" fill="none" stroke="#408da9" stroke-width="1" />

    <!-- Clean dimension label -->
    <line x1="272" y1="240" x2="344" y2="240" stroke="#111" stroke-width="1.1" />
    <text x="308" y="259" class="d3_serif" font-size="16" font-style="italic" text-anchor="middle">H hours</text>

    <line x1="346" y1="198" x2="363" y2="198" stroke="#111" stroke-width="1.2" marker-end="url(#d3_arrow)" />
    <rect x="365" y="169" width="60" height="54" rx="2" class="d3_model" />
    <text x="395" y="193" class="d3_serif d3_small" text-anchor="middle">forecast</text>
    <text x="395" y="211" class="d3_serif d3_small" text-anchor="middle">model</text>
    <line x1="425" y1="198" x2="442" y2="198" stroke="#111" stroke-width="1.2" marker-end="url(#d3_arrow)" />
    <text x="451" y="204" class="d3_math" text-anchor="middle">ŷ</text>

    <!-- Subpanel 3: Contemporaneous state -->
    <text x="575" y="88" class="d3_serif d3_panelTitle" text-anchor="middle">Contemporaneous state</text>
    <text x="575" y="106" class="d3_serif d3_panelSub" text-anchor="middle">Synchronized external</text>
    <text x="575" y="124" class="d3_serif d3_panelSub" text-anchor="middle">measurements</text>

    <g class="d3_math">
      <text x="501" y="149" font-size="18">s</text><text x="512" y="139" font-size="10">ext</text><text x="512" y="155" font-size="10">t</text>
      <text x="535" y="149" font-size="18" font-style="normal">= [</text>
      <text x="564" y="149" font-size="18">s</text><text x="574" y="155" font-size="10">t</text><text x="574" y="139" font-size="10">(1)</text>
      <text x="599" y="149" font-size="18" font-style="normal">, …,</text>
      <text x="637" y="149" font-size="18">s</text><text x="647" y="155" font-size="10">t</text><text x="647" y="139" font-size="10">(k)</text>
      <text x="665" y="149" font-size="18" font-style="normal">]</text>
    </g>

    <rect x="476" y="165" width="85" height="85" rx="2" fill="#f6fbfd" stroke="#4f8fb3" stroke-width="1.1" />
    <text x="518.5" y="186" class="d3_serif" font-size="15" text-anchor="middle">SCADA states</text>
    <text x="518.5" y="204" class="d3_serif" font-size="15" text-anchor="middle">grid signals</text>
    <text x="518.5" y="222" class="d3_serif" font-size="15" text-anchor="middle">turbine states</text>
    <g fill="#111"><circle cx="518.5" cy="233" r="1.2" /><circle cx="518.5" cy="238" r="1.2" /><circle cx="518.5" cy="243" r="1.2" /></g>

    <line x1="563" y1="207" x2="584" y2="207" stroke="#111" stroke-width="1.2" marker-end="url(#d3_arrow)" />
    <rect x="587" y="186" width="62" height="52" rx="2" class="d3_model" />
    <text x="618" y="208" class="d3_serif d3_small" text-anchor="middle">forecast</text>
    <text x="618" y="225" class="d3_serif d3_small" text-anchor="middle">model</text>
    <line x1="649" y1="207" x2="668" y2="207" stroke="#111" stroke-width="1.2" marker-end="url(#d3_arrow)" />
    <text x="677" y="213" class="d3_math" text-anchor="middle">ŷ</text>
  </svg>
""")

    # -------------------------------------------------------------
    # Panel D4: Implementation Fidelity
    # -------------------------------------------------------------
    svg_parts.append("""  <!-- ==================== Panel D4 ==================== -->
  <svg id="panel_D4" x="726" y="460" width="717" height="326" viewBox="0 0 699 299" preserveAspectRatio="none">
    <defs>
      <style>
        .d4_serif { font-family: 'Times New Roman', 'DejaVu Serif', serif; fill:#111; }
        .d4_title { font-weight:700; fill:#111; }
        .d4_italic { font-style:italic; }
        .d4_eq { font-family: 'Times New Roman', 'Segoe UI Symbol', 'Cambria Math', 'DejaVu Serif', serif; font-style:italic; fill:#101010; }
      </style>
      <marker id="d4_arrowStart" markerWidth="7" markerHeight="7" refX="1" refY="3.5" orient="auto">
        <path d="M7,0 L0,3.5 L7,7" fill="#111" />
      </marker>
      <marker id="d4_arrowEnd" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
        <path d="M0,0 L7,3.5 L0,7" fill="#111" />
      </marker>
    </defs>

    <!-- Outer card -->
    <rect x="1.5" y="1.5" width="696" height="296" rx="9" fill="#fffef8" stroke="#294e62" stroke-width="2" />
    <rect x="4.5" y="4.5" width="690" height="290" rx="7" fill="none" stroke="#f2bf2d" stroke-width="2.2" />

    <!-- Header badge & title -->
    <rect x="8" y="8" width="75" height="47" rx="2" fill="#fffbea" stroke="#e3a82b" stroke-width="1.1" />
    <line x1="84" y1="8" x2="84" y2="55" stroke="#e3a82b" stroke-width="1.1" />
    <text x="45.5" y="43" text-anchor="middle" class="d4_serif d4_title" font-size="36">D4</text>
    <text x="100" y="32" class="d4_serif d4_title" font-size="24">Implementation Fidelity</text>
    <text x="102" y="53" class="d4_serif d4_italic" font-size="19.2">Can the published formulation be verified in executable code?</text>
    <line x1="8" y1="59" x2="691" y2="59" stroke="#e3a82b" stroke-width="1.1" />

    <!-- Top section: Code availability -->
    <text x="19" y="84" class="d4_serif d4_title" font-size="18.5">Code availability</text>

    <!-- Box 1: Public (Streamlined code snippet) -->
    <rect x="197" y="64" width="235" height="106" rx="4" fill="#f9fdff" stroke="#86b8d4" stroke-width="1.4" />
    <text x="314.5" y="82" text-anchor="middle" class="d4_serif d4_title" font-size="18.5">Public</text>
    <text x="314.5" y="98" text-anchor="middle" class="d4_serif" font-size="15">Source code available</text>
    <rect x="210" y="103" width="209" height="61" rx="4" fill="#edf7ff" stroke="#62aee6" stroke-width="1.2" />

    <!-- Clean editor window: file tab + loss.py + return L_phys -->
    <rect x="218" y="111" width="72" height="19" rx="2" fill="#d8edff" stroke="#90c7f0" stroke-width="0.8" />
    <text x="254" y="125" text-anchor="middle" font-family="'Liberation Mono','DejaVu Sans Mono',monospace" font-size="14" font-weight="bold" fill="#0f5bb5">loss.py</text>
    <text x="225" y="150" font-family="'Liberation Mono','DejaVu Sans Mono',monospace" font-size="16" font-weight="bold" fill="#1c73dc">return </text>
    <text x="284" y="150" class="d4_eq" font-size="18" fill="#1c73dc">ℒ</text>
    <text x="299" y="154" class="d4_serif d4_italic" font-size="13" fill="#1c73dc">phys</text>

    <!-- Box 2: Not found -->
    <rect x="436" y="64" width="245" height="106" rx="4" fill="#fbfdfe" stroke="#86b8d4" stroke-width="1.4" />
    <text x="558.5" y="82" text-anchor="middle" class="d4_serif d4_title" font-size="18.5">Not found</text>
    <text x="558.5" y="98" text-anchor="middle" class="d4_serif" font-size="15">No accessible source code</text>
    <rect x="448" y="103" width="221" height="61" rx="4" fill="#fff" stroke="#7fb1ce" stroke-width="1.5" stroke-dasharray="6,4" />
    <text x="558.5" y="129" class="d4_serif" font-size="16" font-weight="600" fill="#4a5568" text-anchor="middle">Repository omitted</text>
    <text x="558.5" y="149" class="d4_serif" font-size="15" font-style="italic" fill="#718096" text-anchor="middle">(Code not public)</text>

    <!-- Middle divider: Equation-code parity -->
    <line x1="8" y1="176" x2="691" y2="176" stroke="#e3a82b" stroke-width="1.1" />
    <text x="18" y="193" class="d4_serif d4_title" font-size="18.5">Equation–code parity</text>
    <text x="198" y="193" class="d4_serif d4_italic" font-size="16">If code is public, evaluate equation–code parity</text>
    <line x1="8" y1="198" x2="691" y2="198" stroke="#e3a82b" stroke-width="1.1" />

    <!-- Bottom subpanel: Parity check (Widen Published equation box to 138 to prevent border collision) -->
    <rect x="12" y="205" width="138" height="79" rx="4" fill="#fffdfc" stroke="#eab0b0" stroke-width="1.2" />
    <text x="81" y="222" text-anchor="middle" class="d4_serif d4_title" font-size="16">Published equation</text>
    <rect x="23" y="229" width="116" height="47" rx="4" fill="#ffe8ed" stroke="#ff9aac" stroke-width="1.15" />
    <g class="d4_eq">
      <text x="47" y="261" font-size="26">ℒ</text>
      <text x="62" y="266" font-size="13">pub</text>
      <text x="85" y="261" font-size="26">(θ)</text>
    </g>

    <line x1="154" y1="248" x2="173" y2="248" stroke="#111" stroke-width="1.7" marker-start="url(#d4_arrowStart)" marker-end="url(#d4_arrowEnd)" />

    <rect x="177" y="205" width="118" height="79" rx="4" fill="#fffdfc" stroke="#9fbfd3" stroke-width="1.2" />
    <text x="236" y="222" text-anchor="middle" class="d4_serif d4_title" font-size="16.5">Source code</text>
    <rect x="185" y="229" width="102" height="47" rx="4" fill="#eaf6ff" stroke="#8fc4e2" stroke-width="1.15" />
    <g class="d4_eq">
      <text x="203" y="261" font-size="26">ℒ</text>
      <text x="218" y="266" font-size="13">exec</text>
      <text x="248" y="261" font-size="26">(θ)</text>
    </g>

    <rect x="300" y="205" width="190" height="79" rx="4" fill="#effcf5" stroke="#79c7a5" stroke-width="1.2" />
    <text x="395" y="222" text-anchor="middle" class="d4_serif d4_title" font-size="18" fill="#1b5e20">Aligned</text>
    <text x="395" y="238" text-anchor="middle" class="d4_serif" font-size="15">Executable implementation</text>
    <text x="395" y="250" text-anchor="middle" class="d4_serif" font-size="15">matches formulation</text>
    <rect x="307" y="254" width="176" height="24" rx="4" fill="#dff8e8" />
    <g class="d4_eq">
      <text x="324" y="271" font-size="19">ℒ</text>
      <text x="336" y="275" font-size="10">pub</text>
      <text x="352" y="271" font-size="19">(θ)</text>
      <text x="381" y="271" font-size="18">=</text>
      <text x="401" y="271" font-size="19">ℒ</text>
      <text x="413" y="275" font-size="10">exec</text>
      <text x="436" y="271" font-size="19">(θ)</text>
    </g>

    <rect x="496" y="205" width="191" height="79" rx="4" fill="#fff1f4" stroke="#d4a1af" stroke-width="1.2" />
    <text x="591.5" y="222" text-anchor="middle" class="d4_serif d4_title" font-size="18" fill="#c62828">Mismatch</text>
    <text x="591.5" y="238" text-anchor="middle" class="d4_serif" font-size="15">Discrepancy between</text>
    <text x="591.5" y="250" text-anchor="middle" class="d4_serif" font-size="15">formulation and code</text>
    <rect x="504" y="254" width="175" height="24" rx="4" fill="#f8e4ea" />
    <g class="d4_eq">
      <text x="521" y="271" font-size="19">ℒ</text>
      <text x="533" y="275" font-size="10">pub</text>
      <text x="549" y="271" font-size="19">(θ)</text>
      <text x="578" y="271" font-size="18">≠</text>
      <text x="598" y="271" font-size="19">ℒ</text>
      <text x="610" y="275" font-size="10">exec</text>
      <text x="633" y="271" font-size="19">(θ)</text>
    </g>
  </svg>
""")

    # -------------------------------------------------------------
    # Panel D5: Evaluation Validity & Harm Reporting (Schema v4)
    # -------------------------------------------------------------
    svg_parts.append("""  <!-- ==================== Panel D5 ==================== -->
  <svg id="panel_D5" x="5" y="791" width="1438" height="285" viewBox="0 0 1400 260" preserveAspectRatio="none">
    <defs>
      <marker id="d5_arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L8,4 L0,8 z" fill="#111" />
      </marker>
      <style>
        .d5_serif { font-family: 'Times New Roman', 'DejaVu Serif', serif; fill:#111; }
        .d5_math { font-family: 'Times New Roman', 'Segoe UI Symbol', 'Cambria Math', 'DejaVu Serif', serif; font-style:italic; fill:#111; }
      </style>
    </defs>
    <rect x="0" y="0" width="1400" height="260" fill="white" />

    <!-- Outer card -->
    <rect x="5" y="2" width="1390" height="252" rx="5" fill="#f7f0ff" stroke="#7d29d6" stroke-width="2.1" />
    <rect x="10" y="6" width="76" height="45" rx="2" fill="#eee2ff" stroke="#a55ce8" stroke-width="1.2" />
    <text x="48" y="39" class="d5_serif" font-size="38" font-weight="bold" text-anchor="middle">D5</text>
    <text x="100" y="29" class="d5_serif" font-size="26" font-weight="bold" text-anchor="start">Evaluation Validity &amp; Harm Reporting</text>
    <text x="102" y="51" class="d5_serif" font-size="19" font-style="italic" text-anchor="start" fill="#2d3748">Does the evaluation isolate the physical prior and expose failure regimes?</text>

    <!-- Two major sub-containers: Ablation Semantics (left) and Failure/Harm Reporting (right) -->
    <rect x="12" y="55" width="729" height="193" rx="4" fill="#f9fbff" stroke="#92a6bc" stroke-width="1.2" />
    <rect x="746" y="55" width="642" height="193" rx="4" fill="#f9fbff" stroke="#92a6bc" stroke-width="1.2" />

    <!-- Subfield Headers (y=72 text; subboxes start at y=81 leaving 9px padding, preventing descender collision) -->
    <text x="20" y="72" class="d5_serif" font-size="21" font-weight="bold">Ablation semantics</text>
    <text x="757" y="72" class="d5_serif" font-size="21" font-weight="bold">Failure / harm reporting</text>

    <!-- Subpanel boxes (y=81 to y=242, height 161) -->
    <rect x="15" y="81" width="223" height="161" rx="3" fill="#fbfdff" stroke="#8ba3ba" stroke-width="1.1" />
    <rect x="240" y="81" width="264" height="161" rx="3" fill="#fbfdff" stroke="#8ba3ba" stroke-width="1.1" />
    <rect x="506" y="81" width="232" height="161" rx="3" fill="#fbfdff" stroke="#8ba3ba" stroke-width="1.1" />
    <rect x="750" y="81" width="315" height="161" rx="3" fill="#fbfdff" stroke="#8ba3ba" stroke-width="1.1" />
    <rect x="1071" y="81" width="315" height="161" rx="3" fill="#fbfdff" stroke="#8ba3ba" stroke-width="1.1" />

    <!-- Category 1: Constraint on/off -->
    <text x="131" y="96" class="d5_serif" font-size="19" font-weight="bold" text-anchor="middle">Constraint on/off</text>
    <text x="135" y="109" class="d5_serif" font-size="13" text-anchor="middle" fill="#2d3748">Identical backbone; λ = 0 ↔ λ &gt; 0</text>
    <line x1="62" y1="206" x2="230" y2="206" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
    <line x1="62" y1="206" x2="62" y2="124" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
    <text x="145" y="226" class="d5_serif" font-size="16" text-anchor="middle">Forecast horizon h</text>
    <text x="36" y="174" class="d5_serif" font-size="13.5" transform="rotate(-90 36 174)" text-anchor="middle">Normalized metric (↓)</text>
    <polyline points="62.0,195.0 80.7,193.9 99.3,192.2 118.0,190.2 136.7,188.0 155.3,185.7 174.0,183.2 192.7,180.6 211.3,177.8 230.0,175.0" fill="none" stroke="#1e76ee" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round" />
    <polyline points="62.0,195.0 80.7,192.4 99.3,188.5 118.0,184.0 136.7,179.0 155.3,173.6 174.0,167.8 192.7,161.8 211.3,155.5 230.0,149.0" fill="none" stroke="#6f8499" stroke-width="2.1" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="7 5" />
    <line x1="70" y1="129" x2="90" y2="129" stroke="#6f8499" stroke-width="2.1" stroke-dasharray="7 5" />
    <text x="95" y="133" class="d5_serif" font-size="13.5">no constraint (λ = 0)</text>
    <line x1="70" y1="143" x2="90" y2="143" stroke="#1e76ee" stroke-width="2.3" />
    <text x="95" y="147" class="d5_serif" font-size="13.5">constrained (λ &gt; 0)</text>

    <!-- Category 2: Module ablation (Refined typography, perfectly aligned colons, generous box margins) -->
    <text x="372" y="96" class="d5_serif" font-size="19" font-weight="bold" text-anchor="middle">Module ablation</text>
    <text x="372" y="112" class="d5_serif" font-size="14" text-anchor="middle" fill="#2d3748">Full model ↔ neural block removed</text>

    <!-- Row 1: Full Pipeline (Center y=141) -->
    <text x="296" y="145" class="d5_serif" font-size="13" font-weight="bold" text-anchor="end" fill="#1b3a57">Full:</text>
    <text x="312" y="145" class="d5_math" font-size="16.5" text-anchor="middle">x</text>
    <line x1="319" y1="141" x2="328" y2="141" stroke="#111" stroke-width="1.2" marker-end="url(#d5_arrow)" />
    <rect x="330" y="123" width="62" height="36" rx="3" fill="#eaf4fb" stroke="#5b86a6" stroke-width="1.2" />
    <text x="361" y="145" class="d5_serif" font-size="11.5" font-weight="bold" text-anchor="middle" fill="#1e405b">backbone</text>
    <line x1="394" y1="141" x2="404" y2="141" stroke="#111" stroke-width="1.2" marker-end="url(#d5_arrow)" />
    <rect x="406" y="123" width="72" height="36" rx="3" fill="#eaf4fb" stroke="#5b86a6" stroke-width="1.2" />
    <text x="442" y="136" class="d5_serif" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#1e405b">constraint</text>
    <text x="442" y="150" class="d5_serif" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#1e405b">module</text>
    <line x1="480" y1="141" x2="490" y2="141" stroke="#111" stroke-width="1.2" marker-end="url(#d5_arrow)" />
    <text x="498" y="146" class="d5_math" font-size="17.5" text-anchor="middle">ŷ</text>

    <!-- Row 2: Ablated Pipeline (Center y=195) -->
    <text x="296" y="199" class="d5_serif" font-size="13" font-weight="bold" text-anchor="end" fill="#1b3a57">Ablated:</text>
    <text x="312" y="199" class="d5_math" font-size="16.5" text-anchor="middle">x</text>
    <line x1="319" y1="195" x2="328" y2="195" stroke="#111" stroke-width="1.2" marker-end="url(#d5_arrow)" />
    <rect x="330" y="177" width="62" height="36" rx="3" fill="#eaf4fb" stroke="#5b86a6" stroke-width="1.2" />
    <text x="361" y="199" class="d5_serif" font-size="11.5" font-weight="bold" text-anchor="middle" fill="#1e405b">backbone</text>
    <line x1="394" y1="195" x2="404" y2="195" stroke="#111" stroke-width="1.2" marker-end="url(#d5_arrow)" />
    <rect x="406" y="177" width="72" height="36" rx="3" fill="#ffffff" stroke="#78909c" stroke-width="1.2" stroke-dasharray="4 3" />
    <text x="442" y="199" class="d5_serif" font-size="11.5" font-style="italic" fill="#78909c" text-anchor="middle">excised</text>
    <line x1="480" y1="195" x2="490" y2="195" stroke="#111" stroke-width="1.2" marker-end="url(#d5_arrow)" />
    <text x="498" y="200" class="d5_math" font-size="17.5" text-anchor="middle">ŷ</text>

    <!-- Category 3: Both (Consistent hierarchy: Full lowest, Module middle, No constraint highest) -->
    <text x="617" y="96" class="d5_serif" font-size="19" font-weight="bold" text-anchor="middle">Both</text>
    <text x="622" y="109" class="d5_serif" font-size="13" text-anchor="middle" fill="#2d3748">Constraint on/off &amp; module ablation</text>
    <line x1="548" y1="206" x2="725" y2="206" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
    <line x1="548" y1="206" x2="548" y2="124" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
    <text x="637" y="226" class="d5_serif" font-size="16" text-anchor="middle">Forecast horizon h</text>
    <text x="522" y="174" class="d5_serif" font-size="13.5" transform="rotate(-90 522 174)" text-anchor="middle">Normalized metric (↓)</text>
    <polyline points="548.0,195.0 567.7,193.9 587.3,192.2 607.0,190.2 626.7,188.0 646.3,185.7 666.0,183.2 685.7,180.6 705.3,177.8 725.0,175.0" fill="none" stroke="#1e76ee" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round" />
    <polyline points="548.0,195.0 567.7,193.1 587.3,190.3 607.0,187.1 626.7,183.5 646.3,179.6 666.0,175.5 685.7,171.2 705.3,166.7 725.0,162.0" fill="none" stroke="#18b94c" stroke-width="2.1" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="7 4" />
    <polyline points="548.0,195.0 567.7,192.4 587.3,188.5 607.0,184.0 626.7,179.0 646.3,173.6 666.0,167.8 685.7,161.8 705.3,155.5 725.0,149.0" fill="none" stroke="#6f8499" stroke-width="2.1" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="7 4" />
    <line x1="556" y1="129" x2="574" y2="129" stroke="#6f8499" stroke-width="2.1" stroke-dasharray="7 4" />
    <text x="578" y="133" class="d5_serif" font-size="13.5">no constraint</text>
    <line x1="556" y1="143" x2="574" y2="143" stroke="#18b94c" stroke-width="2.1" stroke-dasharray="7 4" />
    <text x="578" y="147" class="d5_serif" font-size="13.5">module ablated</text>
    <line x1="556" y1="157" x2="574" y2="157" stroke="#1e76ee" stroke-width="2.3" />
    <text x="578" y="161" class="d5_serif" font-size="13.5">full model</text>

    <!-- Category 4: Reported (PURELY SCHEMATIC failure boundary alpha*, ZERO text collisions) -->
    <text x="907" y="97" class="d5_serif" font-size="19" font-weight="bold" text-anchor="middle">Reported</text>
    <text x="907" y="113" class="d5_serif" font-size="15" text-anchor="middle" fill="#2d3748">Failure boundary disclosed</text>

    <line x1="790" y1="207" x2="1030" y2="207" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
    <line x1="790" y1="207" x2="790" y2="120" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
    <text x="910" y="227" class="d5_serif" font-size="16" text-anchor="middle">Corruption / stress factor α</text>
    <text x="758" y="170" class="d5_serif" font-size="14" transform="rotate(-90 758 170)" text-anchor="middle">Normalized metric (↓)</text>

    <!-- Regimes shading (Baseline at y=170) -->
    <polygon points="790,207 1030,207 1030,170 790,170" fill="#eaf7ed" opacity="0.85" />
    <polygon points="790,170 1030,170 1030,120 790,120" fill="#fff1f2" opacity="0.6" />

    <!-- Baseline: green dashed line at y=170 -->
    <line x1="790" y1="170" x2="1030" y2="170" stroke="#2e7d32" stroke-width="1.2" stroke-dasharray="5 4" />

    <!-- Quadrant 4 (Bottom-Right): Baseline label stacked cleanly below green line -->
    <text x="1026" y="183" class="d5_serif" font-size="11.5" font-weight="bold" text-anchor="end" fill="#2e7d32">Baseline</text>
    <text x="1026" y="195" class="d5_serif" font-size="10" font-style="italic" text-anchor="end" fill="#2e7d32">(no physics)</text>

    <!-- Quadrant 3 (Bottom-Left): Beneficial Regime stacked cleanly below curve -->
    <text x="843" y="193" class="d5_serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#1b5e20">Beneficial</text>
    <text x="843" y="204" class="d5_serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#1b5e20">regime</text>

    <!-- Center Crossover: Balanced schematic crossover line at x=895 -->
    <line x1="895" y1="122" x2="895" y2="207" stroke="#d32f2f" stroke-width="1.3" stroke-dasharray="4 3" />

    <!-- Quadrant 1 (Top-Left): Failure boundary label pointing right to crossover -->
    <text x="890" y="132" class="d5_serif" font-size="11.5" font-style="italic" font-weight="bold" text-anchor="end" fill="#d32f2f">empirical</text>
    <text x="890" y="145" class="d5_serif" font-size="11.5" font-style="italic" font-weight="bold" text-anchor="end" fill="#d32f2f">crossover α* →</text>

    <!-- Quadrant 2 (Top-Right): Adverse regime label placed above curve -->
    <text x="960" y="138" class="d5_serif" font-size="12.5" font-weight="bold" text-anchor="middle" fill="#c62828">Adverse regime</text>

    <!-- Shaded uncertainty band passing through crossover (895, 170) -->
    <polygon points="790.0,181.0 825.0,179.0 860.0,175.0 895.0,170.0 935.0,162.0 980.0,153.0 1030.0,143.0 1030.0,154.0 980.0,163.0 935.0,172.0 895.0,170.0 860.0,181.0 825.0,185.0 790.0,187.0" fill="#ffcdd2" opacity="0.45" />

    <!-- Red curve: starts in beneficial (y=183), crosses at (895, 170), rises into adverse (y=147) -->
    <polyline points="790.0,183.0 825.0,181.0 860.0,177.0 895.0,170.0 935.0,163.0 980.0,155.0 1030.0,147.0" fill="none" stroke="#d32f2f" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round" />

    <!-- Category 5: Not reported (Clean symmetric design, zero clutter) -->
    <text x="1228" y="97" class="d5_serif" font-size="19" font-weight="bold" text-anchor="middle">Not reported</text>
    <text x="1228" y="113" class="d5_serif" font-size="15" text-anchor="middle" fill="#2d3748">Failure boundary omitted</text>

    <line x1="1111" y1="207" x2="1351" y2="207" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
    <line x1="1111" y1="207" x2="1111" y2="120" stroke="#111" stroke-width="1.1" marker-end="url(#d5_arrow)" />
    <text x="1231" y="227" class="d5_serif" font-size="16" text-anchor="middle">Corruption / stress factor α</text>
    <text x="1079" y="170" class="d5_serif" font-size="14" transform="rotate(-90 1079 170)" text-anchor="middle">Normalized metric (↓)</text>

    <!-- Regimes shading: Nominal evaluated region (green below baseline, pale red above baseline) -->
    <polygon points="1111,207 1190,207 1190,170 1111,170" fill="#eaf7ed" opacity="0.85" />
    <polygon points="1111,170 1190,170 1190,120 1111,120" fill="#fff1f2" opacity="0.4" />

    <!-- Unexamined territory shading (x=1190 to 1351) with soft wash -->
    <polygon points="1190,207 1351,207 1351,120 1190,120" fill="#fdf2f4" opacity="0.65" />

    <!-- Baseline: green dashed line at y=170 across entire plot -->
    <line x1="1111" y1="170" x2="1351" y2="170" stroke="#2e7d32" stroke-width="1.2" stroke-dasharray="5 4" />
    <text x="1347" y="183" class="d5_serif" font-size="11.5" font-weight="bold" text-anchor="end" fill="#2e7d32">Baseline</text>
    <text x="1347" y="195" class="d5_serif" font-size="10" font-style="italic" text-anchor="end" fill="#2e7d32">(no physics)</text>

    <!-- Evaluation cutoff boundary dashed line at x=1190 -->
    <line x1="1190" y1="120" x2="1190" y2="207" stroke="#546e7a" stroke-width="1.3" stroke-dasharray="4 3" />
    <text x="1185" y="132" class="d5_serif" font-size="11.5" font-style="italic" font-weight="bold" text-anchor="end" fill="#546e7a">evaluation</text>
    <text x="1185" y="145" class="d5_serif" font-size="11.5" font-style="italic" font-weight="bold" text-anchor="end" fill="#546e7a">cutoff →</text>

    <!-- Nominal region labels below curve -->
    <text x="1150" y="193" class="d5_serif" font-size="11.5" font-weight="bold" text-anchor="middle" fill="#1b5e20">Nominal</text>
    <text x="1150" y="204" class="d5_serif" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#1b5e20">region only</text>

    <!-- Evaluated curve inside nominal region only: blue solid curve ending at x=1190 -->
    <polyline points="1111.0,183.0 1137.0,181.5 1163.0,179.0 1190.0,175.0" fill="none" stroke="#1e76ee" stroke-width="2.3" stroke-linecap="round" />
    <circle cx="1190" cy="175" r="3.0" fill="#1e76ee" />

    <!-- Clean, elegant text centered in unexamined region -->
    <text x="1270" y="140" class="d5_serif" font-size="12.5" font-weight="bold" text-anchor="middle" fill="#c62828">Downstream failure</text>
    <text x="1270" y="154" class="d5_serif" font-size="11.5" font-weight="bold" text-anchor="middle" fill="#c62828">regimes unexamined</text>
  </svg>

</svg>
""")

    return "".join(svg_parts)

def main():
    root = Path(__file__).resolve().parents[1]
    svg_content = build_perfect_svg()

    # Validate XML
    ET.fromstring(svg_content)
    print("XML validation passed successfully.")

    # Write SVG
    svg_out = root / "outputs" / "constraint_lifecycle_taxonomy_assembled.svg"
    svg_out.write_text(svg_content, encoding="utf-8")
    print(f"Written refined SVG to: {svg_out}")

    # Generate PDF and PNG
    pdf_out = root / "outputs" / "constraint_lifecycle_taxonomy_assembled.pdf"
    png_out = root / "outputs" / "constraint_lifecycle_taxonomy_assembled.png"
    cairosvg.svg2pdf(bytestring=svg_content.encode("utf-8"), write_to=str(pdf_out))
    print(f"Generated PDF: {pdf_out} ({pdf_out.stat().st_size} bytes)")
    cairosvg.svg2png(bytestring=svg_content.encode("utf-8"), write_to=str(png_out), dpi=300)
    print(f"Generated PNG: {png_out} ({png_out.stat().st_size} bytes)")

    # Also update outputs/fig_constraint_lifecycle.*
    shutil.copy2(svg_out, root / "outputs" / "fig_constraint_lifecycle.svg")
    shutil.copy2(pdf_out, root / "outputs" / "fig_constraint_lifecycle.pdf")
    shutil.copy2(png_out, root / "outputs" / "fig_constraint_lifecycle.png")

    # Sync to targets
    targets = [
        root / "01_论文投稿材料" / "active_manuscript" / "figures",
        root / "01_论文投稿材料" / "chinese_manuscript" / "figures",
        root / "03_图表与绘图素材",
        root / "02_论文正文与LaTeX",
    ]
    for target_dir in targets:
        if target_dir.exists():
            dest_pdf = target_dir / "fig_constraint_lifecycle.pdf"
            dest_svg = target_dir / "fig_constraint_lifecycle.svg"
            dest_png = target_dir / "fig_constraint_lifecycle.png"
            shutil.copy2(pdf_out, dest_pdf)
            shutil.copy2(svg_out, dest_svg)
            shutil.copy2(png_out, dest_png)
            print(f"Synced figure to: {target_dir}")

    print("All figure sync operations completed successfully!")

if __name__ == "__main__":
    main()
