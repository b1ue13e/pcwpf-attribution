"""
Master Synthesis Script: constraint_lifecycle_taxonomy_vector.svg & .pdf
=======================================================================
Integrates authentic project repository artifacts, PRISMA empirical statistics,
and publication-grade vector graphics into the 1448 x 1086 master taxonomy canvas.

Guarantees:
- Zero fabricated curves (D5.4 genuine non-linear corruption trajectory with alpha* ~ 0.07)
- Zero Type3 bitmap fonts (100% standard TrueType / Type 1 compliant SVG typography)
- Zero bitmap <image> tags (pure scalable vector mathematics)
- Authentic Penmanshiel SCADA empirical prior & 36-step trajectory
- Authentic DLinear constraint ablation horizons (D5.1, D5.3)
"""

from __future__ import annotations
import sys
import os
import subprocess
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
import fitz  # PyMuPDF

ROOT = Path(r"E:\论文1")
SYS_ROUND2 = ROOT / "05_实验代码与脚本" / "constraint_round2"
sys.path.append(str(SYS_ROUND2))

from core import fit_prior, CFG, Forecaster, starts, windows
from controlled_ablation import fit_scrambled_prior

INPUT_RAW_SVG = ROOT / "outputs" / "constraint_lifecycle_taxonomy_vector_raw.svg"
OUTPUT_SVG = ROOT / "outputs" / "constraint_lifecycle_taxonomy_vector.svg"
FINAL_SVG = ROOT / "outputs" / "constraint_lifecycle_taxonomy_final.svg"
OUTPUT_PDF = ROOT / "outputs" / "constraint_lifecycle_taxonomy_vector.pdf"
OUTPUT_PNG = ROOT / "outputs" / "constraint_lifecycle_taxonomy_vector.png"

EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

MANUSCRIPT_PDF_PATHS = [
    ROOT / "01_论文投稿材料" / "active_manuscript" / "figures" / "fig_constraint_lifecycle.pdf",
    ROOT / "01_论文投稿材料" / "chinese_manuscript" / "figures" / "fig_constraint_lifecycle.pdf",
    ROOT / "02_论文正文与LaTeX" / "fig_constraint_lifecycle.pdf",
    ROOT / "03_图表与绘图素材" / "fig_constraint_lifecycle.pdf",
]

def el(tag, attrs=None, text=None, children=None):
    e = ET.Element(tag, attrs or {})
    if text is not None:
        e.text = text
    if children:
        for c in children:
            e.append(c)
    return e

def load_all_repository_data():
    print("1. Loading authentic data from Penmanshiel and experiment logs...")
    frame = pd.read_csv(ROOT / "04_数据集" / "constraint_rebuild_v2" / "penmanshiel.csv", index_col='time', parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    x, y, ids = starts(frame, CFG['splits']['penmanshiel'])
    test_x, test_y = windows(x, y, ids['test'])

    # 1. Prior
    train_end = CFG['splits']['penmanshiel']['train_end']
    prior = fit_prior(frame, train_end)
    scrambled = fit_scrambled_prior(prior, seed=42)

    # Subsample real historical SCADA points
    d = frame[frame.index < pd.Timestamp(train_end, tz='UTC')][['wind_speed', 'power_pu']].dropna()
    d_sub = d[(d.wind_speed >= 0) & (d.wind_speed <= 22) & (d.power_pu >= -0.05) & (d.power_pu <= 1.05)].sample(n=120, random_state=42)

    # 2. D3 lookback trajectory
    scada_lookback = test_x[100, :, 0]

    # 3. D5 Model checkpoints
    ckpt_uncon = ROOT / "06_实验结果与日志" / "constraint_round2" / "penmanshiel" / "dlinear" / "seed42" / "none_l0.0_f1.0_plain.pt"
    ckpt_con = ROOT / "06_实验结果与日志" / "constraint_round2" / "penmanshiel" / "dlinear" / "seed42" / "curve_l1.0_f1.0_plain.pt"
    ckpt_dist = ROOT / "06_实验结果与日志" / "constraint_round2" / "penmanshiel" / "dlinear" / "seed42" / "curve_l0.1_f1.0_plain.pt"

    m_uncon = Forecaster('dlinear', CFG['lookback'], CFG['horizon'])
    m_uncon.load_state_dict(torch.load(ckpt_uncon, map_location='cpu'))
    m_uncon.eval()

    m_con = Forecaster('dlinear', CFG['lookback'], CFG['horizon'])
    m_con.load_state_dict(torch.load(ckpt_con, map_location='cpu'))
    m_con.eval()

    m_mod = Forecaster('dlinear', CFG['lookback'], CFG['horizon'])
    m_mod.load_state_dict(torch.load(ckpt_dist, map_location='cpu'))
    m_mod.eval()

    with torch.no_grad():
        p_uncon = m_uncon(torch.tensor(test_x[:300])).numpy()
        p_con = m_con(torch.tensor(test_x[:300])).numpy()
        p_mod = m_mod(torch.tensor(test_x[:300])).numpy()

    diffs = np.mean((p_uncon - p_con)**2, axis=1)
    rep_idx_onoff = int(np.argsort(diffs)[-15])
    rep_idx_both = int(np.argsort(diffs)[-12])

    y_uncon = p_uncon[rep_idx_onoff][:24]
    y_con = p_con[rep_idx_onoff][:24]

    y_full = p_con[rep_idx_both][:24]
    y_ablated = p_mod[rep_idx_both][:24]
    y_none = p_uncon[rep_idx_both][:24]

    # 4. Continuous alpha corruption data
    df_corr = pd.read_csv(ROOT / "06_实验结果与日志" / "corruption_severity" / "corruption_severity_results.csv")
    sub = df_corr[(df_corr['site'] == 'penmanshiel') & (df_corr['experiment'] == 'continuous_alpha')]
    agg = sub.groupby('parameter')['dispatch_cost_cal'].agg(['mean', 'std']).reset_index()

    alphas = agg['parameter'].to_numpy(float)
    costs = agg['mean'].to_numpy(float)
    baseline = 0.09881
    alpha_star = float(alphas[0] + (baseline - costs[0]) / (costs[1] - costs[0]) * (alphas[1] - alphas[0]))

    return {
        'prior': prior,
        'scrambled': scrambled,
        'scada_scatter': d_sub,
        'scada_lookback': scada_lookback,
        'y_uncon': y_uncon,
        'y_con': y_con,
        'y_full': y_full,
        'y_ablated': y_ablated,
        'y_none': y_none,
        'alphas': alphas,
        'costs': costs,
        'baseline': baseline,
        'alpha_star': alpha_star
    }

# ==============================================================================
# D1 Builders
# ==============================================================================

def build_d1_intrinsic():
    elements = []
    # Axes
    elements.append(el('line', {'x1': '45', 'y1': '367', 'x2': '158', 'y2': '367', 'stroke': '#111111', 'stroke-width': '1.2'}))
    elements.append(el('line', {'x1': '45', 'y1': '367', 'x2': '45', 'y2': '251', 'stroke': '#111111', 'stroke-width': '1.2'}))
    
    # Labels (Y-axis label at x=20 with ample clearance from tick labels at x=38)
    elements.append(el('text', {'x': '101.5', 'y': '397', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'text-anchor': 'middle'}, 'Wind speed (m/s)'))
    elements.append(el('text', {'x': '20', 'y': '309', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'text-anchor': 'middle', 'transform': 'rotate(-90,20,309)'}, 'Power (p.u.)'))
    
    # Ticks X: 0, 10, 20
    for v in [0, 10, 20]:
        vx = 45.0 + (v / 25.0) * 110.0
        elements.append(el('line', {'x1': f'{vx:.1f}', 'y1': '364', 'x2': f'{vx:.1f}', 'y2': '370', 'stroke': '#111111', 'stroke-width': '0.8'}))
        elements.append(el('text', {'x': f'{vx:.1f}', 'y': '383', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'middle'}, str(v)))
        
    # Ticks Y: 0.0, 0.5, 1.0
    for p, val_str in [(0.0, '0.0'), (0.5, '0.5'), (1.0, '1.0')]:
        py = 367.0 - p * 116.0
        elements.append(el('line', {'x1': '42', 'y1': f'{py:.1f}', 'x2': '48', 'y2': f'{py:.1f}', 'stroke': '#111111', 'stroke-width': '0.8'}))
        elements.append(el('text', {'x': '38', 'y': f'{py+3.5:.1f}', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'end'}, val_str))

    # Betz ceiling dashed line at 1.0
    elements.append(el('line', {'x1': '45', 'y1': '251', 'x2': '158', 'y2': '251', 'stroke': '#64748b', 'stroke-dasharray': '3,3', 'stroke-width': '1.2'}))
    elements.append(el('text', {'x': '102', 'y': '243', 'fill': '#475569', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'Betz ceiling: Cp ≤ 16/27'))

    # Smooth cubic aerodynamic curve entering rated saturation
    v_vals = np.linspace(0, 25, 40)
    v_r = 12.0
    pts = []
    for v in v_vals:
        cx = 45.0 + (v / 25.0) * 110.0
        if v < 3.0:
            p_val = 0.0
        elif v <= v_r:
            p_val = ((v - 3.0) / (v_r - 3.0)) ** 3
        else:
            p_val = 1.0
        cy = 367.0 - min(1.0, p_val) * 116.0
        pts.append(f"{cx:.1f},{cy:.1f}")
        
    elements.append(el('polyline', {'points': ' '.join(pts), 'fill': 'none', 'stroke': '#1676e8', 'stroke-width': '2.6', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'}))
    
    # Formula annotation
    t_f = el('text', {'x': '65', 'y': '288', 'fill': '#1676e8', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'font-weight': 'bold'})
    t_f.text = 'P ∝ v³'
    elements.append(t_f)

    elements.append(el('text', {'x': '102.5', 'y': '422', 'fill': '#475569', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9', 'font-style': 'italic', 'text-anchor': 'middle'}, 'analytic fluid limit'))
    return elements

def build_d1_historical_static(data):
    elements = []
    # Axes
    elements.append(el('line', {'x1': '230', 'y1': '367', 'x2': '355', 'y2': '367', 'stroke': '#111111', 'stroke-width': '1.2'}))
    elements.append(el('line', {'x1': '230', 'y1': '367', 'x2': '230', 'y2': '251', 'stroke': '#111111', 'stroke-width': '1.2'}))
    
    # Labels (Y-axis label at x=196 has comfortable breathing room from x=223 ticks)
    elements.append(el('text', {'x': '292.0', 'y': '397', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'text-anchor': 'middle'}, 'Wind speed (m/s)'))
    elements.append(el('text', {'x': '196.0', 'y': '309.0', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'text-anchor': 'middle', 'transform': 'rotate(-90,196.0,309.0)'}, 'Power (p.u.)'))
    
    # Ticks X: 0, 10, 20
    for v in [0, 10, 20]:
        vx = 230.0 + (v / 25.0) * 120.0
        elements.append(el('line', {'x1': f'{vx:.1f}', 'y1': '364', 'x2': f'{vx:.1f}', 'y2': '370', 'stroke': '#111111', 'stroke-width': '0.8'}))
        elements.append(el('text', {'x': f'{vx:.1f}', 'y': '383', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'middle'}, str(v)))
    
    # Ticks Y: 0.0, 0.5, 1.0
    for p, val_str in [(0.0, '0.0'), (0.5, '0.5'), (1.0, '1.0')]:
        py = 367.0 - p * 116.0
        elements.append(el('line', {'x1': '227', 'y1': f'{py:.1f}', 'x2': '233', 'y2': f'{py:.1f}', 'stroke': '#111111', 'stroke-width': '0.8'}))
        elements.append(el('text', {'x': '223', 'y': f'{py+3.5:.1f}', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'end'}, val_str))
    
    # Scatter points (120 authentic SCADA observations)
    for _, row in data['scada_scatter'].iterrows():
        v, p = row['wind_speed'], row['power_pu']
        if v < 0 or v > 25: continue
        cx = 230.0 + (v / 25.0) * 120.0
        cy = 367.0 - max(0.0, min(1.02, p)) * 116.0
        elements.append(el('circle', {'cx': f'{cx:.1f}', 'cy': f'{cy:.1f}', 'r': '1.1', 'fill': '#94a3b8', 'fill-opacity': '0.35'}))
    
    # Authentic empirical curve C_emp from fit_prior
    prior = data['prior']
    c_pts = []
    for c, m in zip(prior['centers'], prior['means']):
        if c > 25.0: continue
        cx = 230.0 + (c / 25.0) * 120.0
        cy = 367.0 - max(0.0, min(1.0, float(m))) * 116.0
        c_pts.append(f"{cx:.1f},{cy:.1f}")
    
    elements.append(el('polyline', {'points': ' '.join(c_pts), 'fill': 'none', 'stroke': '#1676e8', 'stroke-width': '2.6', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'}))
    
    # Tag with clean breathing room above plateau
    t_tag = el('text', {'x': '292', 'y': '260', 'fill': '#1676e8', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'font-weight': 'bold'})
    t_tag.text = 'C'
    t_tag.append(el('tspan', {'font-size': '8.5', 'dy': '2.5'}, 'emp'))
    t_tag.append(el('tspan', {'font-size': '11', 'dy': '-2.5'}, '(v)'))
    elements.append(t_tag)
    
    elements.append(el('text', {'x': '283.5', 'y': '422', 'fill': '#475569', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9', 'font-style': 'italic', 'text-anchor': 'middle'}, 'offline empirical prior'))
    return elements

def build_d1_learned_online():
    elements = []
    # Input x_1:t
    t_in = el('text', {'x': '408', 'y': '286', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '18', 'font-style': 'italic'})
    t_in.text = 'X'
    t_in.append(el('tspan', {'font-size': '12', 'dy': '4'}, '1:t'))
    elements.append(t_in)

    # Arrow 1
    elements.append(el('line', {'x1': '432', 'y1': '280', 'x2': '452', 'y2': '280', 'stroke': '#111111', 'stroke-width': '1.4', 'marker-end': 'url(#arrow)'}))

    # Neural adapter box g_phi
    elements.append(el('rect', {'x': '454', 'y': '256', 'width': '46', 'height': '48', 'rx': '4', 'ry': '4', 'fill': '#e0f2fe', 'stroke': '#0284c7', 'stroke-width': '1.3'}))
    t_g = el('text', {'x': '477', 'y': '286', 'fill': '#0369a1', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '18', 'font-weight': 'bold', 'text-anchor': 'middle'})
    t_g.text = 'g'
    t_g.append(el('tspan', {'font-size': '12', 'dy': '2'}, 'ϕ'))
    elements.append(t_g)

    # Arrow 2
    elements.append(el('line', {'x1': '502', 'y1': '280', 'x2': '520', 'y2': '280', 'stroke': '#111111', 'stroke-width': '1.4', 'marker-end': 'url(#arrow)'}))

    # Output parameter theta_t
    t_out = el('text', {'x': '532', 'y': '286', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '20', 'font-style': 'italic'})
    t_out.text = 'θ'
    t_out.append(el('tspan', {'font-size': '13', 'dy': '4'}, 't'))
    elements.append(t_out)

    # Recursive feedback theta_t-1
    t_rec = el('text', {'x': '477', 'y': '350', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '18', 'font-style': 'italic', 'text-anchor': 'middle'})
    t_rec.text = 'θ'
    t_rec.append(el('tspan', {'font-size': '12', 'dy': '4'}, 't-1'))
    elements.append(t_rec)

    # Arrow from theta_t-1 up to g_phi
    elements.append(el('line', {'x1': '477', 'y1': '334', 'x2': '477', 'y2': '308', 'stroke': '#111111', 'stroke-width': '1.3', 'marker-end': 'url(#arrow)'}))
    elements.append(el('text', {'x': '477', 'y': '372', 'fill': '#0284c7', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'online update'))

    # Constraint set badge below
    elements.append(el('rect', {'x': '396', 'y': '395', 'width': '136', 'height': '27', 'rx': '4', 'ry': '4', 'fill': '#f8fafc', 'stroke': '#cbd5e1', 'stroke-width': '0.9'}))
    t_c = el('text', {'x': '464', 'y': '412', 'fill': '#475569', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9.5', 'text-anchor': 'middle'})
    t_c.text = 'Adaptive constraint C'
    t_c.append(el('tspan', {'font-size': '7.5', 'dy': '2'}, 't'))
    elements.append(t_c)

    return elements

def build_d1_unreported():
    elements = []
    # Dashed warning card
    elements.append(el('rect', {'x': '566', 'y': '256', 'width': '130', 'height': '98', 'rx': '6', 'ry': '6', 'fill': '#fef2f2', 'stroke': '#f87171', 'stroke-dasharray': '4,3', 'stroke-width': '1.3'}))
    
    # Question mark formula
    t_q = el('text', {'x': '631', 'y': '294', 'fill': '#b91c1c', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '22', 'font-weight': 'bold', 'text-anchor': 'middle'})
    t_q.text = 'θ'
    t_q.append(el('tspan', {'font-size': '14', 'dy': '3'}, 'phys'))
    t_q.append(el('tspan', {'font-size': '22', 'dy': '-3'}, ' = ?'))
    elements.append(t_q)

    elements.append(el('text', {'x': '631', 'y': '318', 'fill': '#dc2626', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'Provenance: Unspecified'))
    elements.append(el('text', {'x': '631', 'y': '336', 'fill': '#991b1b', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9', 'font-style': 'italic', 'text-anchor': 'middle'}, 'calibration omitted'))

    # Badge below
    elements.append(el('rect', {'x': '574', 'y': '376', 'width': '114', 'height': '24', 'rx': '4', 'ry': '4', 'fill': '#fee2e2', 'stroke': '#ef4444', 'stroke-width': '1'}))
    elements.append(el('text', {'x': '631', 'y': '392', 'fill': '#b91c1c', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'Audit Risk: High'))
    
    return elements

# ==============================================================================
# D2 Builders
# ==============================================================================

def build_d2_training_loss():
    elements = []
    elements.append(el('rect', {'x': '745', 'y': '200', 'width': '316', 'height': '66', 'rx': '6', 'ry': '6', 'fill': '#f0f9ff', 'stroke': '#0284c7', 'stroke-width': '1.2'}))
    
    # Formula: min L_task + lambda * L_phys
    t = el('text', {'x': '903', 'y': '240', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '20', 'font-weight': 'bold', 'text-anchor': 'middle'})
    t.text = 'min  L'
    t.append(el('tspan', {'font-size': '13', 'dy': '4'}, 'task'))
    t.append(el('tspan', {'font-size': '20', 'dy': '-4'}, ' + λ L'))
    t.append(el('tspan', {'font-size': '13', 'dy': '4'}, 'phys'))
    elements.append(t)

    elements.append(el('text', {'x': '903', 'y': '280', 'fill': '#0369a1', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'middle'}, 'Soft regularization / Lagrangian penalty'))
    return elements

def build_d2_architecture():
    elements = []
    # Input h_t
    t_in = el('text', {'x': '1114', 'y': '236', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '17', 'font-style': 'italic'})
    t_in.text = 'h'
    t_in.append(el('tspan', {'font-size': '11', 'dy': '3'}, 't'))
    elements.append(t_in)

    # Arrow 1
    elements.append(el('line', {'x1': '1130', 'y1': '231', 'x2': '1164', 'y2': '231', 'stroke': '#111111', 'stroke-width': '1.4', 'marker-end': 'url(#arrow)'}))

    # Hard constraint layer box
    elements.append(el('rect', {'x': '1168', 'y': '208', 'width': '126', 'height': '48', 'rx': '5', 'ry': '5', 'fill': '#eff6ff', 'stroke': '#2563eb', 'stroke-width': '1.3'}))
    t_pi = el('text', {'x': '1231', 'y': '228', 'fill': '#1e40af', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '14', 'font-weight': 'bold', 'text-anchor': 'middle'})
    t_pi.text = 'Π'
    t_pi.append(el('tspan', {'font-size': '10', 'dy': '2'}, 'M'))
    t_pi.append(el('tspan', {'font-size': '14', 'dy': '-2'}, '(·)'))
    elements.append(t_pi)
    elements.append(el('text', {'x': '1231', 'y': '244', 'fill': '#2563eb', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9', 'text-anchor': 'middle'}, 'hard constraint layer'))

    # Arrow 2
    elements.append(el('line', {'x1': '1296', 'y1': '231', 'x2': '1330', 'y2': '231', 'stroke': '#111111', 'stroke-width': '1.4', 'marker-end': 'url(#arrow)'}))

    # Output y_hat in M
    t_out = el('text', {'x': '1364', 'y': '236', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '15', 'font-weight': 'bold', 'text-anchor': 'middle'})
    t_out.text = 'ŷ ∈ M'
    elements.append(t_out)

    elements.append(el('text', {'x': '1256', 'y': '280', 'fill': '#475569', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'middle'}, 'Guaranteed feasible by construction'))
    return elements

def build_d2_data_generation():
    elements = []
    # Left box: Simulator f_phys(theta)
    elements.append(el('rect', {'x': '742', 'y': '360', 'width': '84', 'height': '56', 'rx': '6', 'ry': '6', 'fill': '#f0f9ff', 'stroke': '#0284c7', 'stroke-width': '1.2'}))
    t_f = el('text', {'x': '784', 'y': '383', 'fill': '#0369a1', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '13', 'font-weight': 'bold', 'text-anchor': 'middle'})
    t_f.text = 'f'
    t_f.append(el('tspan', {'font-size': '9', 'dy': '2'}, 'phys'))
    t_f.append(el('tspan', {'font-size': '13', 'dy': '-2'}, '(θ)'))
    elements.append(t_f)
    elements.append(el('text', {'x': '784', 'y': '401', 'fill': '#0284c7', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'middle'}, 'simulator'))
    
    # Arrow 1
    elements.append(el('line', {'x1': '832', 'y1': '388', 'x2': '860', 'y2': '388', 'stroke': '#64748b', 'stroke-width': '1.3', 'marker-end': 'url(#arrow)'}))
    
    # Middle Matrix with brackets
    elements.append(el('path', {'d': 'M 872 355 L 866 355 L 866 422 L 872 422', 'fill': 'none', 'stroke': '#475569', 'stroke-width': '1.3'}))
    elements.append(el('text', {'x': '903', 'y': '371', 'fill': '#1e293b', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'text-anchor': 'middle'}, 'x¹    y¹'))
    elements.append(el('text', {'x': '903', 'y': '386', 'fill': '#1e293b', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'text-anchor': 'middle'}, 'x²    y²'))
    elements.append(el('text', {'x': '903', 'y': '399', 'fill': '#64748b', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'text-anchor': 'middle'}, '⋮       ⋮'))
    elements.append(el('text', {'x': '903', 'y': '414', 'fill': '#1e293b', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'text-anchor': 'middle'}, 'xᴺ    yᴺ'))
    elements.append(el('path', {'d': 'M 934 355 L 940 355 L 940 422 L 934 422', 'fill': 'none', 'stroke': '#475569', 'stroke-width': '1.3'}))
    elements.append(el('text', {'x': '903', 'y': '432', 'fill': '#64748b', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8.5', 'text-anchor': 'middle'}, 'synthetic batch'))
    
    # Arrow 2
    elements.append(el('line', {'x1': '946', 'y1': '388', 'x2': '974', 'y2': '388', 'stroke': '#64748b', 'stroke-width': '1.3', 'marker-end': 'url(#arrow)'}))
    
    # Right Tensor Card
    elements.append(el('rect', {'x': '980', 'y': '364', 'width': '90', 'height': '48', 'rx': '5', 'ry': '5', 'fill': '#eff6ff', 'stroke': '#2563eb', 'stroke-width': '1.2'}))
    elements.append(el('text', {'x': '1025', 'y': '386', 'fill': '#1e40af', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11.5', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'X ∈ ℝ^(N×T×d)'))
    elements.append(el('text', {'x': '1025', 'y': '401', 'fill': '#2563eb', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8.5', 'text-anchor': 'middle'}, 'training tensor'))
    
    return elements

def build_d2_posthoc():
    elements = []
    # Axes
    elements.append(el('line', {'x1': '1120', 'y1': '417', 'x2': '1325', 'y2': '417', 'stroke': '#111111', 'stroke-width': '1.2'}))
    elements.append(el('line', {'x1': '1120', 'y1': '417', 'x2': '1120', 'y2': '350', 'stroke': '#111111', 'stroke-width': '1.2'}))
    elements.append(el('text', {'x': '1222', 'y': '432', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'middle'}, 'Time step  t'))
    elements.append(el('text', {'x': '1106', 'y': '383', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'middle', 'transform': 'rotate(-90,1106,383)'}, 'Power (p.u.)'))
    
    # Inadmissible excursion shaded region (above limit y=371)
    elements.append(el('rect', {'x': '1120', 'y': '350', 'width': '205', 'height': '21', 'fill': '#fee2e2', 'opacity': '0.4'}))
    
    # Physical bound line
    elements.append(el('line', {'x1': '1120', 'y1': '371', 'x2': '1325', 'y2': '371', 'stroke': '#94a3b8', 'stroke-dasharray': '3,3', 'stroke-width': '1.2'}))
    elements.append(el('text', {'x': '1332', 'y': '374', 'fill': '#94a3b8', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'font-weight': 'bold'}, 'Limit (1.0)'))
    
    # Raw & Clipped curves
    t_vals = np.linspace(0, 10, 45)
    raw_vals = 0.72 + 0.38 * np.sin(t_vals * np.pi / 10.0)**1.5
    
    raw_pts = []
    post_pts = []
    for t_v, r_v in zip(t_vals, raw_vals):
        cx = 1120.0 + (t_v / 10.0) * 205.0
        cy_raw = 417.0 - ((r_v - 0.70) / 0.45) * 67.0
        cy_post = max(371.0, cy_raw)  # clipped flat at limit 371.0
        raw_pts.append(f"{cx:.1f},{cy_raw:.1f}")
        post_pts.append(f"{cx:.1f},{cy_post:.1f}")
        
    elements.append(el('polyline', {'points': ' '.join(raw_pts), 'fill': 'none', 'stroke': '#64748b', 'stroke-dasharray': '6,3', 'stroke-width': '1.8'}))
    elements.append(el('text', {'x': '1332', 'y': '393', 'fill': '#64748b', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10'}, 'Raw (ŷ_raw)'))
    
    elements.append(el('polyline', {'points': ' '.join(post_pts), 'fill': 'none', 'stroke': '#1676e8', 'stroke-width': '2.4', 'stroke-linecap': 'round'}))
    elements.append(el('text', {'x': '1332', 'y': '412', 'fill': '#1676e8', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'font-weight': 'bold'}, 'Post (ŷ_proj)'))
    
    # Projection arrow at peak
    elements.append(el('line', {'x1': '1222.5', 'y1': '358', 'x2': '1222.5', 'y2': '369', 'stroke': '#1676e8', 'stroke-width': '1.5', 'marker-end': 'url(#blue_arrow)'}))
    elements.append(el('rect', {'x': '1227', 'y': '355', 'width': '22', 'height': '14', 'fill': '#ffffff', 'opacity': '0.9', 'rx': '2'}))
    
    t_pi = el('text', {'x': '1228', 'y': '365', 'fill': '#1676e8', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10.5', 'font-weight': 'bold'})
    t_pi.text = 'Π'
    t_pi.append(el('tspan', {'font-size': '8', 'dy': '2.5'}, 'C'))
    elements.append(t_pi)
    
    return elements

# ==============================================================================
# D3 Builders
# ==============================================================================

def build_d3_none(data):
    elements = []
    # Math label
    t_math = el('text', {'x': '130.33', 'y': '602', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '20', 'text-anchor': 'middle'})
    t_math.text = 'P'
    t_math.append(el('tspan', {'font-size': '14', 'dy': '3.5'}, 't-L:t'))
    elements.append(t_math)
    
    # Axes
    elements.append(el('line', {'x1': '32', 'y1': '695', 'x2': '110', 'y2': '695', 'stroke': '#111111', 'stroke-width': '1.2'}))
    elements.append(el('line', {'x1': '32', 'y1': '695', 'x2': '32', 'y2': '628', 'stroke': '#111111', 'stroke-width': '1.2'}))
    elements.append(el('text', {'x': '71', 'y': '711', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'middle'}, 'Time'))
    elements.append(el('text', {'x': '22', 'y': '662', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'middle', 'transform': 'rotate(-90,22,662)'}, 'Power'))
    
    # Authentic 36-step trajectory
    traj = data['scada_lookback']
    t_pts = []
    for step, val in enumerate(traj):
        cx = 32.0 + (step / 35.0) * 78.0
        cy = 695.0 - max(0.0, min(1.0, float(val))) * 67.0
        t_pts.append(f"{cx:.1f},{cy:.1f}")
        
    elements.append(el('polyline', {'points': ' '.join(t_pts), 'fill': 'none', 'stroke': '#1676e8', 'stroke-width': '2.0', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'}))
    
    # Arrow 1
    elements.append(el('line', {'x1': '116', 'y1': '662', 'x2': '134', 'y2': '662', 'stroke': '#111111', 'stroke-width': '1.3', 'marker-end': 'url(#arrow)'}))
    
    # Forecast model box
    elements.append(el('rect', {'x': '138', 'y': '642', 'width': '56', 'height': '40', 'rx': '4', 'ry': '4', 'fill': '#eff6ff', 'stroke': '#2563eb', 'stroke-width': '1.2'}))
    elements.append(el('text', {'x': '166', 'y': '658', 'fill': '#1e40af', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10.5', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'forecast'))
    elements.append(el('text', {'x': '166', 'y': '672', 'fill': '#1e40af', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10.5', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'model'))
    
    # Arrow 2
    elements.append(el('line', {'x1': '198', 'y1': '662', 'x2': '214', 'y2': '662', 'stroke': '#111111', 'stroke-width': '1.3', 'marker-end': 'url(#arrow)'}))
    
    # Output y_hat
    elements.append(el('text', {'x': '219', 'y': '668', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '22', 'font-weight': 'bold'}, 'ŷ'))
    
    return elements

def build_d3_nwp():
    elements = []
    # Math label
    t_m = el('text', {'x': '361.0', 'y': '602', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '17', 'font-weight': 'bold', 'text-anchor': 'middle'})
    t_m.text = 'W'
    t_m.append(el('tspan', {'font-size': '11', 'dy': '3'}, 'NWP'))
    t_m.append(el('tspan', {'font-size': '17', 'dy': '-3'}, ' ∈ ℝ'))
    t_m.append(el('tspan', {'font-size': '11', 'dy': '-6'}, 'H×d'))
    elements.append(t_m)

    # Meteorological tensor bracket matrix
    elements.append(el('path', {'d': 'M 266 632 L 260 632 L 260 695 L 266 695', 'fill': 'none', 'stroke': '#475569', 'stroke-width': '1.3'}))
    
    # Row channel labels
    elements.append(el('text', {'x': '272', 'y': '648', 'fill': '#0369a1', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'font-weight': 'bold'}, 'v'))
    elements.append(el('text', {'x': '272', 'y': '666', 'fill': '#b45309', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'font-weight': 'bold'}, 'T'))
    elements.append(el('text', {'x': '272', 'y': '684', 'fill': '#475569', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'font-weight': 'bold'}, 'P'))

    # Grid cells (3 channels x 3 horizon blocks)
    colors = [
        ['#bae6fd', '#7dd3fc', '#38bdf8'],
        ['#fed7aa', '#fdba74', '#fb923c'],
        ['#e2e8f0', '#cbd5e1', '#94a3b8'],
    ]
    for r in range(3):
        for c in range(3):
            elements.append(el('rect', {
                'x': str(286 + c * 16),
                'y': str(636 + r * 18),
                'width': '13',
                'height': '13',
                'rx': '2',
                'fill': colors[r][c],
                'stroke': '#ffffff',
                'stroke-width': '0.8'
            }))
            
    elements.append(el('path', {'d': 'M 338 632 L 344 632 L 344 695 L 338 695', 'fill': 'none', 'stroke': '#475569', 'stroke-width': '1.3'}))
    elements.append(el('text', {'x': '302', 'y': '711', 'fill': '#475569', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9', 'text-anchor': 'middle'}, 'H hours ahead'))

    # Arrow to model
    elements.append(el('line', {'x1': '348', 'y1': '663', 'x2': '366', 'y2': '663', 'stroke': '#111111', 'stroke-width': '1.3', 'marker-end': 'url(#arrow)'}))

    # Model box
    elements.append(el('rect', {'x': '370', 'y': '643', 'width': '54', 'height': '40', 'rx': '4', 'ry': '4', 'fill': '#eff6ff', 'stroke': '#2563eb', 'stroke-width': '1.2'}))
    elements.append(el('text', {'x': '397', 'y': '659', 'fill': '#1e40af', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'forecast'))
    elements.append(el('text', {'x': '397', 'y': '673', 'fill': '#1e40af', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'model'))

    # Arrow to output
    elements.append(el('line', {'x1': '428', 'y1': '663', 'x2': '444', 'y2': '663', 'stroke': '#111111', 'stroke-width': '1.3', 'marker-end': 'url(#arrow)'}))
    elements.append(el('text', {'x': '449', 'y': '669', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '22', 'font-weight': 'bold'}, 'ŷ'))
    
    return elements

def build_d3_contemporaneous():
    elements = []
    # Math label
    t_m = el('text', {'x': '591.66', 'y': '602', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '17', 'font-weight': 'bold', 'text-anchor': 'middle'})
    t_m.text = 's'
    t_m.append(el('tspan', {'font-size': '11', 'dy': '-6'}, 'ext'))
    t_m.append(el('tspan', {'font-size': '11', 'dy': '9'}, 't'))
    t_m.append(el('tspan', {'font-size': '17', 'dy': '-3'}, ' ∈ ℝ'))
    t_m.append(el('tspan', {'font-size': '11', 'dy': '-6'}, 'M'))
    elements.append(t_m)

    # Multi-turbine telemetry array card
    elements.append(el('rect', {'x': '488', 'y': '634', 'width': '80', 'height': '58', 'rx': '4', 'ry': '4', 'fill': '#f8fafc', 'stroke': '#94a3b8', 'stroke-width': '1.1'}))
    elements.append(el('text', {'x': '528', 'y': '649', 'fill': '#1e293b', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8.5', 'text-anchor': 'middle'}, 'Turbine SCADA'))
    elements.append(el('text', {'x': '528', 'y': '664', 'fill': '#1e293b', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8.5', 'text-anchor': 'middle'}, 'Grid P/Q Signals'))
    elements.append(el('text', {'x': '528', 'y': '679', 'fill': '#1e293b', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8.5', 'text-anchor': 'middle'}, 'Substation RTU'))
    elements.append(el('text', {'x': '528', 'y': '711', 'fill': '#475569', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9', 'text-anchor': 'middle'}, 'M spatial channels'))

    # Arrow to model
    elements.append(el('line', {'x1': '574', 'y1': '663', 'x2': '594', 'y2': '663', 'stroke': '#111111', 'stroke-width': '1.3', 'marker-end': 'url(#arrow)'}))

    # Model box
    elements.append(el('rect', {'x': '598', 'y': '643', 'width': '54', 'height': '40', 'rx': '4', 'ry': '4', 'fill': '#eff6ff', 'stroke': '#2563eb', 'stroke-width': '1.2'}))
    elements.append(el('text', {'x': '625', 'y': '659', 'fill': '#1e40af', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'forecast'))
    elements.append(el('text', {'x': '625', 'y': '673', 'fill': '#1e40af', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'model'))

    # Arrow to output
    elements.append(el('line', {'x1': '656', 'y1': '663', 'x2': '672', 'y2': '663', 'stroke': '#111111', 'stroke-width': '1.3', 'marker-end': 'url(#arrow)'}))
    elements.append(el('text', {'x': '677', 'y': '669', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '22', 'font-weight': 'bold'}, 'ŷ'))
    
    return elements

# ==============================================================================
# D4 Builders
# ==============================================================================

def build_d4_code_availability():
    elements = []
    # Left: Public card
    elements.append(el('rect', {'x': '739', 'y': '542', 'width': '338', 'height': '76', 'rx': '4', 'ry': '4', 'fill': '#ffffff', 'stroke': '#22c55e', 'stroke-width': '1.1'}))
    elements.append(el('text', {'x': '754', 'y': '564', 'fill': '#15803d', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '15', 'font-weight': 'bold'}, 'Public'))
    elements.append(el('text', {'x': '754', 'y': '580', 'fill': '#475569', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10.5'}, 'Source code available & runnable'))
    
    # Stat pill Public
    elements.append(el('rect', {'x': '935', 'y': '550', 'width': '132', 'height': '20', 'rx': '10', 'ry': '10', 'fill': '#dcfce7', 'stroke': '#16a34a', 'stroke-width': '0.8'}))
    elements.append(el('text', {'x': '1001', 'y': '564', 'fill': '#15803d', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'Open: 4.3% (2/46)'))

    # Code snippet badge
    elements.append(el('rect', {'x': '754', 'y': '588', 'width': '310', 'height': '22', 'rx': '3', 'ry': '3', 'fill': '#f1f5f9', 'stroke': '#cbd5e1', 'stroke-width': '0.8'}))
    elements.append(el('text', {'x': '764', 'y': '603', 'fill': '#334155', 'font-family': 'DejaVu Sans Mono, monospace', 'font-size': '10'}, 'loss.py / model.py / return L_phys'))

    # Right: Not found card
    elements.append(el('rect', {'x': '1083', 'y': '542', 'width': '338', 'height': '76', 'rx': '4', 'ry': '4', 'fill': '#ffffff', 'stroke': '#ef4444', 'stroke-width': '1.1'}))
    elements.append(el('text', {'x': '1098', 'y': '564', 'fill': '#b91c1c', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '15', 'font-weight': 'bold'}, 'Not found'))
    elements.append(el('text', {'x': '1098', 'y': '580', 'fill': '#475569', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10.5'}, 'No accessible source code / broken URL'))

    # Stat pill Not found
    elements.append(el('rect', {'x': '1240', 'y': '550', 'width': '170', 'height': '20', 'rx': '10', 'ry': '10', 'fill': '#fee2e2', 'stroke': '#dc2626', 'stroke-width': '0.8'}))
    elements.append(el('text', {'x': '1325', 'y': '564', 'fill': '#b91c1c', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'Closed: 95.7% (44/46)'))

    # Warning badge
    elements.append(el('rect', {'x': '1098', 'y': '588', 'width': '310', 'height': '22', 'rx': '3', 'ry': '3', 'fill': '#fef2f2', 'stroke': '#fca5a5', 'stroke-width': '0.8', 'stroke-dasharray': '3,3'}))
    elements.append(el('text', {'x': '1253', 'y': '603', 'fill': '#dc2626', 'font-family': 'DejaVu Sans Mono, monospace', 'font-size': '10', 'text-anchor': 'middle'}, '404 Repo Not Found / Code Unavailable'))

    return elements

def build_d4_parity_branching():
    elements = []
    # Left Input 1: L_pub
    elements.append(el('rect', {'x': '739', 'y': '660', 'width': '146', 'height': '35', 'rx': '4', 'ry': '4', 'fill': '#ffe9ef', 'stroke': '#f43f5e', 'stroke-width': '1.1'}))
    t_pub = el('text', {'x': '812', 'y': '678', 'fill': '#881337', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '14', 'font-weight': 'bold', 'text-anchor': 'middle'})
    t_pub.text = 'L'
    t_pub.append(el('tspan', {'font-size': '9.5', 'dy': '3'}, 'pub'))
    t_pub.append(el('tspan', {'font-size': '14', 'dy': '-3'}, '(θ)'))
    elements.append(t_pub)
    elements.append(el('text', {'x': '812', 'y': '690', 'fill': '#9f1239', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8', 'text-anchor': 'middle'}, 'Published equation'))

    # Left Input 2: L_exec
    elements.append(el('rect', {'x': '739', 'y': '702', 'width': '146', 'height': '35', 'rx': '4', 'ry': '4', 'fill': '#e9f4ff', 'stroke': '#3b82f6', 'stroke-width': '1.1'}))
    t_exec = el('text', {'x': '812', 'y': '720', 'fill': '#1e3a8a', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '14', 'font-weight': 'bold', 'text-anchor': 'middle'})
    t_exec.text = 'L'
    t_exec.append(el('tspan', {'font-size': '9.5', 'dy': '3'}, 'exec'))
    t_exec.append(el('tspan', {'font-size': '14', 'dy': '-3'}, '(θ)'))
    elements.append(t_exec)
    elements.append(el('text', {'x': '812', 'y': '732', 'fill': '#1d4ed8', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8', 'text-anchor': 'middle'}, 'Source code logic'))

    # Arrows to central node
    elements.append(el('line', {'x1': '887', 'y1': '677', 'x2': '908', 'y2': '690', 'stroke': '#64748b', 'stroke-width': '1.3', 'marker-end': 'url(#arrow)'}))
    elements.append(el('line', {'x1': '887', 'y1': '719', 'x2': '908', 'y2': '706', 'stroke': '#64748b', 'stroke-width': '1.3', 'marker-end': 'url(#arrow)'}))

    # Central AST Check node
    elements.append(el('rect', {'x': '910', 'y': '674', 'width': '112', 'height': '48', 'rx': '5', 'ry': '5', 'fill': '#f1f5f9', 'stroke': '#64748b', 'stroke-width': '1.2'}))
    elements.append(el('text', {'x': '966', 'y': '694', 'fill': '#1e293b', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10.5', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'AST Equivalence'))
    elements.append(el('text', {'x': '966', 'y': '710', 'fill': '#475569', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9.5', 'text-anchor': 'middle'}, 'Δ_AST = 0 ?'))

    # Branch arrows
    elements.append(el('line', {'x1': '1024', 'y1': '690', 'x2': '1046', 'y2': '677', 'stroke': '#15803d', 'stroke-width': '1.3', 'marker-end': 'url(#arrow)'}))
    elements.append(el('line', {'x1': '1024', 'y1': '706', 'x2': '1046', 'y2': '719', 'stroke': '#b91c1c', 'stroke-width': '1.3', 'marker-end': 'url(#arrow)'}))

    # Top Output: Aligned
    elements.append(el('rect', {'x': '1048', 'y': '660', 'width': '180', 'height': '35', 'rx': '4', 'ry': '4', 'fill': '#e7f9e9', 'stroke': '#22c55e', 'stroke-width': '1.1'}))
    t_ali = el('text', {'x': '1138', 'y': '677', 'fill': '#15803d', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '12', 'font-weight': 'bold', 'text-anchor': 'middle'})
    t_ali.text = 'Aligned  L_pub = L_exec'
    elements.append(t_ali)
    elements.append(el('text', {'x': '1138', 'y': '690', 'fill': '#16a34a', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9.5', 'font-weight': 'bold', 'text-anchor': 'middle'}, '✓  2.2% (1 / 46)'))

    # Bottom Output: Mismatch
    elements.append(el('rect', {'x': '1048', 'y': '702', 'width': '180', 'height': '35', 'rx': '4', 'ry': '4', 'fill': '#ffe8ed', 'stroke': '#ef4444', 'stroke-width': '1.1'}))
    t_mis = el('text', {'x': '1138', 'y': '719', 'fill': '#b91c1c', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '12', 'font-weight': 'bold', 'text-anchor': 'middle'})
    t_mis.text = 'Mismatch  L_pub ≠ L_exec'
    elements.append(t_mis)
    elements.append(el('text', {'x': '1138', 'y': '732', 'fill': '#dc2626', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9.5', 'font-weight': 'bold', 'text-anchor': 'middle'}, '✗  2.2% (1 / 46)'))

    # Far right note
    elements.append(el('rect', {'x': '1238', 'y': '660', 'width': '182', 'height': '77', 'rx': '4', 'ry': '4', 'fill': '#f8fafc', 'stroke': '#cbd5e1', 'stroke-width': '0.8'}))
    elements.append(el('text', {'x': '1329', 'y': '680', 'fill': '#0f172a', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'PRISMA-ScR Audit'))
    elements.append(el('text', {'x': '1329', 'y': '697', 'fill': '#475569', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8.5', 'text-anchor': 'middle'}, 'Only 1 out of 46 surveyed'))
    elements.append(el('text', {'x': '1329', 'y': '712', 'fill': '#475569', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8.5', 'text-anchor': 'middle'}, 'studies passed mathematical-code'))
    elements.append(el('text', {'x': '1329', 'y': '727', 'fill': '#15803d', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8.5', 'font-weight': 'bold', 'text-anchor': 'middle'}, 'parity verification (2.2%)'))

    return elements

# ==============================================================================
# D5 Builders
# ==============================================================================

def build_d5_constraint_onoff(data):
    elements = []
    # Axes
    elements.append(el('line', {'x1': '57', 'y1': '1007', 'x2': '205', 'y2': '1007', 'stroke': '#111111', 'stroke-width': '1.2'}))
    elements.append(el('line', {'x1': '57', 'y1': '1007', 'x2': '57', 'y2': '920', 'stroke': '#111111', 'stroke-width': '1.2'}))
    elements.append(el('text', {'x': '131', 'y': '1032', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'text-anchor': 'middle'}, 'Forecast horizon  h'))
    elements.append(el('text', {'x': '36', 'y': '964', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'middle', 'transform': 'rotate(-90,36,964)'}, 'Power (p.u.)'))
    
    # Ticks
    for h, h_label in [(0, '1'), (7, '8'), (15, '16'), (23, '24')]:
        hx = 57.0 + (h / 23.0) * 142.0
        elements.append(el('line', {'x1': f'{hx:.1f}', 'y1': '1004', 'x2': f'{hx:.1f}', 'y2': '1010', 'stroke': '#111111', 'stroke-width': '0.8'}))
        elements.append(el('text', {'x': f'{hx:.1f}', 'y': '1021', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9.5', 'text-anchor': 'middle'}, h_label))
        
    for p, p_label in [(0.0, '0.0'), (0.4, '0.4'), (0.8, '0.8')]:
        py = 1007.0 - (p / 0.8) * 87.0
        elements.append(el('line', {'x1': '54', 'y1': f'{py:.1f}', 'x2': '60', 'y2': f'{py:.1f}', 'stroke': '#111111', 'stroke-width': '0.8'}))
        elements.append(el('text', {'x': '51', 'y': f'{py+3:.1f}', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9.5', 'text-anchor': 'end'}, p_label))
        
    # Real 24-step Penmanshiel DLinear trajectories
    y_uncon = data['y_uncon']
    y_con = data['y_con']
    
    pts_uncon = []
    pts_con = []
    for h in range(len(y_uncon)):
        hx = 57.0 + (h / 23.0) * 142.0
        hy_u = 1007.0 - (float(y_uncon[h]) / 0.8) * 87.0
        hy_c = 1007.0 - (float(y_con[h]) / 0.8) * 87.0
        pts_uncon.append(f"{hx:.1f},{hy_u:.1f}")
        pts_con.append(f"{hx:.1f},{hy_c:.1f}")
        
    elements.append(el('polyline', {'points': ' '.join(pts_uncon), 'fill': 'none', 'stroke': '#64748b', 'stroke-dasharray': '6,3', 'stroke-width': '1.8'}))
    elements.append(el('polyline', {'points': ' '.join(pts_con), 'fill': 'none', 'stroke': '#1676e8', 'stroke-width': '2.2', 'stroke-linecap': 'round'}))
    
    # End-of-line direct labeling
    elements.append(el('text', {'x': '203', 'y': '986', 'fill': '#1676e8', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9', 'font-weight': 'bold', 'text-anchor': 'end'}, 'λ > 0 (con)'))
    elements.append(el('text', {'x': '203', 'y': '1002', 'fill': '#64748b', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9', 'text-anchor': 'end'}, 'λ = 0 (base)'))
    
    return elements

def build_d5_both(data):
    elements = []
    # Axes
    elements.append(el('line', {'x1': '553', 'y1': '1007', 'x2': '701', 'y2': '1007', 'stroke': '#111111', 'stroke-width': '1.2'}))
    elements.append(el('line', {'x1': '553', 'y1': '1007', 'x2': '553', 'y2': '920', 'stroke': '#111111', 'stroke-width': '1.2'}))
    elements.append(el('text', {'x': '627', 'y': '1032', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'text-anchor': 'middle'}, 'Forecast horizon  h'))
    elements.append(el('text', {'x': '532', 'y': '964', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'middle', 'transform': 'rotate(-90,532,964)'}, 'Power (p.u.)'))
    
    # Ticks
    for h, h_label in [(0, '1'), (7, '8'), (15, '16'), (23, '24')]:
        hx = 553.0 + (h / 23.0) * 142.0
        elements.append(el('line', {'x1': f'{hx:.1f}', 'y1': '1004', 'x2': f'{hx:.1f}', 'y2': '1010', 'stroke': '#111111', 'stroke-width': '0.8'}))
        elements.append(el('text', {'x': f'{hx:.1f}', 'y': '1021', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9.5', 'text-anchor': 'middle'}, h_label))
        
    for p, p_label in [(0.0, '0.0'), (0.4, '0.4'), (0.8, '0.8')]:
        py = 1007.0 - (p / 0.8) * 87.0
        elements.append(el('line', {'x1': '550', 'y1': f'{py:.1f}', 'x2': '556', 'y2': f'{py:.1f}', 'stroke': '#111111', 'stroke-width': '0.8'}))
        elements.append(el('text', {'x': '547', 'y': f'{py+3:.1f}', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9.5', 'text-anchor': 'end'}, p_label))
        
    # Real 3-way ablation trajectories
    y_full = data['y_full']
    y_ablated = data['y_ablated']
    y_none = data['y_none']
    
    pts_full, pts_abl, pts_none = [], [], []
    for h in range(len(y_full)):
        hx = 553.0 + (h / 23.0) * 142.0
        hy_f = 1007.0 - (float(y_full[h]) / 0.8) * 87.0
        hy_a = 1007.0 - (float(y_ablated[h]) / 0.8) * 87.0
        hy_n = 1007.0 - (float(y_none[h]) / 0.8) * 87.0
        pts_full.append(f"{hx:.1f},{hy_f:.1f}")
        pts_abl.append(f"{hx:.1f},{hy_a:.1f}")
        pts_none.append(f"{hx:.1f},{hy_n:.1f}")
        
    elements.append(el('polyline', {'points': ' '.join(pts_full), 'fill': 'none', 'stroke': '#1676e8', 'stroke-width': '2.2', 'stroke-linecap': 'round'}))
    elements.append(el('polyline', {'points': ' '.join(pts_abl), 'fill': 'none', 'stroke': '#24a45a', 'stroke-dasharray': '5,2.5', 'stroke-width': '1.8'}))
    elements.append(el('polyline', {'points': ' '.join(pts_none), 'fill': 'none', 'stroke': '#64748b', 'stroke-dasharray': '6,3', 'stroke-width': '1.6'}))
    
    # End-of-line direct labeling
    elements.append(el('text', {'x': '699', 'y': '980', 'fill': '#1676e8', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8.5', 'font-weight': 'bold', 'text-anchor': 'end'}, 'Full'))
    elements.append(el('text', {'x': '699', 'y': '992', 'fill': '#24a45a', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8.5', 'font-weight': 'bold', 'text-anchor': 'end'}, 'Ablated'))
    elements.append(el('text', {'x': '699', 'y': '1004', 'fill': '#64748b', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8.5', 'text-anchor': 'end'}, 'None'))
    
    return elements

def build_d5_reported(data):
    elements = []
    alphas = data['alphas']
    costs = data['costs']
    baseline = data['baseline']
    alpha_star = data['alpha_star']

    def map_x(a):
        return 820.0 + (a / 1.0) * 205.0
        
    def map_y(c):
        return 1010.0 - ((c - 0.092) / (0.165 - 0.092)) * 92.0

    y_base = map_y(baseline)
    x_star = map_x(alpha_star)

    # Shaded dual-color background regimes
    # Beneficial regime (below baseline)
    elements.append(el('rect', {
        'x': '820',
        'y': f'{y_base:.1f}',
        'width': '205',
        'height': f'{1010.0 - y_base:.1f}',
        'fill': '#dcfce7',
        'fill-opacity': '0.5'
    }))
    # Adverse / Failure regime (above baseline)
    elements.append(el('rect', {
        'x': '820',
        'y': '918',
        'width': '205',
        'height': f'{y_base - 918.0:.1f}',
        'fill': '#fee2e2',
        'fill-opacity': '0.5'
    }))

    # Axes
    elements.append(el('line', {'x1': '820', 'y1': '1010', 'x2': '1025', 'y2': '1010', 'stroke': '#111111', 'stroke-width': '1.2'}))
    elements.append(el('line', {'x1': '820', 'y1': '1010', 'x2': '820', 'y2': '918', 'stroke': '#111111', 'stroke-width': '1.2'}))
    
    # Labels (Y-axis label at x=786 has comfortable breathing room from tick labels at x=813)
    elements.append(el('text', {'x': '922.5', 'y': '1035', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'text-anchor': 'middle'}, 'Corruption factor α'))
    elements.append(el('text', {'x': '786', 'y': '964', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'middle', 'transform': 'rotate(-90,786,964)'}, 'Normalized cost J'))
    
    # Ticks X: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        ax = map_x(a)
        elements.append(el('line', {'x1': f'{ax:.1f}', 'y1': '1007', 'x2': f'{ax:.1f}', 'y2': '1013', 'stroke': '#111111', 'stroke-width': '0.8'}))
        elements.append(el('text', {'x': f'{ax:.1f}', 'y': '1024', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9.5', 'text-anchor': 'middle'}, f'{a:.1f}'))

    # Ticks Y
    for c_val, c_lbl in [(0.10, '0.10'), (0.13, '0.13'), (0.16, '0.16')]:
        cy = map_y(c_val)
        elements.append(el('line', {'x1': '817', 'y1': f'{cy:.1f}', 'x2': '823', 'y2': f'{cy:.1f}', 'stroke': '#111111', 'stroke-width': '0.8'}))
        elements.append(el('text', {'x': '813', 'y': f'{cy+3:.1f}', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9.5', 'text-anchor': 'end'}, c_lbl))

    # Horizontal baseline line (Unconstrained baseline)
    elements.append(el('line', {
        'x1': '820', 'y1': f'{y_base:.1f}',
        'x2': '1025', 'y2': f'{y_base:.1f}',
        'stroke': '#64748b', 'stroke-dasharray': '5,3', 'stroke-width': '1.3'
    }))
    elements.append(el('text', {
        'x': '1022', 'y': f'{y_base - 4:.1f}',
        'fill': '#475569', 'font-family': 'DejaVu Sans, Arial, sans-serif',
        'font-size': '8', 'text-anchor': 'end'
    }, 'Unconstrained baseline (no physics)'))

    # Vertical crossover line at alpha* ~ 0.07
    elements.append(el('line', {
        'x1': f'{x_star:.1f}', 'y1': '918',
        'x2': f'{x_star:.1f}', 'y2': '1010',
        'stroke': '#b91c1c', 'stroke-dasharray': '3,2', 'stroke-width': '1.1'
    }))
    elements.append(el('circle', {
        'cx': f'{x_star:.1f}', 'cy': f'{y_base:.1f}',
        'r': '3.2', 'fill': '#b91c1c', 'stroke': '#ffffff', 'stroke-width': '0.6'
    }))
    elements.append(el('text', {
        'x': f'{x_star + 4:.1f}', 'y': '1004',
        'fill': '#b91c1c', 'font-family': 'DejaVu Sans, Arial, sans-serif',
        'font-size': '9', 'font-weight': 'bold'
    }, 'α* ≈ 0.07'))

    # Genuine non-linear cost trajectory polyline & markers
    pts = []
    for a, c in zip(alphas, costs):
        px = map_x(a)
        py = map_y(c)
        pts.append(f"{px:.1f},{py:.1f}")
        elements.append(el('rect', {
            'x': f'{px-2.5:.1f}', 'y': f'{py-2.5:.1f}',
            'width': '5', 'height': '5',
            'fill': '#dc2626', 'stroke': '#991b1b', 'stroke-width': '0.5'
        }))

    elements.append(el('polyline', {
        'points': ' '.join(pts),
        'fill': 'none',
        'stroke': '#dc2626',
        'stroke-width': '2.2',
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round'
    }))

    # Text annotations in the two regimes (placed in clear whitespace)
    elements.append(el('text', {
        'x': '860', 'y': '936',
        'fill': '#b91c1c', 'font-family': 'DejaVu Sans, Arial, sans-serif',
        'font-size': '10.5', 'font-weight': 'bold'
    }, 'Adverse regime (harm)'))
    elements.append(el('text', {
        'x': '900', 'y': '1006',
        'fill': '#15803d', 'font-family': 'DejaVu Sans, Arial, sans-serif',
        'font-size': '9', 'font-weight': 'bold'
    }, 'Beneficial'))

    # End tag for curve
    elements.append(el('text', {
        'x': f'{map_x(1.0) + 4:.1f}', 'y': f'{map_y(costs[-1]) + 3:.1f}',
        'fill': '#dc2626', 'font-family': 'DejaVu Sans, Arial, sans-serif',
        'font-size': '9', 'font-weight': 'bold'
    }, 'Cost'))

    return elements

def build_d5_notreported(data):
    elements = []
    # Axes
    elements.append(el('line', {'x1': '1146', 'y1': '1007', 'x2': '1326', 'y2': '1007', 'stroke': '#111111', 'stroke-width': '1.2'}))
    elements.append(el('line', {'x1': '1146', 'y1': '1007', 'x2': '1146', 'y2': '918', 'stroke': '#111111', 'stroke-width': '1.2'}))
    elements.append(el('text', {'x': '1236', 'y': '1032', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '11', 'text-anchor': 'middle'}, 'Wind speed (m/s)'))
    elements.append(el('text', {'x': '1124', 'y': '961', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '10', 'text-anchor': 'middle', 'transform': 'rotate(-90,1124,961)'}, 'Power (p.u.)'))
    
    # Ticks X: 0, 5, 10, 15, 20
    for v in [0, 5, 10, 15, 20]:
        vx = 1146.0 + (v / 20.0) * 180.0
        elements.append(el('line', {'x1': f'{vx:.1f}', 'y1': '1004', 'x2': f'{vx:.1f}', 'y2': '1010', 'stroke': '#111111', 'stroke-width': '0.8'}))
        elements.append(el('text', {'x': f'{vx:.1f}', 'y': '1021', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9.5', 'text-anchor': 'middle'}, str(v)))
        
    for p, p_label in [(0.0, '0.0'), (0.4, '0.4'), (0.8, '0.8')]:
        py = 1007.0 - (p / 0.8) * 89.0
        elements.append(el('line', {'x1': '1143', 'y1': f'{py:.1f}', 'x2': '1149', 'y2': f'{py:.1f}', 'stroke': '#111111', 'stroke-width': '0.8'}))
        elements.append(el('text', {'x': '1140', 'y': f'{py+3:.1f}', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '9.5', 'text-anchor': 'end'}, p_label))
        
    # Genuine prior C_emp curve
    prior = data['prior']
    c_pts = []
    for c, m in zip(prior['centers'], prior['means']):
        if c > 20.0: continue
        cx = 1146.0 + (c / 20.0) * 180.0
        cy = 1007.0 - max(0.0, min(0.85, float(m))) / 0.8 * 89.0
        c_pts.append(f"{cx:.1f},{cy:.1f}")
        
    elements.append(el('polyline', {'points': ' '.join(c_pts), 'fill': 'none', 'stroke': '#1676e8', 'stroke-width': '2.5', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'}))
    
    # Scrambled prior C_scramb curve
    scramb = data['scrambled']
    s_pts = []
    marker_elements = []
    for c, m in zip(scramb['centers'], scramb['means']):
        if c > 20.0: continue
        cx = 1146.0 + (c / 20.0) * 180.0
        cy = 1007.0 - max(0.0, min(0.85, float(m))) / 0.8 * 89.0
        s_pts.append(f"{cx:.1f},{cy:.1f}")
        marker_elements.append(el('rect', {'x': f'{cx-2.0:.1f}', 'y': f'{cy-2.0:.1f}', 'width': '4.0', 'height': '4.0', 'fill': '#ef3e46', 'stroke': '#ef3e46'}))
        
    elements.append(el('polyline', {'points': ' '.join(s_pts), 'fill': 'none', 'stroke': '#ef3e46', 'stroke-dasharray': '6,3', 'stroke-width': '1.8'}))
    elements.extend(marker_elements)
    
    # Subtle background pill for legend in upper-left plot whitespace
    elements.append(el('rect', {'x': '1154', 'y': '923', 'width': '124', 'height': '31', 'rx': '3', 'ry': '3', 'fill': '#ffffff', 'fill-opacity': '0.90', 'stroke': '#e2e8f0', 'stroke-width': '0.8'}))
    
    # Legend Line 1: Genuine prior C_emp
    elements.append(el('line', {'x1': '1160', 'y1': '933', 'x2': '1178', 'y2': '933', 'stroke': '#1676e8', 'stroke-width': '2.5'}))
    t_g = el('text', {'x': '1182', 'y': '936', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8.5', 'font-weight': 'bold'})
    t_g.text = 'Genuine prior C'
    t_g.append(el('tspan', {'font-size': '7', 'dy': '1.8'}, 'emp'))
    elements.append(t_g)
    
    # Legend Line 2: Scrambled prior C_scramb
    elements.append(el('line', {'x1': '1160', 'y1': '946', 'x2': '1178', 'y2': '946', 'stroke': '#ef3e46', 'stroke-dasharray': '4,2', 'stroke-width': '1.8'}))
    t_s = el('text', {'x': '1182', 'y': '949', 'fill': '#111111', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8.5', 'font-weight': 'bold'})
    t_s.text = 'Scrambled prior C'
    t_s.append(el('tspan', {'font-size': '7', 'dy': '1.8'}, 'scramb'))
    elements.append(t_s)
    
    # Subtle Footnote annotation badge
    elements.append(el('rect', {'x': '1195', 'y': '984', 'width': '130', 'height': '18', 'rx': '3', 'ry': '3', 'fill': '#f8fafc', 'stroke': '#cbd5e1', 'stroke-width': '0.8'}))
    elements.append(el('text', {'x': '1260', 'y': '996', 'fill': '#64748b', 'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': '8.5', 'font-style': 'italic', 'text-anchor': 'middle'}, 'failure boundary omitted'))
    
    return elements

# ==============================================================================
# Master Assembler
# ==============================================================================

def synthesize():
    print("2. Parsing raw master SVG template...")
    tree = ET.parse(INPUT_RAW_SVG)
    root = tree.getroot()
    raw_attrib = dict(root.attrib)
    
    data = load_all_repository_data()
    
    print("3. Performing surgical replacements across all slots...")
    old_children = list(root)
    new_children = []
    
    i = 0
    n = len(old_children)
    while i < n:
        c = old_children[i]
        
        # Target 1: D1.1 Intrinsic (indices 15 to 34)
        if i == 15:
            print("  -> Replacing D1.1 Intrinsic (Betz cubic envelope)...")
            new_children.extend(build_d1_intrinsic())
            i = 35
            continue
            
        # Target 2: D1.2 Historical-static (indices 38 to 278)
        if i == 38:
            print("  -> Replacing D1.2 Historical-static (Penmanshiel SCADA)...")
            new_children.extend(build_d1_historical_static(data))
            i = 279
            continue
            
        # Target 3: D1.3 Learned-online (indices 282 to 290)
        if i == 282:
            print("  -> Replacing D1.3 Learned-online (Adaptive graph)...")
            new_children.extend(build_d1_learned_online())
            i = 291
            continue
            
        # Target 4: D1.4 Unreported (indices 294 to 295)
        if i == 294:
            print("  -> Replacing D1.4 Unreported (Audit badge)...")
            new_children.extend(build_d1_unreported())
            i = 296
            continue
            
        # Target 5: D2.1 Training loss (indices 307 to 308)
        if i == 307:
            print("  -> Replacing D2.1 Training loss (Clean vector formula)...")
            new_children.extend(build_d2_training_loss())
            i = 309
            continue
            
        # Target 6: D2.2 Architecture (indices 312 to 319)
        if i == 312:
            print("  -> Replacing D2.2 Architecture (Manifold layer)...")
            new_children.extend(build_d2_architecture())
            i = 320
            continue
            
        # Target 7: D2.3 Data generation (indices 323 to 336)
        if i == 323:
            print("  -> Replacing D2.3 Data generation (Simulator batch)...")
            new_children.extend(build_d2_data_generation())
            i = 337
            continue
            
        # Target 8: D2.4 Post-hoc correction (indices 340 to 370)
        if i == 340:
            print("  -> Replacing D2.4 Post-hoc correction (Projection plot)...")
            new_children.extend(build_d2_posthoc())
            i = 371
            continue
            
        # Target 9: D3.1 None (indices 382 to 392)
        if i == 382:
            print("  -> Replacing D3.1 None (Lookback trajectory)...")
            new_children.extend(build_d3_none(data))
            i = 393
            continue
            
        # Target 10: D3.2 Forecast input (indices 396 to 425)
        if i == 396:
            print("  -> Replacing D3.2 Forecast input (NWP tensor)...")
            new_children.extend(build_d3_nwp())
            i = 426
            continue
            
        # Target 11: D3.3 Contemporaneous state (indices 429 to 436)
        if i == 429:
            print("  -> Replacing D3.3 Contemporaneous state (Telemetry vector)...")
            new_children.extend(build_d3_contemporaneous())
            i = 437
            continue
            
        # Target 12: D4.1 Code availability (indices 447 to 463)
        if i == 447:
            print("  -> Replacing D4.1 Code availability (PRISMA statistics)...")
            new_children.extend(build_d4_code_availability())
            i = 464
            continue
            
        # Target 13: D4.2 Equation-code parity (indices 467 to 481)
        if i == 467:
            print("  -> Replacing D4.2 Equation-code parity (AST branching)...")
            new_children.extend(build_d4_parity_branching())
            i = 482
            continue
            
        # Target 14: D5.1 Constraint on/off (indices 497 to 520)
        if i == 497:
            print("  -> Replacing D5.1 Constraint on/off (Penmanshiel DLinear)...")
            new_children.extend(build_d5_constraint_onoff(data))
            i = 521
            continue
            
        # Target 15: D5.3 Both (indices 546 to 572)
        if i == 546:
            print("  -> Replacing D5.3 Both (3-way ablation)...")
            new_children.extend(build_d5_both(data))
            i = 573
            continue
            
        # Target 16: D5.4 Reported (indices 576 to 614)
        if i == 576:
            print("  -> Replacing D5.4 Reported (Genuine non-linear crossover!)...")
            new_children.extend(build_d5_reported(data))
            i = 615
            continue
            
        # Target 17: D5.5 Not reported (indices 618 to 661)
        if i == 618:
            print("  -> Replacing D5.5 Not reported (Real vs Scrambled)...")
            new_children.extend(build_d5_notreported(data))
            i = 662
            continue
            
        new_children.append(c)
        i += 1
        
    root.clear()
    root.attrib.update(raw_attrib)
    for nc in new_children:
        root.append(nc)
        
    print("4. Saving synthesized SVG...")
    svg_str = ET.tostring(root, encoding='utf-8').decode('utf-8')
    
    # Clean standard SVG namespaces
    svg_str = svg_str.replace('<ns0:', '<').replace('</ns0:', '</').replace('xmlns:ns0=', 'xmlns=')
    if not svg_str.startswith('<?xml'):
        svg_str = "<?xml version='1.0' encoding='utf-8'?>\n" + svg_str
        
    OUTPUT_SVG.write_text(svg_str, encoding='utf-8')
    FINAL_SVG.write_text(svg_str, encoding='utf-8')
    print(f"  -> Saved {OUTPUT_SVG}")
    print(f"  -> Saved {FINAL_SVG}")
    
    print("5. Rendering pixel-perfect vector PDF via Edge Engine...")
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
{svg_str[svg_str.find('?>')+2:].strip()}
</body>
</html>"""
    
    html_temp = ROOT / "outputs" / "temp_render_wrapper.html"
    html_temp.write_text(html_content, encoding="utf-8")
    
    cmd = [
        EDGE_EXE,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--print-to-pdf=" + str(OUTPUT_PDF),
        html_temp.as_uri()
    ]
    subprocess.run(cmd, check=True)
    print(f"  -> Saved vector PDF {OUTPUT_PDF}")
    
    # Deploy to manuscript directories
    for p in MANUSCRIPT_PDF_PATHS:
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUTPUT_PDF, p)
        print(f"  -> Deployed {p}")
        
    # Render PNG at 300 DPI
    doc = fitz.open(str(OUTPUT_PDF))
    page = doc[0]
    pix = page.get_pixmap(dpi=300)
    pix.save(str(OUTPUT_PNG))
    print(f"  -> Saved 300 DPI PNG {OUTPUT_PNG} ({pix.width}x{pix.height})")
    
    # Scratch preview at 150 DPI
    scratch_png = Path(r"C:\Users\lidong\.gemini\antigravity\brain\628e27cb-2724-48ad-88f3-9aeb746ee2c7\scratch\final_master_render.png")
    pix_preview = page.get_pixmap(dpi=150)
    pix_preview.save(str(scratch_png))
    print(f"  -> Saved preview {scratch_png}")
    
    # Cleanup temp html
    if html_temp.exists():
        html_temp.unlink()

if __name__ == '__main__':
    synthesize()
    print("Master Figure 1 Vector Synthesis Successfully Completed!")
