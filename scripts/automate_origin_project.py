#!/usr/bin/env python3
"""
Automated OriginLab Project Generator & Exporter for Ultra-LSNT.
Strictly adheres to Applied Energy / Nature publication graphic standards:
- Native OriginLab COM API automation (originpro)
- Pure Arial font across all graph titles, axes, tick labels, and legends
- High-contrast Okabe-Ito / Blues publication palettes
- Proper categorical column chart for Kernel Size ablation (k=3, k=7, k=15)
- Greek symbols (\\g(l) for lambda, \\g(s)\\-(eff) for sigma_eff, R\\+(2) for R^2)
- Zero clipped legends: all legends placed strictly inside plot boundary in whitespace
- Vector / 600 DPI export of publication-grade figures
"""

import originpro as op
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def build_origin_project():
    project_dir = Path('03_图表与绘图素材/origin_figures').resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    op_file = project_dir / 'Ultra_LSNT_Figures.opju'

    print(f'Connecting to OriginLab and opening project: {op_file}')
    op.open(str(op_file))

    # Determine dynamic TrueType Arial font ID
    arial_id = op.lt_int('font("Arial")')
    if arial_id <= 0:
        arial_id = 68
    print(f'Detected Origin TrueType Arial Font ID: {arial_id}')

    def apply_publication_style(gp, title_x, title_y, x_from=None, x_to=None, y_from=None, y_to=None):
        gp.activate()
        op.lt_exec(f'win -a "{gp.name}";')
        
        # Set TrueType Arial font across page and tick labels
        op.lt_exec(f'page.font = {arial_id};')
        op.lt_exec(f'layer.x.label.font = {arial_id};')
        op.lt_exec(f'layer.y.label.font = {arial_id};')
        op.lt_exec('layer.x.label.pts = 13;')
        op.lt_exec('layer.y.label.pts = 13;')
        
        # Box frame with inward ticks and balanced 1.5 pt line thickness
        op.lt_exec('layer.x2.showAxes = 1; layer.y2.showAxes = 1;')
        op.lt_exec('layer.x2.ticks = 0; layer.y2.ticks = 0;')
        op.lt_exec('layer.x.thickness = 1.5; layer.y.thickness = 1.5;')
        op.lt_exec('layer.x2.thickness = 1.5; layer.y2.thickness = 1.5;')
        
        # Set Axis limits if provided
        if x_from is not None and x_to is not None:
            op.lt_exec(f'layer.x.from = {x_from}; layer.x.to = {x_to};')
        if y_from is not None and y_to is not None:
            op.lt_exec(f'layer.y.from = {y_from}; layer.y.to = {y_to};')
            
        # Set Titles with explicit Arial Bold tags
        op.lt_exec(f'label -xb "\\f:Arial(\\b({title_x}))";')
        op.lt_exec(f'label -yl "\\f:Arial(\\b({title_y}))";')

    # ==================== 1. Graph: Kernel Size Sensitivity (Categorical Column) ====================
    wb_kernel = op.find_book('w', 'UserslidongAp')
    if wb_kernel:
        ws = wb_kernel[0]
        old_gp = op.find_graph('Graph_Ablation_Kernel')
        if old_gp:
            old_gp.destroy()

        gp = op.new_graph(template='column')
        gp.name = 'Graph_Ablation_Kernel'
        gl = gp[0]
        p = gl.add_plot(ws, coly='R2_Score', colx='Kernel_Type', type='c')
        p.color = '#0072B2'
        gl.rescale()
        
        apply_publication_style(
            gp,
            title_x='Temporal Embedding Kernel Size',
            title_y='Forecasting Accuracy (R\\+(2) Score)',
            y_from=0.80, y_to=0.88
        )
        # Explicitly remove redundant single-column legend
        gl.remove_label('Legend')
        
        out_png = project_dir / 'Origin_Ablation_Kernel.png'
        gp.save_fig(str(out_png), width=1800)
        print('Exported publication-grade Graph_Ablation_Kernel')

    # ==================== 2. Graph: Router Regularization (Ablation Lambda) ====================
    wb_lambda = op.find_book('w', 'GraphLambdaAb')
    if wb_lambda:
        ws = wb_lambda[0]
        old_gp = op.find_graph('Graph_Ablation_Lambda')
        if old_gp:
            old_gp.destroy()

        gp = op.new_graph(template='scatter')
        gp.name = 'Graph_Ablation_Lambda'
        gl = gp[0]
        p = gl.add_plot(ws, coly='R2_Score', colx='Lambda', type='linesymbol')
        p.color = '#0072B2'
        gl.rescale()
        
        apply_publication_style(
            gp,
            title_x='Router Auxiliary Regularization \\g(l)',
            title_y='Forecasting Accuracy (R\\+(2) Score)',
            x_from=-0.01, x_to=0.11,
            y_from=0.865, y_to=0.890
        )
        # Set line thickness and symbol style
        op.lt_exec('set %C -w 2200; set %C -z 9; set %C -k 2;')
        
        # Configure clean legend positioned inside top-left whitespace in axis units
        op.lt_exec('legendupdate;')
        lbl = gl.label('Legend')
        if lbl:
            lbl.text = '\\f:Arial(\\l(1) Validation R\\+(2))'
            lbl.set_float('x', 0.025)
            lbl.set_float('y', 0.887)
        
        out_png = project_dir / 'Origin_Ablation_Lambda.png'
        gp.save_fig(str(out_png), width=1800)
        print('Exported publication-grade Graph_Ablation_Lambda')

    # ==================== 3. Graph: Diurnal Expert Routing (24-Hour Cycle) ====================
    wb_diurnal = op.find_book('w', 'GraphDiurnalD')
    if wb_diurnal:
        ws = wb_diurnal[0]
        old_gp = op.find_graph('Graph_Diurnal_Routing')
        if old_gp:
            old_gp.destroy()

        gp = op.new_graph(template='line')
        gp.name = 'Graph_Diurnal_Routing'
        gl = gp[0]
        colors = ['#0072B2', '#D55E00', '#009E73', '#E69F00']
        cols = ['Expert_1_WindSpeed', 'Expert_2_Turbulence', 'Expert_3_Temperature', 'Expert_4_Humidity']
        for i, col in enumerate(cols):
            p = gl.add_plot(ws, coly=col, colx='Hour', type='l')
            p.color = colors[i]
            
        gl.rescale()
        apply_publication_style(
            gp,
            title_x='Hour of Day (UTC)',
            title_y='Mean Expert Routing Weight',
            x_from=0, x_to=23,
            y_from=0.10, y_to=0.50
        )
        # Set line widths
        op.lt_exec('for(int ii=1; ii<=4; ii++) { set %C(ii) -w 2200; };')
        
        # Legend placed in open upper-left region (centered at X=7.0, Y=0.470)
        op.lt_exec('legendupdate;')
        lbl = gl.label('Legend')
        if lbl:
            lbl.text = '\\f:Arial(\\l(1) E1: Wind Speed)\r\n\\f:Arial(\\l(2) E2: Turbulence)\r\n\\f:Arial(\\l(3) E3: Temperature)\r\n\\f:Arial(\\l(4) E4: Humidity)'
            lbl.set_float('x', 7.0)
            lbl.set_float('y', 0.470)
        
        out_png = project_dir / 'Origin_Diurnal_Routing.png'
        gp.save_fig(str(out_png), width=1800)
        print('Exported publication-grade Graph_Diurnal_Routing')

    # ==================== 4. Graph: Sensor Noise Robustness ====================
    wb_robust = op.find_book('w', 'GraphRobustne')
    if wb_robust:
        ws = wb_robust[0]
        old_gp = op.find_graph('Graph_Noise_Robustness')
        if old_gp:
            old_gp.destroy()

        gp = op.new_graph(template='line')
        gp.name = 'Graph_Noise_Robustness'
        gl = gp[0]
        models = [
            ('Ultra_LSNT', '#0072B2', 2600, 2),  # Solid circle, thick line
            ('LightGBM', '#E69F00', 1800, 3),    # Square
            ('DLinear', '#009E73', 1800, 4),     # Triangle
            ('PatchTST', '#56B4E9', 1800, 5),    # Diamond
            ('TimeMixer', '#64748B', 1800, 6),   # Inverted triangle
        ]
        for mod, col, lw, sym in models:
            p = gl.add_plot(ws, coly=mod, colx='Sigma_eff', type='linesymbol')
            p.color = col
            
        gl.rescale()
        apply_publication_style(
            gp,
            title_x='Effective Noise Severity \\g(s)\\-(eff)',
            title_y='Forecasting Accuracy (R\\+(2) Score)',
            x_from=0.0, x_to=0.62,
            y_from=0.48, y_to=0.92
        )
        # Set line widths and symbol styles
        for idx, (_, _, lw, sym) in enumerate(models, start=1):
            op.lt_exec(f'set %C({idx}) -w {lw}; set %C({idx}) -z 8; set %C({idx}) -k {sym};')
        
        # Legend placed in bottom-left whitespace (centered at X=0.13, Y=0.58)
        op.lt_exec('legendupdate;')
        op.lt_exec('legend.pts = 9;')
        lbl = gl.label('Legend')
        if lbl:
            lbl.text = '\\f:Arial(\\l(1) Ultra-LSNT (Ours))\r\n\\f:Arial(\\l(2) LightGBM)\r\n\\f:Arial(\\l(3) DLinear)\r\n\\f:Arial(\\l(4) PatchTST)\r\n\\f:Arial(\\l(5) TimeMixer)'
            lbl.set_float('x', 0.13)
            lbl.set_float('y', 0.58)
        
        out_png = project_dir / 'Origin_Noise_Robustness.png'
        gp.save_fig(str(out_png), width=1800)
        print('Exported publication-grade Graph_Noise_Robustness')

    # ==================== 5. Graph: Time-Resolved Power Forecast ====================
    wb_power = op.find_book('w', 'GraphPowerFor')
    if wb_power:
        ws = wb_power[0]
        old_gp = op.find_graph('Graph_Power_Forecast')
        if old_gp:
            old_gp.destroy()

        gp = op.new_graph(template='line')
        gp.name = 'Graph_Power_Forecast'
        gl = gp[0]
        p1 = gl.add_plot(ws, coly='Realized_Power_kW', colx='Step', type='l')
        p1.color = '#0F172A'
        p2 = gl.add_plot(ws, coly='Predicted_Power_kW', colx='Step', type='l')
        p2.color = '#0072B2'
        
        gl.rescale()
        apply_publication_style(
            gp,
            title_x='Observation Step Index (15-min intervals, 64 hours)',
            title_y='Active Wind Power (kW)',
            x_from=0, x_to=256,
            y_from=8000, y_to=78000
        )
        # Line properties
        op.lt_exec('set %C(1) -w 1800;')
        op.lt_exec('set %C(2) -w 2200; set %C(2) -d 2;') # dashed line for prediction
        
        # Legend placed in spacious valley region (centered at X=135, Y=72000)
        op.lt_exec('legendupdate;')
        lbl = gl.label('Legend')
        if lbl:
            lbl.text = '\\f:Arial(\\l(1) Realized Power)\r\n\\f:Arial(\\l(2) Ultra-LSNT Forecast)'
            lbl.set_float('x', 135.0)
            lbl.set_float('y', 72000.0)
        
        out_png = project_dir / 'Origin_Power_Forecast.png'
        gp.save_fig(str(out_png), width=1800)
        print('Exported publication-grade Graph_Power_Forecast')

    # Save origin project
    op.save()
    print('Origin Project successfully updated and saved to:', op_file)
    op.exit()

if __name__ == '__main__':
    build_origin_project()
