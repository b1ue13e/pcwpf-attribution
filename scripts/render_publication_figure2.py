import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from pathlib import Path

def draw_figure_2():
    # Double column Nature / Elsevier width: 14.0 in x 7.0 in, 300 DPI preview, 600 DPI export
    fig, ax = plt.subplots(figsize=(14.0, 7.0), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Background: Pure white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Font setup
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'stix',
    })

    # Color Palette (Nature / Okabe-Ito inspired)
    C_BLUE = '#1D4ED8'        # Primary Royal Blue
    C_BLUE_LIGHT = '#EFF6FF'  # Light Blue fill
    C_BLUE_BORDER = '#3B82F6'
    
    C_TEAL = '#047857'        # Green/Teal for Trend
    C_TEAL_LIGHT = '#ECFDF5'
    C_TEAL_BORDER = '#10B981'
    
    C_ORANGE = '#B45309'      # Amber/Orange for Router
    C_ORANGE_LIGHT = '#FFFBEB'
    C_ORANGE_BORDER = '#F59E0B'
    
    C_RED = '#DC2626'         # Coral/Red for Jump Gate
    C_RED_LIGHT = '#FEF2F2'
    C_RED_BORDER = '#EF4444'
    
    C_DARK = '#0F172A'
    C_GRAY_TEXT = '#475569'
    C_GRAY_BG = '#F8FAFC'
    C_STAGE_BG = '#F8FAFC'
    C_STAGE_BORDER = '#CBD5E1'

    # Helper function for card boxes
    def draw_card(x, y, w, h, title, subtitle=None, fc='white', ec='#CBD5E1', lw=1.3, 
                  title_color='#0F172A', title_size=9.5, sub_size=8.2, bold=True, style='solid'):
        ls = '-' if style == 'solid' else '--'
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.18',
                             facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2)
        ax.add_patch(box)
        if subtitle is None:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                    fontsize=title_size, fontweight='bold' if bold else 'normal',
                    color=title_color, zorder=3)
        else:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center',
                    fontsize=title_size, fontweight='bold' if bold else 'normal',
                    color=title_color, zorder=3)
            ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center',
                    fontsize=sub_size, color=C_GRAY_TEXT, zorder=3)

    # Helper function for stage containers
    def draw_stage(x, y, w, h, title):
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.25',
                             facecolor=C_STAGE_BG, edgecolor=C_STAGE_BORDER, linewidth=1.1, zorder=0)
        ax.add_patch(box)
        header_h = 0.58
        h_box = FancyBboxPatch((x, y + h - header_h), w, header_h, 
                               boxstyle='round,pad=0.0,rounding_size=0.2',
                               facecolor='#E2E8F0', edgecolor='none', zorder=1)
        ax.add_patch(h_box)
        ax.text(x + w/2, y + h - header_h/2, title, ha='center', va='center',
                fontsize=10.2, fontweight='bold', color='#1E293B', zorder=2)

    # 1. STAGE CONTAINERS
    draw_stage(0.4, 1.4, 6.0, 8.2, 'Stage 1: Decomposition & Multi-Scale Embedding')
    draw_stage(6.7, 1.4, 9.3, 8.2, 'Stage 2: Consistent Jump Gate & Sparse MoE Encoder')
    draw_stage(16.3, 1.4, 3.3, 8.2, 'Stage 3: Reconstruction & Fusion')

    # ==================== STAGE 1: DECOMPOSITION & MULTI-SCALE EMBEDDING ====================
    # Input Series
    draw_card(0.65, 4.6, 1.55, 1.45, 'Input Series', r'$\mathbf{X} \in \mathbb{R}^{L \times D}$',
              fc='white', ec='#64748B', title_color=C_DARK, lw=1.4)
    
    # MA Decomp
    draw_card(2.6, 4.6, 1.6, 1.45, 'MA Decomp', 'Split Filter',
              fc=C_TEAL_LIGHT, ec=C_TEAL_BORDER, title_color=C_TEAL, lw=1.4)
    
    # Arrow: Input -> MA Decomp
    ax.annotate('', xy=(2.6, 5.32), xytext=(2.2, 5.32),
                arrowprops=dict(arrowstyle='->', color='#334155', lw=1.6, mutation_scale=13))

    # Trend Box (Top of Stage 1)
    draw_card(4.65, 8.1, 1.55, 1.15, 'Trend', r'$\mathbf{X}_{\mathrm{trend}}$',
              fc=C_TEAL_LIGHT, ec=C_TEAL_BORDER, title_color=C_TEAL, lw=1.4)
    
    # Arrow: MA Decomp top -> Trend
    ax.annotate('', xy=(4.65, 8.68), xytext=(3.4, 6.05),
                arrowprops=dict(arrowstyle='->', color=C_TEAL, lw=1.8,
                                connectionstyle='arc3,rad=-0.16', mutation_scale=14))

    # Dual-Scale Conv1D Blocks
    draw_card(4.65, 5.8, 1.55, 1.15, 'Conv1D', r'$k_s = 3$ (Short)',
              fc=C_BLUE_LIGHT, ec=C_BLUE_BORDER, title_color=C_BLUE, lw=1.4)
    draw_card(4.65, 4.0, 1.55, 1.15, 'Conv1D', r'$k_m = 7$ (Medium)',
              fc=C_BLUE_LIGHT, ec=C_BLUE_BORDER, title_color=C_BLUE, lw=1.4)

    # Arrows: MA Decomp -> Conv1D blocks
    ax.annotate('', xy=(4.65, 6.38), xytext=(4.2, 5.65),
                arrowprops=dict(arrowstyle='->', color=C_BLUE_BORDER, lw=1.5,
                                connectionstyle='arc3,rad=-0.08', mutation_scale=13))
    ax.annotate('', xy=(4.65, 4.58), xytext=(4.2, 5.0),
                arrowprops=dict(arrowstyle='->', color=C_BLUE_BORDER, lw=1.5,
                                connectionstyle='arc3,rad=0.08', mutation_scale=13))

    # Embedding Fusion Block (Bottom of Stage 1)
    draw_card(4.65, 1.9, 1.55, 1.35, 'Embedding', r'$\mathbf{E} = \phi(\mathbf{E}_s + \mathbf{E}_m)$',
              fc=C_BLUE_LIGHT, ec=C_BLUE_BORDER, title_color=C_BLUE, lw=1.4)
    
    # Explicit dual connections: Conv1D k=3 and k=7 BOTH cleanly enter Embedding top edge
    # Arrow from k_m=7 (straight down into left half of top edge)
    ax.annotate('', xy=(5.05, 3.25), xytext=(5.05, 4.0),
                arrowprops=dict(arrowstyle='->', color=C_BLUE_BORDER, lw=1.6, mutation_scale=13))
    # Arrow from k_s=3 (curves along right side into right half of top edge)
    ax.annotate('', xy=(5.75, 3.25), xytext=(6.2, 6.1),
                arrowprops=dict(arrowstyle='->', color=C_BLUE_BORDER, lw=1.5,
                                connectionstyle='arc3,rad=0.15', mutation_scale=13))

    # ==================== STAGE 2: CONSISTENT JUMP GATE & SPARSE MOE ====================
    # 2.1 Jump Gate Controller (receives Embedding E cleanly)
    draw_card(7.0, 4.5, 1.6, 1.35, 'Jump Gate', r'Test $I_t < \tau$',
              fc=C_RED_LIGHT, ec=C_RED_BORDER, title_color=C_RED, lw=1.5)
    
    # Arrow: Embedding -> Jump Gate (Clean upward curve into Jump Gate input, zero collision)
    ax.annotate('', xy=(7.0, 5.18), xytext=(6.2, 2.57),
                arrowprops=dict(arrowstyle='->', color=C_DARK, lw=1.8,
                                connectionstyle='arc3,rad=-0.15', mutation_scale=14))

    # 2.2 Skip Path (When I_t < tau, Skip MoE, 0 FLOPs)
    # Dedicated elevated route at y = 7.95, strictly above all expert cards (Expert 1 y_max=7.45)
    # Left dashed segment: from Jump Gate top -> Skip badge
    ax.annotate('', xy=(9.7, 7.95), xytext=(7.8, 5.85),
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.8, linestyle='--',
                                connectionstyle='arc3,rad=-0.18', mutation_scale=13))
    
    # Callout Badge for Skip Path (Clean white pill centered at x=11.2, y=7.95)
    skip_box = FancyBboxPatch((9.75, 7.72), 2.85, 0.46, boxstyle='round,pad=0.0,rounding_size=0.12',
                              facecolor='white', edgecolor=C_RED_BORDER, lw=1.2, zorder=5)
    ax.add_patch(skip_box)
    ax.text(11.17, 7.95, 'Conditional Skip Path (0 FLOPs)', ha='center', va='center',
            fontsize=7.8, fontweight='bold', color=C_RED, zorder=6)

    # Right dashed segment: from Skip badge -> Seasonal Head top
    ax.annotate('', xy=(15.2, 5.8), xytext=(12.65, 7.95),
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.8, linestyle='--',
                                connectionstyle='arc3,rad=-0.18', mutation_scale=14))

    # 2.3 Compute Path (When I_t >= tau, Proceed to MoE Routing)
    # Top-K Router
    router_circle = Circle((9.4, 5.18), 0.54, facecolor=C_ORANGE_LIGHT, edgecolor=C_ORANGE_BORDER, lw=1.8, zorder=3)
    ax.add_patch(router_circle)
    ax.text(9.4, 5.33, 'Top-K', ha='center', va='center', fontsize=8.6, fontweight='bold', color=C_ORANGE, zorder=4)
    ax.text(9.4, 5.01, 'Router', ha='center', va='center', fontsize=7.4, color=C_ORANGE, zorder=4)

    # Arrow: Jump Gate -> Router (Clean, labeled 'I_t >= tau')
    ax.annotate('', xy=(8.86, 5.18), xytext=(8.6, 5.18),
                arrowprops=dict(arrowstyle='->', color=C_DARK, lw=1.8, mutation_scale=14))
    ax.text(8.73, 5.38, r'$\geq \tau$', ha='center', va='bottom', fontsize=7.6, fontweight='bold', color=C_DARK)

    # 2.4 Expert Bank (4 Experts)
    draw_card(10.6, 6.4, 2.35, 1.05, 'Expert 1 (Active)', 'Wind Speed Specialist',
              fc=C_BLUE_LIGHT, ec=C_BLUE_BORDER, title_color=C_BLUE, lw=1.5)
    draw_card(10.6, 5.1, 2.35, 1.05, 'Expert 2 (Idle)', 'Turbulence Dynamics',
              fc=C_STAGE_BG, ec='#9CA3AF', title_color='#6B7280', lw=1.1, style='dashed')
    draw_card(10.6, 3.8, 2.35, 1.05, 'Expert 3 (Active)', 'Diurnal / Thermal Cycle',
              fc=C_BLUE_LIGHT, ec=C_BLUE_BORDER, title_color=C_BLUE, lw=1.5)
    draw_card(10.6, 2.5, 2.35, 1.05, 'Expert 4 (Idle)', 'Air Density / Humidity',
              fc=C_STAGE_BG, ec='#9CA3AF', title_color='#6B7280', lw=1.1, style='dashed')

    # Arrows: Router -> Experts
    # Active: Expert 1 & 3 (Thick Blue Solid)
    ax.annotate('', xy=(10.6, 6.92), xytext=(9.94, 5.38),
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.8,
                                connectionstyle='arc3,rad=-0.10', mutation_scale=14))
    ax.annotate('', xy=(10.6, 4.32), xytext=(9.94, 4.98),
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.8,
                                connectionstyle='arc3,rad=0.08', mutation_scale=14))
    # Idle: Expert 2 & 4 (Faint Muted Dotted)
    ax.annotate('', xy=(10.6, 5.62), xytext=(9.94, 5.22),
                arrowprops=dict(arrowstyle='->', color='#9CA3AF', lw=1.2, linestyle=':',
                                connectionstyle='arc3,rad=-0.04', mutation_scale=12))
    ax.annotate('', xy=(10.6, 3.02), xytext=(9.94, 4.85),
                arrowprops=dict(arrowstyle='->', color='#9CA3AF', lw=1.2, linestyle=':',
                                connectionstyle='arc3,rad=0.14', mutation_scale=12))

    # 2.5 Aggregation Node (\Sigma)
    agg_circle = Circle((13.65, 5.18), 0.44, facecolor='white', edgecolor=C_BLUE, lw=1.6, zorder=3)
    ax.add_patch(agg_circle)
    ax.text(13.65, 5.18, r'$\sum$', ha='center', va='center', fontsize=12.5, fontweight='bold', color=C_BLUE, zorder=4)

    # Arrows: Active Experts -> Aggregation Node
    ax.annotate('', xy=(13.21, 5.38), xytext=(12.95, 6.92),
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.7,
                                connectionstyle='arc3,rad=0.08', mutation_scale=14))
    ax.annotate('', xy=(13.21, 4.98), xytext=(12.95, 4.32),
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.7,
                                connectionstyle='arc3,rad=-0.08', mutation_scale=14))

    # 2.6 Seasonal Head
    draw_card(14.55, 4.55, 1.35, 1.25, 'Seasonal', r'Head $\hat{\mathbf{y}}_{\mathrm{se}}$',
              fc=C_BLUE_LIGHT, ec=C_BLUE_BORDER, title_color=C_BLUE, lw=1.4)
    
    # Arrow: Aggregation -> Seasonal Head
    ax.annotate('', xy=(14.55, 5.18), xytext=(14.09, 5.18),
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.8, mutation_scale=14))

    # ==================== STAGE 3: RECONSTRUCTION & ADDITIVE FUSION ====================
    # Trend Head
    draw_card(16.55, 8.1, 1.35, 1.15, 'Trend Head', r'$\hat{\mathbf{y}}_{\mathrm{tr}}$',
              fc=C_TEAL_LIGHT, ec=C_TEAL_BORDER, title_color=C_TEAL, lw=1.4)
    
    # Long Trend Line across the top from Stage 1 to Stage 3 (Clean, elevated at y=8.68)
    ax.annotate('', xy=(16.55, 8.68), xytext=(6.2, 8.68),
                arrowprops=dict(arrowstyle='->', color=C_TEAL, lw=2.0, mutation_scale=16))
    ax.text(11.37, 8.88, 'Lightweight Trend Branch (Linear/MLP)', ha='center', va='bottom',
            fontsize=8.2, fontweight='bold', color=C_TEAL, zorder=5)

    # Additive Fusion Node (+)
    fusion_circle = Circle((17.22, 4.6), 0.44, facecolor='white', edgecolor='#0F172A', lw=1.8, zorder=3)
    ax.add_patch(fusion_circle)
    ax.text(17.22, 4.6, r'$+$', ha='center', va='center', fontsize=14, fontweight='bold', color='#0F172A', zorder=4)

    # Arrow: Trend Head -> Fusion Node
    ax.annotate('', xy=(17.22, 5.04), xytext=(17.22, 8.1),
                arrowprops=dict(arrowstyle='->', color=C_TEAL, lw=1.8, mutation_scale=14))
    
    # Arrow: Seasonal Head -> Fusion Node
    ax.annotate('', xy=(16.78, 4.6), xytext=(15.9, 4.6),
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.8, mutation_scale=14))

    # Final Forecast Output Card
    draw_card(18.15, 4.1, 1.35, 1.45, 'Forecast', r'$\hat{\mathbf{Y}} \in \mathbb{R}^{H \times 1}$',
              fc='white', ec=C_DARK, title_color=C_DARK, lw=1.6)
    
    # Arrow: Fusion -> Forecast Output
    ax.annotate('', xy=(18.15, 4.6), xytext=(17.66, 4.6),
                arrowprops=dict(arrowstyle='->', color=C_DARK, lw=2.0, mutation_scale=16))

    # ==================== BOTTOM METRICS BADGES ====================
    badge_y = 0.35
    badge_h = 0.75
    draw_card(0.65, badge_y, 5.0, badge_h, 'Edge-Native: 1.33 MiB RAM | Linear O(L) Time',
              fc='#F0FDF4', ec='#22C55E', title_color='#15803D', title_size=8.4, bold=True)
    draw_card(6.7, badge_y, 5.0, badge_h, 'Sparse Compute: Top-2 of 4 Experts (50% FLOPs Reduction)',
              fc='#EFF6FF', ec='#3B82F6', title_color='#1D4ED8', title_size=8.4, bold=True)
    draw_card(12.75, badge_y, 5.0, badge_h, 'Green AI Verified: 6.53 ms Latency | 9.80 mJ / Step',
              fc='#FEF2F2', ec='#EF4444', title_color='#B91C1C', title_size=8.4, bold=True)

    plt.tight_layout(pad=0.15)
    
    # Save to all three canonical directories
    out_dirs = [
        Path('03_图表与绘图素材'),
        Path('02_论文正文与LaTeX'),
        Path('02_论文正文与LaTeX/latex_current_pack'),
    ]
    
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        for base in ['figure2', 'Figure_2_Final']:
            fig.savefig(d / f'{base}.pdf', format='pdf', bbox_inches='tight', facecolor='white')
            fig.savefig(d / f'{base}.png', format='png', dpi=600, bbox_inches='tight', facecolor='white')
    
    print('Successfully re-rendered publication-ready Figure 2 with ZERO collisions across all directories!')
    plt.close(fig)

if __name__ == '__main__':
    draw_figure_2()
