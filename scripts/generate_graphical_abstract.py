import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from pathlib import Path

def generate_graphical_abstract():
    # Nature / Elsevier 16:9 ratio: 14.0 in x 7.87 in (355.6 mm x 200 mm)
    fig, ax = plt.subplots(figsize=(14.0, 7.87), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56.25)
    ax.axis('off')
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'stix',
    })

    # Color Palette (Nature / Elsevier Morandi Style)
    C_BLUE_DARK   = '#1E3A8A'  # Deep Slate Navy
    C_BLUE_MED    = '#2563EB'  # Royal Blue
    C_BLUE_LIGHT  = '#EFF6FF'  # Very light blue
    C_TEAL_DARK   = '#0F766E'  # Deep Teal
    C_TEAL_MED    = '#0D9488'  # Medium Teal
    C_TEAL_LIGHT  = '#F0FDFA'  # Very light teal
    C_RED_DARK    = '#991B1B'  # Deep Crimson
    C_RED_MED     = '#DC2626'  # Red
    C_RED_LIGHT   = '#FEF2F2'  # Very light red
    C_AMBER_DARK  = '#B45309'  # Deep Amber
    C_AMBER_MED   = '#D97706'  # Amber
    C_AMBER_LIGHT = '#FFFBEB'  # Light Amber
    C_CARD_BG     = '#F8FAFC'  # Card background
    C_CARD_BORDER = '#CBD5E1'  # Card border
    C_TEXT_DARK   = '#0F172A'  # Slate 900
    C_TEXT_MUTED  = '#475569'  # Slate 600

    # Header Title Banner
    ax.text(50, 53.5, "Conditional Inductive Guardrails: Operating Boundaries of Physical Priors in Wind Forecasting",
            ha='center', va='center', fontsize=14.5, fontweight='bold', color=C_TEXT_DARK)
    ax.text(50, 51.5, "Controlled Empirical Identification Across Three Commercial Wind Farms and Electricity Market Balancing",
            ha='center', va='center', fontsize=10.0, color=C_TEXT_MUTED)

    # Helper: Card Drawer
    def draw_box(x, y, w, h, bg=C_CARD_BG, border=C_CARD_BORDER, lw=1.2, radius=0.8):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.0,rounding_size={radius}',
                             facecolor=bg, edgecolor=border, linewidth=lw, zorder=1)
        ax.add_patch(box)

    # ==================== PANEL 1: STATUS QUO & GOVERNANCE DEFICIT (Left: 2 to 31) ====================
    draw_box(2, 2, 29, 47.5, bg='#F8FAFC', border='#94A3B8', lw=1.4, radius=1.0)
    
    # Panel 1 Header
    draw_box(2, 45.2, 29, 4.3, bg='#E2E8F0', border='#94A3B8', radius=0.8)
    ax.text(16.5, 47.35, "1. Status Quo: Governance Deficit", ha='center', va='center',
            fontsize=11.0, fontweight='bold', color=C_BLUE_DARK)

    # 46 Studies Audited
    draw_box(3.5, 39.5, 26, 4.5, bg='white', border='#CBD5E1')
    ax.text(16.5, 42.4, "Systematic Scoping Review (2023–2026)", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=C_TEXT_MUTED)
    ax.text(16.5, 40.7, "N = 46 Contemporary Physics-Informed ML Studies", ha='center', va='center',
            fontsize=9.2, fontweight='bold', color=C_BLUE_MED)

    # Green Claim vs Reality Stack
    draw_box(3.5, 32.5, 26, 5.8, bg='#ECFDF5', border='#059669', lw=1.2)
    ax.text(16.5, 36.2, "100.0% Uniform Positive Claims", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color='#065F46')
    ax.text(16.5, 34.0, "46/46 studies report forecasting accuracy gains", ha='center', va='center',
            fontsize=8.2, color='#047857')

    # Red Reality Ledger
    ax.text(16.5, 30.5, "The Ground Reality Ledger (Audit Findings)", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=C_RED_DARK)

    # 3 Audit Metrics
    draw_box(3.5, 22.0, 26, 7.2, bg=C_RED_LIGHT, border='#F87171', lw=1.0)
    ax.text(5.0, 27.2, "• Failure Regimes:", fontsize=8.5, fontweight='bold', color=C_RED_DARK)
    ax.text(16.0, 27.2, "0.0% reported (0/46)", fontsize=8.5, fontweight='bold', color=C_RED_DARK)
    ax.text(5.0, 25.2, "• Open Software Deficit:", fontsize=8.5, fontweight='bold', color=C_RED_DARK)
    ax.text(18.5, 25.2, "95.7% withheld (44/46)", fontsize=8.5, fontweight='bold', color=C_RED_DARK)
    ax.text(5.0, 23.2, "• Capacity Confounded:", fontsize=8.5, fontweight='bold', color=C_RED_DARK)
    ax.text(18.5, 23.2, "50.0% confounded (23/46)", fontsize=8.5, fontweight='bold', color=C_RED_DARK)

    # AST Software Divergence
    draw_box(3.5, 11.0, 26, 9.8, bg='#FFFBEB', border='#F59E0B', lw=1.1)
    ax.text(16.5, 19.1, "Software AST Graph Divergence", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=C_AMBER_DARK)
    ax.text(16.5, 17.0, "Public PCT Code Reverses JS-Divergence:", ha='center', va='center',
            fontsize=7.8, color=C_TEXT_DARK)
    ax.text(16.5, 15.0, r"Executed PyTorch: $\mathrm{JS}(y_{\mathrm{ref}}, y_{\mathrm{pred}})$ vs Formula $\mathrm{JS}(y_{\mathrm{pred}} \parallel y_{\mathrm{ref}})$",
            ha='center', va='center', fontsize=7.5, color=C_TEXT_DARK)
    ax.text(16.5, 12.8, r"Induces $\mathcal{O}(1/\epsilon)$ gradient singularity ($2.20\times$–$2.72\times$ norm surge)",
            ha='center', va='center', fontsize=7.5, fontweight='bold', color=C_RED_DARK)

    # Bottom Question Callout
    draw_box(3.5, 3.2, 26, 6.5, bg='white', border=C_BLUE_MED, lw=1.3)
    ax.text(16.5, 7.5, "The Core Research Question:", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=C_BLUE_DARK)
    ax.text(16.5, 5.0, "When do physical priors genuinely enhance,\nmerely mimic, or impair operational decisions?",
            ha='center', va='center', fontsize=8.0, color=C_TEXT_DARK)

    # Transition Arrow 1 -> 2
    ax.annotate("", xy=(33.8, 25.75), xytext=(31.2, 25.75),
                arrowprops=dict(arrowstyle="->", color=C_BLUE_DARK, lw=2.2, mutation_scale=16))

    # ==================== PANEL 2: CONTROLLED IDENTIFICATION PROTOCOL (Middle: 34 to 66) ====================
    draw_box(34, 2, 32, 47.5, bg='#F8FAFC', border='#94A3B8', lw=1.4, radius=1.0)
    
    # Panel 2 Header
    draw_box(34, 45.2, 32, 4.3, bg='#E2E8F0', border='#94A3B8', radius=0.8)
    ax.text(50.0, 47.35, "2. Controlled Identification Protocol", ha='center', va='center',
            fontsize=11.0, fontweight='bold', color=C_BLUE_DARK)

    # 2x2 Negative Controls
    draw_box(35.5, 33.5, 29, 10.5, bg='white', border='#CBD5E1')
    ax.text(50.0, 42.3, "Negative Control Crucible (Capacity Strictly Matched)", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=C_TEXT_DARK)
    
    # Branch A: Authentic
    draw_box(36.5, 34.5, 13, 6.2, bg=C_TEAL_LIGHT, border=C_TEAL_MED, lw=1.1)
    ax.text(43.0, 39.0, "Authentic Prior", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_TEAL_DARK)
    ax.text(43.0, 37.0, r"Empirical Curve $C_{\mathrm{emp}}$", ha='center', va='center', fontsize=8.0, color=C_TEXT_DARK)
    ax.text(43.0, 35.3, "True Aerodynamic Geometry", ha='center', va='center', fontsize=7.2, color=C_TEXT_MUTED)

    # Branch B: Pseudo-scrambled
    draw_box(50.5, 34.5, 13, 6.2, bg=C_RED_LIGHT, border=C_RED_MED, lw=1.1)
    ax.text(57.0, 39.0, "Pseudo-Scrambled", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_RED_DARK)
    ax.text(57.0, 37.0, r"Disrupted Prior $C_{\mathrm{scramble}}$", ha='center', va='center', fontsize=8.0, color=C_TEXT_DARK)
    ax.text(57.0, 35.3, "Form- & Weight-Matched", ha='center', va='center', fontsize=7.2, color=C_TEXT_MUTED)

    # Information-Budget Matching
    draw_box(35.5, 22.0, 29, 10.2, bg='#F0FDF4', border='#16A34A', lw=1.2)
    ax.text(50.0, 30.3, "Information-Budget Matching Experiment", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color='#15803D')
    ax.text(50.0, 28.2, r"Controlled Sample Volume: $N_{\mathrm{prior}} = N_{\mathrm{train}} = 500$", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=C_TEXT_DARK)
    ax.text(50.0, 25.8, "• Contemporaneous sample prior retains 77% to 92% of benefit", ha='center', va='center',
            fontsize=7.8, color=C_TEXT_DARK)
    ax.text(50.0, 23.6, "• Proves gains stem from domain geometry, not hidden archive data", ha='center', va='center',
            fontsize=7.8, color=C_TEXT_MUTED)

    # Predefined Stress Axes
    draw_box(35.5, 10.5, 29, 10.2, bg='white', border='#CBD5E1')
    ax.text(50.0, 18.8, "Predefined Operational Stress Axes", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=C_TEXT_DARK)
    
    # Stress Axis 1
    draw_box(36.5, 11.5, 13, 5.8, bg='#F8FAFC', border='#94A3B8')
    ax.text(43.0, 15.6, "Penalty Sweep", ha='center', va='center', fontsize=8.2, fontweight='bold', color=C_BLUE_DARK)
    ax.text(43.0, 13.8, r"$\lambda \in \{0.01, 0.1, 1.0\}$", ha='center', va='center', fontsize=8.0, color=C_TEXT_DARK)
    ax.text(43.0, 12.3, "Separates mimicry", ha='center', va='center', fontsize=7.0, color=C_TEXT_MUTED)

    # Stress Axis 2
    draw_box(50.5, 11.5, 13, 5.8, bg='#F8FAFC', border='#94A3B8')
    ax.text(57.0, 15.6, "Proxy Latency Ladder", ha='center', va='center', fontsize=8.2, fontweight='bold', color=C_BLUE_DARK)
    ax.text(57.0, 13.8, r"Delay $k \in [1\mathrm{h}, 6\mathrm{h}]$", ha='center', va='center', fontsize=8.0, color=C_TEXT_DARK)
    ax.text(57.0, 12.3, "Offline staleness test", ha='center', va='center', fontsize=7.0, color=C_TEXT_MUTED)

    # Bottom Multi-Farm Field Foundation
    draw_box(35.5, 3.2, 29, 6.2, bg='#EFF6FF', border=C_BLUE_MED)
    ax.text(50.0, 7.3, "Empirical Validation Foundation (3 Commercial Wind Farms)", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=C_BLUE_DARK)
    ax.text(50.0, 5.0, "Kelmarsh (12.3 MW) | Hill of Towie (48.3 MW) | Penmanshiel (28.7 MW)\nCoupled with 2017 UK Elexon BSC Half-Hourly Balancing Settlement",
            ha='center', va='center', fontsize=7.2, color=C_TEXT_DARK)

    # Transition Arrow 2 -> 3
    ax.annotate("", xy=(68.8, 25.75), xytext=(66.2, 25.75),
                arrowprops=dict(arrowstyle="->", color=C_BLUE_DARK, lw=2.2, mutation_scale=16))

    # ==================== PANEL 3: EMPIRICAL OPERATING ENVELOPE (Right: 69 to 98) ====================
    draw_box(69, 2, 29, 47.5, bg='#F8FAFC', border='#94A3B8', lw=1.4, radius=1.0)
    
    # Panel 3 Header
    draw_box(69, 45.2, 29, 4.3, bg='#E2E8F0', border='#94A3B8', radius=0.8)
    ax.text(83.5, 47.35, "3. Operational Decision Envelope", ha='center', va='center',
            fontsize=11.0, fontweight='bold', color=C_BLUE_DARK)

    # Regime 1: Guardrail
    draw_box(70.5, 36.5, 26, 7.8, bg=C_TEAL_LIGHT, border=C_TEAL_MED, lw=1.2)
    ax.text(71.8, 42.5, "Regime 1: Inductive Guardrail", fontsize=8.8, fontweight='bold', color=C_TEAL_DARK)
    ax.text(71.8, 40.6, r"Sample-Restricted Regime ($N_{\mathrm{train}} \leq 1{,}583$ on Penmanshiel):", fontsize=7.4, color=C_TEXT_DARK)
    ax.text(71.8, 38.8, r"• Authentic Prior:  -1.22% Downside Risk Protection", fontsize=7.6, fontweight='bold', color=C_TEAL_DARK)
    ax.text(71.8, 37.2, r"• Scrambled Prior: +68.1% Dispatch Penalty Surge", fontsize=7.6, fontweight='bold', color=C_RED_DARK)

    # Regime 2: Mimicry
    draw_box(70.5, 27.5, 26, 7.8, bg='#FFFBEB', border='#F59E0B', lw=1.2)
    ax.text(71.8, 33.5, "Regime 2: Regularization Mimicry", fontsize=8.8, fontweight='bold', color=C_AMBER_DARK)
    ax.text(71.8, 31.6, r"Weak Penalty Schedule ($\lambda = 0.01$ on Kelmarsh):", fontsize=7.4, color=C_TEXT_DARK)
    ax.text(71.8, 29.8, r"• Authentic (0.08810) $\approx$ Scrambled (0.08801)", fontsize=7.6, color=C_TEXT_DARK)
    ax.text(71.8, 28.2, "• Indistinguishable loss proves numerical smoothing, not physics", fontsize=7.2, color=C_TEXT_MUTED)

    # Regime 3: Inversion
    draw_box(70.5, 18.5, 26, 7.8, bg='#FEF2F2', border='#F87171', lw=1.2)
    ax.text(71.8, 24.5, "Regime 3: Large-Sample Inversion", fontsize=8.8, fontweight='bold', color=C_RED_DARK)
    ax.text(71.8, 22.6, r"High Sample Density ($N_{\mathrm{train}} \geq 5{,}000$):", fontsize=7.4, color=C_TEXT_DARK)
    ax.text(71.8, 20.8, "• Unconstrained deep models match or outperform priors", fontsize=7.4, color=C_TEXT_DARK)
    ax.text(71.8, 19.2, "• Fixed priors impose structural bias under rich observations", fontsize=7.2, color=C_TEXT_MUTED)

    # Decision Decoupling & Zero-Bid
    draw_box(70.5, 3.2, 26, 14.0, bg='#EFF6FF', border=C_BLUE_DARK, lw=1.3)
    ax.text(83.5, 15.3, "Decision-Risk Decoupling & Zero-Bid Probe", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=C_BLUE_DARK)
    ax.text(71.8, 13.5, "• RMSE Decoupling: Symmetric error decouples from tail risk.", fontsize=7.2, color=C_TEXT_DARK)
    ax.text(71.8, 11.8, r"• Degenerate Zero-Bid ($q \equiv 0$): Standalone cashflow earns", fontsize=7.2, color=C_TEXT_DARK)
    ax.text(73.5, 10.2, "+£239,450/(turb·yr) on Penmanshiel without forecasting skill!",
            fontsize=7.4, fontweight='bold', color=C_RED_DARK)
    ax.text(71.8, 8.4, "• Proof that unhedged balancing metrics deceive market operators.", fontsize=7.2, color=C_TEXT_MUTED)
    
    # REFORMS-PCWPF Footer
    draw_box(71.5, 4.2, 24, 3.2, bg='white', border='#94A3B8')
    ax.text(83.5, 5.8, "Actionable Governance: 5-Gate REFORMS-PCWPF Checklist", ha='center', va='center',
            fontsize=7.5, fontweight='bold', color=C_BLUE_DARK)

    # Output paths
    out_dir_1 = Path("01_论文投稿材料/submission_package")
    out_dir_2 = Path("01_论文投稿材料")
    out_dir_3 = Path("03_图表与绘图素材")
    for d in [out_dir_1, out_dir_2, out_dir_3]:
        d.mkdir(parents=True, exist_ok=True)
    
    pdf_path_1 = out_dir_1 / "graphical_abstract.pdf"
    png_path_1 = out_dir_1 / "graphical_abstract.png"
    pdf_path_2 = out_dir_2 / "graphical_abstract.pdf"
    png_path_2 = out_dir_2 / "graphical_abstract.png"

    plt.savefig(pdf_path_1, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path_1, format='png', bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path_2, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path_2, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Successfully generated Graphical Abstract:")
    print(f"  {pdf_path_1} ({pdf_path_1.stat().st_size} bytes)")
    print(f"  {png_path_1} ({png_path_1.stat().st_size} bytes)")

if __name__ == "__main__":
    generate_graphical_abstract()
