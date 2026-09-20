"""
Generate Remaining Scientific Glyphs for Figure 1 Constraint Lifecycle Taxonomy
================================================================================
Strict Scientific-Object Language (Zero Generic Pictograms/Slop).
All glyphs are pure mathematical/computational objects matching the paper's
audited formulations and PRISMA-ScR empirical scoping taxonomy.

Covers:
  - D1 Provenance: Intrinsic (Betz Limit), Learned-Online, Unreported
  - D2 Stage: Architecture (Manifold Projection), Training Loss (Regularization)
  - D3 Dependency: NWP Forecast Matrix, Contemporaneous-State Multi-Turbine Vector
  - D4 Fidelity: Verified Aligned (AST Check), Code Mismatch, Not Found (404/95.7%)
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Common styling
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['svg.fonttype'] = 'none'

C_BLUE = '#1d4ed8'
C_CYAN = '#0284c7'
C_GREEN = '#059669'
C_AMBER = '#d97706'
C_RED = '#dc2626'
C_SLATE = '#475569'
C_LIGHT_SLATE = '#64748b'
C_BG_BLUE = '#f0f9ff'
C_BG_GREEN = '#ecfdf5'
C_BG_AMBER = '#fffbeb'
C_BG_RED = '#fef2f2'

# ==============================================================================
# 1. D1: Intrinsic (Betz Analytical Limit & Theoretical Cubic Power Curve)
# ==============================================================================
def generate_d1_intrinsic():
    """
    Theoretical cubic envelope P = 0.5 * rho * A * Cp * v^3 saturating at rated power,
    with Betz limit ceiling Cp <= 16/27.
    """
    print("Generating D1 Intrinsic (Betz Limit & Aerodynamic Envelope)...")
    v = np.linspace(0, 15, 60)
    v_rated = 11.0
    p_theory = np.minimum((v / v_rated)**3, 1.0)
    
    fig = plt.figure(figsize=(1.9, 1.2), dpi=100)
    ax = fig.add_axes([0.10, 0.16, 0.82, 0.74])
    ax.grid(False)
    
    # Betz limit ceiling line
    ax.plot([0, 15], [1.0, 1.0], color=C_LIGHT_SLATE, linestyle=':', linewidth=1.3)
    # Theory curve
    ax.plot(v, p_theory, color=C_CYAN, linestyle='-', linewidth=2.2)
    
    ax.text(15.3, 1.0, r"$C_p \leq \frac{16}{27}$", color=C_LIGHT_SLATE, fontsize=6.8, va='center',
            bbox=dict(boxstyle='square,pad=0.08', facecolor='white', edgecolor='none'))
    ax.text(7.2, 0.58, r"$P \propto v^3$", color=C_CYAN, fontsize=7.2, fontweight='bold')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$v$", fontsize=8.0, color=C_SLATE, labelpad=2)
    ax.set_ylabel("Power", fontsize=7.5, color=C_SLATE, labelpad=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(C_LIGHT_SLATE)
    ax.spines['bottom'].set_color(C_LIGHT_SLATE)
    ax.spines['left'].set_linewidth(0.85)
    ax.spines['bottom'].set_linewidth(0.85)
    ax.set_xlim(-0.5, 21.0)
    ax.set_ylim(-0.05, 1.15)
    
    fig.savefig(OUT_DIR / "fig1_d1_intrinsic.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d1_intrinsic.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done: fig1_d1_intrinsic.svg, fig1_d1_intrinsic.pdf")


# ==============================================================================
# 2. D1: Learned-Online (Dynamic Subnetwork Adaptation Loop)
# ==============================================================================
def generate_d1_learned_online():
    """
    Dynamic parameter subnetwork: theta_t = g_phi(X_t) -> Prior_t
    """
    print("Generating D1 Learned-Online (Dynamic Subnetwork Adaptation)...")
    fig = plt.figure(figsize=(2.2, 1.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # Input telemetry
    ax.text(0.12, 0.52, r"$X_t$", ha='center', va='center', fontsize=8.0, fontweight='bold', color=C_SLATE)
    
    # Arrow 1
    ax.annotate('', xy=(0.28, 0.52), xytext=(0.19, 0.52),
                arrowprops=dict(arrowstyle="->", color=C_SLATE, lw=1.2, mutation_scale=9))
    
    # Subnetwork Box
    box_net = FancyBboxPatch((0.30, 0.22), 0.32, 0.58, boxstyle="round,pad=0.02,rounding_size=0.04",
                             edgecolor=C_CYAN, facecolor=C_BG_BLUE, linewidth=1.1)
    ax.add_patch(box_net)
    ax.text(0.46, 0.59, r"$g_\phi(X_t)$", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#0369a1')
    ax.text(0.46, 0.38, "adaptive net", ha='center', va='center', fontsize=5.8, color=C_CYAN)
    
    # Arrow 2
    ax.annotate('', xy=(0.70, 0.52), xytext=(0.63, 0.52),
                arrowprops=dict(arrowstyle="->", color=C_SLATE, lw=1.2, mutation_scale=9))
    
    # Output parameter
    ax.text(0.85, 0.52, r"$\theta_t \rightarrow \mathcal{C}_t$", ha='center', va='center', fontsize=7.2, fontweight='bold', color=C_BLUE)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.savefig(OUT_DIR / "fig1_d1_learned_online.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d1_learned_online.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done: fig1_d1_learned_online.svg, fig1_d1_learned_online.pdf")


# ==============================================================================
# 3. D1: Unreported (Missing Calibration Basis)
# ==============================================================================
def generate_d1_unreported():
    """
    Omitted calibration basis: theta = ?
    """
    print("Generating D1 Unreported (Omitted Calibration Basis)...")
    fig = plt.figure(figsize=(2.0, 1.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    box = FancyBboxPatch((0.15, 0.20), 0.70, 0.62, boxstyle="round,pad=0.02,rounding_size=0.04",
                         edgecolor='#94a3b8', facecolor='#f8fafc', linewidth=1.1, linestyle='--')
    ax.add_patch(box)
    
    ax.text(0.50, 0.60, r"$\theta_{\mathrm{phys}} = \mathbf{?}$", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#64748b')
    ax.text(0.50, 0.36, "calibration omitted", ha='center', va='center', fontsize=6.2, style='italic', color='#94a3b8')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.savefig(OUT_DIR / "fig1_d1_unreported.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d1_unreported.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done: fig1_d1_unreported.svg, fig1_d1_unreported.pdf")


# ==============================================================================
# 4. D2: Architecture (Computational Graph Manifold Projection Layer)
# ==============================================================================
def generate_d2_architecture():
    """
    Internal layer projection: h -> [ Pi_M ] -> y_hat in M_phys
    """
    print("Generating D2 Architecture (Layer Projection)...")
    fig = plt.figure(figsize=(2.2, 1.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    ax.text(0.12, 0.52, r"$h_t$", ha='center', va='center', fontsize=8.0, fontweight='bold', color=C_SLATE)
    
    # Arrow 1
    ax.annotate('', xy=(0.28, 0.52), xytext=(0.19, 0.52),
                arrowprops=dict(arrowstyle="->", color=C_SLATE, lw=1.2, mutation_scale=9))
    
    # Projection Layer Box
    box_proj = FancyBboxPatch((0.30, 0.22), 0.34, 0.58, boxstyle="round,pad=0.02,rounding_size=0.04",
                              edgecolor=C_GREEN, facecolor=C_BG_GREEN, linewidth=1.1)
    ax.add_patch(box_proj)
    ax.text(0.47, 0.59, r"$\Pi_{\mathcal{M}}(\cdot)$", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#065f46')
    ax.text(0.47, 0.38, "manifold layer", ha='center', va='center', fontsize=5.8, color=C_GREEN)
    
    # Arrow 2
    ax.annotate('', xy=(0.72, 0.52), xytext=(0.65, 0.52),
                arrowprops=dict(arrowstyle="->", color=C_SLATE, lw=1.2, mutation_scale=9))
    
    # Output
    ax.text(0.86, 0.52, r"$\hat{y} \in \mathcal{M}$", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#065f46')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.savefig(OUT_DIR / "fig1_d2_architecture.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d2_architecture.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done: fig1_d2_architecture.svg, fig1_d2_architecture.pdf")


# ==============================================================================
# 5. D2: Training Loss (Loss Regularization Objective)
# ==============================================================================
def generate_d2_training_loss():
    """
    Loss regularization: L = L_task + lambda * L_phys
    """
    print("Generating D2 Training Loss (Regularization)...")
    fig = plt.figure(figsize=(2.2, 1.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    box = FancyBboxPatch((0.06, 0.18), 0.88, 0.64, boxstyle="round,pad=0.02,rounding_size=0.04",
                         edgecolor=C_GREEN, facecolor=C_BG_GREEN, linewidth=1.1)
    ax.add_patch(box)
    
    ax.text(0.50, 0.59, r"$\mathcal{L}_{\mathrm{task}} + \lambda \mathcal{L}_{\mathrm{phys}}$",
            ha='center', va='center', fontsize=7.8, fontweight='bold', color='#065f46')
    ax.text(0.50, 0.36, r"gradient penalty $\min_\theta \mathcal{L}$",
            ha='center', va='center', fontsize=6.0, color='#047857')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.savefig(OUT_DIR / "fig1_d2_training_loss.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d2_training_loss.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done: fig1_d2_training_loss.svg, fig1_d2_training_loss.pdf")


# ==============================================================================
# 6. D3: NWP Forecast (Future Weather Prediction Matrix)
# ==============================================================================
def generate_d3_nwp():
    """
    NWP Weather forecast matrix input: W_NWP in R^{H x d_met}
    """
    print("Generating D3 NWP Forecast Input...")
    fig = plt.figure(figsize=(2.2, 1.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # Left: Inputs bracket
    ax.text(0.18, 0.70, r"$P_{t-L:t}$", ha='center', va='center', fontsize=6.8, color=C_SLATE)
    # NWP forecast matrix bracket
    ax.plot([0.06, 0.04, 0.04, 0.06], [0.55, 0.55, 0.15, 0.15], color=C_AMBER, lw=1.1)
    ax.plot([0.30, 0.32, 0.32, 0.30], [0.55, 0.55, 0.15, 0.15], color=C_AMBER, lw=1.1)
    ax.text(0.18, 0.44, r"$v_{t+1} \;\; \phi_{t+1}$", ha='center', va='center', fontsize=6.0, color='#78350f')
    ax.text(0.18, 0.26, r"$\vdots \qquad \vdots$", ha='center', va='center', fontsize=6.0, color=C_AMBER)
    
    # Arrow
    ax.annotate('', xy=(0.42, 0.48), xytext=(0.35, 0.48),
                arrowprops=dict(arrowstyle="->", color=C_SLATE, lw=1.2, mutation_scale=9))
    
    # Model box
    box_m = FancyBboxPatch((0.44, 0.22), 0.30, 0.54, boxstyle="round,pad=0.02,rounding_size=0.04",
                           edgecolor=C_AMBER, facecolor=C_BG_AMBER, linewidth=1.1)
    ax.add_patch(box_m)
    ax.text(0.59, 0.55, "Forecast\nModel", ha='center', va='center', fontsize=6.2, fontweight='bold', color='#78350f')
    ax.text(0.59, 0.32, r"$+$NWP", ha='center', va='center', fontsize=5.6, color=C_AMBER)
    
    # Arrow out
    ax.annotate('', xy=(0.82, 0.48), xytext=(0.75, 0.48),
                arrowprops=dict(arrowstyle="->", color=C_SLATE, lw=1.2, mutation_scale=9))
    ax.text(0.91, 0.48, r"$\hat{y}$", ha='center', va='center', fontsize=8.0, fontweight='bold', color=C_AMBER)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.savefig(OUT_DIR / "fig1_d3_nwp.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d3_nwp.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done: fig1_d3_nwp.svg, fig1_d3_nwp.pdf")


# ==============================================================================
# 7. D3: Contemporaneous-State (Multi-Turbine Sensor Vector)
# ==============================================================================
def generate_d3_contemporaneous():
    """
    Multi-turbine farm telemetry vector: s_t = [s_t^(1), ..., s_t^(M)]^T
    """
    print("Generating D3 Contemporaneous-State Telemetry...")
    fig = plt.figure(figsize=(2.2, 1.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # Vector bracket
    ax.plot([0.06, 0.04, 0.04, 0.06], [0.82, 0.82, 0.20, 0.20], color=C_AMBER, lw=1.1)
    ax.plot([0.28, 0.30, 0.30, 0.28], [0.82, 0.82, 0.20, 0.20], color=C_AMBER, lw=1.1)
    ax.text(0.17, 0.71, r"$s_t^{(1)}$", ha='center', va='center', fontsize=6.5, color='#78350f')
    ax.text(0.17, 0.52, r"$\vdots$", ha='center', va='center', fontsize=6.5, color=C_AMBER)
    ax.text(0.17, 0.31, r"$s_t^{(M)}$", ha='center', va='center', fontsize=6.5, color='#78350f')
    
    # Arrow
    ax.annotate('', xy=(0.42, 0.51), xytext=(0.33, 0.51),
                arrowprops=dict(arrowstyle="->", color=C_SLATE, lw=1.2, mutation_scale=9))
    
    # Model box
    box_m = FancyBboxPatch((0.44, 0.22), 0.30, 0.54, boxstyle="round,pad=0.02,rounding_size=0.04",
                           edgecolor=C_AMBER, facecolor=C_BG_AMBER, linewidth=1.1)
    ax.add_patch(box_m)
    ax.text(0.59, 0.55, "Forecast\nModel", ha='center', va='center', fontsize=6.2, fontweight='bold', color='#78350f')
    ax.text(0.59, 0.32, "farm state", ha='center', va='center', fontsize=5.5, color=C_AMBER)
    
    # Arrow out
    ax.annotate('', xy=(0.82, 0.51), xytext=(0.75, 0.51),
                arrowprops=dict(arrowstyle="->", color=C_SLATE, lw=1.2, mutation_scale=9))
    ax.text(0.91, 0.51, r"$\hat{y}$", ha='center', va='center', fontsize=8.0, fontweight='bold', color=C_AMBER)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.savefig(OUT_DIR / "fig1_d3_contemporaneous.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d3_contemporaneous.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done: fig1_d3_contemporaneous.svg, fig1_d3_contemporaneous.pdf")


# ==============================================================================
# 8. D4: Verified Aligned (AST Code-Math Parity)
# ==============================================================================
def generate_d4_code_aligned():
    """
    Code-equation alignment verified: L_phys == AST(code)
    """
    print("Generating D4 Verified Aligned (AST Code Parity)...")
    fig = plt.figure(figsize=(2.2, 1.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    box = FancyBboxPatch((0.06, 0.18), 0.88, 0.64, boxstyle="round,pad=0.02,rounding_size=0.04",
                         edgecolor=C_GREEN, facecolor=C_BG_GREEN, linewidth=1.1)
    ax.add_patch(box)
    
    ax.text(0.50, 0.60, r"$\mathcal{L}_{\mathrm{phys}} \equiv \mathrm{AST}(\mathrm{code})$",
            ha='center', va='center', fontsize=7.2, fontweight='bold', color='#065f46')
    ax.text(0.50, 0.36, "\u2713 verified aligned (2.2%)",
            ha='center', va='center', fontsize=6.2, fontweight='bold', color=C_GREEN)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.savefig(OUT_DIR / "fig1_d4_code_aligned.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d4_code_aligned.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done: fig1_d4_code_aligned.svg, fig1_d4_code_aligned.pdf")


# ==============================================================================
# 9. D4: Code Mismatch (Reverse-JS Divergence / Algebraic Discrepancy)
# ==============================================================================
def generate_d4_code_mismatch():
    """
    Algebraic mismatch in public codebase: Math != Code
    """
    print("Generating D4 Code Mismatch (Algebraic Deviation)...")
    fig = plt.figure(figsize=(2.2, 1.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    box = FancyBboxPatch((0.06, 0.18), 0.88, 0.64, boxstyle="round,pad=0.02,rounding_size=0.04",
                         edgecolor=C_AMBER, facecolor=C_BG_AMBER, linewidth=1.1)
    ax.add_patch(box)
    
    ax.text(0.50, 0.60, r"$\mathrm{Math} \neq \mathrm{Code}$",
            ha='center', va='center', fontsize=7.6, fontweight='bold', color='#78350f')
    ax.text(0.50, 0.36, "\u00d7 algebraic mismatch (2.2%)",
            ha='center', va='center', fontsize=6.2, fontweight='bold', color=C_AMBER)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.savefig(OUT_DIR / "fig1_d4_code_mismatch.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d4_code_mismatch.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done: fig1_d4_code_mismatch.svg, fig1_d4_code_mismatch.pdf")


# ==============================================================================
# 10. D4: Not Found (Closed Source / Missing Repository 95.7%)
# ==============================================================================
def generate_d4_not_found():
    """
    Missing software repository (95.7% scoping finding)
    """
    print("Generating D4 Not Found (Missing Repo / 95.7%)...")
    fig = plt.figure(figsize=(2.2, 1.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    box = FancyBboxPatch((0.06, 0.18), 0.88, 0.64, boxstyle="round,pad=0.02,rounding_size=0.04",
                         edgecolor=C_RED, facecolor=C_BG_RED, linewidth=1.1)
    ax.add_patch(box)
    
    ax.text(0.50, 0.60, "404 Repo Not Found",
            ha='center', va='center', fontsize=7.2, fontweight='bold', color='#991b1b')
    ax.text(0.50, 0.36, "closed-source (95.7%)",
            ha='center', va='center', fontsize=6.2, fontweight='bold', color=C_RED)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.savefig(OUT_DIR / "fig1_d4_not_found.svg", format='svg', bbox_inches='tight', transparent=False)
    fig.savefig(OUT_DIR / "fig1_d4_not_found.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("  -> Done: fig1_d4_not_found.svg, fig1_d4_not_found.pdf")


if __name__ == '__main__':
    print("==================================================")
    print("GENERATING REMAINING 10 SCIENTIFIC GLYPHS")
    print("==================================================")
    generate_d1_intrinsic()
    generate_d1_learned_online()
    generate_d1_unreported()
    generate_d2_architecture()
    generate_d2_training_loss()
    generate_d3_nwp()
    generate_d3_contemporaneous()
    generate_d4_code_aligned()
    generate_d4_code_mismatch()
    generate_d4_not_found()
    print("==================================================")
    print("ALL 10 SCIENTIFIC GLYPHS GENERATED SUCCESSFULLY")
    print("==================================================")
