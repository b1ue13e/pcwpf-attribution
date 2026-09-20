"""Render Figure 1: Five-Dimensional Constraint Lifecycle Framework (Schema v4)."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow
from pathlib import Path

fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
ax.set_xlim(0, 11)
ax.set_ylim(0, 6)
ax.axis("off")

# 5 boxes
stages = [
    {
        "dim": "D1: Provenance",
        "title": "Parameter\nProvenance",
        "desc": "Where physics origin:\n• Intrinsic (87.0%)\n• Historical-static (10.9%)\n• Learned-online (2.2%)",
        "color": "#e6f0fa",
        "border": "#2b6cb0",
        "x": 0.4
    },
    {
        "dim": "D2: Stage",
        "title": "Application\nStage",
        "desc": "Where constraint acts:\n• Architecture (50.0%)\n• Training Loss (45.7%)\n• Data Gen (2.2%)\n• Post-hoc Clip (2.2%)",
        "color": "#f0fff4",
        "border": "#2f855a",
        "x": 2.5
    },
    {
        "dim": "D3: Dependency",
        "title": "Inference\nRequirement",
        "desc": "Deployment telemetry:\n• None / SCADA (78.3%)\n• NWP Forecast (13.0%)\n• Real-time State (8.7%)",
        "color": "#fefcbf",
        "border": "#d69e2e",
        "x": 4.6
    },
    {
        "dim": "D4: Fidelity",
        "title": "Software\nFidelity",
        "desc": "Implementation check:\n• Not Found (95.7%)\n• Code Mismatch (2.2%)\n• Verified Aligned (2.2%)",
        "color": "#feebc8",
        "border": "#dd6b20",
        "x": 6.7
    },
    {
        "dim": "D5: Harm",
        "title": "Harm & Ablation\nReporting",
        "desc": "Ablation & safety audit:\n• Strict Isolation (50.0%)\n• Module Ablation (50.0%)\n• Negative Harm (0.0%)",
        "color": "#fed7d7",
        "border": "#e53e3e",
        "x": 8.8
    }
]

w = 1.8
h = 3.8
y = 1.1

for s in stages:
    # Outer box
    ax.add_patch(FancyBboxPatch((s["x"], y), w, h, boxstyle="round,pad=0.08",
                                fc=s["color"], ec=s["border"], lw=2.0))
    # Header box
    ax.add_patch(FancyBboxPatch((s["x"], y + h - 0.7), w, 0.7, boxstyle="round,pad=0.05",
                                fc=s["border"], ec=s["border"], lw=1.0))
    ax.text(s["x"] + w / 2, y + h - 0.35, s["dim"], ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white")
    # Subtitle
    ax.text(s["x"] + w / 2, y + h - 1.25, s["title"], ha="center", va="center",
            fontsize=10, fontweight="bold", color="#2d3748")
    # Divider line
    ax.plot([s["x"] + 0.15, s["x"] + w - 0.15], [y + h - 1.7, y + h - 1.7], color=s["border"], lw=1.0, linestyle="--")
    # Description
    ax.text(s["x"] + 0.15, y + 0.3, s["desc"], ha="left", va="bottom",
            fontsize=8.5, color="#1a202c", linespacing=1.4)

# Arrows between stages
for i in range(len(stages) - 1):
    x_curr = stages[i]["x"] + w
    x_next = stages[i+1]["x"]
    y_mid = y + h / 2
    ax.annotate("", xy=(x_next - 0.05, y_mid), xytext=(x_curr + 0.05, y_mid),
                arrowprops=dict(arrowstyle="->", color="#4a5568", lw=2.2, mutation_scale=18))

# Super-title and flow context
ax.text(5.5, 5.4, "Constraint Lifecycle Taxonomy (Schema v4): Five Orthogonal Dimensions",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#1a202c")
ax.text(5.5, 5.0, "Tracking physical priors from mathematical origin through software implementation to operational deployment",
        ha="center", va="center", fontsize=9.5, style="italic", color="#4a5568")

plt.tight_layout()

# Save to all target locations
targets = [
    Path("03_图表与绘图素材"),
    Path("01_论文投稿材料/active_manuscript/figures"),
    Path("01_论文投稿材料/chinese_manuscript/figures"),
    Path("02_论文正文与LaTeX"),
]

for out_dir in targets:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "fig_constraint_lifecycle.pdf", bbox_inches="tight", facecolor="white")
    if out_dir == Path("03_图表与绘图素材"):
        plt.savefig(out_dir / "fig_constraint_lifecycle.png", bbox_inches="tight", facecolor="white")
    print(f"Exported fig_constraint_lifecycle.pdf to {out_dir}")

print("Rendered Figure 1 (Lifecycle Framework) successfully.")
