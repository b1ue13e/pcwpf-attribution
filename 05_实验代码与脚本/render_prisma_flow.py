"""Render PRISMA-ScR flow diagram as PDF/PNG (matplotlib, publication quality)."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

fig, ax = plt.subplots(figsize=(8.5, 10.5), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis("off")

def box(x, y, w, h, text, fc="#eef4fb", ec="#2b6cb0", fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#1a202c", linespacing=1.25)

def arrow(x_start, y_start, x_end, y_end):
    ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1.8, mutation_scale=15))

# 1. Identification
box(0.8, 12.0, 8.4, 1.4, 
    "IDENTIFICATION\nRecords identified via database searching:\nOpenAlex search (4 queries, 2023–2026): n = 72\nDomain web search queries: n = 17\nTotal raw candidate records: N = 89",
    fc="#ebf8ff", ec="#3182ce", bold=True)

arrow(5.0, 12.0, 5.0, 11.2)

# 2. Screening
box(0.8, 9.8, 5.2, 1.4, 
    "SCREENING\nTitle-level screened records\n(OpenAlex unique records)\n(n = 72)",
    fc="#f7fafc", ec="#4a5568")

arrow(5.0, 9.8, 5.0, 8.8)

# Exclusion 1
arrow(6.0, 10.5, 6.8, 10.5)
box(6.8, 9.8, 2.6, 1.4, 
    "Excluded (n = 19):\n• Non-forecasting (9)\n• Maintenance/ctrl (9)\n• No physics (1)",
    fc="#fff5f5", ec="#e53e3e", fs=9.5)

# 3. Deduplication
box(0.8, 7.4, 5.2, 1.4, 
    "DEDUPLICATION\nCandidate studies assessed (n = 53)\nMerged 5 preprint/journal pairs\nUnique studies in ledger: N = 48",
    fc="#f7fafc", ec="#4a5568")

arrow(5.0, 7.4, 5.0, 6.4)

# 4. Eligibility Check
box(0.8, 5.0, 5.2, 1.4, 
    "ELIGIBILITY\nFull-text assessed for eligibility\n(Formal inclusion criteria applied)\n(N = 48)",
    fc="#f7fafc", ec="#4a5568")

arrow(5.0, 5.0, 5.0, 3.8)

# Exclusion 2
arrow(6.0, 5.7, 6.8, 5.7)
box(6.8, 5.0, 2.6, 1.4, 
    "Excluded (n = 2):\n• S26: Wind speed only\n  (no power constraint)\n• S42: Heuristic only\n  (no explicit equation)",
    fc="#fff5f5", ec="#e53e3e", fs=9.5)

# 5. Included
box(0.8, 2.4, 8.4, 1.4, 
    "INCLUDED IN SCOPING AUDIT\nFinal coded full-text corpus\n(N = 46 independent studies, 2023–2026)\nAll studies coded across lifecycle dimensions D1–D5",
    fc="#f0fff4", ec="#38a169", fs=11, bold=True)

plt.tight_layout()
out_dir = Path("03_图表与绘图素材")
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "prisma_flow_diagram.pdf", bbox_inches="tight", facecolor="white")
plt.savefig(out_dir / "prisma_flow_diagram.png", bbox_inches="tight", facecolor="white")
print("Rendered PRISMA flow diagram (N=46) to PDF and PNG successfully.")
