"""Arbitration of coder_A vs coder_B disagreements, per ObsAudit Protocol v1.

Root cause found: coder_A polluted `observability_level` with enforcement-mode
descriptions (e.g. 'soft distributional' instead of clean L1-L5). coder_B used
clean five-level labels. Arbitration rule:
  - observability_level : take coder_B clean label; where A has SUBSTANTIVE
    disagreement (MLPINN L1-vs-mixed, MMD L2-vs-prior), arbitrate by paper text.
  - enforcement_mode    : take coder_A (richer description), normalize to fixed vocabulary.
  - other fields        : take majority / coder_A when only wording differs.
Output: arbitrated coding table (final pilot).
"""
from __future__ import annotations
import csv

A = r"E:\论文1\08_报告与说明文档\literature_coding_table_v2.csv"
B = r"E:\论文1\08_报告与说明文档\literature_coding_table_coderB.csv"
OUT = r"E:\论文1\08_报告与说明文档\literature_coding_table_arbitrated.csv"


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return {r["paper_id"]: r for r in csv.DictReader(f)}


# 实质仲裁裁决（基于论文原文）
ARB_LEVEL = {
    "MLPINN": ("L1", "ARB: ML-PINN uses turbine intrinsic physics (PMSG/BEM) as simulation constraint; classify L1 (intrinsic), not mixed; forecast-dependence is model not constraint"),
    "MMD-PGF": ("L2", "ARB: MMD constrains joint (wind,power) built from HISTORICAL observed data at train-time; the joint itself is a distributional prior over contemporaneously-observed variables -> L2"),
    "TANG2025": ("L3", "ARB: cut-in/cut-out/rated are L1 but the regularized 'theoretical power curve' is fitted from historical+meteorological data (forecast-dependent proxy) -> dominant constraint L3"),
    "PCT": ("L2", "ARB: PCT constraint input = power-curve KDE + windspeed, contemporaneously observed at train-time -> L2 (not L3/L4); implementation defect is separate from observability"),
    "TgDPF": ("L2", "ARB: same as PCT family; power-curve prior built from historical observation -> L2"),
    "JARA2024": ("L4", "ARB: possible power is state-contingent latent (maintenance/curtailment states) -> L4"),
    "PHYCELL": ("L3", "ARB: PDE constraints on forecast wind field -> L3"),
    "WANG2026": ("L4", "ARB: available power under curtailment is state-contingent latent -> L4"),
}

# 归一化 enforcement_mode 词表
ENF_NORM = {
    "soft (distributional, added to MSE)": "distributional",
    "soft distributional; empirical: independence-joint marginal product": "distributional",
    "soft distributional (MMD regularization)": "distributional",
    "probabilistic (quantile reconstruction)": "distributional",
    "probabilistic white-box": "distributional",
    "soft (PDE residual)": "soft",
    "soft (regularization term)": "soft",
    "hybrid (PINN + Simulink)": "hybrid",
    "N/A": "N/A",
    "soft distributional": "distributional",
    "soft distributional (added to MSE)": "distributional",
    "distributional": "distributional",
    "soft": "soft",
    "hybrid": "hybrid",
}
# deployment_source_stated 归一化
DS_NORM = {"NWP-like forecast": "not stated", "uses curtailment labels": "stated",
           "uses labels": "stated", "not stated": "not stated", "stated": "stated",
           "partial": "partial", "N/A": "N/A", "spatial neighbors": "stated"}


def main():
    a = load(A); b = load(B)
    out_rows = []
    for pid in sorted(a):
        ra, rb = a[pid], b[pid]
        lvl = ARB_LEVEL.get(pid, (rb.get("observability_level", "UNKNOWN"), ""))[0]
        enf_raw = ra["enforcement_mode"]
        enf = ENF_NORM.get(enf_raw, enf_raw)
        ds_raw = ra["deployment_source_stated"]
        ds = DS_NORM.get(ds_raw, ds_raw)
        row = dict(ra)
        row["observability_level"] = lvl
        row["enforcement_mode"] = enf
        row["deployment_source_stated"] = ds
        row["coder"] = "arbitrated"
        row["notes"] = (ra.get("notes", "") + " | " + ARB_LEVEL.get(pid, ("", ""))[1]).strip(" |")
        out_rows.append(row)
    fields = list(out_rows[0].keys())
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"arbitrated table written: {OUT} ({len(out_rows)} rows)")
    # 快速报告仲裁后的 level 分布
    from collections import Counter
    print("observability_level 分布:", dict(Counter(r['observability_level'] for r in out_rows)))
    print("enforcement_mode 分布:", dict(Counter(r['enforcement_mode'] for r in out_rows)))


if __name__ == "__main__":
    main()
