"""Inter-rater agreement calculation between coder_A (v3) and coder_C (independent coder).

Evaluates the 5 lifecycle taxonomy dimensions (D1 to D5) across 26 papers:
- D1: Provenance (intrinsic, historical-static, forecast-derived, learned-online, unreported)
- D2: Application stage (training-loss, post-hoc-correction, inference-architecture, data-generation, unreported)
- D3: Inference requirement (none, forecast-input, contemporaneous-state, latent-state, unreported)
- D4: Implementation fidelity (code-aligned, code-mismatch, no-code, parity-only, unverified)
- D5: Harm reporting (none-reported, ablation-absent, harm-quantified, substitution-tested)

Usage:
    python interrater_agreement_final.py
"""
from __future__ import annotations
import csv
from collections import Counter
import os

DIMENSIONS = [
    ("D1_provenance", "coderC_D1", "D1 Provenance"),
    ("D2_stage", "coderC_D2", "D2 Application Stage"),
    ("D3_inference_req", "coderC_D3", "D3 Inference Requirement"),
    ("D4_fidelity", "coderC_D4", "D4 Implementation Fidelity"),
    ("D5_harm", "coderC_D5", "D5 Harm Reporting"),
]

def cohen_kappa(a: list[str], b: list[str]):
    """Compute raw agreement and Cohen's Kappa between two raters."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return 1.0, 1.0
    agree = sum(1 for x, y in zip(a, b) if x == y)
    po = agree / n
    cats = sorted(set(a) | set(b))
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1.0 - pe)
    return po, kappa

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    a_path = os.path.join(base_dir, "literature_coding_table_v3_final.csv")
    c_path = os.path.join(base_dir, "literature_coding_table_coderC_blank.csv")
    if not os.path.exists(c_path):
        c_path = os.path.join(base_dir, "literature_coding_table_coderC_filled.csv")

    with open(a_path, encoding="utf-8-sig") as f:
        a_rows = {r["paper_id"]: r for r in csv.DictReader(f)}
    with open(c_path, encoding="utf-8-sig") as f:
        c_rows = {r["paper_id"]: r for r in csv.DictReader(f)}

    common_pids = [r["paper_id"] for r in csv.DictReader(open(a_path, encoding="utf-8-sig")) if r["paper_id"] in c_rows]
    n = len(common_pids)

    print("=" * 80)
    print(f"Inter-Rater Agreement Report: coder_A vs coder_C (N = {n} papers)")
    print("=" * 80)
    print(f"{'Dimension':<28s} {'Agreement':>10s} {'Cohen κ':>10s} {'Disagreements':>14s}")
    print("-" * 80)

    for a_col, c_col, label in DIMENSIONS:
        va = [a_rows[p][a_col].strip() for p in common_pids]
        vc = [c_rows[p][c_col].strip() for p in common_pids]
        po, kappa = cohen_kappa(va, vc)
        dis = [p for p in common_pids if a_rows[p][a_col].strip() != c_rows[p][c_col].strip()]
        print(f"{label:<28s} {po*100:>9.1f}% {kappa:>10.3f} {len(dis):>14d}")

    print("\n" + "=" * 80)
    print("D4 Schema Note (Semantic Alignment):")
    print("In v3 raw coding, coder_A used 'unverified' for uninspected papers while coder_C used 'no-code'.")
    va_d4_aligned = ['no-code' if a_rows[p]['D4_fidelity'].strip() == 'unverified' else a_rows[p]['D4_fidelity'].strip() for p in common_pids]
    vc_d4 = [c_rows[p]['coderC_D4'].strip() for p in common_pids]
    po_d4, kappa_d4 = cohen_kappa(va_d4_aligned, vc_d4)
    print(f"D4 (after aligning unverified -> no-code): Agreement = {po_d4*100:.1f}%, Cohen κ = {kappa_d4:.3f}")

    print("\n" + "=" * 80)
    print("Detailed Disagreements for Arbitration:")
    print("=" * 80)
    for a_col, c_col, label in DIMENSIONS:
        dis = [(p, a_rows[p][a_col].strip(), c_rows[p][c_col].strip()) for p in common_pids if a_rows[p][a_col].strip() != c_rows[p][c_col].strip()]
        if dis:
            print(f"\n[{label}] ({len(dis)} differences):")
            for pid, va, vc in dis:
                print(f"  - {pid:<16s}: coder_A = {va:<22s} | coder_C = {vc:<22s}")

if __name__ == "__main__":
    main()
