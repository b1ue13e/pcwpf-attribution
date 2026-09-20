"""Inter-rater agreement between coder_A (v2) and coder_B on 7 judgment fields.

Fields: observability_level, oracle_leakage, deployment_source_stated,
        enforcement_mode, harm_reported, substitution_test, code_available.
Reports per-field Cohen's kappa + raw agreement, and flags disagreements.
"""
from __future__ import annotations
import csv
from collections import Counter


def load(path):
    rows = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows[r["paper_id"]] = r
    return rows


FIELDS = ["observability_level", "oracle_leakage", "deployment_source_stated",
          "enforcement_mode", "harm_reported", "substitution_test", "code_available"]


def cohen_kappa(a, b):
    """Cohen's kappa for two categorical sequences (list-aligned)."""
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for c in cats:
        pa = a.count(c) / n
        pb = b.count(c) / n
        pe += pa * pb
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    a = load(r"E:\论文1\08_报告与说明文档\literature_coding_table_v2.csv")
    b = load(r"E:\论文1\08_报告与说明文档\literature_coding_table_coderB.csv")
    common = sorted(set(a) & set(b))
    print(f"共同 paper_id: {len(common)}")
    print(f"{'字段':28s} {'一致率':>6s} {'kappa':>8s}  不一致示例")
    all_disagreements = []
    for f in FIELDS:
        va = [a[p][f] for p in common]
        vb = [b[p][f] for p in common]
        agree = sum(1 for x, y in zip(va, vb) if x == y)
        rate = agree / len(common)
        kappa = cohen_kappa(va, vb)
        dis = [(p, a[p][f], b[p][f]) for p in common if a[p][f] != b[p][f]]
        all_disagreements += dis
        dis_str = "; ".join(f"{p}:A={x}/B={y}" for p, x, y in dis[:4])
        print(f"{f:28s} {rate:6.2%} {kappa:8.3f}  {dis_str}")
    print(f"\n总不一致判定数: {len(all_disagreements)} （{len(common)} 篇 × {len(FIELDS)} 字段）")
    # 报告需要仲裁的字段
    print("\n=== 需仲裁的分歧 ===")
    for p, x, y in all_disagreements:
        print(f"  {p}: A={x} vs B={y}")


if __name__ == "__main__":
    main()
