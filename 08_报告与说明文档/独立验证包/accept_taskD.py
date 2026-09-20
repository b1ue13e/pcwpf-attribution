"""验收脚本：Task D（PRISMA 排除）——验证者判定 vs 我们 title 规则。

用法：
    python accept_taskD.py --screener <验证者填好的 screening csv>

- 读取验证者的 screening 表（72 行，decision 列 include/exclude/need-abstract）
  与我们的 prisma_ledger.csv（source=openalex；assessed→include，excluded-title→exclude）。
- 输出一致率 + 分歧清单 + need-abstract 统计。
- 只读不改；仲裁由人做。
"""
from __future__ import annotations
import argparse
import pandas as pd

BASE = "E:/论文1"


def load_ours():
    led = pd.read_csv(f"{BASE}/tmp/prisma_ledger.csv")
    oa = led[led["source"] == "openalex"].copy()
    oa["ours_decision"] = oa["stage"].map({"assessed": "include", "excluded-title": "exclude"})
    return oa[["odoi", "title", "ours_decision"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screener", required=True, help="验证者填好的 screening csv")
    args = ap.parse_args()

    ours = load_ours()
    theirs = pd.read_csv(args.screener)

    # 按 odoi 对齐（验证者表 odoi 列）；ours 保留 title，theirs 只取 decision 列
    ours = ours.rename(columns={"odoi": "odoi_key"})
    theirs = theirs.rename(columns={"odoi": "odoi_key"})
    m = ours.merge(theirs[["odoi_key", "decision", "decision_basis",
                           "exclusion_reason", "notes"]], on="odoi_key", how="left")

    # 只统计两者都有判定且非 need-abstract 的行
    judged = m[(m["decision"].isin(["include", "exclude"])) & (m["ours_decision"].notna())]
    agree = (judged["ours_decision"] == judged["decision"]).mean()
    print(f"可判定行数: {len(m)}")
    print(f"双方都有判定: {len(judged)}")
    print(f"一致率: {agree:.1%}")

    need_abstract = m[m["decision"] == "need-abstract"]
    print(f"验证者标 need-abstract: {len(need_abstract)} 篇")

    print("\n=== 分歧清单（我们 vs 验证者）===")
    diff = judged[judged["ours_decision"] != judged["decision"]]
    for _, r in diff.iterrows():
        print(f"  [{r['ours_decision']:>8} vs {r['decision']:<8}] {r['title'][:70]}")

    print("\n按验证者判定的分布:")
    print(m["decision"].value_counts().to_string())


if __name__ == "__main__":
    main()
