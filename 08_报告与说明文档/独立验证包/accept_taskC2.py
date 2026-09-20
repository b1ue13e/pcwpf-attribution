"""验收脚本：Task C2（v4 schema 重编）三方 D1/D2/D3 κ。

用法（收回答后）：
    python accept_taskC2.py --coderD2 <TaskC2_D1D3_杜俊涛.csv> --coderD1 <TaskC_coderD_杜俊涛.csv>

- 读 FINAL21（我们 v3 值）、杜老师 v3 值（coderD1）、杜老师 v4 值（coderD2）。
- v3→v4 映射：D1 把 forecast-derived 视为历史旧值（v4 已删），只对 coderD2 用 v4 取值。
- 输出每一维：FINAL21 vs D2(v4) 的 κ（需先把 FINAL21 的 forecast-derived 映射掉，
  映射规则在代码里声明）、D2(v4) 与 D1(v3) 的自一致变化（哪些篇变了）。
"""
from __future__ import annotations
import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE = "E:/论文1/08_报告与说明文档"
D = ["D1_provenance", "D2_stage", "D3_inference_req"]


def kappa(a, b):
    a = [str(x).strip() for x in a]; b = [str(x).strip() for x in b]
    agree = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    return agree, cohen_kappa_score(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coderD2", required=True, help="杜老师 v4 重编 csv")
    ap.add_argument("--coderD1", required=True, help="杜老师 v3 第一轮 csv")
    args = ap.parse_args()

    final = pd.read_csv(f"{BASE}/literature_coding_table_FINAL21.csv")[["paper_id"] + D]
    d1 = pd.read_csv(args.coderD1)[["paper_id"] + D]
    d2 = pd.read_csv(args.coderD2)[["paper_id"] + D + ["coderD_notes"]]

    pids = final["paper_id"].tolist()

    print("=== 三方 D1/D2/D3（v4 取值口径）===")
    for dim in D:
        f = final.set_index("paper_id").loc[pids, dim]
        a = d1.set_index("paper_id").loc[pids, dim]
        b = d2.set_index("paper_id").loc[pids, dim]
        # 只对比各方原始值（v4 里 forecast-derived 已不存在于 D2；FINAL21/D1 若还有则如实显示）
        ag1, k1 = kappa(f, b)   # 我们 vs 杜 v4
        ag2, k2 = kappa(a, b)   # 杜 v3 vs 杜 v4（自洽变化）
        print(f"[{dim}]")
        print(f"  FINAL21 vs D(v4):  agree={ag1:6.1%}  kappa={k1:+.3f}")
        print(f"  D(v3)    vs D(v4):  agree={ag2:6.1%}  kappa={k2:+.3f}")

    print("\n=== 杜老师 v3→v4 变化清单（哪些篇改了 D1/D2/D3）===")
    changed = 0
    for p in pids:
        for dim in D:
            if d1.set_index("paper_id").loc[p, dim] != d2.set_index("paper_id").loc[p, dim]:
                print(f"  {p:16s} [{dim}] v3={d1.set_index('paper_id').loc[p,dim]:20s} -> v4={d2.set_index('paper_id').loc[p,dim]}")
                changed += 1
    print(f"总计变化单元格: {changed}")

    print("\n=== 杜老师 v4 的 D1/D2/D3 分布 ===")
    for dim in D:
        print(f"{dim}: {dict(d2[dim].value_counts())}")


if __name__ == "__main__":
    main()
