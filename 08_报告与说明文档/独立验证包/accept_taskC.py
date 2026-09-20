"""验收脚本：Task C（第三编码员 vs FINAL21）三方 κ 对比。

用法（收回验证者 coderD 表后）：
    python accept_taskC.py --coderD <验证者填好的csv路径>

- 读取 FINAL21（我们的最终表）与 coderC（杜俊涛老师编码）与 coderD（新验证者）。
- 收齐 coderC + coderD 后输出三方（FINAL21 vs C / FINAL21 vs D / C vs D）每维的
  一致率 + Cohen's κ。
- 只读不改，不写任何结论（仲裁由人做，脚本只给数）。
"""
from __future__ import annotations
import argparse, csv
import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE = "E:/论文1/08_报告与说明文档"
DIMS = ["D1_provenance", "D2_stage", "D3_inference_req", "D4_fidelity", "D5_harm"]
ALIAS = {"D1": "D1_provenance", "D2": "D2_stage", "D3": "D3_inference_req",
         "D4": "D4_fidelity", "D5": "D5_harm"}


def kappa(a, b):
    a = [str(x).strip() for x in a]
    b = [str(x).strip() for x in b]
    agree = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    k = cohen_kappa_score(a, b)
    return agree, k


def load_coderC():
    c = pd.read_csv(f"{BASE}/literature_coding_table_coderC_blank.csv",
                    usecols=["paper_id", "coderC_D1", "coderC_D2", "coderC_D3",
                             "coderC_D4", "coderC_D5"])
    return c.rename(columns={"coderC_D1": "D1_provenance", "coderC_D2": "D2_stage",
                             "coderC_D3": "D3_inference_req", "coderC_D4": "D4_fidelity",
                             "coderC_D5": "D5_harm"})[["paper_id"] + DIMS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coderD", required=True, help="验证者填好的 coderD csv")
    args = ap.parse_args()

    final = pd.read_csv(f"{BASE}/literature_coding_table_FINAL21.csv")[["paper_id"] + DIMS]
    coderC = load_coderC()
    coderD = pd.read_csv(args.coderD)[["paper_id"] + DIMS]

    # 对齐 paper_id 顺序
    pids = final["paper_id"].tolist()
    m = final.merge(coderC, on="paper_id", suffixes=("", "_C")).merge(
        coderD, on="paper_id", suffixes=("_final", "_D"))
    m = m.set_index("paper_id").loc[pids].reset_index()
    # 列名整理
    colmap = {}
    for d in DIMS:
        # final 列保留原名；coderC 是 d+"_C"；coderD 是 d+"_D"（取决于 merge suffixes）
        pass
    # 直接用三份独立 df 对齐最稳
    f = final.set_index("paper_id").loc[pids]
    c = coderC.set_index("paper_id").loc[pids]
    dd = coderD.set_index("paper_id").loc[pids]

    print(f"{'pairs':<16s} {'agree':>8s} {'kappa':>8s}")
    print("-" * 40)
    for d in DIMS:
        a1, k1 = kappa(f[d], c[d])
        a2, k2 = kappa(f[d], dd[d])
        a3, k3 = kappa(c[d], dd[d])
        print(f"[{d}]")
        print(f"  FINAL21 vs C:   {a1:7.1%}  {k1:+.3f}")
        print(f"  FINAL21 vs D:   {a2:7.1%}  {k2:+.3f}")
        print(f"  C       vs D:   {a3:7.1%}  {k3:+.3f}")

    # 分歧清单：FINAL21 vs D
    print("\n=== FINAL21 vs D 分歧清单 ===")
    for d in DIMS:
        diff = [(p, f.loc[p, d], dd.loc[p, d]) for p in pids if f.loc[p, d] != dd.loc[p, d]]
        if diff:
            print(f"[{d}] ({len(diff)}):")
            for p, a, b in diff:
                print(f"  {p:16s} final={a:22s} D={b}")


if __name__ == "__main__":
    main()
