"""验收脚本：第二批 D1/D2/D3 全文级 + ablation 语义复核。

用法（收回答后）：
    python accept_taskC4.py --coderV2 <第二批csv> --coderV1 <第一批corpus48_编码_杜俊涛.csv>

- 对比 v1（前置预编）vs v2（全文级）的 D1/D2/D3 变动数 + 每维分布。
- 统计 ablation_type 分布，标出 module-ablation 需要改 d5_ablation_present 的篇目。
- 只读不改。
"""
from __future__ import annotations
import argparse
import pandas as pd

D = ["D1_provenance", "D2_stage", "D3_inference_req"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coderV2", required=True)
    ap.add_argument("--coderV1", required=True)
    args = ap.parse_args()

    v1 = pd.read_csv(args.coderV1)
    v2 = pd.read_csv(args.coderV2)
    m = v1.merge(v2, on="study_id", suffixes=("_v1", "_v2"))
    print(f"合并 {len(m)} 篇")

    print("\n=== D1/D2/D3 预编(v1) vs 全文级(v2) 变动 ===")
    for dim in D:
        a = m[dim + "_v1"].astype(str)
        b = m[dim + "_v2"].astype(str)
        changed = (a != b).sum()
        print(f"{dim:20s} 变动 {changed}/{len(m)}")
        # 分布对比
        print(f"    v1: {dict(a.value_counts())}")
        print(f"    v2: {dict(b.value_counts())}")

    print("\n=== v2 全文级 D1/D2/D3 最终分布 ===")
    for dim in D:
        print(f"{dim}: {dict(m[dim+'_v2'].value_counts())}")

    print("\n=== ablation_type 分布（语义边界复核）===")
    if 'ablation_type' in v2.columns:
        print(dict(v2['ablation_type'].value_counts()))
        ma = v2[v2['ablation_type'].isin(['module-ablation'])]
        print(f"module-ablation（需把 d5_ablation_present 从 yes 改 no）: {len(ma)} 篇")
        for _, r in ma.iterrows():
            print(f"  {r['study_id']}: {str(r['title'])[:60]}")
    else:
        print("无 ablation_type 列")


if __name__ == "__main__":
    main()
