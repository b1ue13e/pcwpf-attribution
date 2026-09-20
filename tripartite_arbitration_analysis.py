#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三方独立盲审（Coder A 杜俊涛, Coder B, Coder C）信度对比与仲裁分析脚本
"""

import os
import sys
import csv
import math
from collections import Counter

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def cohen_kappa(y1, y2, categories=None):
    assert len(y1) == len(y2)
    n = len(y1)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, (0.0, 0.0)
    if categories is None:
        categories = sorted(list(set(y1) | set(y2)))
    k = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        if a in cat_to_idx and b in cat_to_idx:
            matrix[cat_to_idx[a]][cat_to_idx[b]] += 1
    po = sum(matrix[i][i] for i in range(k)) / n
    row_sums = [sum(matrix[i][j] for j in range(k)) for i in range(k)]
    col_sums = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row_sums[i] * col_sums[i]) for i in range(k)) / (n * n)
    if abs(1.0 - pe) < 1e-9:
        # Constant label across all items: 0/0 is mathematically undefined (NaN)
        kappa = float('nan')
        se = float('nan')
        ci = (float('nan'), float('nan'))
    else:
        kappa = (po - pe) / (1.0 - pe)
        var = (po * (1 - po)) / (n * (1 - pe) ** 2) if n > 1 else 0.0
        se = math.sqrt(max(0.0, var))
        ci = (max(-1.0, kappa - 1.96 * se), min(1.0, kappa + 1.96 * se))
    return po, pe, kappa, se, ci

def fleiss_kappa(data_matrix, n_categories):
    """
    data_matrix: N x k 矩阵，每行为某一样本各类别被评定的次数之和
    n_categories: k
    """
    N = len(data_matrix)
    n = sum(data_matrix[0])  # 每样本的评分人数 (3人)
    if N == 0 or n <= 1:
        return 0.0, 0.0
    
    # pj: 各类别的总比例
    pj = [sum(data_matrix[i][j] for i in range(N)) / (N * n) for j in range(n_categories)]
    pe = sum(p ** 2 for p in pj)
    
    # Pi: 各样本的一致度
    Pi = []
    for i in range(N):
        s = sum(data_matrix[i][j] ** 2 for j in range(n_categories))
        pi = (s - n) / (n * (n - 1))
        Pi.append(pi)
    po = sum(Pi) / N
    
    if abs(1.0 - pe) < 1e-9:
        return float('nan'), po
    return (po - pe) / (1.0 - pe), po

def load_table(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return {r["study_id"]: r for r in csv.DictReader(f)}

def main():
    base_dir = r"e:\论文1"
    path_a = os.path.join(base_dir, "08_报告与说明文档", "coderA_杜俊涛_20篇基准对照表.csv")
    path_b = os.path.join(base_dir, "盲审填写表_CoderB_已填写.csv")
    path_c = os.path.join(base_dir, "盲审填写表_CoderC_已填写.csv")
    
    a_dict = load_table(path_a)
    b_dict = load_table(path_b)
    c_dict = load_table(path_c)
    
    studies = sorted(list(set(a_dict.keys()) & set(b_dict.keys()) & set(c_dict.keys())))
    print(f"三方共同编码文献数: N = {len(studies)}\n")
    
    fields = [
        ("inclusion_check", "纳排资格 (Inclusion)"),
        ("D1_provenance", "D1: 参数来源 (Provenance)"),
        ("D2_stage", "D2: 施加阶段 (Stage)"),
        ("D3_inference_req", "D3: 推理依赖 (Inference Req)"),
        ("D4_fidelity", "D4: 代码保真度 (Fidelity)"),
        ("d5_ablation_type", "D5: 消融类型 (Ablation Type)"),
        ("d5_negative_reported", "D5: 负面效应 (Harm Reporting)")
    ]
    
    print(f"{'审计维度':22s} | {'A vs B Kappa':12s} | {'A vs C Kappa':12s} | {'B vs C Kappa':12s} | {'三方Fleiss Kappa':15s}")
    print("-" * 80)
    
    disagreements = []
    
    for f_key, f_name in fields:
        va = [a_dict[s].get(f_key, "").strip().lower() for s in studies]
        vb = [b_dict[s].get(f_key, "").strip().lower() for s in studies]
        vc = [c_dict[s].get(f_key, "").strip().lower() for s in studies]
        
        po_ab, _, k_ab, _, _ = cohen_kappa(va, vb)
        po_ac, _, k_ac, _, _ = cohen_kappa(va, vc)
        po_bc, _, k_bc, _, _ = cohen_kappa(vb, vc)
        
        # Fleiss' Kappa
        cats = sorted(list(set(va) | set(vb) | set(vc)))
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        matrix = [[0] * len(cats) for _ in range(len(studies))]
        for i in range(len(studies)):
            matrix[i][cat_to_idx[va[i]]] += 1
            matrix[i][cat_to_idx[vb[i]]] += 1
            matrix[i][cat_to_idx[vc[i]]] += 1
        fk, po_f = fleiss_kappa(matrix, len(cats))
        
        def fmt_k(k, po=None):
            if math.isnan(k):
                if po is not None and abs(po - 1.0) < 1e-9:
                    return "NaN (Po=100%)"
                return "NaN"
            return f"{k:.3f}"
            
        print(f"{f_name:22s} | {fmt_k(k_ab, po_ab):>12s} | {fmt_k(k_ac, po_ac):>12s} | {fmt_k(k_bc, po_bc):>12s} | {fmt_k(fk, po_f):>15s}")
        
        # 记录分歧
        for s in studies:
            val_a = a_dict[s].get(f_key, "").strip().lower()
            val_b = b_dict[s].get(f_key, "").strip().lower()
            val_c = c_dict[s].get(f_key, "").strip().lower()
            if not (val_a == val_b == val_c):
                disagreements.append({
                    "study_id": s,
                    "field": f_name,
                    "field_key": f_key,
                    "A": val_a,
                    "B": val_b,
                    "C": val_c,
                    "notes_B": b_dict[s].get("coder_notes", ""),
                    "notes_C": c_dict[s].get("coder_notes", "")
                })
                
    # 2. 纳入文献子集 (Included Studies Subset, N = 18)
    studies_inc = [s for s in studies if a_dict[s].get("inclusion_check", "").strip().lower() == "include"]
    print("\n" + "=" * 80)
    print(f"纳入文献子集信度统计 (N = {len(studies_inc)} 篇，排除 S26 与 S42 负控制):")
    print(f"{'审计维度':22s} | {'A vs B Kappa':12s} | {'A vs C Kappa':12s} | {'B vs C Kappa':12s} | {'三方Fleiss Kappa':15s}")
    print("-" * 80)
    for f_key, f_name in fields[1:]:
        va = [a_dict[s].get(f_key, "").strip().lower() for s in studies_inc]
        vb = [b_dict[s].get(f_key, "").strip().lower() for s in studies_inc]
        vc = [c_dict[s].get(f_key, "").strip().lower() for s in studies_inc]
        
        po_ab, _, k_ab, _, _ = cohen_kappa(va, vb)
        po_ac, _, k_ac, _, _ = cohen_kappa(va, vc)
        po_bc, _, k_bc, _, _ = cohen_kappa(vb, vc)
        
        cats = sorted(list(set(va) | set(vb) | set(vc)))
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        matrix = [[0] * len(cats) for _ in range(len(studies_inc))]
        for i in range(len(studies_inc)):
            matrix[i][cat_to_idx[va[i]]] += 1
            matrix[i][cat_to_idx[vb[i]]] += 1
            matrix[i][cat_to_idx[vc[i]]] += 1
        fk, po_f = fleiss_kappa(matrix, len(cats))
        
        print(f"{f_name:22s} | {fmt_k(k_ab, po_ab):>12s} | {fmt_k(k_ac, po_ac):>12s} | {fmt_k(k_bc, po_bc):>12s} | {fmt_k(fk, po_f):>15s}")
        
    print("\n" + "=" * 80)
    print(f"三方分歧详细列表 (共 {len(disagreements)} 处):")
    for d in disagreements:
        print(f"[{d['study_id']}] 字段: {d['field']} -> A: {d['A']} | B: {d['B']} | C: {d['C']}")

if __name__ == "__main__":
    main()
