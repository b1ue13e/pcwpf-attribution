#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双人独立盲审 Cohen's Kappa 自动化计算与仲裁报告生成脚本
用于评估 Coder A (杜俊涛) 与 Coder B 在 20 篇代表性文献抽样上的双人盲审一致性。
"""

import sys
import os
import csv
import math
from collections import Counter

# 兼容 Windows GBK 控制台编码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def cohen_kappa(y1, y2, categories=None):
    """
    计算两个分类序列的 Cohen's Kappa 系数及其渐近标准误与 95% 置信区间
    """
    assert len(y1) == len(y2)
    n = len(y1)
    if n == 0:
        return 0.0, 0.0, (0.0, 0.0), 0.0
    
    if categories is None:
        categories = sorted(list(set(y1) | set(y2)))
        
    k = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        if a in cat_to_idx and b in cat_to_idx:
            matrix[cat_to_idx[a]][cat_to_idx[b]] += 1
            
    # 计算观测一致率 Po
    po = sum(matrix[i][i] for i in range(k)) / n
    
    # 计算边际概率与期望一致率 Pe
    row_sums = [sum(matrix[i][j] for j in range(k)) for i in range(k)]
    col_sums = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row_sums[i] * col_sums[i]) for i in range(k)) / (n * n)
    
    if abs(1.0 - pe) < 1e-9:
        kappa = float('nan')
        se = float('nan')
        ci = (float('nan'), float('nan'))
    else:
        kappa = (po - pe) / (1.0 - pe)
        # Fleiss / Cohen 渐近方差估计
        var = (po * (1 - po)) / (n * (1 - pe) ** 2) if n > 1 else 0.0
        se = math.sqrt(max(0.0, var))
        ci = (max(-1.0, kappa - 1.96 * se), min(1.0, kappa + 1.96 * se))
        
    return po, pe, kappa, se, ci

def kappa_qualitative(kappa):
    """Landis & Koch (1977) 标准分类解释"""
    if kappa >= 0.81:
        return "Almost Perfect (几乎完全一致)"
    elif kappa >= 0.61:
        return "Substantial (高度一致)"
    elif kappa >= 0.41:
        return "Moderate (中度一致)"
    elif kappa >= 0.21:
        return "Fair (一般一致)"
    elif kappa >= 0.00:
        return "Slight (轻微一致)"
    else:
        return "Poor (一致性极低/低于偶然)"

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    coder_a_path = os.path.join(base_dir, "coderA_杜俊涛_20篇基准对照表.csv")
    
    # 支持命令行指定 Coder B 的文件路径，默认查找同目录下的完成表或空白表
    if len(sys.argv) > 1:
        coder_b_path = sys.argv[1]
    else:
        possible_paths = [
            os.path.join(base_dir, "盲审填写表_CoderB_已完成.csv"),
            os.path.join(base_dir, "盲审填写表_CoderB_completed.csv"),
            os.path.join(base_dir, "盲审填写表_CoderB_空白表.csv")
        ]
        coder_b_path = None
        for p in possible_paths:
            if os.path.exists(p):
                coder_b_path = p
                break
                
    if not coder_b_path or not os.path.exists(coder_b_path):
        print(f"[错误] 未找到第二编码员的填写表格！请确认已保存为：{coder_b_path}")
        print("用法: python calculate_cohen_kappa.py [Coder_B_file.csv]")
        sys.exit(1)
        
    print(f"正在比对双人独立盲审结果:")
    print(f"  Coder A (第一编码员基准): {coder_a_path}")
    print(f"  Coder B (第二编码员结果): {coder_b_path}\n")
    
    with open(coder_a_path, "r", encoding="utf-8-sig") as f:
        rows_a = {r["study_id"]: r for r in csv.DictReader(f)}
        
    with open(coder_b_path, "r", encoding="utf-8-sig") as f:
        rows_b = {r["study_id"]: r for r in csv.DictReader(f)}
        
    # 共同审阅的文献 ID
    common_studies = [sid for sid in sorted(rows_a.keys()) if sid in rows_b]
    n_common = len(common_studies)
    
    # 待核验的核心分类字段
    fields_to_evaluate = [
        ("inclusion_check", "纳排资格 (Inclusion)"),
        ("D1_provenance", "D1: 参数来源 (Provenance)"),
        ("D2_stage", "D2: 施加阶段 (Application Stage)"),
        ("D3_inference_req", "D3: 推理依赖 (Inference Requirement)"),
        ("D4_fidelity", "D4: 代码保真度 (Fidelity)"),
        ("d5_ablation_type", "D5: 消融类型 (Ablation Type)"),
        ("d5_negative_reported", "D5: 负面效应报告 (Harm Reporting)")
    ]
    
    report_lines = []
    report_lines.append("# 双人独立盲审信度分析与仲裁总表 (Inter-Rater Reliability Report)")
    report_lines.append(f"\n- 样本容量: $N = {n_common}$ 篇代表性抽样文献")
    report_lines.append(f"- 评价编码员: Coder A (杜俊涛) 与 Coder B (独立第二编码员)")
    report_lines.append(f"- 统计指标: 观测一致率 ($P_o$), 偶然一致率 ($P_e$), Cohen's $\\kappa$ 及 95% 置信区间\n")
    
    report_lines.append("## 一、 核心维度信度统计汇总")
    report_lines.append("| 审计维度 | 样本数 $N$ | 观测一致率 $P_o$ | 偶然一致率 $P_e$ | Cohen's $\\kappa$ | 95% 置信区间 | 一致性评级 |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
    
    disagreements = []
    
    for f_key, f_name in fields_to_evaluate:
        vals_a = []
        vals_b = []
        for sid in common_studies:
            va = (rows_a[sid].get(f_key) or "").strip().lower()
            vb = (rows_b[sid].get(f_key) or "").strip().lower()
            if va and vb:
                vals_a.append(va)
                vals_b.append(vb)
                if va != vb:
                    disagreements.append({
                        "study_id": sid,
                        "title": rows_a[sid].get("title", ""),
                        "field": f_name,
                        "field_key": f_key,
                        "coder_a": va,
                        "evidence_a": rows_a[sid].get("evidence_quote", "") or rows_a[sid].get("page_section_evidence", ""),
                        "coder_b": vb,
                        "evidence_b": rows_b[sid].get("evidence_quote", "") or rows_b[sid].get("page_section_evidence", "")
                    })
                    
        po, pe, kappa, se, ci = cohen_kappa(vals_a, vals_b)
        rating = kappa_qualitative(kappa)
        report_lines.append(f"| **{f_name}** | {len(vals_a)} | {po*100:.1f}% | {pe*100:.1f}% | **{kappa:.3f}** | [{ci[0]:.3f}, {ci[1]:.3f}] | {rating} |")
        
    report_lines.append("\n---\n")
    report_lines.append("## 二、 编码分歧与仲裁对照表 (Disagreement Table for Arbitration)")
    if disagreements:
        report_lines.append(f"共检测到 **{len(disagreements)} 处**独立编码分歧，请在仲裁会议中根据原文依据逐项裁定：\n")
        report_lines.append("| 文献 ID | 分歧维度 | Coder A 判定与依据 | Coder B 判定与依据 | 仲裁决议建议 |")
        report_lines.append("| :---: | :--- | :--- | :--- | :--- |")
        for d in disagreements:
            report_lines.append(f"| **{d['study_id']}** | {d['field']} | **{d['coder_a']}**<br><small>“{d['evidence_a'][:80]}...”</small> | **{d['coder_b']}**<br><small>“{d['evidence_b'][:80]}...”</small> | 待讨论 |")
    else:
        report_lines.append("[提示] 全字段达到 100% 完全一致！无需额外召开仲裁会议。")
        
    out_md_path = os.path.join(base_dir, "double_coding_agreement_report.md")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print("\n".join(report_lines[:25]))
    print(f"\n[完成] 完整信度报告与仲裁表已生成: {out_md_path}")

if __name__ == "__main__":
    main()
