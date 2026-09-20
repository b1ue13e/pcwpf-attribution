# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

df_a = pd.read_csv('e:/论文1/08_报告与说明文档/coderA_杜俊涛_20篇基准对照表.csv').sort_values('study_id').reset_index(drop=True)
df_b = pd.read_csv('e:/论文1/盲审填写表_CoderB_已填写.csv').sort_values('study_id').reset_index(drop=True)
df_c = pd.read_csv('e:/论文1/盲审填写表_CoderC_已填写.csv').sort_values('study_id').reset_index(drop=True)

# Build Final Consensus DataFrame based on arbitration decisions
consensus = df_c.copy()

# Ensure excluded rows S26 and S42 are clean 'n/a'
for col in ['D1_provenance', 'D2_stage', 'D3_inference_req', 'D4_fidelity', 'd5_ablation_type', 'd5_negative_reported']:
    consensus.loc[consensus['inclusion_check'] == 'exclude', col] = 'n/a'

consensus_path = 'e:/论文1/08_报告与说明文档/文献盲审_20篇代表性文献最终仲裁共识表.csv'
consensus.to_csv(consensus_path, index=False, encoding='utf-8-sig')
print(f'Final consensus saved to: {consensus_path}')

dimensions = [
    ('inclusion_check', '纳排资格 (Inclusion Check)'),
    ('D1_provenance', 'D1: 参数来源 (Provenance)'),
    ('D2_stage', 'D2: 施加阶段 (Application Stage)'),
    ('D3_inference_req', 'D3: 推理依赖 (Inference Dependency)'),
    ('D4_fidelity', 'D4: 代码保真度 (Code Fidelity)'),
    ('d5_ablation_type', 'D5: 消融类型 (Ablation Type)'),
    ('d5_negative_reported', 'D5: 负面效应报告 (Harm Reporting)')
]

def clean_val(x):
    if pd.isna(x):
        return 'n/a'
    x = str(x).strip().lower()
    if x in ['', 'nan']:
        return 'n/a'
    return x

def fleiss_kappa(ratings, n_categories):
    n_subjects, _ = ratings.shape
    n_raters = ratings.sum(axis=1)[0]
    p_j = ratings.sum(axis=0) / (n_subjects * n_raters)
    P_i = ((ratings * ratings).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e_bar = (p_j * p_j).sum()
    if 1.0 - P_e_bar == 0:
        return 1.0
    return (P_bar - P_e_bar) / (1.0 - P_e_bar)

results = []

for field, desc in dimensions:
    vals_a = df_a[field].apply(clean_val).values
    vals_b = df_b[field].apply(clean_val).values
    vals_c = df_c[field].apply(clean_val).values
    
    k_ab = cohen_kappa_score(vals_a, vals_b)
    k_ac = cohen_kappa_score(vals_a, vals_c)
    k_bc = cohen_kappa_score(vals_b, vals_c)
    
    categories = sorted(list(set(vals_a) | set(vals_b) | set(vals_c)))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    counts = np.zeros((len(vals_a), len(categories)))
    for i in range(len(vals_a)):
        counts[i, cat_to_idx[vals_a[i]]] += 1
        counts[i, cat_to_idx[vals_b[i]]] += 1
        counts[i, cat_to_idx[vals_c[i]]] += 1
    k_fleiss = fleiss_kappa(counts, len(categories))
    
    # Also calculate on N=18 included studies
    inc_mask = (consensus['inclusion_check'] == 'include').values
    if field != 'inclusion_check':
        k_bc_inc = cohen_kappa_score(vals_b[inc_mask], vals_c[inc_mask])
    else:
        k_bc_inc = k_bc
        
    results.append({
        'Dimension': desc,
        'A vs B': round(k_ab, 3),
        'A vs C': round(k_ac, 3),
        'B vs C (全量N=20)': round(k_bc, 3),
        'B vs C (纳入N=18)': round(k_bc_inc, 3),
        '三方 Fleiss': round(k_fleiss, 3)
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))