"""Read current revision and save compact, reproducible audit evidence; no training."""
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
A = ROOT / '06_实验结果与日志/constraint_round2/analysis'
M = ROOT / '01_论文投稿材料/active_manuscript'
R = ROOT / '08_报告与说明文档'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

ledger = pd.read_csv(A / 'corpus46_scoping_ledger.csv', keep_default_na=False)
inc = ledger[ledger.inclusion_check.eq('include')]
arb = pd.read_csv(R / '文献盲审_20篇代表性文献最终仲裁共识表.csv', keep_default_na=False)
fields = [('D1_provenance', 'D1_provenance'), ('D2_stage', 'D2_stage'),
          ('D3_inference_req', 'D3_inference_req'), ('D4_fidelity', 'D4_fidelity'),
          ('ablation_type', 'd5_ablation_type')]
conflicts = []
for row in arb.to_dict('records'):
    found = ledger[ledger.study_id.eq(row['study_id'])]
    if len(found) != 1:
        continue
    current = found.iloc[0]
    for left, right in fields:
        if current[left] != row[right]:
            conflicts.append({'study_id': row['study_id'], 'field': left,
                             'table_ledger': current[left], 'arbitration': row[right]})

retrieval = pd.read_csv(R / 'corpus_rebuild_20260907/retrieval_records.csv')
screening = pd.read_csv(R / 'corpus_rebuild_20260907/screening_pending.csv')
query = pd.read_csv(R / 'corpus_rebuild_20260907/query_log.csv')

patch = pd.read_csv(A / 'patchtst_settlement_table.csv')
patch_deltas = []
for site in patch.site.unique():
    d = patch[patch.site.eq(site) & patch.policy.eq('quantile_calibrated')].set_index('method')
    cols = ['cost_dyn_mean', 'shortfall_turb_yr_mean', 'net_cashflow_turb_yr_mean']
    delta = d.loc['distribution_l0.01_f1.0_plain', cols] - d.loc['none_l0.0_f1.0_plain', cols]
    patch_deltas.append({'site': site, **{k: float(v) for k,v in delta.items()}})

boot = pd.read_csv(A / 'dynamic_price_paired_intervals.csv')
kel = boot[boot.site.eq('kelmarsh') & boot.method.eq('distribution_f1.0_plain') &
           boot.policy.eq('quantile_calibrated') &
           boot.contrast_type.eq('vs_unconstrained_dlinear_same_policy')].iloc[0]
kel_cols = ['delta_cashout_per_turb_yr_gbp', 'ci95_cashout_low', 'ci95_cashout_high']

# Exact uniform-distribution risks, retaining all fractions until reporting.
def urisk(a, b, p):
    assert a <= p <= b
    return (2*(p-a)**2 + (b-p)**2) / (2*(b-a))

ab = [(F(1,5), F(3,5)), (F(3,10), F(9,10))]
quantiles = [a+(b-a)/3 for a,b in ab]
bayes = sum(urisk(a,b,q)/2 for (a,b),q in zip(ab,quantiles))
math_results = {}
for name, forecasts in [('A_mean', [F(2,5), F(3,5)]),
                        ('B_low_spread_shift', [F(7,20), F(3,5)]),
                        ('B_high_spread_shift', [F(2,5), F(11,20)])]:
    weights = [1/(b-a) for a,b in ab]
    delta = sum(w*(q-f) for w,q,f in zip(weights,quantiles,forecasts))/sum(weights)
    risk = sum(urisk(a,b,f+delta)/2 for (a,b),f in zip(ab,forecasts))
    math_results[name] = {'delta': float(delta), 'risk': float(risk),
                          'excess_over_bayes': float(risk-bayes), 'risk_fraction': str(risk)}

keys = re.findall(r'(?m)^@\w+\s*\{\s*([^,]+)', (M/'refs.bib').read_text(encoding='utf-8'))
log = (M/'main.log').read_text(encoding='utf-8', errors='replace')
files = list((M/'sections').glob('*.tex')) + [M/'main.tex', M/'refs.bib',
       ROOT/'05_实验代码与脚本/constraint_round2/evaluate_patchtst_settlement.py',
       ROOT/'05_实验代码与脚本/constraint_round2/export_latex_tables.py',
       A/'patchtst_settlement_table.csv', A/'dynamic_price_settlement_table2.csv',
       A/'dynamic_price_paired_intervals.csv', A/'corpus46_scoping_ledger.csv',
       R/'corpus_rebuild_20260907/flow_status.json']

report = {
 'scope': '2026-09-08 16:14 source revision, with evaluation/table/PDF outputs rerun in review',
 'ledger': {'rows': len(ledger), 'included': len(inc), 'unique_study_ids': ledger.study_id.nunique(),
            'sha256_equals_old_corpus48': sha(A/'corpus46_scoping_ledger.csv') == sha(R/'corpus48_编码_全维度集成版_杜俊涛.csv'),
            'counts': {c: inc[c].value_counts().to_dict() for c in
              ['D1_provenance','D2_stage','D3_inference_req','D4_fidelity','ablation_type','d5_negative_effect_reported']},
            'conflicts_against_saved_arbitration': conflicts},
 'retrieval': {'raw_rows': len(retrieval), 'screening_rows': len(screening),
               'screening_status': screening.screening_decision.value_counts().to_dict(),
               'sum_query_hits': int(query.records.sum()),
               'current_summary_json': json.loads((R/'corpus_rebuild_20260907/flow_status.json').read_text())},
 'patchtst_distribution_minus_unconstrained_calibrated': patch_deltas,
 'kelmarsh_dlinear_calibrated_contrast': {c: float(kel[c]) for c in kel_cols},
 'appendix_exact_risk_checks': math_results,
 'bibtex_duplicate_keys': {k:v for k,v in Counter(keys).items() if v>1},
 'compile_log': {'overfull_hbox':log.count('Overfull \\hbox'),
                 'overfull_vbox':log.count('Overfull \\vbox'),
                 'undefined_mentions':log.lower().count('undefined'),
                 'missing_characters':log.count('Missing character:')},
 'source_hashes': {str(p.relative_to(ROOT)):sha(p) for p in files}
}
(OUT/'verified_revision.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'},ensure_ascii=False,indent=2))
