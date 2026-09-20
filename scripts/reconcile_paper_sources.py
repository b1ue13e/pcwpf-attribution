from pathlib import Path
import pandas as pd

ROOT = Path(r'E:/论文1')
SRC = ROOT / '08_报告与说明文档/corpus48_编码_全维度集成版_杜俊涛.csv'
OUT = ROOT / '06_实验结果与日志/constraint_round2/analysis/corpus46_scoping_ledger.csv'
REPORT = ROOT / '08_报告与说明文档/canonical_ledger_reconciliation.md'

df = pd.read_csv(SRC, encoding='utf-8')
df = df[df['inclusion_check'].str.lower().eq('include')].copy()
updates = {
    'S15': {'D3_inference_req': 'contemporaneous-state'},
    'S18': {'D2_stage': 'inference-architecture', 'D3_inference_req': 'contemporaneous-state'},
    'S19': {'ablation_type': 'constraint-on-off'},
    'S27': {'D3_inference_req': 'forecast-input', 'ablation_type': 'module-ablation'},
    'S34': {'ablation_type': 'module-ablation'},
    'S39': {'D3_inference_req': 'forecast-input'},
}
for sid, vals in updates.items():
    for col, val in vals.items():
        df.loc[df.study_id.eq(sid), col] = val

df = df.sort_values('study_id').reset_index(drop=True)
df.to_csv(OUT, index=False, encoding='utf-8-sig')

def counts(col):
    return df[col].value_counts(dropna=False).to_dict()
lines = [
    '# Canonical literature ledger reconciliation', '',
    '- Canonical source: `corpus48_编码_全维度集成版_杜俊涛.csv`, filtered to `inclusion_check=include`.',
    '- Final included studies: **46** (48 ledger rows minus 2 excluded records).',
    '- Six field-level conflicts in the 20-study arbitration sample were resolved using the saved consensus table; this is a reconciliation of existing records, not a new independent full-corpus reread.',
    '', '## Counts used by the manuscript', '',
    f'- D1 provenance: `{counts("D1_provenance")}`',
    f'- D2 stage: `{counts("D2_stage")}`',
    f'- D3 inference requirement: `{counts("D3_inference_req")}`',
    f'- D4 fidelity: `{counts("D4_fidelity")}`',
    f'- Ablation type: `{counts("ablation_type")}`',
    f'- Negative-effect report: `{counts("d5_negative_effect_reported")}`',
    '', '## Publication-status note', '',
    'The ledger contains 31 journal records, 4 conference records, 10 records with SSRN/arXiv/working-paper identifiers, and 1 record without a DOI. These identifiers are not sufficient to assert peer-review status for every record; the manuscript therefore reports the 46-study cohort without a peer-reviewed/preprint breakdown until record-level status is verified.',
]
REPORT.write_text('\n'.join(lines), encoding='utf-8')
print(f'wrote {OUT} and {REPORT}')
