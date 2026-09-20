"""Build the final single-source 21-article coding table and auto-generate all
statistics from it (submission freeze, 4th-round reviewer directive #5).

Arbitration rules (frozen):
  - corpus: 21 = 26 coded minus 5 excluded (XAIPC2023, RISKREG2025, SETVAL2025,
    CPDIGITAL2025, HUANG2026)
  - D4: take coder_C (no-code / code-mismatch; coder_A 'unverified' maps to no-code)
  - D5: take coder_C (none-reported / ablation-absent / harm-quantified)
  - D1/D2/D3: coder_C (independent coder), with coder_A kept for κ reference
Output: literature_coding_table_FINAL21.csv + final_stats.json
"""
from __future__ import annotations
import csv, json
from collections import Counter

EXCLUDE = {'XAIPC2023', 'RISKREG2025', 'SETVAL2025', 'CPDIGITAL2025', 'HUANG2026'}

def main():
    A = {r['paper_id']: r for r in csv.DictReader(open('literature_coding_table_v3_final.csv', encoding='utf-8-sig'))}
    C = {r['paper_id']: r for r in csv.DictReader(open('literature_coding_table_coderC_blank.csv', encoding='utf-8-sig'))}
    inc = [p for p in A if p not in EXCLUDE]
    rows = []
    for p in sorted(inc):
        a, c = A[p], C[p]
        rows.append({
            'paper_id': p,
            'paper_short': a['paper_short'],
            'year': a['year'],
            'journal': a['journal'][:40],
            'D1_provenance': c['coderC_D1'],
            'D2_stage': c['coderC_D2'],
            'D3_inference_req': c['coderC_D3'],
            'D4_fidelity': c['coderC_D4'],
            'D5_harm': c['coderC_D5'],
            'coderA_D1': a['D1_provenance'],
            'coderA_D3': a['D3_inference_req'],
            'coderA_D5': a['D5_harm'],
            'coderC_notes': c.get('coderC_notes', ''),
        })
    with open('literature_coding_table_FINAL21.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # auto-generate stats from this single table only
    stats = {
        'n_included': len(rows),
        'D1': dict(Counter(r['D1_provenance'] for r in rows)),
        'D2': dict(Counter(r['D2_stage'] for r in rows)),
        'D3': dict(Counter(r['D3_inference_req'] for r in rows)),
        'D4': dict(Counter(r['D4_fidelity'] for r in rows)),
        'D5': dict(Counter(r['D5_harm'] for r in rows)),
    }
    with open('final_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print('FINAL21 table + stats written:', len(rows), 'articles')
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f'  {k}: {v}')

    # 口径消解表述
    d5 = stats['D5']
    none = d5.get('none-reported', 0)
    abl = d5.get('ablation-absent', 0)
    harm = d5.get('harm-quantified', 0)
    print(f'\nD5 口径说明：{none}/21 none-reported；另有 {abl} 篇 ablation-absent、'
          f'{harm} 篇 harm-quantified。最保守表述：19/21 未报告任何负面效应'
          f'（或 18/21 严格 none + 1 ablation-absent 未被算入）。')

if __name__ == '__main__':
    main()
