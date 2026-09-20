"""Read-only audit of active evidence; writes only into this audit directory.

No model training, manuscript edits, or mutation of original result files.
The zero-bid calculation diagnoses a partial settlement metric, not a trading policy.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RES = ROOT / '06_实验结果与日志/constraint_round2'
ANA = RES / 'analysis'
CODE = ROOT / '05_实验代码与脚本/constraint_round2'
DATA = ROOT / '04_数据集/constraint_rebuild_v2'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tracked = list(ANA.glob('*.csv')) + [CODE / n for n in
        ['core.py', 'protocol.json', 'assess.py', 'patchtst_core.py',
         'evaluate_dynamic_price.py', 'evaluate_patchtst_settlement.py']]
    tracked += [RES / 'analysis_plan.json', RES / 'run_provenance.json']
    tracked += list((ROOT / '01_论文投稿材料/active_manuscript/sections').glob('*.tex'))
    tracked += [ROOT / '01_论文投稿材料/active_manuscript/main.tex']
    evidence = {'audit_utc': datetime.now(timezone.utc).isoformat(),
                'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in tracked}}

    dl = read_csv(ANA / 'dynamic_price_settlement_table2.csv')
    pt = read_csv(ANA / 'patchtst_settlement_table.csv')
    contrasts = []
    for site in ['kelmarsh', 'hill_of_towie']:
        for policy in ['uncalibrated', 'quantile_calibrated']:
            rows = {v['method']: v for v in dl if v['site'] == site and
                    v['policy'] == policy and v['period'] == 'all'}
            a, b = rows['distribution_f1.0_plain'], rows['none_f1.0_plain']
            item = dict(site=site, backbone='DLinear', policy=policy)
            for output, column in [('delta_dynamic_cost', 'cost_dyn_mean'),
                    ('delta_static_cost', 'cost_static_mean'),
                    ('delta_shortfall_gbp_turb_year', 'cashout_per_turb_yr_gbp_mean'),
                    ('delta_net_cashflow_gbp_turb_year', 'net_cashflow_per_turb_yr_gbp_mean')]:
                item[output] = float(a[column]) - float(b[column])
            contrasts.append(item)
    for site in ['kelmarsh', 'hill_of_towie', 'penmanshiel']:
        rows = {v['method']: v for v in pt if v['site'] == site and
                v['policy'] == 'quantile_calibrated'}
        a, b = rows['distribution_l0.01_f1.0_plain'], rows['none_l0.0_f1.0_plain']
        item = dict(site=site, backbone='PatchTST', policy='quantile_calibrated')
        for output, column in [('delta_dynamic_cost', 'cost_dyn_mean'),
                ('delta_shortfall_gbp_turb_year', 'shortfall_turb_yr_mean'),
                ('delta_net_cashflow_gbp_turb_year', 'net_cashflow_turb_yr_mean')]:
            item[output] = float(a[column]) - float(b[column])
        contrasts.append(item)
    pd.DataFrame(contrasts).to_csv(OUT / 'recomputed_contrasts.csv', index=False)
    evidence['contrasts'] = contrasts
    evidence['primary_endpoint'] = json.loads((RES / 'analysis_plan.json').read_text(encoding='utf-8'))
    evidence['primary_intervals'] = [v for v in read_csv(ANA / 'paired_cost_intervals.csv')
        if v['method'] == 'distribution_f1.0_plain' and v['policy'] == 'quantile_calibrated']

    corpus = read_csv(ANA / 'corpus46_scoping_ledger.csv')
    included = [v for v in corpus if v['inclusion_check'] == 'include']
    evidence['corpus_counts'] = {key: dict(Counter(v[key] for v in included)) for key in
        ['D1_provenance', 'D2_stage', 'D3_inference_req', 'D4_fidelity', 'ablation_type',
         'd5_negative_effect_reported', 'd5_harm_metric_reported']}
    evidence['corpus_internal_ablation_conflicts'] = [
        {k: v[k] for k in ['study_id', 'ablation_type', 'd5_ablation_strict', 'ablation_note']}
        for v in included if (v['ablation_type'] == 'module-ablation' and v['d5_ablation_strict'] == 'yes')
        or (v['ablation_type'] in ['both', 'constraint-on-off'] and v['d5_ablation_strict'] == 'no')]
    cons_path = ROOT / '08_报告与说明文档/文献盲审_20篇代表性文献最终仲裁共识表.csv'
    consensus = {v['study_id']: v for v in read_csv(cons_path)}
    conflicts = []
    for v in included:
        if v['study_id'] not in consensus:
            continue
        for a, b in [('D1_provenance', 'D1_provenance'), ('D2_stage', 'D2_stage'),
                     ('D3_inference_req', 'D3_inference_req'), ('D4_fidelity', 'D4_fidelity'),
                     ('ablation_type', 'd5_ablation_type')]:
            if v[a] != consensus[v['study_id']][b]:
                conflicts.append([v['study_id'], a, v[a], consensus[v['study_id']][b]])
    evidence['current_consensus_conflicts'] = conflicts
    evidence['training_manifests'] = {}
    evidence['checkpoint_counts'] = {}
    for site in ['kelmarsh', 'hill_of_towie', 'penmanshiel', 'smarteole']:
        evidence['checkpoint_counts'][site] = {backbone: len(list((RES / site / backbone).rglob('*.pt')))
                                               for backbone in ['dlinear', 'patchtst']}
        manifest = json.loads((RES / site / 'dlinear/seed42/manifest.json').read_text())
        evidence['training_manifests'][site] = manifest
        manifest['prepared_data_hash_matches'] = manifest['data_sha256'] == sha(DATA / f'{site}.csv')
        manifest['core_hash_matches'] = manifest['code_sha256'] == sha(CODE / 'core.py')
        manifest['protocol_hash_matches'] = manifest['protocol_sha256'] == sha(CODE / 'protocol.json')

    # Independent reconstruction of the SAME test-origin eligibility logic.
    # Evaluates a deliberately noncompetitive zero-bid diagnostic using real prices.
    prices = pd.read_csv(ROOT / '04_数据集/elexon_system_prices_2016_2024.csv')
    prices.index = pd.to_datetime(prices['startTime'], utc=True)
    prices = prices['systemBuyPrice']
    assert prices.index.is_unique
    zero_rows = []
    for site, turbines, rated in [('kelmarsh', 6, 12300.), ('hill_of_towie', 21, 48300.),
                                 ('penmanshiel', 14, 28700.)]:
        f = pd.read_csv(DATA / f'{site}.csv', index_col='time', parse_dates=True)
        f.index = pd.to_datetime(f.index, utc=True)
        assert f.index.is_unique
        assert (f.index.to_series().diff().dropna() == pd.Timedelta(minutes=10)).all()
        raw = f[['power_pu', 'wind_speed']].copy()
        raw['wind_speed'] /= 15.
        filled = raw.ffill(limit=3).to_numpy(np.float32)
        y = f['power_pu'].to_numpy(np.float32)
        ids = np.arange(36, len(f) - 36 + 1, 6)
        valid = np.array([np.isfinite(filled[i-36:i]).all() and np.isfinite(y[i:i+36]).all()
                          for i in ids])
        ids = ids[valid]
        ids = ids[(f.index[ids] >= pd.Timestamp('2017-01-01', tz='UTC')) &
                  (f.index[ids+35] < pd.Timestamp('2018-01-01', tz='UTC'))]
        assert len(ids) == evidence['training_manifests'][site]['windows']['test']
        target = (ids[:, None] + np.arange(6)).ravel()
        timeline = pd.DataFrame({'y': y[target], 'sp':
            (f.index[target] - pd.Timedelta(minutes=10)).floor('30min')})
        sp = timeline.groupby('sp')['y'].agg(['sum', 'size'])
        sp = sp[sp['size'] == 3].copy()
        sp['price'] = prices.reindex(sp.index)
        assert sp['price'].notna().all()
        sp['net'] = sp['price'] * sp['sum'] * rated / 6000.
        daily = sp.groupby(sp.index.strftime('%Y-%m-%d'))['net'].sum()
        origin_days = set(f.index[ids].strftime('%Y-%m-%d'))
        daily = daily[daily.index.isin(origin_days)]
        zero_rows.append(dict(site=site, test_origins=len(ids), complete_settlement_periods=len(sp),
                              scored_days=len(daily), zero_bid_shortfall_gbp=0.,
                              zero_bid_net_component_gbp_turb_year=float(daily.mean()*365/turbines)))
    evidence['zero_bid_metric_diagnostic'] = zero_rows
    pd.DataFrame(zero_rows).to_csv(OUT / 'zero_bid_metric_diagnostic.csv', index=False)
    evidence['source_hashes_unchanged_during_audit'] = all(
        sha(ROOT / path) == value for path, value in evidence['source_hashes'].items())
    (OUT / 'verified_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'contrasts': contrasts, 'zero_bid_diagnostic': zero_rows,
                      'ledger_conflicts': evidence['corpus_internal_ablation_conflicts'],
                      'consensus_conflicts': conflicts,
                      'sources_unchanged': evidence['source_hashes_unchanged_during_audit'],
                      'output': str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
