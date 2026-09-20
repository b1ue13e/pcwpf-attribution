from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RES = ROOT / '06_实验结果与日志/constraint_round2'
CODE = ROOT / '05_实验代码与脚本/constraint_round2'
DATA = ROOT / '04_数据集/constraint_rebuild_v2'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
ev = {}
t = pd.read_csv(RES/'analysis/dynamic_price_settlement_table2.csv')
t = t[(t.period=='all') & (t.policy=='quantile_calibrated')]
cols = ['site','method','cost_static_mean','cashout_per_turb_yr_gbp_mean','net_cashflow_per_turb_yr_gbp_mean']
ev['dlinear_calibrated'] = t[cols].to_dict('records')
p = pd.read_csv(RES/'analysis/patchtst_settlement_table.csv')
ev['patchtst_calibrated'] = p[p.policy=='quantile_calibrated'].to_dict('records')
c = pd.read_csv(RES/'analysis/corpus46_scoping_ledger.csv')
ev['corpus_rows'] = len(c)
c = c[c.inclusion_check=='include'].copy()
ev['corpus_included_rows'] = len(c)
ev['corpus_counts'] = {k:c[k].value_counts(dropna=False).to_dict() for k in ['D4_fidelity','ablation_type','d5_negative_effect_reported','d5_harm_metric_reported','d5_prior_substitution_tested']}
ev['ssrn_rows'] = c[c.doi.str.contains('ssrn',case=False,na=False)][['study_id','doi','title']].to_dict('records')
ev['manifests'] = []
for site in ['kelmarsh','hill_of_towie','penmanshiel']:
    m = json.loads((RES/site/'dlinear/seed42/manifest.json').read_text())
    ev['manifests'].append({'site':site,'windows':m['windows'],'data_hash_matches':m['data_sha256']==sha(DATA/f'{site}.csv'),'code_hash_matches':m['code_sha256']==sha(CODE/'core.py'),'protocol_hash_matches':m['protocol_sha256']==sha(CODE/'protocol.json'),'dlinear_checkpoints':len(list((RES/site/'dlinear').rglob('*.pt'))),'patchtst_checkpoints':len(list((RES/site/'patchtst').rglob('*.pt')))})
prices = pd.read_csv(ROOT/'04_数据集/elexon_system_prices_2016_2024.csv')
prices.index = pd.to_datetime(prices.startTime,utc=True)
assert prices.index.is_unique
price = prices.systemBuyPrice
ev['zero_bid_diagnostic'] = []
for site, nt, cap in [('kelmarsh',6,12300),('hill_of_towie',21,48300),('penmanshiel',14,28700)]:
    f = pd.read_csv(DATA/f'{site}.csv',index_col='time',parse_dates=True)
    f.index = pd.to_datetime(f.index,utc=True)
    assert f.index.is_unique
    assert (f.index.to_series().diff().dropna()==pd.Timedelta(minutes=10)).all()
    raw=f[['power_pu','wind_speed']].copy();raw.wind_speed/=15
    x=raw.ffill(limit=3).to_numpy(np.float32);y=f.power_pu.to_numpy(np.float32)
    ids=np.arange(36,len(f)-36+1,6)
    ok=np.array([np.isfinite(x[i-36:i]).all() and np.isfinite(y[i:i+36]).all() for i in ids])
    ids=ids[ok];ids=ids[(f.index[ids]>=pd.Timestamp('2017-01-01',tz='UTC')) & (f.index[ids+35]<pd.Timestamp('2018-01-01',tz='UTC'))]
    targets=(ids[:,None]+np.arange(6)).ravel()
    assert len(targets)==len(np.unique(targets))
    sp=(f.index[targets]-pd.Timedelta(minutes=10)).floor('30min')
    z=pd.DataFrame({'sp':sp,'y':y[targets]}).groupby('sp').agg(total=('y','sum'),n=('y','count'))
    z=z[z.n==3];z['price']=price.reindex(z.index)
    assert z.price.notna().all()
    z['net']=z.total*cap/6000*z.price
    daily=z.net.groupby(z.index.strftime('%Y-%m-%d')).sum()
    ev['zero_bid_diagnostic'].append({'site':site,'origins':len(ids),'complete_settlement_periods':len(z),'observed_days':len(daily),'zero_bid_shortfall':0,'zero_bid_net_component_per_turb_year':float(daily.mean()*365/nt)})
P=np.array([.99,.01]);Q=P[::-1];M=(P+Q)/2
kl=lambda a,b:float(np.sum(a*np.log(a/b)))
ev['js_diagnostic']={'standard_JS':.5*(kl(P,M)+kl(Q,M)),'reverse_JS':.5*(kl(M,P)+kl(M,Q)),'reverse_JS_swapped':.5*(kl(M,Q)+kl(M,P)),'ln2':float(np.log(2))}
(OUT/'verified_evidence.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in ev.items() if k not in ['patchtst_calibrated']},ensure_ascii=False,indent=2))
