"""Seed summaries and paired moving-block uncertainty; no seed-as-site inflation."""
from pathlib import Path
import argparse,json,re
import numpy as np
import pandas as pd

def summarize(root):
    root=Path(root);out=root/'analysis';out.mkdir(exist_ok=True)
    files=list(root.glob('*/*/seed*/test_metrics.csv'))+list(root.glob('classical/test_metrics.csv'))+list(root.glob('dlinear/test_metrics.csv'))
    metric_file_count=len(files)
    m=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    m['method_group']=m.method.str.replace(r'_l[0-9.]+_f','_f',regex=True)
    metrics=['rmse_pu','mae_pu','p95_abs_error_pu','nameplate_exceedance_rate','cost_ratio2']
    table=m.groupby(['site','backbone','method_group','scenario','clipped','period'])[metrics].agg(['mean','std']).reset_index()
    table.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in table.columns]
    table.to_csv(out/'summary.csv',index=False)
    clean=table[(table.scenario=='clean')&(table.period=='all')&(table.clipped==True)]
    clean.to_csv(out/'clean_clipped_summary.csv',index=False)
    rows=[]
    # Read one site/backbone at a time. Average optimization replicates within day.
    for site in ['kelmarsh','penmanshiel','smarteole']:
        for kind in ['linear','gru']:
            files=list((root/site/kind).glob('seed*/daily_metrics.csv'))
            if not files:continue
            d=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
            d['method_group']=d.method.str.replace(r'_l[0-9.]+_f','_f',regex=True)
            d=d[d.scenario.isin(['clean','power_gaussian_005','wind_gaussian_05ms'])]
            daily=d.groupby(['method_group','scenario','clipped','day'])[['mse','cost_ratio2','n_origins']].mean().reset_index()
            for (scenario,clip),g in daily.groupby(['scenario','clipped']):
                baseline=g[g.method_group=='none_f1.0_plain'].set_index('day')
                for method,h in g.groupby('method_group'):
                    if method=='none_f1.0_plain':continue
                    a=h.set_index('day').join(baseline,lsuffix='_a',rsuffix='_b').dropna()
                    n=len(a)
                    assert np.allclose(a.n_origins_a,a.n_origins_b),'Mismatched forecast origins'
                    rng=np.random.default_rng(9207);block=min(7,n);draw=rng.integers(0,n,size=(1000,int(np.ceil(n/block))))
                    ix=((draw[:,:,None]+np.arange(block))%n).reshape(1000,-1)[:,:n]
                    w=a.n_origins_a.to_numpy();ea=a.mse_a.to_numpy();eb=a.mse_b.to_numpy()
                    delta=np.sqrt((ea[ix]*w[ix]).sum(1)/w[ix].sum(1))-np.sqrt((eb[ix]*w[ix]).sum(1)/w[ix].sum(1))
                    point=float(np.sqrt(np.average(ea,weights=w))-np.sqrt(np.average(eb,weights=w)))
                    cost=((a.cost_ratio2_a-a.cost_ratio2_b).to_numpy()[ix]*w[ix]).sum(1)/w[ix].sum(1)
                    rows.append({'site':site,'backbone':kind,'method_group':method,'scenario':scenario,'clipped':clip,
                        'days':n,'block_days':block,'delta_rmse_pu':point,'ci95_low':float(np.quantile(delta,.025)),
                        'ci95_high':float(np.quantile(delta,.975)),'cost_ratio2_delta':float(np.average(a.cost_ratio2_a-a.cost_ratio2_b,weights=w)),
                        'cost_ci95_low':float(np.quantile(cost,.025)),'cost_ci95_high':float(np.quantile(cost,.975)),
                        'interpretation':'pointwise exploratory CI; not multiplicity adjusted; negative favors treatment'})
    pd.DataFrame(rows).to_csv(out/'paired_7day_block_bootstrap.csv',index=False)
    (out/'analysis_manifest.json').write_text(json.dumps({'metric_files':metric_file_count,'total_metric_rows':len(m),
        'neural_training_completed':len(list(root.glob('*/*/seed*/TRAINING_COMPLETE'))),
        'neural_evaluation_completed':len(list(root.glob('*/*/seed*/EVALUATION_COMPLETE'))),
        'classical_complete':(root/'classical/COMPLETE').exists(),
        'uncertainty':'circular 7-day blocks on available scored days, averaged over optimization seeds; pointwise exploratory',
        'caveat':'missing days can make a block span more than 7 calendar days; not population-level field generalization'},indent=2))
    print(clean[['site','backbone','method_group','rmse_pu_mean']].to_string(index=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();summarize(a.root)
