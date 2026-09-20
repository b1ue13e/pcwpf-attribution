from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

def main(root):
    root=Path(root);out=root/'analysis';out.mkdir(exist_ok=True)
    files=list(root.glob('*/assessment/metrics.csv'));m=pd.concat([pd.read_csv(p) for p in files])
    m['group']=m.method.str.replace(r'_l[0-9.]+_f','_f',regex=True)
    table=m.groupby(['site','group','scenario','ratio','policy','period'])[['rmse_pu','cost']].agg(['mean','std']).reset_index()
    table.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in table.columns]
    table.to_csv(out/'summary.csv',index=False);rows=[]
    for p in root.glob('*/assessment/daily.csv'):
        d=pd.read_csv(p);site=d.site.iloc[0];d=d[(d.scenario=='clean')&(d.ratio==2)].copy()
        d['group']=d.method.str.replace(r'_l[0-9.]+_f','_f',regex=True)
        seeds=sorted(d.seed.unique());calendar=pd.date_range(d.day.min(),d.day.max(),freq='D').strftime('%Y-%m-%d');N=len(calendar)
        def arr(g,col):return g.pivot(index='seed',columns='day',values=col).reindex(index=seeds,columns=calendar).fillna(0).to_numpy()
        for policy in ['uncalibrated','quantile_calibrated']:
            q=d[d.policy==policy];base=q[q.group=='none_f1.0_plain'];w=arr(base,'n_origins')
            basecost=arr(base,'cost')
            for method in ['bounds_f1.0_plain','curve_f1.0_plain','distribution_f1.0_plain','ridge','persistence']:
                a=q[q.group==method];assert len(a)==len(base)
                assert np.allclose(arr(a,'n_origins'),w)
                delta=arr(a,'cost')-basecost
                rng=np.random.default_rng(20907);R=2000
                ss=rng.integers(0,len(seeds),(R,len(seeds)))
                begins=rng.integers(0,N,(R,int(np.ceil(N/7))))
                ii=((begins[:,:,None]+np.arange(7))%N).reshape(R,-1)[:,:N]
                ww=w[ss[:,:,None],ii[:,None,:]];dd=delta[ss[:,:,None],ii[:,None,:]]
                samples=(dd*ww).sum((1,2))/ww.sum((1,2))
                rows.append({'site':site,'method':method,'policy':policy,'delta_cost':float((delta*w).sum()/w.sum()),
                    'ci95_low':float(np.quantile(samples,.025)),'ci95_high':float(np.quantile(samples,.975)),
                    'three_constraint_fwer_low':float(np.quantile(samples,.05/6)),'three_constraint_fwer_high':float(np.quantile(samples,1-.05/6)),
                    'calendar_days':N,'seeds':len(seeds),'interpretation':'negative favors method; paired seed and exact-calendar 7-day circular block bootstrap; familywise interval applies only to three predeclared constraint contrasts'})
    pd.DataFrame(rows).to_csv(out/'paired_cost_intervals.csv',index=False)
    print(table[(table.scenario=='clean')&(table.ratio==2)&(table.period=='all')&table.group.isin(['none_f1.0_plain','bounds_f1.0_plain','curve_f1.0_plain','distribution_f1.0_plain','ridge','persistence'])].to_string(index=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();main(a.root)
