"""Separate held-out evaluation after validation-selection lock has been written."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import torch
from benchmark import CFG,HERE,Forecaster,fit_prior,infer,perturb,starts,windows

def select(root):
    rows=[]
    for p in Path(root).glob('*/*/seed*/validation_trials.csv'):
        site,kind,seed=p.parts[-4:-1]
        if site not in CFG['development_sites']:continue
        d=pd.read_csv(p)
        for key,g in d.groupby(['family','factor','control']):
            best=g.sort_values(['val_mse','lambda']).iloc[0].to_dict()
            rows.append(dict(best,site=site,backbone=kind,seed=int(seed[4:])))
    assert len(rows)==2*2*5*12,f'Incomplete development selection: {len(rows)}'
    selected=pd.DataFrame(rows);selected.to_csv(Path(root)/'selection.csv',index=False)
    external=[]
    for key,g in selected.groupby(['backbone','family','factor','control']):
        counts=g['lambda'].value_counts();lam=float(min(counts[counts==counts.max()].index))
        external.append(dict(zip(['backbone','family','factor','control'],key),**{'lambda':lam}))
    (Path(root)/'external_selection.json').write_text(json.dumps(external,indent=2))
    lock={'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
          'selection_sha256':hashlib.sha256((Path(root)/'selection.csv').read_bytes()).hexdigest(),
          'rule':'CLEAN validation only; external lambda modal across development sites and seeds',
          'n_selected':len(rows)}
    (Path(root)/'SELECTION_LOCK.json').write_text(json.dumps(lock,indent=2))
    print(lock)

def score(pred,y,times,clip,info):
    if clip:pred=np.clip(pred,0,1)
    err=pred-y
    rows=[]
    # Operation uses only the next hour: no repeatedly counting overlapping 6h delivery windows.
    short=np.maximum(err[:,:6],0).mean(1);surplus=np.maximum(-err[:,:6],0).mean(1)
    base=pd.DataFrame({'day':pd.to_datetime(times).strftime('%Y-%m-%d'),
                       'mse':(err**2).mean(1),'mae':np.abs(err).mean(1),
                       'violation':((pred<0)|(pred>1)).mean(1),
                       'shortfall':short,'surplus':surplus})
    for h in [6,18,36]:base[f'mse_lead{h}']=err[:,h-1]**2
    for ratio in [1,2,5]:base[f'cost_ratio{ratio}']=ratio*short+surplus
    daily=base.groupby('day').mean().reset_index();daily['n_origins']=base.groupby('day').size().values
    for k,v in dict(info,clipped=clip).items():daily[k]=v
    for period,mask in [('all',np.ones(len(y),bool))]+[(f'Q{q}',pd.to_datetime(times).quarter==q) for q in [1,2,3,4]]:
        if not mask.any():continue
        e=err[mask];p=pred[mask]
        row=dict(info,clipped=clip,period=period,n_origins=int(mask.sum()),rmse_pu=float(np.sqrt(np.mean(e**2))),
                 mae_pu=float(np.abs(e).mean()),p95_abs_error_pu=float(np.quantile(np.abs(e),.95)),
                 nameplate_exceedance_rate=float(((p<0)|(p>1)).mean()))
        for h in [6,18,36]:row[f'rmse_lead{h}_pu']=float(np.sqrt(np.mean(e[:,h-1]**2)))
        for ratio in [1,2,5]:row[f'cost_ratio{ratio}']=float((ratio*short[mask]+surplus[mask]).mean())
        rows.append(row)
    return rows,daily

def run(a):
    root=Path(a.root);assert (root/'SELECTION_LOCK.json').exists()
    directory=root/a.site/a.backbone/f'seed{a.seed}'
    d=pd.read_csv(Path(a.data)/f'{a.site}.csv',index_col='time',parse_dates=True);d.index=pd.to_datetime(d.index,utc=True)
    x,y,ids=starts(d,CFG['splits'][a.site]);testids=ids['test'];times=d.index[testids]
    testy=windows(x,y,testids)[1]
    selected=pd.read_csv(root/'selection.csv')
    if a.site in CFG['development_sites']:
        chosen=selected[(selected.site==a.site)&(selected.backbone==a.backbone)&(selected.seed==a.seed)].to_dict('records')
    else:
        chosen=[v for v in json.loads((root/'external_selection.json').read_text()) if v['backbone']==a.backbone]
        for v in chosen:v['tag']=f"{v['family']}_l{v['lambda']}_f{v['factor']}_{v['control']}"
    outputs=[];daily=[];device='cuda' if torch.cuda.is_available() else 'cpu'
    arrays={s:windows(perturb(x,s,a.seed+9000),y,testids)[0] for s in CFG['scenarios']}
    for row in chosen:
        model=Forecaster(a.backbone,CFG['lookback'],CFG['horizon']).to(device)
        model.load_state_dict(torch.load(directory/f"{row['tag']}.pt",map_location=device,weights_only=True))
        for scenario,xx in arrays.items():
            pred=infer(model,xx,device)
            info={'site':a.site,'backbone':a.backbone,'seed':a.seed,'method':row['tag'],'scenario':scenario}
            for clip in [False,True]:
                r,b=score(pred,testy,times,clip,info);outputs.extend(r);daily.append(b)
            archive=directory/'predictions';archive.mkdir(exist_ok=True)
            np.savez_compressed(archive/f"{row['tag']}_{scenario}.npz",prediction=pred,truth=testy,origin_ns=times.asi8)
    pd.DataFrame(outputs).to_csv(directory/'test_metrics.csv',index=False)
    pd.concat(daily).to_csv(directory/'daily_metrics.csv',index=False)
    (directory/'EVALUATION_COMPLETE').write_text('All frozen selections and predeclared scenarios evaluated\n')
    print('evaluated',a.site,a.backbone,a.seed,len(outputs),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--select',action='store_true');ap.add_argument('--root',required=True)
    ap.add_argument('--data');ap.add_argument('--site');ap.add_argument('--backbone');ap.add_argument('--seed',type=int)
    a=ap.parse_args()
    if a.select:select(a.root)
    else:run(a)
