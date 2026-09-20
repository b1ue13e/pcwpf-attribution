"""Round two: held-out calibration, identical decision policy across predictors."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from core import CFG,HERE,Forecaster,starts,windows,perturb,infer

def lock(root,data):
    root=Path(root);rows=[];ridge=[]
    for site in CFG['development_sites']:
        for seed in CFG['seed_list']:
            p=root/site/'dlinear'/f'seed{seed}'/'validation_trials.csv';d=pd.read_csv(p)
            assert len(d)==30
            for _,g in d.groupby(['family','factor','control']):
                row=g.sort_values(['val_mse','lambda']).iloc[0].to_dict();rows.append(dict(row,site=site,seed=seed,backbone='dlinear'))
        f=pd.read_csv(Path(data)/f'{site}.csv',index_col='time',parse_dates=True);f.index=pd.to_datetime(f.index,utc=True)
        x,y,ids=starts(f,CFG['splits'][site]);xt,yt=windows(x,y,ids['train']);xv,yv=windows(x,y,ids['val'])
        for alpha in [.1,1.,10.]:
            m=Ridge(alpha=alpha).fit(xt.reshape(len(xt),-1),yt)
            ridge.append({'site':site,'alpha':alpha,'val_mse':float(np.mean((m.predict(xv.reshape(len(xv),-1))-yv)**2))})
    table=pd.DataFrame(rows);assert len(table)==120;table.to_csv(root/'selection.csv',index=False)
    external=[]
    for key,g in table.groupby(['backbone','family','factor','control']):
        vc=g['lambda'].value_counts();lam=float(min(vc[vc==vc.max()].index))
        external.append(dict(zip(['backbone','family','factor','control'],key),**{'lambda':lam}))
    (root/'external_selection.json').write_text(json.dumps(external,indent=2))
    rd=pd.DataFrame(ridge);rd.to_csv(root/'ridge_validation.csv',index=False)
    choices=rd.sort_values(['val_mse','alpha']).groupby('site').first().alpha.to_dict()
    choices.update({s:min(choices.values()) for s in CFG['external_sites']})
    (root/'ridge_selection.json').write_text(json.dumps(choices,indent=2))
    manifest={'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
        'selection_sha256':hashlib.sha256((root/'selection.csv').read_bytes()).hexdigest(),'status':'frozen before Hill of Towie test evaluation'}
    (root/'SELECTION_LOCK.json').write_text(json.dumps(manifest,indent=2));print(manifest)

def metrics(pred,y,times,delta,info):
    out=[];daily=[];base=np.clip(pred,0,1)
    for ratio in [1,2,5]:
        for policy in ['uncalibrated','quantile_calibrated']:
            p=base if policy=='uncalibrated' else np.clip(base+delta[ratio],0,1)
            e=p-y;cost=(ratio*np.maximum(e[:,:6],0)+np.maximum(-e[:,:6],0)).mean(1)
            detail=pd.DataFrame({'day':times.strftime('%Y-%m-%d'),'mse':(e**2).mean(1),'cost':cost})
            g=detail.groupby('day');a=g.mean();a['n_origins']=g.size();a=a.reset_index()
            for k,v in dict(info,ratio=ratio,policy=policy).items():a[k]=v
            daily.append(a)
            for period,mask in [('all',np.ones(len(y),bool))]+[(f'Q{q}',times.quarter==q) for q in [1,2,3,4]]:
                if not mask.any():continue
                out.append(dict(info,ratio=ratio,policy=policy,period=period,n_origins=int(mask.sum()),
                    rmse_pu=float(np.sqrt(np.mean(e[mask]**2))),cost=float(cost[mask].mean())))
    return out,daily

def run(a):
    root=Path(a.root);assert (root/'SELECTION_LOCK.json').exists();site=a.site
    f=pd.read_csv(Path(a.data)/f'{site}.csv',index_col='time',parse_dates=True);f.index=pd.to_datetime(f.index,utc=True)
    x,y,ids=starts(f,CFG['splits'][site]);xt,yt=windows(x,y,ids['train']);xc,yc=windows(x,y,ids['cal'])
    yy=windows(x,y,ids['test'])[1];times=f.index[ids['test']]
    alpha=json.loads((root/'ridge_selection.json').read_text())[site]
    ridge=Ridge(alpha=alpha).fit(xt.reshape(len(xt),-1),yt)
    out=root/site/'assessment';out.mkdir(parents=True,exist_ok=True);records=[];days=[];calibration=[]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for seed in CFG['seed_list']:
        if site in CFG['development_sites']:
            t=pd.read_csv(root/'selection.csv');chosen=t[(t.site==site)&(t.seed==seed)].to_dict('records')
        else:
            chosen=json.loads((root/'external_selection.json').read_text())
            for r in chosen:r['tag']=f"{r['family']}_l{r['lambda']}_f{r['factor']}_{r['control']}"
        predictors={'ridge':lambda z:ridge.predict(z.reshape(len(z),-1)),
                    'persistence':lambda z:np.repeat(z[:,-1,0,None],CFG['horizon'],axis=1)}
        for r in chosen:
            m=Forecaster('dlinear',CFG['lookback'],CFG['horizon']).to(device)
            m.load_state_dict(torch.load(root/site/'dlinear'/f'seed{seed}'/f"{r['tag']}.pt",map_location=device,weights_only=True))
            predictors[r['tag']]=lambda z,m=m:infer(m,z,device)
        for method,predict in predictors.items():
            cp=np.clip(predict(xc),0,1)
            delta={r:np.quantile(yc-cp,1/(r+1),axis=0) for r in [1,2,5]}
            for ratio,b in delta.items():calibration.append({'site':site,'seed':seed,'method':method,'ratio':ratio,'delta_by_lead':b.tolist()})
            for scenario in CFG['scenarios']:
                xx,_=windows(perturb(x,scenario,seed+9000),y,ids['test']);p=predict(xx)
                info={'site':site,'seed':seed,'method':method,'scenario':scenario}
                r,d=metrics(p,yy,times,delta,info);records.extend(r);days.extend(d)
                if seed==42 and scenario=='clean':
                    np.savez_compressed(out/f'{method}_sample.npz',prediction=p,truth=yy,first_target_ns=times.asi8,calibration_truth=yc,calibration_prediction=cp)
        print('assessed',site,seed,flush=True)
    pd.DataFrame(records).to_csv(out/'metrics.csv',index=False);pd.concat(days).to_csv(out/'daily.csv',index=False)
    (out/'calibration.json').write_text(json.dumps(calibration))
    (out/'COMPLETE').write_text(json.dumps({'windows':{k:len(v) for k,v in ids.items()},'policy':'calibration computed on disjoint validation half; clean calibration for every test scenario'}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--lock',action='store_true');ap.add_argument('--root',required=True);ap.add_argument('--data',required=True);ap.add_argument('--site');a=ap.parse_args()
    if a.lock:lock(a.root,a.data)
    else:run(a)
