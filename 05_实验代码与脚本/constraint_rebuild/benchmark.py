"""Controlled prospective benchmark. Run selftest before a matrix is dispatched."""
from pathlib import Path
import argparse,copy,hashlib,json,os,time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'protocol.json').read_text())
torch.set_num_threads(2)

class Forecaster(nn.Module):
    def __init__(self,kind,L,H):
        super().__init__();self.kind=kind
        if kind=='linear':self.net=nn.Linear(L*2,H)
        elif kind=='gru':self.gru=nn.GRU(2,32,batch_first=True);self.head=nn.Linear(32,H)
        else:raise ValueError(kind)
    def forward(self,x):
        if self.kind=='linear':return self.net(x.flatten(1))
        _,h=self.gru(x);return self.head(h[-1])

def starts(frame,split):
    L,H,S=CFG['lookback'],CFG['horizon'],CFG['origin_stride']
    raw=frame[['power_pu','wind_speed']].copy()
    raw['wind_speed']/=15.
    filled=raw.ffill(limit=3).to_numpy(np.float32)
    y=frame.power_pu.to_numpy(np.float32)
    ids=np.arange(L,len(frame)-H+1,S)
    # Eligibility uses unmodified observed data. Perturbations cannot change the scored set.
    good=np.array([np.isfinite(filled[i-L:i]).all() and np.isfinite(y[i:i+H]).all() for i in ids])
    ids=ids[good];ts=frame.index
    tr=pd.Timestamp(split['train_end'],tz='UTC');va=pd.Timestamp(split['validation_end'],tz='UTC');end=pd.Timestamp(split['test_end'],tz='UTC')
    masks={'train':(ts[ids+H-1]<tr),'val':(ts[ids]>=tr)&(ts[ids+H-1]<va),
           'test':(ts[ids]>=va)&(ts[ids+H-1]<end)}
    return filled,y,{k:ids[np.asarray(v)] for k,v in masks.items()}

def windows(x,y,ids):
    L,H=CFG['lookback'],CFG['horizon']
    return np.stack([x[i-L:i] for i in ids]),np.stack([y[i:i+H] for i in ids])

def perturb(x,scenario,seed):
    a=x.copy();rng=np.random.default_rng(seed)
    if scenario.startswith('power_gaussian_005'):a[:,0]+=rng.normal(0,.05,len(a))
    elif scenario=='wind_gaussian_05ms':a[:,1]+=rng.normal(0,.5/15,len(a))
    elif scenario=='power_drift_005':a[:,0]+=.05*np.sin(np.arange(len(a))*2*np.pi/(144*7))
    elif scenario=='power_missing_10pct_ffill':
        a[rng.random(len(a))<.1,0]=np.nan
        a[:,0]=pd.Series(a[:,0]).ffill().fillna(0).values
    elif scenario!='clean':raise ValueError(scenario)
    if scenario.endswith('_denoised'):
        a[:,0]=pd.Series(a[:,0]).rolling(3,min_periods=1).median().values
    return a

def fit_prior(frame,train_end):
    # Only pre-cutoff contemporaneous measured pairs; future wind is NEVER used at inference.
    d=frame[frame.index<pd.Timestamp(train_end,tz='UTC')][['wind_speed','power_pu']].dropna()
    edges=np.arange(0,31.,1.);centers=(edges[:-1]+edges[1:])/2
    ix=np.clip(np.digitize(d.wind_speed,edges)-1,0,len(centers)-1)
    means=[];hist=[];grid=np.linspace(-.1,1.2,40)
    allp=d.power_pu.to_numpy()
    global_pdf=np.exp(-.5*((allp[:,None]-grid)/.06)**2).mean(0);global_pdf/=global_pdf.sum()
    for k in range(len(centers)):
        p=allp[ix==k]
        means.append(float(p.mean()) if len(p)>=20 else np.nan)
        h=np.exp(-.5*((p[:,None]-grid)/.06)**2).mean(0) if len(p)>=20 else global_pdf.copy()
        hist.append(h/(h.sum()+1e-12))
    means=pd.Series(means).interpolate(limit_direction='both').fillna(float(np.mean(allp))).values
    return {'centers':centers,'means':means.astype(np.float32),'hist':np.asarray(hist,np.float32),'grid':grid.astype(np.float32)}

def penalty(pred,x,family,factor,prior):
    if family=='none':return pred.sum()*0
    if family=='bounds':return (F.relu(-pred)**2+F.relu(pred-factor)**2).mean()
    idx=(x[:,-1,1]*15).long().clamp(0,len(prior['means'])-1)
    if family=='curve':
        # A lagged-wind persistence proxy, NOT future weather or a physical hard bound.
        target=prior['means'][idx,None]*factor
        return ((pred-target)**2).mean()
    if family=='distribution':
        p=torch.exp(-.5*((pred[:,:,None]/factor-prior['grid'])/.06)**2).mean(1)+1e-8
        p=p/p.sum(1,keepdim=True);q=prior['hist'][idx]+1e-8;q=q/q.sum(1,keepdim=True)
        m=(p+q)/2
        return .5*(p*(p.log()-m.log())+q*(q.log()-m.log())).sum(1).mean()
    raise ValueError(family)

def infer(model,x,device):
    model.eval();out=[]
    with torch.no_grad():
        for i in range(0,len(x),512):out.append(model(torch.as_tensor(x[i:i+512],device=device)).cpu().numpy())
    return np.concatenate(out)

def train(kind,seed,family,lam,factor,control,trainxy,valxy,prior,device,epochs):
    np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    model=Forecaster(kind,CFG['lookback'],CFG['horizon']).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'],weight_decay=.001 if control=='l2' else 0.)
    xt,yt=[torch.as_tensor(a,device=device) for a in trainxy]
    vp=valxy[1];best=float('inf');state=None;stale=0;history=[]
    pri={k:torch.as_tensor(v,device=device) for k,v in prior.items()}
    # Every configuration gets the same initialization and shuffle seed.
    rng=np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        for ix in np.array_split(rng.permutation(len(xt)),int(np.ceil(len(xt)/CFG['batch_size']))):
            x=xt[ix];y=yt[ix];opt.zero_grad();pred=model(x)
            loss=F.mse_loss(pred,y)+lam*penalty(pred,x,family,factor,pri)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        value=float(np.mean((infer(model,valxy[0],device)-vp)**2));history.append(value)
        if value<best:
            best=value;state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=CFG['patience']:break
    model.load_state_dict(state)
    return model,best,history

def configs():
    yield 'none',0.,1.,'plain'
    yield 'none',0.,1.,'l2'
    yield 'none',0.,1.,'augmentation'
    for family in ['bounds','curve','distribution']:
        for factor in CFG['misspecification_factors']:
            for lam in CFG['lambdas']:yield family,lam,factor,'plain'

def run(args):
    out=Path(args.out)/args.site/args.backbone/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(Path(args.data)/f'{args.site}.csv',index_col='time',parse_dates=True)
    frame.index=pd.to_datetime(frame.index,utc=True)
    assert frame.index.is_unique
    assert (frame.index.to_series().diff().dropna()==pd.Timedelta(minutes=10)).all()
    x,y,ids=starts(frame,CFG['splits'][args.site])
    assert min(map(len,ids.values()))>=30,{k:len(v) for k,v in ids.items()}
    trainxy=windows(x,y,ids['train']);valxy=windows(x,y,ids['val'])
    prior=fit_prior(frame,CFG['splits'][args.site]['train_end'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest={'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'data_sha256':hashlib.sha256((Path(args.data)/f'{args.site}.csv').read_bytes()).hexdigest(),
              'windows':{k:len(v) for k,v in ids.items()},'seed':args.seed,'backbone':args.backbone,
              'epochs_budget':args.epochs,'smoke':args.smoke,'torch':torch.__version__,'device':device}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    trials=[];choices=list(configs())
    if args.site==CFG['external_site']:
        selection=Path(args.out)/'external_selection.json'
        assert selection.exists(),'External site requires transferred development selection'
        choices=[(r['family'],r['lambda'],r['factor'],r['control']) for r in json.loads(selection.read_text()) if r['backbone']==args.backbone]
    if args.smoke:choices=[choices[0],next(c for c in choices if c[0]=='bounds')]
    for family,lam,factor,control in choices:
        tag=f'{family}_l{lam}_f{factor}_{control}'
        checkpoint=out/f'{tag}.pt';meta=out/f'{tag}.json'
        if meta.exists():trials.append(json.loads(meta.read_text()));continue
        actual_train=trainxy
        if control=='augmentation':
            # Fixed sequence-level augmentation, exactly the evaluated noise magnitude.
            aug=windows(perturb(x,'power_gaussian_005',args.seed+7000),y,ids['train'])
            actual_train=aug  # same samples, batches and maximum updates as the clean arm
        model,value,history=train(args.backbone,args.seed,family,lam,factor,control,actual_train,valxy,prior,device,args.epochs)
        torch.save(model.state_dict(),checkpoint)
        row={'tag':tag,'family':family,'lambda':lam,'factor':factor,'control':control,'val_mse':value,'history':history}
        meta.write_text(json.dumps(row));trials.append(row);print(args.site,args.backbone,args.seed,tag,value,flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k!='history'} for r in trials]).to_csv(out/'validation_trials.csv',index=False)
    # Test is a separate stage, only after development selection is frozen.
    (out/'TRAINING_COMPLETE').write_text('validation candidates only; test not evaluated\n')

def selftest():
    torch.manual_seed(0);x=torch.randn(2,36,2,requires_grad=True)
    m=Forecaster('linear',36,36);m(x).sum().backward();assert (x.grad.abs().sum((0,1))>0).all()
    p=torch.tensor([-.2,.4,1.4]);y=torch.tensor([0.,.7,1.]);assert ((p.clamp(0,1)-y).abs()<=(p-y).abs()).all()
    a=np.zeros((100,2),np.float32);z=perturb(a,'power_gaussian_005',4);assert np.all(z[:,1]==0)
    assert np.array_equal(z,perturb(a,'power_gaussian_005',4))
    for cut in [30,60]:assert np.array_equal(z[:cut],perturb(a[:cut],'power_gaussian_005',4))
    idx=pd.date_range('2016-01-01',periods=1000,freq='10min',tz='UTC')
    d=pd.DataFrame({'power_pu':.5,'wind_speed':8.},index=idx);d.iloc[400,0]=np.nan
    _,_,ids=starts(d,{'train_end':'2016-01-04','validation_end':'2016-01-06','test_end':'2016-01-08'})
    assert all(not (i<=400<i+36) for ar in ids.values() for i in ar)
    print('PASS: channel gradients, projection control, causal sequence perturbation, no target imputation',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--site',default='kelmarsh');ap.add_argument('--backbone',default='linear',choices=['linear','gru'])
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--data',default=str(HERE.parents[1]/'04_数据集/constraint_rebuild_v1'))
    ap.add_argument('--out',default=str(HERE.parents[1]/'06_实验结果与日志/constraint_rebuild_v1'))
    ap.add_argument('--epochs',type=int,default=CFG['epochs']);args=ap.parse_args()
    if args.selftest:selftest()
    else:run(args)
