"""Explicit univariate DLinear control: decomposition plus two temporal maps."""
from pathlib import Path
import copy,json
import numpy as np
import pandas as pd
import torch
from torch import nn
from benchmark import CFG,infer,starts,windows,perturb
from evaluate import score
BASE=Path(__file__).resolve().parent.parent

class DLinear(nn.Module):
    def __init__(self):
        super().__init__();L,H=CFG['lookback'],CFG['horizon']
        self.avg=nn.AvgPool1d(25,stride=1)
        self.seasonal=nn.Linear(L,H);self.trend=nn.Linear(L,H)
        nn.init.constant_(self.seasonal.weight,1/L);nn.init.constant_(self.trend.weight,1/L)
    def forward(self,x):
        p=x[:,:,0];pad=torch.cat([p[:,:1].repeat(1,12),p,p[:,-1:].repeat(1,12)],1)
        trend=self.avg(pad[:,None]).squeeze(1)
        return self.seasonal(p-trend)+self.trend(trend)

def main():
    out=BASE/'results/dlinear';out.mkdir(exist_ok=True);records=[];days=[];logs=[]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for site in ['kelmarsh','penmanshiel','smarteole']:
        d=pd.read_csv(BASE/'data'/f'{site}.csv',index_col='time',parse_dates=True);d.index=pd.to_datetime(d.index,utc=True)
        x,y,ids=starts(d,CFG['splits'][site]);xt,yt=windows(x,y,ids['train']);xv,yv=windows(x,y,ids['val'])
        ty=windows(x,y,ids['test'])[1];times=d.index[ids['test']]
        for seed in CFG['seed_list']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);m=DLinear().to(device);opt=torch.optim.Adam(m.parameters(),lr=CFG['learning_rate'])
            tx=torch.as_tensor(xt,device=device);targ=torch.as_tensor(yt,device=device);best=float('inf');stale=0
            history=[]
            for ep in range(CFG['epochs']):
                m.train()
                for ix in np.array_split(rng.permutation(len(xt)),int(np.ceil(len(xt)/CFG['batch_size']))):
                    opt.zero_grad();loss=((m(tx[ix])-targ[ix])**2).mean();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
                v=float(np.mean((infer(m,xv,device)-yv)**2));history.append(v)
                if v<best:best=v;state=copy.deepcopy(m.state_dict());stale=0
                else:stale+=1
                if stale>=CFG['patience']:break
            m.load_state_dict(state);torch.save(state,out/f'{site}_{seed}.pt')
            logs.append({'site':site,'seed':seed,'val_mse':best,'epochs':len(history)})
            for scenario in CFG['scenarios']:
                xx,_=windows(perturb(x,scenario,seed+9000),y,ids['test']);pred=infer(m,xx,device)
                info={'site':site,'backbone':'dlinear','seed':seed,'method':'dlinear','scenario':scenario}
                for clip in [False,True]:
                    a,b=score(pred,ty,times,clip,info);records.extend(a);days.append(b)
            print(site,seed,best,flush=True)
    pd.DataFrame(records).to_csv(out/'test_metrics.csv',index=False);pd.concat(days).to_csv(out/'daily_metrics.csv',index=False)
    pd.DataFrame(logs).to_csv(out/'validation_trials.csv',index=False)
    (out/'COMPLETE').write_text('DLinear univariate baseline, no physics penalty. Added after initial matrix as user-requested comparator; no test tuning.\n')
if __name__=='__main__':main()
