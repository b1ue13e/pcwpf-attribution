"""Persistence, lagged curve, ridge and LightGBM under the exact same origins."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/'vendor'))
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from benchmark import CFG,fit_prior,perturb,starts,windows
from evaluate import score

def main():
    records=[];daily=[];selected={};trials=[]
    out=BASE/'results/classical';out.mkdir(parents=True,exist_ok=True)
    for site in ['kelmarsh','penmanshiel','smarteole']:
        d=pd.read_csv(BASE/'data'/f'{site}.csv',index_col='time',parse_dates=True);d.index=pd.to_datetime(d.index,utc=True)
        x,y,ids=starts(d,CFG['splits'][site]);xt,yt=windows(x,y,ids['train']);xv,yv=windows(x,y,ids['val'])
        prior=fit_prior(d,CFG['splits'][site]['train_end']);times=d.index[ids['test']];testy=windows(x,y,ids['test'])[1]
        models={}
        for method,grid in [('ridge',[.1,1.,10.]),('lightgbm',[7,15,31])]:
            if site=='smarteole':
                vs=[selected[(s,method)] for s in ['kelmarsh','penmanshiel']]
                grid=[min(vs)]  # two development sites: modal with smallest-value tie break
            candidates=[]
            for param in grid:
                if method=='ridge':m=Ridge(alpha=param)
                else:m=MultiOutputRegressor(LGBMRegressor(n_estimators=100,num_leaves=int(param),learning_rate=.05,
                                    verbosity=-1,n_jobs=2,random_state=42),n_jobs=1)
                m.fit(xt.reshape(len(xt),-1),yt)
                loss=float(np.mean((m.predict(xv.reshape(len(xv),-1))-yv)**2))
                candidates.append((loss,param,m));trials.append({'site':site,'method':method,'param':param,'val_mse':loss})
                print(site,method,param,loss,flush=True)
            best=min(candidates,key=lambda a:(a[0],a[1]));selected[(site,method)]=best[1];models[method]=best[2]
        # Train once. Multiple evaluation seeds below are noise replicates, not model retraining.
        for seed in CFG['seed_list']:
            for scenario in CFG['scenarios']:
                xx,_=windows(perturb(x,scenario,seed+9000),y,ids['test'])
                predictions={'persistence':np.repeat(xx[:,-1,0,None],CFG['horizon'],axis=1),
                    'lagged_curve':np.repeat(np.interp(xx[:,-1,1]*15,prior['centers'],prior['means'])[:,None],CFG['horizon'],axis=1)}
                for method,m in models.items():predictions[method]=m.predict(xx.reshape(len(xx),-1))
                for method,pred in predictions.items():
                    info={'site':site,'backbone':'classical','seed':seed,'training_seed':42,'method':method,'scenario':scenario}
                    for clip in [False,True]:
                        r,b=score(pred,testy,times,clip,info);records.extend(r);daily.append(b)
                np.savez_compressed(out/f'{site}_seed{seed}_{scenario}.npz',truth=testy,origin_ns=times.asi8,**predictions)
        pd.DataFrame(records).to_csv(out/'test_metrics.csv',index=False)
        pd.concat(daily).to_csv(out/'daily_metrics.csv',index=False)
        pd.DataFrame(trials).to_csv(out/'validation_trials.csv',index=False)
    (out/'COMPLETE').write_text('Seed field denotes paired evaluation noise; models trained once with random_state=42\n')
if __name__=='__main__':main()
