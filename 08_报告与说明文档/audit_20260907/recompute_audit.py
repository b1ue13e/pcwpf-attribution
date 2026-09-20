from pathlib import Path
import json, hashlib, importlib.util
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
f = pd.read_csv(ROOT / '04_数据集/wind_final.csv')
m = pd.read_csv(ROOT / '04_数据集/wind_main.csv')
t = pd.to_datetime(m.DATATIME)
n = int(len(f) * .8)
p = f.power.to_numpy()
upper = float(p[:n].max())
facts = {'rows':len(f), 'unique_timestamps':int(t.nunique()),
         'duplicate_timestamp_rows':int(t.duplicated().sum()),
         'exact_duplicate_rows':int(m.duplicated().sum()),
         'power_equals_PREPOWER':bool(np.allclose(p,m.PREPOWER)),
         'power_equals_ROUND_POWER':bool(np.allclose(p,m['ROUND(A.POWER,0)'],equal_nan=True)),
         'train_max':upper,'test_max':float(p[n:].max()),
         'test_above_train_max':int((p[n:]>upper).sum()),
         'test_rows':len(p[n:]),'date_min':str(t.min()),'date_max':str(t.max()),
         'train_end_time':str(t.iloc[n-1]),'test_start_time':str(t.iloc[n])}
for h in [24,96]:
    delta=(t.shift(-h)-t).dt.total_seconds()/3600
    facts[f'row_span_{h}_hours_quantiles']=delta.quantile([0,.25,.5,.75,1]).to_dict()
r=pd.read_csv(ROOT/'06_实验结果与日志/guardrail_ablation_results.csv')
agg=r.groupby(['arm','sigma','lambda']).agg(
    rmse_mean=('rmse','mean'),rmse_std=('rmse','std'),mae_mean=('mae','mean'),
    r2_mean=('r2','mean'),violation_mean=('violation_rate','mean')).reset_index()
agg.to_csv(OUT/'guardrail_recomputed.csv',index=False)
# Read-only diagnostic: verify the computational graph, without training.
try:
    import torch
    spec=importlib.util.spec_from_file_location('guardrail',ROOT/'05_实验代码与脚本/run_guardrail_ablation_windcn.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    torch.manual_seed(42);net=mod.DLinear();x=torch.randn(2,96,6,requires_grad=True)
    out=net(x)[:,:,-1];out.sum().backward()
    facts['input_gradient_abs_sum_per_channel']=x.grad.abs().sum((0,1)).tolist()
    y=x.detach().clone();y[:,:,:5]+=100
    facts['nonpower_perturbation_max_output_change']=float((net(y)[:,:,-1]-out.detach()).abs().max())
except Exception as exc:
    facts['graph_check_error']=str(exc)
# Inventory includes historical, smoke and active outputs; it is not an endorsement.
items=[]
for folder in ['06_实验结果与日志','05_实验代码与脚本/results','.experiment-to-paper']:
    for path in (ROOT/folder).rglob('*'):
        if path.is_file():
            items.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,
                          'suffix':path.suffix,'mtime':path.stat().st_mtime})
pd.DataFrame(items).to_csv(OUT/'local_result_inventory.csv',index=False)
facts['inventory_files']=len(items)
facts['sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
 for path in [ROOT/'04_数据集/wind_final.csv',ROOT/'04_数据集/wind_main.csv',
 ROOT/'06_实验结果与日志/guardrail_ablation_results.csv',ROOT/'05_实验代码与脚本/run_guardrail_ablation_windcn.py']}
(OUT/'verified_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(facts,ensure_ascii=False,indent=2))
