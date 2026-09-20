"""Require real populated coding tables; report pre-adjudication agreement only."""
import argparse,json
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score,confusion_matrix

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('coder_a');p.add_argument('coder_b');p.add_argument('--out',required=True);a=p.parse_args()
    x=pd.read_csv(a.coder_a,keep_default_na=False);y=pd.read_csv(a.coder_b,keep_default_na=False)
    keys=['study_id','constraint_id'];fields=['D1','D2','D3','D4','ablation_type','negative_effect_reported','harm_metric_reported','prior_substitution_tested']
    for frame in [x,y]:
        for col in keys+fields+['coder_id','coded_utc','fulltext_version','page_section','evidence_quote']:
            assert col in frame and frame[col].astype(str).str.strip().ne('').all(),f'Missing {col}'
        assert not frame.duplicated(keys).any(),'Duplicate constraint key'
    assert set(x.coder_id).isdisjoint(set(y.coder_id)),'Independent coder IDs required; authors must attest actual independence'
    assert set(map(tuple,x[keys].values))==set(map(tuple,y[keys].values)),'Matched constraint set required'
    m=x.merge(y,on=keys,suffixes=('_a','_b'));result={}
    for field in fields:
        va=m[field+'_a'];vb=m[field+'_b'];labels=sorted(set(va)|set(vb))
        result[field]={'n':len(m),'agreement':float((va==vb).mean()),
                       'kappa':float(cohen_kappa_score(va,vb)) if len(labels)>1 else None,
                       'labels':labels,'confusion_matrix':confusion_matrix(va,vb,labels=labels).tolist()}
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
