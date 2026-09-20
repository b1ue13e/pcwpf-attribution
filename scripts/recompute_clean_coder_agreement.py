from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
ROOT=Path(r'E:/论文1/08_报告与说明文档')
dfs=[pd.read_csv(ROOT/f'盲审填写表_Coder{x}_独立编码.csv',encoding='utf-8') for x in 'ABC']
fields=['inclusion_check','D1_provenance','D2_stage','D3_inference_req','D4_fidelity','d5_ablation_type','d5_negative_reported']
rows=[]
for field in fields:
 vals=[d[field].fillna('NA').astype(str).str.strip().str.lower() for d in dfs]
 inc=dfs[0]['inclusion_check'].fillna('NA').astype(str).str.strip().str.lower().eq('include')
 for scope,mask in [('all',pd.Series(True,index=vals[0].index)),('included',inc)]:
  for i,j in [(0,1),(0,2),(1,2)]:
   a,b=vals[i][mask],vals[j][mask]; cats=sorted(set(a)|set(b)); k=cohen_kappa_score(a,b,labels=cats) if len(cats)>1 else np.nan
   rows.append({'scope':scope,'field':field,'pair':f'{i+1}{j+1}','n':len(a),'po':float((a.values==b.values).mean()),'kappa':k})
out=ROOT/'clean_coder_agreement_recomputed.csv'; pd.DataFrame(rows).to_csv(out,index=False,encoding='utf-8-sig')
lines=['# Clean-material coder agreement recomputation','', '- Inputs: the three independent coding tables dated 2026-09-08.','- Missing labels normalized to `NA`; constant-marginal kappa reported as `NaN`.','- This reports agreement for available tables; it does not by itself prove that coding was blinded to earlier materials.','']
for scope in ['all','included']:
 lines += [f'## {scope}','', '| field | pair | n | observed agreement | Cohen kappa |','|---|---:|---:|---:|---:|']
 for r in rows:
  if r['scope']==scope: lines.append(f"| {r['field']} | {r['pair']} | {r['n']} | {r['po']:.3f} | {'NaN' if pd.isna(r['kappa']) else f'{r['kappa']:.3f}'} |")
 lines.append('')
(ROOT/'clean_coder_agreement_recomputed.md').write_text('`n'.join(lines),encoding='utf-8')
print(f'wrote {out}')


