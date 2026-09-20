"""Traceable retrieval and metadata checking; never invent human screening/coding."""
from pathlib import Path
import concurrent.futures,datetime,hashlib,json,re,time
from urllib.parse import quote
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'08_报告与说明文档/corpus_rebuild_20260907'
QUERIES=['physics-constrained wind','physics-informed wind power','physics-guided wind','physics-embedded wind power','wind power curve deep learning']
def get(url,params=None):
    r=requests.get(url,params=params,timeout=35,headers={'User-Agent':'ConstraintLifecycleAudit/1.0'})
    r.raise_for_status();return r.json(),r.url
def doi_check(row):
    doi=str(row.get('doi','')).strip()
    if doi in ['nan','']:return dict(study_id=row['study_id'],doi='',status='missing_identifier')
    try:
        d,url=get('https://api.crossref.org/works/'+quote(doi,safe=''))
        m=d['message'];p=OUT/'metadata'/f"{row['study_id']}.json";p.write_text(json.dumps(d,ensure_ascii=False),encoding='utf-8')
        return dict(study_id=row['study_id'],doi=doi,status='metadata_resolved_not_fulltext_verified',
                    source=url,title=m.get('title',[''])[0],type=m.get('type'),
                    online_date=str(m.get('published-online',m.get('published',{})).get('date-parts')),
                    title_agreement=re.sub(r'\W','',m.get('title',[''])[0]).lower()==re.sub(r'\W','',row['title']).lower())
    except Exception as e:return dict(study_id=row['study_id'],doi=doi,status='unresolved_not_proof_invalid',error=str(e))
def main():
    (OUT/'metadata').mkdir(parents=True,exist_ok=True);(OUT/'raw_queries').mkdir(exist_ok=True)
    old=pd.read_csv(ROOT/'08_报告与说明文档/study_level_ledger_48.csv')
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        checked=list(pool.map(doi_check,old.to_dict('records')))
    pd.DataFrame(checked).to_csv(OUT/'identifier_checks.csv',index=False)
    records=[];querylog=[]
    for qid,q in enumerate(QUERIES,1):
        cursor='*';pages=0;count=0;status='complete'
        while cursor:
            try:
                d,url=get('https://api.openalex.org/works',{'filter':f'title.search:{q},from_publication_date:2023-01-01,to_publication_date:2026-09-07',
                    'per-page':200,'cursor':cursor})
                pages+=1;(OUT/'raw_queries'/f'Q{qid}_{pages}.json').write_text(json.dumps(d,ensure_ascii=False),encoding='utf-8')
                for w in d['results']:
                    records.append({'record_id':w['id'],'query_id':f'Q{qid}','source':'OpenAlex',
                        'doi':(w.get('doi') or '').replace('https://doi.org/','').lower(),'title':w.get('display_name'),
                        'publication_date':w.get('publication_date'),'retrieval_url':url,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
                count+=len(d['results']);cursor=d['meta'].get('next_cursor') if d['results'] else None
            except Exception as e:status=str(e);break
        querylog.append({'query_id':f'Q{qid}','query':q,'pages':pages,'records':count,'status':status})
        print(querylog[-1],flush=True)
    pd.DataFrame(querylog).to_csv(OUT/'query_log.csv',index=False)
    if records:
        allr=pd.DataFrame(records);allr.to_csv(OUT/'retrieval_records.csv',index=False)
        unique=allr.drop_duplicates('record_id').copy()
        unique['screening_decision']='pending';unique['reason']='';unique['fulltext_url']='';unique['reviewer']=''
        unique.to_csv(OUT/'screening_pending.csv',index=False)
        counts={'raw_query_hits':len(allr),'unique_records':len(unique),'duplicate_query_hits':len(allr)-len(unique),
                'screened':0,'included':None,'status':'retrieval only; no invented inclusion counts'}
    else:counts={'raw_query_hits':0,'status':'retrieval unavailable; do not infer no literature'}
    (OUT/'flow_status.json').write_text(json.dumps(counts,indent=2))
    # A clean human worksheet has no prefilled legacy judgments.
    template=old[['study_id','doi','title']].copy()
    for col in ['inclusion','exclusion_reason','constraint_id','D1','D2','D3','code_url','code_commit','D4',
                'ablation_type','negative_effect_reported','harm_metric_reported','prior_substitution_tested',
                'fulltext_version','page_section','evidence_quote','coder_id','coded_utc']:template[col]=''
    template.to_csv(OUT/'independent_coding_blank.csv',index=False)
    print(counts,flush=True)
if __name__=='__main__':main()
