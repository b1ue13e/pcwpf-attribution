from pathlib import Path
import hashlib,json,zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'04_数据集/constraint_rebuild_v2';RAW=OUT/'raw_hill_of_towie'
if __name__=='__main__':
    meta=pd.read_csv(RAW/'Hill_of_Towie_turbine_metadata.csv');entities=set(meta['Station ID']);assert len(entities)==21
    rated=float(meta['Rated power (kW)'].sum());parts=[];raw_hashes=[]
    for year in [2016,2017]:
        path=RAW/f'{year}.zip';h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        raw_hashes.append({'file':path.name,'sha256':h.hexdigest()})
        with zipfile.ZipFile(path) as z:
            for month in range(1,13):
                key=['TimeStamp','StationId'];power='wtc_ActPower_mean'
                wind=['wtc_AcWindSp_mean','wtc_PrWindSp_mean','wtc_SeWindSp_mean']
                a=pd.read_csv(z.open(f'tblSCTurGrid_{year}_{month:02}.csv'),usecols=key+[power])
                b=pd.read_csv(z.open(f'tblSCTurbine_{year}_{month:02}.csv'),usecols=key+wind)
                for d in [a,b]:
                    assert not d.duplicated(key).any()
                    assert set(d.StationId)<=entities
                d=a.merge(b,on=key,how='outer',validate='one_to_one')
                d['TimeStamp']=pd.to_datetime(d.TimeStamp,utc=True,format='mixed');g=d.groupby('TimeStamp')
                v=pd.DataFrame({'power_kw_signed':g[power].sum(min_count=21)})
                for src,name in zip(wind,['wind_speed','wind_primary','wind_secondary']):v[name]=g[src].mean().where(g[src].count()==21)
                v['power_entities']=g[power].count();parts.append(v)
                print(year,month,len(v),flush=True)
    out=pd.concat(parts).sort_index()
    duplicate_timestamps=int(out.index.duplicated().sum())
    if duplicate_timestamps:
        duplicate=out[out.index.duplicated(keep=False)]
        for col in out:
            assert (duplicate.groupby(level=0)[col].nunique(dropna=False)<=1).all(),f'Conflicting monthly boundary: {col}'
        out=out[~out.index.duplicated(keep='first')]
    assert out.index.is_unique
    out=out.reindex(pd.date_range('2016-01-01','2017-12-31 23:50',freq='10min',tz='UTC'))
    out.index.name='time';out['power_kw']=out.power_kw_signed.clip(lower=0);out['power_pu']=out.power_kw/rated;out['rated_kw']=rated
    path=OUT/'hill_of_towie.csv';out.to_csv(path)
    train=out[out.index<pd.Timestamp('2016-12-01',tz='UTC')];diff=(train.wind_primary-train.wind_secondary).dropna()
    report={'source':'https://zenodo.org/records/20204946','license':'CC-BY-4.0','rated_kw':rated,'turbines':21,
        'time_semantics':'UTC end of measurement period, no artificial shift','power_source':'tblSCTurGrid/wtc_ActPower_mean kW',
        'wind_source':'tblSCTurbine/wtc_AcWindSp_mean m/s','negative_target_policy':'retain signed source and use max(signed,0) generation target',
        'identical_month_boundary_duplicates':duplicate_timestamps,'rows':len(out),'valid_target':int(out.power_pu.notna().sum()),'missing_fraction':float(out.power_pu.isna().mean()),
        'training_sensor_disagreement':{'n':len(diff),'median_abs_mps':float(diff.abs().median()),'p95_abs_mps':float(diff.abs().quantile(.95)),
            'caveat':'pairwise sensor disagreement, not ground-truth measurement error'},
        'raw_files':raw_hashes,'prepared_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT/'hill_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report,flush=True)
