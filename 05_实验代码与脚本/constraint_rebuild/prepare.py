"""Rebuild measured farm power from raw SCADA; preserve the regular time grid."""
from pathlib import Path
from io import StringIO
import hashlib,json,re,zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'04_数据集/open_cross_site_wind'
OUT=ROOT/'04_数据集/constraint_rebuild_v1'

def greenbyte(path):
    frames=[]
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if not member.startswith('Turbine_Data_'):continue
            text=z.read(member).decode('utf-8-sig',errors='replace')
            lines=text.splitlines();start=next(i for i,s in enumerate(lines) if s.startswith('# Date and time,'))
            d=pd.read_csv(StringIO('\n'.join(lines[start:])),usecols=lambda c:c in ['# Date and time','Power (kW)','Wind speed (m/s)'])
            d.columns=['time' if c=='# Date and time' else 'power' if c=='Power (kW)' else 'wind' for c in d.columns]
            entity=next(s for s in lines[:start] if s.startswith('# Turbine:')).split(':',1)[1].strip()
            d['entity']=entity
            # Conservatively treat timestamps as interval starts: values usable 10 min later.
            d['time']=pd.to_datetime(d.time,utc=True)+pd.Timedelta(minutes=10)
            assert not d.duplicated(['entity','time']).any(),member
            frames.append(d)
    return frames

def farm_table(frames,expected,commercial):
    d=pd.concat(frames,ignore_index=True)
    assert not d.duplicated(['entity','time']).any(),'duplicate entity/time across files'
    assert d.entity.nunique()==expected
    d=d[d.time>=pd.Timestamp(commercial,tz='UTC')]
    g=d.groupby('time',sort=True)
    out=pd.DataFrame({'power_kw_signed':g.power.sum(min_count=expected),'wind_speed':g.wind.mean(),
                      'power_entities':g.power.count(),'wind_entities':g.wind.count()})
    out.loc[out.wind_entities<expected,'wind_speed']=np.nan
    out=out.reindex(pd.date_range(out.index.min(),out.index.max(),freq='10min',tz='UTC'))
    out.index.name='time'
    return out

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);meta={}
    for site,folder,count,commercial,record in [
        ('kelmarsh','kelmarsh_uk',6,'2016-04-15','5841834'),
        ('penmanshiel','penmanshiel_scotland',14,'2016-09-01','5946808')]:
        static=next((RAW/folder/'raw').glob('*WT_static.csv'))
        names=pd.read_csv(static);rated=float(names['Rated power (kW)'].sum())
        paths=sorted((RAW/folder/'raw').glob('*SCADA_201[67]*.zip'))
        assert any('2017' in p.name for p in paths),'Full-year test download required'
        frames=[]
        for p in paths:frames.extend(greenbyte(p))
        out=farm_table(frames,count,commercial)
        meta[site]=save(site,out,rated,paths,f'https://zenodo.org/records/{record}',
            '10-min measured Power (kW); complete turbine fleet required; timestamp shifted +10min conservatively',
            'development site; 2016 train/validation, fixed-model 2017 seasonal tests')
    p=RAW/'smarteole_france/raw/SMARTEOLE-WFC-open-dataset.zip'
    with zipfile.ZipFile(p) as z:
        member=next(n for n in z.namelist() if 'SCADA_1minData.csv' in n)
        d=pd.read_csv(z.open(member))
    powers=[c for c in d if c.startswith('active_power_') and c.endswith('_avg')]
    winds=[c for c in d if c.startswith('wind_speed_') and c.endswith('_avg')]
    assert len(powers)==7 and len(winds)==7
    idx=pd.to_datetime(d.time,utc=True)
    assert not idx.duplicated().any()
    x=pd.DataFrame({'power_kw_signed':d[powers].sum(axis=1,min_count=7).values,
                    'wind_speed':d[winds].mean(axis=1).where(d[winds].count(axis=1)==7).values},index=idx)
    # 10 source minutes per completed interval; gaps are not interpolated.
    g=x.resample('10min',closed='left',label='right')
    out=g.mean();cnt=g.count()
    out=out.where(cnt>=10);out['power_entities']=np.where(cnt.power_kw_signed>=10,7,0)
    out['wind_entities']=np.where(cnt.wind_speed>=10,7,0);out.index.name='time'
    meta['smarteole']=save('smarteole',out,14350.,[p],'https://zenodo.org/records/7342466',
        '7 measured active_power_*_avg (kW); 10 complete source minutes per interval',
        'held out from NEW protocol tuning; historically inspected; Feb-May 2020 campaign, publisher removed turbine-stop periods; not pristine or all-season validation')
    (OUT/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)

def save(site,out,rated,paths,url,semantics,role):
    assert out.index.is_unique and out.index.is_monotonic_increasing
    assert (out.index.to_series().diff().dropna()==pd.Timedelta(minutes=10)).all()
    # Task is nonnegative electrical generation; signed meter readings remain alongside target.
    out['power_kw']=out.power_kw_signed.clip(lower=0)
    out['power_pu']=out.power_kw/rated
    out['rated_kw']=rated
    path=OUT/f'{site}.csv';out.to_csv(path)
    m={'site':site,'rated_kw':rated,'target':'nonnegative measured generation max(signed Power,0), no upper clipping',
       'source':url,'semantics':semantics,'role':role,'frequency_minutes':10,
       'start':str(out.index.min()),'end':str(out.index.max()),'rows':len(out),
       'valid_target':int(out.power_pu.notna().sum()),'negative_signed_count':int((out.power_kw_signed<0).sum()),
       'above_nameplate_count':int((out.power_pu>1).sum()),
       'missing_target_fraction':float(out.power_pu.isna().mean()),
       'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
       'raw_files':[{'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    return m

if __name__=='__main__':prepare()
