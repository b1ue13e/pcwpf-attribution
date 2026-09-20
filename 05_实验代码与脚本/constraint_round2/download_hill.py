from pathlib import Path
import concurrent.futures,hashlib,json,requests
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'04_数据集/constraint_rebuild_v2/raw_hill_of_towie'
def download(f):
    p=OUT/f['key'];expected=f['checksum'].split(':')[-1]
    def digest(path):
        h=hashlib.md5()
        with path.open('rb') as stream:
            for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
        return h.hexdigest()
    if p.exists() and digest(p)==expected:return p.name+' verified existing'
    url=f"https://zenodo.org/records/20204946/files/{p.name}?download=1"
    part=p.with_suffix(p.suffix+'.part')
    with requests.get(url,stream=True,timeout=(20,120)) as r:
        r.raise_for_status()
        with part.open('wb') as s:
            for b in r.iter_content(4*1024*1024):s.write(b)
    assert digest(part)==expected,p
    part.replace(p);return p.name+' verified downloaded'
if __name__=='__main__':
    d=json.loads((OUT/'record.json').read_text(encoding='utf-8'))
    files=[f for f in d['files'] if f['key'] in ['2016.zip','2017.zip','README.md','Hill_of_Towie_turbine_metadata.csv','Hill_of_Towie_turbine_fields_description.csv','Hill_of_Towie_tables_description.csv']]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for future in concurrent.futures.as_completed([pool.submit(download,f) for f in files]):print(future.result(),flush=True)
