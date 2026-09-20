"""Download the next complete calendar year, validating the publisher checksum."""
from pathlib import Path
import concurrent.futures, hashlib, json, requests

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'04_数据集/open_cross_site_wind'
def download(job):
    record,folder,f=job
    dest=RAW/folder/'raw'/f['key']
    expected=f['checksum'].split(':')[-1]
    def md5(p):
        h=hashlib.md5()
        with p.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    if dest.exists() and md5(dest)==expected:return f'{dest.name}: verified existing'
    url=f"https://zenodo.org/records/{record}/files/{f['key']}?download=1"
    tmp=dest.with_suffix('.zip.part')
    with requests.get(url,stream=True,timeout=(20,90)) as r:
        r.raise_for_status()
        with tmp.open('wb') as stream:
            for chunk in r.iter_content(1024*1024):stream.write(chunk)
    assert md5(tmp)==expected, f'Checksum mismatch: {dest}'
    tmp.replace(dest)
    return f'{dest.name}: downloaded and checksum verified'
if __name__=='__main__':
    jobs=[]
    for record,folder in [('5841834','kelmarsh_uk'),('5946808','penmanshiel_scotland')]:
        d=json.loads((RAW/f'_manifests/zenodo_{record}.json').read_text(encoding='utf-8-sig'))
        jobs.extend((record,folder,f) for f in d['files'] if 'SCADA_2017' in f['key'])
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        for result in pool.map(download,jobs):print(result,flush=True)
