from pathlib import Path
import argparse,concurrent.futures,os,subprocess,sys
B=Path(__file__).resolve().parent.parent
ENV=dict(os.environ,PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',LD_LIBRARY_PATH='/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--external',required=True);p.add_argument('--development',required=True);a=p.parse_args()
    def train(job):
        seed,gpu=job;env=dict(ENV,CUDA_VISIBLE_DEVICES=str(gpu))
        with (B/'logs'/f'{a.external}_{seed}.log').open('w') as f:
            return subprocess.run([sys.executable,'-u',str(B/'code/core.py'),'--site',a.external,'--backbone','dlinear','--seed',str(seed),'--data',str(B/'data'),'--out',str(B/'results')],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    with concurrent.futures.ThreadPoolExecutor(4) as pool:r=list(pool.map(train,[(s,i%4) for i,s in enumerate([42,123,456,789,1024])]))
    assert all(code==0 for code in r),r
    for site in [a.development,a.external]:
        with (B/'logs'/f'{site}_assessment.log').open('w') as f:
            rc=subprocess.run([sys.executable,'-u',str(B/'code/assess.py'),'--site',site,'--data',str(B/'data'),'--root',str(B/'results')],env=dict(ENV,CUDA_VISIBLE_DEVICES='0'),stdout=f,stderr=subprocess.STDOUT).returncode
        assert rc==0,site
        print('completed',site,flush=True)
