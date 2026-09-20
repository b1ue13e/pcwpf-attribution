"""Bounded GPU worker queue, isolated to this run directory; resume-safe."""
from pathlib import Path
import argparse,concurrent.futures,json,os,queue,subprocess,sys
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--backbone',required=True);ap.add_argument('--gpus',type=int,required=True)
    ap.add_argument('--sites',nargs='+',default=['kelmarsh','penmanshiel']);ap.add_argument('--evaluate',action='store_true');a=ap.parse_args()
    cfg=json.loads((HERE/'protocol.json').read_text());q=queue.Queue()
    for site in a.sites:
        for seed in cfg['seed_list']:q.put((site,seed))
    (BASE/'logs').mkdir(exist_ok=True)
    def worker(gpu):
        while True:
            try:site,seed=q.get_nowait()
            except queue.Empty:return
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     LD_LIBRARY_PATH='/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
            cmd=[sys.executable,'-u',str(HERE/('evaluate.py' if a.evaluate else 'benchmark.py')),'--site',site,'--seed',str(seed),'--backbone',a.backbone,
                 '--data',str(BASE/'data'),'--root' if a.evaluate else '--out',str(BASE/'results')]
            with (BASE/'logs'/f'{site}_{a.backbone}_{seed}{"_eval" if a.evaluate else ""}.log').open('a') as log:
                rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            print(json.dumps({'site':site,'seed':seed,'backbone':a.backbone,'gpu':gpu,'returncode':rc}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(a.gpus) as pool:list(pool.map(worker,range(a.gpus)))
