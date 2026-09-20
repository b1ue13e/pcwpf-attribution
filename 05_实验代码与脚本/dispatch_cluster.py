"""
Distributed 3D Controlled Ablation Matrix Dispatcher.
Utilizes all 10x RTX 4090 GPUs across Node 1 (port 25329) and Node 2 (port 25979).

Allocation:
Node 1 (4 GPUs):
  - GPU 0: kelmarsh, seed 42
  - GPU 1: kelmarsh, seed 43
  - GPU 2: kelmarsh, seed 44
  - GPU 3: kelmarsh, seed 45

Node 2 (6 GPUs):
  - GPU 0: hill_of_towie, seed 42
  - GPU 1: hill_of_towie, seed 43
  - GPU 2: hill_of_towie, seed 44
  - GPU 3: penmanshiel, seed 42
  - GPU 4: penmanshiel, seed 43
  - GPU 5: penmanshiel, seed 44
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import paramiko
from cluster_runner import NODES, get_client, run_cmd

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / '05_实验代码与脚本/constraint_round2'
DATA_DIR = ROOT / '04_数据集/constraint_rebuild_v2'
PRICE_FILE = ROOT / '04_数据集/elexon_system_prices_2016_2024.csv'
LOCAL_RES_DIR = ROOT / '06_实验结果与日志/ablation_3d_results'
LOCAL_RES_DIR.mkdir(parents=True, exist_ok=True)

REMOTE_BASE = '/root/paper_ablation_matrix'
REMOTE_DATA = f"{REMOTE_BASE}/data"
REMOTE_CODE = f"{REMOTE_BASE}/code"
REMOTE_OUT = f"{REMOTE_BASE}/output"

JOB_SPEC = {
    'node1': [
        {'site': 'kelmarsh', 'seed': 42, 'gpu': 0},
        {'site': 'kelmarsh', 'seed': 43, 'gpu': 1},
        {'site': 'kelmarsh', 'seed': 44, 'gpu': 2},
        {'site': 'kelmarsh', 'seed': 45, 'gpu': 3},
    ],
    'node2': [
        {'site': 'hill_of_towie', 'seed': 42, 'gpu': 0},
        {'site': 'hill_of_towie', 'seed': 43, 'gpu': 1},
        {'site': 'hill_of_towie', 'seed': 44, 'gpu': 2},
        {'site': 'penmanshiel', 'seed': 42, 'gpu': 3},
        {'site': 'penmanshiel', 'seed': 43, 'gpu': 4},
        {'site': 'penmanshiel', 'seed': 44, 'gpu': 5},
    ]
}

def sync_payload(node_name):
    print(f"[{node_name}] Preparing directories and syncing payload...")
    run_cmd(node_name, f"mkdir -p {REMOTE_DATA} {REMOTE_CODE} {REMOTE_OUT}")
    
    client = get_client(node_name)
    sftp = client.open_sftp()
    try:
        # Sync Data
        for site in ['kelmarsh', 'hill_of_towie', 'penmanshiel']:
            loc_f = DATA_DIR / f"{site}.csv"
            rem_f = f"{REMOTE_DATA}/{site}.csv"
            print(f"[{node_name}] Uploading {loc_f.name}...")
            sftp.put(str(loc_f), rem_f)
            
        print(f"[{node_name}] Uploading prices CSV...")
        sftp.put(str(PRICE_FILE), f"{REMOTE_DATA}/elexon_system_prices_2016_2024.csv")
        
        # Sync Code
        code_files = [
            'core.py', 'protocol.json', 'controlled_ablation.py',
            'audit_evaluator.py', 'run_cluster_ablation.py'
        ]
        for cf in code_files:
            loc_c = CODE_DIR / cf
            rem_c = f"{REMOTE_CODE}/{cf}"
            sftp.put(str(loc_c), rem_c)
            
        print(f"[{node_name}] Sync complete.")
    finally:
        sftp.close()
        client.close()

def launch_jobs(node_name):
    jobs = JOB_SPEC[node_name]
    print(f"[{node_name}] Generating launch_all.sh and starting {len(jobs)} parallel GPU jobs...")
    
    script_lines = ["#!/bin/bash"]
    if node_name == 'node1':
        script_lines.append("export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:$LD_LIBRARY_PATH")
    script_lines.append(f"cd {REMOTE_CODE}")
    
    for j in jobs:
        site = j['site']
        seed = j['seed']
        gpu = j['gpu']
        log_file = f"{REMOTE_OUT}/job_{site}_s{seed}_gpu{gpu}.log"
        cmd = (
            f"python3 run_cluster_ablation.py "
            f"--site {site} --seed {seed} --gpu {gpu} "
            f"--data_path {REMOTE_DATA} "
            f"--prices_path {REMOTE_DATA}/elexon_system_prices_2016_2024.csv "
            f"--out_dir {REMOTE_OUT} </dev/null > {log_file} 2>&1 &"
        )
        script_lines.append(cmd)
        
    script_content = "\n".join(script_lines) + "\n"
    
    client = get_client(node_name)
    sftp = client.open_sftp()
    remote_sh = f"{REMOTE_BASE}/launch_all.sh"
    with sftp.open(remote_sh, 'w') as f:
        f.write(script_content)
    sftp.chmod(remote_sh, 0o755)
    sftp.close()
    client.close()
    
    run_cmd(node_name, f"bash {remote_sh}")
    print(f"[{node_name}] Successfully launched all {len(jobs)} GPU jobs.")

def monitor_progress():
    print("\n=== Monitoring 10 GPU Jobs across Cluster ===")
    total_expected = sum(len(jobs) for jobs in JOB_SPEC.values())
    
    start_time = time.time()
    while True:
        completed = 0
        status_reports = []
        for name in NODES:
            # Check CSV count
            out, _ = run_cmd(name, f"find {REMOTE_OUT} -name 'ablation_results_*.csv' | wc -l")
            n_done = int(out.strip() or 0)
            completed += n_done
            
            # Check running python processes
            out_ps, _ = run_cmd(name, "ps aux | grep 'run_cluster_ablation.py' | grep -v grep | wc -l")
            n_running = int(out_ps.strip() or 0)
            
            # Check sample log tail
            tail, _ = run_cmd(name, f"tail -n 1 {REMOTE_OUT}/*.log 2>/dev/null | grep -E '(COMPLETED|Arm)' | tail -n 2")
            status_reports.append(f"[{name.upper()}] Done: {n_done} | Running: {n_running} | Recent: {tail.strip()}")
            
        elapsed = int(time.time() - start_time)
        print(f"[{elapsed}s elapsed] Completed {completed}/{total_expected} jobs. Status:\n  " + "\n  ".join(status_reports), flush=True)
        
        if completed >= total_expected:
            print(f"\nAll {total_expected} jobs completed successfully in {elapsed} seconds!", flush=True)
            break
            
        time.sleep(10)

def collect_results():
    print("\nDownloading result CSVs from cluster...")
    for name in NODES:
        client = get_client(name)
        sftp = client.open_sftp()
        try:
            out_files, _ = run_cmd(name, f"find {REMOTE_OUT} -name 'ablation_results_*.csv'")
            for line in out_files.strip().split('\n'):
                if not line.strip():
                    continue
                rem_p = line.strip()
                fname = os.path.basename(rem_p)
                loc_p = LOCAL_RES_DIR / fname
                sftp.get(rem_p, str(loc_p))
                print(f"[{name}] Downloaded {fname}")
        finally:
            sftp.close()
            client.close()

def aggregate_summary():
    print("\nAggregating final results...")
    csvs = list(LOCAL_RES_DIR.glob('ablation_results_*.csv'))
    if not csvs:
        print("No CSV files found!")
        return
    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    summary_path = LOCAL_RES_DIR / 'ablation_3d_master_summary.csv'
    df.to_csv(summary_path, index=False)
    print(f"Saved master summary with {len(df)} rows to {summary_path}")
    
    # Print key insights: comparison of calibrated Pinball Loss and static cost for ratio=2 (q=0.333)
    p2 = df[(df.ratio == 2) & (df.policy == 'calibrated')].copy()
    grouped = p2.groupby(['site', 'arm']).agg(
        cost_mean=('cost_static', 'mean'),
        cost_std=('cost_static', 'std'),
        pinball_mean=('pinball_loss', 'mean'),
        rmse_mean=('rmse', 'mean'),
        cashflow_mean=('net_cashflow_per_turb_yr', 'mean'),
        shortfall_mean=('shortfall_per_turb_yr', 'mean')
    ).round(6)
    
    table_path = LOCAL_RES_DIR / 'ablation_3d_key_comparison_table.csv'
    grouped.to_csv(table_path)
    print(f"Saved key comparison table to {table_path}")
    print("\n=== Key Ablation Comparison (Ratio 2 / Quantile 0.333, Calibrated) ===")
    print(grouped.to_string())

if __name__ == '__main__':
    # Step 1: Sync to both nodes
    for name in NODES:
        sync_payload(name)
        
    # Step 2: Clean previous remote runs
    for name in NODES:
        run_cmd(name, f"rm -rf {REMOTE_OUT}/*")
        
    # Step 3: Launch across 10 GPUs
    for name in NODES:
        launch_jobs(name)
        
    # Step 4: Monitor until completion
    monitor_progress()
    
    # Step 5: Download results
    collect_results()
    
    # Step 6: Master aggregation & insight table
    aggregate_summary()
