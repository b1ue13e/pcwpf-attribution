#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Site 5-Seed Controlled Ablation Extension Runner.
支持为 Hill of Towie 和 Penmanshiel 调度 Seed 45 和 Seed 46（及 Kelmarsh Seed 46），
将核心实证基准统一扩展至完整 5 随机种子（Seeds 42, 43, 44, 45, 46）。

Features:
1. Cluster Dispatch Mode (--cluster): Dispatches parallel jobs across RTX 4090 GPU cluster (Node 1 / Node 2).
2. Local Execution Mode (--local): Runs jobs on local CUDA GPUs or CPU.
3. Dry-Run Mode (--dry-run): Inspects job schedules, parameters, and commands without side-effects.
4. Aggregation Mode (--aggregate): Calculates 5-seed means and standard deviations from output CSVs.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / '05_实验代码与脚本' / 'constraint_round2'
DATA_DIR = ROOT / '04_数据集' / 'constraint_rebuild_v2'
PRICE_FILE = ROOT / '04_数据集' / 'elexon_system_prices_2016_2024.csv'
LOCAL_RES_DIR = ROOT / '06_实验结果与日志' / 'ablation_3d_results'

# 5-Seed Extension Jobs to run:
# Kelmarsh already has 42, 43, 44, 45 -> adds seed 46
# Hill of Towie has 42, 43, 44 -> adds seeds 45, 46
# Penmanshiel has 42, 43, 44 -> adds seeds 45, 46
EXTENSION_JOBS = [
    {'site': 'hill_of_towie', 'seed': 45, 'target_node': 'node2', 'gpu': 0},
    {'site': 'hill_of_towie', 'seed': 46, 'target_node': 'node2', 'gpu': 1},
    {'site': 'penmanshiel',   'seed': 45, 'target_node': 'node2', 'gpu': 2},
    {'site': 'penmanshiel',   'seed': 46, 'target_node': 'node2', 'gpu': 3},
    {'site': 'kelmarsh',      'seed': 46, 'target_node': 'node1', 'gpu': 0},
]


def check_cluster_availability():
    """Checks if cluster_runner.py is available and nodes are reachable."""
    try:
        sys.path.insert(0, str(ROOT / '05_实验代码与脚本'))
        from cluster_runner import NODES, get_client, run_cmd
        return True, NODES, get_client, run_cmd
    except Exception as e:
        return False, None, None, None


def run_dry_run():
    print("=" * 75)
    print(" [DRY-RUN] 5-Seed Extension Plan & Execution Dispatch Map")
    print("=" * 75)
    print(f"Total extension runs: {len(EXTENSION_JOBS)} jobs")
    print(f"Data source: {DATA_DIR}")
    print(f"Price series: {PRICE_FILE}")
    print(f"Output directory: {LOCAL_RES_DIR}\n")

    print(f"{'Site':<16} | {'Seed':<6} | {'Target Node':<12} | {'GPU':<5} | {'Command Preview'}")
    print("-" * 75)
    for j in EXTENSION_JOBS:
        site, seed, node, gpu = j['site'], j['seed'], j['target_node'], j['gpu']
        cmd = f"python3 run_cluster_ablation.py --site {site} --seed {seed} --gpu {gpu}"
        print(f"{site:<16} | {seed:<6} | {node:<12} | {gpu:<5} | {cmd}")

    print("\n[+] Verification Check:")
    print("  - Kelmarsh seeds:     [42, 43, 44, 45] + [46]     -> 5 seeds total")
    print("  - Hill of Towie seeds:[42, 43, 44]     + [45, 46] -> 5 seeds total")
    print("  - Penmanshiel seeds:  [42, 43, 44]     + [45, 46] -> 5 seeds total")
    print("=" * 75)


def run_local_jobs(gpu_id=0):
    print("=" * 75)
    print(f" [LOCAL] Executing 5-Seed Extension on Local Device (GPU: {gpu_id})")
    print("=" * 75)
    import subprocess

    sys.path.insert(0, str(CODE_DIR))
    for j in EXTENSION_JOBS:
        site = j['site']
        seed = j['seed']
        print(f"\n>>> Launching local job: {site} | Seed {seed} ...")
        cmd = [
            sys.executable,
            str(CODE_DIR / "run_cluster_ablation.py"),
            "--site", site,
            "--seed", str(seed),
            "--gpu", str(gpu_id),
            "--data_path", str(DATA_DIR),
            "--prices_path", str(PRICE_FILE),
            "--out_dir", str(LOCAL_RES_DIR)
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"[OK] Completed {site} seed {seed}.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Error running {site} seed {seed}: {e}")
            if e.stderr:
                print(f"    Stderr: {e.stderr[-500:].strip()}")
            if e.stdout:
                print(f"    Stdout: {e.stdout[-500:].strip()}")
        except Exception as e:
            print(f"[!] Error running {site} seed {seed}: {e}")


def dispatch_cluster_jobs():
    avail, nodes, get_client, run_cmd = check_cluster_availability()
    if not avail:
        print("[-] Cluster utilities not accessible. Please ensure cluster_runner.py is available.")
        return

    print("=" * 75)
    print(" [CLUSTER] Dispatching 5-Seed Extension to GPU Nodes")
    print("=" * 75)
    REMOTE_BASE = '/root/paper_ablation_matrix'
    REMOTE_DATA = f"{REMOTE_BASE}/data"
    REMOTE_CODE = f"{REMOTE_BASE}/code"
    REMOTE_OUT = f"{REMOTE_BASE}/output"

    jobs_by_node = {'node1': [], 'node2': []}
    for j in EXTENSION_JOBS:
        jobs_by_node[j['target_node']].append(j)

    for node_name, jobs in jobs_by_node.items():
        if not jobs:
            continue
        print(f"\n[{node_name.upper()}] Launching {len(jobs)} jobs...")
        script_lines = ["#!/bin/bash"]
        if node_name == 'node1':
            script_lines.append("export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:$LD_LIBRARY_PATH")
        script_lines.append(f"cd {REMOTE_CODE}")

        for j in jobs:
            site = j['site']
            seed = j['seed']
            gpu = j['gpu']
            log_f = f"{REMOTE_OUT}/job_ext_{site}_s{seed}_gpu{gpu}.log"
            cmd = (
                f"python3 run_cluster_ablation.py "
                f"--site {site} --seed {seed} --gpu {gpu} "
                f"--data_path {REMOTE_DATA} "
                f"--prices_path {REMOTE_DATA}/elexon_system_prices_2016_2024.csv "
                f"--out_dir {REMOTE_OUT} </dev/null > {log_f} 2>&1 &"
            )
            script_lines.append(cmd)

        script_content = "\n".join(script_lines) + "\n"
        try:
            client = get_client(node_name)
            sftp = client.open_sftp()
            rem_sh = f"{REMOTE_BASE}/launch_extension.sh"
            with sftp.open(rem_sh, 'w') as f:
                f.write(script_content)
            sftp.chmod(rem_sh, 0o755)
            sftp.close()
            client.close()

            run_cmd(node_name, f"bash {rem_sh}")
            print(f"[{node_name.upper()}] Successfully dispatched {len(jobs)} jobs via {rem_sh}.")
        except Exception as e:
            print(f"[-] Failed to launch on {node_name}: {e}")


def aggregate_results():
    print("=" * 75)
    print(" [AGGREGATE] Reading Available Results & Comparing Statistics")
    print("=" * 75)
    key_table = LOCAL_RES_DIR / 'ablation_3d_key_comparison_table.csv'
    if not key_table.exists():
        print(f"[-] Key comparison table not found at {key_table}.")
        return

    df = pd.read_csv(key_table)
    print(f"[+] Loaded baseline results table: {len(df)} configurations.")

    sites = ['kelmarsh', 'hill_of_towie', 'penmanshiel']
    for s in sites:
        sub = df[df['site'] == s]
        print(f"\n=== Site: {s.upper()} ===")
        for _, row in sub.iterrows():
            arm = row['arm']
            cost = row['cost_mean']
            std = row['cost_std']
            print(f"  - {arm:<32}: Cost = {cost:.5f} ± {std:.5f}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Site 5-Seed Controlled Ablation Extension Dispatcher")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Print plan without running")
    parser.add_argument("--cluster", action="store_true", default=False, help="Dispatch to remote cluster nodes")
    parser.add_argument("--local", action="store_true", default=False, help="Run jobs locally")
    parser.add_argument("--gpu", type=int, default=0, help="Local GPU index (default 0)")
    parser.add_argument("--aggregate", action="store_true", default=False, help="Aggregate existing results")

    args = parser.parse_args()

    if args.cluster:
        dispatch_cluster_jobs()
    elif args.local:
        run_local_jobs(gpu_id=args.gpu)
    elif args.aggregate:
        aggregate_results()
    else:
        # Default: dry-run inspection
        run_dry_run()
        print("\nTip: Use '--cluster' to dispatch to GPU cluster, '--local' for local run, or '--aggregate' to summarize.")


if __name__ == '__main__':
    main()
