"""
Fix protobuf mismatch on Node 2 and execute remaining Hill of Towie & Penmanshiel jobs.
Also download Node 1's completed Kelmarsh results.
"""

import os
import sys
import time
from pathlib import Path
import paramiko
from cluster_runner import NODES, get_client, run_cmd
from dispatch_cluster import JOB_SPEC, REMOTE_BASE, REMOTE_DATA, REMOTE_CODE, REMOTE_OUT, LOCAL_RES_DIR, collect_results, aggregate_summary

def fix_protobuf_node2():
    print("Fixing protobuf on Node 2...")
    out, err = run_cmd('node2', 'pip install "protobuf<4.21.0" -q')
    print("Pip result:", out.strip())
    test_cmd = """python3 -c "import torch; opt = torch.optim.Adam([torch.nn.Parameter(torch.randn(2,2))]); print('Adam test SUCCESS on Node 2!')" """
    out_test, err_test = run_cmd('node2', test_cmd)
    print(out_test.strip())
    if err_test.strip():
        print("Warning/Err:", err_test.strip())

def launch_node2():
    print("Launching Node 2 jobs (Hill of Towie & Penmanshiel across 6 GPUs)...")
    run_cmd('node2', f"rm -rf {REMOTE_OUT}/*")
    
    jobs = JOB_SPEC['node2']
    script_lines = [
        "#!/bin/bash",
        "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python",
        f"cd {REMOTE_CODE}"
    ]
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
    
    client = get_client('node2')
    sftp = client.open_sftp()
    remote_sh = f"{REMOTE_BASE}/launch_all.sh"
    with sftp.open(remote_sh, 'w') as f:
        f.write(script_content)
    sftp.chmod(remote_sh, 0o755)
    sftp.close()
    client.close()
    
    run_cmd('node2', f"bash {remote_sh}")
    print(f"Successfully started all {len(jobs)} jobs on Node 2.")

def monitor_node2():
    print("\nMonitoring Node 2 jobs...")
    total_expected = len(JOB_SPEC['node2'])
    start_time = time.time()
    
    while True:
        out, _ = run_cmd('node2', f"find {REMOTE_OUT} -name 'ablation_results_*.csv' | wc -l")
        n_done = int(out.strip() or 0)
        
        out_ps, _ = run_cmd('node2', "ps aux | grep 'run_cluster_ablation.py' | grep -v grep | wc -l")
        n_running = int(out_ps.strip() or 0)
        
        tail, _ = run_cmd('node2', f"tail -n 1 {REMOTE_OUT}/*.log 2>/dev/null | grep -E '(COMPLETED|Arm)' | tail -n 3")
        
        elapsed = int(time.time() - start_time)
        print(f"[{elapsed}s] Node 2 Done: {n_done}/{total_expected} | Running: {n_running} | Recent:\n{tail.strip()}", flush=True)
        
        if n_done >= total_expected:
            print(f"\nNode 2 all {total_expected} jobs completed in {elapsed}s!", flush=True)
            break
            
        if n_running == 0 and n_done < total_expected and elapsed > 20:
            print("\nWarning: No jobs running but not all finished! Checking logs:")
            err_log, _ = run_cmd('node2', f"cat {REMOTE_OUT}/*.log | tail -n 20")
            print(err_log)
            break
            
        time.sleep(10)

if __name__ == '__main__':
    fix_protobuf_node2()
    launch_node2()
    monitor_node2()
    collect_results()
    aggregate_summary()
