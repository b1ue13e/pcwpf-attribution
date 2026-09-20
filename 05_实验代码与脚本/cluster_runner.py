"""
Cluster Runner Utility for Distributed Research Experiments.
Manages compute nodes:
- Node 1: 192.168.17.251:25329 (4x RTX 4090, 192 CPUs, 503GB RAM)
- Node 2: 192.168.17.251:25979 (6x RTX 4090, 192 CPUs, 503GB RAM)
Password: root / 061022
"""

import os
from pathlib import Path
import paramiko

NODES = {
    'node1': {'host': '192.168.17.251', 'port': 25329, 'user': 'root', 'pass': '061022', 'gpus': 4},
    'node2': {'host': '192.168.17.251', 'port': 25979, 'user': 'root', 'pass': '061022', 'gpus': 6}
}

def get_client(node_name):
    cfg = NODES[node_name]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(cfg['host'], port=cfg['port'], username=cfg['user'], password=cfg['pass'], timeout=10)
    return client

def run_cmd(node_name, cmd):
    client = get_client(node_name)
    try:
        if node_name == 'node1':
            cmd = "export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:$LD_LIBRARY_PATH && " + cmd
        stdin, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        return out, err
    finally:
        client.close()

def sync_dir(node_name, local_dir, remote_dir):
    client = get_client(node_name)
    sftp = client.open_sftp()
    try:
        # ensure remote dir exists
        run_cmd(node_name, f"mkdir -p {remote_dir}")
        for root, _, files in os.walk(local_dir):
            rel = os.path.relpath(root, local_dir).replace('\\', '/')
            rem_sub = f"{remote_dir}/{rel}" if rel != '.' else remote_dir
            run_cmd(node_name, f"mkdir -p {rem_sub}")
            for f in files:
                loc_path = os.path.join(root, f)
                rem_path = f"{rem_sub}/{f}"
                sftp.put(loc_path, rem_path)
    finally:
        sftp.close()
        client.close()

def check_env():
    for name in NODES:
        print(f"=== Compute Node: {name.upper()} ({NODES[name]['host']}:{NODES[name]['port']}) ===")
        cmd = "python3 -c \"import torch; print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | GPU Count: {torch.cuda.device_count()}')\""
        out, err = run_cmd(name, cmd)
        print(out.strip())
        out_gpus, _ = run_cmd(name, "nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader")
        for line in out_gpus.strip().split('\n'):
            print(f"  GPU {line.strip()}")
        if err.strip():
            print("Warning:", err.strip())
        print()

if __name__ == '__main__':
    check_env()
