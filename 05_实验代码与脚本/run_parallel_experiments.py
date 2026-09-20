"""
并行实验运行器 - 充分利用GPU
同时运行多个数据集的实验以提高GPU利用率
"""

import subprocess
import sys
import time
import os
import json
from multiprocessing import Process, Queue
import threading

# 实验配置
DATASETS = ['wind_cn', 'electricity', 'air_quality', 'gefcom']
SEEDS = [42, 123, 456]
MAX_PARALLEL = 3  # 同时运行的最大进程数（根据GPU内存调整）


def run_traditional_single(dataset, seed, queue):
    """运行单个传统基线实验。"""
    script = f"""
import sys
sys.path.append('/root')
from quick_baselines_experiment import QuickBaselinesExperiment
exp = QuickBaselinesExperiment()

# 只运行指定数据集和种子
models = ['ARIMA', 'SVR', 'LSTM', 'CNN-LSTM']
results = []

for model_name in models:
    print(f"Running {{model_name}} on {dataset} (seed={seed})...")
    if model_name == 'ARIMA':
        result = exp.run_arima('{dataset}', {seed})
    elif model_name == 'SVR':
        result = exp.run_svr('{dataset}', {seed})
    elif model_name == 'LSTM':
        result = exp.run_lstm('{dataset}', {seed})
    elif model_name == 'CNN-LSTM':
        result = exp.run_cnn_lstm('{dataset}', {seed})
    
    if result:
        results.append(result)
        print(f"  Result: R2={{result['R2']:.4f}}")

# 保存结果
import pandas as pd
if results:
    df = pd.DataFrame(results)
    df.to_csv(f'/root/experiment_logs/traditional_{dataset}_{seed}.csv', index=False)
    print(f"Saved traditional_{dataset}_{seed}.csv")
"""
    
    log_file = f"/root/experiment_logs/traditional_{dataset}_{seed}.log"
    with open(log_file, 'w') as f:
        f.write(f"Starting {dataset} seed {seed}\n")
        process = subprocess.Popen(
            [sys.executable, '-c', script],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd='/root'
        )
        process.wait()
        f.write(f"Completed with status {process.returncode}\n")
    
    queue.put((dataset, seed, process.returncode))


def run_coa_bilstm_single(dataset, seed, queue):
    """运行单个COA-BiLSTM实验。"""
    # 创建临时脚本
    temp_script = f"""
import sys
sys.path.append('/root')
import torch

# 设置GPU设备（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

from coa_bilstm_experiment_v2 import run_coa_bilstm_experiment

try:
    result = run_coa_bilstm_experiment('{dataset}', {seed})
    print(f"Completed {dataset} seed {seed}: R2={{result['metrics']['R2']:.4f}}")
except Exception as e:
    print(f"Error: {{e}}")
    import traceback
    traceback.print_exc()
"""
    
    log_file = f"/root/experiment_logs/coa_{dataset}_{seed}.log"
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            [sys.executable, '-c', temp_script],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd='/root',
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
        )
        process.wait()
    
    queue.put(('COA', dataset, seed, process.returncode))


def parallel_runner(target_func, tasks, max_parallel=3):
    """并行运行任务。"""
    from multiprocessing import Process, Queue
    
    queue = Queue()
    running = []
    completed = []
    remaining = list(tasks)
    
    print(f"Starting parallel execution: {len(tasks)} tasks, max {max_parallel} parallel")
    
    while remaining or running:
        # 启动新任务
        while len(running) < max_parallel and remaining:
            task = remaining.pop(0)
            p = Process(target=target_func, args=(*task, queue))
            p.start()
            running.append((p, task))
            print(f"  Started {task} (PID: {p.pid})")
            time.sleep(5)  # 避免同时启动导致资源竞争
        
        # 检查完成的任务
        for p, task in list(running):
            if not p.is_alive():
                running.remove((p, task))
                completed.append(task)
                try:
                    result = queue.get_nowait()
                    print(f"  Completed {task}: {result}")
                except:
                    print(f"  Completed {task}")
        
        time.sleep(2)
    
    print(f"All {len(completed)} tasks completed")
    return completed


def run_all_traditional_parallel():
    """并行运行所有传统基线实验。"""
    print("\n" + "="*70)
    print("并行运行传统基线实验")
    print("="*70)
    
    tasks = []
    for dataset in DATASETS:
        for seed in SEEDS:
            tasks.append((dataset, seed))
    
    parallel_runner(run_traditional_single, tasks, max_parallel=MAX_PARALLEL)
    
    # 合并结果
    print("\n合并所有结果...")
    import pandas as pd
    
    all_results = []
    for dataset in DATASETS:
        for seed in SEEDS:
            csv_file = f"/root/experiment_logs/traditional_{dataset}_{seed}.csv"
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                all_results.append(df)
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv('/root/traditional_baselines_80_20.csv', index=False)
        print(f"Saved traditional_baselines_80_20.csv ({len(combined)} rows)")


def run_all_coa_parallel():
    """并行运行所有COA-BiLSTM实验。"""
    print("\n" + "="*70)
    print("并行运行 COA-BiLSTM 实验")
    print("="*70)
    
    tasks = []
    for dataset in DATASETS:
        for seed in SEEDS:
            tasks.append((dataset, seed))
    
    # COA-BiLSTM 需要更多GPU内存，减少并行度
    parallel_runner(run_coa_bilstm_single, tasks, max_parallel=2)


def main():
    """主函数。"""
    print("="*70)
    print("GPU并行实验运行器")
    print("="*70)
    print(f"GPU: RTX 4090 D (24GB)")
    print(f"Max parallel tasks: {MAX_PARALLEL}")
    print(f"Datasets: {DATASETS}")
    print(f"Seeds: {SEEDS}")
    
    # 1. 传统基线（可以高度并行）
    run_all_traditional_parallel()
    
    # 2. COA-BiLSTM（适度并行）
    run_all_coa_parallel()
    
    print("\n" + "="*70)
    print("所有实验完成!")
    print("="*70)


if __name__ == "__main__":
    main()
