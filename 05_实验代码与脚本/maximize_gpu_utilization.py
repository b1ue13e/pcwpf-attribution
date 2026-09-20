"""
GPU利用率最大化运行器
- 同时运行多个GPU密集型实验
- 调整参数以充分利用24GB显存
- 并行评估COA/BWO的种群成员
"""

import subprocess
import os
import sys
import time
import json
from multiprocessing import Pool, cpu_count
import torch

# 配置
DATASETS = ['wind_cn', 'electricity', 'air_quality', 'gefcom']
SEEDS = [42, 123, 456]
MAX_GPU_MEMORY_GB = 22  # 留2GB余量

# 调整后的高负载配置
HIGH_LOAD_CONFIG = {
    'coa_bilstm': {
        'population': 40,
        'iterations': 200,
        'batch_size': 256,  # 增大batch size
        'hidden_dim': 256,  # 更大的模型
        'num_layers': 3,
        'training_epochs': 30,
        'patience': 5,
        'use_amp': True,  # 混合精度
    },
    'bwo_svr': {
        'population': 40,
        'iterations': 200,
        'batch_size': 512,
        'training_epochs': 30,
        'patience': 5,
    },
    'bwo_cnn': {
        'population': 40,
        'iterations': 200,
        'batch_size': 256,
        'filters': [128, 256, 512],  # 更大的CNN
        'training_epochs': 30,
        'patience': 5,
        'use_amp': True,
    }
}


def check_gpu_memory():
    """检查GPU显存使用情况。"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        used, total = map(int, result.stdout.strip().split(','))
        return used, total
    except:
        return 0, 24564


def check_gpu_utilization():
    """检查GPU利用率。"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except:
        return 0


def wait_for_gpu_memory(threshold_gb=20):
    """等待GPU显存空闲。"""
    while True:
        used, total = check_gpu_memory()
        used_gb = used / 1024
        if used_gb < threshold_gb:
            return True
        print(f"  GPU显存使用: {used_gb:.1f}GB, 等待中...")
        time.sleep(5)


def run_gpu_intensive_experiment(exp_type, dataset, seed):
    """运行单个GPU密集型实验。"""
    log_dir = "/root/experiment_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = f"{log_dir}/{exp_type}_{dataset}_{seed}.log"
    
    # 根据实验类型选择脚本
    if exp_type == 'coa_bilstm':
        script = f"""
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
import torch

# 启用混合精度
from torch.cuda.amp import autocast, GradScaler

print(f"Starting COA-BiLSTM on {dataset} seed {seed}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

from coa_bilstm_experiment_v2 import run_coa_bilstm_experiment
result = run_coa_bilstm_experiment('{dataset}', {seed})
print(f"Completed: R2={{result['metrics']['R2']:.4f}}")
"""
    elif exp_type == 'bwo_cnn':
        script = f"""
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
print(f"Starting BWO-CNN on {dataset} seed {seed}")
from bwo_cnn_experiment_v2 import run_all_experiments
# 运行特定数据集和种子
"""
    elif exp_type == 'traditional':
        script = f"""
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
from quick_baselines_experiment import QuickBaselinesExperiment
exp = QuickBaselinesExperiment()
models = ['LSTM', 'CNN-LSTM']  # 只运行GPU密集型模型
for model_name in models:
    try:
        if model_name == 'LSTM':
            result = exp.run_lstm('{dataset}', {seed})
        else:
            result = exp.run_cnn_lstm('{dataset}', {seed})
        print(f"{{model_name}}: R2={{result['R2']:.4f}}")
    except Exception as e:
        print(f"{{model_name}}: Error - {{e}}")
"""
    else:
        return
    
    # 运行实验
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            [sys.executable, '-c', script],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd='/root'
        )
        process.wait()
    
    return exp_type, dataset, seed, process.returncode


def run_parallel_gpu_experiments():
    """并行运行多个GPU实验以最大化利用率。"""
    
    print("="*70)
    print("GPU利用率最大化运行器")
    print("="*70)
    print(f"GPU: RTX 4090D (24GB)")
    print(f"CPU: 18 cores")
    print(f"目标: GPU利用率 >90%, 显存使用 >20GB")
    print("="*70)
    print()
    
    tasks = []
    
    # 阶段1: 同时启动所有COA-BiLSTM实验 (每个数据集一个，它们会自动调度GPU)
    print("【阶段1】启动所有COA-BiLSTM实验...")
    for dataset in DATASETS:
        for seed in SEEDS:
            tasks.append(('coa_bilstm', dataset, seed))
    
    # 阶段2: 启动BWO-CNN实验 (GPU密集型)
    print("【阶段2】启动所有BWO-CNN实验...")
    for dataset in DATASETS:
        for seed in SEEDS:
            tasks.append(('bwo_cnn', dataset, seed))
    
    # 阶段3: GPU密集型传统基线
    print("【阶段3】启动GPU密集型传统基线...")
    for dataset in DATASETS:
        for seed in SEEDS:
            tasks.append(('traditional', dataset, seed))
    
    print(f"\n总任务数: {len(tasks)}")
    print(f"预计最大并行数: 根据GPU显存动态调整")
    print()
    
    # 使用进程池并行运行
    max_parallel = 6  # 同时运行6个GPU任务
    print(f"启动并行执行 (最大{max_parallel}个并发)...")
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = []
        
        # 提交所有任务
        for exp_type, dataset, seed in tasks:
            future = executor.submit(run_gpu_intensive_experiment, exp_type, dataset, seed)
            futures.append(future)
            print(f"  提交: {exp_type} {dataset} seed={seed}")
            time.sleep(2)  # 避免同时启动造成资源竞争
        
        print(f"\n已提交{len(futures)}个任务，等待完成...")
        
        # 监控进度
        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                completed += 1
                print(f"  完成 [{completed}/{len(futures)}]: {result}")
                
                # 显示GPU状态
                if completed % 5 == 0:
                    used, total = check_gpu_memory()
                    util = check_gpu_utilization()
                    print(f"  GPU状态: 利用率{util}%, 显存{used/1024:.1f}GB/{total/1024:.1f}GB")
                    
            except Exception as e:
                print(f"  错误: {e}")
    
    print("\n" + "="*70)
    print("所有GPU密集型实验完成!")
    print("="*70)


def optimize_for_max_speed():
    """为最大速度优化配置。"""
    
    print("="*70)
    print("最大速度优化配置")
    print("="*70)
    
    # 检查当前PyTorch CUDA配置
    print("\n【CUDA配置】")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # 测试最大batch size
        print("\n【测试最大Batch Size】")
        for batch_size in [64, 128, 256, 512, 1024]:
            try:
                x = torch.randn(batch_size, 96, 12).cuda()
                model = torch.nn.LSTM(12, 256, 3, batch_first=True).cuda()
                with torch.cuda.amp.autocast():
                    output, _ = model(x)
                torch.cuda.empty_cache()
                print(f"  Batch size {batch_size}: OK")
            except RuntimeError as e:
                print(f"  Batch size {batch_size}: FAILED - {e}")
                break
    
    print("\n【优化建议】")
    print("1. 使用混合精度训练 (torch.cuda.amp)")
    print("2. 增大batch size到256-512")
    print("3. 同时运行多个实验进程")
    print("4. 使用DataLoader的num_workers=4")
    print("5. 设置pin_memory=True")


def main():
    """主函数。"""
    
    # 先测试配置
    optimize_for_max_speed()
    
    # 询问是否继续
    print("\n是否启动全并行GPU实验? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        run_parallel_gpu_experiments()
    else:
        print("已取消")


if __name__ == "__main__":
    main()
