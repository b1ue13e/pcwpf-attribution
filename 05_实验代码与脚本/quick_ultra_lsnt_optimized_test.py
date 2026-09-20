"""
快速Ultra-LSNT-Lite优化测试
跳过编译的autotune阶段，直接测量推理延迟
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

sys.path.append(os.getcwd())
from ultra_lsnt_lite import UltraLSNTLiteForecaster

# 禁用torch.compile的autotune
os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '0'

def quick_optimized_test():
    print("🚀 快速Ultra-LSNT-Lite优化测试")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 优化配置
    config = {
        'input_dim': 8,
        'hidden_dim': 96,
        'seq_len': 96,
        'pred_len': 24,
        'num_blocks': 1,
        'num_experts': 2,
        'top_k': 1,
        'scales': [1, 2]
    }
    
    # 创建模型
    model = UltraLSNTLiteForecaster(config).to(device)
    model.eval()
    
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    print(f"模型大小: {total_params * 4 / (1024*1024):.2f} MB")
    
    # 使用torch.compile但不进行autotune
    print("应用torch.compile（无autotune）...")
    try:
        model = torch.compile(model, mode='max-autotune')
    except Exception as e:
        print(f"编译失败: {e}")
    
    # 创建测试输入
    batch_size = 256
    test_input = torch.randn(batch_size, config['seq_len'], config['input_dim']).to(device)
    
    # 预热
    print("预热...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_input)
    
    # 测量推理延迟
    print("测量推理延迟...")
    inference_times = []
    with torch.no_grad():
        for _ in range(20):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            
            _ = model(test_input)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            
            inference_times.append((end - start) * 1000)  # ms
    
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    print(f"\n优化后推理延迟:")
    print(f"  平均: {avg_time:.2f} ms (±{std_time:.2f} ms)")
    print(f"  最小: {min_time:.2f} ms")
    print(f"  最大: {max_time:.2f} ms")
    print(f"  测试次数: {len(inference_times)}")
    
    # 能耗估算
    # RTX 4090 D: 推理功耗~150W
    power_watts = 150.0
    energy_per_inference_mj = power_watts * (avg_time / 1000) * 1000
    
    # 保存结果
    results = {
        "model": "Ultra-LSNT-Lite-Optimized-Quick",
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_parameters": total_params,
        "avg_inference_time_ms": avg_time,
        "std_inference_time_ms": std_time,
        "min_inference_time_ms": min_time,
        "max_inference_time_ms": max_time,
        "estimated_energy_per_inference_mj": energy_per_inference_mj,
        "hidden_dim": config['hidden_dim'],
        "num_blocks": config['num_blocks'],
        "num_experts": config['num_experts'],
        "top_k": config['top_k'],
    }
    
    df = pd.DataFrame([results])
    output_csv = "ultra_lsnt_lite_optimized_quick_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✅ 快速测试结果已保存: {output_csv}")
    
    return results

if __name__ == '__main__':
    quick_optimized_test()