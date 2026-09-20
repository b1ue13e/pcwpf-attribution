"""
DLinear推理延迟与能耗测试脚本
基于TimeMixer效率测试模板，为DLinear提供全面的效率基准测试
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import psutil
import gc
from pathlib import Path

sys.path.append(os.getcwd())

# 导入DLinear模型
try:
    from run_dlinear import DLinear
    print("✅ 成功导入 DLinear 模型")
except ImportError as e:
    print(f"❌ 导入DLinear时出错: {e}")
    # 如果导入失败，直接在此定义DLinear
    class MovingAvg(nn.Module):
        def __init__(self, kernel_size, stride):
            super().__init__()
            self.kernel_size = kernel_size
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

        def forward(self, x):
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=1)
            x = x.permute(0, 2, 1)
            x = self.avg(x)
            x = x.permute(0, 2, 1)
            return x

    class SeriesDecomp(nn.Module):
        def __init__(self, kernel_size):
            super().__init__()
            self.moving_avg = MovingAvg(kernel_size, stride=1)

        def forward(self, x):
            moving_mean = self.moving_avg(x)
            res = x - moving_mean
            return res, moving_mean

    class DLinear(nn.Module):
        def __init__(self, seq_len, pred_len, enc_in):
            super().__init__()
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.decompsition = SeriesDecomp(25)
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)

        def forward(self, x):
            seasonal_init, trend_init = self.decompsition(x)
            seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            x = seasonal_output + trend_output
            return x.permute(0, 2, 1)[:, :, -1]  # 预测单变量

# 导入数据加载工具
try:
    from ultra_lsnt_timeseries import (
        load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
        compute_metrics
    )
    print("✅ 成功导入 ultra_lsnt_timeseries 工具库")
except ImportError as e:
    print(f"❌ 导入超时序列模块时出错: {e}")
    sys.exit(1)

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
    
    return memory_mb, gpu_memory

def estimate_energy_consumption(model, input_tensor, device):
    """
    估算能耗（基于FLOPs和硬件假设）
    """
    try:
        from thop import profile
        # 计算FLOPs
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        
        # 假设硬件配置
        if device.type == 'cuda':
            # RTX 4090D: 约 50 TFLOPs, 功耗 450W
            tflops = 50.0
            power_watts = 450.0
        else:
            # CPU假设: 0.5 TFLOPs, 功耗 65W (桌面CPU)
            # 树莓派4B: 0.005 TFLOPs (5 GFLOPS), 功耗 3W
            tflops = 0.005  # 树莓派4B: 5 GFLOPS
            power_watts = 3.0
        
        # 计算理论推理时间（秒）
        inference_time_s = flops / (tflops * 1e12)
        # 计算能耗（焦耳）
        energy_joules = power_watts * inference_time_s
        
        return {
            "flops": flops,
            "params": params,
            "estimated_inference_time_s": inference_time_s,
            "estimated_energy_joules": energy_joules,
            "estimated_energy_mj": energy_joules * 1000
        }
    except ImportError:
        print("⚠️  未安装thop，跳过FLOPs和能耗估算")
        return None

def run_dlinear_efficiency_benchmark():
    print("==========================================================")
    print("🚀 启动DLinear效率基准测试")
    print("==========================================================")
    
    # 强制使用CPU以模拟树莓派环境
    device = torch.device('cpu')
    print(f"测试设备: {device} (模拟树莓派环境)")
    print("注意: 使用CPU模式模拟树莓派4B Cortex-A72环境")
    
    # 加载数据
    print("\n📊 加载数据...")
    data_path = 'wind_final.csv'
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        # 创建虚拟数据用于测试
        print("使用虚拟数据进行测试")
        data = np.random.randn(1000, 8).astype(np.float32)
    else:
        data, _ = load_csv_data(data_path, 'power')
    
    input_dim = data.shape[1]
    print(f"数据形状: {data.shape}, 输入维度: {input_dim}")
    
    # 配置参数
    seq_len = 96
    pred_len = 24
    batch_size = 1  # 树莓派上通常批量大小为1
    
    ts_config = TimeSeriesConfig(seq_len=seq_len, pred_len=pred_len, target='power')
    train_config = TrainConfig(batch_size=batch_size, epochs=1)  # 仅用于创建数据加载器
    
    # 创建数据加载器
    train_loader, _, test_loader, scaler = create_dataloaders(
        data, ts_config, train_config
    )
    
    # 1. 初始化DLinear模型
    print("\n" + "="*60)
    print("1. 模型初始化与参数量统计")
    print("="*60)
    
    model = DLinear(seq_len, pred_len, input_dim).to(device)
    
    # 模型参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小 (MB): {total_params * 4 / (1024*1024):.2f}")  # 假设float32
    
    # 2. 推理延迟测试
    print("\n" + "="*60)
    print("2. 推理延迟测试 (CPU模拟树莓派)")
    print("="*60)
    
    model.eval()
    
    # 预热
    print("预热模型...")
    with torch.no_grad():
        for bx, _ in train_loader:
            bx = bx.to(device)
            _ = model(bx)
            break
    
    # 单次推理测试
    print("单次推理测试...")
    single_inference_times = []
    with torch.no_grad():
        for bx, _ in train_loader:
            bx = bx.to(device)
            
            # 多次测量取平均
            for _ in range(50):
                start_time = time.perf_counter()
                _ = model(bx)
                end_time = time.perf_counter()
                single_inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
            break  # 只测试一个批次
    
    single_inference_time = np.mean(single_inference_times) if single_inference_times else 0
    single_inference_std = np.std(single_inference_times) if single_inference_times else 0
    print(f"单次推理延迟: {single_inference_time:.2f} ms (±{single_inference_std:.2f} ms)")
    print(f"最小延迟: {np.min(single_inference_times):.2f} ms")
    print(f"最大延迟: {np.max(single_inference_times):.2f} ms")
    
    # 批量推理测试（小批量）
    print("小批量推理测试 (batch=4)...")
    batch_size = 4
    test_config = TrainConfig(batch_size=batch_size)
    _, _, batch_loader, _ = create_dataloaders(data, ts_config, test_config)
    
    batch_inference_times = []
    with torch.no_grad():
        for i, (bx, _) in enumerate(batch_loader):
            if i >= 10:  # 限制测试批次数量
                break
            bx = bx.to(device)
            
            start_time = time.perf_counter()
            _ = model(bx)
            end_time = time.perf_counter()
            
            batch_inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    avg_inference_time = np.mean(batch_inference_times) if batch_inference_times else 0
    std_inference_time = np.std(batch_inference_times) if batch_inference_times else 0
    min_inference_time = np.min(batch_inference_times) if batch_inference_times else 0
    max_inference_time = np.max(batch_inference_times) if batch_inference_times else 0
    
    print(f"批量推理延迟 (batch={batch_size}): {avg_inference_time:.2f} ms (±{std_inference_time:.2f} ms)")
    print(f"最小延迟: {min_inference_time:.2f} ms")
    print(f"最大延迟: {max_inference_time:.2f} ms")
    print(f"测试批次数: {len(batch_inference_times)}")
    
    # 3. 内存占用测试
    print("\n" + "="*60)
    print("3. 内存占用测试")
    print("="*60)
    
    memory_before, gpu_memory_before = get_memory_usage()
    print(f"初始化后内存: {memory_before:.1f} MB, GPU内存: {gpu_memory_before:.1f} MB")
    
    # 进行一次推理以测量内存增量
    bx, _ = next(iter(batch_loader))
    bx = bx.to(device)
    
    with torch.no_grad():
        _ = model(bx)
    
    memory_after, gpu_memory_after = get_memory_usage()
    print(f"推理后内存: {memory_after:.1f} MB, GPU内存: {gpu_memory_after:.1f} MB")
    print(f"内存增量: {memory_after - memory_before:.1f} MB")
    print(f"GPU内存增量: {gpu_memory_after - gpu_memory_before:.1f} MB")
    
    # 4. 能耗估算
    print("\n" + "="*60)
    print("4. 能耗估算（基于理论计算）")
    print("="*60)
    
    # 创建测试输入
    test_input = torch.randn(1, seq_len, input_dim).to(device)
    energy_estimation = estimate_energy_consumption(model, test_input, device)
    
    if energy_estimation:
        print(f"FLOPs: {energy_estimation['flops']:,.0f}")
        print(f"参数量: {energy_estimation['params']:,.0f}")
        print(f"估算推理时间: {energy_estimation['estimated_inference_time_s']*1000:.4f} ms")
        print(f"估算能耗 (J): {energy_estimation['estimated_energy_joules']:.6f}")
        print(f"估算能耗 (mJ): {energy_estimation['estimated_energy_mj']:.2f}")
    else:
        print("⚠️  无法进行详细的能耗估算（需要安装thop库）")
    
    # 树莓派4B具体估算
    print("\n树莓派4B详细估算:")
    print("硬件假设: Cortex-A72 @ 1.5GHz, 5 GFLOPS, 功耗 3W")
    
    # 基于实测延迟计算能耗
    pi_power_watts = 3.0  # 树莓派4B满载功耗
    pi_energy_per_inference_j = pi_power_watts * (single_inference_time / 1000)  # 功率×时间=能量
    pi_energy_per_inference_mj = pi_energy_per_inference_j * 1000
    
    print(f"实测单次推理延迟: {single_inference_time:.2f} ms")
    print(f"树莓派单次推理能耗: {pi_energy_per_inference_mj:.2f} mJ")
    print(f"理论最大推理频率: {1000/single_inference_time:.1f} Hz")
    
    # 5. 保存结果
    print("\n" + "="*60)
    print("5. 保存测试结果")
    print("="*60)
    
    # 准备结果数据
    results = {
        "model": "DLinear",
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024*1024),
        
        # 推理延迟
        "single_inference_time_ms": single_inference_time,
        "single_inference_std_ms": single_inference_std,
        "avg_inference_time_ms": avg_inference_time,
        "std_inference_time_ms": std_inference_time,
        "min_inference_time_ms": min_inference_time,
        "max_inference_time_ms": max_inference_time,
        "inference_batch_count": len(batch_inference_times),
        
        # 内存
        "memory_usage_mb": memory_after - memory_before,
        "gpu_memory_usage_mb": gpu_memory_after - gpu_memory_before,
        
        # 树莓派能耗
        "raspberry_pi_energy_mj": pi_energy_per_inference_mj,
        "raspberry_pi_power_watts": pi_power_watts,
        "raspberry_pi_inference_frequency_hz": 1000/single_inference_time if single_inference_time > 0 else 0,
        
        # 配置
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "prediction_length": pred_len,
        "input_dimension": input_dim,
    }
    
    # 添加能耗估算数据
    if energy_estimation:
        results.update({
            "flops": float(energy_estimation['flops']) if energy_estimation['flops'] else 0,
            "estimated_inference_time_ms": energy_estimation['estimated_inference_time_s'] * 1000 if energy_estimation['estimated_inference_time_s'] else 0,
            "estimated_energy_mj": energy_estimation['estimated_energy_mj'] if energy_estimation['estimated_energy_mj'] else 0,
        })
    
    # 保存为CSV
    df = pd.DataFrame([results])
    output_csv = "dlinear_latency_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ 效率测试结果已保存: {output_csv}")
    
    # 保存详细报告
    report_file = "dlinear_latency_report.txt"
    with open(report_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("DLinear 推理延迟与能耗测试报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"测试时间: {results['timestamp']}\n")
        f.write(f"测试设备: {results['device']} (模拟树莓派环境)\n\n")
        
        f.write("1. 模型统计\n")
        f.write("-"*40 + "\n")
        f.write(f"总参数量: {results['total_parameters']:,}\n")
        f.write(f"可训练参数量: {results['trainable_parameters']:,}\n")
        f.write(f"模型大小: {results['model_size_mb']:.2f} MB\n\n")
        
        f.write("2. 推理延迟 (树莓派模拟)\n")
        f.write("-"*40 + "\n")
        f.write(f"单次推理延迟: {results['single_inference_time_ms']:.2f} ms (±{results['single_inference_std_ms']:.2f} ms)\n")
        f.write(f"批量推理延迟: {results['avg_inference_time_ms']:.2f} ms (±{results['std_inference_time_ms']:.2f} ms)\n")
        f.write(f"最小延迟: {results['min_inference_time_ms']:.2f} ms\n")
        f.write(f"最大延迟: {results['max_inference_time_ms']:.2f} ms\n\n")
        
        f.write("3. 树莓派能耗分析\n")
        f.write("-"*40 + "\n")
        f.write(f"单次推理能耗: {results['raspberry_pi_energy_mj']:.2f} mJ\n")
        f.write(f"功耗: {results['raspberry_pi_power_watts']:.1f} W\n")
        f.write(f"理论最大推理频率: {results['raspberry_pi_inference_frequency_hz']:.1f} Hz\n")
        f.write(f"每小时最大推理次数: {results['raspberry_pi_inference_frequency_hz']*3600:.0f}\n\n")
        
        f.write("4. 内存使用\n")
        f.write("-"*40 + "\n")
        f.write(f"内存增量: {results['memory_usage_mb']:.1f} MB\n")
        f.write(f"GPU内存增量: {results['gpu_memory_usage_mb']:.1f} MB\n\n")
        
        f.write("5. 测试配置\n")
        f.write("-"*40 + "\n")
        f.write(f"批大小: {results['batch_size']}\n")
        f.write(f"输入序列长度: {results['sequence_length']}\n")
        f.write(f"预测长度: {results['prediction_length']}\n")
        f.write(f"输入维度: {results['input_dimension']}\n")
    
    print(f"✅ 详细报告已保存: {report_file}")
    
    # 与现有Ultra-LSNT数据对比
    print("\n" + "="*60)
    print("6. 与Ultra-LSNT延迟对比 (参考数据)")
    print("="*60)
    
    # 读取现有的Ultra-LSNT延迟数据
    ultra_lsnt_latency = 15.29  # ms (来自timemixer_ultralsn_lite_comparison_report.md)
    time_mixer_latency = 3.36   # ms
    
    print(f"DLinear延迟: {single_inference_time:.2f} ms")
    print(f"Ultra-LSNT延迟 (GPU): {ultra_lsnt_latency:.2f} ms")
    print(f"TimeMixer延迟 (GPU): {time_mixer_latency:.2f} ms")
    
    if single_inference_time > 0:
        ratio_vs_ultralsnt = single_inference_time / ultra_lsnt_latency
        ratio_vs_timemixer = single_inference_time / time_mixer_latency
        print(f"DLinear/Ultra-LSNT延迟比: {ratio_vs_ultralsnt:.2f}x")
        print(f"DLinear/TimeMixer延迟比: {ratio_vs_timemixer:.2f}x")
        
        # 保存对比数据
        comparison_file = "dlinear_vs_ultralsnt_comparison.txt"
        with open(comparison_file, "w") as f:
            f.write("DLinear vs Ultra-LSNT 延迟对比分析\n")
            f.write("="*50 + "\n")
            f.write(f"DLinear (CPU模拟树莓派): {single_inference_time:.2f} ms\n")
            f.write(f"Ultra-LSNT (GPU RTX 4090): {ultra_lsnt_latency:.2f} ms\n")
            f.write(f"TimeMixer (GPU RTX 4090): {time_mixer_latency:.2f} ms\n")
            f.write(f"延迟比 DLinear/Ultra-LSNT: {ratio_vs_ultralsnt:.2f}x\n")
            f.write(f"延迟比 DLinear/TimeMixer: {ratio_vs_timemixer:.2f}x\n")
            f.write("\n注: 对比需考虑硬件差异 (CPU vs GPU)\n")
        
        print(f"✅ 对比分析已保存: {comparison_file}")
    
    print("\n" + "="*60)
    print("🎉 DLinear效率测试完成!")
    print("="*60)

if __name__ == '__main__':
    run_dlinear_efficiency_benchmark()