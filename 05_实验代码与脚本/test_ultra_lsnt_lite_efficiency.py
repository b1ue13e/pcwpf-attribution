"""
Ultra-LSNT-Lite效率测试脚本
基于TimeMixer测试方式，提供与TimeMixer的公平对比
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

sys.path.append(os.getcwd())

# 导入必要模块
try:
    from ultra_lsnt_timeseries import (
        load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
        compute_metrics
    )
    print("✅ 成功导入 ultra_lsnt_timeseries 工具库")
except ImportError as e:
    print(f"❌ 导入超时序列模块时出错: {e}")
    sys.exit(1)

from ultra_lsnt_lite import UltraLSNTLiteForecaster

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
    
    return memory_mb, gpu_memory

def run_ultra_lsnt_lite_efficiency_test():
    print("==========================================================")
    print("🚀 启动Ultra-LSNT-Lite效率基准测试")
    print("==========================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    
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
    
    # 配置参数（与TimeMixer测试保持一致）
    seq_len = 96
    pred_len = 24
    batch_size = 256
    
    ts_config = TimeSeriesConfig(seq_len=seq_len, pred_len=pred_len, target='power')
    train_config = TrainConfig(batch_size=batch_size, epochs=3)  # 减少epochs用于基准测试
    
    # 创建数据加载器
    train_loader, _, test_loader, scaler = create_dataloaders(
        data, ts_config, train_config
    )
    
    # 1. 初始化Ultra-LSNT-Lite模型
    print("\n" + "="*60)
    print("1. 模型初始化与参数量统计")
    print("="*60)
    
    model_config = {
        'input_dim': input_dim,
        'hidden_dim': 128,
        'seq_len': seq_len,
        'pred_len': pred_len,
        'num_blocks': 2,  # 比原始Ultra-LSNT少
        'num_experts': 4,  # 比原始Ultra-LSNT少
        'top_k': 2,  # 比原始Ultra-LSNT少
        'scales': [1, 2, 4]  # 与TimeMixer相同的多尺度因子
    }
    
    model = UltraLSNTLiteForecaster(model_config).to(device)
    
    # 模型参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小 (MB): {total_params * 4 / (1024*1024):.2f}")
    
    # 2. 推理延迟测试
    print("\n" + "="*60)
    print("2. 推理延迟测试")
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
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.perf_counter()
            
            _ = model(bx)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.perf_counter()
            
            single_inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
            break  # 只测试一次
    
    single_inference_time = np.mean(single_inference_times) if single_inference_times else 0
    print(f"单次推理延迟: {single_inference_time:.2f} ms")
    
    # 批量推理测试
    print("批量推理测试...")
    batch_inference_times = []
    with torch.no_grad():
        for i, (bx, _) in enumerate(test_loader):
            if i >= 20:  # 限制测试批次数量
                break
            bx = bx.to(device)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.perf_counter()
            
            _ = model(bx)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.perf_counter()
            
            batch_inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    avg_inference_time = np.mean(batch_inference_times)
    std_inference_time = np.std(batch_inference_times)
    min_inference_time = np.min(batch_inference_times)
    max_inference_time = np.max(batch_inference_times)
    
    print(f"平均推理延迟: {avg_inference_time:.2f} ms (±{std_inference_time:.2f} ms)")
    print(f"最小推理延迟: {min_inference_time:.2f} ms")
    print(f"最大推理延迟: {max_inference_time:.2f} ms")
    print(f"测试批次数: {len(batch_inference_times)}")
    
    # 3. 内存占用测试
    print("\n" + "="*60)
    print("3. 内存占用测试")
    print("="*60)
    
    memory_before, gpu_memory_before = get_memory_usage()
    print(f"初始化后内存: {memory_before:.1f} MB, GPU内存: {gpu_memory_before:.1f} MB")
    
    # 进行一次推理以测量内存增量
    bx, _ = next(iter(test_loader))
    bx = bx.to(device)
    
    with torch.no_grad():
        _ = model(bx)
    
    memory_after, gpu_memory_after = get_memory_usage()
    print(f"推理后内存: {memory_after:.1f} MB, GPU内存: {gpu_memory_after:.1f} MB")
    print(f"内存增量: {memory_after - memory_before:.1f} MB")
    print(f"GPU内存增量: {gpu_memory_after - gpu_memory_before:.1f} MB")
    
    # 4. 训练时间测试（短时间）
    print("\n" + "="*60)
    print("4. 训练时间测试（3个epoch）")
    print("="*60)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epoch_times = []
    epoch_losses = []
    
    for epoch in range(3):
        epoch_start = time.perf_counter()
        epoch_loss = 0
        batch_count = 0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out, aux_loss = model(bx)
            loss = criterion(out, by) + 0.01 * aux_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        epoch_losses.append(epoch_loss / batch_count)
        
        print(f"  Epoch {epoch+1}/3: 训练时间: {epoch_time:.2f} 秒, 平均损失: {epoch_losses[-1]:.4f}")
    
    avg_epoch_time = np.mean(epoch_times)
    print(f"平均每个epoch训练时间: {avg_epoch_time:.2f} 秒")
    
    # 5. 保存结果
    print("\n" + "="*60)
    print("5. 保存测试结果")
    print("="*60)
    
    # 准备结果数据
    results = {
        "model": "Ultra-LSNT-Lite",
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024*1024),
        
        # 推理延迟
        "single_inference_time_ms": single_inference_time,
        "avg_inference_time_ms": avg_inference_time,
        "std_inference_time_ms": std_inference_time,
        "min_inference_time_ms": min_inference_time,
        "max_inference_time_ms": max_inference_time,
        "inference_batch_count": len(batch_inference_times),
        
        # 内存
        "memory_usage_mb": memory_after - memory_before,
        "gpu_memory_usage_mb": gpu_memory_after - gpu_memory_before,
        
        # 训练时间
        "avg_epoch_time_sec": avg_epoch_time,
        "total_training_time_sec": sum(epoch_times),
        
        # 配置
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "prediction_length": pred_len,
        "input_dimension": input_dim,
        "hidden_dim": model_config['hidden_dim'],
        "num_blocks": model_config['num_blocks'],
        "num_experts": model_config['num_experts'],
        "top_k": model_config['top_k'],
    }
    
    # 保存为CSV
    df = pd.DataFrame([results])
    output_csv = "ultra_lsnt_lite_efficiency_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ 效率测试结果已保存: {output_csv}")
    print(df.to_string())
    
    # 保存详细报告
    report_file = "ultra_lsnt_lite_efficiency_report.txt"
    with open(report_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("Ultra-LSNT-Lite 效率基准测试报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"测试时间: {results['timestamp']}\n")
        f.write(f"测试设备: {results['device']}\n\n")
        
        f.write("1. 模型统计\n")
        f.write("-"*40 + "\n")
        f.write(f"总参数量: {results['total_parameters']:,}\n")
        f.write(f"可训练参数量: {results['trainable_parameters']:,}\n")
        f.write(f"模型大小: {results['model_size_mb']:.2f} MB\n")
        f.write(f"隐藏维度: {results['hidden_dim']}\n")
        f.write(f"块数: {results['num_blocks']}\n")
        f.write(f"专家数: {results['num_experts']}\n")
        f.write(f"Top-K: {results['top_k']}\n\n")
        
        f.write("2. 推理延迟\n")
        f.write("-"*40 + "\n")
        f.write(f"单次推理延迟: {results['single_inference_time_ms']:.2f} ms\n")
        f.write(f"平均推理延迟: {results['avg_inference_time_ms']:.2f} ms\n")
        f.write(f"延迟标准差: {results['std_inference_time_ms']:.2f} ms\n")
        f.write(f"最小延迟: {results['min_inference_time_ms']:.2f} ms\n")
        f.write(f"最大延迟: {results['max_inference_time_ms']:.2f} ms\n")
        f.write(f"测试批次数: {results['inference_batch_count']}\n\n")
        
        f.write("3. 内存使用\n")
        f.write("-"*40 + "\n")
        f.write(f"内存增量: {results['memory_usage_mb']:.1f} MB\n")
        f.write(f"GPU内存增量: {results['gpu_memory_usage_mb']:.1f} MB\n\n")
        
        f.write("4. 训练效率\n")
        f.write("-"*40 + "\n")
        f.write(f"平均每个epoch训练时间: {results['avg_epoch_time_sec']:.2f} 秒\n")
        f.write(f"总训练时间: {results['total_training_time_sec']:.2f} 秒\n\n")
        
        f.write("5. 测试配置\n")
        f.write("-"*40 + "\n")
        f.write(f"批大小: {results['batch_size']}\n")
        f.write(f"输入序列长度: {results['sequence_length']}\n")
        f.write(f"预测长度: {results['prediction_length']}\n")
        f.write(f"输入维度: {results['input_dimension']}\n")
    
    print(f"✅ 详细报告已保存: {report_file}")
    
    print("\n" + "="*60)
    print("🎉 Ultra-LSNT-Lite效率测试完成!")
    print("="*60)
    
    return results

if __name__ == '__main__':
    run_ultra_lsnt_lite_efficiency_test()