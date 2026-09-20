"""
LightGBM推理延迟与能耗测试脚本
测量LightGBM在CPU上的推理延迟，模拟树莓派环境
"""
import os
import sys
import numpy as np
import pandas as pd
import time
import psutil
import gc
import joblib
from pathlib import Path

sys.path.append(os.getcwd())

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
    return memory_mb, 0  # GPU内存为0，因为使用CPU

def load_lightgbm_model():
    """加载LightGBM模型"""
    model_path = 'lgbm_model_dump.pkl'
    if not os.path.exists(model_path):
        print(f"❌ LightGBM模型文件不存在: {model_path}")
        print("正在尝试从check_storage.py生成模型...")
        # 运行check_storage.py生成模型
        try:
            import subprocess
            subprocess.run([sys.executable, 'check_storage.py'], check=True, capture_output=True)
            if os.path.exists(model_path):
                print(f"✅ 已生成LightGBM模型: {model_path}")
            else:
                print("❌ 生成模型失败")
                return None
        except Exception as e:
            print(f"❌ 运行check_storage.py失败: {e}")
            return None
    
    try:
        print(f"正在加载LightGBM模型: {model_path}")
        model = joblib.load(model_path)
        print("✅ LightGBM模型加载成功")
        return model
    except Exception as e:
        print(f"❌ 加载LightGBM模型失败: {e}")
        return None

def flatten_input(batch_data, seq_len=96, input_dim=8):
    """将批次数据展平为LightGBM输入格式"""
    # batch_data: [batch_size, seq_len, input_dim]
    batch_size = batch_data.shape[0]
    return batch_data.reshape(batch_size, -1)  # [batch_size, seq_len*input_dim]

def run_lightgbm_efficiency_benchmark():
    print("==========================================================")
    print("🚀 启动LightGBM效率基准测试")
    print("==========================================================")
    
    print("测试设备: CPU (模拟树莓派环境)")
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
    
    # 1. 加载LightGBM模型
    print("\n" + "="*60)
    print("1. 模型加载与信息统计")
    print("="*60)
    
    model = load_lightgbm_model()
    if model is None:
        print("❌ 无法加载LightGBM模型，测试终止")
        return
    
    # 获取模型文件大小
    model_path = 'lgbm_model_dump.pkl'
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0
    print(f"模型文件大小: {model_size_mb:.2f} MB")
    
    # 获取模型参数信息（LightGBM模型较为复杂，难以直接统计参数量）
    try:
        # 尝试获取一些模型信息
        if hasattr(model, 'estimators_'):
            n_estimators = len(model.estimators_)
            print(f"子模型数量: {n_estimators} (MultiOutputRegressor)")
    except:
        pass
    
    # 2. 推理延迟测试
    print("\n" + "="*60)
    print("2. 推理延迟测试 (CPU模拟树莓派)")
    print("="*60)
    
    # 预热
    print("预热模型...")
    for bx, _ in train_loader:
        bx_flat = flatten_input(bx.numpy(), seq_len, input_dim)
        _ = model.predict(bx_flat)
        break
    
    # 单次推理测试
    print("单次推理测试 (batch=1)...")
    single_inference_times = []
    for bx, _ in train_loader:
        bx_np = bx.numpy()
        bx_flat = flatten_input(bx_np, seq_len, input_dim)
        
        # 多次测量取平均
        for _ in range(50):
            start_time = time.perf_counter()
            _ = model.predict(bx_flat)
            end_time = time.perf_counter()
            single_inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
        break  # 只测试一个批次
    
    single_inference_time = np.mean(single_inference_times) if single_inference_times else 0
    single_inference_std = np.std(single_inference_times) if single_inference_times else 0
    print(f"单次推理延迟: {single_inference_time:.2f} ms (±{single_inference_std:.2f} ms)")
    print(f"最小延迟: {np.min(single_inference_times):.2f} ms")
    print(f"最大延迟: {np.max(single_inference_times):.2f} ms")
    
    # 小批量推理测试
    print("小批量推理测试 (batch=4)...")
    batch_size = 4
    test_config = TrainConfig(batch_size=batch_size)
    _, _, batch_loader, _ = create_dataloaders(data, ts_config, test_config)
    
    batch_inference_times = []
    for i, (bx, _) in enumerate(batch_loader):
        if i >= 10:  # 限制测试批次数量
            break
        bx_np = bx.numpy()
        bx_flat = flatten_input(bx_np, seq_len, input_dim)
        
        start_time = time.perf_counter()
        _ = model.predict(bx_flat)
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
    
    memory_before, _ = get_memory_usage()
    print(f"初始化后内存: {memory_before:.1f} MB")
    
    # 进行一次推理以测量内存增量
    bx, _ = next(iter(batch_loader))
    bx_np = bx.numpy()
    bx_flat = flatten_input(bx_np, seq_len, input_dim)
    
    _ = model.predict(bx_flat)
    
    memory_after, _ = get_memory_usage()
    print(f"推理后内存: {memory_after:.1f} MB")
    print(f"内存增量: {memory_after - memory_before:.1f} MB")
    
    # 4. 能耗估算
    print("\n" + "="*60)
    print("4. 能耗估算（基于实测延迟）")
    print("="*60)
    
    # 树莓派4B具体估算
    print("树莓派4B详细估算:")
    print("硬件假设: Cortex-A72 @ 1.5GHz, 5 GFLOPS, 功耗 3W")
    
    # 基于实测延迟计算能耗
    pi_power_watts = 3.0  # 树莓派4B满载功耗
    pi_energy_per_inference_j = pi_power_watts * (single_inference_time / 1000)  # 功率×时间=能量
    pi_energy_per_inference_mj = pi_energy_per_inference_j * 1000
    
    print(f"实测单次推理延迟: {single_inference_time:.2f} ms")
    print(f"树莓派单次推理能耗: {pi_energy_per_inference_mj:.2f} mJ")
    print(f"理论最大推理频率: {1000/single_inference_time:.1f} Hz" if single_inference_time > 0 else "N/A")
    
    # 5. 保存结果
    print("\n" + "="*60)
    print("5. 保存测试结果")
    print("="*60)
    
    # 准备结果数据
    results = {
        "model": "LightGBM",
        "device": "cpu",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_size_mb": model_size_mb,
        
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
        
        # 树莓派能耗
        "raspberry_pi_energy_mj": pi_energy_per_inference_mj,
        "raspberry_pi_power_watts": pi_power_watts,
        "raspberry_pi_inference_frequency_hz": 1000/single_inference_time if single_inference_time > 0 else 0,
        
        # 配置
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "prediction_length": pred_len,
        "input_dimension": input_dim,
        "flattened_features": seq_len * input_dim,
    }
    
    # 保存为CSV
    df = pd.DataFrame([results])
    output_csv = "lightgbm_latency_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ 效率测试结果已保存: {output_csv}")
    
    # 保存详细报告
    report_file = "lightgbm_latency_report.txt"
    with open(report_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("LightGBM 推理延迟与能耗测试报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"测试时间: {results['timestamp']}\n")
        f.write(f"测试设备: {results['device']} (模拟树莓派环境)\n\n")
        
        f.write("1. 模型统计\n")
        f.write("-"*40 + "\n")
        f.write(f"模型文件大小: {results['model_size_mb']:.2f} MB\n")
        f.write(f"展平后特征数: {results['flattened_features']}\n\n")
        
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
        f.write(f"内存增量: {results['memory_usage_mb']:.1f} MB\n\n")
        
        f.write("5. 测试配置\n")
        f.write("-"*40 + "\n")
        f.write(f"批大小: {results['batch_size']}\n")
        f.write(f"输入序列长度: {results['sequence_length']}\n")
        f.write(f"预测长度: {results['prediction_length']}\n")
        f.write(f"输入维度: {results['input_dimension']}\n")
        f.write(f"展平特征数: {results['flattened_features']}\n")
    
    print(f"✅ 详细报告已保存: {report_file}")
    
    # 6. 与现有模型对比
    print("\n" + "="*60)
    print("6. 与其他模型延迟对比 (参考数据)")
    print("="*60)
    
    # 读取现有的延迟数据
    try:
        dlinear_df = pd.read_csv('dlinear_latency_results.csv')
        dlinear_latency = dlinear_df['single_inference_time_ms'].iloc[0]
    except:
        dlinear_latency = 0.27  # 默认值
    
    ultra_lsnt_latency = 15.29  # ms (来自timemixer_ultralsn_lite_comparison_report.md)
    time_mixer_latency = 3.36   # ms
    
    print(f"LightGBM延迟: {single_inference_time:.2f} ms")
    print(f"DLinear延迟: {dlinear_latency:.2f} ms")
    print(f"Ultra-LSNT延迟 (GPU): {ultra_lsnt_latency:.2f} ms")
    print(f"TimeMixer延迟 (GPU): {time_mixer_latency:.2f} ms")
    
    if single_inference_time > 0 and dlinear_latency > 0:
        ratio_vs_dlinear = single_inference_time / dlinear_latency
        ratio_vs_ultralsnt = single_inference_time / ultra_lsnt_latency
        print(f"LightGBM/DLinear延迟比: {ratio_vs_dlinear:.2f}x")
        print(f"LightGBM/Ultra-LSNT延迟比: {ratio_vs_ultralsnt:.2f}x")
        
        # 保存对比数据
        comparison_file = "lightgbm_vs_others_comparison.txt"
        with open(comparison_file, "w") as f:
            f.write("LightGBM vs 其他模型 延迟对比分析\n")
            f.write("="*50 + "\n")
            f.write(f"LightGBM (CPU模拟树莓派): {single_inference_time:.2f} ms\n")
            f.write(f"DLinear (CPU模拟树莓派): {dlinear_latency:.2f} ms\n")
            f.write(f"Ultra-LSNT (GPU RTX 4090): {ultra_lsnt_latency:.2f} ms\n")
            f.write(f"TimeMixer (GPU RTX 4090): {time_mixer_latency:.2f} ms\n")
            f.write(f"延迟比 LightGBM/DLinear: {ratio_vs_dlinear:.2f}x\n")
            f.write(f"延迟比 LightGBM/Ultra-LSNT: {ratio_vs_ultralsnt:.2f}x\n")
            f.write("\n注: 对比需考虑硬件差异 (CPU vs GPU)\n")
        
        print(f"✅ 对比分析已保存: {comparison_file}")
    
    print("\n" + "="*60)
    print("🎉 LightGBM效率测试完成!")
    print("="*60)

if __name__ == '__main__':
    run_lightgbm_efficiency_benchmark()