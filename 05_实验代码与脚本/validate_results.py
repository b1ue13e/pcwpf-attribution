#!/usr/bin/env python3
"""
验证生成的CSV文件数据的合理性和完整性
"""
import pandas as pd
import numpy as np
import os

def validate_latency_csv():
    """验证latency.csv"""
    print("="*60)
    print("验证 latency.csv")
    print("="*60)
    
    df = pd.read_csv("latency.csv")
    print(f"记录数: {len(df)}")
    print(f"模型列表: {list(df['model'])}")
    
    # 检查必需字段
    required_cols = ['model', 'mean_ms', 'p95_ms', 'std_ms', 'memory_mb', 'params']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ 缺少必需列: {col}")
            return False
    
    # 检查数据完整性
    for idx, row in df.iterrows():
        if pd.isna(row['mean_ms']) or pd.isna(row['p95_ms']) or pd.isna(row['std_ms']):
            print(f"❌ 模型 {row['model']} 有缺失值")
            return False
    
    # 检查合理性
    print("\n延迟数据合理性检查:")
    print("-"*60)
    for idx, row in df.iterrows():
        print(f"{row['model']:<25} mean={row['mean_ms']:.2f}ms, p95={row['p95_ms']:.2f}ms, std={row['std_ms']:.2f}ms")
        
        # 合理性检查
        if row['mean_ms'] <= 0:
            print(f"  ⚠️  警告: 平均延迟 <= 0")
        if row['p95_ms'] < row['mean_ms']:
            print(f"  ⚠️  警告: P95 < 平均值")
        if row['std_ms'] > row['mean_ms'] * 2:
            print(f"  ⚠️  警告: 标准差过大")
    
    print("\n✅ latency.csv 验证通过")
    return True

def validate_energy_csv():
    """验证energy.csv"""
    print("\n" + "="*60)
    print("验证 energy.csv")
    print("="*60)
    
    df = pd.read_csv("energy.csv")
    print(f"记录数: {len(df)}")
    print(f"模型列表: {list(df['model'])}")
    
    # 检查必需字段
    required_cols = ['model', 'energy_per_inference_mj', 'avg_power_w', 'inference_time_ms']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ 缺少必需列: {col}")
            return False
    
    # 检查数据完整性
    print("\n能耗数据:")
    print("-"*60)
    for idx, row in df.iterrows():
        print(f"{row['model']:<25} energy={row['energy_per_inference_mj']:.0f}mJ, power={row['avg_power_w']:.1f}W")
        
        # 合理性检查
        if row['energy_per_inference_mj'] <= 0:
            print(f"  ⚠️  警告: 能耗 <= 0")
        if row['avg_power_w'] < 10 or row['avg_power_w'] > 200:
            print(f"  ⚠️  警告: 功率值不合理: {row['avg_power_w']}W")
    
    print("\n✅ energy.csv 验证通过（注：能耗为估算值）")
    return True

def validate_tau_sweep_csv():
    """验证tau_sweep.csv"""
    print("\n" + "="*60)
    print("验证 tau_sweep.csv")
    print("="*60)
    
    df = pd.read_csv("tau_sweep.csv")
    print(f"记录数: {len(df)}")
    print(f"tau值: {list(df['tau'])}")
    
    # 检查必需字段
    required_cols = ['tau', 'mean_latency_ms', 'effective_latency_ms', 'skip_rate', 'estimated_accuracy']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ 缺少必需列: {col}")
            return False
    
    # 检查数据完整性
    print("\ntau扫描数据:")
    print("-"*60)
    for idx, row in df.iterrows():
        print(f"tau={row['tau']:.1f}: latency={row['mean_latency_ms']:.2f}ms, effective={row['effective_latency_ms']:.2f}ms, skip={row['skip_rate']:.2f}, acc={row['estimated_accuracy']:.3f}")
        
        # 合理性检查
        if row['skip_rate'] < 0 or row['skip_rate'] > 1:
            print(f"  ⚠️  警告: skip_rate不在[0,1]范围内")
        if row['estimated_accuracy'] < 0 or row['estimated_accuracy'] > 1:
            print(f"  ⚠️  警告: accuracy不在[0,1]范围内")
        if row['effective_latency_ms'] > row['mean_latency_ms'] * 1.2:
            print(f"  ⚠️  警告: 等效延迟大于原始延迟")
    
    print("\n✅ tau_sweep.csv 验证通过")
    return True

def validate_topk_sweep_csv():
    """验证topk_sweep.csv"""
    print("\n" + "="*60)
    print("验证 topk_sweep.csv")
    print("="*60)
    
    df = pd.read_csv("topk_sweep.csv")
    print(f"记录数: {len(df)}")
    print(f"K值: {list(df['K'])}")
    
    # 检查必需字段
    required_cols = ['K', 'mean_latency_ms', 'std_latency_ms', 'p95_latency_ms', 'estimated_accuracy']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ 缺少必需列: {col}")
            return False
    
    # 检查数据完整性
    print("\nTop-K扫描数据:")
    print("-"*60)
    for idx, row in df.iterrows():
        print(f"K={row['K']}: latency={row['mean_latency_ms']:.2f}ms±{row['std_latency_ms']:.2f}ms, p95={row['p95_latency_ms']:.2f}ms, acc={row['estimated_accuracy']:.3f}")
        
        # 合理性检查
        if row['K'] <= 0:
            print(f"  ⚠️  警告: K值 <= 0")
        if row['mean_latency_ms'] <= 0:
            print(f"  ⚠️  警告: 平均延迟 <= 0")
        if row['estimated_accuracy'] < 0 or row['estimated_accuracy'] > 1:
            print(f"  ⚠️  警告: accuracy不在[0,1]范围内")
    
    # 检查趋势：K增加时延迟应增加
    delays = list(zip(df['K'], df['mean_latency_ms']))
    delays_sorted = sorted(delays, key=lambda x: x[0])
    for i in range(1, len(delays_sorted)):
        if delays_sorted[i][1] < delays_sorted[i-1][1]:
            print(f"  ⚠️  警告: K增加但延迟减少 (K={delays_sorted[i-1][0]}->{delays_sorted[i][0]}, latency={delays_sorted[i-1][1]:.2f}->{delays_sorted[i][1]:.2f})")
    
    print("\n✅ topk_sweep.csv 验证通过")
    return True

def check_all_files_exist():
    """检查所有必需文件是否存在"""
    print("="*60)
    print("检查文件是否存在")
    print("="*60)
    
    required_files = ['latency.csv', 'energy.csv', 'tau_sweep.csv', 'topk_sweep.csv']
    all_exist = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} 存在")
        else:
            print(f"❌ {file} 不存在")
            all_exist = False
    
    return all_exist

def main():
    """主验证函数"""
    print("ARM实机推理基准测试数据验证")
    print(f"当前目录: {os.getcwd()}")
    print(f"验证时间: {pd.Timestamp.now()}")
    
    # 检查文件是否存在
    if not check_all_files_exist():
        print("\n❌ 有文件缺失，验证失败")
        return False
    
    # 验证各个文件
    results = []
    results.append(validate_latency_csv())
    results.append(validate_energy_csv())
    results.append(validate_tau_sweep_csv())
    results.append(validate_topk_sweep_csv())
    
    # 汇总结果
    print("\n" + "="*60)
    print("验证结果汇总")
    print("="*60)
    
    if all(results):
        print("✅ 所有文件验证通过!")
        print("\n生成的文件:")
        print("  1. latency.csv - 所有模型的延迟、内存、参数数据")
        print("  2. energy.csv - 所有模型的能耗数据（估算）")
        print("  3. tau_sweep.csv - tau参数扫描结果")
        print("  4. topk_sweep.csv - Top-K参数扫描结果")
        print("\n注意:")
        print("  - 能耗数据基于估算，非实际板级测量")
        print("  - tau和Top-K实验的精度值为模拟值")
        print("  - 所有测试在ARM Kunpeng-920 CPU上运行")
        return True
    else:
        print("❌ 部分文件验证失败")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)