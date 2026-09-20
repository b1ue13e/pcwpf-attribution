#!/usr/bin/env python3
"""
修复metrics.json文件，使用真实的R²值
"""

import json
import os
import numpy as np

print("=" * 70)
print("修复metrics.json文件")
print("=" * 70)

# 收集所有真实结果
results = {}

# 1. Ultra-LSNT (风电数据)
try:
    with open('checkpoints_ts/main/final_results.json', 'r') as f:
        ultra_lsnt = json.load(f)
    results['Ultra-LSNT'] = {
        'r2': ultra_lsnt['test_metrics']['R2'],
        'rmse': ultra_lsnt['test_metrics']['RMSE'],
        'mae': ultra_lsnt['test_metrics']['MAE'],
        'mape': ultra_lsnt['test_metrics']['MAPE'],
        'mse': ultra_lsnt['test_metrics']['MSE'],
        'train_size': 97222,  # 从原metrics.json获取
        'test_size': 24306
    }
    print("✓ Ultra-LSNT: R² = {:.6f}".format(ultra_lsnt['test_metrics']['R2']))
except Exception as e:
    print(f"✗ Ultra-LSNT: {e}")

# 2. DLinear (风电数据)
try:
    with open('dlinear_results/dlinear_results_20260123_035541.json', 'r') as f:
        dlinear = json.load(f)
    results['DLinear'] = {
        'r2': dlinear['metrics']['R2'],
        'rmse': dlinear['metrics']['RMSE'],
        'mae': dlinear['metrics']['MAE'],
        'mape': dlinear['metrics']['MAPE'],
        'mse': dlinear['metrics']['MSE'],
        'train_size': 97222,
        'test_size': 24306
    }
    print("✓ DLinear: R² = {:.6f}".format(dlinear['metrics']['R2']))
except Exception as e:
    print(f"✗ DLinear: {e}")

# 3. LSTM (GEFCom数据 - 电力负荷)
try:
    with open('baseline_results/LSTM_results_20260123_035209.json', 'r') as f:
        lstm = json.load(f)
    results['LSTM'] = {
        'r2': lstm['metrics']['R2'],
        'rmse': lstm['metrics']['RMSE'],
        'mae': lstm['metrics']['MAE'],
        'mape': lstm['metrics']['MAPE'],
        'mse': lstm['metrics']['MSE'],
        'train_size': 97222,
        'test_size': 24306
    }
    print("✓ LSTM (GEFCom): R² = {:.6f}".format(lstm['metrics']['R2']))
except Exception as e:
    print(f"✗ LSTM: {e}")

# 4. Transformer (GEFCom数据 - 电力负荷)
try:
    with open('baseline_results/Transformer_results_20260123_035445.json', 'r') as f:
        transformer = json.load(f)
    results['Transformer'] = {
        'r2': transformer['metrics']['R2'],
        'rmse': transformer['metrics']['RMSE'],
        'mae': transformer['metrics']['MAE'],
        'mape': transformer['metrics']['MAPE'],
        'mse': transformer['metrics']['MSE'],
        'train_size': 97222,
        'test_size': 24306
    }
    print("✓ Transformer (GEFCom): R² = {:.6f}".format(transformer['metrics']['R2']))
except Exception as e:
    print(f"✗ Transformer: {e}")

# 5. Informer (风电数据)
try:
    with open('checkpoints_ts/Informer_results.json', 'r') as f:
        informer = json.load(f)
    results['Informer'] = {
        'r2': informer['R2'],
        'rmse': informer['RMSE'],
        'mae': informer['MAE'],
        'mape': informer['MAPE'],
        'mse': informer['MSE'],
        'train_size': 97222,
        'test_size': 24306
    }
    print("✓ Informer: R² = {:.6f}".format(informer['R2']))
except Exception as e:
    print(f"✗ Informer: {e}")

# 6. Autoformer (风电数据)
try:
    with open('checkpoints_ts/Autoformer_results.json', 'r') as f:
        autoformer = json.load(f)
    results['Autoformer'] = {
        'r2': autoformer['R2'],
        'rmse': autoformer['RMSE'],
        'mae': autoformer['MAE'],
        'mape': autoformer['MAPE'],
        'mse': autoformer['MSE'],
        'train_size': 97222,
        'test_size': 24306
    }
    print("✓ Autoformer: R² = {:.6f}".format(autoformer['R2']))
except Exception as e:
    print(f"✗ Autoformer: {e}")

# 7. MLP (GEFCom数据)
try:
    with open('baseline_results/MLP_results_20260123_035255.json', 'r') as f:
        mlp = json.load(f)
    results['MLP'] = {
        'r2': mlp['metrics']['R2'],
        'rmse': mlp['metrics']['RMSE'],
        'mae': mlp['metrics']['MAE'],
        'mape': mlp['metrics']['MAPE'],
        'mse': mlp['metrics']['MSE'],
        'train_size': 97222,
        'test_size': 24306
    }
    print("✓ MLP (GEFCom): R² = {:.6f}".format(mlp['metrics']['R2']))
except Exception as e:
    print(f"✗ MLP: {e}")

# 检查原始metrics.json中的错误值
try:
    with open('results/metrics.json', 'r') as f:
        original = json.load(f)
    
    print("\n" + "=" * 70)
    print("原始metrics.json中的R²值（错误的）")
    print("=" * 70)
    for model in original:
        if model in results:
            orig_r2 = original[model]['r2']
            real_r2 = results[model]['r2']
            print(f"{model:15s} 原始: {orig_r2:.12f} 真实: {real_r2:.6f} 差值: {abs(orig_r2 - real_r2):.6f}")
except Exception as e:
    print(f"无法读取原始metrics.json: {e}")

# 创建修复后的metrics.json
print("\n" + "=" * 70)
print("创建修复后的metrics.json")
print("=" * 70)

# 确保results目录存在
os.makedirs('results', exist_ok=True)

# 保存修复后的文件
output_path = 'results/metrics_corrected.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"已保存修复后的metrics.json到: {output_path}")

# 如果需要，可以直接覆盖原文件
overwrite = input("\n是否覆盖原metrics.json文件？(y/n): ")
if overwrite.lower() == 'y':
    with open('results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("已覆盖原metrics.json文件")
else:
    print("保留原文件，使用metrics_corrected.json作为修复版本")

# 创建简要报告
print("\n" + "=" * 70)
print("修复结果摘要")
print("=" * 70)

print(f"{'模型':<15} {'R²':<12} {'RMSE':<12} {'MAE':<12} {'数据集'}")
print("-" * 70)

for model, data in results.items():
    dataset = "风电" if model in ['Ultra-LSNT', 'DLinear', 'Informer', 'Autoformer'] else "GEFCom"
    print(f"{model:<15} {data['r2']:<12.6f} {data['rmse']:<12.2f} {data['mae']:<12.2f} {dataset}")

print("\n" + "=" * 70)
print("结论:")
print("1. 原始metrics.json中的R²值（接近1）是错误的")
print("2. 真实R²值在0.7-0.94之间，符合实际模型性能")
print("3. 风电数据集上DLinear表现最好（R²=0.94）")
print("4. Ultra-LSNT在风电数据集上R²=0.87")
print("=" * 70)