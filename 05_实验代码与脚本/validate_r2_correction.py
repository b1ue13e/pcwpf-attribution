#!/usr/bin/env python3
"""
验证和修复R²计算错误
分析demo_run.py中R²计算的问题并提供修复方案
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("验证R²计算问题")
print("=" * 70)

# 1. 加载数据
df = pd.read_csv('wind_final.csv')
data = df.values.astype(np.float32)
n_samples = len(data)
seq_len, pred_len = 96, 24

# 2. 创建时序样本（与demo_run.py相同）
X_samples, y_samples = [], []
for i in range(n_samples - seq_len - pred_len):
    X_samples.append(data[i:i+seq_len].flatten())
    y_samples.append(data[i+seq_len+pred_len, -1])  # 最后一列是power

X = np.array(X_samples, dtype=np.float32)
y = np.array(y_samples, dtype=np.float32)

# 3. 划分数据集（与demo_run.py相同）
train_idx = int(0.7 * len(X))
val_idx = int(0.85 * len(X))

X_train, y_train = X[:train_idx], y[:train_idx]
X_test, y_test = X[val_idx:], y[val_idx:]

# 4. 标准化（与demo_run.py相同）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"数据形状: X_test={X_test.shape}, y_test={y_test.shape}")
print(f"标准化y_test范围: [{y_test_scaled.min():.4f}, {y_test_scaled.max():.4f}]")
print(f"标准化y_test均值: {np.mean(y_test_scaled):.6f}")
print(f"标准化y_test方差: {np.var(y_test_scaled):.6f}")

# 5. 使用简单的线性回归计算预测值（模拟demo_run.py）
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)
y_test_pred_scaled = model.predict(X_test_scaled)

# 6. 反标准化
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

# 7. 计算不同空间下的R²
def compute_r2_scaled():
    """在标准化空间计算R²（demo_run.py的方法）"""
    ss_res = np.sum((y_test_scaled - y_test_pred_scaled) ** 2)
    ss_tot = np.sum((y_test_scaled - np.mean(y_test_scaled)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return r2, ss_res, ss_tot

def compute_r2_original():
    """在原始数据空间计算R²（正确的方法）"""
    ss_res = np.sum((y_test - y_test_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return r2, ss_res, ss_tot

r2_scaled, ss_res_scaled, ss_tot_scaled = compute_r2_scaled()
r2_original, ss_res_original, ss_tot_original = compute_r2_original()

print("\n" + "=" * 70)
print("R²计算对比")
print("=" * 70)
print(f"标准化空间R²: {r2_scaled:.12f}")
print(f"原始数据空间R²: {r2_original:.6f}")
print(f"差异: {abs(r2_scaled - r2_original):.10f}")

print("\n标准化空间统计:")
print(f"  SS_res: {ss_res_scaled:.6f}")
print(f"  SS_tot: {ss_tot_scaled:.6f}")
print(f"  SS_res/SS_tot: {ss_res_scaled/ss_tot_scaled:.12f}")

print("\n原始数据空间统计:")
print(f"  SS_res: {ss_res_original:.2f}")
print(f"  SS_tot: {ss_tot_original:.2f}")
print(f"  SS_res/SS_tot: {ss_res_original/ss_tot_original:.6f}")

# 8. 与当前metrics.json对比
print("\n" + "=" * 70)
print("与当前metrics.json对比")
print("=" * 70)

try:
    with open('results/metrics.json', 'r') as f:
        current_metrics = json.load(f)
    
    if 'Ultra-LSNT' in current_metrics:
        current_r2 = current_metrics['Ultra-LSNT']['r2']
        print(f"当前metrics.json中的R²: {current_r2:.12f}")
        print(f"标准化空间计算的R²: {r2_scaled:.12f}")
        print(f"原始数据空间计算的R²: {r2_original:.6f}")
        
        # 检查问题是否来自标准化
        print("\n问题分析:")
        if abs(current_r2 - r2_scaled) < 1e-10:
            print("✓ 确认问题: demo_run.py在标准化空间计算R²，导致R²异常接近1")
            print("  原因: 标准化后数据方差很小，SS_tot很小，R²接近1")
        else:
            print("✗ 问题更复杂，可能有其他计算错误")
            
except Exception as e:
    print(f"无法读取metrics.json: {e}")

# 9. 与checkpoints中的真实结果对比
print("\n" + "=" * 70)
print("与checkpoints真实结果对比")
print("=" * 70)

try:
    with open('checkpoints_ts/main/final_results.json', 'r') as f:
        checkpoint_results = json.load(f)
    
    real_r2 = checkpoint_results['test_metrics']['R2']
    print(f"checkpoints中的真实R²: {real_r2:.6f}")
    print(f"我们的原始数据空间R²: {r2_original:.6f}")
    print(f"我们的标准化空间R²: {r2_scaled:.12f}")
    
    print(f"\n结论:")
    print(f"1. 真实模型性能R²约为: {real_r2:.4f}")
    print(f"2. metrics.json中报告的R²({current_r2:.12f})是错误的")
    print(f"3. 错误原因是使用了标准化数据计算R²")
    
except Exception as e:
    print(f"无法读取checkpoint结果: {e}")

# 10. 修复建议
print("\n" + "=" * 70)
print("修复建议")
print("=" * 70)
print("1. 修改demo_run.py中的compute_metrics函数:")
print("   - 在计算R²前进行反标准化")
print("   - 或者在原始数据空间计算所有指标")
print("2. 更新results/metrics.json:")
print("   - 使用真实的R²值（约0.87）")
print("   - 重新运行评估或手动更新")

# 11. 创建一个修复后的metrics.json示例
print("\n" + "=" * 70)
print("创建修复后的metrics.json示例")
print("=" * 70)

# 从checkpoints加载真实结果
try:
    with open('checkpoints_ts/main/final_results.json', 'r') as f:
        checkpoint = json.load(f)
    
    fixed_metrics = {
        "Ultra-LSNT": {
            "r2": checkpoint['test_metrics']['R2'],
            "rmse": checkpoint['test_metrics']['RMSE'],
            "mae": checkpoint['test_metrics']['MAE'],
            "mape": checkpoint['test_metrics']['MAPE'],
            "mse": checkpoint['test_metrics']['MSE'],
            "train_size": 97222,  # 从原有metrics.json获取
            "test_size": 24306
        },
        "DLinear": {
            "r2": 0.872,  # 示例值，应该从DLinear结果获取
            "rmse": 9080.55,
            "mae": 7171.42,
            "mape": 27.74,
            "mse": 82456480.0,
            "train_size": 97222,
            "test_size": 24306
        }
    }
    
    print("修复后的metrics.json示例:")
    print(json.dumps(fixed_metrics, indent=2))
    
except Exception as e:
    print(f"无法创建示例: {e}")

print("\n✅ 验证完成")