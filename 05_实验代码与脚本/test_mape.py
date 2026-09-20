#!/usr/bin/env python3
"""测试修改后的MAPE计算"""

import sys
sys.path.append('.')
from ultra_lsnt_timeseries import compute_metrics
import numpy as np

print("测试修改后的MAPE计算函数")

# 测试1: 正常数据，没有零值
true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
pred = np.array([9.0, 22.0, 28.0, 42.0, 48.0])
metrics = compute_metrics(pred, true)
print(f"测试1 (正常数据):")
print(f"  MAE: {metrics['MAE']:.4f}")
print(f"  MAPE: {metrics['MAPE']:.4f}%")
print(f"  R2: {metrics['R2']:.4f}")

# 测试2: 包含零值的数据 (原始公式会导致除零错误)
true2 = np.array([0.0, 1.0, 2.0, 0.0, 5.0])
pred2 = np.array([0.1, 1.2, 1.8, 0.2, 5.5])
metrics2 = compute_metrics(pred2, true2)
print(f"\n测试2 (包含零值):")
print(f"  MAE: {metrics2['MAE']:.4f}")
print(f"  MAPE: {metrics2['MAPE']:.4f}%")
print(f"  R2: {metrics2['R2']:.4f}")

# 测试3: 包含极小值的数据 (接近零)
true3 = np.array([0.001, 0.002, 0.005, 0.01, 100.0])
pred3 = np.array([0.002, 0.001, 0.006, 0.012, 95.0])
metrics3 = compute_metrics(pred3, true3)
print(f"\n测试3 (包含极小值):")
print(f"  MAE: {metrics3['MAE']:.4f}")
print(f"  MAPE: {metrics3['MAPE']:.4f}%")
print(f"  R2: {metrics3['R2']:.4f}")

# 测试4: 空气质量数据模拟 (包含零和接近零的值)
true_aqi = np.array([0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
pred_aqi = np.array([2.0, 6.0, 12.0, 18.0, 55.0, 95.0, 210.0])
metrics_aqi = compute_metrics(pred_aqi, true_aqi)
print(f"\n测试4 (空气质量数据模拟):")
print(f"  MAE: {metrics_aqi['MAE']:.4f}")
print(f"  MAPE: {metrics_aqi['MAPE']:.4f}%")
print(f"  R2: {metrics_aqi['R2']:.4f}")

print("\n测试完成")