#!/usr/bin/env python3
"""测试修改后的模型加载逻辑"""

import sys
sys.path.append('.')
import torch
import os
import json

# 测试LSNTConfig
from ultra_lsnt_timeseries import LSNTConfig, UltraLSNTForecaster, TimeSeriesConfig

print("测试修改后的模型加载逻辑")

# 创建一个模拟配置
config = LSNTConfig(input_dim=8, hidden_dim=128, num_blocks=3, num_experts=4, top_k=2)
ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')

# 创建模型
model = UltraLSNTForecaster(config, ts_config)

print(f"模型创建成功: {model}")
print(f"输入维度: {config.input_dim}")
print(f"隐藏维度: {config.hidden_dim}")

# 测试状态字典加载（模拟）
state_dict = model.state_dict()
print(f"\n状态字典键数量: {len(state_dict)}")

# 测试修改后的加载逻辑（模拟strict=False场景）
try:
    # 创建一个不同配置的模型（hidden_dim不匹配）
    config2 = LSNTConfig(input_dim=8, hidden_dim=256, num_blocks=3, num_experts=4, top_k=2)
    model2 = UltraLSNTForecaster(config2, ts_config)
    
    print(f"\n测试strict=False加载...")
    print(f"模型1 hidden_dim: {config.hidden_dim}")
    print(f"模型2 hidden_dim: {config2.hidden_dim}")
    
    # 尝试加载权重（应该会失败，但会提供错误信息）
    try:
        model2.load_state_dict(state_dict, strict=True)
        print("  strict=True 加载成功（预期失败，但可能在某些情况下成功）")
    except RuntimeError as e:
        print(f"  strict=True 加载失败（预期）: {str(e)[:100]}...")
        
        # 尝试strict=False
        try:
            model2.load_state_dict(state_dict, strict=False)
            print("  strict=False 加载成功")
        except RuntimeError as e2:
            print(f"  strict=False 加载也失败: {str(e2)[:100]}...")
    
except Exception as e:
    print(f"测试过程中出现异常: {e}")

# 测试配置保存和加载
print(f"\n测试配置保存/加载...")
config_dict = config.to_dict()
print(f"配置字典: {config_dict['hidden_dim']}")

# 验证修改后的MAPE计算
from ultra_lsnt_timeseries import compute_metrics
import numpy as np

# 测试包含零值的数据
true = np.array([0.0, 10.0, 0.0, 20.0, 30.0])
pred = np.array([1.0, 12.0, 2.0, 18.0, 32.0])
metrics = compute_metrics(pred, true)
print(f"\nMAPE测试:")
print(f"  MAPE值: {metrics['MAPE']:.4f}% (应该是一个合理值，而不是异常高的值)")

# 验证字体警告已消除
print(f"\n字体配置测试:")
import matplotlib.pyplot as plt
print(f"  当前字体设置: {plt.rcParams.get('font.family', '未设置')}")
print(f"  期望: 使用默认字体，不再硬编码Times New Roman")

print("\n所有测试完成")