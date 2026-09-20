#!/usr/bin/env python3
import numpy as np
import os
import json

print("检查PatchTST预测数据...")
patch_pred = np.load('checkpoints_ts/PatchTST_preds.npy')
patch_true = np.load('checkpoints_ts/PatchTST_trues.npy')
print(f"PatchTST_preds形状: {patch_pred.shape}")
print(f"PatchTST_trues形状: {patch_true.shape}")

print("\n检查其他模型预测...")
for f in os.listdir('checkpoints_ts'):
    if f.endswith('_preds.npy'):
        model = f.replace('_preds.npy', '')
        print(f"{model}: 存在预测文件")

print("\n检查是否有历史DM测试结果...")
if os.path.exists('dm_test_output.log'):
    with open('dm_test_output.log', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '样本数' in line:
                print(f"日志中的样本信息: {line.strip()}")

print("\n从full_sota_comparison.json获取参考性能...")
with open('checkpoints_ts/full_sota_comparison.json', 'r') as f:
    data = json.load(f)
    for model in ['Ultra-LSNT', 'PatchTST', 'Transformer', 'TimeMixer', 'iTransformer']:
        if model in data:
            print(f"{model}: R²={data[model]['R2']:.4f}, RMSE={data[model]['RMSE']:.2f}")