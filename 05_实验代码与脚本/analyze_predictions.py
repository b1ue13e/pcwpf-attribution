#!/usr/bin/env python3
"""
分析不同模型的预测结果，为调度比较做准备
"""
import numpy as np
import json
import os

def load_predictions(model_name):
    """加载不同模型的预测数据"""
    if model_name == "Ultra-LSNT":
        # 主模型的预测
        path = "checkpoints_ts/main/predictions.npz"
        if os.path.exists(path):
            data = np.load(path)
            # 需要查看数据结构
            print(f"Ultra-LSNT NPZ keys: {list(data.keys())}")
            preds = data['predictions']
            trues = data['trues'] if 'trues' in data else None
            return preds, trues
        else:
            print(f"警告: {path} 不存在")
            return None, None
    elif model_name == "DLinear":
        path = "dlinear_results/dlinear_samples_20260123_035541.npz"
        if os.path.exists(path):
            data = np.load(path)
            print(f"DLinear NPZ keys: {list(data.keys())}")
            # 假设有 'predictions' 和 'trues' 键
            preds = data['predictions'] if 'predictions' in data else None
            trues = data['trues'] if 'trues' in data else None
            return preds, trues
        else:
            print(f"警告: {path} 不存在")
            return None, None
    elif model_name == "Transformer":
        # 这里Transformer可能指的是Autoformer或Informer
        # 我们需要找到对应的预测文件
        # 暂时返回None
        return None, None
    else:
        return None, None

def main():
    print("分析预测结果...")
    
    # 检查文件
    models = ["Ultra-LSNT", "DLinear", "Autoformer", "Informer"]
    
    for model in models:
        print(f"\n=== {model} ===")
        preds, trues = load_predictions(model)
        if preds is not None:
            print(f"预测形状: {preds.shape}")
            print(f"预测范围: [{preds.min():.1f}, {preds.max():.1f}]")
            print(f"均值: {preds.mean():.1f} ± {preds.std():.1f}")
        else:
            print("无预测数据")
    
    # 检查调度模型需要的数据
    print("\n=== 调度模型参数 ===")
    # 加载一个样本用于测试
    sample_path = "checkpoints_ts/main/predictions.npz"
    if os.path.exists(sample_path):
        data = np.load(sample_path)
        if 'predictions' in data:
            preds = data['predictions']
            # 取第一个样本的24小时预测
            sample_pred = preds[0, :]
            print(f"样本预测 (24小时): {sample_pred.shape}")
            print(f"样本值: {sample_pred}")
            
            # 计算不确定性（基于预测误差统计）
            # 这里简单使用固定比例
            wind_std = sample_pred * 0.2
            print(f"不确定性 (20%): {wind_std}")
        else:
            print("预测文件中没有'predictions'键")
    else:
        print(f"文件不存在: {sample_path}")

if __name__ == "__main__":
    main()