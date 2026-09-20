#!/usr/bin/env python3
"""
运行DM测试：Ultra-LSNT vs PatchTST (ICLR 2023)
使用现有的DM测试框架和保存的预测结果
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
import os
import json
import sys
from datetime import datetime

# ==================== DM测试函数 (与现有脚本相同) ====================
def dm_test_newey_west(actual, pred1, pred2, criterion="MSE"):
    """
    Diebold-Mariano Test with Newey-West HAC variance estimator
    完全匹配附录描述：
    - 损失函数: MSE
    - 预测范围: 单步 (H=1)
    - 方差估计: Newey-West HAC 协方差估计器
    - 滞后阶数: ⌊4(T/100)^{2/9}⌋
    - 零假设: 两模型预测精度相等
    - 备择假设: Ultra-LSNT 预期损失更低 (单侧检验)
    
    Returns:
        dm_stat, p_value (单侧)
    """
    e1 = actual - pred1
    e2 = actual - pred2
    
    T = len(e1)
    
    if criterion == "MSE":
        d = e1**2 - e2**2
    elif criterion == "MAE":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("criterion must be 'MSE' or 'MAE'")
        
    d_mean = np.mean(d)
    
    # 计算滞后阶数 ⌊4(T/100)^{2/9}⌋
    lag = int(np.floor(4 * (T/100) ** (2/9)))
    # 确保滞后不超过 T-1
    lag = min(lag, T - 1)
    
    # 自协方差计算函数
    def autocovariance(Xi, k):
        if k == 0:
            return np.mean((Xi - d_mean) ** 2)
        else:
            return np.mean((Xi[k:] - d_mean) * (Xi[:-k] - d_mean))
    
    # 计算 gamma_0 到 gamma_lag
    gamma = np.zeros(lag + 1)
    for k in range(lag + 1):
        gamma[k] = autocovariance(d, k)
    
    # Newey-West HAC 方差估计 (Bartlett 核)
    var_d = gamma[0]
    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)  # Bartlett 核权重
        var_d += 2 * weight * gamma[k]
    
    # 防止除零
    if var_d <= 0:
        var_d = 1e-6
    
    # DM 统计量
    dm_stat = d_mean / np.sqrt(var_d / T)
    
    # 单侧 p 值 (备择假设: Ultra-LSNT 损失更小, 即 d_mean < 0)
    p_value = stats.norm.cdf(dm_stat)
    
    return dm_stat, p_value

# ==================== 加载预测数据 ====================
def load_predictions(model_name):
    """加载保存的预测结果"""
    pred_path = f'checkpoints_ts/{model_name}_preds.npy'
    true_path = f'checkpoints_ts/{model_name}_trues.npy'
    
    if not os.path.exists(pred_path):
        print(f"❌ 找不到预测文件: {pred_path}")
        return None, None
    
    if not os.path.exists(true_path):
        print(f"❌ 找不到真实值文件: {true_path}")
        return None, None
    
    try:
        preds = np.load(pred_path)
        trues = np.load(true_path)
        
        # 展平为一维数组
        if preds.ndim > 1:
            preds = preds.flatten()
        if trues.ndim > 1:
            trues = trues.flatten()
        
        # 确保长度一致
        min_len = min(len(preds), len(trues))
        preds = preds[:min_len]
        trues = trues[:min_len]
        
        print(f"✅ 加载 {model_name}: {len(preds)} 个预测样本")
        return preds, trues
    except Exception as e:
        print(f"❌ 加载 {model_name} 数据时出错: {e}")
        return None, None

def compute_metrics(preds, trues):
    """计算性能指标"""
    mse = np.mean((preds - trues) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - trues))
    
    # R² 计算
    ss_res = np.sum((trues - preds) ** 2)
    ss_tot = np.sum((trues - np.mean(trues)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse
    }

# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("运行DM测试：Ultra-LSNT vs PatchTST (ICLR 2023)")
    print("=" * 80)
    
    # 1. 加载Ultra-LSNT预测
    print("\n🤖 加载Ultra-LSNT预测...")
    preds_ultra, trues_ultra = load_predictions('Ultra-LSNT')
    if preds_ultra is None:
        # 尝试其他可能的文件名
        print("尝试加载 main 目录下的预测...")
        main_pred_path = 'checkpoints_ts/main/preds.npy'
        if os.path.exists(main_pred_path):
            preds_ultra = np.load(main_pred_path).flatten()
            trues_ultra = np.load('checkpoints_ts/main/trues.npy').flatten()
        else:
            print("❌ 无法加载Ultra-LSNT预测")
            return
    
    # 2. 加载PatchTST预测
    print("\n🤖 加载PatchTST预测...")
    preds_patch, trues_patch = load_predictions('PatchTST')
    if preds_patch is None:
        print("❌ 无法加载PatchTST预测")
        return
    
    # 3. 确保使用相同的真实值
    if len(trues_ultra) != len(trues_patch):
        print(f"⚠️ 样本数量不一致: Ultra-LSNT={len(trues_ultra)}, PatchTST={len(trues_patch)}")
        min_len = min(len(trues_ultra), len(trues_patch))
        trues_ultra = trues_ultra[:min_len]
        trues_patch = trues_patch[:min_len]
        preds_ultra = preds_ultra[:min_len]
        preds_patch = preds_patch[:min_len]
    
    # 检查真实值是否一致（应该相同）
    if not np.allclose(trues_ultra, trues_patch, rtol=1e-3):
        print("⚠️ 警告: Ultra-LSNT和PatchTST的真实值不完全一致")
        max_diff = np.max(np.abs(trues_ultra - trues_patch))
        print(f"   最大差异: {max_diff:.2f}")
        # 使用平均值
        actual = (trues_ultra + trues_patch) / 2
    else:
        actual = trues_ultra
    
    print(f"\n📊 样本统计:")
    print(f"   样本数: {len(actual)}")
    print(f"   实际值范围: [{actual.min():.2f}, {actual.max():.2f}]")
    print(f"   实际值均值: {actual.mean():.2f} ± {actual.std():.2f}")
    
    # 4. 计算性能指标
    metrics_ultra = compute_metrics(preds_ultra, actual)
    metrics_patch = compute_metrics(preds_patch, actual)
    
    print(f"\n📈 性能对比:")
    print(f"   Ultra-LSNT:   R²={metrics_ultra['R2']:.4f}, RMSE={metrics_ultra['RMSE']:.2f}, MAE={metrics_ultra['MAE']:.2f}")
    print(f"   PatchTST:     R²={metrics_patch['R2']:.4f}, RMSE={metrics_patch['RMSE']:.2f}, MAE={metrics_patch['MAE']:.2f}")
    print(f"   差异 (Ultra-LSNT - PatchTST):")
    print(f"     ΔR²: {metrics_ultra['R2'] - metrics_patch['R2']:.4f}")
    print(f"     ΔRMSE: {metrics_ultra['RMSE'] - metrics_patch['RMSE']:.2f}")
    print(f"     ΔMAE: {metrics_ultra['MAE'] - metrics_patch['MAE']:.2f}")
    
    # 5. 运行DM测试
    print(f"\n⚔️ 运行DM Test (Newey-West HAC)...")
    
    # 使用MSE作为损失函数
    dm_stat, p_value = dm_test_newey_west(actual, preds_ultra, preds_patch, criterion="MSE")
    
    print(f"   DM统计量: {dm_stat:.4f}")
    print(f"   P值 (单侧): {p_value:.4e}")
    
    # 解释结果
    if dm_stat < 0:
        if p_value < 0.01:
            conclusion = "**Ultra-LSNT显著优于PatchTST (p<0.01)**"
        elif p_value < 0.05:
            conclusion = "*Ultra-LSNT显著优于PatchTST (p<0.05)"
        else:
            conclusion = "Ultra-LSNT优于PatchTST但不显著"
        print(f"   结论: {conclusion}")
        print(f"   (负DM统计量表示Ultra-LSNT误差更小)")
    else:
        if p_value < 0.01:
            conclusion = "**PatchTST显著优于Ultra-LSNT (p<0.01)**"
        elif p_value < 0.05:
            conclusion = "*PatchTST显著优于Ultra-LSNT (p<0.05)"
        else:
            conclusion = "PatchTST优于Ultra-LSNT但不显著"
        print(f"   结论: {conclusion}")
        print(f"   (正DM统计量表示PatchTST误差更小)")
    
    # 6. 保存结果
    result = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'wind_final.csv',
        'samples': len(actual),
        'ultra_lsnt': {
            'r2': float(metrics_ultra['R2']),
            'rmse': float(metrics_ultra['RMSE']),
            'mae': float(metrics_ultra['MAE']),
        },
        'patchtst': {
            'r2': float(metrics_patch['R2']),
            'rmse': float(metrics_patch['RMSE']),
            'mae': float(metrics_patch['MAE']),
        },
        'dm_test': {
            'statistic': float(dm_stat),
            'p_value': float(p_value),
            'criterion': 'MSE',
            'conclusion': conclusion,
        },
        'reference_values': {
            'ultra_lsnt_r2': 0.8727728724479675,  # 来自Table 3
            'patchtst_r2': 0.8666042983531952,    # 来自full_sota_comparison.json
        }
    }
    
    os.makedirs('dm_results', exist_ok=True)
    result_file = f"dm_results/dm_patchtst_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 结果已保存: {result_file}")
    
    # 7. 生成LaTeX格式表格
    print(f"\n📋 LaTeX格式结果表:")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Diebold-Mariano显著性检验结果 (Ultra-LSNT vs PatchTST - ICLR 2023)}")
    print("\\label{tab:dm_patchtst}")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("模型 & R² & RMSE & DM统计量 \\\\")
    print("\\hline")
    print(f"Ultra-LSNT & {metrics_ultra['R2']:.4f} & {metrics_ultra['RMSE']:.2f} & --- \\\\")
    print(f"PatchTST & {metrics_patch['R2']:.4f} & {metrics_patch['RMSE']:.2f} & {dm_stat:.4f} \\\\")
    print("\\hline")
    print(f"\\multicolumn{{4}}{{l}}{{p值 (单侧): {p_value:.4e}}} \\\\")
    print(f"\\multicolumn{{4}}{{l}}{{结论: {conclusion}}} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    
    print(f"\n✅ DM测试完成!")
    print(f"💡 关键发现: {conclusion}")

if __name__ == '__main__':
    main()