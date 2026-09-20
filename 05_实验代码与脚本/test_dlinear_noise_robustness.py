"""
测试DLinear在20%高斯噪声下的R²性能
使用新训练的权重
"""
import torch
import numpy as np
import os
import json
from run_dlinear import DLinear
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig, compute_metrics
)
from noise_utils import inject_industrial_noise

def test_dlinear_noise_robustness():
    print("🔬 测试DLinear噪声鲁棒性 (20%高斯噪声)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 数据加载
    data, _ = load_csv_data('wind_final.csv', 'power')
    seq_len = 96
    pred_len = 24
    input_dim = data.shape[1]
    
    ts_config = TimeSeriesConfig(seq_len=seq_len, pred_len=pred_len, target='power')
    train_config = TrainConfig(batch_size=256)
    
    _, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)
    target_std = scaler.std[-1]
    target_mean = scaler.mean[-1]
    
    # 2. 加载训练好的DLinear模型
    model = DLinear(seq_len, pred_len, input_dim).to(device)
    
    # 尝试加载新训练的权重
    weight_path = 'dlinear_weights/dlinear_trained.pth'
    if os.path.exists(weight_path):
        print(f"✅ 加载训练好的DLinear权重: {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 显示模型性能
        if 'metrics' in checkpoint:
            print(f"  原始R²: {checkpoint['metrics']['R2']:.4f}")
    else:
        print("⚠️  未找到训练好的权重，使用随机权重")
    
    model.eval()
    
    # 3. 测试不同噪声水平
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    r2_results = []
    
    print(f"\n📊 噪声测试 (高斯噪声)")
    print("-" * 50)
    
    for noise_level in noise_levels:
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                by = by.to(device)
                
                # 注入噪声
                bx_noisy = inject_industrial_noise(bx, 'gaussian', noise_level)
                
                # 推理
                pred = model(bx_noisy)
                
                all_preds.append(pred.cpu().numpy())
                all_trues.append(by.cpu().numpy())
        
        # 合并结果
        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        
        # 反归一化
        preds_orig = preds * target_std + target_mean
        trues_orig = trues * target_std + target_mean
        
        # 计算R²
        metrics = compute_metrics(preds_orig, trues_orig)
        r2 = metrics['R2']
        r2_results.append(r2)
        
        print(f"噪声 {noise_level:.1f} | R² = {r2:.4f} | MSE = {metrics['MSE']:.0f}")
    
    # 4. 保存结果
    results = {
        'model': 'DLinear',
        'noise_levels': noise_levels,
        'r2_scores': r2_results,
        'clean_r2': r2_results[0] if r2_results else 0,
        'noise_20_r2': r2_results[2] if len(r2_results) > 2 else 0,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'device': str(device),
        'data_info': {
            'dataset': 'wind_final.csv',
            'seq_len': seq_len,
            'pred_len': pred_len,
            'input_dim': input_dim,
            'noise_type': 'gaussian'
        }
    }
    
    # 保存JSON
    output_file = 'dlinear_noise_robustness_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存: {output_file}")
    
    # 5. 生成报告
    report_file = 'dlinear_noise_robustness_report.txt'
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DLinear 噪声鲁棒性测试报告 (20%高斯噪声)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"测试时间: {results['timestamp']}\n")
        f.write(f"测试设备: {results['device']}\n")
        f.write(f"数据集: {results['data_info']['dataset']}\n\n")
        
        f.write("噪声水平与R²性能:\n")
        f.write("-" * 40 + "\n")
        for nl, r2 in zip(noise_levels, r2_results):
            f.write(f"噪声 {nl:.1f}: R² = {r2:.4f}\n")
        
        f.write("\n关键指标:\n")
        f.write("-" * 40 + "\n")
        f.write(f"干净数据R²: {results['clean_r2']:.4f}\n")
        f.write(f"噪声20% R²: {results['noise_20_r2']:.4f}\n")
        
        if results['clean_r2'] > 0:
            degradation = ((results['clean_r2'] - results['noise_20_r2']) / results['clean_r2']) * 100
            f.write(f"性能下降: {degradation:.1f}%\n")
        
        f.write("\n结论:\n")
        f.write("-" * 40 + "\n")
        if results['noise_20_r2'] > 0.6:
            f.write("DLinear在20%高斯噪声下仍保持良好性能 (>0.6 R²)\n")
        elif results['noise_20_r2'] > 0:
            f.write("DLinear在噪声环境下性能有所下降，但仍为正预测能力\n")
        else:
            f.write("DLinear在噪声环境下性能严重下降\n")
    
    print(f"✅ 报告已生成: {report_file}")
    
    # 6. 与现有数据对比
    print(f"\n📈 与现有数据对比")
    print("-" * 50)
    
    try:
        with open('gbdt_results_complete.json', 'r') as f:
            gbdt_data = json.load(f)
        
        if 'ultra_lsnt_scores' in gbdt_data and 'lightgbm_scores' in gbdt_data:
            noise_idx = 2  # 0.2噪声水平索引
            ultra_r2 = gbdt_data['ultra_lsnt_scores'][noise_idx]
            lgbm_r2 = gbdt_data['lightgbm_scores'][noise_idx]
            dlinear_r2 = results['noise_20_r2']
            
            print(f"噪声20%下各模型R²:")
            print(f"  DLinear: {dlinear_r2:.4f}")
            print(f"  Ultra-LSNT: {ultra_r2:.4f}")
            print(f"  LightGBM: {lgbm_r2:.4f}")
            
            # 保存对比
            comparison_file = 'noise_20_comparison.txt'
            with open(comparison_file, 'w') as f:
                f.write("噪声20%下模型性能对比\n")
                f.write("=" * 50 + "\n")
                f.write(f"DLinear:      R² = {dlinear_r2:.4f}\n")
                f.write(f"Ultra-LSNT:   R² = {ultra_r2:.4f}\n")
                f.write(f"LightGBM:     R² = {lgbm_r2:.4f}\n")
            
            print(f"✅ 对比结果已保存: {comparison_file}")
    except Exception as e:
        print(f"⚠️  无法与现有数据对比: {e}")
    
    return results

import time

if __name__ == '__main__':
    test_dlinear_noise_robustness()