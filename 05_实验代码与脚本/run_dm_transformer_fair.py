#!/usr/bin/env python3
"""
重新运行DM测试：Ultra-LSNT vs Transformer (使用与Table 3完全相同的配置)
解决Table 3与DM测试结果的矛盾
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
import os
import json
import sys
from datetime import datetime

# 引入项目模块
sys.path.append(os.getcwd())
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    UltraLSNTForecaster, LSNTConfig, compute_metrics
)

# ==================== Transformer基线模型 (与Table 3相同) ====================
class TransformerBaseline(nn.Module):
    """与Table 3完全相同的Transformer配置"""
    def __init__(self, input_dim, seq_len=96, pred_len=24, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, d_model)
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # Transformer编码层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_proj = nn.Linear(d_model, pred_len)
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.input_proj(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        # 取最后一个时间步的特征进行预测
        x = x[:, -1, :]
        out = self.output_proj(x)
        return out

# ==================== DM测试函数 (与附录相同) ====================
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

# ==================== 加载或训练Transformer到Table 3性能 ====================
def get_transformer_predictions():
    """获取Transformer的预测结果，确保与Table 3性能一致"""
    print("🤖 加载/训练Transformer (Table 3配置)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载数据
    data_path = 'wind_final.csv'
    target = 'power'
    
    data, _ = load_csv_data(data_path, target)
    if data is None:
        print("❌ 数据加载失败")
        return None, None
    
    # Table 3配置
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target=target)
    train_config = TrainConfig(
        batch_size=256,
        epochs=100,  # 与Table 3训练配置一致
        lr=0.001,
        patience=20
    )
    
    train_loader, val_loader, test_loader, scaler = create_dataloaders(data, ts_config, train_config)
    input_dim = data.shape[1]
    
    # 2. 检查是否有保存的Transformer checkpoint
    transformer_ckpt_path = 'checkpoints_ts/transformer_best.pth'
    if os.path.exists(transformer_ckpt_path):
        print(f"✅ 找到Transformer checkpoint: {transformer_ckpt_path}")
        model = TransformerBaseline(input_dim, ts_config.seq_len, ts_config.pred_len).to(device)
        ckpt = torch.load(transformer_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    else:
        print("⚠️ 未找到Transformer checkpoint，重新训练...")
        # 重新训练Transformer以达到Table 3性能
        model = TransformerBaseline(input_dim, ts_config.seq_len, ts_config.pred_len).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
        criterion = nn.MSELoss()
        
        # 训练循环
        model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(train_config.epochs):
            epoch_loss = 0
            steps = 0
            
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                steps += 1
            
            avg_loss = epoch_loss / steps
            
            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    loss = criterion(out, by)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # 早停逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                }, transformer_ckpt_path)
            else:
                patience_counter += 1
            
            if patience_counter >= train_config.patience:
                print(f"⏹️ 早停在epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{train_config.epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # 加载最佳模型
        ckpt = torch.load(transformer_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    
    # 3. 推理
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            preds.append(out.cpu().numpy())
            trues.append(by.cpu().numpy())
    
    # 反归一化
    target_mean = scaler.mean[-1]
    target_std = scaler.std[-1]
    
    preds = np.concatenate(preds, axis=0) * target_std + target_mean
    trues = np.concatenate(trues, axis=0) * target_std + target_mean
    
    # 验证性能是否与Table 3一致
    metrics = compute_metrics(preds, trues)
    print(f"📊 Transformer性能: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
    
    # Table 3中的参考值
    table3_r2 = 0.743889570236206
    table3_rmse = 2916.826416015625
    
    if abs(metrics['R2'] - table3_r2) > 0.05:
        print(f"⚠️ 警告: Transformer R² ({metrics['R2']:.4f}) 与Table 3 ({table3_r2:.4f}) 差异较大")
    
    return preds.flatten(), trues.flatten(), metrics

# ==================== 加载Ultra-LSNT预测 ====================
def get_ultra_lsnt_predictions():
    """获取Ultra-LSNT的预测结果"""
    print("🤖 加载Ultra-LSNT预测...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_path = 'wind_final.csv'
    target = 'power'
    
    data, _ = load_csv_data(data_path, target)
    if data is None:
        return None, None
    
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target=target)
    train_config = TrainConfig(batch_size=256)
    
    train_loader, val_loader, test_loader, scaler = create_dataloaders(data, ts_config, train_config)
    input_dim = data.shape[1]
    
    # 加载Ultra-LSNT模型
    model_config_path = 'checkpoints_ts/main/model_config.json'
    if os.path.exists(model_config_path):
        with open(model_config_path, 'r') as f:
            config_dict = json.load(f)
        config_dict['input_dim'] = input_dim
        model_config = LSNTConfig(**config_dict)
    else:
        model_config = LSNTConfig(input_dim=input_dim)
    
    model = UltraLSNTForecaster(model_config, ts_config).to(device)
    
    # 加载权重
    ckpt_path = 'checkpoints_ts/main/best_model.pth'
    if not os.path.exists(ckpt_path):
        print(f"❌ 找不到Ultra-LSNT权重 {ckpt_path}")
        return None, None
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model_state_dict = model.state_dict()
    ckpt_state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # 处理权重不匹配
    filtered_state_dict = {}
    for k, v in ckpt_state_dict.items():
        if k in model_state_dict and model_state_dict[k].shape == v.shape:
            filtered_state_dict[k] = v
    
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict, strict=False)
    print(f"✅ 加载Ultra-LSNT权重，成功加载 {len(filtered_state_dict)}/{len(model_state_dict)} 个参数")
    
    # 推理
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            out, _ = model(bx)
            preds.append(out.cpu().numpy())
            trues.append(by.cpu().numpy())
    
    # 反归一化
    target_mean = scaler.mean[-1]
    target_std = scaler.std[-1]
    
    preds = np.concatenate(preds, axis=0) * target_std + target_mean
    trues = np.concatenate(trues, axis=0) * target_std + target_mean
    
    # 验证性能
    metrics = compute_metrics(preds, trues)
    print(f"📊 Ultra-LSNT性能: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
    
    return preds.flatten(), trues.flatten(), metrics

# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("重新运行DM测试：Ultra-LSNT vs Transformer (使用Table 3相同配置)")
    print("=" * 80)
    
    # 1. 获取Ultra-LSNT预测
    preds_ultra, trues_ultra, metrics_ultra = get_ultra_lsnt_predictions()
    if preds_ultra is None:
        return
    
    # 2. 获取Transformer预测
    preds_trans, trues_trans, metrics_trans = get_transformer_predictions()
    if preds_trans is None:
        return
    
    # 3. 确保使用相同的真实值
    if len(trues_ultra) != len(trues_trans):
        min_len = min(len(trues_ultra), len(trues_trans))
        trues_ultra = trues_ultra[:min_len]
        trues_trans = trues_trans[:min_len]
        preds_ultra = preds_ultra[:min_len]
        preds_trans = preds_trans[:min_len]
    
    # 检查真实值是否一致
    if not np.allclose(trues_ultra, trues_trans, rtol=1e-3):
        print("⚠️ 警告: Ultra-LSNT和Transformer的真实值不完全一致")
        print(f"   差异: max={np.max(np.abs(trues_ultra - trues_trans)):.2f}")
    
    # 使用Ultra-LSNT的真实值作为基准
    actual = trues_ultra
    
    print(f"\n📊 样本统计:")
    print(f"   Ultra-LSNT样本数: {len(preds_ultra)}")
    print(f"   Transformer样本数: {len(preds_trans)}")
    print(f"   实际值范围: [{actual.min():.2f}, {actual.max():.2f}]")
    print(f"   实际值均值: {actual.mean():.2f} ± {actual.std():.2f}")
    
    print(f"\n📈 性能对比:")
    print(f"   Ultra-LSNT:   R²={metrics_ultra['R2']:.4f}, RMSE={metrics_ultra['RMSE']:.2f}")
    print(f"   Transformer:  R²={metrics_trans['R2']:.4f}, RMSE={metrics_trans['RMSE']:.2f}")
    print(f"   差异 (Ultra-LSNT - Transformer):")
    print(f"     ΔR²: {metrics_ultra['R2'] - metrics_trans['R2']:.4f}")
    print(f"     ΔRMSE: {metrics_ultra['RMSE'] - metrics_trans['RMSE']:.2f}")
    
    # 4. 运行DM测试
    print(f"\n⚔️ 运行DM Test (Newey-West HAC)...")
    
    # 使用MSE作为损失函数
    dm_stat, p_value = dm_test_newey_west(actual, preds_ultra, preds_trans, criterion="MSE")
    
    print(f"   DM Statistic: {dm_stat:.4f}")
    print(f"   P-Value (单侧): {p_value:.4e}")
    
    # 解释结果
    if dm_stat < 0:
        if p_value < 0.01:
            conclusion = "**Ultra-LSNT显著优于Transformer (p<0.01)**"
        elif p_value < 0.05:
            conclusion = "*Ultra-LSNT显著优于Transformer (p<0.05)"
        else:
            conclusion = "Ultra-LSNT优于Transformer但不显著"
        print(f"   结论: {conclusion}")
        print(f"   (负DM统计量表示Ultra-LSNT误差更小)")
    else:
        if p_value < 0.01:
            conclusion = "**Transformer显著优于Ultra-LSNT (p<0.01)**"
        elif p_value < 0.05:
            conclusion = "*Transformer显著优于Ultra-LSNT (p<0.05)"
        else:
            conclusion = "Transformer优于Ultra-LSNT但不显著"
        print(f"   结论: {conclusion}")
        print(f"   (正DM统计量表示Transformer误差更小)")
    
    # 5. 保存结果
    result = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'wind_final.csv',
        'samples': len(actual),
        'ultra_lsnt': {
            'r2': float(metrics_ultra['R2']),
            'rmse': float(metrics_ultra['RMSE']),
            'mae': float(metrics_ultra['MAE']),
        },
        'transformer': {
            'r2': float(metrics_trans['R2']),
            'rmse': float(metrics_trans['RMSE']),
            'mae': float(metrics_trans['MAE']),
        },
        'dm_test': {
            'statistic': float(dm_stat),
            'p_value': float(p_value),
            'criterion': 'MSE',
            'conclusion': conclusion,
        },
        'table3_reference': {
            'ultra_lsnt_r2': 0.8727728724479675,
            'transformer_r2': 0.743889570236206,
            'ultra_lsnt_rmse': 9080.5546875,
            'transformer_rmse': 2916.826416015625,
        }
    }
    
    os.makedirs('dm_results', exist_ok=True)
    result_file = f"dm_results/dm_transformer_fair_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 结果已保存: {result_file}")
    
    # 6. 生成LaTeX格式表格
    print(f"\n📋 LaTeX格式结果表:")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Diebold-Mariano显著性检验结果 (Ultra-LSNT vs Transformer - 使用Table 3配置)}")
    print("\\label{tab:dm_transformer_fair}")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("模型 & R² & RMSE & DM统计量 \\\\")
    print("\\hline")
    print(f"Ultra-LSNT & {metrics_ultra['R2']:.4f} & {metrics_ultra['RMSE']:.2f} & --- \\\\")
    print(f"Transformer & {metrics_trans['R2']:.4f} & {metrics_trans['RMSE']:.2f} & {dm_stat:.4f} \\\\")
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