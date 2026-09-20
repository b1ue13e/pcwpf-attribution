"""
为electricity数据集运行Ultra-LSNT-Lite评估
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

sys.path.append('/root/论文1')

from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    compute_metrics
)
from ultra_lsnt_lite import UltraLSNTLiteForecaster

def evaluate_on_electricity():
    print("=" * 60)
    print("🚀 Ultra-LSNT-Lite on Electricity Dataset")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载数据
    data_path = '/root/论文1/Load_history.csv'
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None
    
    # electricity数据集的目标列是'load'
    data, _ = load_csv_data(data_path, 'load')
    input_dim = data.shape[1]
    print(f"数据形状: {data.shape}, 输入维度: {input_dim}")
    
    # 配置参数
    seq_len = 96
    pred_len = 24
    batch_size = 256
    epochs = 3
    
    ts_config = TimeSeriesConfig(seq_len=seq_len, pred_len=pred_len, target='load')
    train_config = TrainConfig(batch_size=batch_size, epochs=epochs)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        data, ts_config, train_config
    )
    
    # 优化配置
    model_config = {
        'input_dim': input_dim,
        'hidden_dim': 96,
        'seq_len': seq_len,
        'pred_len': pred_len,
        'num_blocks': 1,
        'num_experts': 2,
        'top_k': 1,
        'scales': [1, 2]
    }
    
    print("\n📋 模型配置:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # 创建模型
    model = UltraLSNTLiteForecaster(model_config).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 模型参数量: {total_params:,}")
    
    # 训练模型
    print("\n🏋️ 训练模型...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
    
    # 评估模型
    print("\n📊 评估模型...")
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.numpy())
    
    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # 反标准化
    predictions = scaler.inverse_transform(predictions)
    targets = scaler.inverse_transform(targets)
    
    # 计算指标
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print("\n" + "=" * 60)
    print("📈 评估结果:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print("=" * 60)
    
    # 保存结果
    results = {
        'dataset': 'electricity',
        'model': 'Ultra-LSNT-Lite',
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'params': total_params
    }
    
    return results

if __name__ == "__main__":
    results = evaluate_on_electricity()
    
    if results:
        # 保存到CSV
        import csv
        output_file = '/root/experiment_logs/ultra_lsnt_electricity_results.csv'
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['dataset', 'model', 'R2', 'RMSE', 'MAE', 'params'])
            writer.writeheader()
            writer.writerow(results)
        print(f"\n✅ 结果已保存到: {output_file}")
