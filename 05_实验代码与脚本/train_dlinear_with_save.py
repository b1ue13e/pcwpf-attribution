"""
训练DLinear并保存权重，用于噪声鲁棒性测试
"""
import torch
import torch.nn as nn
import numpy as np
import os
import time
from run_dlinear import DLinear
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig, compute_metrics
)

def train_and_save_dlinear():
    print("🚀 训练DLinear并保存权重...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 数据
    data, _ = load_csv_data('wind_final.csv', 'power')
    if data is None:
        print("❌ 数据加载失败")
        return
    
    seq_len = 96
    pred_len = 24
    input_dim = data.shape[1]
    
    ts_config = TimeSeriesConfig(seq_len=seq_len, pred_len=pred_len, target='power')
    train_config = TrainConfig(batch_size=64, epochs=10, lr=0.005)
    
    train_loader, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)
    
    # 2. 模型
    model = DLinear(seq_len, pred_len, input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    criterion = nn.MSELoss()
    
    # 3. 训练
    print("开始训练...")
    model.train()
    training_losses = []
    
    for epoch in range(train_config.epochs):
        epoch_loss = 0
        epoch_start = time.time()
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{train_config.epochs}: Loss={avg_loss:.4f}, Time={epoch_time:.1f}s")
    
    # 4. 测试
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            preds.append(out.cpu().numpy())
            trues.append(by.cpu().numpy())
    
    # 反归一化
    target_mean, target_std = scaler.mean[-1], scaler.std[-1]
    preds = np.concatenate(preds) * target_std + target_mean
    trues = np.concatenate(trues) * target_std + target_mean
    
    metrics = compute_metrics(preds, trues)
    print(f"\n✅ DLinear 训练完成")
    print(f"   R²: {metrics['R2']:.4f}")
    print(f"   MSE: {metrics['MSE']:.2f}")
    print(f"   MAE: {metrics['MAE']:.2f}")
    
    # 5. 保存权重
    os.makedirs('dlinear_weights', exist_ok=True)
    weight_path = 'dlinear_weights/dlinear_trained.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'seq_len': seq_len,
            'pred_len': pred_len,
            'input_dim': input_dim
        },
        'metrics': metrics,
        'timestamp': time.strftime("%Y%m%d_%H%M%S")
    }, weight_path)
    
    print(f"✅ 权重已保存: {weight_path}")
    
    # 6. 保存测试结果
    result_path = 'dlinear_weights/dlinear_results.json'
    import json
    results = {
        'model': 'DLinear',
        'R2': float(metrics['R2']),
        'MSE': float(metrics['MSE']),
        'MAE': float(metrics['MAE']),
        'RMSE': float(metrics['RMSE']),
        'MAPE': float(metrics['MAPE']),
        'training_losses': training_losses,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'device': str(device),
        'config': {
            'seq_len': seq_len,
            'pred_len': pred_len,
            'input_dim': input_dim,
            'epochs': train_config.epochs,
            'batch_size': train_config.batch_size,
            'learning_rate': train_config.lr
        }
    }
    
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ 结果已保存: {result_path}")
    
    return model, weight_path, metrics

if __name__ == '__main__':
    train_and_save_dlinear()