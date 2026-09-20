"""
轻量化 Mamba (S6) 对比实验
简化实现，用于获取基准性能数据
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.getcwd())
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    compute_metrics
)

# 简化版Mamba块 (基于S6结构)
class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * d_model)
        
        # 简化实现：使用线性层模拟
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, self.d_conv, groups=self.d_inner, padding=self.d_conv-1)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        xz = self.in_proj(x)  # [B, L, 2*D_inner]
        x, z = xz.chunk(2, dim=-1)
        
        # 简化卷积
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        
        # 简化状态空间模型
        dt = self.dt_proj(x)
        # 简化处理：直接使用线性变换
        y = x + dt * 0.1
        y = y * torch.sigmoid(z)
        
        return self.out_proj(y)

class MambaForecaster(nn.Module):
    def __init__(self, input_dim, seq_len=96, pred_len=24, d_model=64, n_layers=4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            SimpleMambaBlock(d_model) for _ in range(n_layers)
        ])
        self.decoder = nn.Linear(d_model * seq_len, pred_len)
        
    def forward(self, x):
        # x: [B, L, D]
        x = self.embedding(x)
        for block in self.blocks:
            x = x + block(x)
        
        # 全局池化 + 解码
        B, L, D = x.shape
        x = x.reshape(B, -1)
        out = self.decoder(x).unsqueeze(-1)  # [B, pred_len, 1]
        return out, torch.tensor(0.0)  # 保持与Ultra-LSNT相同的返回格式

def run_mamba_comparison():
    print("🤖 运行 Mamba (S6) 对比实验...")
    
    # 加载数据
    data, _ = load_csv_data('wind_final.csv', 'power')
    input_dim = data.shape[1]
    
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    train_config = TrainConfig(batch_size=64, epochs=10)  # 快速训练
    
    train_loader, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaForecaster(input_dim).to(device)
    
    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(train_config.epochs):
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out, _ = model(bx)
            loss = criterion(out.squeeze(), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{train_config.epochs}: Loss = {train_loss/len(train_loader):.4f}")
    
    # 测试
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            out, _ = model(bx)
            preds.append(out.cpu().numpy())
            trues.append(by.cpu().numpy())
    
    # 计算指标
    target_std = scaler.std[-1]
    target_mean = scaler.mean[-1]
    p = np.concatenate(preds) * target_std + target_mean
    y = np.concatenate(trues) * target_std + target_mean
    
    metrics = compute_metrics(p, y)
    print(f"📊 Mamba (S6) 性能:")
    print(f"  R²: {metrics['R2']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    
    # 保存结果
    import json
    results = {
        'model': 'Mamba_S6',
        'R2': float(metrics['R2']),
        'RMSE': float(metrics['RMSE']),
        'MAE': float(metrics['MAE']),
        'params': sum(p.numel() for p in model.parameters())
    }
    
    with open('mamba_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Mamba对比实验完成，结果保存到 mamba_results.json")

if __name__ == '__main__':
    run_mamba_comparison()
