"""
为electricity数据集运行Ultra-LSNT-Lite评估 (修复维度)
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

sys.path.append('/root/论文1')

def load_and_prepare_data(filepath):
    """加载并准备数据为3D格式"""
    print(f"加载数据: {filepath}")
    
    df = pd.read_csv(filepath, thousands=',')
    df = df.drop(columns=['zone_id', 'year', 'month', 'day'], errors='ignore')
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    # 数据形状: (samples, 24) -> 转换为3D (samples, seq_len, features)
    # 我们将24个h值作为seq_len=24, features=1
    data = df.values  # (31720, 24)
    data = data.reshape(data.shape[0], data.shape[1], 1)  # (31720, 24, 1)
    
    print(f"  数据形状: {data.shape}")
    return data

def create_sequences(data, seq_len=24, pred_len=1):
    """创建时间序列样本"""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])  # (seq_len, 1)
        y.append(data[i+seq_len:i+seq_len+pred_len, -1])  # 预测下一个时间步
    
    X = np.array(X)  # (samples, seq_len, 1)
    y = np.array(y).reshape(-1, pred_len)  # (samples, pred_len)
    
    return X, y

def main():
    print("=" * 60)
    print("🚀 Ultra-LSNT-Lite on Electricity (v3)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 加载数据
    data = load_and_prepare_data('/root/论文1/Load_history.csv')
    
    # 创建序列
    X, y = create_sequences(data, seq_len=24, pred_len=1)
    print(f"X: {X.shape}, y: {y.shape}")
    
    # 划分数据集
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_train = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # DataLoader
    train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128)
    
    # 模型
    from ultra_lsnt_lite import UltraLSNTLiteForecaster
    
    model = UltraLSNTLiteForecaster({
        'input_dim': 1,
        'hidden_dim': 64,
        'seq_len': 24,
        'pred_len': 1,
        'num_blocks': 1,
        'num_experts': 2,
        'top_k': 1,
        'scales': [1]
    }).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"📊 参数量: {params:,}\n")
    
    # 训练
    print("🏋️ 训练...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(3):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: {total_loss/len(train_loader):.6f}")
    
    # 评估
    print("\n📊 评估...")
    model.eval()
    preds, targets = [], []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            targets.append(yb.numpy())
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))
    
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    print(f"\n📈 R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    # 保存
    import json
    result = {
        'dataset': 'electricity',
        'model': 'Ultra-LSNT-Lite',
        'R2': float(r2),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'params': params
    }
    with open('/root/experiment_logs/ultra_lsnt_electricity_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("\n✅ 已保存结果")

if __name__ == "__main__":
    main()
