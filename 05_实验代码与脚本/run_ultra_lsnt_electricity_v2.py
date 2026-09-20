"""
为electricity数据集运行Ultra-LSNT-Lite评估 (适配h1-h24格式)
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

sys.path.append('/root/论文1')

def load_electricity_data(filepath):
    """加载electricity数据，使用h24作为目标"""
    print(f"加载数据: {filepath}")
    
    # 读取CSV，处理千分位分隔符
    df = pd.read_csv(filepath, thousands=',')
    print(f"  原始形状: {df.shape}")
    print(f"  列: {list(df.columns)}")
    
    # 删除非特征列
    drop_cols = ['zone_id', 'year', 'month', 'day']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # 将所有h列转换为数值
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除有缺失值的行
    df = df.dropna()
    print(f"  处理后形状: {df.shape}")
    
    return df.values, list(df.columns)

def create_simple_dataloaders(data, seq_len=96, pred_len=24, batch_size=256):
    """简化版数据加载器"""
    # 数据形状: (samples, features)
    # features: h1-h24 (24个负荷值)
    
    # 使用前seq_len个时间步预测pred_len个时间步
    # 这里我们将每个样本的24小时作为24个时间步
    
    X, y = [], []
    for i in range(len(data) - 1):
        # 使用当前样本的所有h值作为输入
        # 预测下一个样本的h24（最后一小时负荷）
        X.append(data[i])
        y.append(data[i+1][-1])  # h24是最后一列
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    print(f"  X形状: {X.shape}, y形状: {y.shape}")
    
    # 划分训练/测试集 (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    
    # 创建DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler_y

def evaluate_on_electricity():
    print("=" * 60)
    print("🚀 Ultra-LSNT-Lite on Electricity Dataset (v2)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载数据
    data_path = '/root/论文1/Load_history.csv'
    data, columns = load_electricity_data(data_path)
    input_dim = data.shape[1]
    
    # 创建数据加载器
    train_loader, test_loader, scaler_y = create_simple_dataloaders(data, seq_len=24, pred_len=1, batch_size=256)
    
    # 模型配置
    from ultra_lsnt_lite import UltraLSNTLiteForecaster
    
    model_config = {
        'input_dim': input_dim,
        'hidden_dim': 64,
        'seq_len': 24,
        'pred_len': 1,
        'num_blocks': 1,
        'num_experts': 2,
        'top_k': 1,
        'scales': [1]
    }
    
    print("\n📋 模型配置:", model_config)
    
    model = UltraLSNTLiteForecaster(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 参数量: {total_params:,}")
    
    # 训练
    print("\n🏋️ 训练模型 (3 epochs)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(3):
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.6f}")
    
    # 评估
    print("\n📊 评估模型...")
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.numpy())
    
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # 反标准化
    predictions = scaler_y.inverse_transform(predictions)
    targets = scaler_y.inverse_transform(targets)
    
    # 计算指标
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"📈 结果: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    print("=" * 60)
    
    return {
        'dataset': 'electricity',
        'model': 'Ultra-LSNT-Lite',
        'R2': float(r2),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'params': total_params
    }

if __name__ == "__main__":
    results = evaluate_on_electricity()
    if results:
        import json
        with open('/root/experiment_logs/ultra_lsnt_electricity_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n✅ 结果已保存")
