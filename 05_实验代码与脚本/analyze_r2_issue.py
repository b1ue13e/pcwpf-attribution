import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import json

print("=== Analyzing R2 Anomaly Issue ===")

# 加载数据
df = pd.read_csv('wind_final.csv')
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

data = df.values.astype(np.float32)
n_samples = len(data)
seq_len, pred_len = 96, 24

# 创建时序样本
X_samples, y_samples = [], []
for i in range(n_samples - seq_len - pred_len):
    X_samples.append(data[i:i+seq_len].flatten())
    y_samples.append(data[i+seq_len+pred_len, -1])

X = np.array(X_samples, dtype=np.float32)
y = np.array(y_samples, dtype=np.float32)

print(f"创建了 {len(X)} 个样本")

# 划分数据集
train_idx = int(0.7 * len(X))
val_idx = int(0.85 * len(X))

X_train, y_train = X[:train_idx], y[:train_idx]
X_test, y_test = X[val_idx:], y[val_idx:]

print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"\n标准化后数据统计:")
print(f"y_test_scaled 范围: [{y_test_scaled.min():.6f}, {y_test_scaled.max():.6f}]")
print(f"y_test_scaled 均值: {np.mean(y_test_scaled):.6f}")
print(f"y_test_scaled 标准差: {np.std(y_test_scaled):.6f}")

# 训练模型并计算R²
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print(f"\n预测结果统计:")
print(f"y_test_pred 范围: [{y_test_pred.min():.6f}, {y_test_pred.max():.6f}]")
print(f"y_test_pred 均值: {np.mean(y_test_pred):.6f}")
print(f"y_test_pred 标准差: {np.std(y_test_pred):.6f}")

# 计算R²
ss_res = np.sum((y_test_scaled - y_test_pred) ** 2)
ss_tot = np.sum((y_test_scaled - np.mean(y_test_scaled)) ** 2)
r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

print(f"\nR2 Calculation Details:")
print(f"SS_res: {ss_res:.10f}")
print(f"SS_tot: {ss_tot:.10f}")
print(f"R2: {r2:.10f}")

# 检查是否存在数值精度问题
print(f"\nNumerical Precision Analysis:")
print(f"SS_res / SS_tot: {ss_res/ss_tot:.10f}")
print(f"1 - (SS_res / SS_tot): {1 - (ss_res/ss_tot):.10f}")

# 与当前metrics.json对比
print(f"\nComparison with current metrics.json:")
try:
    with open('results/metrics.json', 'r') as f:
        current_metrics = json.load(f)
    
    if 'Ultra-LSNT' in current_metrics:
        current_r2 = current_metrics['Ultra-LSNT']['r2']
        print(f"当前metrics.json中的R²: {current_r2}")
        print(f"实际计算的R²: {r2}")
        print(f"差异: {abs(current_r2 - r2)}")
except Exception as e:
    print(f"Cannot read current metrics.json: {e}")

# 检查是否可能是标准化导致的异常
print(f"\nStandardization Impact Analysis:")
print(f"原始y_test范围: [{y_test.min():.2f}, {y_test.max():.2f}]")
print(f"标准化y_test范围: [{y_test_scaled.min():.6f}, {y_test_scaled.max():.6f}]")

# 如果SS_tot非常小，可能导致R²异常接近1
if ss_tot < 1e-10:
    print("Warning: SS_tot is very small, may cause R2 to be abnormally close to 1!")
    print("This may be due to numerical precision issues caused by small data range after standardization")
else:
    print(f"SS_tot size normal: {ss_tot}")