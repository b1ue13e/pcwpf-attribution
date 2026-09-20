"""
Ultra-LSNT 演示脚本 - 使用 Scikit-learn 替代 PyTorch
用于展示数据处理和结果生成
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import os
import json
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🚀 Ultra-LSNT 演示训练")
print("="*70)

# 1. 加载数据
print("\n📊 加载数据...")
try:
    df = pd.read_csv('wind_final.csv')
    print(f"   ✓ 数据形状: {df.shape}")
    print(f"   ✓ 列名: {df.columns.tolist()}")
except Exception as e:
    print(f"   ❌ 数据加载失败: {e}")
    exit(1)

# 2. 数据预处理
print("\n🔧 数据预处理...")
data = df.values.astype(np.float32)
n_samples = len(data)
seq_len, pred_len = 96, 24

# 创建时序样本
X_samples, y_samples = [], []
for i in range(n_samples - seq_len - pred_len):
    X_samples.append(data[i:i+seq_len].flatten())
    y_samples.append(data[i+seq_len+pred_len, -1])  # 最后一列是power

X = np.array(X_samples, dtype=np.float32)
y = np.array(y_samples, dtype=np.float32)

print(f"   ✓ 创建了 {len(X)} 个样本")
print(f"   ✓ 特征维度: {X.shape[1]}")

# 划分数据集
train_idx = int(0.7 * len(X))
val_idx = int(0.85 * len(X))

X_train, y_train = X[:train_idx], y[:train_idx]
X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
X_test, y_test = X[val_idx:], y[val_idx:]

print(f"   ✓ 训练集: {len(X_train)}")
print(f"   ✓ 验证集: {len(X_val)}")
print(f"   ✓ 测试集: {len(X_test)}")

# 3. 标准化
print("\n📈 数据标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print("   ✓ 数据标准化完成")

# 4. 训练模型
print("\n🤖 训练模型...")

# Ultra-LSNT (用LR简化模拟)
print("   [1/3] 训练 Ultra-LSNT (线性回归模拟)...")
model_lsnt = LinearRegression()
model_lsnt.fit(X_train_scaled, y_train_scaled)
y_val_pred_lsnt = model_lsnt.predict(X_val_scaled)
y_test_pred_lsnt = model_lsnt.predict(X_test_scaled)
print("   ✓ Ultra-LSNT 训练完成")

# DLinear (LR)
print("   [2/3] 训练 DLinear (线性回归)...")
model_dlinear = LinearRegression()
model_dlinear.fit(X_train_scaled, y_train_scaled)
y_val_pred_dlinear = model_dlinear.predict(X_val_scaled)
y_test_pred_dlinear = model_dlinear.predict(X_test_scaled)
print("   ✓ DLinear 训练完成")

# GBDT
print("   [3/3] 训练 GBDT...")
model_gbdt = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=5)
model_gbdt.fit(X_train_scaled, y_train_scaled)
y_val_pred_gbdt = model_gbdt.predict(X_val_scaled)
y_test_pred_gbdt = model_gbdt.predict(X_test_scaled)
print("   ✓ GBDT 训练完成")

# 5. 评估指标
def compute_metrics(pred, true):
    """计算评估指标"""
    pred = pred.flatten()
    true = true.flatten()
    
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - true))
    
    mask = true != 0
    mape = np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100 if mask.any() else 0
    
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

print("\n📊 计算评估指标...")

# 验证集指标
metrics_val = {
    'Ultra-LSNT': compute_metrics(y_val_pred_lsnt, y_val_scaled),
    'DLinear': compute_metrics(y_val_pred_dlinear, y_val_scaled),
    'GBDT': compute_metrics(y_val_pred_gbdt, y_val_scaled),
}

# 测试集指标
metrics_test = {
    'Ultra-LSNT': compute_metrics(y_test_pred_lsnt, y_test_scaled),
    'DLinear': compute_metrics(y_test_pred_dlinear, y_test_scaled),
    'GBDT': compute_metrics(y_test_pred_gbdt, y_test_scaled),
}

print("\n✅ 验证集结果:")
for model_name, metrics in metrics_val.items():
    print(f"\n  {model_name}:")
    for metric, value in metrics.items():
        print(f"    {metric}: {value:.6f}")

print("\n✅ 测试集结果:")
for model_name, metrics in metrics_test.items():
    print(f"\n  {model_name}:")
    for metric, value in metrics.items():
        print(f"    {metric}: {value:.6f}")

# 6. 保存结果
print("\n💾 保存结果...")
results = {
    'val_metrics': metrics_val,
    'test_metrics': metrics_test,
    'models': ['Ultra-LSNT', 'DLinear', 'GBDT'],
}

os.makedirs('results', exist_ok=True)
with open('results/metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

# 保存预测结果
np.savez('results/predictions.npz',
         test_pred_lsnt=y_test_pred_lsnt,
         test_pred_dlinear=y_test_pred_dlinear,
         test_pred_gbdt=y_test_pred_gbdt,
         test_true=y_test_scaled,
         val_pred_lsnt=y_val_pred_lsnt,
         val_pred_dlinear=y_val_pred_dlinear,
         val_pred_gbdt=y_val_pred_gbdt,
         val_true=y_val_scaled)

print("   ✓ 结果已保存到 results/")

# 7. 生成图表
print("\n📈 生成图表...")

# 创建图表目录
os.makedirs('figures', exist_ok=True)

# 图1: 模型对比 (测试集)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ultra-LSNT vs Baselines - 测试集对比', fontsize=16, fontweight='bold')

# 1.1 预测 vs 真实值
ax = axes[0, 0]
n_plot = min(200, len(y_test_scaled))
ax.plot(y_test_scaled[:n_plot], 'o-', label='True', alpha=0.7, linewidth=2)
ax.plot(y_test_pred_lsnt[:n_plot], 's-', label='Ultra-LSNT', alpha=0.7, linewidth=2)
ax.plot(y_test_pred_dlinear[:n_plot], '^-', label='DLinear', alpha=0.7, linewidth=2)
ax.plot(y_test_pred_gbdt[:n_plot], 'v-', label='GBDT', alpha=0.7, linewidth=2)
ax.set_xlabel('时间步')
ax.set_ylabel('标准化功率')
ax.set_title('(a) 预测 vs 真实值')
ax.legend()
ax.grid(True, alpha=0.3)

# 1.2 R² 对比
ax = axes[0, 1]
models = ['Ultra-LSNT', 'DLinear', 'GBDT']
r2_values = [metrics_test[m]['R2'] for m in models]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax.bar(models, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('R² Score')
ax.set_title('(b) R² 对比')
ax.set_ylim([0, 1])
for bar, val in zip(bars, r2_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 1.3 RMSE 对比
ax = axes[1, 0]
rmse_values = [metrics_test[m]['RMSE'] for m in models]
bars = ax.bar(models, rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('RMSE')
ax.set_title('(c) RMSE 对比')
for bar, val in zip(bars, rmse_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 1.4 散点图
ax = axes[1, 1]
ax.scatter(y_test_scaled[:500], y_test_pred_lsnt[:500], alpha=0.5, s=30, label='Ultra-LSNT')
ax.scatter(y_test_scaled[:500], y_test_pred_dlinear[:500], alpha=0.5, s=30, label='DLinear')
min_val = min(y_test_scaled.min(), y_test_pred_lsnt.min())
max_val = max(y_test_scaled.max(), y_test_pred_lsnt.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
ax.set_xlabel('真实值')
ax.set_ylabel('预测值')
ax.set_title('(d) 预测精度散点图')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=150, bbox_inches='tight')
print("   ✓ 图1: model_comparison.png")
plt.close()

# 图2: 指标汇总表
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# 准备表数据
table_data = []
table_data.append(['模型', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R²'])

for model in models:
    m = metrics_test[model]
    row = [model, f"{m['MSE']:.6f}", f"{m['RMSE']:.6f}", f"{m['MAE']:.6f}", 
           f"{m['MAPE']:.2f}", f"{m['R2']:.6f}"]
    table_data.append(row)

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# 设置表头颜色
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 交替行颜色
for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F0F0F0')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

plt.title('测试集评估指标汇总', fontsize=14, fontweight='bold', pad=20)
plt.savefig('figures/metrics_table.png', dpi=150, bbox_inches='tight')
print("   ✓ 图2: metrics_table.png")
plt.close()

# 图3: 误差分布
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('测试集误差分布', fontsize=14, fontweight='bold')

errors = {
    'Ultra-LSNT': y_test_scaled - y_test_pred_lsnt,
    'DLinear': y_test_scaled - y_test_pred_dlinear,
    'GBDT': y_test_scaled - y_test_pred_gbdt,
}

for idx, (model, error) in enumerate(errors.items()):
    axes[idx].hist(error, bins=50, color=colors[idx], alpha=0.7, edgecolor='black')
    axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[idx].set_xlabel('误差')
    axes[idx].set_ylabel('频次')
    axes[idx].set_title(model)
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/error_distribution.png', dpi=150, bbox_inches='tight')
print("   ✓ 图3: error_distribution.png")
plt.close()

# 图4: 验证集 vs 测试集
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('验证集 vs 测试集性能对比', fontsize=14, fontweight='bold')

metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'R2']
x = np.arange(len(models))
width = 0.35

for idx, metric in enumerate([0, 3]):  # MSE 和 R²
    ax = axes[idx]
    metric_name = metrics_to_plot[metric]
    
    val_vals = [metrics_val[m][metric_name] for m in models]
    test_vals = [metrics_test[m][metric_name] for m in models]
    
    ax.bar(x - width/2, val_vals, width, label='验证集', alpha=0.8, color='#4ECDC4')
    ax.bar(x + width/2, test_vals, width, label='测试集', alpha=0.8, color='#FF6B6B')
    
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} 对比')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/val_test_comparison.png', dpi=150, bbox_inches='tight')
print("   ✓ 图4: val_test_comparison.png")
plt.close()

# 图5: 时序预测
fig, ax = plt.subplots(figsize=(15, 6))

n_show = 300
x_axis = np.arange(n_show)

ax.plot(x_axis, y_test_scaled[:n_show], 'o-', label='实际值', color='black', 
        linewidth=2.5, markersize=4, alpha=0.8)
ax.plot(x_axis, y_test_pred_lsnt[:n_show], 's-', label='Ultra-LSNT', 
        linewidth=2, markersize=3, alpha=0.7)
ax.plot(x_axis, y_test_pred_dlinear[:n_show], '^-', label='DLinear', 
        linewidth=2, markersize=3, alpha=0.7)
ax.plot(x_axis, y_test_pred_gbdt[:n_show], 'v-', label='GBDT', 
        linewidth=2, markersize=3, alpha=0.7)

ax.fill_between(x_axis, y_test_pred_lsnt[:n_show], y_test_scaled[:n_show], 
                 alpha=0.2, color='#FF6B6B', label='Ultra-LSNT 误差')

ax.set_xlabel('时间步', fontsize=12)
ax.set_ylabel('标准化功率值', fontsize=12)
ax.set_title('测试集时序预测结果 (前300步)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/time_series_prediction.png', dpi=150, bbox_inches='tight')
print("   ✓ 图5: time_series_prediction.png")
plt.close()

print("\n" + "="*70)
print("🎉 所有图表已生成！")
print("="*70)
print("\n📁 生成的文件:")
print("   📊 results/metrics.json - 评估指标")
print("   📊 results/predictions.npz - 预测结果")
print("   📈 figures/model_comparison.png")
print("   📈 figures/metrics_table.png")
print("   📈 figures/error_distribution.png")
print("   📈 figures/val_test_comparison.png")
print("   📈 figures/time_series_prediction.png")
print("\n✅ 演示完成！")
