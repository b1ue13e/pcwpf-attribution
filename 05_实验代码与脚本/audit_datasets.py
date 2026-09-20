import pandas as pd
import numpy as np

print('=== 数据集审计表 ===')
print()

# 1. 风电机组数据
df = pd.read_csv('prepared_priority_datasets/wind_turbine_primary.csv')
print('【风电机组运行数据集】')
print(f'  形状: {df.shape}')
print(f'  时间范围: {df.datetime.iloc[0]} ~ {df.datetime.iloc[-1]}')
print(f'  特征数: {len(df.columns)-2}')  # 减去datetime和power
print(f'  功率范围: {df.power.min():.2f} ~ {df.power.max():.2f} kW')
print(f'  总缺失率: {df.isnull().sum().sum() / (df.shape[0]*df.shape[1]):.4f}')
print(f'  零功率比例: {(df.power == 0).mean():.4f}')
print(f'  关键字段: windspeed, prepower, winddirection, temperature, humidity, pressure')
print(f'  负功率数: {(df.power < 0).sum()}')
print(f'  功率>50000数: {(df.power > 50000).sum()}')

# 检查observed faults
print(f'  连续相同功率计数: {(df.power.diff() == 0).sum()}')
print()

# 2. 新疆数据
df = pd.read_csv('prepared_priority_datasets/xinjiang_wind_2019.csv')
print('【新疆风电场数据】')
print(f'  形状: {df.shape}')
print(f'  时间范围: {df.datetime.iloc[0]} ~ {df.datetime.iloc[-1]}')
print(f'  特征数: {len(df.columns)-2}')
print(f'  功率范围: {df.power.min():.2f} ~ {df.power.max():.2f} MW')
print(f'  总缺失率: {df.isnull().sum().sum() / (df.shape[0]*df.shape[1]):.4f}')
print()

# 3. 美国数据
df = pd.read_csv('prepared_priority_datasets/us_wind_2012.csv')
print('【美国风电场数据】')
print(f'  形状: {df.shape}')
print(f'  时间范围: {df.datetime.iloc[0]} ~ {df.datetime.iloc[-1]}')
print(f'  特征数: {len(df.columns)-2}')
print(f'  功率范围: {df.power.min():.2f} ~ {df.power.max():.2f} MW')
print(f'  总缺失率: {df.isnull().sum().sum() / (df.shape[0]*df.shape[1]):.4f}')
print()

# 4. 原始风场数据（论文主数据）
df = pd.read_csv('wind_final.csv')
print('【原始Wind(CN)数据（论文主数据）】')
print(f'  形状: {df.shape}')
print(f'  功率范围: {df.power.min():.2f} ~ {df.power.max():.2f} kW')
print(f'  特征数: {len(df.columns)-2}')

# 兼容性检查
print()
print('=== 与当前(L,H)=(96,24)协议兼容性 ===')
for name, file in [('风电机组', 'wind_turbine_primary'), ('新疆', 'xinjiang_wind_2019'), ('美国', 'us_wind_2012')]:
    df = pd.read_csv(f'prepared_priority_datasets/{file}.csv')
    n_samples = len(df)
    min_needed = 96 + 24  # seq_len + pred_len
    print(f'{name}: {n_samples}样本, 最少需要{min_needed}, {"兼容" if n_samples >= min_needed*10 else "样本偏少"}')
