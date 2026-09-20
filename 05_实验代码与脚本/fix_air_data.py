import pandas as pd
import numpy as np

print("🧹 启动空气质量数据深度清洗 (V2 - 强力去噪版)...")

# 1. 读取数据
try:
    df = pd.read_csv('air_quality.csv', dtype=str)
    print(f"   读取成功，原始大小: {df.shape}")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    exit()

cols_map = {
    'aqi指数': 'AQI', 'pm2.5': 'PM2.5', 'pm10': 'PM10',
    'so2': 'SO2', 'no2': 'NO2', 'co': 'CO', 'o3': 'O3'
}

# 2. 清洗循环
for col_cn, col_en in cols_map.items():
    if col_cn in df.columns:
        # 转数字
        df[col_cn] = pd.to_numeric(df[col_cn], errors='coerce')
        # 重命名
        df = df.rename(columns={col_cn: col_en})

        # === 【新增】强力去极值 ===
        # 计算 99.9% 分位数 (剔除 0.1% 的极端异常值)
        # 比如把 6593 这种离谱数字变成正常的上限值
        upper_limit = df[col_en].quantile(0.999)
        lower_limit = 0  # 空气质量不能是负数

        # 执行截断
        df[col_en] = df[col_en].clip(lower=lower_limit, upper=upper_limit)

        print(f"   ✓ {col_en}: 范围限制在 [0, {upper_limit:.2f}]")

# 3. 筛选与保存
final_cols = [c for c in cols_map.values() if c in df.columns]
df_clean = df[final_cols].dropna()

output_file = 'air_quality_ready.csv'
df_clean.to_csv(output_file, index=False)
print(f"\n✅ 强力清洗完成: {output_file}")
print(f"   最终样本: {len(df_clean)}")