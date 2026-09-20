"""
clean_gefcom.py - 专门清洗 GEFCom2012 数据
"""
import pandas as pd
import numpy as np


def clean_gefcom():
    print("🧹 开始清洗 GEFCom 数据...")

    # 1. 读取数据
    try:
        df = pd.read_csv('Load_history.csv')
        print(f"   读取成功，原始形状: {df.shape}")
    except FileNotFoundError:
        print("❌ 没找到 Load_history.csv，请把它复制到当前目录！")
        return

    # 2. 移除千分位逗号（如果有）
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '')

    # 3. 宽表转长表 (关键步骤)
    # 原始格式: zone_id, year, month, day, h1, h2 ... h24
    # 目标格式: date, load

    # 只要 Zone 1 的数据（做实验只选一个区域就够了，没必要全跑）
    # 通常 Zone 1 数据质量最好
    if 'zone_id' in df.columns:
        df = df[df['zone_id'] == 1].copy()
        print("   已筛选 Zone 1 数据")

    # 提取小时列 (h1...h24)
    hour_cols = [c for c in df.columns if c.startswith('h') and c[1:].isdigit()]

    # 融合 (Melt)
    id_vars = ['year', 'month', 'day']
    df_melted = df.melt(id_vars=id_vars, value_vars=hour_cols, var_name='hour_str', value_name='load')

    # 处理小时 (h1 -> 0, h24 -> 23)
    df_melted['hour'] = df_melted['hour_str'].str.replace('h', '').astype(int) - 1

    # 生成标准时间戳
    df_melted['date'] = pd.to_datetime(df_melted[['year', 'month', 'day', 'hour']])

    # 排序
    df_final = df_melted.sort_values('date').reset_index(drop=True)

    # 只保留需要的列
    output_df = df_final[['date', 'load']]

    # 去除空值
    output_df = output_df.dropna()

    print(f"   清洗完成！最终形状: {output_df.shape}")
    print(f"   时间范围: {output_df['date'].min()} 到 {output_df['date'].max()}")

    # 4. 保存
    output_df.to_csv('gefcom_ready.csv', index=False)
    print("✅ 已保存为: gefcom_ready.csv (直接用这个跑训练！)")


if __name__ == '__main__':
    clean_gefcom()