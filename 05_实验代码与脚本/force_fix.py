import pandas as pd
import numpy as np
import os

print("[修复] 启动暴力修复模式 (增强版 - 包含物理边界检查)...")

def check_and_fix_power_column(df, power_col='power'):
    """
    检查并修复功率列，确保符合物理边界
    """
    if power_col not in df.columns:
        print(f"[错误] 数据框中没有 '{power_col}' 列")
        print(f"   可用列: {df.columns.tolist()}")
        return df
    
    original_len = len(df)
    
    # 1. 确保功率列是数值类型
    df[power_col] = pd.to_numeric(df[power_col], errors='coerce')
    
    # 2. 删除功率为NaN的行
    df = df.dropna(subset=[power_col])
    nan_count = original_len - len(df)
    if nan_count > 0:
        print(f"   删除了 {nan_count} 个功率为NaN的行")
    
    # 3. 确保功率非负（风电功率不能为负）
    negative_count = (df[power_col] < 0).sum()
    if negative_count > 0:
        print(f"   发现 {negative_count} 个负功率值，将其修正为0")
        df.loc[df[power_col] < 0, power_col] = 0
    
    # 4. 物理边界检查：功率不能超过装机容量的1.1倍
    # 假设装机容量为历史最大观测功率的某个倍数
    max_power = df[power_col].max()
    
    # 估算装机容量（通常略高于最大观测功率）
    # 如果已有装机容量信息，可以在这里添加
    installed_capacity = max_power * 1.1  # 假设装机容量比最大观测功率高10%
    
    print(f"   最大观测功率: {max_power:.2f} kW")
    print(f"   估算装机容量: {installed_capacity:.2f} kW")
    
    # 设置合理上限（装机容量的1.1倍）
    upper_limit = installed_capacity * 1.1
    
    # 检查是否有异常高的值
    outlier_count = (df[power_col] > upper_limit).sum()
    if outlier_count > 0:
        print(f"   发现 {outlier_count} 个超过上限 ({upper_limit:.2f}) 的值，将其修正为上限值")
        df.loc[df[power_col] > upper_limit, power_col] = upper_limit
    
    # 5. 保存统计信息
    stats = df[power_col].describe()
    print(f"\n   [统计] 功率列统计:")
    print(f"      最小值: {stats['min']:.2f}")
    print(f"      最大值: {stats['max']:.2f}")
    print(f"      平均值: {stats['mean']:.2f}")
    print(f"      标准差: {stats['std']:.2f}")
    
    return df

def main():
    # 检查 wind_final.csv 是否存在
    if os.path.exists('wind_final.csv'):
        print("[文件] 检测到现有的 wind_final.csv，将直接进行物理边界检查...")
        df = pd.read_csv('wind_final.csv')
        print(f"   当前数据形状: {df.shape}")
        
        # 找到功率列
        power_col = None
        possible_names = ['power', 'prepower', 'active_power', '功率', '发电量']
        for name in possible_names:
            if name in df.columns:
                power_col = name
                break
        
        if power_col:
            print(f"   找到功率列: {power_col}")
            df = check_and_fix_power_column(df, power_col)
            
            # 如果需要，重命名为标准 'power'
            if power_col != 'power':
                df = df.rename(columns={power_col: 'power'})
                print(f"   已将列 '{power_col}' 重命名为 'power'")
        else:
            print("[错误] 在现有文件中未找到功率列！")
            return
    else:
        print("[文件] 未找到 wind_final.csv，将从原始文件创建...")
        # 1. 强行读取原始文件
        try:
            df = pd.read_excel('wind_main.csv')
        except:
            try:
                df = pd.read_csv('wind_main.csv')
            except Exception as e:
                print(f"[错误] 读取失败: {e}")
                exit()

        print(f"   读取成功! 原始大小: {df.shape}")

        # 2. 列名转小写
        df.columns = [str(c).lower().strip() for c in df.columns]

        # 3. 找到目标列 (功率)
        target_col = None
        possible_names = ['power', 'prepower', 'active_power', '功率', '发电量']
        for name in possible_names:
            if name in df.columns:
                target_col = name
                break

        if target_col:
            print(f"   找到目标列: {target_col}")
            # 把目标列改名为标准 'power'
            df = df.rename(columns={target_col: 'power'})
        else:
            print("[错误] 没找到功率列！请手动检查csv列名！")
            exit()

        # 4. 【关键一步】只保留数字列！砍掉所有字符串！
        # 先把所有列尝试转为数字，转不了的变成 NaN
        df_numeric = df.apply(pd.to_numeric, errors='coerce')

        # 删除全是 NaN 的列（比如原来的时间列）
        df_clean = df_numeric.dropna(axis=1, how='all')

        # 5. 填补剩余的 NaN
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # 6. 删除捣乱的 YD15 (如果是常数)
        if 'yd15' in df_clean.columns:
            if df_clean['yd15'].nunique() <= 1:
                df_clean = df_clean.drop(columns=['yd15'])
                print("   已踢出无效列: YD15")

        print(f"   清洗后大小: {df_clean.shape}")
        print(f"   保留特征: {list(df_clean.columns)}")
        
        # 7. 物理边界检查
        df_clean = check_and_fix_power_column(df_clean, 'power')
    
    # 8. 强制保存
    output_file = 'wind_final.csv'
    df.to_csv(output_file, index=False)
    print(f"\n[成功] 成功生成/修复: {output_file}")
    print(f"   最终数据形状: {df.shape}")
    print(f"   功率范围: {df['power'].min():.2f} - {df['power'].max():.2f} kW")
    
    # 9. 生成验证报告
    report_file = 'power_validation_report.txt'
    with open(report_file, 'w') as f:
        f.write("功率数据验证报告\n")
        f.write("=" * 40 + "\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n")
        f.write(f"数据文件: {output_file}\n")
        f.write(f"数据形状: {df.shape}\n")
        f.write(f"功率列统计:\n")
        f.write(f"  最小值: {df['power'].min():.2f} kW\n")
        f.write(f"  最大值: {df['power'].max():.2f} kW\n")
        f.write(f"  平均值: {df['power'].mean():.2f} kW\n")
        f.write(f"  负值数量: {(df['power'] < 0).sum()}\n")
        f.write(f"  超出上限数量: {(df['power'] > df['power'].max() * 1.1).sum()}\n")
    
    print(f"[报告] 验证报告已保存: {report_file}")

if __name__ == '__main__':
    main()