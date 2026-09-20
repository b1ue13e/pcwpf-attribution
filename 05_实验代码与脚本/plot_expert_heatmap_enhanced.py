import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许多次加载 OpenMP（绕过 OMP 错误）

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd
from datetime import datetime

# 引入你的核心库
sys.path.append(os.getcwd())
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    UltraLSNTForecaster, LSNTConfig
)

# 设置字体
plt.rcParams['font.family'] = 'sans-serif'
# # # # plt.rcParams['font.serif'] = ['DejaVu Sans']  # 注释掉以避免警告  # 注释掉以避免警告  # 注释掉以避免警告  # 注释掉以避免警告

def extract_expert_activation(model, test_loader, device, num_experts=4):
    """
    从测试数据中提取专家激活概率
    """
    target_router = model.blocks[0].moe_router  # 提取第一层的 Router
    
    # 初始化统计矩阵
    hourly_experts = {h: np.zeros(num_experts) for h in range(24)}
    counts = {h: 0 for h in range(24)}
    
    print("[扫描] 正在扫描测试集数据...")
    current_idx = 0
    total_samples = 0
    
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            
            # 手动跑前向传播到 Router 层
            enc_out = model.encoder(bx)
            norm_out = model.blocks[0].input_norm(enc_out)
            logits = target_router.router(norm_out)  # (Batch, num_experts)
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            
            batch_size = probs.shape[0]
            
            for i in range(batch_size):
                # 简单估算时间 (假设数据是连续的15分钟间隔)
                # 96个点 = 1天。 idx % 96 可以映射到一天的时间
                # 0-95 -> 0-23点
                time_step = current_idx % 96
                hour = int(time_step / 4)  # 15min * 4 = 1 hour
                
                if hour < 24:
                    hourly_experts[hour] += probs[i]
                    counts[hour] += 1
                
                current_idx += 1
                total_samples += 1
            
            if total_samples >= 96 * 100:  # 限制样本数，防止内存溢出
                break
    
    # 计算平均激活概率
    heatmap_data = np.zeros((num_experts, 24))
    for h in range(24):
        if counts[h] > 0:
            heatmap_data[:, h] = hourly_experts[h] / counts[h]
    
    return heatmap_data, counts, total_samples

def analyze_wind_data():
    """分析风电数据的专家分工模式"""
    print("[风电] 分析风电数据专家分工模式...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载风电数据
    data_path = 'wind_final.csv'
    if not os.path.exists(data_path):
        print(f"[错误] 找不到 {data_path}，请先运行 force_fix.py 或 data_preprocess.py")
        return None
    
    try:
        data, feature_names = load_csv_data(data_path, 'power')
        print(f"[成功] 风电数据加载成功: {data.shape}")
    except ValueError as e:
        print(f"[错误] 风电数据加载失败: {e}")
        return None
    
    # 2. 配置
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    train_config = TrainConfig(batch_size=32)  # 适当增大batch_size以提高效率
    _, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)
    
    # 3. 加载模型 - 使用与现有权重匹配的配置
    input_dim = data.shape[1]
    model_config = LSNTConfig(
        input_dim=input_dim,
        hidden_dim=128,           # 匹配现有权重
        num_blocks=4,             # 匹配现有权重
        num_experts=4,            # 匹配现有权重
        top_k=2,                  # 通常为num_experts的一半
        heteroscedastic_moe=False # 现有权重无uncertainty_net
    )
    model = UltraLSNTForecaster(model_config, ts_config).to(device)

    ckpt_path = 'checkpoints_ts/main/best_model.pth'
    if not os.path.exists(ckpt_path):
        print("[错误] 找不到模型权重，请先训练主模型")
        return None

    print(f"  加载权重文件: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 获取模型状态字典
    if 'model_state_dict' in checkpoint:
        pretrained_dict = checkpoint['model_state_dict']
    else:
        pretrained_dict = checkpoint
    
    # 获取当前模型状态字典
    model_dict = model.state_dict()
    
    # 过滤掉不匹配的键
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                      if k in model_dict and model_dict[k].shape == v.shape}
    
    # 更新模型字典
    model_dict.update(pretrained_dict)
    
    # 加载过滤后的权重
    model.load_state_dict(model_dict)
    
    # 检查加载了多少参数
    matched_keys = len(pretrained_dict)
    total_keys = len(model_dict)
    print(f"  权重加载完成: {matched_keys}/{total_keys} 个参数匹配")
    
    if matched_keys < total_keys:
        print(f"  警告: 有 {total_keys - matched_keys} 个参数未从检查点加载")
    
    model.eval()
    print("[成功] 风电模型已加载")
    
    # 4. 提取专家激活
    heatmap_data, counts, total_samples = extract_expert_activation(model, test_loader, device)
    
    # 5. 保存结果
    result_file = 'wind_expert_activation.csv'
    df_result = pd.DataFrame(heatmap_data.T, columns=[f'Expert_{i+1}' for i in range(4)])
    df_result['Hour'] = range(24)
    df_result['Count'] = [counts[h] for h in range(24)]
    df_result.to_csv(result_file, index=False)
    print(f"[成功] 风电专家激活数据已保存: {result_file}")
    
    return heatmap_data

def analyze_air_quality_data():
    """分析空气质量数据的专家分工模式"""
    print("[空气质量] 分析空气质量数据专家分工模式...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载空气质量数据
    data_path = 'air_quality_ready.csv'
    if not os.path.exists(data_path):
        print(f"[错误] 找不到 {data_path}，请确保文件存在")
        return None
    
    try:
        # 空气质量数据的目标列是 'AQI'
        data, feature_names = load_csv_data(data_path, 'AQI')
        print(f"[成功] 空气质量数据加载成功: {data.shape}")
        print(f"   特征列: {feature_names}")
    except ValueError as e:
        print(f"[错误] 空气质量数据加载失败: {e}")
        # 尝试自动检测目标列
        try:
            df = pd.read_csv(data_path)
            print(f"   数据列名: {df.columns.tolist()}")
            print(f"   数据形状: {df.shape}")
            
            # 尝试使用第一列作为目标
            target_col = df.columns[0]
            print(f"   尝试使用目标列: {target_col}")
            data, feature_names = load_csv_data(data_path, target_col)
        except Exception as e2:
            print(f"[错误] 再次失败: {e2}")
            return None
    
    # 2. 配置 (使用与风电相同的配置以确保公平比较)
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='AQI')
    train_config = TrainConfig(batch_size=32)
    _, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)
    
    # 3. 加载相同的模型权重 (使用在风电数据上训练的模型) - 使用权重过滤方法
    input_dim = data.shape[1]
    model_config = LSNTConfig(
        input_dim=input_dim,
        hidden_dim=128,           # 匹配现有权重
        num_blocks=4,             # 匹配现有权重
        num_experts=4,            # 匹配现有权重
        top_k=2,                  # 通常为num_experts的一半
        heteroscedastic_moe=False # 现有权重无uncertainty_net
    )
    model = UltraLSNTForecaster(model_config, ts_config).to(device)
    
    ckpt_path = 'checkpoints_ts/main/best_model.pth'
    if not os.path.exists(ckpt_path):
        print("[错误] 找不到模型权重")
        return None
    
    print(f"  加载权重文件: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 获取模型状态字典
    if 'model_state_dict' in checkpoint:
        pretrained_dict = checkpoint['model_state_dict']
    else:
        pretrained_dict = checkpoint
    
    # 获取当前模型状态字典
    model_dict = model.state_dict()
    
    # 过滤掉不匹配的键（特别是输入维度不匹配的键）
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                      if k in model_dict and model_dict[k].shape == v.shape}
    
    # 更新模型字典
    model_dict.update(pretrained_dict)
    
    # 加载过滤后的权重
    model.load_state_dict(model_dict)
    
    # 检查加载了多少参数
    matched_keys = len(pretrained_dict)
    total_keys = len(model_dict)
    print(f"  权重加载完成: {matched_keys}/{total_keys} 个参数匹配")
    
    if matched_keys < total_keys:
        print(f"  警告: 有 {total_keys - matched_keys} 个参数未从检查点加载")
    
    model.eval()
    print("[成功] 空气质量模型已加载 (使用风电训练的权重，经过过滤)")
    
    # 4. 提取专家激活
    heatmap_data, counts, total_samples = extract_expert_activation(model, test_loader, device)
    
    # 5. 保存结果
    result_file = 'air_quality_expert_activation.csv'
    df_result = pd.DataFrame(heatmap_data.T, columns=[f'Expert_{i+1}' for i in range(4)])
    df_result['Hour'] = range(24)
    df_result['Count'] = [counts[h] for h in range(24)]
    df_result.to_csv(result_file, index=False)
    print(f"[成功] 空气质量专家激活数据已保存: {result_file}")
    
    return heatmap_data

def plot_comparison_heatmaps(wind_heatmap, air_heatmap):
    """绘制风电和空气质量数据的专家分工对比图"""
    if wind_heatmap is None or air_heatmap is None:
        print("[错误] 无法生成对比图：缺少数据")
        return
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # 风电数据热图
    sns.heatmap(wind_heatmap, cmap="Reds", ax=axes[0],
                xticklabels=range(24),
                yticklabels=[f'Expert {i + 1}' for i in range(4)],
                cbar_kws={'label': 'Activation Probability'})
    axes[0].set_xlabel('Hour of Day (0-23)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Experts', fontsize=12, fontweight='bold')
    axes[0].set_title('Wind Power: Expert Specialization', fontsize=14, fontweight='bold')
    
    # 空气质量数据热图
    sns.heatmap(air_heatmap, cmap="Blues", ax=axes[1],
                xticklabels=range(24),
                yticklabels=[f'Expert {i + 1}' for i in range(4)],
                cbar_kws={'label': 'Activation Probability'})
    axes[1].set_xlabel('Hour of Day (0-23)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Experts', fontsize=12, fontweight='bold')
    axes[1].set_title('Air Quality: Expert Specialization', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存对比图
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'fig_expert_comparison_{timestamp}.png'
    plt.savefig(output_file, dpi=300)
    plt.savefig('fig_expert_comparison_latest.png', dpi=300)
    print(f"[成功] 专家分工对比图已保存: {output_file}")
    print(f"[成功] 最新对比图已保存: fig_expert_comparison_latest.png")
    
    # 生成分析报告
    generate_analysis_report(wind_heatmap, air_heatmap)

def generate_analysis_report(wind_heatmap, air_heatmap):
    """生成专家分工分析报告"""
    if wind_heatmap is None or air_heatmap is None:
        return
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("专家分工模式分析报告")
    report_lines.append("=" * 60)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 分析风电数据
    report_lines.append("风电数据专家分工分析:")
    report_lines.append("-" * 40)
    
    for expert_idx in range(4):
        expert_row = wind_heatmap[expert_idx]
        peak_hour = np.argmax(expert_row)
        peak_value = np.max(expert_row)
        avg_activation = np.mean(expert_row)
        
        report_lines.append(f"Expert {expert_idx + 1}:")
        report_lines.append(f"  峰值激活时间: {peak_hour}:00")
        report_lines.append(f"  峰值激活概率: {peak_value:.4f}")
        report_lines.append(f"  平均激活概率: {avg_activation:.4f}")
        
        # 找出激活概率较高的时间段
        high_activation_hours = [h for h in range(24) if expert_row[h] > avg_activation * 1.2]
        if high_activation_hours:
            report_lines.append(f"  高激活时间段: {high_activation_hours}")
        report_lines.append("")
    
    # 分析空气质量数据
    report_lines.append("空气质量数据专家分工分析:")
    report_lines.append("-" * 40)
    
    for expert_idx in range(4):
        expert_row = air_heatmap[expert_idx]
        peak_hour = np.argmax(expert_row)
        peak_value = np.max(expert_row)
        avg_activation = np.mean(expert_row)
        
        report_lines.append(f"Expert {expert_idx + 1}:")
        report_lines.append(f"  峰值激活时间: {peak_hour}:00")
        report_lines.append(f"  峰值激活概率: {peak_value:.4f}")
        report_lines.append(f"  平均激活概率: {avg_activation:.4f}")
        
        # 找出激活概率较高的时间段
        high_activation_hours = [h for h in range(24) if expert_row[h] > avg_activation * 1.2]
        if high_activation_hours:
            report_lines.append(f"  高激活时间段: {high_activation_hours}")
        report_lines.append("")
    
    # 比较两种数据的专家分工模式
    report_lines.append("专家分工模式对比分析:")
    report_lines.append("-" * 40)
    
    # 计算专家分工的相关性
    correlations = []
    for i in range(4):
        corr = np.corrcoef(wind_heatmap[i], air_heatmap[i])[0, 1]
        correlations.append(corr)
        report_lines.append(f"Expert {i + 1} 相关性: {corr:.4f}")
    
    avg_correlation = np.mean(correlations)
    report_lines.append(f"平均相关性: {avg_correlation:.4f}")
    
    if avg_correlation > 0.7:
        report_lines.append("结论: 专家在两种数据上表现出相似的分工模式")
    elif avg_correlation > 0.4:
        report_lines.append("结论: 专家分工模式存在一定相似性")
    else:
        report_lines.append("结论: 专家分工模式差异较大")
    
    # 保存报告
    report_file = 'expert_specialization_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"[成功] 专家分工分析报告已保存: {report_file}")
    
    # 打印报告摘要
    print("\n" + "=" * 60)
    print("专家分工分析报告摘要")
    print("=" * 60)
    for line in report_lines[:20]:  # 只打印前20行作为摘要
        print(line)

def main():
    """主函数：分析风电和空气质量数据的专家分工"""
    print("=" * 70)
    print("[分析] 专家分工模式分析工具 (增强版)")
    print("=" * 70)
    print("分析目标：")
    print("  1. 风电数据的专家分工模式")
    print("  2. 空气质量数据的专家分工模式")
    print("  3. 两种数据模式的对比分析")
    print("")
    
    # 检查必要文件
    ckpt_path = 'checkpoints_ts/main/best_model.pth'
    if not os.path.exists(ckpt_path):
        print(f"[错误] 找不到模型权重文件: {ckpt_path}")
        print("   请先运行主模型训练或确认文件路径")
        return
    
    print(f"[成功] 找到模型权重: {ckpt_path}")
    
    # 分析风电数据
    wind_heatmap = analyze_wind_data()
    
    # 分析空气质量数据
    air_heatmap = analyze_air_quality_data()
    
    # 生成对比图和分析报告
    if wind_heatmap is not None and air_heatmap is not None:
        plot_comparison_heatmaps(wind_heatmap, air_heatmap)
        print("\n[完成] 分析完成！")
        print("生成的文件:")
        print("  - wind_expert_activation.csv (风电专家激活数据)")
        print("  - air_quality_expert_activation.csv (空气质量专家激活数据)")
        print("  - fig_expert_comparison_*.png (专家分工对比图)")
        print("  - fig_expert_comparison_latest.png (最新对比图)")
        print("  - expert_specialization_report.txt (分析报告)")
    elif wind_heatmap is not None:
        print("[警告] 只完成了风电数据分析，空气质量数据分析失败")
    elif air_heatmap is not None:
        print("[警告] 只完成了空气质量数据分析，风电数据分析失败")
    else:
        print("[错误] 两种数据分析都失败，请检查数据和模型")

if __name__ == '__main__':
    main()