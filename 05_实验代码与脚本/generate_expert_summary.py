"""
生成专家分析汇总报告（增强版）
包含专家切换频率检查和关键指标分析
"""
import os
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime

def generate_expert_summary():
    print("生成专家分析汇总报告（增强版）...")
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("Ultra-LSNT 专家分析汇总报告（增强版）")
    report_lines.append("=" * 70)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_lines.append(f"生成时间: {timestamp}")
    report_lines.append("")
    
    # 1. 检查生成的文件
    report_lines.append("1. 生成文件清单")
    report_lines.append("-" * 40)
    
    expert_files = [
        ("expert_switching_analysis_report*.txt", "专家切换分析报告"),
        ("expert_switching_event_*.png", "专家切换可视化图"),
        ("wind_expert_activation*.csv", "风电专家激活数据"),
        ("expert_analysis_summary.txt", "专家分析汇总报告"),
    ]
    
    for pattern, description in expert_files:
        files = glob.glob(pattern)
        if files:
            report_lines.append(f"✓ {description}:")
            for f in files[:3]:  # 最多显示3个文件
                report_lines.append(f"    - {os.path.basename(f)}")
            if len(files) > 3:
                report_lines.append(f"    ... 共 {len(files)} 个文件")
        else:
            report_lines.append(f"✗ {description}: 未找到")
    
    # 2. 检查模型权重
    report_lines.append("\n2. 模型权重状态")
    report_lines.append("-" * 40)
    
    weight_files = [
        "checkpoints_ts/linear_branch/best_model.pth",
        "checkpoints_ts/main/best_model.pth",
    ]
    
    for weight_file in weight_files:
        if os.path.exists(weight_file):
            file_size = os.path.getsize(weight_file) / (1024*1024)  # MB
            report_lines.append(f"✓ {weight_file}: {file_size:.1f} MB")
        else:
            report_lines.append(f"✗ {weight_file}: 不存在")
    
    # 3. 关键分析结果和指标
    report_lines.append("\n3. 关键分析结果和指标")
    report_lines.append("-" * 40)
    
    # 3.1 尝试读取专家切换分析报告
    report_files = glob.glob("expert_switching_analysis_report*.txt")
    if report_files:
        try:
            with open(report_files[0], "r", encoding='utf-8') as f:
                lines = f.readlines()
                # 提取关键信息
                for line in lines:
                    if "最活跃的专家" in line or "风速突变阈值" in line or "分析事件数" in line or "结论:" in line:
                        report_lines.append(f"  {line.strip()}")
        except:
            report_lines.append("  无法读取专家切换分析报告")
    else:
        report_lines.append("  未找到专家切换分析报告")
    
    # 3.2 分析专家切换频率（如果有时序数据）
    report_lines.append("\n  专家切换频率分析:")
    try:
        # 查找专家激活数据文件
        activation_files = glob.glob("wind_expert_activation*.csv")
        if activation_files:
            df = pd.read_csv(activation_files[0])
            print(f"  读取激活数据文件: {activation_files[0]}, 形状: {df.shape}")
            
            # 检查是否有时间序列数据（包含top_expert列）
            if 'top_expert' in df.columns:
                # 计算专家切换率：即 Top-1 专家发生改变的频率
                switches = (df['top_expert'].shift(1) != df['top_expert']).sum()
                total_steps = len(df) - 1  # 排除第一个
                switch_rate = switches / total_steps if total_steps > 0 else 0
                
                report_lines.append(f"    关键指标: 专家切换总次数: {switches} (反映了模型对风速波动的敏感度)")
                report_lines.append(f"    切换频率: {switch_rate:.2%} ({switches}/{total_steps} 个时间步发生切换)")
                
                # 分析专家主导情况
                expert_counts = df['top_expert'].value_counts().sort_index()
                report_lines.append(f"    各专家主导时间步统计:")
                for expert, count in expert_counts.items():
                    proportion = count / len(df)
                    report_lines.append(f"      Expert {expert}: {count} 步 ({proportion:.1%})")
            else:
                # 事件级别的数据（我们生成的格式）
                # 检查是否有专家权重列
                expert_cols = [col for col in df.columns if col.startswith('Expert_') or col.startswith('expert_')]
                if len(expert_cols) >= 2:  # 至少有2个专家列
                    report_lines.append(f"    检测到 {len(expert_cols)} 个专家的激活权重数据")
                    
                    # 计算每个事件中最活跃的专家
                    if len(df) > 0:
                        # 对于每一行（事件），找出权重最高的专家
                        expert_arrays = []
                        for col in expert_cols:
                            expert_arrays.append(df[col].values)
                        
                        # 转换为numpy数组
                        weights_matrix = np.column_stack(expert_arrays)  # (n_events, n_experts)
                        top_experts = np.argmax(weights_matrix, axis=1) + 1  # 1-based
                        
                        # 统计专家主导事件数
                        expert_event_counts = {}
                        for expert in top_experts:
                            expert_event_counts[expert] = expert_event_counts.get(expert, 0) + 1
                        
                        report_lines.append(f"    专家主导事件统计:")
                        for expert, count in sorted(expert_event_counts.items()):
                            proportion = count / len(df)
                            report_lines.append(f"      Expert {expert}: {count} 个事件 ({proportion:.1%})")
                        
                        # 计算事件间专家切换次数
                        if len(top_experts) > 1:
                            event_switches = (top_experts[:-1] != top_experts[1:]).sum()
                            event_switch_rate = event_switches / (len(top_experts) - 1)
                            report_lines.append(f"    事件间专家切换次数: {event_switches} (切换率: {event_switch_rate:.2%})")
                    
                    # 分析专家权重分布
                    report_lines.append(f"    专家权重统计:")
                    for i, col in enumerate(expert_cols, 1):
                        mean_weight = df[col].mean()
                        std_weight = df[col].std()
                        report_lines.append(f"      {col}: 均值={mean_weight:.4f}, 标准差={std_weight:.4f}")
                else:
                    report_lines.append("    警告: 数据格式不包含专家权重列，无法计算切换频率")
        else:
            report_lines.append("    未找到专家激活数据文件")
    except Exception as e:
        report_lines.append(f"    专家切换频率分析错误: {str(e)}")
    
    # 4. 建议和下一步工作
    report_lines.append("\n4. 建议和下一步工作")
    report_lines.append("-" * 40)
    report_lines.append("  • 检查专家切换可视化图，分析风速突变时的专家响应")
    report_lines.append("  • 分析专家分工模式：不同专家是否处理不同的风速模式（如稳定风、突变风）")
    report_lines.append("  • 验证专家激活权重与物理变量的相关性（如风速、湍流强度）")
    report_lines.append("  • 考虑调整线性分支权重以优化模型性能")
    report_lines.append("  • 可以尝试不同的突变检测阈值以获得更多分析案例")
    report_lines.append("  • 建议生成时间序列级别的专家激活数据以进行更精细的切换频率分析")
    
    # 5. 技术备注
    report_lines.append("\n5. 技术备注")
    report_lines.append("-" * 40)
    report_lines.append("  • 当前分析基于标准Ultra-LSNT模型（hidden_dim=128, num_experts=4）")
    report_lines.append("  • 风速突变检测阈值: 5.0 m/s，时间窗口: 12个时间步")
    report_lines.append("  • 专家切换频率反映了模型对输入变化的敏感度和适应能力")
    report_lines.append("  • 高切换频率可能表示模型对不同风速模式有细粒度响应")
    
    # 保存报告
    report_text = "\n".join(report_lines)
    with open("expert_analysis_summary_enhanced.txt", "w", encoding='utf-8') as f:
        f.write(report_text)
    
    # 同时更新原有的expert_analysis_summary.txt
    with open("expert_analysis_summary.txt", "w", encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print("\n专家分析汇总报告已保存到: expert_analysis_summary.txt 和 expert_analysis_summary_enhanced.txt")

if __name__ == '__main__':
    generate_expert_summary()
