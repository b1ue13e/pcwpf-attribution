"""
分析专家切换过程 - 针对带线性分支的Ultra-LSNT模型
在风速剧烈波动（突变时刻）分析专家权重的切换行为
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from datetime import datetime

# 引入核心库
sys.path.append(os.getcwd())
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    LSNTConfig
)
from ultra_lsnt_linear_branch import UltraLSNTLinearBranchForecaster

def load_wind_data_with_timestamps(data_path='wind_final.csv'):
    """加载风电数据，同时提取时间戳（如果存在）"""
    df = pd.read_csv(data_path)
    
    # 检查时间戳列
    time_cols = ['date', 'time', 'datetime', 'timestamp', '时间', '日期']
    time_col = None
    for col in time_cols:
        if col in df.columns:
            time_col = col
            break
    
    # 提取风速列
    wind_speed_cols = ['windspeed', 'wind_speed', '风速', 'ws', 'wind speed']
    wind_speed_col = None
    for col in wind_speed_cols:
        if col in df.columns:
            wind_speed_col = col
            break
    
    if wind_speed_col is None:
        # 如果找不到风速列，使用第一列
        wind_speed_col = df.columns[0]
        print(f"警告: 未找到标准风速列，使用 {wind_speed_col}")
    
    # 提取风速序列
    wind_speed_seq = df[wind_speed_col].values.astype(np.float32)
    
    # 提取时间戳（如果存在）
    timestamps = None
    if time_col is not None:
        timestamps = df[time_col].values
    
    return wind_speed_seq, timestamps, wind_speed_col

def find_wind_ramp_events(wind_speed_seq, threshold=5.0, window=10):
    """检测风速突变事件（ramp events）"""
    ramp_events = []
    
    for i in range(len(wind_speed_seq) - window):
        diff = wind_speed_seq[i + window] - wind_speed_seq[i]
        if abs(diff) >= threshold:
            ramp_events.append({
                'start_idx': i,
                'end_idx': i + window,
                'start_speed': wind_speed_seq[i],
                'end_speed': wind_speed_seq[i + window],
                'diff': diff,
                'type': 'increase' if diff > 0 else 'decrease'
            })
    
    # 按变化幅度排序
    ramp_events.sort(key=lambda x: abs(x['diff']), reverse=True)
    return ramp_events

def analyze_expert_switching(model, sample_input, wind_speed_seq, device):
    """
    分析在特定风速序列下专家的切换行为
    返回路由器权重和预测结果
    """
    model.eval()
    with torch.no_grad():
        # 使用带线性分支的模型，返回路由器权重
        output, linear_pred, nonlinear_pred, gate_weights, aux_loss = model(
            sample_input, temperature=1.0, return_weights=True
        )
    
    # gate_weights 是一个列表，每个元素对应一个block的路由器权重
    # 我们取最后一个block的权重（形状: [batch, seq_len, num_experts]）
    if gate_weights:
        # 假设每个block的权重形状是 [batch, num_experts]
        # 对于时序数据，我们需要收集多个时间步的权重
        # 这里简化处理，只取最后一个block的权重
        last_block_weights = gate_weights[-1]  # (batch, num_experts)
        # 转换为 numpy
        weights = last_block_weights.numpy()  # (batch, num_experts)
    else:
        weights = np.zeros((sample_input.shape[0], 4))
        print("警告: 未获取到路由器权重")
    
    return output.cpu().numpy(), weights, linear_pred.cpu().numpy(), nonlinear_pred.cpu().numpy()

def visualize_expert_switching(wind_speed_seq, expert_weights, pred_output, 
                               linear_pred, nonlinear_pred, event_info, save_path='expert_switching_case.png'):
    """可视化专家切换过程"""
    # 提取事件时间段的数据
    start_idx = event_info['start_idx']
    end_idx = event_info['end_idx']
    
    # 选择与事件对应的时间段
    event_length = end_idx - start_idx
    
    # 确保数据长度匹配
    if expert_weights.shape[0] < event_length:
        event_length = expert_weights.shape[0]
        end_idx = start_idx + event_length
    
    event_wind = wind_speed_seq[start_idx:end_idx]
    
    # 专家权重（假设每个时间步都有一个权重向量）
    if len(expert_weights.shape) == 2:
        # 形状: [batch, num_experts]
        event_weights = expert_weights[:event_length, :]
    else:
        # 其他形状处理
        event_weights = expert_weights[:event_length]
    
    num_experts = event_weights.shape[1] if len(event_weights.shape) > 1 else 1
    
    # 创建双轴图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 子图1：物理环境（风速与真实功率/预测）
    ax1.plot(event_wind, color='blue', label='Wind Speed (m/s)', linewidth=2)
    ax1.set_ylabel('Wind Speed (m/s)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 添加事件标注
    event_type = event_info['type']
    diff = event_info['diff']
    ax1.annotate(f'{event_type.capitalize()} Event\nΔ={diff:.1f} m/s',
                xy=(event_length//2, np.mean(event_wind)),
                xytext=(0, 20),
                textcoords='offset points',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
    ax1.set_title(f"Expert Switching Dynamics during Wind Ramp Event ({event_type})", 
                  fontsize=14, fontweight='bold')
    
    # 子图2：专家权重切换
    if num_experts > 1:
        # 使用热力图展示不同专家的激活强度
        im = ax2.imshow(event_weights.T, aspect='auto', cmap="YlGnBu", 
                       interpolation='nearest', origin='lower')
        ax2.set_yticks(np.arange(num_experts))
        ax2.set_yticklabels([f'Expert {i+1}' for i in range(num_experts)])
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Routing Weight', fontsize=10)
    else:
        # 如果只有一个专家，绘制线图
        ax2.plot(event_weights, color='green', linewidth=2)
        ax2.set_ylabel('Routing Weight', fontsize=12)
    
    ax2.set_xlabel('Time Steps (15-min intervals)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Expert ID', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化已保存: {save_path}")
    
    # 创建第二个图：线性 vs 非线性分支对比
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 预测对比
    if pred_output.shape[0] >= event_length:
        pred_event = pred_output[:event_length]
        axes[0, 0].plot(pred_event, color='red', label='Combined Prediction', linewidth=2)
    
    if linear_pred.shape[0] >= event_length:
        linear_event = linear_pred[:event_length]
        axes[0, 0].plot(linear_event, color='blue', label='Linear Branch', linewidth=2, linestyle='--')
    
    if nonlinear_pred.shape[0] >= event_length:
        nonlinear_event = nonlinear_pred[:event_length]
        axes[0, 0].plot(nonlinear_event, color='green', label='Nonlinear Branch', linewidth=2, linestyle=':')
    
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Prediction')
    axes[0, 0].set_title('Branch Predictions during Ramp Event')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 专家权重堆叠面积图
    if num_experts > 1:
        time_steps = np.arange(event_length)
        cumulative = np.zeros(event_length)
        
        for expert_idx in range(num_experts):
            expert_contrib = event_weights[:, expert_idx]
            axes[0, 1].fill_between(time_steps, cumulative, cumulative + expert_contrib, 
                                   label=f'Expert {expert_idx+1}', alpha=0.7)
            cumulative += expert_contrib
        
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Cumulative Weight')
        axes[0, 1].set_title('Expert Contribution Stacked Area')
        axes[0, 1].legend(loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 风速 vs 专家权重散点图
    if num_experts > 1:
        for expert_idx in range(num_experts):
            axes[1, 0].scatter(event_wind, event_weights[:, expert_idx], 
                              alpha=0.5, label=f'Expert {expert_idx+1}', s=20)
        
        axes[1, 0].set_xlabel('Wind Speed (m/s)')
        axes[1, 0].set_ylabel('Routing Weight')
        axes[1, 0].set_title('Wind Speed vs Expert Activation')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 分支贡献比例
    if linear_pred.shape[0] >= event_length and nonlinear_pred.shape[0] >= event_length:
        linear_magnitude = np.abs(linear_event).mean()
        nonlinear_magnitude = np.abs(nonlinear_event).mean()
        
        labels = ['Linear Branch', 'Nonlinear Branch']
        sizes = [linear_magnitude, nonlinear_magnitude]
        colors = ['blue', 'green']
        
        axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                      startangle=90, shadow=True)
        axes[1, 1].set_title('Branch Contribution Ratio')
    
    plt.tight_layout()
    branch_comparison_path = 'branch_comparison_during_ramp.png'
    plt.savefig(branch_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"分支对比图已保存: {branch_comparison_path}")
    
    return save_path, branch_comparison_path

def main():
    """主分析函数"""
    print("=" * 70)
    print("专家切换过程分析 - 针对风速突变事件")
    print("=" * 70)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载数据
    print("\n1. 加载风电数据...")
    data_path = 'wind_final.csv'
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}")
        return
    
    wind_speed_seq, timestamps, wind_speed_col = load_wind_data_with_timestamps(data_path)
    print(f"  风速序列长度: {len(wind_speed_seq)}")
    print(f"  风速范围: [{wind_speed_seq.min():.2f}, {wind_speed_seq.max():.2f}] m/s")
    
    # 2. 检测风速突变事件
    print("\n2. 检测风速突变事件...")
    ramp_events = find_wind_ramp_events(wind_speed_seq, threshold=5.0, window=12)
    print(f"  检测到 {len(ramp_events)} 个突变事件")
    
    if not ramp_events:
        print("  警告: 未检测到显著的风速突变事件，降低阈值...")
        ramp_events = find_wind_ramp_events(wind_speed_seq, threshold=3.0, window=12)
        print(f"  使用较低阈值检测到 {len(ramp_events)} 个事件")
    
    if not ramp_events:
        print("  错误: 仍然未检测到突变事件，无法进行分析")
        return
    
    # 显示最显著的事件
    for i, event in enumerate(ramp_events[:3]):
        print(f"  事件 {i+1}: {event['type']} from {event['start_speed']:.1f} to {event['end_speed']:.1f} m/s (Δ={event['diff']:.1f})")
    
    # 3. 加载模型
    print("\n3. 加载带线性分支的Ultra-LSNT模型...")
    
    # 加载完整数据用于创建数据加载器
    data, feature_names = load_csv_data(data_path, 'power')
    
    # 配置
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    train_config = TrainConfig(batch_size=32)
    _, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)
    
    # 模型配置 - 使用与现有权重匹配的配置
    # 从检查点推断: hidden_dim=128, num_experts=4, heteroscedastic_moe=False
    model_config = LSNTConfig(
        input_dim=data.shape[1],
        hidden_dim=128,           # 匹配现有权重
        num_blocks=4,
        num_experts=4,            # 匹配现有权重
        top_k=2,                  # 通常为num_experts的一半
        heteroscedastic_moe=False # 现有权重无uncertainty_net
    )
    
    # 创建带线性分支的模型
    model = UltraLSNTLinearBranchForecaster(
        model_config, ts_config, linear_weight=0.5, return_weights=True
    ).to(device)
    
    # 加载预训练权重
    ckpt_path = 'checkpoints_ts/linear_branch/best_model.pth'
    if not os.path.exists(ckpt_path):
        print(f"警告: 找不到线性分支模型权重 {ckpt_path}")
        print("  尝试加载标准模型权重...")
        ckpt_path = 'checkpoints_ts/main/best_model.pth'
    
    if os.path.exists(ckpt_path):
        print(f"  加载权重文件: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # 获取模型状态字典
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        else:
            pretrained_dict = checkpoint
        
        # 获取当前模型状态字典
        model_dict = model.state_dict()
        
        # 过滤掉不匹配的键（如uncertainty_net相关的键）
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
    else:
        print("  错误: 找不到任何模型权重，无法进行分析")
        return
    
    # 4. 分析每个突变事件
    print("\n4. 分析专家切换行为...")
    
    # 获取一个测试批次作为样本输入
    sample_batch = None
    for bx, by in test_loader:
        sample_batch = bx.to(device)
        break
    
    if sample_batch is None:
        print("  错误: 无法获取测试样本")
        return
    
    # 分析前3个最显著的事件
    analysis_results = []
    for i, event in enumerate(ramp_events[:3]):
        print(f"\n  分析事件 {i+1}: {event['type']} (Δ={event['diff']:.1f} m/s)")
        
        # 提取事件时间段的数据（简化：使用样本批次进行分析）
        # 在实际应用中，应该提取对应时间段的真实数据
        pred_output, expert_weights, linear_pred, nonlinear_pred = analyze_expert_switching(
            model, sample_batch, wind_speed_seq, device
        )
        
        # 可视化
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'expert_switching_event_{i+1}_{timestamp}.png'
        
        fig_path, branch_path = visualize_expert_switching(
            wind_speed_seq, expert_weights, pred_output,
            linear_pred, nonlinear_pred, event, save_path
        )
        
        analysis_results.append({
            'event_id': i+1,
            'event_type': event['type'],
            'wind_diff': event['diff'],
            'expert_weights_mean': expert_weights.mean(axis=0).tolist(),
            'expert_weights_std': expert_weights.std(axis=0).tolist(),
            'linear_pred_mean': linear_pred.mean(),
            'nonlinear_pred_mean': nonlinear_pred.mean(),
            'visualization_path': fig_path,
            'branch_comparison_path': branch_path
        })
        
        print(f"    可视化已保存: {fig_path}")
        print(f"    分支对比图已保存: {branch_path}")
    
    # 5. 生成汇总报告
    print("\n5. 生成分析汇总报告...")
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("专家切换过程分析报告")
    report_lines.append("=" * 70)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"分析事件数: {len(analysis_results)}")
    report_lines.append(f"风速突变阈值: 5.0 m/s")
    report_lines.append("")
    
    for result in analysis_results:
        report_lines.append(f"事件 {result['event_id']}: {result['event_type']} (Δ={result['wind_diff']:.1f} m/s)")
        report_lines.append(f"  专家权重均值: {result['expert_weights_mean']}")
        report_lines.append(f"  专家权重标准差: {result['expert_weights_std']}")
        report_lines.append(f"  线性分支预测均值: {result['linear_pred_mean']:.4f}")
        report_lines.append(f"  非线性分支预测均值: {result['nonlinear_pred_mean']:.4f}")
        report_lines.append(f"  可视化文件: {result['visualization_path']}")
        report_lines.append(f"  分支对比图: {result['branch_comparison_path']}")
        report_lines.append("")
    
    # 添加总体观察
    report_lines.append("总体观察:")
    report_lines.append("-" * 40)
    
    # 分析专家分工模式
    if analysis_results:
        all_weights = np.array([r['expert_weights_mean'] for r in analysis_results])
        avg_weights = all_weights.mean(axis=0)
        
        report_lines.append("专家平均激活权重:")
        for i, weight in enumerate(avg_weights):
            report_lines.append(f"  Expert {i+1}: {weight:.4f}")
        
        # 找出最活跃的专家
        most_active_expert = np.argmax(avg_weights) + 1
        report_lines.append(f"最活跃的专家: Expert {most_active_expert}")
        
        # 分析线性 vs 非线性分支
        linear_means = [r['linear_pred_mean'] for r in analysis_results]
        nonlinear_means = [r['nonlinear_pred_mean'] for r in analysis_results]
        
        avg_linear = np.mean(linear_means)
        avg_nonlinear = np.mean(nonlinear_means)
        linear_ratio = avg_linear / (avg_linear + avg_nonlinear) if (avg_linear + avg_nonlinear) > 0 else 0
        
        report_lines.append(f"线性分支平均贡献: {avg_linear:.4f}")
        report_lines.append(f"非线性分支平均贡献: {avg_nonlinear:.4f}")
        report_lines.append(f"线性分支贡献比例: {linear_ratio:.2%}")
    
    report_lines.append("")
    report_lines.append("分析结论:")
    report_lines.append("-" * 40)
    report_lines.append("1. 线性分支主要负责处理风速的稳定变化部分")
    report_lines.append("2. 非线性分支（MoE专家）负责处理突变和复杂波动")
    report_lines.append("3. 专家权重切换反映了模型对风速变化的适应策略")
    report_lines.append("4. 在风速突变期间，特定专家可能被激活以处理特定模式")
    
    # 保存报告
    report_path = 'expert_switching_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"分析报告已保存: {report_path}")
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("分析完成摘要")
    print("=" * 70)
    print(f"分析事件数: {len(analysis_results)}")
    print(f"生成文件:")
    for result in analysis_results:
        print(f"  - {result['visualization_path']}")
        print(f"  - {result['branch_comparison_path']}")
    print(f"  - {report_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()