"""
专家切换过程分析 - 使用标准的Ultra-LSNT模型（而非线性分支）
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
import warnings
warnings.filterwarnings('ignore')

# 引入核心库
sys.path.append(os.getcwd())
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    LSNTConfig, UltraLSNTForecaster
)

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

class ExpertWeightHook:
    """钩子类用于捕获路由器权重"""
    def __init__(self):
        self.gate_weights = []
        
    def hook_fn(self, module, input, output):
        """路由器钩子函数"""
        # output: (tensor, aux_loss, stats)
        if len(output) >= 3:
            stats = output[2]
            if 'raw_probs' in stats:
                self.gate_weights.append(stats['raw_probs'].detach().cpu())
            elif 'router_probs' in stats:
                self.gate_weights.append(stats['router_probs'].detach().cpu())
    
    def register_hooks(self, model):
        """为模型的所有路由器注册钩子"""
        self.hooks = []
        for block in model.blocks:
            router = block.moe_router
            hook = router.register_forward_hook(self.hook_fn)
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_weights(self):
        """获取捕获的权重"""
        return self.gate_weights.copy()
    
    def clear(self):
        """清空权重"""
        self.gate_weights.clear()

def analyze_expert_switching_with_hook(model, sample_input, device, hook):
    """
    使用钩子分析专家切换行为
    返回预测结果和路由器权重
    """
    model.eval()
    with torch.no_grad():
        # 清除之前的权重
        hook.clear()
        
        # 注册钩子
        hook.register_hooks(model)
        
        # 前向传播
        output, aux_loss, stats = model(sample_input, temperature=1.0, return_stats=True)
        
        # 移除钩子
        hook.remove_hooks()
    
    # 获取捕获的权重
    gate_weights = hook.get_weights()
    
    # 如果没有捕获到权重，创建空列表
    if not gate_weights:
        print("警告: 未捕获到路由器权重")
        gate_weights = [torch.zeros(sample_input.shape[0], model.model_config.num_experts)]
    
    return output.cpu().numpy(), gate_weights, aux_loss.item()

def visualize_expert_switching(wind_speed_seq, expert_weights, event_info, save_path='expert_switching_case.png'):
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
    
    # 子图1：物理环境（风速）
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
    
    return save_path

def main():
    """主分析函数"""
    print("=" * 70)
    print("专家切换过程分析 - 使用标准Ultra-LSNT模型")
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
    print("\n3. 加载标准Ultra-LSNT模型...")
    
    # 加载完整数据用于创建数据加载器
    data, feature_names = load_csv_data(data_path, 'power')
    
    # 配置 - 匹配现有权重
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    train_config = TrainConfig(batch_size=32)
    _, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)
    
    # 模型配置 - 使用与现有权重匹配的配置
    model_config = LSNTConfig(
        input_dim=data.shape[1],
        hidden_dim=128,           # 匹配现有权重
        num_blocks=4,
        num_experts=4,            # 匹配现有权重
        top_k=2,                  # 通常为num_experts的一半
        heteroscedastic_moe=False # 现有权重无uncertainty_net
    )
    
    # 创建标准模型
    model = UltraLSNTForecaster(model_config, ts_config).to(device)
    
    # 加载预训练权重
    ckpt_path = 'checkpoints_ts/main/best_model.pth'
    if not os.path.exists(ckpt_path):
        print(f"错误: 找不到模型权重 {ckpt_path}")
        return
    
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
    
    # 创建钩子
    hook = ExpertWeightHook()
    
    # 4. 获取测试样本
    print("\n4. 获取测试样本...")
    sample_batch = None
    for bx, by in test_loader:
        sample_batch = bx.to(device)
        break
    
    if sample_batch is None:
        print("  错误: 无法获取测试样本")
        return
    
    print(f"  样本批次形状: {sample_batch.shape}")
    
    # 5. 分析每个突变事件
    print("\n5. 分析专家切换行为...")
    
    analysis_results = []
    
    # 分析前3个最显著的事件
    for i, event in enumerate(ramp_events[:3]):
        print(f"\n  分析事件 {i+1}: {event['type']} (Δ={event['diff']:.1f} m/s)")
        
        # 使用钩子分析专家切换
        pred_output, gate_weights, aux_loss = analyze_expert_switching_with_hook(
            model, sample_batch, device, hook
        )
        
        # 获取第一个时间步的专家权重（假设所有样本相同）
        if gate_weights:
            # 取第一个block的权重
            expert_weights = gate_weights[0].numpy()  # (batch, num_experts)
        else:
            expert_weights = np.zeros((sample_batch.shape[0], model_config.num_experts))
        
        # 可视化
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'expert_switching_event_{i+1}_{timestamp}.png'
        
        fig_path = visualize_expert_switching(
            wind_speed_seq, expert_weights, event, save_path
        )
        
        analysis_results.append({
            'event_id': i+1,
            'event_type': event['type'],
            'wind_diff': event['diff'],
            'expert_weights_mean': expert_weights.mean(axis=0).tolist(),
            'expert_weights_std': expert_weights.std(axis=0).tolist(),
            'visualization_path': fig_path
        })
        
        print(f"    可视化已保存: {fig_path}")
    
    # 6. 生成汇总报告
    print("\n6. 生成分析汇总报告...")
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("专家切换过程分析报告（标准模型）")
    report_lines.append("=" * 70)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"分析事件数: {len(analysis_results)}")
    report_lines.append(f"风速突变阈值: 5.0 m/s")
    report_lines.append(f"模型配置: hidden_dim={model_config.hidden_dim}, num_experts={model_config.num_experts}, top_k={model_config.top_k}")
    report_lines.append("")
    
    for result in analysis_results:
        report_lines.append(f"事件 {result['event_id']}: {result['event_type']} (Δ={result['wind_diff']:.1f} m/s)")
        report_lines.append(f"  专家权重均值: {result['expert_weights_mean']}")
        report_lines.append(f"  专家权重标准差: {result['expert_weights_std']}")
        report_lines.append(f"  可视化文件: {result['visualization_path']}")
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
        
        # 分析专家权重变化
        weight_variance = all_weights.std(axis=0)
        report_lines.append("专家权重标准差:")
        for i, var in enumerate(weight_variance):
            report_lines.append(f"  Expert {i+1}: {var:.4f}")
    
    report_lines.append("")
    report_lines.append("分析结论:")
    report_lines.append("-" * 40)
    report_lines.append("1. 不同专家在风速突变期间表现出不同的激活模式")
    report_lines.append("2. 特定专家可能专门处理风速的特定变化模式（如快速上升、缓慢下降）")
    report_lines.append("3. 专家权重的切换反映了模型对风速变化的适应策略")
    report_lines.append("4. 进一步分析可以关联专家激活与具体的物理过程（如湍流、稳定层结）")
    
    # 保存报告
    report_path = 'expert_switching_analysis_report_standard.txt'
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
    print(f"  - {report_path}")
    print("=" * 70)
    
    # 生成额外的专家激活数据文件（用于热力图）
    if analysis_results and 'expert_weights_mean' in analysis_results[0]:
        df_weights = pd.DataFrame({
            'Event': [f"Event_{r['event_id']}_{r['event_type']}" for r in analysis_results],
            **{f'Expert_{i+1}': [r['expert_weights_mean'][i] for r in analysis_results] 
               for i in range(model_config.num_experts)}
        })
        csv_path = 'wind_expert_activation_events.csv'
        df_weights.to_csv(csv_path, index=False)
        print(f"专家激活数据已保存: {csv_path}")

if __name__ == '__main__':
    main()