import torch
import torch.nn as nn
from ultra_lsnt_timeseries import create_dataloaders, load_csv_data, compute_metrics, TimeSeriesConfig, TrainConfig, \
    LSTMBaseline, MLPBaseline
import numpy as np
import time
import math
import json
import os
from datetime import datetime


# ==========================================
# 🆕 新增：Transformer 基线模型 (SCI 一区质检员)
# ==========================================
class TransformerBaseline(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, d_model)
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # Transformer 编码层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.output_proj = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.input_proj(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        # 取最后一个时间步的特征进行预测
        x = x[:, -1, :]
        out = self.output_proj(x)
        return out, torch.zeros(1)  # Transformer 没有 Auxiliary Loss


# ==========================================
# 训练主逻辑 (增强版：保存结果)
# ==========================================
def train_baseline(model_name, data_path, target, device):
    print(f"\n🥊 正在训练强力基线: {model_name} ...")

    # 创建结果保存目录
    results_dir = 'baseline_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 加载数据
    try:
        data, _ = load_csv_data(data_path, target)
        print(f"✅ 数据加载成功: {data.shape}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return 0, None, None, None

    input_dim = data.shape[1]

    # 配置
    # 注意：这里我们用 seq_len=168 (和你的 Ultra-LSNT 保持一致，这样才公平)
    ts_config = TimeSeriesConfig(seq_len=168, pred_len=24, target=target)
    # Transformer 需要更多的耐心和更小的学习率
    train_config = TrainConfig(batch_size=64, epochs=30, lr=0.0005)

    train_loader, val_loader, test_loader, scaler = create_dataloaders(data, ts_config, train_config)

    # 模型选择
    if model_name == 'LSTM':
        model = LSTMBaseline(input_dim, 128, ts_config.pred_len).to(device)
    elif model_name == 'MLP':
        model = MLPBaseline(input_dim, ts_config.seq_len, 128, ts_config.pred_len).to(device)
    elif model_name == 'Transformer':
        model = TransformerBaseline(input_dim, ts_config.seq_len, ts_config.pred_len).to(device)
    else:
        print(f"❌ 未知模型: {model_name}")
        return 0, None, None, None

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    criterion = nn.MSELoss()

    # 训练循环
    model.train()
    start_time = time.time()
    training_history = []
    
    for epoch in range(train_config.epochs):
        epoch_loss = 0
        steps = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out, _ = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        
        # 记录训练历史
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'time': time.time() - start_time
        })
        
        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"   ⏳ Epoch {epoch + 1}/{train_config.epochs} | Loss: {avg_loss:.4f} | 耗时: {elapsed:.1f}s")

    # 测试
    print(f"   🚀 开始测试 {model_name}...")
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out, _ = model(bx)
            preds.append(out.cpu().numpy())
            trues.append(by.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    target_mean = scaler.mean[-1]
    target_std = scaler.std[-1]
    preds_original = preds * target_std + target_mean
    trues_original = trues * target_std + target_mean

    metrics = compute_metrics(preds_original, trues_original)
    print(f"✅ {model_name} 最终 R2: {metrics['R2']:.4f}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(results_dir, f"{model_name}_results_{timestamp}.json")
    
    result_data = {
        'model_name': model_name,
        'data_path': data_path,
        'target': target,
        'input_dim': input_dim,
        'seq_len': ts_config.seq_len,
        'pred_len': ts_config.pred_len,
        'batch_size': train_config.batch_size,
        'epochs': train_config.epochs,
        'learning_rate': train_config.lr,
        'metrics': metrics,
        'training_history': training_history,
        'predictions_shape': preds.shape,
        'timestamp': timestamp,
        'device': str(device)
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2, default=str)
    
    print(f"💾 结果已保存: {result_file}")
    
    # 保存预测值和真实值（前1000个样本）
    sample_file = os.path.join(results_dir, f"{model_name}_samples_{timestamp}.npz")
    np.savez_compressed(
        sample_file,
        predictions=preds_original[:1000],
        ground_truth=trues_original[:1000]
    )
    print(f"💾 样本数据已保存: {sample_file}")
    
    return metrics['R2'], preds_original, trues_original, result_file


# ==========================================
# 主函数：运行所有基线模型
# ==========================================
def run_all_baselines():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results_summary = []
    
    # 运行所有基线模型
    models_to_run = ['LSTM', 'MLP', 'Transformer']
    
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"🚀 开始训练 {model_name} 模型")
        print(f"{'='*60}")
        
        r2_score, preds, trues, result_file = train_baseline(
            model_name,
            'gefcom_ready.csv',
            'load',
            device
        )
        
        if r2_score > 0:
            results_summary.append({
                'model': model_name,
                'r2_score': r2_score,
                'result_file': result_file
            })
    
    # 打印汇总结果
    print(f"\n{'='*60}")
    print("🏆 【SCI 一区 终极排位赛】")
    print(f"{'='*60}")
    
    for result in results_summary:
        print(f"{result['model']:15s} : {result['r2_score']:.4f}")
    
    print("-" * 40)
    print(f"Ultra-LSNT (Ours)  : 0.7513 (参考值)")
    print(f"{'='*60}")
    
    # 保存汇总结果
    summary_file = 'baseline_results/baseline_summary.json'
    summary_data = {
        'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'models': results_summary,
        'ultra_lsnt_reference': 0.7513
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n📊 汇总结果已保存: {summary_file}")
    
    # 生成比较报告
    generate_comparison_report(results_summary)
    
    return results_summary


# ==========================================
# 生成比较报告
# ==========================================
def generate_comparison_report(results_summary):
    """生成基线模型比较报告"""
    report_lines = []
    report_lines.append("基线模型性能比较报告")
    report_lines.append("=" * 60)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 按R2分数排序
    sorted_results = sorted(results_summary, key=lambda x: x['r2_score'], reverse=True)
    
    report_lines.append("性能排名:")
    report_lines.append("-" * 40)
    for i, result in enumerate(sorted_results, 1):
        report_lines.append(f"{i}. {result['model']:15s} : R² = {result['r2_score']:.4f}")
    
    report_lines.append("")
    report_lines.append("与Ultra-LSNT的比较:")
    report_lines.append("-" * 40)
    ultra_lsnt_score = 0.7513
    
    for result in sorted_results:
        diff = result['r2_score'] - ultra_lsnt_score
        if diff > 0:
            report_lines.append(f"{result['model']}: 优于Ultra-LSNT +{diff:.4f}")
        else:
            report_lines.append(f"{result['model']}: 劣于Ultra-LSNT {diff:.4f}")
    
    report_lines.append("")
    report_lines.append("结论:")
    report_lines.append("-" * 40)
    best_model = sorted_results[0]
    if best_model['r2_score'] > ultra_lsnt_score:
        report_lines.append(f"{best_model['model']} 表现最佳，超越了Ultra-LSNT")
    else:
        report_lines.append("Ultra-LSNT 在所有基线模型中保持最佳性能")
    
    # 保存报告
    report_file = 'baseline_results/baseline_comparison_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📝 比较报告已生成: {report_file}")
    
    # 打印报告摘要
    print("\n📋 报告摘要:")
    print("-" * 40)
    for line in report_lines[:15]:
        print(line)


# ==========================================
# 主程序入口
# ==========================================
if __name__ == '__main__':
    print("🚀 启动基线模型训练系统 (增强版)")
    print("✨ 特性: 自动保存结果、生成报告、性能对比")
    print("-" * 60)
    
    # 检查必要文件
    if not os.path.exists('gefcom_ready.csv'):
        print("❌ 找不到 gefcom_ready.csv 文件")
        print("   请确保数据文件存在或使用正确的路径")
        exit(1)
    
    # 运行所有基线模型
    results = run_all_baselines()
    
    print("\n🎉 基线模型训练完成!")
    print("📁 所有结果保存在 'baseline_results/' 目录中")
    print("📊 查看 summary.json 获取详细结果")
    print("📝 查看 comparison_report.txt 获取分析报告")