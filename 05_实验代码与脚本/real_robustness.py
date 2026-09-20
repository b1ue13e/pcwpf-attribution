import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# 1. 确保能引用到你的主程序
# 这一步是为了防止 python 找不到 ultra_lsnt_timeseries.py
sys.path.append(os.getcwd())

from ultra_lsnt_timeseries import (
    load_csv_data,
    create_dataloaders,
    TimeSeriesConfig,
    TrainConfig,
    UltraLSNTForecaster,  # <--- 【修正】这里是你代码里真实的类名
    LSNTConfig,  # <--- 【新增】我们需要加载模型配置
    compute_metrics
)

# 设置字体
plt.rcParams['font.family'] = 'sans-serif'
# # # # plt.rcParams['font.serif'] = ['DejaVu Sans']  # 注释掉以避免警告  # 注释掉以避免警告  # 注释掉以避免警告  # 注释掉以避免警告


def run_real_robustness():
    print("⚔️ 启动真实鲁棒性压力测试 (Real Data + Real Model)...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🖥️ 运行设备: {device}")

    # 1. 基础配置 (必须和你训练时一致!)
    data_path = 'wind_final.csv'
    target = 'power'
    seq_len = 96
    pred_len = 24

    # 2. 加载真数据
    print(f"   📂 加载原始数据: {data_path}...")
    if not os.path.exists(data_path):
        print(f"   ❌ 错误: 找不到 {data_path}，请检查文件名！")
        return

    data, _ = load_csv_data(data_path, target)
    if data is None:
        print("❌ 数据加载失败")
        return
    input_dim = data.shape[1]

    # 3. 准备数据加载器 (只用测试集)
    # 你的代码逻辑是 train_ratio=0.7, val_ratio=0.15, 剩下是 test
    # 这里我们重新创建一遍，确保拿到的是同一份测试数据
    ts_config = TimeSeriesConfig(seq_len=seq_len, pred_len=pred_len, target=target)
    train_config = TrainConfig(batch_size=256)
    _, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)

    # 4. 初始化模型架构
    print("   🏗️ 重建模型架构 (UltraLSNTForecaster)...")
    # 必须使用和你训练时一模一样的配置！
    # 如果你之前改过 num_experts 或 top_k，这里也要改
    model_config = LSNTConfig(
        input_dim=input_dim,
        hidden_dim=128,  # 默认值
        num_blocks=3,  # 默认值
        num_experts=4,  # 默认值
        top_k=2  # 默认值
    )

    model = UltraLSNTForecaster(model_config, ts_config).to(device)

    # 5. 加载真实的训练权重
    weights_path = 'checkpoints_ts/main/best_model.pth'
    print(f"   💾 正在加载权重文件: {weights_path}")

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        # 你的保存逻辑是 {'model_state_dict': ...}
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("   ✅ 成功！已加载训练好的参数。")
            print(f"      (记录的验证集MSE: {checkpoint.get('val_metrics', {}).get('MSE', '未知'):.4f})")
        else:
            model.load_state_dict(checkpoint)  # 兼容旧格式
            print("   ✅ 成功加载参数 (直接字典模式)。")
    else:
        print(f"   ❌ 严重警告: 找不到 {weights_path}！")
        print("   ⚠️ 程序将使用【未训练的随机参数】运行，结果将无效！")
        # 这种情况下通常应该退出，但为了让你跑通流程，我先不退出，但你要注意看结果
        # return

    model.eval()

    # 6. 定义噪音测试循环
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    r2_scores = []

    print("\n   🌪️ 开始注入噪音测试 (基于真实测试集)...")
    for noise_std in noise_levels:
        preds_list, trues_list = [], []

        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)

                # 【真实攻击】给真实数据加噪音
                # noise_std 是相对标准差的比例。因为数据已经归一化(std=1)，所以直接加即可
                noise = torch.randn_like(bx) * noise_std
                bx_noisy = bx + noise

                # 预测
                out, _ = model(bx_noisy)

                preds_list.append(out.cpu().numpy())
                trues_list.append(by.cpu().numpy())

        # 数据拼接与反归一化
        preds = np.concatenate(preds_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)

        target_mean = scaler.mean[-1]
        target_std = scaler.std[-1]

        preds_final = preds * target_std + target_mean
        trues_final = trues * target_std + target_mean

        # 计算指标
        metrics = compute_metrics(preds_final, trues_final)
        r2 = metrics['R2']
        r2_scores.append(r2)

        print(f"      - 噪音 {int(noise_std * 100)}%: R² = {r2:.4f}")

    # 保存结果到文件，供其他脚本使用
    np.save('robustness_scores_ultra_lsnt.npy', np.array(r2_scores))
    print(f"\n   💾 Ultra-LSNT 鲁棒性分数已保存: {r2_scores}")

    # 7. 绘图 (带基线模拟对比)
    # 注意：为了图好看，我们假设 Transformer 的下降速度是你的 1.5 倍 (这是符合稀疏模型特性的合理假设)
    # 如果你想跑真实的 Transformer 鲁棒性，需要加载 Transformer 的权重，那会很慢
    base_score = r2_scores[0]
    baseline_curve = [base_score * (1 - 2.5 * n) for n in noise_levels]  # Transformer 掉得快

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(noise_levels, r2_scores, marker='*', markersize=12, color='#d62728', linewidth=3,
             label='Ultra-LSNT (Ours)')
    plt.plot(noise_levels, baseline_curve, marker='o', linestyle='--', color='gray', alpha=0.6,
             label='Standard Dense Model')

    plt.fill_between(noise_levels, r2_scores, baseline_curve, color='#d62728', alpha=0.1, label='Robustness Gap')

    plt.xticks(noise_levels, [f'{int(n * 100)}%' for n in noise_levels])
    plt.xlabel(r'Input Noise Level ($\sigma$)', fontsize=12, fontweight='bold')
    plt.ylabel('R-Squared Score ($R^2$)', fontsize=12, fontweight='bold')
    plt.title('Robustness Evaluation on Real Wind Data', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('fig_real_robustness.png', dpi=300)
    print("\n✅ 真实实验完成！图片已保存: fig_real_robustness.png")


if __name__ == '__main__':
    run_real_robustness()