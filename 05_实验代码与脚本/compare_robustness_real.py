import os

# 【必须放在第一行】解决 OMP Error #15
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

# 引入你的主程序
sys.path.append(os.getcwd())
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    UltraLSNTForecaster, LSNTConfig, compute_metrics
)

# 设置字体
plt.rcParams['font.family'] = 'sans-serif'
# # # # plt.rcParams['font.serif'] = ['DejaVu Sans']  # 注释掉以避免警告  # 注释掉以避免警告  # 注释掉以避免警告  # 注释掉以避免警告


# 定义一个标准的 Transformer Baseline
class TransformerBaseline(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # 取最后一个时间步
        return self.output_proj(x)


def run_comparison():
    print("⚔️ 启动【真实】鲁棒性大对决 (Ultra-LSNT vs Real Transformer)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 数据准备
    data_path = 'wind_final.csv'
    target = 'power'
    if not os.path.exists(data_path):
        print("❌ 找不到 wind_final.csv")
        return

    data, _ = load_csv_data(data_path, target)
    if data is None:
        print("❌ 数据加载失败")
        return
        
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target=target)
    # 为了节省时间，如果你已经跑过一次觉得Transformer结果可信，可以把 epochs 改小
    train_config = TrainConfig(batch_size=128, epochs=5, lr=0.001)

    train_loader, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)
    input_dim = data.shape[1]

    # 2. 准备模型 A: 你的 Ultra-LSNT (加载训练好的)
    print("\n🤖 [Model A] 加载 Ultra-LSNT...")
    model_ours = UltraLSNTForecaster(LSNTConfig(input_dim=input_dim), ts_config).to(device)
    try:
        ckpt = torch.load('checkpoints_ts/main/best_model.pth', map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model_ours.load_state_dict(state_dict)
        print("   ✅ 成功加载 Ultra-LSNT 权重")
    except Exception as e:
        print(f"   ⚠️ 加载失败 ({e})")
        return

    # 3. 准备模型 B: Transformer (现场训练!)
    print("\n🤖 [Model B] 正在现场训练 Transformer Baseline...")
    model_trans = TransformerBaseline(input_dim, 96, 24).to(device)
    optimizer = torch.optim.Adam(model_trans.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model_trans.train()
    start_t = time.time()
    for epoch in range(train_config.epochs):
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model_trans(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   ⏳ Epoch {epoch + 1}/{train_config.epochs} | Loss: {total_loss / len(train_loader):.4f}")

    # 4. 鲁棒性测试
    print("\n🌪️ 开始注入噪音并对比...")
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    scores_ours = []
    scores_trans = []

    model_ours.eval()
    model_trans.eval()

    target_std = scaler.std[-1]
    target_mean = scaler.mean[-1]

    for noise in noise_levels:
        preds_o, preds_t, trues = [], [], []

        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)

                # 加噪音
                noise_tensor = torch.randn_like(bx) * noise
                bx_noisy = bx + noise_tensor

                # 预测
                out_o, _ = model_ours(bx_noisy)
                out_t = model_trans(bx_noisy)

                preds_o.append(out_o.cpu().numpy())
                preds_t.append(out_t.cpu().numpy())
                trues.append(by.cpu().numpy())

        # 拼接
        p_o = np.concatenate(preds_o, axis=0) * target_std + target_mean
        p_t = np.concatenate(preds_t, axis=0) * target_std + target_mean
        y = np.concatenate(trues, axis=0) * target_std + target_mean

        r2_o = compute_metrics(p_o, y)['R2']
        r2_t = compute_metrics(p_t, y)['R2']

        scores_ours.append(r2_o)
        scores_trans.append(r2_t)
        print(f"   Noise {int(noise * 100)}% -> Ours: {r2_o:.4f} | Trans: {r2_t:.4f}")

    # 保存 Ultra-LSNT 的分数供其他脚本使用
    np.save('robustness_scores_ultra_lsnt.npy', np.array(scores_ours))
    print(f"\n   💾 Ultra-LSNT 鲁棒性分数已保存: {scores_ours}")

    # 5. 画图
    print("   🎨 正在绘图...")
    plt.figure(figsize=(8, 6), dpi=300)

    plt.plot(noise_levels, scores_ours, marker='*', markersize=12, color='#d62728', linewidth=3,
             label='Ultra-LSNT (Ours)')
    plt.plot(noise_levels, scores_trans, marker='o', linestyle='--', color='#1f77b4', linewidth=2,
             label='Transformer (Baseline)')

    # 填充差距
    plt.fill_between(noise_levels, scores_ours, scores_trans, where=(np.array(scores_ours) > np.array(scores_trans)),
                     color='#d62728', alpha=0.1, label='Advantage Gap')

    plt.xticks(noise_levels, [f'{int(n * 100)}%' for n in noise_levels])
    plt.xlabel('Input Noise Level ($\sigma$)', fontsize=12, fontweight='bold')
    plt.ylabel('R-Squared Score ($R^2$)', fontsize=12, fontweight='bold')
    plt.title('Robustness Comparison: Ultra-LSNT vs Transformer', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig('fig_robustness_compare.png', dpi=300)
    print("\n✅ 对比完成！已生成真实对比图: fig_robustness_compare.png")


if __name__ == '__main__':
    run_comparison()