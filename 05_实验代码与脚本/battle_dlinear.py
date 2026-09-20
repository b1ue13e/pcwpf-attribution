# battle_dlinear.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    UltraLSNTForecaster, LSNTConfig, compute_metrics
)
from noise_utils import inject_industrial_noise


# 简单的 DLinear 定义 (确保自包含)
class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, channels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        # Decompsition Kernel
        self.moving_avg = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, D]
        # Trend
        trend = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        # Seasonal
        seasonal = x - trend

        # Linear
        trend_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)

        x = trend_out + seasonal_out
        return x[:, :, -1]  # 假设预测最后一列


def run_dlinear_battle():
    print("⚔️ [真实对决] DLinear vs Ultra-LSNT...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 数据
    data, _ = load_csv_data('wind_final.csv', 'power')
    if data is None:
        print("❌ 数据加载失败")
        return
        
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    train_config = TrainConfig(batch_size=128, epochs=10)  # 现场训练 Epochs

    train_loader, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)
    input_dim = data.shape[1]

    # 2. 模型 A: Ultra-LSNT (加载权重)
    # 加载训练时使用的配置
    import json
    try:
        with open('checkpoints_ts/main/model_config.json', 'r') as f:
            saved_config = json.load(f)
        print(f"✅ 加载保存的模型配置: hidden_dim={saved_config.get('hidden_dim', 128)}")
        # 确保 heteroscedastic_moe 字段存在（旧配置文件可能没有）
        heteroscedastic_moe = saved_config.get('heteroscedastic_moe', False)
        print(f"   使用 heteroscedastic_moe={heteroscedastic_moe}")
        model_config = LSNTConfig(
            input_dim=input_dim,
            hidden_dim=saved_config.get('hidden_dim', 128),
            output_dim=saved_config.get('output_dim', 1),
            num_blocks=saved_config.get('num_blocks', 3),
            num_experts=saved_config.get('num_experts', 4),
            top_k=saved_config.get('top_k', 2),
            dropout=saved_config.get('dropout', 0.1),
            router_z_loss_coef=saved_config.get('router_z_loss_coef', 0.01),
            router_aux_loss_coef=saved_config.get('router_aux_loss_coef', 0.01),
            router_jitter_noise=saved_config.get('router_jitter_noise', 0.1),
            skip_threshold=saved_config.get('skip_threshold', 0.5),
            dual_mode_expansion=saved_config.get('dual_mode_expansion', 4),
            heteroscedastic_moe=heteroscedastic_moe
        )
    except Exception as e:
        print(f"⚠️ 无法加载配置文件: {e}, 使用默认配置")
        model_config = LSNTConfig(input_dim=input_dim, heteroscedastic_moe=False)
    
    # 打印配置确认
    print(f"📋 模型配置: hidden_dim={model_config.hidden_dim}, num_blocks={model_config.num_blocks}, heteroscedastic_moe={model_config.heteroscedastic_moe}")
    
    model_ours = UltraLSNTForecaster(model_config, ts_config).to(device)
    ckpt = torch.load('checkpoints_ts/main/best_model.pth', map_location=device)
    # 使用 strict=False 加载，忽略缺失的键（如 uncertainty_net）
    missing_keys, unexpected_keys = model_ours.load_state_dict(
        ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt,
        strict=False
    )
    if missing_keys:
        print(f"⚠️ 加载时缺失键: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
    if unexpected_keys:
        print(f"⚠️ 加载时多余键: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    model_ours.eval()

    # 3. 模型 B: DLinear (现场训练)
    print("🤖 正在现场训练 DLinear (请稍候)...")
    model_dlinear = DLinear(96, 24, input_dim).to(device)
    optim = torch.optim.Adam(model_dlinear.parameters(), lr=0.005)
    crit = nn.MSELoss()

    model_dlinear.train()
    for ep in range(train_config.epochs):
        losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optim.zero_grad()
            out = model_dlinear(bx)
            loss = crit(out, by)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        print(f"   Epoch {ep + 1}: Loss {np.mean(losses):.4f}")
    model_dlinear.eval()

    # 4. 鲁棒性测试 (Drift Noise - 线性模型比较怕趋势变化)
    noise_type = 'impulse'
    noise_levels = [0.0, 0.05, 0.10, 0.15]

    scores_ours = []
    scores_dlinear = []

    target_std = scaler.std[-1]
    target_mean = scaler.mean[-1]

    print(f"\n🌪️ 注入毁灭性噪音: {noise_type}...")
    for lvl in noise_levels:
        p_o, p_d, y_t = [], [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)

                # 注入统一噪音
                bx_noisy = inject_industrial_noise(bx, noise_type, lvl)

                out_o, _ = model_ours(bx_noisy)
                out_d = model_dlinear(bx_noisy)

                p_o.append(out_o.cpu().numpy())
                p_d.append(out_d.cpu().numpy())
                y_t.append(by.numpy())

        # 拼接
        p_o = np.concatenate(p_o) * target_std + target_mean
        p_d = np.concatenate(p_d) * target_std + target_mean
        y = np.concatenate(y_t) * target_std + target_mean

        r2_o = compute_metrics(p_o, y)['R2']
        r2_d = compute_metrics(p_d, y)['R2']

        scores_ours.append(r2_o)
        scores_dlinear.append(r2_d)
        print(f"   Level {lvl}: Ours {r2_o:.4f} | DLinear {r2_d:.4f}")

    # 5. 绘图
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(noise_levels, scores_ours, 'r*-', label='Ultra-LSNT')
    plt.plot(noise_levels, scores_dlinear, 'gs--', label='DLinear')
    plt.title(f'Robustness: Drift Noise Comparison')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('fig_battle_dlinear.png')
    print("✅ DLinear 对比图已生成")


if __name__ == '__main__':
    run_dlinear_battle()