import os

# 【新增】必须放在最前面！解决 OMP: Error #15 报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np

from ultra_lsnt_timeseries import UltraLSNTForecaster, LSNTConfig, TimeSeriesConfig

# 设置学术字体
plt.rcParams['font.family'] = 'sans-serif'
# # # # plt.rcParams['font.serif'] = ['DejaVu Sans']  # 注释掉以避免警告  # 注释掉以避免警告  # 注释掉以避免警告  # 注释掉以避免警告


class TransformerBaseline(nn.Module):
    """与 compare_robustness_real 中一致的 Transformer Baseline，用于效率对比"""
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
        x = x[:, -1, :]
        return self.output_proj(x)


def benchmark():
    print("🚀 启动算力边界测试 (Computational Boundary Test)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🖥️ 测试设备: {device}")

    # 测试序列长度：从 96 (1天) 到 2048 (约20天)
    seq_lengths = [96, 192, 336, 512, 1024, 2048]
    times_trans = []
    times_ours = []

    for L in seq_lengths:
        print(f"   ⚡ 正在测试序列长度 L = {L} ...")

        try:
            # 为公平起见，两边都用相同的特征维度（与风电数据一致，一般为 8）
            input_dim = 8
            seq_len = L

            # 定义真实 Ultra-LSNT 模型（来自 ultra_lsnt_timeseries.py）
            ts_config = TimeSeriesConfig(seq_len=seq_len, pred_len=24, target='power')
            model_cfg = LSNTConfig(input_dim=input_dim)
            model_ours = UltraLSNTForecaster(model_cfg, ts_config).to(device)

            # 定义 Transformer Baseline
            model_trans = TransformerBaseline(input_dim, seq_len, 24).to(device)

            # 构造一次性随机输入（只关心前向耗时，不需要真实数据）
            dummy_input = torch.randn(1, seq_len, input_dim, device=device)

            # 预热
            with torch.no_grad():
                for _ in range(5):
                    model_trans(dummy_input)
                    model_ours(dummy_input)[0]

            # 测速 Transformer
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            with torch.no_grad():
                for _ in range(50):
                    model_trans(dummy_input)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            # 计算单次推理时间 (ms)
            times_trans.append((time.time() - start) / 50 * 1000)

            # 测速 Ultra-LSNT
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            with torch.no_grad():
                for _ in range(50):
                    model_ours(dummy_input)[0]
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times_ours.append((time.time() - start) / 50 * 1000)

        except RuntimeError as e:
            print(f"   ❌ 显存溢出 (OOM) at L={L}: {e}")
            break

    # 画图
    print("   🎨 正在绘图...")
    plt.figure(figsize=(8, 6), dpi=300)

    # 截取有效数据（防止中间OOM导致长度不一致）
    valid_len = min(len(times_trans), len(times_ours))
    valid_seq = seq_lengths[:valid_len]

    plt.plot(valid_seq, times_trans[:valid_len], marker='o', linestyle='--', color='#ff7f0e',
             label='Transformer (O($L^2$))')
    plt.plot(valid_seq, times_ours[:valid_len], marker='*', markersize=12, linestyle='-', color='#d62728',
             linewidth=2.5, label='Ultra-LSNT (O($L$))')

    plt.xlabel('Input Sequence Length ($L$)', fontsize=12, fontweight='bold')
    plt.ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    plt.title('Efficiency Boundary: Inference Latency vs. Sequence Length', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('fig_efficiency.png', bbox_inches='tight', dpi=300)
    print("✅ 算力测试完成！已生成 fig_efficiency.png")


if __name__ == '__main__':
    benchmark()