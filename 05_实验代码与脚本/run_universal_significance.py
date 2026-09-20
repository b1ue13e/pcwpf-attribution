import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import sys
import os
import gc
import copy

# 引入你的模型库
sys.path.append(os.getcwd())
from ultra_lsnt_timeseries import (
    load_csv_data, TimeSeriesDataset, UltraLSNTForecaster, LSNTConfig, TimeSeriesConfig,
    LSTMBaseline
)

try:
    from compare_robustness_real import TransformerBaseline
    from battle_dlinear import DLinear
except ImportError:
    print("缺少 Transformer 或 DLinear 定义文件！")
    exit()

plt.rcParams['font.family'] = 'sans-serif'
# # # # plt.rcParams['font.serif'] = ['DejaVu Sans']  # 注释掉以避免警告  # 注释掉以避免警告  # 注释掉以避免警告  # 注释掉以避免警告

# ================= 配置区域 =================
DATASETS_TO_TEST = [
    ("Wind (CN)", "wind_final.csv", "power"),
    ("Wind (US)", "wind_us.csv", "power (MW)"),
    ("GEFCom Load", "gefcom_ready.csv", "load"),
    ("Air Quality", "air_quality_ready.csv", "AQI")
]

# 减少训练轮数以避免内存问题
TRAIN_EPOCHS = 20


# ===========================================

def dm_test(actual, pred1, pred2):
    """Diebold-Mariano Test"""
    e1 = (actual - pred1) ** 2
    e2 = (actual - pred2) ** 2
    d = e1 - e2
    d_mean = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d == 0: return 0.0, 1.0
    stat = d_mean / np.sqrt(var_d / len(d))
    p_val = 2 * (1 - norm.cdf(abs(stat)))
    return stat, p_val


def train_model_with_early_stopping(model, train_loader, epochs, device):
    """带学习率衰减和早停的训练循环"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 关键：学习率衰减，每 20 轮降低一半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    best_weights = None
    patience = 10  # 10轮不提升就早停
    patience_counter = 0

    model.train()
    for ep in range(epochs):
        epoch_loss = 0
        steps = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            if isinstance(out, tuple): out = out[0]
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        scheduler.step()  # 更新学习率
        avg_loss = epoch_loss / steps

        # 简易早停逻辑 (基于训练Loss，因为我们主要为了收敛)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break  # 早停

    # 加载最佳权重
    if best_weights:
        model.load_state_dict(best_weights)
    return model


def run_dm_matrix_for_dataset(name, path, target):
    print(f"\n[{name}] 启动全模型 DM Test 矩阵生成 (优化版)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(path):
        print(f"跳过: 找不到文件 {path}")
        return

    data, _ = load_csv_data(path, target)
    if data is None: return

    # 预处理
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    seq_len, pred_len = 96, 24
    input_dim = data.shape[1]

    # 减少batch size以降低内存使用
    train_loader = DataLoader(TimeSeriesDataset(train_scaled, seq_len, pred_len), batch_size=64, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(TimeSeriesDataset(test_scaled, seq_len, pred_len), batch_size=64, shuffle=False,
                             num_workers=0)

    # 定义模型
    model_classes = {
        'Ultra-LSNT': lambda: UltraLSNTForecaster(LSNTConfig(input_dim=input_dim),
                                                  TimeSeriesConfig(seq_len, pred_len)).to(device),
        'Transformer': lambda: TransformerBaseline(input_dim, seq_len, pred_len).to(device),
        'DLinear': lambda: DLinear(seq_len, pred_len, input_dim).to(device),
        'LSTM': lambda: LSTMBaseline(input_dim, 128, pred_len).to(device)
    }

    predictions = {}
    ground_truth = None

    # 串行训练
    for m_name, model_creator in model_classes.items():
        print(f"   Training {m_name}...", end=" ")
        model = model_creator()

        # 使用更智能的训练函数
        model = train_model_with_early_stopping(model, train_loader, TRAIN_EPOCHS, device)
        print("Done.")

        # 推理
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                if isinstance(out, tuple): out = out[0]
                preds.append(out.cpu().numpy())
                trues.append(by.cpu().numpy())

        flat_pred = np.concatenate(preds).flatten()
        flat_true = np.concatenate(trues).flatten()
        tgt_std = scaler.scale_[-1]
        tgt_mean = scaler.mean_[-1]
        predictions[m_name] = flat_pred * tgt_std + tgt_mean

        if ground_truth is None:
            ground_truth = flat_true * tgt_std + tgt_mean

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # 计算矩阵
    print(f"   Calculating Matrix...")
    model_names = list(model_classes.keys())
    n = len(model_names)
    p_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j: p_matrix[i, j] = np.nan; continue

            name_i, name_j = model_names[i], model_names[j]
            stat, p_val = dm_test(ground_truth, predictions[name_i], predictions[name_j])
            p_matrix[i, j] = p_val

            if name_i == 'Ultra-LSNT':
                if stat < 0 and p_val < 0.05:
                    res = "显著胜出 (WIN)"
                elif stat > 0 and p_val < 0.05:
                    res = "显著落败 (LOSS)"
                else:
                    res = "无显著差异 (TIE)"
                print(f"      vs {name_j:12}: p={p_val:.4f} -> {res}")

    # 绘图
    plt.figure(figsize=(9, 7), dpi=300)
    sns.heatmap(p_matrix, annot=True, fmt=".3f", cmap="Greens_r",
                xticklabels=model_names, yticklabels=model_names,
                cbar_kws={'label': 'P-value (< 0.05 is Significant)'},
                vmin=0, vmax=0.1)
    plt.title(f'DM Test Significance - {name}', fontsize=14)
    plt.tight_layout()

    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    save_path = f'fig_dm_matrix_{safe_name}.png'
    plt.savefig(save_path)
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    print("启动全领域显著性检验流水线 (High Performance Mode)...")
    for name, path, target in DATASETS_TO_TEST:
        run_dm_matrix_for_dataset(name, path, target)
    print("\n完成！")