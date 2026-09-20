import torch
import numpy as np
from ultra_lsnt_timeseries import (
    UltraLSNTForecaster, LSNTConfig, TimeSeriesConfig,
    load_csv_data, create_dataloaders, TrainConfig, compute_metrics
)
# 我们借用之前写的 TransformerBaseline 类
from compare_robustness_real import TransformerBaseline


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_stats():
    print("📊 正在计算模型参数量和完整指标...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 准备数据
    data_path = 'wind_final.csv'
    data, _ = load_csv_data(data_path, 'power')
    input_dim = data.shape[1]
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    train_config = TrainConfig(batch_size=256)
    _, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)

    # 2. 定义模型
    # Ultra-LSNT
    model_ours = UltraLSNTForecaster(LSNTConfig(input_dim=input_dim), ts_config).to(device)
    # Transformer
    model_trans = TransformerBaseline(input_dim, 96, 24).to(device)

    # 3. 计算参数量
    params_ours = count_parameters(model_ours)
    params_trans = count_parameters(model_trans)

    print(f"\n📏 模型轻量化对比:")
    print(f"   Transformer Params: {params_trans:,}")
    print(f"   Ultra-LSNT Params : {params_ours:,}")
    print(f"   👉 节省比例: {(1 - params_ours / params_trans) * 100:.2f}%")

    # 4. 加载权重跑 MAE/RMSE (以 Ultra-LSNT 为例)
    print("\n📉 正在计算 Ultra-LSNT 的详细误差指标...")
    try:
        ckpt = torch.load('checkpoints_ts/main/best_model.pth', map_location=device)
        model_ours.load_state_dict(ckpt['model_state_dict'])
        model_ours.eval()

        preds, trues = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                out, _ = model_ours(bx)
                preds.append(out.cpu().numpy())
                trues.append(by.cpu().numpy())

        # 反归一化
        target_mean, target_std = scaler.mean[-1], scaler.std[-1]
        preds = np.concatenate(preds) * target_std + target_mean
        trues = np.concatenate(trues) * target_std + target_mean

        metrics = compute_metrics(preds, trues)
        print("✅ Ultra-LSNT 完整指标:")
        print(f"   R²  : {metrics['R2']:.4f}")
        print(f"   MAE : {metrics['MAE']:.4f}")
        print(f"   RMSE: {metrics['RMSE']:.4f}")

    except Exception as e:
        print(f"⚠️ 无法加载权重计算指标: {e}")


if __name__ == '__main__':
    get_stats()