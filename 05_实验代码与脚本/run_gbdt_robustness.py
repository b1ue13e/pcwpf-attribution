# run_gbdt_robustness.py
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

# 引入你的项目
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    compute_metrics, UltraLSNTForecaster, LSNTConfig
)
from noise_utils import inject_industrial_noise

plt.rcParams['font.family'] = 'sans-serif'
# # # # plt.rcParams['font.serif'] = ['DejaVu Sans']  # 注释掉以避免警告  # 注释掉以避免警告  # 注释掉以避免警告  # 注释掉以避免警告


def run_real_battle():
    print("⚔️ [真实对决] LightGBM vs Ultra-LSNT (数据实时计算中)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 准备数据
    data_path = 'wind_final.csv'
    target = 'power'
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target=target)
    # Batch size 大一点为了推理快
    train_config = TrainConfig(batch_size=512)

    data, _ = load_csv_data(data_path, target)
    if data is None:
        print("❌ 数据加载失败")
        return
        
    input_dim = data.shape[1]

    # 创建 DataLoader
    train_loader, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)

    # 获取反归一化参数
    target_std = scaler.std[-1]
    target_mean = scaler.mean[-1]

    # -----------------------------------------------------------
    # 2. 准备选手 A: Ultra-LSNT (必须加载真实权重!)
    # -----------------------------------------------------------
    print("🤖 加载 Ultra-LSNT 真实权重...")
    # 加载模型配置
    import json
    config_path = 'checkpoints_ts/main/model_config.json'
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    # 强制使用检查点的hidden_dim=128
    if "hidden_dim" in config_dict:
        config_dict["hidden_dim"] = config_dict["hidden_dim"]  # 使用检查点值
    else:
        config_dict["hidden_dim"] = 128  # 默认值
    config = LSNTConfig(**config_dict)
    config.input_dim = input_dim  # 确保输入维度正确
    model_ours = UltraLSNTForecaster(config, ts_config).to(device)
    ckpt_path = 'checkpoints_ts/main/best_model.pth'
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ 找不到权重文件 {ckpt_path}！请先运行主训练脚本。")

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError:
        print(f"⚠️  警告：未找到权重文件 {ckpt_path}，使用随机权重")
        return
    ckpt_state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # 使用 strict=False 加载权重，忽略缺失键
    missing_keys, unexpected_keys = model_ours.load_state_dict(ckpt_state_dict, strict=False)
    if missing_keys:
        print(f"   ⚠️ 加载时缺失键: {len(missing_keys)} 个")
    if unexpected_keys:
        print(f"   ⚠️ 加载时多余键: {len(unexpected_keys)} 个")
    print(f"   ✅ Ultra-LSNT 就位，权重加载完成（忽略不匹配）")

    # -----------------------------------------------------------
    # 3. 准备选手 B: LightGBM (现场训练!)
    # -----------------------------------------------------------
    print("🌲 正在训练 LightGBM (使用全量训练集)...")

    # 辅助函数：把 DataLoader 转成 numpy
    def to_numpy(loader):
        X, y = [], []
        for bx, by in loader:
            # Flatten input: [B, L, D] -> [B, L*D]
            X.append(bx.reshape(bx.size(0), -1).numpy())
            y.append(by.numpy())
        return np.concatenate(X), np.concatenate(y)

    X_train, y_train = to_numpy(train_loader)
    # 注意：测试集不要在这里转，因为我们要加 Tensor 级别的噪音

    lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1, random_state=42)
    model_lgbm = MultiOutputRegressor(lgbm)
    model_lgbm.fit(X_train, y_train)
    print("   ✅ LightGBM 训练完毕")

    # -----------------------------------------------------------
    # 4. 真实鲁棒性测试 (Impulse Noise)
    # -----------------------------------------------------------
    noise_type = 'gaussian'  # <--- 改这里！！！
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]  # 加大难度

    scores_ours = []
    scores_lgbm = []

    print(f"\n🌪️ 开始注入真实噪音 ({noise_type})...")

    for level in noise_levels:
        preds_o, preds_l, trues = [], [], []

        # 遍历测试集
        for bx, by in test_loader:
            bx = bx.to(device)

            # === 关键：先加噪音，再给两个模型 ===
            bx_noisy = inject_industrial_noise(bx, noise_type, level)

            # 1. Ultra-LSNT 推理 (Tensor)
            with torch.no_grad():
                out_o, _ = model_ours(bx_noisy)
                preds_o.append(out_o.cpu().numpy())

            # 2. LightGBM 推理 (Tensor -> Numpy -> Flatten)
            # 必须用加上了同样噪音的数据！
            X_numpy_noisy = bx_noisy.cpu().numpy().reshape(bx.size(0), -1)
            out_l = model_lgbm.predict(X_numpy_noisy)
            preds_l.append(out_l)

            trues.append(by.numpy())

        # 拼接 & 反归一化
        p_o = np.concatenate(preds_o) * target_std + target_mean
        p_l = np.concatenate(preds_l) * target_std + target_mean
        y_true = np.concatenate(trues) * target_std + target_mean

        # 计算 R2
        r2_o = compute_metrics(p_o, y_true)['R2']
        r2_l = compute_metrics(p_l, y_true)['R2']

        scores_ours.append(r2_o)
        scores_lgbm.append(r2_l)

        print(f"   Noise {level:.2f} | Ultra-LSNT: {r2_o:.4f} vs LightGBM: {r2_l:.4f}")

    # -----------------------------------------------------------
    # 5. 绘图
    # -----------------------------------------------------------
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(noise_levels, scores_ours, 'r*-', linewidth=3, markersize=12, label='Ultra-LSNT (Ours)')
    plt.plot(noise_levels, scores_lgbm, 'bo--', linewidth=2, markersize=8, label='LightGBM (Baseline)')

    plt.xlabel(f'Noise Level ({noise_type})', fontsize=12, fontweight='bold')
    plt.ylabel('R-Squared ($R^2$)', fontsize=12, fontweight='bold')
    plt.title(f'Robustness: Ultra-LSNT vs GBDT (Noise: {noise_type.title()})', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig('fig_gbdt_battle.png', bbox_inches='tight')
    print("\n✅ 真实数据图表已生成: fig_gbdt_battle.png")


if __name__ == '__main__':
    run_real_battle()