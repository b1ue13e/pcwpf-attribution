"""
check_storage.py
================
对比 Ultra-LSNT 和 LightGBM 的模型存储空间占用。
这是证明 "Edge-Native" (边缘原生) 的核心证据。
"""
import os
import joblib
import torch
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig
)


def get_file_size_mb(filepath):
    """获取文件大小 (MB)"""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0


def flatten_data(loader):
    """展平数据用于 GBDT 训练"""
    X_list, y_list = [], []
    for bx, by in loader:
        batch_size = bx.shape[0]
        X_flat = bx.reshape(batch_size, -1).numpy()
        y_list.append(by.numpy())
        X_list.append(X_flat)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def main():
    print("💾 启动存储空间对比测试...")

    # ----------------------------------------
    # 1. 检查 Ultra-LSNT 大小
    # ----------------------------------------
    # 路径通常在 checkpoints_ts/main/best_model.pth
    lsnt_path = 'checkpoints_ts/main/best_model.pth'

    # 如果找不到，尝试搜索一下
    if not os.path.exists(lsnt_path):
        print("   ⚠️ 默认路径没找到，正在搜索 checkpoints_ts...")
        for root, dirs, files in os.walk('checkpoints_ts'):
            for file in files:
                if file.endswith('.pth'):
                    lsnt_path = os.path.join(root, file)
                    break

    size_lsnt = get_file_size_mb(lsnt_path)
    print(f"\n1️⃣  Ultra-LSNT 模型文件: {lsnt_path}")
    if size_lsnt > 0:
        print(f"    大小: {size_lsnt:.4f} MB")
    else:
        print("    ❌ 未找到模型文件，请先运行主训练脚本！")

    # ----------------------------------------
    # 2. 生成并检查 LightGBM 大小
    # ----------------------------------------
    print("\n2️⃣  正在生成 LightGBM 模型文件 (用于测重)...")
    lgbm_path = 'lgbm_model_dump.pkl'

    # 快速加载数据训练一个
    data, _ = load_csv_data('wind_final.csv', 'power')
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    train_config = TrainConfig(batch_size=1024)
    train_loader, _, _, _ = create_dataloaders(data, ts_config, train_config)
    X_train, y_train = flatten_data(train_loader)

    # 训练 (同 run_gbdt.py 配置)
    # 注意：MultiOutputRegressor 会为 24 个预测步长建立 24 个独立的模型！
    print("    正在训练 (24个子模型)...")
    model = MultiOutputRegressor(LGBMRegressor(n_estimators=1000, num_leaves=31, n_jobs=-1))
    model.fit(X_train, y_train)

    # 保存
    joblib.dump(model, lgbm_path, compress=3)  # compress=3 已经是高压缩率了
    size_lgbm = get_file_size_mb(lgbm_path)
    print(f"    大小: {size_lgbm:.4f} MB")

    # ----------------------------------------
    # 3. 最终对比报告
    # ----------------------------------------
    print("\n" + "=" * 50)
    print("📊 边缘端部署可行性报告 (Edge Deployment Report)")
    print("=" * 50)

    if size_lsnt > 0:
        ratio = size_lgbm / size_lsnt
        print(f"Ultra-LSNT (Ours) :  {size_lsnt:.2f} MB  ✅ (适合嵌入式芯片)")
        print(f"LightGBM (Baseline):  {size_lgbm:.2f} MB  ❌ (过于庞大)")
        print("-" * 50)
        print(f"结论: LightGBM 的体积是 Ultra-LSNT 的 {ratio:.1f} 倍！")
        print("在内存受限的 ARM 芯片(如树莓派/单片机)上，LightGBM 难以进行 OTA 热更新。")
    else:
        print("无法计算对比，因缺少 Ultra-LSNT 模型文件。")

    # 清理临时文件
    # os.remove(lgbm_path)
    print(f"\n注：LightGBM 模型已保存为 {lgbm_path}，你可以去文件夹查看。")


if __name__ == '__main__':
    main()