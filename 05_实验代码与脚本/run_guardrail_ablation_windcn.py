#!/usr/bin/env python3
"""
3-Arm Guardrail Ablation Experiment on Wind-CN Dataset
Arm 1: No Guardrail (Standard MSE training, unconstrained inference)
Arm 2: Post-hoc Hard Clipping (Standard MSE training, inference clipped to [0, P_rated])
Arm 3: Joint Guardrail (Differentiable soft penalty during training, swept across lambda)

Protocol:
- Backbone: Standard DLinear with Series Decomposition (Zeng et al., AAAI 2023)
- Seeds: [42, 123, 456, 789, 1024] (5 independent runs)
- Noise levels: sigma in [0.0, 0.2, 0.4, 0.6]
- Lambda sweep for Arm 3: [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
- Metrics: RMSE (kW), MAE (kW), Violation Rate (%)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    def __init__(self, seq_len=96, pred_len=24, enc_in=6):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomposition = SeriesDecomp(25)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

def load_and_prepare_data(csv_path: str, seq_len=96, pred_len=24, step=2):
    df = pd.read_csv(csv_path)
    feature_cols = ['windspeed', 'winddirection', 'temperature', 'humidity', 'pressure', 'power']
    data = df[feature_cols].values.astype(np.float32)
    
    n_samples = len(data)
    train_end = int(n_samples * 0.8)
    train_raw = data[:train_end]
    test_raw = data[train_end:]

    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0)
    std[std == 0] = 1.0

    train_scaled = (train_raw - mean) / std
    test_scaled = (test_raw - mean) / std

    p_min_raw = 0.0
    p_max_raw = float(np.max(train_raw[:, -1]))
    p_mean = mean[-1]
    p_std = std[-1]
    
    y_min_norm = (p_min_raw - p_mean) / p_std
    y_max_norm = (p_max_raw - p_mean) / p_std

    def make_windows(arr):
        xs, ys = [], []
        max_idx = len(arr) - seq_len - pred_len + 1
        for i in range(0, max_idx, step):
            xs.append(arr[i : i + seq_len])
            ys.append(arr[i + seq_len : i + seq_len + pred_len, -1])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    x_train, y_train = make_windows(train_scaled)
    x_test, y_test = make_windows(test_scaled)

    info = {
        'mean': mean,
        'std': std,
        'p_mean': p_mean,
        'p_std': p_std,
        'p_min_raw': p_min_raw,
        'p_max_raw': p_max_raw,
        'y_min_norm': y_min_norm,
        'y_max_norm': y_max_norm,
        'n_train': len(x_train),
        'n_test': len(x_test)
    }
    return x_train, y_train, x_test, y_test, info

def train_dlinear(
    x_train, y_train, 
    info,
    device,
    lambda_guard=0.0,
    epochs=10,
    batch_size=128,
    lr=0.003
):
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    enc_in = x_train.shape[2]
    seq_len = x_train.shape[1]
    pred_len = y_train.shape[1]
    
    model = DLinear(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_crit = nn.MSELoss()
    
    y_min_norm = info['y_min_norm']
    y_max_norm = info['y_max_norm']

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            pred = out[:, :, -1]
            
            loss = mse_crit(pred, by)
            if lambda_guard > 0.0:
                lower_viol = torch.relu(y_min_norm - pred)
                upper_viol = torch.relu(pred - y_max_norm)
                guard_loss = torch.mean(lower_viol**2 + upper_viol**2)
                loss = loss + lambda_guard * guard_loss
                
            loss.backward()
            optimizer.step()
            
    return model

def evaluate_model(model, x_test, y_test, info, device, sigma=0.0, post_hoc_clip=False, noise_seed=None):
    model.eval()
    
    if sigma > 0.0:
        if noise_seed is not None:
            rng = np.random.RandomState(noise_seed)
            noise = rng.normal(0.0, sigma, size=x_test.shape).astype(np.float32)
        else:
            noise = np.random.normal(0.0, sigma, size=x_test.shape).astype(np.float32)
        x_eval = x_test + noise
    else:
        x_eval = x_test.copy()

    with torch.no_grad():
        x_tensor = torch.from_numpy(x_eval).to(device)
        preds_list = []
        batch_size = 512
        for i in range(0, len(x_eval), batch_size):
            bx = x_tensor[i : i + batch_size]
            out = model(bx)
            preds_list.append(out[:, :, -1].cpu().numpy())
        pred_norm = np.concatenate(preds_list, axis=0)

    p_mean = info['p_mean']
    p_std = info['p_std']
    p_min_raw = info['p_min_raw']
    p_max_raw = info['p_max_raw']

    pred_raw = pred_norm * p_std + p_mean
    y_true_raw = y_test * p_std + p_mean

    raw_violations = np.sum((pred_raw < p_min_raw) | (pred_raw > p_max_raw))
    total_points = pred_raw.size
    violation_rate = float(100.0 * raw_violations / total_points)

    if post_hoc_clip:
        pred_eval = np.clip(pred_raw, p_min_raw, p_max_raw)
        eval_violation_rate = 0.0
    else:
        pred_eval = pred_raw
        eval_violation_rate = violation_rate

    rmse = float(np.sqrt(np.mean((pred_eval - y_true_raw) ** 2)))
    mae = float(np.mean(np.abs(pred_eval - y_true_raw)))
    ss_tot = np.sum((y_true_raw - np.mean(y_true_raw)) ** 2)
    ss_res = np.sum((y_true_raw - pred_eval) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'violation_rate': eval_violation_rate,
        'raw_violation_rate': violation_rate
    }

def run_experiments():
    parser = argparse.ArgumentParser(description="3-Arm Guardrail Ablation Experiment on Wind-CN")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024])
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.0, 0.2, 0.4, 0.6])
    parser.add_argument('--lambda_sweep', type=float, nargs='+', default=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--step', type=int, default=2)
    parser.add_argument('--quick', action='store_true', help="Run fast smoke test")
    args = parser.parse_args()

    if args.quick:
        seeds = [42]
        noise_levels = [0.0, 0.4]
        lambda_sweep = [0.0, 0.1]
        epochs = 1
        step = 8
        print(">>> RUNNING IN QUICK SMOKE-TEST MODE <<<")
    else:
        seeds = args.seeds
        noise_levels = args.noise_levels
        lambda_sweep = args.lambda_sweep
        epochs = args.epochs
        step = args.step

    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting 3-Arm Guardrail Experiment on {device} ===")

    data_path = Path("04_数据集/wind_final.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    x_train, y_train, x_test, y_test, info = load_and_prepare_data(str(data_path), step=step)
    print(f"Dataset: train windows={info['n_train']}, test windows={info['n_test']}")
    print(f"Power range: [{info['p_min_raw']:.1f}, {info['p_max_raw']:.1f}] kW (mean={info['p_mean']:.1f}, std={info['p_std']:.1f})")

    records = []

    print("\n--- Running Arm 1 (No Guardrail) and Arm 2 (Post-hoc Clipping) ---")
    for seed in seeds:
        set_seed(seed)
        print(f"Training Base DLinear (Seed {seed})...")
        base_model = train_dlinear(x_train, y_train, info, device, lambda_guard=0.0, epochs=epochs, batch_size=args.batch_size, lr=args.lr)

        for sigma in noise_levels:
            n_seed = seed * 1000 + int(sigma * 100)
            m1 = evaluate_model(base_model, x_test, y_test, info, device, sigma=sigma, post_hoc_clip=False, noise_seed=n_seed)
            records.append({
                'arm': 'Arm 1: No Guardrail',
                'seed': seed,
                'sigma': sigma,
                'lambda': 0.0,
                'post_hoc_clip': False,
                'rmse': m1['rmse'],
                'mae': m1['mae'],
                'r2': m1['r2'],
                'violation_rate': m1['violation_rate']
            })

            m2 = evaluate_model(base_model, x_test, y_test, info, device, sigma=sigma, post_hoc_clip=True, noise_seed=n_seed)
            records.append({
                'arm': 'Arm 2: Post-hoc Clipping',
                'seed': seed,
                'sigma': sigma,
                'lambda': 0.0,
                'post_hoc_clip': True,
                'rmse': m2['rmse'],
                'mae': m2['mae'],
                'r2': m2['r2'],
                'violation_rate': m2['violation_rate']
            })

    print("\n--- Running Arm 3 (Joint Guardrail: Lambda Sweep) ---")
    for lam in lambda_sweep:
        if lam == 0.0:
            continue
        print(f"\nTraining Joint Guardrail models with lambda={lam}...")
        for seed in seeds:
            set_seed(seed)
            joint_model = train_dlinear(x_train, y_train, info, device, lambda_guard=lam, epochs=epochs, batch_size=args.batch_size, lr=args.lr)

            for sigma in noise_levels:
                n_seed = seed * 1000 + int(sigma * 100)
                m3_raw = evaluate_model(joint_model, x_test, y_test, info, device, sigma=sigma, post_hoc_clip=False, noise_seed=n_seed)
                records.append({
                    'arm': 'Arm 3: Joint Guardrail (Raw)',
                    'seed': seed,
                    'sigma': sigma,
                    'lambda': lam,
                    'post_hoc_clip': False,
                    'rmse': m3_raw['rmse'],
                    'mae': m3_raw['mae'],
                    'r2': m3_raw['r2'],
                    'violation_rate': m3_raw['violation_rate']
                })
                m3_clipped = evaluate_model(joint_model, x_test, y_test, info, device, sigma=sigma, post_hoc_clip=True, noise_seed=n_seed)
                records.append({
                    'arm': 'Arm 3: Joint Guardrail (Clipped)',
                    'seed': seed,
                    'sigma': sigma,
                    'lambda': lam,
                    'post_hoc_clip': True,
                    'rmse': m3_clipped['rmse'],
                    'mae': m3_clipped['mae'],
                    'r2': m3_clipped['r2'],
                    'violation_rate': m3_clipped['violation_rate']
                })

    df_results = pd.DataFrame(records)
    out_csv = Path("06_实验结果与日志/guardrail_ablation_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved raw experimental records to {out_csv} ({len(df_results)} rows)")

    agg_df = df_results.groupby(['arm', 'sigma', 'lambda', 'post_hoc_clip']).agg(
        rmse_mean=('rmse', 'mean'),
        rmse_std=('rmse', 'std'),
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std'),
        vr_mean=('violation_rate', 'mean'),
        vr_std=('violation_rate', 'std')
    ).reset_index()

    summary_csv = Path("06_实验结果与日志/guardrail_ablation_summary.csv")
    agg_df.to_csv(summary_csv, index=False)
    print(f"Saved aggregated summary to {summary_csv}")

    print("\n" + "="*80)
    print("TABLE 3: 3-ARM GUARDRAIL ABLATION ACROSS NOISE REGIMES (5 SEEDS)")
    print("="*80)
    target_arms = [
        ('Arm 1: No Guardrail', 0.0),
        ('Arm 2: Post-hoc Clipping', 0.0),
        ('Arm 3: Joint Guardrail (Raw)', 0.1),
        ('Arm 3: Joint Guardrail (Clipped)', 0.1)
    ]
    for sigma in noise_levels:
        print(f"\nNoise Level: sigma = {sigma:.1f} ({'Clean' if sigma==0.0 else 'Sensor Perturbation'})")
        print(f"{'Strategy / Arm':<35} | {'RMSE (kW)':<22} | {'MAE (kW)':<22} | {'Violation Rate (%)':<18}")
        print("-" * 105)
        for arm_name, lam in target_arms:
            sub = agg_df[(agg_df['arm'] == arm_name) & (np.isclose(agg_df['sigma'], sigma)) & (np.isclose(agg_df['lambda'], lam))]
            if len(sub) > 0:
                row = sub.iloc[0]
                rmse_str = f"{row['rmse_mean']:8.1f} +/- {row['rmse_std']:5.1f}"
                mae_str = f"{row['mae_mean']:8.1f} +/- {row['mae_std']:5.1f}"
                vr_str = f"{row['vr_mean']:5.2f}% +/- {row['vr_std']:4.2f}%"
                print(f"{arm_name:<35} | {rmse_str:<22} | {mae_str:<22} | {vr_str:<18}")

    print("\n--- Generating Figure 4: Pareto Trade-off Curve ---")
    plt.figure(figsize=(8, 5.5), dpi=300)
    
    plot_sigmas = [0.0, 0.4, 0.6]
    colors = {0.0: '#1f77b4', 0.4: '#ff7f0e', 0.6: '#d62728'}
    markers = {0.0: 'o', 0.4: 's', 0.6: '^'}
    labels = {0.0: 'Clean (sigma=0.0)', 0.4: 'Moderate noise (sigma=0.4)', 0.6: 'Severe noise (sigma=0.6)'}

    for sig in plot_sigmas:
        sig_data = []
        for lam in lambda_sweep:
            if lam == 0.0:
                arm_query = 'Arm 1: No Guardrail'
            else:
                arm_query = 'Arm 3: Joint Guardrail (Raw)'
            sub = agg_df[(agg_df['arm'] == arm_query) & (np.isclose(agg_df['sigma'], sig)) & (np.isclose(agg_df['lambda'], lam))]
            if len(sub) > 0:
                row = sub.iloc[0]
                sig_data.append((lam, row['rmse_mean'], row['vr_mean'], row['rmse_std'], row['vr_std']))
        
        sig_data.sort(key=lambda x: x[0])
        lams = [x[0] for x in sig_data]
        rmses = [x[1] for x in sig_data]
        vrs = [x[2] for x in sig_data]

        plt.plot(vrs, rmses, marker=markers[sig], color=colors[sig], label=labels[sig], linewidth=2.0, markersize=8)
        for i, lam in enumerate(lams):
            plt.annotate(f"$\\lambda={lam}$", (vrs[i], rmses[i]), textcoords="offset points", xytext=(6, 4), fontsize=9, alpha=0.85)

    plt.xlabel('Physical Envelope Violation Rate (%)', fontsize=11, fontweight='bold')
    plt.ylabel('Prediction Error RMSE (kW)', fontsize=11, fontweight='bold')
    plt.title('Figure 4: Pareto Frontier of Prediction Accuracy vs. Physical Violation Rate', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    fig4_pdf = Path("03_图表与绘图素材/fig_guardrail_pareto.pdf")
    fig4_png = Path("03_图表与绘图素材/fig_guardrail_pareto.png")
    fig4_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig4_pdf)
    plt.savefig(fig4_png)
    plt.close()
    print(f"Saved Figure 4 to {fig4_pdf} and {fig4_png}")

    elapsed = time.time() - start_time
    print(f"\nAll experiments completed successfully in {elapsed:.1f}s!")

if __name__ == '__main__':
    run_experiments()
