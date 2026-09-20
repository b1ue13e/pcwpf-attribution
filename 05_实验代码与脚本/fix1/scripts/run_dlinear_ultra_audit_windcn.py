#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from windcn_audit_common import (
    ROOT,
    apply_zscore,
    audit_row,
    fit_zscore,
    inverse_target_scale,
    load_windcn_matrix,
    make_multivariate_windows,
    maybe_limit_windows,
    read_split_from_manifest,
    scenario_name,
    utc_run_id,
    write_csv,
    ZScoreStats
)
from ultra_lsnt_timeseries import (
    UltraLSNTForecaster, LSNTConfig, TimeSeriesConfig
)

DEFAULT_OUT_CSV = ROOT / "results" / "supplementary_evidence" / "dlinear_ultra_audit_windcn.csv"
DEFAULT_META_JSON = ROOT / "results" / "supplementary_evidence" / "dlinear_ultra_audit_windcn.meta.json"

# ==========================================
# DLinear Implementation (Simplified)
# ==========================================
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
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
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decompsition = SeriesDecomp(25)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1) # [Batch, Pred, Channel]

def train_model(model, train_loader, device, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            # Assuming target is last channel
            pred = out[:, :, -1]
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    return model

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="wind_final.csv")
    parser.add_argument("--target", type=str, default="power")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--sigma-cfgs", type=str, default="0.0,0.4") # 0.4 * 1.5 = 0.60
    parser.add_argument("--out-csv", type=str, default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    sigma_cfgs = [float(x) for x in args.sigma_cfgs.split(",")]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Loading
    data, feature_names, target_idx = load_windcn_matrix(data_path=args.data, target=args.target)
    train_end, test_end = read_split_from_manifest(len(data), dataset_name="wind_cn")
    train_data = data[:train_end]
    test_data = data[train_end:test_end]

    stats = fit_zscore(train_data)
    train_scaled = apply_zscore(train_data, stats)
    test_scaled = apply_zscore(test_data, stats)

    # Windows
    x_train, y_train = make_multivariate_windows(train_scaled, args.seq_len, args.pred_len, target_idx)
    x_test, y_test = make_multivariate_windows(test_scaled, args.seq_len, args.pred_len, target_idx)
    
    # Loader
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    run_id = utc_run_id("_dlinear_ultra_audit_windcn")
    rows = []
    
    # ==========================
    # 1. DLinear Audit
    # ==========================
    print("Auditing DLinear...")
    dlinear = DLinear(args.seq_len, args.pred_len, x_train.shape[2]).to(device)
    dlinear = train_model(dlinear, train_loader, device, epochs=args.epochs, lr=0.005)
    
    dlinear.eval()
    with torch.no_grad():
        for sigma_cfg in sigma_cfgs:
            if np.isclose(sigma_cfg, 0.0):
                x_eval = x_test
            else:
                from windcn_audit_common import inject_gaussian_windows
                x_eval = inject_gaussian_windows(x_test, sigma_cfg)
            
            x_tensor = torch.from_numpy(x_eval).to(device)
            out = dlinear(x_tensor)
            pred = out[:, :, -1].cpu().numpy()
            
            pred_denorm = inverse_target_scale(pred, stats, target_idx=target_idx)
            y_true_denorm = inverse_target_scale(y_test, stats, target_idx=target_idx)
            
            rows.append(audit_row(
                model="DLinear",
                scenario=scenario_name(sigma_cfg),
                sigma_cfg=sigma_cfg,
                pred_denorm=pred_denorm,
                true_denorm=y_true_denorm,
                run_id=run_id
            ))

    # ==========================
    # 2. Ultra-LSNT Audit
    # ==========================
    print("Auditing Ultra-LSNT...")
    # Try to load existing checkpoint to match "Panel A" performance if possible, 
    # otherwise train.
    ckpt_path = ROOT / "results" / "supplementary_evidence" / "ultra_lambda005_finetuned_from_main.pth"
    
    ultra_config = LSNTConfig(input_dim=x_train.shape[2], hidden_dim=64, num_experts=4, top_k=2)
    ts_config = TimeSeriesConfig(seq_len=args.seq_len, pred_len=args.pred_len, target=args.target)
    ultra = UltraLSNTForecaster(ultra_config, ts_config).to(device)
    
    if ckpt_path.exists():
        print(f"Loading Ultra-LSNT checkpoint: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            # Handle state dict structure if needed
            sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            ultra.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"Checkpoint load failed: {e}. Retraining...")
            # Simple train loop for Ultra if load fails
            optimizer = torch.optim.AdamW(ultra.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            ultra.train()
            for epoch in range(5): # Short train for audit if fallback
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    out, _ = ultra(bx)
                    pred = out.squeeze(-1)
                    loss = criterion(pred, by)
                    loss.backward()
                    optimizer.step()
    else:
        print("No checkpoint found. Training Ultra-LSNT from scratch...")
        optimizer = torch.optim.AdamW(ultra.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        ultra.train()
        for epoch in range(10): 
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out, _ = ultra(bx)
                pred = out.squeeze(-1)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

    ultra.eval()
    with torch.no_grad():
        for sigma_cfg in sigma_cfgs:
            if np.isclose(sigma_cfg, 0.0):
                x_eval = x_test
            else:
                from windcn_audit_common import inject_gaussian_windows
                x_eval = inject_gaussian_windows(x_test, sigma_cfg)
            
            x_tensor = torch.from_numpy(x_eval).to(device)
            out, _ = ultra(x_tensor)
            pred = out.squeeze(-1).cpu().numpy()
            
            pred_denorm = inverse_target_scale(pred, stats, target_idx=target_idx)
            y_true_denorm = inverse_target_scale(y_test, stats, target_idx=target_idx)
            
            rows.append(audit_row(
                model="Ultra-LSNT",
                scenario=scenario_name(sigma_cfg),
                sigma_cfg=sigma_cfg,
                pred_denorm=pred_denorm,
                true_denorm=y_true_denorm,
                run_id=run_id
            ))

    write_csv(rows, args.out_csv, columns=["model", "scenario", "sigma_cfg", "sigma_eff", "R2", "RMSE", "MAE", "residual_gaussian_pvalue", "run_id"])
    print(f"Audit completed. Results saved to {args.out_csv}")
    return 0

if __name__ == "__main__":
    main()
