"""
Unified Cluster Runner for the 3D Controlled Ablation Experiment Matrix.

Arms:
1. vanilla_unconstrained: DLinear(Power), lambda=0
2. info_symmetric_unconstrained: DualChannelDLinear(Power + Wind), lambda=0
3. genuine_curve: DLinear(Power) + Curve prior, lambda in [0.01, 0.1, 1.0]
4. genuine_dist: DLinear(Power) + Distribution prior, lambda in [0.01, 0.1, 1.0]
5. pseudo_scrambled_curve: DLinear(Power) + Scrambled Curve prior, lambda in [0.01, 0.1, 1.0]
6. pseudo_scrambled_dist: DLinear(Power) + Scrambled Distribution prior, lambda in [0.01, 0.1, 1.0]
7. lag_scan_curve: DLinear(Power) + Curve prior, lag in [0, 6, 12, 18, 36]
8. zero_bid & persistence: Baselines for audit comparison
"""

import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from core import CFG, Forecaster, starts, windows, fit_prior, infer
from controlled_ablation import DualChannelDLinear, fit_scrambled_prior, controlled_penalty
from audit_evaluator import pinball_loss, evaluate_predictions, settlement_diagnostic

TURBINE_CONFIGS = {
    'kelmarsh': {'nt': 6, 'cap': 12300},
    'hill_of_towie': {'nt': 21, 'cap': 48300},
    'penmanshiel': {'nt': 14, 'cap': 28700}
}

def train_model(model, trainxy, valxy, prior, family, lam, factor, device, epochs, patience, lag_step=0):
    opt = torch.optim.Adam(model.parameters(), lr=CFG['learning_rate'])
    xt, yt = [torch.as_tensor(a, device=device) for a in trainxy]
    vp = valxy[1]
    best_loss = float('inf')
    best_state = None
    stale = 0
    pri_t = {k: torch.as_tensor(v, device=device) for k, v in prior.items()}
    rng = np.random.default_rng(42)

    for epoch in range(epochs):
        model.train()
        for ix in np.array_split(rng.permutation(len(xt)), int(np.ceil(len(xt) / CFG['batch_size']))):
            x = xt[ix]
            y = yt[ix]
            opt.zero_grad()
            pred = model(x)
            mse = F.mse_loss(pred, y)
            pen = controlled_penalty(pred, x, family, factor, pri_t, lag_step=lag_step)
            loss = mse + lam * pen
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered in epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            v_pred = infer(model, valxy[0], device)
            v_loss = float(np.mean((v_pred - vp) ** 2))

        if v_loss < best_loss:
            best_loss = v_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_loss

def run_experiment(site, seed, gpu_id, data_path, prices_path, out_dir):
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    out_path = Path(out_dir) / site / f'seed{seed}'
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[{site.upper()} | Seed {seed}] Starting on device: {device}")
    
    # Load site data
    f = pd.read_csv(Path(data_path) / f'{site}.csv', index_col='time', parse_dates=True)
    f.index = pd.to_datetime(f.index, utc=True)
    
    # Load prices
    prices = pd.read_csv(prices_path)
    prices.index = pd.to_datetime(prices.startTime, utc=True)
    price_series = prices.systemBuyPrice
    
    # Preprocess splits
    split = CFG['splits'][site]
    x, y, ids = starts(f, split)
    trainxy = windows(x, y, ids['train'])
    valxy = windows(x, y, ids['val'])
    calxy = windows(x, y, ids['cal'])
    testxy = windows(x, y, ids['test'])
    
    times_test = f.index[ids['test']]
    y_test = testxy[1]
    y_cal = calxy[1]
    
    # Fit genuine and scrambled priors
    genuine_prior = fit_prior(f, split['train_end'])
    scrambled_prior = fit_scrambled_prior(genuine_prior, seed=seed)
    
    results = []
    
    # Define experimental arms
    arms = []
    # 1. Vanilla unconstrained
    arms.append(('vanilla_unconstrained', 'none', 0.0, 1.0, genuine_prior, 0, 'power_only'))
    # 2. Information symmetry (Dual Channel)
    arms.append(('info_symmetric_unconstrained', 'none', 0.0, 1.0, genuine_prior, 0, 'dual_channel'))
    
    # 3 & 4: Genuine Curve & Distribution
    for lam in [0.01, 0.1, 1.0]:
        arms.append((f'genuine_curve_l{lam}', 'curve', lam, 1.0, genuine_prior, 0, 'power_only'))
        arms.append((f'genuine_dist_l{lam}', 'distribution', lam, 1.0, genuine_prior, 0, 'power_only'))
        
    # 5 & 6: Pseudo Scrambled Curve & Distribution
    for lam in [0.01, 0.1, 1.0]:
        arms.append((f'pseudo_scrambled_curve_l{lam}', 'curve', lam, 1.0, scrambled_prior, 0, 'power_only'))
        arms.append((f'pseudo_scrambled_dist_l{lam}', 'distribution', lam, 1.0, scrambled_prior, 0, 'power_only'))
        
    # 7: Lag Scan (use lam=0.1)
    for lag in [6, 12, 18, 36]:
        arms.append((f'lag_scan_curve_k{lag}', 'curve', 0.1, 1.0, genuine_prior, lag, 'power_only'))
        arms.append((f'lag_scan_dist_k{lag}', 'distribution', 0.1, 1.0, genuine_prior, lag, 'power_only'))

    for arm_name, family, lam, factor, pri, lag_step, arch in arms:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if arch == 'dual_channel':
            model = DualChannelDLinear(CFG['lookback'], CFG['horizon']).to(device)
        else:
            model = Forecaster('dlinear', CFG['lookback'], CFG['horizon']).to(device)
            
        model, val_loss = train_model(
            model, trainxy, valxy, pri, family, lam, factor, device,
            epochs=CFG['epochs'], patience=CFG['patience'], lag_step=lag_step
        )
        
        # Predictions
        cal_pred = np.clip(infer(model, calxy[0], device), 0.0, 1.0)
        test_pred = np.clip(infer(model, testxy[0], device), 0.0, 1.0)
        
        # Calibration delta for ratios [1, 2, 5]
        deltas = {r: np.quantile(y_cal - cal_pred, 1.0 / (r + 1.0), axis=0) for r in [1, 2, 5]}
        
        # Evaluate for uncalibrated and calibrated
        for policy in ['uncalibrated', 'calibrated']:
            for ratio in [1, 2, 5]:
                q = 1.0 / (ratio + 1.0)
                if policy == 'uncalibrated':
                    p_disp = test_pred
                else:
                    p_disp = np.clip(test_pred + deltas[ratio], 0.0, 1.0)
                    
                # Static dispatch cost (Pinball loss * (ratio + 1))
                diff = p_disp - y_test
                # Asymmetric dispatch cost: ratio * surplus + 1 * shortfall
                cost_static = float(np.mean(ratio * np.maximum(diff[:, :6], 0.0) + np.maximum(-diff[:, :6], 0.0)))
                pb_loss = float(pinball_loss(y_test[:, :6], p_disp[:, :6], q))
                
                # 30-minute settlement cashflow diagnostic
                # Compute average 30m aggregate over first 6 steps (1 hour / 2 SPs)
                sp_pred = p_disp[:, :3].mean(1)
                sp_true = y_test[:, :3].mean(1)
                
                # Sample 30-minute timestamps
                sp_times = (times_test - pd.Timedelta(minutes=10)).floor('30min')
                df_sp = pd.DataFrame({'sp': sp_times, 'pred': sp_pred, 'true': sp_true})
                df_agg = df_sp.groupby('sp').agg(pred=('pred', 'mean'), true=('true', 'mean'))
                df_agg['price'] = price_series.reindex(df_agg.index)
                df_agg = df_agg.dropna()
                
                cfg_t = TURBINE_CONFIGS[site]
                imbalance_mwh = (df_agg['true'] - df_agg['pred']) * cfg_t['cap'] / 1000.0 * 0.5
                shortfall_mwh = np.maximum(0.0, df_agg['pred'] - df_agg['true']) * cfg_t['cap'] / 1000.0 * 0.5
                
                daily_cashflow = (imbalance_mwh * df_agg['price']).groupby(df_agg.index.strftime('%Y-%m-%d')).sum()
                daily_shortfall = (shortfall_mwh * df_agg['price']).groupby(df_agg.index.strftime('%Y-%m-%d')).sum()
                
                annual_cashflow_per_turb = float(daily_cashflow.mean() * 365.25 / cfg_t['nt'])
                annual_shortfall_per_turb = float(daily_shortfall.mean() * 365.25 / cfg_t['nt'])
                
                results.append({
                    'site': site,
                    'seed': seed,
                    'arm': arm_name,
                    'family': family,
                    'lambda': lam,
                    'lag_step': lag_step,
                    'policy': policy,
                    'ratio': ratio,
                    'quantile': q,
                    'cost_static': cost_static,
                    'pinball_loss': pb_loss,
                    'rmse': float(np.sqrt(np.mean((p_disp - y_test) ** 2))),
                    'mae': float(np.mean(np.abs(p_disp - y_test))),
                    'bias': float(np.mean(p_disp - y_test)),
                    'net_cashflow_per_turb_yr': annual_cashflow_per_turb,
                    'shortfall_per_turb_yr': annual_shortfall_per_turb
                })
        print(f"[{site} | s{seed}] Arm {arm_name} done. (Val MSE: {val_loss:.6f})", flush=True)

    # Baselines: Zero-bid and Persistence
    for b_name in ['zero_bid', 'persistence']:
        if b_name == 'zero_bid':
            p_base = np.zeros_like(y_test)
        else:
            p_base = np.repeat(testxy[0][:, -1, 0, None], CFG['horizon'], axis=1)
            
        for ratio in [1, 2, 5]:
            q = 1.0 / (ratio + 1.0)
            diff = p_base - y_test
            cost_static = float(np.mean(ratio * np.maximum(diff[:, :6], 0.0) + np.maximum(-diff[:, :6], 0.0)))
            pb_loss = float(pinball_loss(y_test[:, :6], p_base[:, :6], q))
            
            sp_pred = p_base[:, :3].mean(1)
            sp_true = y_test[:, :3].mean(1)
            sp_times = (times_test - pd.Timedelta(minutes=10)).floor('30min')
            df_sp = pd.DataFrame({'sp': sp_times, 'pred': sp_pred, 'true': sp_true})
            df_agg = df_sp.groupby('sp').agg(pred=('pred', 'mean'), true=('true', 'mean'))
            df_agg['price'] = price_series.reindex(df_agg.index).dropna()
            
            cfg_t = TURBINE_CONFIGS[site]
            imbalance_mwh = (df_agg['true'] - df_agg['pred']) * cfg_t['cap'] / 1000.0 * 0.5
            shortfall_mwh = np.maximum(0.0, df_agg['pred'] - df_agg['true']) * cfg_t['cap'] / 1000.0 * 0.5
            
            daily_cashflow = (imbalance_mwh * df_agg['price']).groupby(df_agg.index.strftime('%Y-%m-%d')).sum()
            daily_shortfall = (shortfall_mwh * df_agg['price']).groupby(df_agg.index.strftime('%Y-%m-%d')).sum()
            
            results.append({
                'site': site,
                'seed': seed,
                'arm': b_name,
                'family': 'baseline',
                'lambda': 0.0,
                'lag_step': 0,
                'policy': 'uncalibrated',
                'ratio': ratio,
                'quantile': q,
                'cost_static': cost_static,
                'pinball_loss': pb_loss,
                'rmse': float(np.sqrt(np.mean((p_base - y_test) ** 2))),
                'mae': float(np.mean(np.abs(p_base - y_test))),
                'bias': float(np.mean(p_base - y_test)),
                'net_cashflow_per_turb_yr': float(daily_cashflow.mean() * 365.25 / cfg_t['nt']),
                'shortfall_per_turb_yr': float(daily_shortfall.mean() * 365.25 / cfg_t['nt'])
            })

    res_df = pd.DataFrame(results)
    csv_file = out_path / f'ablation_results_{site}_seed{seed}.csv'
    res_df.to_csv(csv_file, index=False)
    print(f"[{site} | s{seed}] COMPLETED -> Saved {len(res_df)} rows to {csv_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', required=True, choices=['kelmarsh', 'hill_of_towie', 'penmanshiel'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--prices_path', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    
    run_experiment(args.site, args.seed, args.gpu, args.data_path, args.prices_path, args.out_dir)
