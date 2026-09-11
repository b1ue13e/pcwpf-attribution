"""
Information-Budget Matched Experiment:
Disentangling Physical Inductive Bias from Auxiliary Historical Information Budget.

At N_train = 500 (approx. 21 days of data):
Arm A: Model 500 + No Prior (lambda = 0)
Arm B: Model 500 + Genuine Prior fitted on same 500 samples (lambda = 1.0) [Budget-Matched Physical Prior]
Arm C: Model 500 + Genuine Prior fitted on full historical pool (7-10 months) (lambda = 1.0) [Historical Data-Enriched Prior]
Arm D: Model 500 + Scrambled Prior fitted on same 500 samples (lambda = 1.0) [Budget-Matched Scrambled Control]
Arm E: Model 500 + Scrambled Prior fitted on full historical pool (lambda = 1.0) [Historical Scrambled Control]

Sites:
- Kelmarsh (flat terrain, N_sub=500 vs full 7-month prior)
- Hill of Towie (complex upland, N_sub=500 vs full 10-month prior)

Seeds: [42, 123, 456, 789, 1024]
"""

import sys, os, copy, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "constraint_round2"))
from core import CFG, Forecaster, starts, windows, fit_prior, infer
from controlled_ablation import fit_scrambled_prior, controlled_penalty

DATA_DIR = Path(r"e:\论文1\04_数据集\constraint_rebuild_v2")
OUT_DIR = Path(r"e:\论文1\06_实验结果与日志\information_budget")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]
SITES = ['kelmarsh', 'hill_of_towie']
N_SUB = 500

def train_subsample(model, xt_sub, yt_sub, valxy, prior, family, lam, factor, device, seed, epochs=30, patience=5):
    opt = torch.optim.Adam(model.parameters(), lr=CFG['learning_rate'])
    xt = torch.as_tensor(xt_sub, device=device)
    yt = torch.as_tensor(yt_sub, device=device)
    vp = valxy[1]
    best_loss = float('inf')
    best_state = None
    stale = 0
    pri_t = {k: torch.as_tensor(v, device=device) for k, v in prior.items()}
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        for ix in np.array_split(rng.permutation(len(xt)), int(np.ceil(len(xt) / CFG['batch_size']))):
            x = xt[ix]
            y = yt[ix]
            opt.zero_grad()
            pred = model(x)
            mse = F.mse_loss(pred, y)
            pen = controlled_penalty(pred, x, family, factor, pri_t, lag_step=0)
            loss = mse + lam * pen
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

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

def run_budget_experiment():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Running Information Budget Experiment on {device}...")
    
    all_results = []
    
    for site in SITES:
        print(f"\n==================== SITE: {site.upper()} (N_train={N_SUB}) ====================")
        f = pd.read_csv(DATA_DIR / f"{site}.csv", index_col='time', parse_dates=True)
        f.index = pd.to_datetime(f.index, utc=True)
        split = CFG['splits'][site]
        
        x, y, ids = starts(f, split)
        trainxy_full = windows(x, y, ids['train'])
        valxy = windows(x, y, ids['val'])
        calxy = windows(x, y, ids['cal'])
        testxy = windows(x, y, ids['test'])
        
        y_test = testxy[1]
        y_cal = calxy[1]
        
        # 1. Full historical prior (7-10 months)
        prior_full = fit_prior(f, split['train_end'])
        
        # 2. Local subsample window for prior
        tr_ids = ids['train']
        sub_ids = tr_ids[-N_SUB:]
        t_start = f.index[sub_ids[0] - CFG['lookback']]
        t_end = split['train_end']
        f_sub = f[(f.index >= t_start) & (f.index < pd.Timestamp(t_end, tz='UTC'))]
        prior_local = fit_prior(f_sub, t_end)
        
        # Model training data: the exact same 500 origins
        xt_sub = trainxy_full[0][-N_SUB:]
        yt_sub = trainxy_full[1][-N_SUB:]
        
        for seed in SEEDS:
            # Generate scrambled counterparts
            scrambled_full = fit_scrambled_prior(prior_full, seed=seed)
            scrambled_local = fit_scrambled_prior(prior_local, seed=seed)
            
            arms = [
                ('A_no_prior', 'none', 0.0, prior_full, 'none', '500_samples'),
                ('B_genuine_local_budget', 'curve', 1.0, prior_local, 'genuine', '500_samples'),
                ('C_genuine_full_historical', 'curve', 1.0, prior_full, 'genuine', 'full_history'),
                ('D_scrambled_local_budget', 'curve', 1.0, scrambled_local, 'scrambled', '500_samples'),
                ('E_scrambled_full_historical', 'curve', 1.0, scrambled_full, 'scrambled', 'full_history'),
            ]
            
            for arm_name, family, lam, pri, pri_type, pri_budget in arms:
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                model = Forecaster('dlinear', CFG['lookback'], CFG['horizon']).to(device)
                t0 = time.time()
                model, val_loss = train_subsample(
                    model, xt_sub, yt_sub, valxy, pri, family, lam, 1.0, device, seed
                )
                elapsed = time.time() - t0
                
                # Predictions
                cal_pred = np.clip(infer(model, calxy[0], device), 0.0, 1.0)
                test_pred = np.clip(infer(model, testxy[0], device), 0.0, 1.0)
                
                # Quantile calibration for r=2 (q=1/3)
                delta_r2 = np.quantile(y_cal - cal_pred, 1.0 / 3.0, axis=0)
                p_calibrated = np.clip(test_pred + delta_r2, 0.0, 1.0)
                
                for pol_name, p_disp in [('uncalibrated', test_pred), ('calibrated', p_calibrated)]:
                    diff = p_disp - y_test
                    cost_disp = float(np.mean(2.0 * np.maximum(diff[:, :6], 0.0) + np.maximum(-diff[:, :6], 0.0)))
                    rmse = float(np.sqrt(np.mean((p_disp - y_test) ** 2)))
                    mae = float(np.mean(np.abs(p_disp - y_test)))
                    
                    all_results.append({
                        'site': site,
                        'n_train': N_SUB,
                        'seed': seed,
                        'arm': arm_name,
                        'prior_type': pri_type,
                        'prior_budget': pri_budget,
                        'policy': pol_name,
                        'dispatch_cost_r2': cost_disp,
                        'rmse': rmse,
                        'mae': mae,
                        'train_time_sec': elapsed
                    })
                
                print(f"[{site:12s} | s{seed:4d}] {arm_name:28s} -> Cost(cal): {cost_disp:.5f}, RMSE: {rmse:.4f} ({elapsed:.1f}s)")
                
    df = pd.DataFrame(all_results)
    out_csv = OUT_DIR / "information_budget_experiment_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSAVED ALL RESULTS -> {out_csv} ({len(df)} records)")
    
    # Generate summary pivot for calibrated policy
    summary = df[df['policy'] == 'calibrated'].groupby(['site', 'arm', 'prior_type', 'prior_budget'])['dispatch_cost_r2'].agg(['mean', 'std']).reset_index()
    print("\n=== SUMMARY TABLE (Calibrated Dispatch Cost r=2, N_train=500) ===")
    print(summary.to_string(index=False))
    summary_csv = OUT_DIR / "information_budget_experiment_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"SAVED SUMMARY -> {summary_csv}")

if __name__ == '__main__':
    run_budget_experiment()
