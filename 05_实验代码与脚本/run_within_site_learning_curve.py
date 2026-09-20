"""
Within-Site Learning Curve Benchmark:
Decoupling sample size from site topography and turbine confounding.

Sites:
- Kelmarsh (flat terrain, Senvion MM92, full N=5,098)
- Hill of Towie (complex upland, Siemens SWT-2.3-82, full N=7,035)

Sample size regimes:
N_train in [500, 1000, 1500, 3000, 5000]

Arms:
1. Unconstrained DLinear (lambda = 0)
2. Genuine Physical Power Curve Prior (lambda = 1.0)
3. Pseudo-Scrambled Negative Control Prior (lambda = 1.0)

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
OUT_DIR = Path(r"e:\论文1\06_实验结果与日志\learning_curve")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZES = [500, 1000, 1500, 3000, 5000]
SEEDS = [42, 123, 456, 789, 1024]
SITES = ['kelmarsh', 'hill_of_towie']

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

def run_suite():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Running Within-Site Learning Curve on {device}...")
    
    all_results = []
    
    for site in SITES:
        print(f"\n==================== SITE: {site.upper()} ====================")
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
        
        genuine_prior = fit_prior(f, split['train_end'])
        
        total_train_samples = len(trainxy_full[0])
        print(f"Full training pool: {total_train_samples} samples")
        
        for n_sub in SAMPLE_SIZES:
            if n_sub > total_train_samples:
                continue
            
            print(f"\n--- Sample Size N_train = {n_sub} ---")
            
            for seed in SEEDS:
                scrambled_prior = fit_scrambled_prior(genuine_prior, seed=seed)
                
                # Subsample the training data: use the last n_sub samples to represent recent history
                xt_sub = trainxy_full[0][-n_sub:]
                yt_sub = trainxy_full[1][-n_sub:]
                
                arms = [
                    ('unconstrained', 'none', 0.0, genuine_prior),
                    ('genuine_prior', 'curve', 1.0, genuine_prior),
                    ('scrambled_prior', 'curve', 1.0, scrambled_prior),
                ]
                
                for arm_name, family, lam, pri in arms:
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
                    
                    # Compute costs
                    for pol_name, p_disp in [('uncalibrated', test_pred), ('calibrated', p_calibrated)]:
                        diff = p_disp - y_test
                        # Normalized static dispatch cost: ratio=2 (surplus=2*surplus + 1*shortfall)
                        cost_disp = float(np.mean(2.0 * np.maximum(diff[:, :6], 0.0) + np.maximum(-diff[:, :6], 0.0)))
                        rmse = float(np.sqrt(np.mean((p_disp - y_test) ** 2)))
                        mae = float(np.mean(np.abs(p_disp - y_test)))
                        
                        all_results.append({
                            'site': site,
                            'n_train': n_sub,
                            'seed': seed,
                            'arm': arm_name,
                            'policy': pol_name,
                            'dispatch_cost_r2': cost_disp,
                            'rmse': rmse,
                            'mae': mae,
                            'train_time_sec': elapsed
                        })
                    
                    print(f"[{site:12s} | N={n_sub:4d} | s{seed:4d}] {arm_name:16s} -> Cost(cal): {cost_disp:.5f}, RMSE: {rmse:.4f} ({elapsed:.1f}s)")
    
    df = pd.DataFrame(all_results)
    out_csv = OUT_DIR / "within_site_learning_curve_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSAVED ALL RESULTS -> {out_csv} ({len(df)} records)")
    
    # Generate summary pivot
    summary = df[df['policy'] == 'calibrated'].groupby(['site', 'n_train', 'arm'])['dispatch_cost_r2'].agg(['mean', 'std']).reset_index()
    print("\n=== SUMMARY TABLE (Calibrated Dispatch Cost r=2) ===")
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "within_site_learning_curve_summary.csv", index=False)

if __name__ == '__main__':
    run_suite()
