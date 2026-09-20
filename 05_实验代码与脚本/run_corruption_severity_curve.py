"""
Realistic Prior Corruption and Engineering Drift Severity Benchmark:
Bridging the gap between binary scrambled prior (+68.1%) and real-world degradation.

Evaluates two complementary dimensions of prior misspecification:
1. Continuous corruption severity: alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
   where P_alpha(v) = (1 - alpha) * P_genuine(v) + alpha * P_scrambled(v)
2. Realistic engineering drift:
   - Turbine aerodynamic aging derating: scale in [1.0, 0.95, 0.90, 0.85, 0.80]
   - Anemometer calibration shift / sensor bias: wind speed scaling in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]

Sites:
- Penmanshiel (N_train = 1,583, sample-restricted regime)
- Kelmarsh (N_train = 5,098, well-sampled regime)

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
OUT_DIR = Path(r"e:\论文1\06_实验结果与日志\corruption_severity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]
SITES = ['penmanshiel', 'kelmarsh']
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
AGING_SCALES = [1.0, 0.95, 0.90, 0.85, 0.80]
SENSOR_BIASES = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]

def train_model(model, trainxy, valxy, prior, family, lam, factor, device, seed, epochs=30, patience=5):
    opt = torch.optim.Adam(model.parameters(), lr=CFG['learning_rate'])
    xt = torch.as_tensor(trainxy[0], device=device)
    yt = torch.as_tensor(trainxy[1], device=device)
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
    print(f"Running Corruption Severity & Drift Suite on {device}...")
    
    all_results = []
    
    for site in SITES:
        print(f"\n==================== SITE: {site.upper()} ====================")
        f = pd.read_csv(DATA_DIR / f"{site}.csv", index_col='time', parse_dates=True)
        f.index = pd.to_datetime(f.index, utc=True)
        split = CFG['splits'][site]
        
        x, y, ids = starts(f, split)
        trainxy = windows(x, y, ids['train'])
        valxy = windows(x, y, ids['val'])
        calxy = windows(x, y, ids['cal'])
        testxy = windows(x, y, ids['test'])
        
        y_test = testxy[1]
        y_cal = calxy[1]
        
        genuine_prior = fit_prior(f, split['train_end'])
        
        # 1. Continuous Corruption Gradient: alpha in [0.0, 1.0]
        print("\n--- Part 1: Continuous Corruption Gradient alpha in [0.0, 1.0] ---")
        for alpha in ALPHAS:
            for seed in SEEDS:
                scrambled_prior = fit_scrambled_prior(genuine_prior, seed=seed)
                
                # Interpolate prior means and hist
                interp_prior = copy.deepcopy(genuine_prior)
                interp_prior['means'] = (1.0 - alpha) * genuine_prior['means'] + alpha * scrambled_prior['means']
                interp_prior['hist'] = (1.0 - alpha) * genuine_prior['hist'] + alpha * scrambled_prior['hist']
                
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = Forecaster('dlinear', CFG['lookback'], CFG['horizon']).to(device)
                
                t0 = time.time()
                model, _ = train_model(model, trainxy, valxy, interp_prior, 'curve', 1.0, 1.0, device, seed)
                elapsed = time.time() - t0
                
                cal_pred = np.clip(infer(model, calxy[0], device), 0.0, 1.0)
                test_pred = np.clip(infer(model, testxy[0], device), 0.0, 1.0)
                
                delta_r2 = np.quantile(y_cal - cal_pred, 1.0 / 3.0, axis=0)
                p_cal = np.clip(test_pred + delta_r2, 0.0, 1.0)
                
                diff = p_cal - y_test
                cost_cal = float(np.mean(2.0 * np.maximum(diff[:, :6], 0.0) + np.maximum(-diff[:, :6], 0.0)))
                rmse = float(np.sqrt(np.mean((p_cal - y_test) ** 2)))
                
                all_results.append({
                    'site': site,
                    'experiment': 'continuous_alpha',
                    'parameter': alpha,
                    'seed': seed,
                    'dispatch_cost_cal': cost_cal,
                    'rmse': rmse,
                    'time_sec': elapsed
                })
                print(f"[{site:12s} | alpha={alpha:.1f} | s{seed:4d}] Cost: {cost_cal:.5f}, RMSE: {rmse:.4f}")
                
        # 2. Engineering Drift: Turbine Aging Derating
        print("\n--- Part 2: Turbine Aging Capacity Derating ---")
        for scale in AGING_SCALES:
            for seed in SEEDS:
                # Prior assuming scaled rated power (e.g. derated due to blade wear)
                aged_prior = copy.deepcopy(genuine_prior)
                aged_prior['means'] = genuine_prior['means'] * scale
                
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = Forecaster('dlinear', CFG['lookback'], CFG['horizon']).to(device)
                
                t0 = time.time()
                model, _ = train_model(model, trainxy, valxy, aged_prior, 'curve', 1.0, 1.0, device, seed)
                elapsed = time.time() - t0
                
                cal_pred = np.clip(infer(model, calxy[0], device), 0.0, 1.0)
                test_pred = np.clip(infer(model, testxy[0], device), 0.0, 1.0)
                
                delta_r2 = np.quantile(y_cal - cal_pred, 1.0 / 3.0, axis=0)
                p_cal = np.clip(test_pred + delta_r2, 0.0, 1.0)
                
                diff = p_cal - y_test
                cost_cal = float(np.mean(2.0 * np.maximum(diff[:, :6], 0.0) + np.maximum(-diff[:, :6], 0.0)))
                rmse = float(np.sqrt(np.mean((p_cal - y_test) ** 2)))
                
                all_results.append({
                    'site': site,
                    'experiment': 'turbine_aging',
                    'parameter': scale,
                    'seed': seed,
                    'dispatch_cost_cal': cost_cal,
                    'rmse': rmse,
                    'time_sec': elapsed
                })
                print(f"[{site:12s} | aging={scale:.2f} | s{seed:4d}] Cost: {cost_cal:.5f}, RMSE: {rmse:.4f}")
                
        # 3. Engineering Drift: Anemometer Sensor Calibration Shift
        print("\n--- Part 3: Sensor Calibration Shift ---")
        for bias in SENSOR_BIASES:
            for seed in SEEDS:
                # Anemometer calibration shift: wind speed shifted by bias factor
                biased_prior = copy.deepcopy(genuine_prior)
                # Curve evaluated with shifted wind speed mapping
                # i.e., prior centers or means shifted
                shifted_means = np.interp(
                    genuine_prior['centers'] * bias,
                    genuine_prior['centers'],
                    genuine_prior['means']
                ).astype(np.float32)
                biased_prior['means'] = shifted_means
                
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = Forecaster('dlinear', CFG['lookback'], CFG['horizon']).to(device)
                
                t0 = time.time()
                model, _ = train_model(model, trainxy, valxy, biased_prior, 'curve', 1.0, 1.0, device, seed)
                elapsed = time.time() - t0
                
                cal_pred = np.clip(infer(model, calxy[0], device), 0.0, 1.0)
                test_pred = np.clip(infer(model, testxy[0], device), 0.0, 1.0)
                
                delta_r2 = np.quantile(y_cal - cal_pred, 1.0 / 3.0, axis=0)
                p_cal = np.clip(test_pred + delta_r2, 0.0, 1.0)
                
                diff = p_cal - y_test
                cost_cal = float(np.mean(2.0 * np.maximum(diff[:, :6], 0.0) + np.maximum(-diff[:, :6], 0.0)))
                rmse = float(np.sqrt(np.mean((p_cal - y_test) ** 2)))
                
                all_results.append({
                    'site': site,
                    'experiment': 'sensor_bias',
                    'parameter': bias,
                    'seed': seed,
                    'dispatch_cost_cal': cost_cal,
                    'rmse': rmse,
                    'time_sec': elapsed
                })
                print(f"[{site:12s} | sensor_bias={bias:.2f} | s{seed:4d}] Cost: {cost_cal:.5f}, RMSE: {rmse:.4f}")

    df = pd.DataFrame(all_results)
    out_csv = OUT_DIR / "corruption_severity_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSAVED ALL CORRUPTION RESULTS -> {out_csv} ({len(df)} records)")
    
    summary = df.groupby(['site', 'experiment', 'parameter'])['dispatch_cost_cal'].agg(['mean', 'std']).reset_index()
    summary.to_csv(OUT_DIR / "corruption_severity_summary.csv", index=False)
    print("\nSummary saved to corruption_severity_summary.csv")

if __name__ == '__main__':
    run_suite()
