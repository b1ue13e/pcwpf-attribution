"""
PCT Empirical Contrast: Official Reverse-JS vs Corrected Standard-JS.
Answering the reviewer challenge:
"Does correcting the JS divergence argument order materially change training or forecasting?"

Loss formulations evaluated:
1. Baseline: MSE loss only (lambda = 0)
2. Official Reverse-JS:
   loss = MSE + lambda * 0.5 * (F.kl_div(pred.log(), m, reduction='batchmean') + F.kl_div(targ.log(), m, reduction='batchmean'))
   (computes KL(m || pred) + KL(m || targ) due to PyTorch F.kl_div(input, target) semantics)
3. Corrected Standard-JS:
   loss = MSE + lambda * 0.5 * (F.kl_div(m.log(), pred, reduction='batchmean') + F.kl_div(m.log(), targ, reduction='batchmean'))
   (computes genuine KL(pred || m) + KL(targ || m))

Sites: Kelmarsh and Penmanshiel
Lambda grid: [0.01, 0.1, 1.0]
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

DATA_DIR = Path(r"e:\论文1\04_数据集\constraint_rebuild_v2")
OUT_DIR = Path(r"e:\论文1\06_实验结果与日志\pct_js_contrast")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]
SITES = ['kelmarsh', 'penmanshiel']
LAMBDAS = [0.01, 0.1, 1.0]

def compute_js_official(pred_dist, targ_dist):
    # Official PCT implementation: F.kl_div(p.log(), m)
    # PyTorch F.kl_div(input, target) = target * (log(target) - input)
    # Passing pred_dist.log() as input and m as target calculates m * (log(m) - log(pred)) = KL(m || pred)
    m = 0.5 * (pred_dist + targ_dist)
    p_log = torch.log(pred_dist.clamp(min=1e-8))
    q_log = torch.log(targ_dist.clamp(min=1e-8))
    m_safe = m.clamp(min=1e-8)
    # Reduction sum over bins, then mean over batch
    kl1 = F.kl_div(p_log, m_safe, reduction='batchmean')
    kl2 = F.kl_div(q_log, m_safe, reduction='batchmean')
    return 0.5 * (kl1 + kl2)

def compute_js_corrected(pred_dist, targ_dist):
    # Corrected standard implementation: F.kl_div(m.log(), p)
    # Passing m.log() as input and pred_dist as target calculates pred * (log(pred) - log(m)) = KL(pred || m)
    m = 0.5 * (pred_dist + targ_dist)
    m_log = torch.log(m.clamp(min=1e-8))
    p_safe = pred_dist.clamp(min=1e-8)
    q_safe = targ_dist.clamp(min=1e-8)
    kl1 = F.kl_div(m_log, p_safe, reduction='batchmean')
    kl2 = F.kl_div(m_log, q_safe, reduction='batchmean')
    return 0.5 * (kl1 + kl2)

def train_pct_variant(model, trainxy, valxy, prior, mode, lam, device, seed, epochs=30, patience=5):
    opt = torch.optim.Adam(model.parameters(), lr=CFG['learning_rate'])
    xt = torch.as_tensor(trainxy[0], device=device)
    yt = torch.as_tensor(trainxy[1], device=device)
    vp = valxy[1]
    best_loss = float('inf')
    best_state = None
    stale = 0
    pri_t = {k: torch.as_tensor(v, device=device) for k, v in prior.items()}
    rng = np.random.default_rng(seed)
    
    grad_norms = []
    pen_values = []

    for epoch in range(epochs):
        model.train()
        for ix in np.array_split(rng.permutation(len(xt)), int(np.ceil(len(xt) / CFG['batch_size']))):
            x = xt[ix]
            y = yt[ix]
            opt.zero_grad()
            pred = model(x)
            mse = F.mse_loss(pred, y)
            
            if mode == 'baseline' or lam == 0.0:
                pen = torch.tensor(0.0, device=device)
            else:
                idx = (x[:, -1, 1] * 15.0).long().clamp(0, len(pri_t['means']) - 1)
                # Compute distribution proxy
                p_dist = torch.exp(-0.5 * ((pred[:, :, None] - pri_t['grid']) / 0.06) ** 2).mean(1) + 1e-8
                p_dist = p_dist / p_dist.sum(1, keepdim=True)
                q_dist = pri_t['hist'][idx] + 1e-8
                q_dist = q_dist / q_dist.sum(1, keepdim=True)
                
                if mode == 'official_reverse':
                    pen = compute_js_official(p_dist, q_dist)
                elif mode == 'corrected_standard':
                    pen = compute_js_corrected(p_dist, q_dist)
                else:
                    raise ValueError(mode)
                    
            loss = mse + lam * pen
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered in epoch {epoch}")
            loss.backward()
            
            # Record grad norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)
            pen_values.append(pen.item())
            
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
    return model, best_loss, float(np.mean(grad_norms)), float(np.mean(pen_values))

def run_suite():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Running PCT JS Empirical Contrast Suite on {device}...")
    
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
        
        # 1. Baseline (lambda=0)
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            model = Forecaster('dlinear', CFG['lookback'], CFG['horizon']).to(device)
            t0 = time.time()
            model, val_loss, mean_gnorm, _ = train_pct_variant(
                model, trainxy, valxy, genuine_prior, 'baseline', 0.0, device, seed
            )
            elapsed = time.time() - t0
            
            cal_pred = np.clip(infer(model, calxy[0], device), 0.0, 1.0)
            test_pred = np.clip(infer(model, testxy[0], device), 0.0, 1.0)
            delta_r2 = np.quantile(y_cal - cal_pred, 1.0 / 3.0, axis=0)
            p_cal = np.clip(test_pred + delta_r2, 0.0, 1.0)
            
            diff = p_cal - y_test
            cost_cal = float(np.mean(2.0 * np.maximum(diff[:, :6], 0.0) + np.maximum(-diff[:, :6], 0.0)))
            rmse = float(np.sqrt(np.mean((p_cal - y_test) ** 2)))
            mae = float(np.mean(np.abs(p_cal - y_test)))
            
            all_results.append({
                'site': site,
                'mode': 'unconstrained_baseline',
                'lambda': 0.0,
                'seed': seed,
                'dispatch_cost_cal': cost_cal,
                'rmse': rmse,
                'mae': mae,
                'val_mse': val_loss,
                'mean_grad_norm': mean_gnorm,
                'time_sec': elapsed
            })
            print(f"[{site:12s} | baseline | s{seed:4d}] Cost: {cost_cal:.5f}, RMSE: {rmse:.4f}")
            
        # 2. Compare Official Reverse-JS vs Corrected Standard-JS across Lambdas
        for lam in LAMBDAS:
            for mode in ['official_reverse', 'corrected_standard']:
                for seed in SEEDS:
                    torch.manual_seed(seed); np.random.seed(seed)
                    model = Forecaster('dlinear', CFG['lookback'], CFG['horizon']).to(device)
                    t0 = time.time()
                    model, val_loss, mean_gnorm, mean_pen = train_pct_variant(
                        model, trainxy, valxy, genuine_prior, mode, lam, device, seed
                    )
                    elapsed = time.time() - t0
                    
                    cal_pred = np.clip(infer(model, calxy[0], device), 0.0, 1.0)
                    test_pred = np.clip(infer(model, testxy[0], device), 0.0, 1.0)
                    delta_r2 = np.quantile(y_cal - cal_pred, 1.0 / 3.0, axis=0)
                    p_cal = np.clip(test_pred + delta_r2, 0.0, 1.0)
                    
                    diff = p_cal - y_test
                    cost_cal = float(np.mean(2.0 * np.maximum(diff[:, :6], 0.0) + np.maximum(-diff[:, :6], 0.0)))
                    rmse = float(np.sqrt(np.mean((p_cal - y_test) ** 2)))
                    mae = float(np.mean(np.abs(p_cal - y_test)))
                    
                    all_results.append({
                        'site': site,
                        'mode': mode,
                        'lambda': lam,
                        'seed': seed,
                        'dispatch_cost_cal': cost_cal,
                        'rmse': rmse,
                        'mae': mae,
                        'val_mse': val_loss,
                        'mean_grad_norm': mean_gnorm,
                        'mean_penalty_value': mean_pen,
                        'time_sec': elapsed
                    })
                    print(f"[{site:12s} | {mode:18s} | l={lam:.2f} | s{seed:4d}] Cost: {cost_cal:.5f}, RMSE: {rmse:.4f}, GradNorm: {mean_gnorm:.4f}")

    df = pd.DataFrame(all_results)
    out_csv = OUT_DIR / "pct_js_contrast_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSAVED ALL PCT RESULTS -> {out_csv} ({len(df)} records)")
    
    summary = df.groupby(['site', 'mode', 'lambda'])[['dispatch_cost_cal', 'rmse', 'mean_grad_norm', 'val_mse']].agg(['mean', 'std']).reset_index()
    summary.to_csv(OUT_DIR / "pct_js_contrast_summary.csv", index=False)
    print("\n=== SUMMARY TABLE ===")
    print(summary.to_string())

if __name__ == '__main__':
    run_suite()
