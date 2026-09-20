"""
evaluate_dynamic_price.py
=========================
Audit and evaluation of wind power dispatch models under real-world dynamic
electricity balancing settlement prices (UK Elexon 2017 System Buy Price).

Produces:
1. dynamic_price_settlement_table2.csv: Table 2 comprehensive performance metrics
2. dynamic_price_paired_intervals.csv: 7-day circular block paired bootstrap confidence intervals
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

# Reconfigure stdout for UTF-8 compatibility on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure local modules can be imported
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from core import CFG, Forecaster, infer, starts, windows

FARM_SPECS = {
    'hill_of_towie': {
        'name': 'Hill of Towie',
        'rated_kw': 48300.0,
        'turbines': 21,
        'turbine_model': 'Siemens SWT-2.3-82',
        'site_type': 'external_blind_test'
    },
    'kelmarsh': {
        'name': 'Kelmarsh',
        'rated_kw': 12300.0,
        'turbines': 6,
        'turbine_model': 'Senvion MM82',
        'site_type': 'core_development'
    }
}

METHODS = [
    'none_f1.0_plain',
    'bounds_f1.0_plain',
    'curve_f1.0_plain',
    'distribution_f1.0_plain',
    'ridge',
    'persistence'
]

METHOD_PRETTY_NAMES = {
    'none_f1.0_plain': 'Unconstrained DLinear',
    'bounds_f1.0_plain': 'Bounds Constrained DLinear (lambda=0.01)',
    'curve_f1.0_plain': 'Power Curve Constrained DLinear (lambda=0.01)',
    'distribution_f1.0_plain': 'Distribution Prior DLinear (lambda=0.01)',
    'ridge': 'Ridge Regression',
    'persistence': 'Persistence Baseline'
}

def load_elexon_prices(elexon_path):
    print(f"Loading Elexon settlement prices from {elexon_path}...")
    df_elexon = pd.read_csv(elexon_path)
    df_elexon['startTime'] = pd.to_datetime(df_elexon['startTime'], utc=True)
    price_map = dict(zip(df_elexon['startTime'], df_elexon['systemBuyPrice']))
    print(f"  Loaded {len(price_map):,} Elexon price records (range: {df_elexon['startTime'].min()} to {df_elexon['startTime'].max()})")
    return price_map

def run_evaluation(root, data, elexon_path, out_dir, sites=None, bootstrap_reps=2000, seed_boot=20907):
    root = Path(root)
    data = Path(data)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    price_map = load_elexon_prices(elexon_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Inference device: {device}")
    
    if sites is None:
        sites = ['hill_of_towie', 'kelmarsh']
        
    all_table_records = []
    all_bootstrap_records = []
    
    for site in sites:
        print(f"\n=======================================================")
        print(f"Evaluating Site: {site} ({FARM_SPECS[site]['name']})")
        print(f"=======================================================")
        
        scada_file = data / f"{site}.csv"
        f = pd.read_csv(scada_file, index_col='time', parse_dates=True)
        f.index = pd.to_datetime(f.index, utc=True)
        
        x, y, ids = starts(f, CFG['splits'][site])
        xt, yt = windows(x, y, ids['train'])
        xx, yy = windows(x, y, ids['test'])
        times = f.index[ids['test']]
        n_test = len(ids['test'])
        print(f"  Test origins: {n_test:,} (delivery horizon: next-hour h=1..6)")
        
        # Precompute prices matrix (n_test, 6)
        prices = np.zeros((n_test, 6), dtype=np.float64)
        missing_price_count = 0
        for idx_i, origin_idx in enumerate(ids['test']):
            for h in range(6):
                target_time = f.index[origin_idx + h]
                interval_start = target_time - pd.Timedelta(minutes=10)
                elexon_start = interval_start.floor('30min')
                if elexon_start in price_map:
                    prices[idx_i, h] = price_map[elexon_start]
                else:
                    missing_price_count += 1
                    
        assert missing_price_count == 0, f"Error: {missing_price_count} test points failed Elexon alignment!"
        mean_price = prices.mean()
        weights = prices / mean_price
        print(f"  Elexon prices matched: mean = {mean_price:.4f} GBP/MWh (min = {prices.min():.2f}, max = {prices.max():.2f})")
        
        # Load Ridge model
        ridge_sel = json.loads((root / 'ridge_selection.json').read_text())
        alpha = ridge_sel[site]
        ridge = Ridge(alpha=alpha).fit(xt.reshape(len(xt), -1), yt)
        
        # Load calibration deltas for ratio=2
        cal_path = root / site / 'assessment' / 'calibration.json'
        cal_data = json.loads(cal_path.read_text())
        cal_map = {(c['seed'], c['method'], c['ratio']): np.array(c['delta_by_lead']) for c in cal_data}
        
        # Build timeline dataframe for true Settlement Period aggregation
        step_records = []
        for origin_idx in ids['test']:
            for h in range(6):
                target_idx = origin_idx + h
                t_step = f.index[target_idx]
                step_records.append((origin_idx, h, target_idx, t_step))
        df_timeline_site = pd.DataFrame(step_records, columns=['origin_idx', 'h', 'target_idx', 't_step'])
        df_timeline_site['interval_start'] = df_timeline_site['t_step'] - pd.Timedelta(minutes=10)
        df_timeline_site['sp_start'] = df_timeline_site['interval_start'].dt.floor('30min')
        df_timeline_site['price_sbp'] = df_timeline_site['sp_start'].map(price_map)
        df_timeline_site['y'] = yy[:, :6].reshape(-1)

        daily_list = []
        period_list = []
        
        p_rated = FARM_SPECS[site]['rated_kw']
        n_turb = FARM_SPECS[site]['turbines']
        
        for seed in CFG['seed_list']:
            # Determine which models to load
            if site in CFG['development_sites']:
                t = pd.read_csv(root / 'selection.csv')
                chosen = t[(t.site == site) & (t.seed == seed)].to_dict('records')
            else:
                chosen = json.loads((root / 'external_selection.json').read_text())
                for r in chosen:
                    r['tag'] = f"{r['family']}_l{r['lambda']}_f{r['factor']}_{r['control']}"
                    
            predictors = {
                'ridge': ridge.predict(xx.reshape(len(xx), -1)),
                'persistence': np.repeat(xx[:, -1, 0, None], CFG['horizon'], axis=1)
            }
            tag_map = {'ridge': 'ridge', 'persistence': 'persistence'}
            
            for r in chosen:
                group = f"{r['family']}_f{r['factor']}_{r['control']}"
                if group in METHODS:
                    m = Forecaster('dlinear', CFG['lookback'], CFG['horizon']).to(device)
                    ckpt = root / site / 'dlinear' / f'seed{seed}' / f"{r['tag']}.pt"
                    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
                    predictors[group] = infer(m, xx, device)
                    tag_map[group] = r['tag']
                    
            for group in METHODS:
                pred = predictors[group]
                tag = tag_map[group]
                base = np.clip(pred[:, :6], 0, 1)
                delta = cal_map[(seed, tag, 2)][:6]
                
                for policy in ['uncalibrated', 'quantile_calibrated']:
                    p = base if policy == 'uncalibrated' else np.clip(base + delta, 0, 1)
                    y_eval = yy[:, :6]
                    e = p - y_eval
                    
                    # Metric A: Price-Weighted Normalized Cost (Dimensionless, mean over 6 steps)
                    cost_dyn_steps = weights * (2.0 * np.maximum(e, 0) + np.maximum(-e, 0))
                    cost_dyn = cost_dyn_steps.mean(axis=1) # (n_test,)
                    
                    # Static unweighted cost (Table 1 cost)
                    cost_static_steps = 2.0 * np.maximum(e, 0) + np.maximum(-e, 0)
                    cost_static = cost_static_steps.mean(axis=1)
                    
                    df_origins = pd.DataFrame({
                        'day': times.strftime('%Y-%m-%d'),
                        'quarter': times.quarter,
                        'cost_dyn': cost_dyn,
                        'cost_static': cost_static,
                        'sq_err': (e ** 2).mean(axis=1)
                    })
                    
                    # Metric B & C: True BSC 30-minute netted shortfall cashout & balancing cashflow
                    # Under UK Elexon BSC settlement rules (P305), cashout is settled per 30-minute Settlement Period (SP).
                    # Map unique delivery intervals to sp_start = (target_end - 10min).floor('30min').
                    df_timeline = df_timeline_site.copy()
                    df_timeline['p'] = p.reshape(-1)
                    df_timeline['e'] = df_timeline['p'] - df_timeline['y']
                    
                    sp_agg = df_timeline.groupby('sp_start').agg(
                        e_sum=('e', 'sum'),
                        n_steps=('e', 'count'),
                        price=('price_sbp', 'first')
                    ).reset_index()
                    sp_agg['day'] = sp_agg['sp_start'].dt.strftime('%Y-%m-%d')
                    sp_agg['quarter'] = sp_agg['sp_start'].dt.quarter
                    
                    # Filter to complete 3-step SPs (3x10min = 30min intact delivery)
                    sp_intact = sp_agg[sp_agg['n_steps'] == 3].copy()
                    sp_intact['e_mwh'] = sp_intact['e_sum'] * (p_rated / 6000.0)
                    sp_intact['cashout_gbp'] = sp_intact['price'] * np.maximum(sp_intact['e_mwh'], 0)
                    sp_intact['surplus_gbp'] = sp_intact['price'] * np.maximum(-sp_intact['e_mwh'], 0)
                    sp_intact['net_cashflow_gbp'] = sp_intact['surplus_gbp'] - sp_intact['cashout_gbp']
                    sp_intact['cashout_per_turb'] = sp_intact['cashout_gbp'] / n_turb
                    sp_intact['net_cashflow_per_turb'] = sp_intact['net_cashflow_gbp'] / n_turb
                    
                    # Daily SP aggregation
                    sp_day = sp_intact.groupby('day').agg(
                        cashout_daily_sum_gbp=('cashout_gbp', 'sum'),
                        cashout_daily_per_turb=('cashout_per_turb', 'sum'),
                        net_cashflow_daily_sum_gbp=('net_cashflow_gbp', 'sum'),
                        net_cashflow_daily_per_turb=('net_cashflow_per_turb', 'sum'),
                        n_sps=('sp_start', 'count')
                    ).reset_index()
                    
                    orig_day = df_origins.groupby('day').agg(
                        cost_dyn=('cost_dyn', 'mean'),
                        cost_static=('cost_static', 'mean'),
                        n_origins=('cost_dyn', 'count')
                    ).reset_index()
                    
                    d_agg = pd.merge(orig_day, sp_day, on='day', how='inner')
                    d_agg['quarter'] = pd.to_datetime(d_agg['day']).dt.quarter
                    
                    for k, v in {'site': site, 'seed': seed, 'method': group, 'policy': policy}.items():
                        d_agg[k] = v
                    daily_list.append(d_agg)
                    
                    # Period aggregations (all, Q1, Q2, Q3, Q4)
                    periods = [('all', np.ones(len(df_origins), dtype=bool), np.ones(len(sp_intact), dtype=bool), np.ones(len(d_agg), dtype=bool))] + [
                        (f'Q{q}', df_origins['quarter'] == q, sp_intact['quarter'] == q, d_agg['quarter'] == q) for q in [1, 2, 3, 4]
                    ]
                    
                    for period_name, mask_orig, mask_sp, mask_day in periods:
                        if not mask_orig.any() or not mask_sp.any():
                            continue
                        sub_origins = mask_orig.sum()
                        sub_cost_dyn = df_origins[mask_orig]['cost_dyn'].mean()
                        sub_cost_static = df_origins[mask_orig]['cost_static'].mean()
                        sub_rmse = np.sqrt(df_origins[mask_orig]['sq_err'].mean())
                        
                        sub_cashout_total = sp_intact[mask_sp]['cashout_gbp'].sum()
                        sub_cashout_daily_mean = d_agg[mask_day]['cashout_daily_sum_gbp'].mean()
                        sub_cashout_per_turb_yr = d_agg[mask_day]['cashout_daily_per_turb'].mean() * 365.0
                        
                        sub_net_total = sp_intact[mask_sp]['net_cashflow_gbp'].sum()
                        sub_net_daily_mean = d_agg[mask_day]['net_cashflow_daily_sum_gbp'].mean()
                        sub_net_per_turb_yr = d_agg[mask_day]['net_cashflow_daily_per_turb'].mean() * 365.0
                        
                        period_list.append({
                            'site': site,
                            'seed': seed,
                            'method': group,
                            'policy': policy,
                            'period': period_name,
                            'n_origins': int(sub_origins),
                            'rmse_pu': float(sub_rmse),
                            'cost_dyn': float(sub_cost_dyn),
                            'cost_static': float(sub_cost_static),
                            'cashout_total_gbp': float(sub_cashout_total),
                            'cashout_daily_gbp': float(sub_cashout_daily_mean),
                            'cashout_per_turb_yr_gbp': float(sub_cashout_per_turb_yr),
                            'net_cashflow_total_gbp': float(sub_net_total),
                            'net_cashflow_daily_gbp': float(sub_net_daily_mean),
                            'net_cashflow_per_turb_yr_gbp': float(sub_net_per_turb_yr)
                        })

        df_period = pd.DataFrame(period_list)
        
        # Aggregate across 5 seeds for Table 2
        agg_funcs = {
            'rmse_pu': ['mean', 'std'],
            'cost_dyn': ['mean', 'std'],
            'cost_static': ['mean', 'std'],
            'cashout_total_gbp': ['mean', 'std'],
            'cashout_daily_gbp': ['mean', 'std'],
            'cashout_per_turb_yr_gbp': ['mean', 'std'],
            'net_cashflow_total_gbp': ['mean', 'std'],
            'net_cashflow_daily_gbp': ['mean', 'std'],
            'net_cashflow_per_turb_yr_gbp': ['mean', 'std']
        }
        
        summary_site = df_period.groupby(['site', 'method', 'policy', 'period']).agg(agg_funcs).reset_index()
        summary_site.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c for c in summary_site.columns]
        
        # Add metadata
        summary_site['rated_kw'] = p_rated
        summary_site['turbines'] = n_turb
        summary_site['method_name'] = summary_site['method'].map(METHOD_PRETTY_NAMES)
        all_table_records.append(summary_site)
        
        # -------------------------------------------------------------
        # Bootstrap Inference: 7-day Circular Block Paired Bootstrap
        # -------------------------------------------------------------
        df_daily_site = pd.concat(daily_list, ignore_index=True)
        seeds = sorted(df_daily_site.seed.unique())
        calendar = pd.date_range(df_daily_site.day.min(), df_daily_site.day.max(), freq='D').strftime('%Y-%m-%d')
        N = len(calendar)
        scored_days = int(df_daily_site.day.nunique())
        scale_annual = 365.0 / scored_days
        
        def pivot_array(sub_df, val_col):
            return sub_df.pivot(index='seed', columns='day', values=val_col).reindex(index=seeds, columns=calendar).fillna(0).to_numpy()
            
        print(f"  Running {bootstrap_reps:,} paired circular block bootstrap iterations (block length = 7 days, calendar days = {N}, scored days = {scored_days})...")
        rng = np.random.default_rng(seed_boot)
        ss = rng.integers(0, len(seeds), (bootstrap_reps, len(seeds)))
        begins = rng.integers(0, N, (bootstrap_reps, int(np.ceil(N / 7.0))))
        ii = ((begins[:, :, None] + np.arange(7)) % N).reshape(bootstrap_reps, -1)[:, :N]
        
        CONSTRAINT_METHODS = ['bounds_f1.0_plain', 'curve_f1.0_plain', 'distribution_f1.0_plain']

        # 1. Within-policy contrasts vs Unconstrained DLinear
        for policy in ['uncalibrated', 'quantile_calibrated']:
            q = df_daily_site[df_daily_site.policy == policy]
            base = q[q.method == 'none_f1.0_plain']
            w = pivot_array(base, 'n_origins')
            base_cost_dyn = pivot_array(base, 'cost_dyn')
            base_cost_static = pivot_array(base, 'cost_static')
            base_cashout_turb = pivot_array(base, 'cashout_daily_per_turb')
            base_net_turb = pivot_array(base, 'net_cashflow_daily_per_turb')
            
            ww = w[ss[:, :, None], ii[:, None, :]]
            ww_sum = ww.sum((1, 2))
            ww_days = (w > 0).astype(np.float64)[ss[:, :, None], ii[:, None, :]]
            ww_days_sum = ww_days.sum(axis=2) # (R, len(seeds))
            
            for method in ['bounds_f1.0_plain', 'curve_f1.0_plain', 'distribution_f1.0_plain', 'ridge', 'persistence']:
                a = q[q.method == method]
                delta_cost_dyn = pivot_array(a, 'cost_dyn') - base_cost_dyn
                delta_cost_static = pivot_array(a, 'cost_static') - base_cost_static
                delta_cashout_turb = pivot_array(a, 'cashout_daily_per_turb') - base_cashout_turb
                delta_net_turb = pivot_array(a, 'net_cashflow_daily_per_turb') - base_net_turb
                
                # Bootstrap samples
                dd_dyn = delta_cost_dyn[ss[:, :, None], ii[:, None, :]]
                samples_dyn = (dd_dyn * ww).sum((1, 2)) / ww_sum
                
                dd_static = delta_cost_static[ss[:, :, None], ii[:, None, :]]
                samples_static = (dd_static * ww).sum((1, 2)) / ww_sum
                
                # Annualized cashout difference: daily mean in resample * 365.0
                dd_cash = delta_cashout_turb[ss[:, :, None], ii[:, None, :]]
                samples_cash = ((dd_cash.sum(axis=2) / ww_days_sum) * 365.0).mean(axis=1) # (R,)
                
                dd_net = delta_net_turb[ss[:, :, None], ii[:, None, :]]
                samples_net = ((dd_net.sum(axis=2) / ww_days_sum) * 365.0).mean(axis=1) # (R,)
                
                pt_cost_dyn = float((delta_cost_dyn * w).sum() / w.sum())
                pt_cost_static = float((delta_cost_static * w).sum() / w.sum())
                pt_cash = float(delta_cashout_turb.sum(axis=1).mean() * scale_annual)
                pt_net = float(delta_net_turb.sum(axis=1).mean() * scale_annual)
                
                is_constraint = method in CONSTRAINT_METHODS
                all_bootstrap_records.append({
                    'site': site,
                    'contrast_type': 'vs_unconstrained_dlinear_same_policy',
                    'method': method,
                    'method_name': METHOD_PRETTY_NAMES[method],
                    'policy': policy,
                    'reference': 'none_f1.0_plain',
                    # Metric A: Dynamic Price Weighted Cost
                    'delta_cost_dyn': pt_cost_dyn,
                    'ci95_cost_dyn_low': float(np.quantile(samples_dyn, 0.025)),
                    'ci95_cost_dyn_high': float(np.quantile(samples_dyn, 0.975)),
                    'fwer_cost_dyn_low': float(np.quantile(samples_dyn, 0.05 / 6.0)) if is_constraint else np.nan,
                    'fwer_cost_dyn_high': float(np.quantile(samples_dyn, 1.0 - 0.05 / 6.0)) if is_constraint else np.nan,
                    # Static Cost (Table 1 homogeneous)
                    'delta_cost_static': pt_cost_static,
                    'ci95_cost_static_low': float(np.quantile(samples_static, 0.025)),
                    'ci95_cost_static_high': float(np.quantile(samples_static, 0.975)),
                    # Metric B: Shortfall Cashout in GBP (£/turb-yr)
                    'delta_cashout_per_turb_yr_gbp': pt_cash,
                    'ci95_cashout_low': float(np.quantile(samples_cash, 0.025)),
                    'ci95_cashout_high': float(np.quantile(samples_cash, 0.975)),
                    'farm_annual_delta_cashout_gbp': pt_cash * n_turb,
                    'ci95_farm_cashout_low': float(np.quantile(samples_cash, 0.025)) * n_turb,
                    'ci95_farm_cashout_high': float(np.quantile(samples_cash, 0.975)) * n_turb,
                    # Metric C: Two-Sided Full Market Net Cashflow in GBP (£/turb-yr)
                    'delta_net_cashflow_per_turb_yr_gbp': pt_net,
                    'ci95_net_cashflow_low': float(np.quantile(samples_net, 0.025)),
                    'ci95_net_cashflow_high': float(np.quantile(samples_net, 0.975)),
                    'farm_annual_delta_net_cashflow_gbp': pt_net * n_turb,
                    'ci95_farm_net_cashflow_low': float(np.quantile(samples_net, 0.025)) * n_turb,
                    'ci95_farm_net_cashflow_high': float(np.quantile(samples_net, 0.975)) * n_turb,
                    'calendar_days': N,
                    'scored_days': scored_days,
                    'seeds': len(seeds),
                    'interpretation': 'For cost/cashout: negative favours method over unconstrained DLinear. For net cashflow: positive favours method over unconstrained DLinear.'
                })

        # 2. Calibration Gain Contrast (Quantile Calibrated minus Uncalibrated for EACH method)
        for method in METHODS:
            q_uncal = df_daily_site[(df_daily_site.policy == 'uncalibrated') & (df_daily_site.method == method)]
            q_cal = df_daily_site[(df_daily_site.policy == 'quantile_calibrated') & (df_daily_site.method == method)]
            
            w = pivot_array(q_uncal, 'n_origins')
            cost_uncal = pivot_array(q_uncal, 'cost_dyn')
            cost_cal = pivot_array(q_cal, 'cost_dyn')
            cost_uncal_static = pivot_array(q_uncal, 'cost_static')
            cost_cal_static = pivot_array(q_cal, 'cost_static')
            cash_uncal = pivot_array(q_uncal, 'cashout_daily_per_turb')
            cash_cal = pivot_array(q_cal, 'cashout_daily_per_turb')
            net_uncal = pivot_array(q_uncal, 'net_cashflow_daily_per_turb')
            net_cal = pivot_array(q_cal, 'net_cashflow_daily_per_turb')
            
            delta_cal_cost = cost_cal - cost_uncal
            delta_cal_static = cost_cal_static - cost_uncal_static
            delta_cal_cash = cash_cal - cash_uncal
            delta_cal_net = net_cal - net_uncal
            
            ww = w[ss[:, :, None], ii[:, None, :]]
            ww_sum = ww.sum((1, 2))
            ww_days = (w > 0).astype(np.float64)[ss[:, :, None], ii[:, None, :]]
            ww_days_sum = ww_days.sum(axis=2)
            
            dd_c = delta_cal_cost[ss[:, :, None], ii[:, None, :]]
            samples_cal_cost = (dd_c * ww).sum((1, 2)) / ww_sum
            
            dd_c_static = delta_cal_static[ss[:, :, None], ii[:, None, :]]
            samples_cal_static = (dd_c_static * ww).sum((1, 2)) / ww_sum
            
            dd_k = delta_cal_cash[ss[:, :, None], ii[:, None, :]]
            samples_cal_cash = ((dd_k.sum(axis=2) / ww_days_sum) * 365.0).mean(axis=1)
            
            dd_net = delta_cal_net[ss[:, :, None], ii[:, None, :]]
            samples_cal_net = ((dd_net.sum(axis=2) / ww_days_sum) * 365.0).mean(axis=1)
            
            pt_cal_cost = float((delta_cal_cost * w).sum() / w.sum())
            pt_cal_static = float((delta_cal_static * w).sum() / w.sum())
            pt_cal_cash = float(delta_cal_cash.sum(axis=1).mean() * scale_annual)
            pt_cal_net = float(delta_cal_net.sum(axis=1).mean() * scale_annual)
            
            all_bootstrap_records.append({
                'site': site,
                'contrast_type': 'calibration_gain_calibrated_minus_uncalibrated',
                'method': method,
                'method_name': METHOD_PRETTY_NAMES[method],
                'policy': 'quantile_calibrated_vs_uncalibrated',
                'reference': f'{method} (uncalibrated)',
                'delta_cost_dyn': pt_cal_cost,
                'ci95_cost_dyn_low': float(np.quantile(samples_cal_cost, 0.025)),
                'ci95_cost_dyn_high': float(np.quantile(samples_cal_cost, 0.975)),
                'fwer_cost_dyn_low': np.nan,
                'fwer_cost_dyn_high': np.nan,
                'delta_cost_static': pt_cal_static,
                'ci95_cost_static_low': float(np.quantile(samples_cal_static, 0.025)),
                'ci95_cost_static_high': float(np.quantile(samples_cal_static, 0.975)),
                'delta_cashout_per_turb_yr_gbp': pt_cal_cash,
                'ci95_cashout_low': float(np.quantile(samples_cal_cash, 0.025)),
                'ci95_cashout_high': float(np.quantile(samples_cal_cash, 0.975)),
                'farm_annual_delta_cashout_gbp': pt_cal_cash * n_turb,
                'ci95_farm_cashout_low': float(np.quantile(samples_cal_cash, 0.025)) * n_turb,
                'ci95_farm_cashout_high': float(np.quantile(samples_cal_cash, 0.975)) * n_turb,
                'delta_net_cashflow_per_turb_yr_gbp': pt_cal_net,
                'ci95_net_cashflow_low': float(np.quantile(samples_cal_net, 0.025)),
                'ci95_net_cashflow_high': float(np.quantile(samples_cal_net, 0.975)),
                'farm_annual_delta_net_cashflow_gbp': pt_cal_net * n_turb,
                'ci95_farm_net_cashflow_low': float(np.quantile(samples_cal_net, 0.025)) * n_turb,
                'ci95_farm_net_cashflow_high': float(np.quantile(samples_cal_net, 0.975)) * n_turb,
                'calendar_days': N,
                'scored_days': scored_days,
                'seeds': len(seeds),
                'interpretation': 'Negative indicates cost reduction (economic savings) achieved by offline quantile calibration.'
            })

        # 3. Cross-Paradigm Inversion Contrast:
        # Unconstrained Calibrated DLinear vs Uncalibrated Physical Constraints
        q_dlinear_cal = df_daily_site[(df_daily_site.policy == 'quantile_calibrated') & (df_daily_site.method == 'none_f1.0_plain')]
        w = pivot_array(q_dlinear_cal, 'n_origins')
        ww = w[ss[:, :, None], ii[:, None, :]]
        ww_sum = ww.sum((1, 2))
        ww_days = (w > 0).astype(np.float64)[ss[:, :, None], ii[:, None, :]]
        ww_days_sum = ww_days.sum(axis=2)
        cost_dlinear_cal = pivot_array(q_dlinear_cal, 'cost_dyn')
        cost_dlinear_cal_static = pivot_array(q_dlinear_cal, 'cost_static')
        cash_dlinear_cal = pivot_array(q_dlinear_cal, 'cashout_daily_per_turb')
        net_dlinear_cal = pivot_array(q_dlinear_cal, 'net_cashflow_daily_per_turb')
        
        for phys_method in ['distribution_f1.0_plain', 'curve_f1.0_plain', 'bounds_f1.0_plain']:
            q_phys_uncal = df_daily_site[(df_daily_site.policy == 'uncalibrated') & (df_daily_site.method == phys_method)]
            cost_phys_uncal = pivot_array(q_phys_uncal, 'cost_dyn')
            cost_phys_uncal_static = pivot_array(q_phys_uncal, 'cost_static')
            cash_phys_uncal = pivot_array(q_phys_uncal, 'cashout_daily_per_turb')
            net_phys_uncal = pivot_array(q_phys_uncal, 'net_cashflow_daily_per_turb')
            
            # Contrast: DLinear Calibrated minus Physical Uncalibrated
            # Negative means calibrated DLinear is cheaper than uncalibrated physical constraint!
            delta_inv_cost = cost_dlinear_cal - cost_phys_uncal
            delta_inv_static = cost_dlinear_cal_static - cost_phys_uncal_static
            delta_inv_cash = cash_dlinear_cal - cash_phys_uncal
            delta_inv_net = net_dlinear_cal - net_phys_uncal
            
            dd_ic = delta_inv_cost[ss[:, :, None], ii[:, None, :]]
            samples_inv_cost = (dd_ic * ww).sum((1, 2)) / ww_sum
            
            dd_ic_static = delta_inv_static[ss[:, :, None], ii[:, None, :]]
            samples_inv_static = (dd_ic_static * ww).sum((1, 2)) / ww_sum
            
            dd_ik = delta_inv_cash[ss[:, :, None], ii[:, None, :]]
            samples_inv_cash = ((dd_ik.sum(axis=2) / ww_days_sum) * 365.0).mean(axis=1)
            
            dd_inet = delta_inv_net[ss[:, :, None], ii[:, None, :]]
            samples_inv_net = ((dd_inet.sum(axis=2) / ww_days_sum) * 365.0).mean(axis=1)
            
            pt_inv_cost = float((delta_inv_cost * w).sum() / w.sum())
            pt_inv_static = float((delta_inv_static * w).sum() / w.sum())
            pt_inv_cash = float(delta_inv_cash.sum(axis=1).mean() * scale_annual)
            pt_inv_net = float(delta_inv_net.sum(axis=1).mean() * scale_annual)
            
            all_bootstrap_records.append({
                'site': site,
                'contrast_type': 'calibration_inversion_calibrated_dlinear_minus_uncalibrated_physical',
                'method': 'none_f1.0_plain (calibrated)',
                'method_name': 'Decoupled Calibrated Unconstrained DLinear',
                'policy': 'quantile_calibrated_vs_uncalibrated_physical',
                'reference': f'{phys_method} (uncalibrated)',
                'delta_cost_dyn': pt_inv_cost,
                'ci95_cost_dyn_low': float(np.quantile(samples_inv_cost, 0.025)),
                'ci95_cost_dyn_high': float(np.quantile(samples_inv_cost, 0.975)),
                'fwer_cost_dyn_low': np.nan,
                'fwer_cost_dyn_high': np.nan,
                'delta_cost_static': pt_inv_static,
                'ci95_cost_static_low': float(np.quantile(samples_inv_static, 0.025)),
                'ci95_cost_static_high': float(np.quantile(samples_inv_static, 0.975)),
                'delta_cashout_per_turb_yr_gbp': pt_inv_cash,
                'ci95_cashout_low': float(np.quantile(samples_inv_cash, 0.025)),
                'ci95_cashout_high': float(np.quantile(samples_inv_cash, 0.975)),
                'farm_annual_delta_cashout_gbp': pt_inv_cash * n_turb,
                'ci95_farm_cashout_low': float(np.quantile(samples_inv_cash, 0.025)) * n_turb,
                'ci95_farm_cashout_high': float(np.quantile(samples_inv_cash, 0.975)) * n_turb,
                'delta_net_cashflow_per_turb_yr_gbp': pt_inv_net,
                'ci95_net_cashflow_low': float(np.quantile(samples_inv_net, 0.025)),
                'ci95_net_cashflow_high': float(np.quantile(samples_inv_net, 0.975)),
                'farm_annual_delta_net_cashflow_gbp': pt_inv_net * n_turb,
                'ci95_farm_net_cashflow_low': float(np.quantile(samples_inv_net, 0.025)) * n_turb,
                'ci95_farm_net_cashflow_high': float(np.quantile(samples_inv_net, 0.975)) * n_turb,
                'calendar_days': N,
                'scored_days': scored_days,
                'seeds': len(seeds),
                'interpretation': 'Negative proves calibrated unconstrained model strictly beats uncalibrated physical constraint under real market prices.'
            })

    # Concatenate and save results
    df_table2 = pd.concat(all_table_records, ignore_index=True)
    table2_file = out_dir / 'dynamic_price_settlement_table2.csv'
    df_table2.to_csv(table2_file, index=False)
    print(f"\n[Artifact Saved] Formal Table 2: {table2_file} ({len(df_table2)} rows)")
    
    df_bootstrap = pd.DataFrame(all_bootstrap_records)
    boot_file = out_dir / 'dynamic_price_paired_intervals.csv'
    df_bootstrap.to_csv(boot_file, index=False)
    print(f"[Artifact Saved] Bootstrap Intervals: {boot_file} ({len(df_bootstrap)} contrast rows)")
    
    # Print clean formatted summary of Table 2 (period == 'all')
    print("\n" + "="*80)
    print("TABLE 2 SUMMARY: Operational Market Cashflow & Settlement Audit under Real-World Dynamic Prices (Period: Full Year 2017)")
    print("="*80)
    main_view = df_table2[df_table2.period == 'all'][
        ['site', 'method', 'policy', 'rmse_pu_mean', 'cost_dyn_mean', 'cost_dyn_std', 'cost_static_mean', 'cashout_per_turb_yr_gbp_mean', 'cashout_daily_gbp_mean', 'cashout_total_gbp_mean']
    ]
    print(main_view.to_string(index=False))
    
    return df_table2, df_bootstrap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate wind forecasting policies under 2017 dynamic Elexon balancing settlement prices.")
    parser.add_argument('--root', default='e:/论文1/06_实验结果与日志/constraint_round2', help="Root path for experimental models and selections")
    parser.add_argument('--data', default='e:/论文1/04_数据集/constraint_rebuild_v2', help="Path to SCADA datasets")
    parser.add_argument('--elexon', default='e:/论文1/04_数据集/elexon_system_prices_2016_2024.csv', help="Path to Elexon system prices CSV")
    parser.add_argument('--out', default='e:/论文1/06_实验结果与日志/constraint_round2/analysis', help="Output directory for analysis artifacts")
    parser.add_argument('--bootstrap_reps', type=int, default=2000, help="Number of bootstrap resamples (default 2000)")
    parser.add_argument('--site', default=None, choices=['hill_of_towie', 'kelmarsh'], help="Optional specific site")
    args = parser.parse_args()
    
    sites = [args.site] if args.site else ['hill_of_towie', 'kelmarsh']
    run_evaluation(args.root, args.data, args.elexon, args.out, sites=sites, bootstrap_reps=args.bootstrap_reps)
