"""
evaluate_patchtst_settlement.py
================================
Evaluates trained PatchTST models under real-world 2017 UK Elexon BSC 
30-minute physical netting settlement prices across Kelmarsh, Hill of Towie, 
and Penmanshiel.
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))

from core import starts, windows, infer
from patchtst_core import PatchTST

FARM_SPECS = {
    'hill_of_towie': {
        'name': 'Hill of Towie',
        'rated_kw': 48300.0,
        'turbines': 21,
        'turbine_model': 'Siemens SWT-2.3-82',
    },
    'kelmarsh': {
        'name': 'Kelmarsh',
        'rated_kw': 12300.0,
        'turbines': 6,
        'turbine_model': 'Senvion MM82',
    },
    'penmanshiel': {
        'name': 'Penmanshiel',
        'rated_kw': 28700.0,
        'turbines': 14,
        'turbine_model': 'Senvion MM82',
    }
}

METHODS = [
    ('none_l0.0_f1.0_plain', 'Unconstrained PatchTST'),
    ('bounds_l0.01_f1.0_plain', 'Bounds Constrained PatchTST'),
    ('curve_l0.01_f1.0_plain', 'Power Curve Constrained PatchTST'),
    ('distribution_l0.01_f1.0_plain', 'Distribution Prior PatchTST'),
]

SEEDS = CFG['seed_list']

def load_elexon_prices(elexon_path):
    print(f"Loading Elexon settlement prices from {elexon_path}...")
    df_elexon = pd.read_csv(elexon_path)
    df_elexon['startTime'] = pd.to_datetime(df_elexon['startTime'], utc=True)
    price_map = dict(zip(df_elexon['startTime'], df_elexon['systemBuyPrice']))
    print(f"  Loaded {len(price_map):,} Elexon price records.")
    return price_map

def run_settlement_evaluation():
    data_dir = ROOT / '04_数据集' / 'constraint_rebuild_v2'
    price_csv = ROOT / '04_数据集' / 'elexon_system_prices_2016_2024.csv'
    res_dir = ROOT / '06_实验结果与日志' / 'constraint_round2'
    out_dir = res_dir / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    price_map = load_elexon_prices(price_csv)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_site_records = []

    for site, spec in FARM_SPECS.items():
        print(f"\n=======================================================")
        print(f"Evaluating Site: {site} ({spec['name']})")
        print(f"=======================================================")

        scada_path = data_dir / f'{site}.csv'
        f = pd.read_csv(scada_path, index_col='time', parse_dates=True)
        f.index = pd.to_datetime(f.index, utc=True)
        split_cfg = CFG['splits'][site]
        x, y, ids = starts(f, split_cfg)

        cal_x, cal_y = windows(x, y, ids['cal'])
        xx, yy = windows(x, y, ids['test'])
        n_test = len(ids['test'])
        times = f.index[ids['test']]
        print(f"  Test origins: {n_test:,} (delivery horizon: next-hour h=1..6)")

        # Precompute prices matrix (n_test, 6)
        prices = np.zeros((n_test, 6), dtype=np.float64)
        for idx_i, origin_idx in enumerate(ids['test']):
            for h in range(6):
                target_time = f.index[origin_idx + h]
                interval_start = target_time - pd.Timedelta(minutes=10)
                elexon_start = interval_start.floor('30min')
                prices[idx_i, h] = price_map.get(elexon_start, np.nan)

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

        p_rated = spec['rated_kw']
        n_turb = spec['turbines']

        for tag, m_name in METHODS:
            for policy in ['uncalibrated', 'quantile_calibrated']:
                seed_rmse = []
                seed_cost_dyn = []
                seed_shortfall_turb_yr = []
                seed_net_cashflow_turb_yr = []
                seed_total_cashout = []

                for seed in SEEDS:
                    pt_path = res_dir / site / 'patchtst' / f'seed{seed}' / f'{tag}.pt'
                    if not pt_path.exists():
                        continue

                    model = PatchTST(L=CFG['lookback'], H=CFG['horizon']).to(device)
                    model.load_state_dict(torch.load(pt_path, map_location=device))
                    model.eval()

                    # Infer on cal
                    cal_pred = infer(model, cal_x, device)[:, :6]
                    res_cal = cal_y[:, :6] - cal_pred
                    delta = np.quantile(res_cal, 1.0 / 3.0, axis=0) # tau = 1/3 for r=2

                    # Infer on test
                    test_pred = infer(model, xx, device)[:, :6]
                    base = np.clip(test_pred, 0, 1)
                    p = base if policy == 'uncalibrated' else np.clip(base + delta, 0, 1)

                    y_eval = yy[:, :6]
                    e = p - y_eval

                    # RMSE
                    rmse = np.sqrt(np.mean(e ** 2))
                    seed_rmse.append(rmse)

                    # Dynamic cost (Price-Weighted Normalized Cost)
                    # e = p - y. Shortfall (p > y, e > 0) penalized at 2.0, surplus (y > p, -e > 0) at 1.0
                    pinball = 2.0 * np.maximum(e, 0) + np.maximum(-e, 0)
                    mean_sbp = np.nanmean(prices)
                    w_t = prices / mean_sbp
                    cost_dyn = np.nanmean(w_t * pinball)
                    seed_cost_dyn.append(cost_dyn)

                    # BSC 30-minute physical netting settlement
                    df_tl = df_timeline_site.copy()
                    df_tl['p'] = p.reshape(-1)
                    df_tl['e'] = df_tl['p'] - df_tl['y']

                    sp_agg = df_tl.groupby('sp_start').agg(
                        e_sum=('e', 'sum'),
                        n_steps=('e', 'count'),
                        price=('price_sbp', 'first')
                    ).reset_index()

                    # Complete intact SPs (3 steps)
                    sp_intact = sp_agg[sp_agg['n_steps'] == 3].copy()
                    sp_intact['e_mwh'] = sp_intact['e_sum'] * (p_rated / 6000.0)
                    sp_intact['cashout_gbp'] = sp_intact['price'] * np.maximum(sp_intact['e_mwh'], 0)
                    sp_intact['surplus_gbp'] = sp_intact['price'] * np.maximum(-sp_intact['e_mwh'], 0)
                    sp_intact['net_cashflow_gbp'] = sp_intact['surplus_gbp'] - sp_intact['cashout_gbp']
                    sp_intact['cashout_per_turb'] = sp_intact['cashout_gbp'] / n_turb
                    sp_intact['net_cashflow_per_turb'] = sp_intact['net_cashflow_gbp'] / n_turb

                    # Daily SP aggregation matching evaluate_dynamic_price.py
                    sp_intact['day'] = sp_intact['sp_start'].dt.strftime('%Y-%m-%d')
                    sp_day = sp_intact.groupby('day').agg(
                        cashout_daily_per_turb=('cashout_per_turb', 'sum'),
                        net_cashflow_daily_per_turb=('net_cashflow_per_turb', 'sum')
                    ).reset_index()

                    shortfall_turb_yr = sp_day['cashout_daily_per_turb'].mean() * 365.0
                    net_turb_yr = sp_day['net_cashflow_daily_per_turb'].mean() * 365.0
                    total_cashout = sp_intact['cashout_gbp'].sum()

                    seed_shortfall_turb_yr.append(shortfall_turb_yr)
                    seed_net_cashflow_turb_yr.append(net_turb_yr)
                    seed_total_cashout.append(total_cashout)

                if not seed_rmse:
                    continue

                rec = {
                    'site': site,
                    'backbone': 'PatchTST',
                    'method': tag,
                    'method_name': m_name,
                    'policy': policy,
                    'rmse_pu_mean': float(np.mean(seed_rmse)),
                    'rmse_pu_std': float(np.std(seed_rmse)),
                    'cost_dyn_mean': float(np.mean(seed_cost_dyn)),
                    'cost_dyn_std': float(np.std(seed_cost_dyn)),
                    'shortfall_turb_yr_mean': float(np.mean(seed_shortfall_turb_yr)),
                    'shortfall_turb_yr_std': float(np.std(seed_shortfall_turb_yr)),
                    'net_cashflow_turb_yr_mean': float(np.mean(seed_net_cashflow_turb_yr)),
                    'net_cashflow_turb_yr_std': float(np.std(seed_net_cashflow_turb_yr)),
                    'total_cashout_mean': float(np.mean(seed_total_cashout)),
                }
                all_site_records.append(rec)
                print(f"[{site}] {m_name} ({policy:19s}) | RMSE={rec['rmse_pu_mean']:.4f} | Cost_dyn={rec['cost_dyn_mean']:.5f} | Shortfall=\u00a3{rec['shortfall_turb_yr_mean']:,.2f} | NetCashflow=\u00a3{rec['net_cashflow_turb_yr_mean']:,.2f}")

    df_out = pd.DataFrame(all_site_records)
    df_out.to_csv(out_dir / 'patchtst_settlement_table.csv', index=False)
    print(f"\nSettlement evaluation complete. Saved to {out_dir / 'patchtst_settlement_table.csv'}")

if __name__ == '__main__':
    run_settlement_evaluation()
