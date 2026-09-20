# -*- coding: utf-8 -*-
import os
import sys
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / '05_实验代码与脚本' / 'constraint_round2'
DATA_DIR = ROOT / '04_数据集' / 'constraint_rebuild_v2'
PRICE_FILE = ROOT / '04_数据集' / 'elexon_system_prices_2016_2024.csv'
LOCAL_RES_DIR = ROOT / '06_实验结果与日志' / 'ablation_3d_results'

SITES = ['kelmarsh', 'hill_of_towie', 'penmanshiel']
SEEDS = [42, 123, 456, 789, 1024]

def main():
    LOCAL_RES_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 75)
    print(' Unified 5-Seed Ablation Runner across 3 Sites & Seeds [42, 123, 456, 789, 1024]')
    print('=' * 75)

    for site in SITES:
        for seed in SEEDS:
            flat_csv = LOCAL_RES_DIR / f'ablation_results_{site}_seed{seed}.csv'
            sub_csv = LOCAL_RES_DIR / site / f'seed{seed}' / f'ablation_results_{site}_seed{seed}.csv'
            
            # If already exists and valid, ensure flat_csv exists
            if flat_csv.exists() and flat_csv.stat().st_size > 1000:
                print(f'[*] Exists: {site} seed {seed} ({flat_csv.name})')
                continue
            elif sub_csv.exists() and sub_csv.stat().st_size > 1000:
                print(f'[*] Copying from sub-dir to flat: {sub_csv} -> {flat_csv}')
                shutil.copy2(sub_csv, flat_csv)
                continue

            print(f'\n>>> Running: {site} | Seed {seed} on GPU...')
            cmd = [
                sys.executable,
                str(CODE_DIR / 'run_cluster_ablation.py'),
                '--site', site,
                '--seed', str(seed),
                '--gpu', '0',
                '--data_path', str(DATA_DIR),
                '--prices_path', str(PRICE_FILE),
                '--out_dir', str(LOCAL_RES_DIR)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f'[-] Error in {site} seed {seed}: {res.stderr}')
                sys.exit(1)
            
            # Copy to flat directory
            if sub_csv.exists():
                shutil.copy2(sub_csv, flat_csv)
                print(f'[+] Completed and saved: {flat_csv.name}')
            else:
                print(f'[-] Warning: {sub_csv} was not found!')

    print('\n' + '=' * 75)
    print(' Aggregating 5-Seed Master Summary & Key Comparison Table')
    print('=' * 75)

    all_dfs = []
    for site in SITES:
        for seed in SEEDS:
            f = LOCAL_RES_DIR / f'ablation_results_{site}_seed{seed}.csv'
            if not f.exists():
                print(f'[-] Missing file: {f}')
                continue
            df_s = pd.read_csv(f)
            df_s['site'] = site
            df_s['seed'] = seed
            all_dfs.append(df_s)

    master_df = pd.concat(all_dfs, ignore_index=True)
    master_path = LOCAL_RES_DIR / 'ablation_3d_master_summary.csv'
    master_df.to_csv(master_path, index=False)
    print(f'[+] Saved master summary ({len(master_df)} rows) to: {master_path}')

    # Key comparison table
    p2 = master_df[(master_df.ratio == 2) & (master_df.policy == 'calibrated')].copy()
    grouped = p2.groupby(['site', 'arm']).agg(
        cost_mean=('cost_static', 'mean'),
        cost_std=('cost_static', 'std'),
        pinball_mean=('pinball_loss', 'mean'),
        pinball_std=('pinball_loss', 'std'),
        rmse_mean=('rmse', 'mean'),
        cashflow_mean=('net_cashflow_per_turb_yr', 'mean'),
        shortfall_mean=('shortfall_per_turb_yr', 'mean'),
        n_seeds=('seed', 'count')
    ).round(6)

    key_table = LOCAL_RES_DIR / 'ablation_3d_key_comparison_table.csv'
    grouped.to_csv(key_table)
    print(f'[+] Saved key comparison table to: {key_table}')
    print('\nKey Results Summary (Ratio 2 / Quantile 1/3, Calibrated):')
    print(grouped.to_string())

if __name__ == '__main__':
    main()
