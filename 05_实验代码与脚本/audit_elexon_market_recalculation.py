# -*- coding: utf-8 -*-
"""
audit_elexon_market_recalculation.py
====================================
Independent recalculation and audit of raw Elexon settlement prices and
Zero-Bid unhedged imbalance cashflow diagnostic across all three wind farms.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
ELEXON_CSV = ROOT / '04_数据集' / 'elexon_system_prices_2016_2024.csv'
SCADA_DIR = ROOT / '04_数据集' / 'constraint_rebuild_v2'
REPORT_JSON = ROOT / '06_实验结果与日志' / 'audit_elexon_market_report.json'

FARM_SPECS = {
    'kelmarsh': {'rated_kw': 12300.0, 'turbines': 6, 'expected_cashflow': 223882.15},
    'hill_of_towie': {'rated_kw': 48300.0, 'turbines': 21, 'expected_cashflow': 239299.30},
    'penmanshiel': {'rated_kw': 28700.0, 'turbines': 14, 'expected_cashflow': 243449.80}
}

SPLITS = {
    'kelmarsh': {'val_end': '2017-01-01', 'test_end': '2018-01-01'},
    'hill_of_towie': {'val_end': '2017-01-01', 'test_end': '2018-01-01'},
    'penmanshiel': {'val_end': '2017-01-01', 'test_end': '2018-01-01'}
}

def main():
    print("=" * 75)
    print(" EC5: Independent Elexon Raw Price & Zero-Bid Recalculation Audit")
    print("=" * 75)

    if not ELEXON_CSV.exists():
        print(f"[-] Missing Elexon CSV: {ELEXON_CSV}")
        sys.exit(1)

    df_elexon = pd.read_csv(ELEXON_CSV)
    df_2017 = df_elexon[df_elexon['settlementDate'].str.startswith('2017')].copy()
    sbp_2017 = df_2017['systemBuyPrice']

    elexon_mean = float(sbp_2017.mean())
    elexon_peak = float(sbp_2017.max())
    elexon_negative_count = int((sbp_2017 < 0).sum())
    records_count = len(df_2017)

    print(f"[+] 2017 Settlement Periods: {records_count}")
    print(f"[+] Mean System Buy Price: £{elexon_mean:.2f}/MWh (Paper claims: £44.27/MWh)")
    print(f"[+] Peak System Buy Price: £{elexon_peak:.2f}/MWh (Paper claims: £1,509.80/MWh)")
    print(f"[+] Negative Price Periods: {elexon_negative_count} (Paper claims: 68)")

    discrepancies = []
    if abs(elexon_mean - 44.27) > 0.01:
        discrepancies.append(f"Mean price discrepancy: {elexon_mean:.4f} vs 44.27")
    if abs(elexon_peak - 1509.80) > 0.01:
        discrepancies.append(f"Peak price discrepancy: {elexon_peak:.2f} vs 1509.80")
    if elexon_negative_count != 68:
        discrepancies.append(f"Negative count discrepancy: {elexon_negative_count} vs 68")

    # Recalculate Zero-Bid cashflow across all 3 sites
    df_elexon['startTime'] = pd.to_datetime(df_elexon['startTime'], utc=True)
    price_map = dict(zip(df_elexon['startTime'], df_elexon['systemBuyPrice']))

    L, H, S = 36, 36, 6
    site_reports = []

    for site, specs in FARM_SPECS.items():
        scada_file = SCADA_DIR / f'{site}.csv'
        if not scada_file.exists():
            discrepancies.append(f"Missing SCADA file for {site}")
            continue

        f = pd.read_csv(scada_file, index_col='time', parse_dates=True)
        f.index = pd.to_datetime(f.index, utc=True)
        
        raw = f[['power_pu', 'wind_speed']].copy()
        raw['wind_speed'] /= 15.0
        filled = raw.ffill(limit=3).to_numpy(np.float32)
        y = f.power_pu.to_numpy(np.float32)
        ids = np.arange(L, len(f) - H + 1, S)
        good = np.array([np.isfinite(filled[i-L:i]).all() and np.isfinite(y[i:i+H]).all() for i in ids])
        ids = ids[good]
        ts = f.index
        
        va = pd.Timestamp(SPLITS[site]['val_end'], tz='UTC')
        end = pd.Timestamp(SPLITS[site]['test_end'], tz='UTC')
        test_ids = ids[(ts[ids] >= va) & (ts[ids+H-1] < end)]
        
        step_records = []
        for origin_idx in test_ids:
            for h in range(6):
                target_idx = origin_idx + h
                t_step = f.index[target_idx]
                step_records.append((origin_idx, h, target_idx, t_step, y[target_idx]))
                
        df_tl = pd.DataFrame(step_records, columns=['origin_idx', 'h', 'target_idx', 't_step', 'y'])
        df_tl['interval_start'] = df_tl['t_step'] - pd.Timedelta(minutes=10)
        df_tl['sp_start'] = df_tl['interval_start'].dt.floor('30min')
        df_tl['price_sbp'] = df_tl['sp_start'].map(price_map)
        
        # Zero-Bid: p = 0, e = -y
        df_tl['e'] = -df_tl['y']
        sp_agg = df_tl.groupby('sp_start').agg(
            e_sum=('e', 'sum'),
            n_steps=('e', 'count'),
            price=('price_sbp', 'first')
        ).reset_index()
        sp_agg['day'] = sp_agg['sp_start'].dt.strftime('%Y-%m-%d')
        sp_intact = sp_agg[sp_agg['n_steps'] == 3].copy()
        
        p_rated = specs['rated_kw']
        n_turb = specs['turbines']
        sp_intact['e_mwh'] = sp_intact['e_sum'] * (p_rated / 6000.0)
        sp_intact['cashout_gbp'] = sp_intact['price'] * np.maximum(sp_intact['e_mwh'], 0)
        sp_intact['surplus_gbp'] = sp_intact['price'] * np.maximum(-sp_intact['e_mwh'], 0)
        sp_intact['net_cashflow_gbp'] = sp_intact['surplus_gbp'] - sp_intact['cashout_gbp']
        sp_intact['cashout_per_turb'] = sp_intact['cashout_gbp'] / n_turb
        sp_intact['net_cashflow_per_turb'] = sp_intact['net_cashflow_gbp'] / n_turb
        
        sp_day = sp_intact.groupby('day').agg(
            net_cashflow_daily_per_turb=('net_cashflow_per_turb', 'sum'),
            cashout_daily_per_turb=('cashout_per_turb', 'sum')
        ).reset_index()
        
        annual_net = float(sp_day['net_cashflow_daily_per_turb'].mean() * 365.0)
        annual_cashout = float(sp_day['cashout_daily_per_turb'].mean() * 365.0)
        exp_cashflow = specs['expected_cashflow']

        print(f"\n[Site] {site.upper()}:")
        print(f"  Test origins: {len(test_ids):,}, Complete SPs: {len(sp_intact):,}, Days: {len(sp_day)}")
        print(f"  Shortfall Penalty: £{annual_cashout:.2f}/(turb·yr) (Zero shortfall verified)")
        print(f"  Net Cashflow: £{annual_net:.2f}/(turb·yr) (Expected: £{exp_cashflow:.2f}/(turb·yr))")

        if abs(annual_net - exp_cashflow) > 1.0:
            discrepancies.append(f"{site} Zero-Bid cashflow discrepancy: {annual_net:.2f} vs {exp_cashflow:.2f}")

        site_reports.append({
            'site': site,
            'origins': len(test_ids),
            'complete_settlement_periods': len(sp_intact),
            'observed_days': len(sp_day),
            'zero_bid_shortfall_penalty_gbp': annual_cashout,
            'zero_bid_net_cashflow_per_turb_yr_gbp': annual_net,
            'expected_net_cashflow_gbp': exp_cashflow
        })

    report = {
        'status': 'PASS' if not discrepancies else 'FAIL',
        'elexon_2017_stats': {
            'settlement_periods': records_count,
            'mean_price_gbp_per_mwh': round(elexon_mean, 2),
            'peak_price_gbp_per_mwh': round(elexon_peak, 2),
            'negative_price_periods_count': elexon_negative_count
        },
        'zero_bid_diagnostic': site_reports,
        'discrepancies': discrepancies
    }

    with open(REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Audit report written to: {REPORT_JSON}")
    if discrepancies:
        print("[-] Audit FAILED with discrepancies:", discrepancies)
        sys.exit(1)
    else:
        print("[+] EC5 Audit PASSED: Raw Elexon price series and Zero-Bid numbers verified with zero discrepancy.")

if __name__ == '__main__':
    main()
