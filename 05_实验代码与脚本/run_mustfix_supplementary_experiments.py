import os
import io
import json
import math
import time
import contextlib
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import psutil

from ultra_lsnt_timeseries import (
    load_csv_data,
    create_dataloaders,
    TimeSeriesConfig,
    TrainConfig,
    LSNTConfig,
    UltraLSNTForecaster,
    evaluate,
    compute_metrics,
)
from ultra_lsnt_lite import UltraLSNTLiteForecaster
from run_dlinear import DLinear
from wind_dispatch_model import get_default_params, build_dispatch_model, solve_dispatch, parse_solution, evaluate_cost


OUT_DIR = os.path.join('results', 'supplementary_evidence')
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def log(msg):
    print(msg, flush=True)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_main_data_and_loaders(batch_size=128, num_workers=0):
    data, _ = load_csv_data('wind_final.csv', 'power')
    ts = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    tc = TrainConfig(batch_size=batch_size, epochs=1, num_workers=num_workers)
    train_loader, val_loader, test_loader, scaler = create_dataloaders(data, ts, tc)
    return data, ts, tc, train_loader, val_loader, test_loader, scaler


def make_ultra_model_from_main(ts_config, device):
    cfgj = json.load(open('checkpoints_ts/main/model_config.json', 'r', encoding='utf-8'))
    cfg = LSNTConfig(
        input_dim=cfgj['input_dim'],
        hidden_dim=cfgj['hidden_dim'],
        output_dim=1,
        num_blocks=cfgj['num_blocks'],
        num_experts=cfgj['num_experts'],
        top_k=cfgj['top_k'],
        dropout=cfgj.get('dropout', 0.1),
        router_z_loss_coef=cfgj.get('router_z_loss_coef', 0.01),
        router_aux_loss_coef=cfgj.get('router_aux_loss_coef', 0.01),
        router_jitter_noise=cfgj.get('router_jitter_noise', 0.1),
        skip_threshold=cfgj.get('skip_threshold', 0.5),
        dual_mode_expansion=cfgj.get('dual_mode_expansion', 4),
        probabilistic_mode='none',
        heteroscedastic_moe=False,
    )
    model = UltraLSNTForecaster(cfg, ts_config).to(device)
    ckpt = torch.load('checkpoints_ts/main/best_model.pth', map_location=device)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        log(f'[WARN] ultra load mismatch: missing={len(missing)}, unexpected={len(unexpected)}')
    return model


def generate_ultra_predictions(model, test_loader, scaler, device, tag='main'):
    model.eval()
    metrics, efficiency, preds, trues = evaluate(model, test_loader, device, temperature=0.1, scaler_obj=scaler)
    out = os.path.join(OUT_DIR, f'ultra_predictions_{tag}.npz')
    np.savez_compressed(out, predictions=preds, ground_truth=trues)
    with open(os.path.join(OUT_DIR, f'ultra_metrics_{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'efficiency': efficiency, 'tag': tag}, f, indent=2, ensure_ascii=False)
    return preds, trues, metrics, efficiency


def generate_dlinear_predictions(test_loader, scaler, device):
    ckpt = torch.load('dlinear_weights/dlinear_trained.pth', map_location=device)
    cfg = ckpt['config']
    model = DLinear(cfg['seq_len'], cfg['pred_len'], cfg['input_dim']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            preds.append(out.cpu().numpy())
            trues.append(by.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    target_mean, target_std = scaler.mean[-1], scaler.std[-1]
    preds = preds * target_std + target_mean
    trues = trues * target_std + target_mean

    np.savez_compressed(os.path.join(OUT_DIR, 'dlinear_predictions_full.npz'), predictions=preds, ground_truth=trues)
    return preds, trues


def compute_curtailment(solution, demand, wind_mean):
    g = solution['thermal']
    e_ch = solution['charge']
    e_dis = solution['discharge']
    curtail = []
    for t in range(len(wind_mean)):
        max_acceptable_wind = demand - (g[t] + e_dis[t] - e_ch[t])
        curtail.append(max(wind_mean[t] - max_acceptable_wind, 0.0))
    return float(np.sum(curtail))


def dispatch_day_from_pred(wind_mean, params):
    wind_std = np.maximum(np.abs(wind_mean) * 0.2, 1e-6)
    _, _, _, x0, _ = build_dispatch_model(params, wind_mean, wind_std)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        x = solve_dispatch(params, wind_mean, wind_std)
    status = 'optimized' if x is not None else 'fallback_x0'
    if x is None:
        x = x0
    sol = parse_solution(x, params)
    c = evaluate_cost(sol, params, wind_mean, wind_std)
    curtail = compute_curtailment(sol, params['demand'], wind_mean)
    return {
        'total_cost_cny': float(c['total_cost']),
        'fuel_cost_cny': float(c['fuel_cost']),
        'reserve_cost_cny': float(c['reserve_cost']),
        'purchase_cost_cny': float(c['purchase_cost']),
        'storage_cost_cny': float(c['storage_cost']),
        'startup_shutdown_cost_cny': float(c['startup_cost'] + c['shutdown_cost']),
        'total_reserve_kw': float(np.sum(sol['reserve'])),
        'avg_reserve_kw': float(np.mean(sol['reserve'])),
        'total_curtailment_kwh': float(curtail),
        'solver_status': status,
    }


def run_rolling_dispatch_with_ultra(ultra_preds, dlinear_preds, days=120):
    timemixer_preds = np.load('checkpoints_ts/TimeMixer_preds.npy')
    itrans_preds = np.load('checkpoints_ts/iTransformer_preds.npy')

    preds_map = {
        'Ultra-LSNT': ultra_preds,
        'DLinear': dlinear_preds,
        'TimeMixer': timemixer_preds,
        'iTransformer': itrans_preds,
    }

    params = get_default_params()
    params['T'] = 24
    params['dt'] = 1

    daily_rows = []
    for model, arr in preds_map.items():
        n = min(days, arr.shape[0])
        for i in range(n):
            r = dispatch_day_from_pred(arr[i].astype(float), params)
            r['model'] = model
            r['day_idx'] = i
            daily_rows.append(r)

    daily = pd.DataFrame(daily_rows)
    daily_path = os.path.join(OUT_DIR, 'dispatch_rolling_with_ultra_daily.csv')
    daily.to_csv(daily_path, index=False)

    summary = (
        daily.groupby('model', as_index=False)
        .agg(
            days=('day_idx', 'count'),
            total_cost_cny_mean=('total_cost_cny', 'mean'),
            total_cost_cny_std=('total_cost_cny', 'std'),
            total_reserve_kw_mean=('total_reserve_kw', 'mean'),
            avg_reserve_kw_mean=('avg_reserve_kw', 'mean'),
            total_curtailment_kwh_mean=('total_curtailment_kwh', 'mean'),
            solver_optimized_ratio=('solver_status', lambda s: float((s == 'optimized').mean())),
        )
        .sort_values('total_cost_cny_mean')
    )
    summary_path = os.path.join(OUT_DIR, 'dispatch_rolling_with_ultra_summary.csv')
    summary.to_csv(summary_path, index=False)

    return daily, summary


def build_lite_orig():
    cfg = {
        'input_dim': 8,
        'hidden_dim': 128,
        'seq_len': 96,
        'pred_len': 24,
        'num_blocks': 2,
        'num_experts': 4,
        'top_k': 2,
        'scales': [1, 2, 4],
    }
    return UltraLSNTLiteForecaster(cfg)


def build_lite_opt():
    cfg = {
        'input_dim': 8,
        'hidden_dim': 96,
        'seq_len': 96,
        'pred_len': 24,
        'num_blocks': 1,
        'num_experts': 2,
        'top_k': 1,
        'scales': [1, 2],
    }
    return UltraLSNTLiteForecaster(cfg)


def benchmark_cpu_batch1(models_dict, input_tensor, warmup=50, runs=250, threads=1):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except Exception:
        pass

    proc = psutil.Process(os.getpid())
    rows = []

    for name, (model, forward_fn, note) in models_dict.items():
        model = model.to('cpu').eval()
        x = input_tensor.to('cpu')

        rss_before = proc.memory_info().rss / (1024 * 1024)

        with torch.no_grad():
            for _ in range(warmup):
                _ = forward_fn(model, x)

        times = []
        peak_rss = rss_before
        with torch.no_grad():
            for i in range(runs):
                t0 = time.perf_counter()
                _ = forward_fn(model, x)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)
                if (i + 1) % 10 == 0:
                    peak_rss = max(peak_rss, proc.memory_info().rss / (1024 * 1024))

        rows.append({
            'model': name,
            'threads': threads,
            'warmup_runs': warmup,
            'measure_runs': runs,
            'latency_mean_ms': float(np.mean(times)),
            'latency_std_ms': float(np.std(times, ddof=1)),
            'latency_p50_ms': float(np.percentile(times, 50)),
            'latency_p95_ms': float(np.percentile(times, 95)),
            'latency_min_ms': float(np.min(times)),
            'latency_max_ms': float(np.max(times)),
            'peak_rss_mb': float(peak_rss),
            'note': note,
        })

    df = pd.DataFrame(rows).sort_values('latency_mean_ms')
    out = os.path.join(OUT_DIR, 'cpu_batch1_ultra_family_latency.csv')
    df.to_csv(out, index=False)
    return df


def set_tau(model, tau):
    tau = float(tau)
    eps = 1e-6
    tau = min(max(tau, eps), 1 - eps)
    raw = math.log(tau / (1 - tau))
    for blk in model.blocks:
        blk.dual_prop.skip_gate.threshold.data.fill_(raw)


def run_tau_sweep(ts_config, test_loader, scaler, device, cpu_input):
    taus = [0.3, 0.5, 0.7, 0.9]
    rows = []

    for tau in taus:
        model = make_ultra_model_from_main(ts_config, device)
        set_tau(model, tau)

        # Accuracy + skip rate
        metrics, efficiency, _, _ = evaluate(model, test_loader, device, temperature=0.1, scaler_obj=scaler)

        # CPU batch=1 latency at this tau
        model_cpu = deepcopy(model).to('cpu').eval()
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        with torch.no_grad():
            for _ in range(30):
                _ = model_cpu(cpu_input.to('cpu'), temperature=0.1)[0]
        tms = []
        with torch.no_grad():
            for _ in range(220):
                t0 = time.perf_counter()
                _ = model_cpu(cpu_input.to('cpu'), temperature=0.1)[0]
                t1 = time.perf_counter()
                tms.append((t1 - t0) * 1000.0)

        rows.append({
            'tau': tau,
            'R2': float(metrics['R2']),
            'RMSE': float(metrics['RMSE']),
            'MAE': float(metrics['MAE']),
            'skip_rate': float(efficiency['avg_skip_rate']),
            'compute_ratio': float(efficiency['avg_compute_ratio']),
            'cpu_batch1_latency_mean_ms': float(np.mean(tms)),
            'cpu_batch1_latency_std_ms': float(np.std(tms, ddof=1)),
        })

    df = pd.DataFrame(rows).sort_values('tau')
    out = os.path.join(OUT_DIR, 'tau_sweep_ultra_pareto.csv')
    df.to_csv(out, index=False)

    # Optional simple pareto plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['svg.fonttype'] = 'none'
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        import matplotlib.pyplot as plt

        plot_df = df.sort_values('cpu_batch1_latency_mean_ms').reset_index(drop=True)
        plt.figure(figsize=(7, 4.5), dpi=200)
        plt.plot(plot_df['cpu_batch1_latency_mean_ms'], plot_df['R2'], marker='o')

        x_max = float(plot_df['cpu_batch1_latency_mean_ms'].max())
        x_min = float(plot_df['cpu_batch1_latency_mean_ms'].min())
        x_pad = max(0.003, (x_max - x_min) * 0.05)
        plt.xlim(x_min - x_pad, x_max + x_pad)
        ann_kw = dict(fontsize=12, fontweight='normal', color='black')
        # Put 0.5/0.9/0.7 labels on one horizontal row below the top line.
        top_y = float(plot_df['R2'].max())
        row_y = top_y - 0.00095

        for _, r in plot_df.iterrows():
            x_i = float(r['cpu_batch1_latency_mean_ms'])
            y_i = float(r['R2'])
            tau_i = round(float(r['tau']), 1)
            label = rf'$\tau={tau_i:.1f}$'
            # Three top labels share one y-level for visual alignment.
            if tau_i in (0.5, 0.9, 0.7):
                if tau_i == 0.5:
                    # Nudge right to avoid overlap with the steep left segment.
                    plt.text(x_i + 0.0018, row_y, label, ha='left', va='top', **ann_kw)
                elif tau_i == 0.7:
                    # Nudge left so right-most text stays inside frame.
                    plt.text(x_i - 0.003, row_y, label, ha='right', va='top', **ann_kw)
                else:
                    plt.text(x_i, row_y, label, ha='center', va='top', **ann_kw)
            elif abs(x_i - x_max) < 1e-9:
                plt.annotate(label, (x_i, y_i), textcoords='offset points', xytext=(-8, -12), ha='right', va='top', **ann_kw)
            else:
                plt.annotate(label, (x_i, y_i), textcoords='offset points', xytext=(4, 4), ha='left', va='bottom', **ann_kw)
        plt.xlabel('CPU batch=1 latency (ms)')
        plt.ylabel('R2')
        plt.title('Tau Sweep Pareto (Latency vs R2)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_base = os.path.join(OUT_DIR, 'tau_sweep_ultra_pareto')
        plt.savefig(out_base + '.png')
        plt.savefig(out_base + '.pdf')
        plt.savefig(out_base + '.svg')
        plt.close()
    except Exception as e:
        log(f'[WARN] tau pareto plot skipped: {e}')

    return df


def evaluate_noisy_r2(model, test_loader, scaler_obj, device, nominal_sigma, alpha=1.5):
    model.eval()
    all_preds, all_trues = [], []
    usage_vars = []
    collapse_idx = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            by = by.to(device)

            # per-sample, per-feature std over time
            std = bx.std(dim=1, keepdim=True)
            noise = torch.randn_like(bx) * (alpha * nominal_sigma) * std
            bx_noisy = bx + noise

            pred, _, stats = model(bx_noisy, temperature=0.1, return_stats=True)

            # router usage stats
            for b in stats['block_stats']:
                usage = np.array(b['router']['expert_usage'], dtype=float)
                usage_vars.append(float(np.var(usage)))
                collapse_idx.append(float(np.max(usage)))

            all_preds.append(pred.cpu().numpy())
            all_trues.append(by.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)

    target_mean, target_std = scaler_obj.mean[-1], scaler_obj.std[-1]
    preds = preds * target_std + target_mean
    trues = trues * target_std + target_mean

    m = compute_metrics(preds, trues)
    m['expert_usage_var_mean'] = float(np.mean(usage_vars)) if usage_vars else np.nan
    m['expert_usage_max_mean'] = float(np.mean(collapse_idx)) if collapse_idx else np.nan
    return m


def quick_finetune_lambda(ts_config, train_loader, test_loader, scaler, device, lambda_value, epochs=2, max_batches=250):
    model = make_ultra_model_from_main(ts_config, device)

    # set lambda in router
    for blk in model.blocks:
        if hasattr(blk, 'moe_router') and hasattr(blk.moe_router, 'aux_loss_coef'):
            blk.moe_router.aux_loss_coef = float(lambda_value)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    mse = nn.MSELoss()

    for ep in range(epochs):
        n = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            pred, aux = model(bx, temperature=1.0)
            loss = mse(pred, by) + 0.01 * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            n += 1
            if n >= max_batches:
                break

    # robustness points (nominal sigma 0.2 / 0.4 => eff 0.3 / 0.6)
    rows = []
    for sigma in [0.2, 0.4]:
        met = evaluate_noisy_r2(model, test_loader, scaler, device, nominal_sigma=sigma, alpha=1.5)
        rows.append({
            'lambda': float(lambda_value),
            'nominal_sigma': sigma,
            'effective_sigma': 1.5 * sigma,
            'R2': float(met['R2']),
            'RMSE': float(met['RMSE']),
            'MAE': float(met['MAE']),
            'expert_usage_var_mean': float(met['expert_usage_var_mean']),
            'expert_usage_max_mean': float(met['expert_usage_max_mean']),
        })

    return pd.DataFrame(rows)


def run_lambda_robustness(ts_config, train_loader, test_loader, scaler, device):
    dfs = []
    for lam in [0.01, 0.05]:
        log(f'[lambda] fine-tune + robustness: lambda={lam}')
        df = quick_finetune_lambda(ts_config, train_loader, test_loader, scaler, device, lambda_value=lam)
        dfs.append(df)
    out_df = pd.concat(dfs, ignore_index=True)
    out = os.path.join(OUT_DIR, 'lambda_robustness_sigma_eff_compare.csv')
    out_df.to_csv(out, index=False)
    return out_df


def main():
    device = get_device()
    log(f'== Device: {device} ==')

    # data/loaders
    data, ts_config, train_cfg, train_loader, val_loader, test_loader, scaler = load_main_data_and_loaders(batch_size=128, num_workers=0)
    log(f'data shape={data.shape}, test_batches={len(test_loader)}')

    # 1) Ultra predictions + rolling dispatch include Ultra
    log('\n[1/4] generating Ultra full predictions...')
    ultra_model = make_ultra_model_from_main(ts_config, device)
    ultra_preds, ultra_trues, ultra_metrics, ultra_eff = generate_ultra_predictions(ultra_model, test_loader, scaler, device, tag='main_fulltest')
    log(f"Ultra test R2={ultra_metrics['R2']:.4f}, RMSE={ultra_metrics['RMSE']:.2f}, skip={ultra_eff['avg_skip_rate']:.4f}")

    log('[1/4] generating DLinear full predictions...')
    dlinear_preds, dlinear_trues = generate_dlinear_predictions(test_loader, scaler, device)
    d_metrics = compute_metrics(dlinear_preds, dlinear_trues)
    log(f"DLinear test R2={d_metrics['R2']:.4f}, RMSE={d_metrics['RMSE']:.2f}")

    log('[1/4] running rolling multi-day dispatch with Ultra/DLinear/TimeMixer/iTransformer...')
    _, roll_summary = run_rolling_dispatch_with_ultra(ultra_preds, dlinear_preds, days=120)
    log('\nRolling summary:')
    log(roll_summary.to_string(index=False))

    # 2) CPU batch=1 latency for Ultra family
    log('\n[2/4] CPU batch=1 latency for Ultra family...')
    cpu_input = torch.from_numpy(data[:96].astype(np.float32)).unsqueeze(0)  # (1,96,8)

    ultra_cpu = make_ultra_model_from_main(ts_config, torch.device('cpu'))
    lite_orig = build_lite_orig()
    lite_opt = build_lite_opt()

    models = {
        'Ultra-LSNT(full)': (ultra_cpu, lambda m, x: m(x, temperature=0.1)[0], 'checkpoint: checkpoints_ts/main/best_model.pth'),
        'Ultra-LSNT-Lite(orig)': (lite_orig, lambda m, x: m(x)[0], 'arch-only (no dedicated lite checkpoint found)'),
        'Ultra-LSNT-Lite(opt)': (lite_opt, lambda m, x: m(x)[0], 'arch-only (optimized config hidden=96,blocks=1,experts=2,top_k=1)'),
    }
    lat_df = benchmark_cpu_batch1(models, cpu_input, warmup=50, runs=250, threads=1)
    log(lat_df.to_string(index=False))

    # 3) tau sweep
    log('\n[3/4] tau sweep + pareto...')
    tau_df = run_tau_sweep(ts_config, test_loader, scaler, device, cpu_input)
    log(tau_df.to_string(index=False))

    # 4) lambda robustness compare
    log('\n[4/4] lambda=0.01 vs 0.05 robustness compare (quick controlled fine-tune)...')
    lam_df = run_lambda_robustness(ts_config, train_loader, test_loader, scaler, device)
    log(lam_df.to_string(index=False))

    log('\nAll required outputs generated under results/supplementary_evidence/')


if __name__ == '__main__':
    main()
