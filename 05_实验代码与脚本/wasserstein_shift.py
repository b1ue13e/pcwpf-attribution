"""Wasserstein distance between train-time constraint target distributions and
the deployment (test) distribution. Quantifies the distribution-shift mechanism
behind the oracle-substitution result (mechanism b).

For each site compute:
  - W1(train_power_marginal, test_power_marginal)
  - W1(train_wind_power_joint, test_wind_power_joint)
  - W1(each_constraint_target, test_power_marginal)  <- the "direction error" of the
    constraint: how far the distribution the model is pulled toward is from the
    distribution it will actually be evaluated on.

Uses 1D/2D exact Wasserstein via order statistics (1D) and joint 2D via
sliced-Wasserstein approximation (cheap, deterministic).
"""
from __future__ import annotations
import argparse, csv
import numpy as np
import pandas as pd


def w1_1d(a, b):
    """Exact 1D Wasserstein-1 via quantile matching (handles unequal sample sizes)."""
    a = np.sort(a); b = np.sort(b)
    # common quantile grid
    n = max(len(a), len(b))
    q = (np.arange(n) + 0.5) / n
    qa = np.interp(q, (np.arange(len(a)) + 0.5) / len(a), a)
    qb = np.interp(q, (np.arange(len(b)) + 0.5) / len(b), b)
    return float(np.mean(np.abs(qa - qb)))


def sliced_w1_2d(a, b, n_proj=64, seed=0):
    """Sliced Wasserstein-1 (deterministic projections on unit circle)."""
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0, np.pi, n_proj)
    total = 0.0
    for th in thetas:
        u = np.cos(th); v = np.sin(th)
        total += w1_1d(a[:, 0] * u + a[:, 1] * v, b[:, 0] * u + b[:, 1] * v)
    return total / len(thetas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="kelmarsh")
    ap.add_argument("--out", default="")
    ap.add_argument("--outcsv", default="")
    args = ap.parse_args()

    SITES = {
        "wind_cn": "../04_数据集/prepared_priority_datasets/wind_cn_core_15min.csv",
        "kelmarsh": "../04_数据集/prepared_priority_datasets/kelmarsh_uk_2016.csv",
        "smarteole": "../04_数据集/prepared_priority_datasets/smarteole_france_2020.csv",
        "penmanshiel": "../04_数据集/prepared_priority_datasets/penmanshiel_scotland_2016.csv",
        "wind_us": "../04_数据集/prepared_priority_datasets/wind_us_core_15min.csv",
    }
    df = pd.read_csv(SITES[args.site])
    ws = df["wind_speed"].to_numpy(np.float64)
    p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]

    # same split convention as the oracle-substitution experiments:
    # 80% top-level then 80/20 within windows (approx: train = first 64%)
    n0 = int(len(ws) * 0.8)           # fit_curve window
    pw_train = p[:n0]
    ws_train = ws[:n0]
    # oracle-substitution uses: X windows built from full, n_tr = 0.8*len(X) -> ~64% of rows
    split = int(len(ws) * 0.64)
    tr_p, te_p = p[:split], p[split:]
    tr_ws, te_ws = ws[:split], ws[split:]

    # constraint targets (same definitions as pct_oracle_substitution.py)
    p_rated = float(np.percentile(tr_p, 99))
    v_in = float(np.percentile(tr_ws, 6))
    v_rated = float(np.percentile(tr_ws, 96))
    # oracle = train marginal power
    oracle_tr = tr_p / max(p_rated, 1e-6)
    # lagged = previous-hour measured power (shift by 6 at 10-min, but our csv could be 15-min;
    # use lag=k as in experiment: shift power back by 6 rows)
    lag = 6
    lagged = np.empty_like(tr_p); lagged[:lag] = tr_p[:lag]; lagged[lag:] = tr_p[:-lag]
    lagged = lagged / max(p_rated, 1e-6)
    # nwp-linear: ideal linear ramp on real wind
    nwp_lin = np.clip((tr_ws - v_in) / max(v_rated - v_in, 1e-6), 0, 1)
    # nwp-measured: bin-mean measured curve
    bins = np.linspace(v_in, v_rated, 20)
    idx = np.digitize(tr_ws, bins)
    means = []
    for i in range(len(bins) - 1):
        m = (tr_p[idx == i].mean() / max(p_rated, 1e-6)) if (idx == i).sum() > 10 else np.nan
        means.append(m)
    bcs = (bins[:-1] + bins[1:]) / 2
    okb = np.isfinite(means)
    nwp_ms = np.clip(np.interp(tr_ws, bcs[okb], np.array(means)[okb], left=0, right=1), 0, 1)

    te_pu = te_p / max(p_rated, 1e-6)

    rows = []
    # marginal power drift
    rows.append(("train_vs_test_power_marginal", w1_1d(oracle_tr, te_pu)))
    # joint drift (wind normalized to [0,1] via /25)
    tr_joint = np.stack([tr_ws / 25.0, oracle_tr], axis=1)
    te_joint = np.stack([te_ws / 25.0, te_pu], axis=1)
    rows.append(("train_vs_test_joint_2d", sliced_w1_2d(tr_joint[:5000], te_joint[:5000])))
    # direction error of each constraint target vs test marginals
    for name, target in [("oracle", oracle_tr), ("lagged", lagged), ("nwp_linear", nwp_lin), ("nwp_measured", nwp_ms)]:
        rows.append((f"{name}_vs_test_power", w1_1d(target, te_pu)))

    line = f"{args.site} | " + " ".join(f"{n}={v:.5f}" for n, v in rows)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    if args.outcsv:
        with open(args.outcsv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["site"] + [n for n, _ in rows])
            w.writerow([args.site] + [f"{v:.5f}" for _, v in rows])


if __name__ == "__main__":
    main()
