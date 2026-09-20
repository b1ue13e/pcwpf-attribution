#!/usr/bin/env python3
"""
One-click runner for all submission add-on experiments on RTX 4090.

Phases:
1) Full backfill pipeline (existing): run_full_backfill_4090.py
2) Cross-domain completion for additional models:
   - ARIMA (auditable), SVR (auditable), SSA-ELM, Mamba (auditable)
3) Multi-seed robustness + significance for core models on Wind(CN):
   - Ultra-LSNT, DLinear, iTransformer, TimeMixer
4) Universal DM significance suite:
   - run_universal_significance.py
5) Ultra-LSNT-Lite robustness completion:
   - clean, gaussian@0.6, spike@0.6
6) Rebuild main4 matrices
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from ultra_lsnt_timeseries import (
    TimeSeriesConfig,
    TrainConfig,
    compute_metrics,
    create_dataloaders,
    load_csv_data,
)
from ultra_lsnt_lite import UltraLSNTLiteForecaster


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
EVIDENCE_DIR = RESULTS_DIR / "supplementary_evidence"

DATASETS: List[Tuple[str, str, str]] = [
    ("Wind (CN)", "wind_final.csv", "power"),
    ("Wind (US)", "wind_us.csv", "power (MW)"),
    ("Air Quality", "air_quality_ready.csv", "AQI"),
    ("GEFCom Load", "gefcom_ready.csv", "load"),
]
TRANSFER_COL = {
    "Wind (US)": "R2_wind_us",
    "Air Quality": "R2_air_quality",
    "GEFCom Load": "R2_gefcom_load",
}


def _force_utf8_stdio() -> None:
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


@dataclass
class StepResult:
    name: str
    ok: bool
    return_code: int
    elapsed_sec: float
    log_file: str
    note: str = ""


def parse_seeds(raw: str) -> List[int]:
    out: List[int] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError("No valid seeds parsed from --seeds")
    return out


def run_subprocess_step(
    name: str,
    cmd: List[str],
    log_dir: Path,
    dry_run: bool = False,
    timeout_sec: Optional[int] = None,
) -> StepResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    if dry_run:
        with log_path.open("w", encoding="utf-8") as f:
            f.write("DRY RUN\n")
            f.write(" ".join(cmd) + "\n")
        return StepResult(name=name, ok=True, return_code=0, elapsed_sec=0.0, log_file=str(log_path))

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[cmd] " + " ".join(cmd) + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        rc = 0
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
            proc.wait(timeout=timeout_sec)
            rc = int(proc.returncode)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = 124
            msg = f"\n[TIMEOUT] step {name} exceeded {timeout_sec} sec\n"
            print(msg, end="")
            f.write(msg)

    return StepResult(
        name=name,
        ok=(rc == 0),
        return_code=rc,
        elapsed_sec=time.time() - t0,
        log_file=str(log_path),
    )


def append_or_raise(
    res: StepResult,
    results: List[StepResult],
    continue_on_error: bool,
) -> None:
    results.append(res)
    if (not res.ok) and (not continue_on_error):
        raise RuntimeError(f"Step failed: {res.name}")


def _inject_gaussian_protocol_seeded(
    x: torch.Tensor,
    sigma_cfg: float,
    alpha: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if sigma_cfg <= 0:
        return x
    std = x.std(dim=1, keepdim=True)
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype) * (alpha * sigma_cfg) * std
    return x + noise


def run_seed_stability_significance(
    seeds: List[int],
    device_name: str,
    out_dir: Path,
    dry_run: bool = False,
) -> Dict[str, str]:
    _force_utf8_stdio()
    out_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        dummy = out_dir / "seed_stability_metrics.csv"
        pd.DataFrame([{"note": "dry_run"}]).to_csv(dummy, index=False)
        return {"seed_metrics_csv": str(dummy)}

    import run_pi_moe_evidence_bundle as rp

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data, _, ts_cfg, _, _, test_loader, scaler = rp.load_main_data()
    ultra = rp.build_ultra_model(ts_cfg, device)
    dlinear = rp.build_dlinear_model(device)
    itr = rp.build_itransformer_model(device, input_dim=data.shape[1])
    timemixer = rp.build_timemixer_model(device, input_dim=data.shape[1])

    predictors = {
        "Ultra-LSNT": lambda x: rp.predict_ultra(ultra, x),
        "DLinear": lambda x: rp.predict_dlinear(dlinear, x),
        "iTransformer": lambda x: rp.predict_itrans(itr, x),
        "TimeMixer": lambda x: rp.predict_timemixer(timemixer, x),
    }

    scenario_cfg = [
        ("clean", "none", 0.0),
        ("gaussian_60pct", "gaussian", 0.4),  # sigma_eff = 1.5 * sigma_cfg = 0.6
        ("spike_60pct", "spike", 0.6),
    ]

    rows: List[Dict] = []
    dm_rows: List[Dict] = []
    for sd in seeds:
        for scenario, mode, sev in scenario_cfg:
            preds_per_model: Dict[str, List[np.ndarray]] = {k: [] for k in predictors}
            trues_all: List[np.ndarray] = []
            for bidx, (x, y) in enumerate(test_loader):
                if mode == "none":
                    x_mod = x
                elif mode == "gaussian":
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed(sd * 100000 + bidx)
                    x_mod = _inject_gaussian_protocol_seeded(x, sigma_cfg=sev, alpha=1.5, generator=gen)
                elif mode == "spike":
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed(sd * 100000 + bidx)
                    x_mod = rp.inject_structured_fault(
                        x,
                        fault_type="spike",
                        severity=sev,
                        generator=gen,
                    )
                else:
                    raise ValueError(mode)

                y_dev = y.to(device)
                x_dev = x_mod.to(device)
                true_phys = y_dev * float(scaler.std[-1]) + float(scaler.mean[-1])
                trues_all.append(true_phys.cpu().numpy())

                for mname, pred_fn in predictors.items():
                    with torch.no_grad():
                        pred_norm = pred_fn(x_dev)
                    pred_phys = pred_norm * float(scaler.std[-1]) + float(scaler.mean[-1])
                    preds_per_model[mname].append(pred_phys.cpu().numpy())

            trues = np.concatenate(trues_all, axis=0)
            per_model_pred: Dict[str, np.ndarray] = {}
            for mname, plist in preds_per_model.items():
                pred = np.concatenate(plist, axis=0)
                per_model_pred[mname] = pred
                m = compute_metrics(pred, trues)
                rows.append(
                    {
                        "seed": sd,
                        "scenario": scenario,
                        "model": mname,
                        "R2": float(m["R2"]),
                        "RMSE": float(m["RMSE"]),
                        "MAE": float(m["MAE"]),
                    }
                )

            # DM style stat: compare Ultra-LSNT with each baseline in this seed+scenario.
            y_flat = trues.reshape(-1)
            ultra_flat = per_model_pred["Ultra-LSNT"].reshape(-1)
            e_u = (y_flat - ultra_flat) ** 2
            for baseline in ["DLinear", "iTransformer", "TimeMixer"]:
                b_flat = per_model_pred[baseline].reshape(-1)
                e_b = (y_flat - b_flat) ** 2
                d = e_u - e_b
                d_mean = float(np.mean(d))
                d_var = float(np.var(d, ddof=1))
                stat = 0.0 if d_var <= 1e-12 else d_mean / np.sqrt(d_var / max(len(d), 1))
                dm_rows.append(
                    {
                        "seed": sd,
                        "scenario": scenario,
                        "baseline": baseline,
                        "dm_stat_ultra_minus_baseline": stat,
                        "mean_loss_diff": d_mean,
                    }
                )

    df = pd.DataFrame(rows)
    dm_df = pd.DataFrame(dm_rows)
    metrics_csv = out_dir / "seed_stability_metrics.csv"
    dm_csv = out_dir / "seed_stability_dm_stats.csv"
    df.to_csv(metrics_csv, index=False)
    dm_df.to_csv(dm_csv, index=False)

    summary = (
        df.groupby(["scenario", "model"], as_index=False)
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            n_seed=("seed", "count"),
        )
        .sort_values(["scenario", "R2_mean"], ascending=[True, False])
    )
    summary_csv = out_dir / "seed_stability_summary_mean_std.csv"
    summary.to_csv(summary_csv, index=False)

    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    sig_rows: List[Dict] = []
    for scenario in sorted(df["scenario"].unique()):
        sub = df[df["scenario"] == scenario]
        u = sub[sub["model"] == "Ultra-LSNT"].sort_values("seed")
        for base in ["DLinear", "iTransformer", "TimeMixer"]:
            b = sub[sub["model"] == base].sort_values("seed")
            merged = pd.merge(
                u[["seed", "R2"]],
                b[["seed", "R2"]],
                on="seed",
                how="inner",
                suffixes=("_ultra", "_baseline"),
            )
            if len(merged) < 2 or wilcoxon is None:
                stat = np.nan
                p = np.nan
            else:
                try:
                    w = wilcoxon(
                        merged["R2_ultra"].values,
                        merged["R2_baseline"].values,
                        zero_method="wilcox",
                        alternative="greater",
                    )
                    stat = float(w.statistic)
                    p = float(w.pvalue)
                except Exception:
                    stat = np.nan
                    p = np.nan
            sig_rows.append(
                {
                    "scenario": scenario,
                    "baseline": base,
                    "n_seed": int(len(merged)),
                    "wilcoxon_stat": stat,
                    "wilcoxon_pvalue_one_sided_ultra_greater": p,
                }
            )
    sig_df = pd.DataFrame(sig_rows)
    sig_csv = out_dir / "seed_stability_significance_wilcoxon.csv"
    sig_df.to_csv(sig_csv, index=False)

    return {
        "seed_metrics_csv": str(metrics_csv),
        "seed_summary_csv": str(summary_csv),
        "seed_significance_csv": str(sig_csv),
        "seed_dm_csv": str(dm_csv),
    }


def _run_ssa_single_dataset(
    data: np.ndarray,
    seq_len: int,
    pred_len: int,
    seed: int,
    ssa_pop_size: int,
    ssa_iters: int,
) -> Dict[str, Dict[str, float]]:
    import run_ssa_elm_auditable_suite as ssa

    arr = data.astype(np.float64)
    target_idx = arr.shape[1] - 1
    n = len(arr)
    n_train = int(0.8 * n)
    if n_train <= seq_len + pred_len + 16:
        raise RuntimeError("Insufficient train samples for SSA-ELM windows.")

    train_raw = arr[:n_train]
    test_raw = arr[n_train:]

    x_scaler = StandardScaler().fit(train_raw)
    y_scaler = StandardScaler().fit(train_raw[:, target_idx].reshape(-1, 1))
    train_x = x_scaler.transform(train_raw).astype(np.float32)
    test_x = x_scaler.transform(test_raw).astype(np.float32)
    train_y = y_scaler.transform(train_raw[:, target_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)
    test_y = y_scaler.transform(test_raw[:, target_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)

    xtr, ytr = ssa.create_windows(train_x, train_y, seq_len, pred_len)
    xte, yte = ssa.create_windows(test_x, test_y, seq_len, pred_len)
    if len(xtr) < 256 or len(xte) < 8:
        raise RuntimeError("Insufficient windows for SSA-ELM.")

    in_dim = xtr.shape[1] * xtr.shape[2]
    xtr_flat = xtr.reshape(xtr.shape[0], in_dim).astype(np.float64)
    xte_flat = xte.reshape(xte.shape[0], in_dim).astype(np.float64)

    tr_obj = min(8000, int(0.75 * len(xtr_flat)))
    va_obj = min(2000, len(xtr_flat) - tr_obj)
    if va_obj <= 64:
        tr_obj = max(128, int(0.8 * len(xtr_flat)))
        va_obj = len(xtr_flat) - tr_obj
    if va_obj <= 32:
        raise RuntimeError("Insufficient objective split for SSA search.")

    x_obj_tr = xtr_flat[:tr_obj]
    y_obj_tr = ytr[:tr_obj]
    x_obj_va = xtr_flat[tr_obj : tr_obj + va_obj]
    y_obj_va = ytr[tr_obj : tr_obj + va_obj]

    bounds = [
        (48.0, 320.0),
        (-7.0, -2.0),
        (0.0, 2.0),
        (0.5, 2.0),
    ]

    def objective(v: np.ndarray) -> float:
        cfg = ssa.decode_cfg(v)
        model = ssa.ELMRegressor(cfg=cfg, seed=seed)
        model.fit(x_obj_tr, y_obj_tr)
        pred = model.predict(x_obj_va)
        r2 = r2_score(y_obj_va.reshape(-1), pred.reshape(-1))
        return float(-r2)

    best_vec, best_score, _ = ssa.ssa_optimize(
        objective_fn=objective,
        bounds=bounds,
        pop_size=ssa_pop_size,
        iters=ssa_iters,
        seed=seed,
    )
    best_cfg = ssa.decode_cfg(best_vec)
    model = ssa.ELMRegressor(cfg=best_cfg, seed=seed)
    model.fit(xtr_flat, ytr)

    pred_clean_s = model.predict(xte_flat)
    pred_clean = y_scaler.inverse_transform(pred_clean_s.reshape(-1, 1)).reshape(pred_clean_s.shape)
    true_clean = y_scaler.inverse_transform(yte.reshape(-1, 1)).reshape(yte.shape)
    m_clean = ssa.metrics(true_clean, pred_clean)

    xte_g = ssa.inject_gaussian(xte, sigma_eff=0.60, seed=seed).reshape(xte.shape[0], in_dim)
    pred_g_s = model.predict(xte_g.astype(np.float64))
    pred_g = y_scaler.inverse_transform(pred_g_s.reshape(-1, 1)).reshape(pred_g_s.shape)
    m_g = ssa.metrics(true_clean, pred_g)

    xte_sp = ssa.inject_spike(xte, severity=0.6, seed=seed + 1).reshape(xte.shape[0], in_dim)
    pred_sp_s = model.predict(xte_sp.astype(np.float64))
    pred_sp = y_scaler.inverse_transform(pred_sp_s.reshape(-1, 1)).reshape(pred_sp_s.shape)
    m_sp = ssa.metrics(true_clean, pred_sp)

    return {
        "clean": m_clean,
        "gaussian_60pct": m_g,
        "spike_60pct": m_sp,
        "search_score": {"best_objective": float(best_score)},
        "model_info": {
            "hidden_units": int(best_cfg.hidden_units),
            "ridge_alpha": float(best_cfg.ridge_alpha),
            "activation": str(best_cfg.activation),
            "input_scale": float(best_cfg.input_scale),
        },
    }


def _run_mamba_single_dataset(
    data: np.ndarray,
    seq_len: int,
    pred_len: int,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    import run_mamba_auditable_suite as mamba

    cfg = mamba.RunConfig(
        seed=seed,
        seq_len=seq_len,
        pred_len=pred_len,
        epochs=epochs,
        batch_size=batch_size,
    )
    target_idx = data.shape[1] - 1
    x_train, y_train, x_test, y_test, scaler = mamba.split_and_scale(data, target_idx=target_idx, cfg=cfg)
    train_loader = mamba.to_loader(x_train, y_train, cfg.batch_size, shuffle=True)
    test_loader = mamba.to_loader(x_test, y_test, cfg.batch_size, shuffle=False)

    model = mamba.AuditableMambaForecaster(
        input_dim=x_train.shape[-1],
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        d_state=cfg.d_state,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    clean = mamba.evaluate(model, test_loader, scaler, target_idx, device, perturb="none", severity=0.0)
    g = mamba.evaluate(model, test_loader, scaler, target_idx, device, perturb="gaussian", severity=0.6)
    sp = mamba.evaluate(model, test_loader, scaler, target_idx, device, perturb="spike", severity=0.6)
    return {"clean": clean, "gaussian_60pct": g, "spike_60pct": sp}


def run_cross_domain_completion(
    seed: int,
    seq_len: int,
    pred_len: int,
    ssa_pop_size: int,
    ssa_iters: int,
    mamba_epochs: int,
    mamba_batch_size: int,
    run_arima_svr: bool,
    run_mamba: bool,
    out_dir: Path,
    dry_run: bool = False,
) -> Dict[str, str]:
    _force_utf8_stdio()
    out_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        dummy = out_dir / "cross_domain_additional_models_clean.csv"
        pd.DataFrame([{"note": "dry_run"}]).to_csv(dummy, index=False)
        return {"cross_domain_clean_csv": str(dummy)}

    arsv = None
    if run_arima_svr:
        try:
            import run_arima_svr_auditable_suite as arsv_mod
            arsv = arsv_mod
        except Exception as e:
            raise RuntimeError(
                "ARIMA/SVR cross-domain phase requires statsmodels/sklearn dependencies. "
                "Please install requirements and rerun, or pass --skip-arima-svr-crossdomain. "
                f"Original error: {e}"
            ) from e

    clean_rows: List[Dict] = []
    robust_rows: List[Dict] = []
    model_meta_rows: List[Dict] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ds_name, csv_name, target in DATASETS:
        path = ROOT / csv_name
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        data, _ = load_csv_data(str(path), target)
        if data is None:
            raise RuntimeError(f"Failed to load dataset: {path}")

        if run_arima_svr:
            assert arsv is not None
            y_raw = data[:, -1].astype(np.float64)
            n = len(y_raw)
            split = 0.8
            n_train = int(n * split)
            y_train = y_raw[:n_train]
            scaler = StandardScaler()
            scaler.fit(y_train.reshape(-1, 1))
            y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
            anchors = arsv.build_anchor_indices(n=n, split=split, seq_len=seq_len, pred_len=pred_len)
            if len(anchors) < 8:
                raise RuntimeError(f"Insufficient anchors for {ds_name}")

            arima = arsv.ArimaAuditable().fit(y_train_scaled)
            svr, svr_fit_meta = arsv.fit_svr_one_step(y_train_scaled, seq_len=seq_len, seed=seed)
            model_meta_rows.append(
                {
                    "dataset": ds_name,
                    "model": "ARIMA (auditable)",
                    "meta": json.dumps(
                        {
                            "selected_order": list(arima.order),
                            "fit_aic": float(arima.fit_aic),
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            model_meta_rows.append(
                {
                    "dataset": ds_name,
                    "model": "SVR (auditable)",
                    "meta": json.dumps(svr_fit_meta, ensure_ascii=False),
                }
            )

            for model_name, model_obj, seed_offset in [
                ("ARIMA (auditable)", arima, 0),
                ("SVR (auditable)", svr, 1),
            ]:
                m_clean = arsv.evaluate_model(
                    y_raw=y_raw,
                    scaler=scaler,
                    anchors=anchors,
                    seq_len=seq_len,
                    pred_len=pred_len,
                    mode="clean",
                    model_obj=model_obj,
                    rng_seed=seed + seed_offset,
                )
                m_g = arsv.evaluate_model(
                    y_raw=y_raw,
                    scaler=scaler,
                    anchors=anchors,
                    seq_len=seq_len,
                    pred_len=pred_len,
                    mode="gaussian:0.60",
                    model_obj=model_obj,
                    rng_seed=seed + seed_offset,
                )
                m_sp = arsv.evaluate_model(
                    y_raw=y_raw,
                    scaler=scaler,
                    anchors=anchors,
                    seq_len=seq_len,
                    pred_len=pred_len,
                    mode="spike:0.6",
                    model_obj=model_obj,
                    rng_seed=seed + seed_offset + 17,
                )
                clean_rows.append(
                    {
                        "dataset": ds_name,
                        "model": model_name,
                        "R2": float(m_clean["R2"]),
                        "RMSE": float(m_clean["RMSE"]),
                        "MAE": float(m_clean["MAE"]),
                    }
                )
                robust_rows.extend(
                    [
                        {
                            "dataset": ds_name,
                            "model": model_name,
                            "scenario": "gaussian_60pct",
                            "R2": float(m_g["R2"]),
                            "RMSE": float(m_g["RMSE"]),
                            "MAE": float(m_g["MAE"]),
                        },
                        {
                            "dataset": ds_name,
                            "model": model_name,
                            "scenario": "spike_60pct",
                            "R2": float(m_sp["R2"]),
                            "RMSE": float(m_sp["RMSE"]),
                            "MAE": float(m_sp["MAE"]),
                        },
                    ]
                )

        ssa_res = _run_ssa_single_dataset(
            data=data,
            seq_len=seq_len,
            pred_len=pred_len,
            seed=seed,
            ssa_pop_size=ssa_pop_size,
            ssa_iters=ssa_iters,
        )
        clean_rows.append(
            {
                "dataset": ds_name,
                "model": "SSA-ELM",
                "R2": float(ssa_res["clean"]["R2"]),
                "RMSE": float(ssa_res["clean"]["RMSE"]),
                "MAE": float(ssa_res["clean"]["MAE"]),
            }
        )
        robust_rows.extend(
            [
                {
                    "dataset": ds_name,
                    "model": "SSA-ELM",
                    "scenario": "gaussian_60pct",
                    "R2": float(ssa_res["gaussian_60pct"]["R2"]),
                    "RMSE": float(ssa_res["gaussian_60pct"]["RMSE"]),
                    "MAE": float(ssa_res["gaussian_60pct"]["MAE"]),
                },
                {
                    "dataset": ds_name,
                    "model": "SSA-ELM",
                    "scenario": "spike_60pct",
                    "R2": float(ssa_res["spike_60pct"]["R2"]),
                    "RMSE": float(ssa_res["spike_60pct"]["RMSE"]),
                    "MAE": float(ssa_res["spike_60pct"]["MAE"]),
                },
            ]
        )
        model_meta_rows.append(
            {
                "dataset": ds_name,
                "model": "SSA-ELM",
                "meta": json.dumps(
                    {
                        **ssa_res["model_info"],
                        **ssa_res["search_score"],
                    },
                    ensure_ascii=False,
                ),
            }
        )

        if run_mamba:
            mb = _run_mamba_single_dataset(
                data=data,
                seq_len=seq_len,
                pred_len=pred_len,
                seed=seed,
                epochs=mamba_epochs,
                batch_size=mamba_batch_size,
                device=device,
            )
            clean_rows.append(
                {
                    "dataset": ds_name,
                    "model": "Mamba (auditable)",
                    "R2": float(mb["clean"]["R2"]),
                    "RMSE": float(mb["clean"]["RMSE"]),
                    "MAE": float(mb["clean"]["MAE"]),
                }
            )
            robust_rows.extend(
                [
                    {
                        "dataset": ds_name,
                        "model": "Mamba (auditable)",
                        "scenario": "gaussian_60pct",
                        "R2": float(mb["gaussian_60pct"]["R2"]),
                        "RMSE": float(mb["gaussian_60pct"]["RMSE"]),
                        "MAE": float(mb["gaussian_60pct"]["MAE"]),
                    },
                    {
                        "dataset": ds_name,
                        "model": "Mamba (auditable)",
                        "scenario": "spike_60pct",
                        "R2": float(mb["spike_60pct"]["R2"]),
                        "RMSE": float(mb["spike_60pct"]["RMSE"]),
                        "MAE": float(mb["spike_60pct"]["MAE"]),
                    },
                ]
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    clean_df = pd.DataFrame(clean_rows)
    robust_df = pd.DataFrame(robust_rows)
    meta_df = pd.DataFrame(model_meta_rows)
    clean_csv = out_dir / "cross_domain_additional_models_clean.csv"
    robust_csv = out_dir / "cross_domain_additional_models_robust.csv"
    meta_csv = out_dir / "cross_domain_additional_models_meta.csv"
    clean_df.to_csv(clean_csv, index=False)
    robust_df.to_csv(robust_csv, index=False)
    meta_df.to_csv(meta_csv, index=False)

    # Build expanded transfer matrix by merging existing 4-model matrix with added models.
    base_path = EVIDENCE_DIR / "multi_domain_transfer_matrix.csv"
    base_df = pd.read_csv(base_path) if base_path.exists() else pd.DataFrame()
    score_map: Dict[str, Dict[str, float]] = {}
    for m in ["ARIMA (auditable)", "SVR (auditable)", "SSA-ELM", "Mamba (auditable)"]:
        score_map[m] = {k: np.nan for k in TRANSFER_COL.values()}

    for _, r in clean_df.iterrows():
        ds = str(r["dataset"]).strip()
        m = str(r["model"]).strip()
        if ds not in TRANSFER_COL:
            continue
        if m not in score_map:
            continue
        score_map[m][TRANSFER_COL[ds]] = float(r["R2"])

    add_rows = []
    for m, cols in score_map.items():
        add_rows.append(
            {
                "model": m,
                "R2_wind_us": cols["R2_wind_us"],
                "R2_air_quality": cols["R2_air_quality"],
                "R2_gefcom_load": cols["R2_gefcom_load"],
                "source": "cross_domain_additional_models_clean",
            }
        )
    add_df = pd.DataFrame(add_rows)

    if not base_df.empty:
        keep_cols = ["model", "R2_wind_us", "R2_air_quality", "R2_gefcom_load", "source"]
        for c in keep_cols:
            if c not in base_df.columns:
                base_df[c] = np.nan
        merged = pd.concat([base_df[keep_cols], add_df], axis=0, ignore_index=True)
        merged = merged.drop_duplicates(subset=["model"], keep="last")
    else:
        merged = add_df

    transfer_all_csv = out_dir / "multi_domain_transfer_matrix_all_models.csv"
    merged.to_csv(transfer_all_csv, index=False)

    # Also copy to evidence directory for table drafting.
    promoted_csv = EVIDENCE_DIR / "multi_domain_transfer_matrix_all_models.csv"
    merged.to_csv(promoted_csv, index=False)

    return {
        "cross_domain_clean_csv": str(clean_csv),
        "cross_domain_robust_csv": str(robust_csv),
        "cross_domain_meta_csv": str(meta_csv),
        "transfer_all_models_csv": str(transfer_all_csv),
    }


def run_lite_robustness_completion(
    seed: int,
    seq_len: int,
    pred_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device_name: str,
    out_dir: Path,
    dry_run: bool = False,
) -> Dict[str, str]:
    _force_utf8_stdio()
    out_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        dummy = out_dir / "lite_robustness_windcn.csv"
        pd.DataFrame([{"note": "dry_run"}]).to_csv(dummy, index=False)
        return {"lite_robustness_csv": str(dummy)}

    import run_pi_moe_evidence_bundle as rp

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data, _ = load_csv_data(str(ROOT / "wind_final.csv"), "power")
    ts_cfg = TimeSeriesConfig(seq_len=seq_len, pred_len=pred_len, target="power")
    tr_cfg = TrainConfig(batch_size=batch_size, epochs=epochs, lr=lr, num_workers=0, patience=max(2, epochs))
    train_loader, _, test_loader, scaler = create_dataloaders(data, ts_cfg, tr_cfg)

    cfg = {
        "input_dim": data.shape[1],
        "hidden_dim": 96,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "num_blocks": 1,
        "num_experts": 2,
        "top_k": 1,
        "scales": [1, 2],
    }
    model = UltraLSNTLiteForecaster(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    hist = []
    model.train()
    for _ in range(epochs):
        losses = []
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad(set_to_none=True)
            out, aux = model(bx)
            loss = criterion(out, by) + 0.01 * aux
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        hist.append(float(np.mean(losses)) if losses else np.nan)

    def eval_scenario(scenario: str, mode: str, sev: float, seed_shift: int) -> Dict[str, float]:
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for bidx, (x, y) in enumerate(test_loader):
                if mode == "none":
                    x_mod = x
                elif mode == "gaussian":
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed((seed + seed_shift) * 100000 + bidx)
                    x_mod = _inject_gaussian_protocol_seeded(x, sigma_cfg=sev, alpha=1.5, generator=gen)
                elif mode == "spike":
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed((seed + seed_shift) * 100000 + bidx)
                    x_mod = rp.inject_structured_fault(x, "spike", sev, gen)
                else:
                    raise ValueError(mode)
                x_dev = x_mod.to(device)
                y_dev = y.to(device)
                pred_norm, _ = model(x_dev)
                pred = pred_norm * float(scaler.std[-1]) + float(scaler.mean[-1])
                true = y_dev * float(scaler.std[-1]) + float(scaler.mean[-1])
                preds.append(pred.cpu().numpy())
                trues.append(true.cpu().numpy())
        p = np.concatenate(preds, axis=0)
        t = np.concatenate(trues, axis=0)
        m = compute_metrics(p, t)
        return {"scenario": scenario, "R2": float(m["R2"]), "RMSE": float(m["RMSE"]), "MAE": float(m["MAE"])}

    rows = [
        eval_scenario("clean", "none", 0.0, 0),
        eval_scenario("gaussian_60pct", "gaussian", 0.4, 1),
        eval_scenario("spike_60pct", "spike", 0.6, 2),
    ]
    clean_r2 = rows[0]["R2"]
    for r in rows:
        r["model"] = "Ultra-LSNT-Lite(opt)"
        r["R2_clean"] = clean_r2
        r["relative_drop_pct"] = 100.0 * (clean_r2 - r["R2"]) / (abs(clean_r2) + 1e-12) if r["scenario"] != "clean" else 0.0

    csv_path = out_dir / "lite_robustness_windcn.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    meta = {
        "script": "run_submission_oneclick_4090.py",
        "phase": "lite_robustness_completion",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "device": str(device),
        "train_history": hist,
        "config": {
            "seq_len": seq_len,
            "pred_len": pred_len,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "model": cfg,
        },
        "output_csv": str(csv_path),
    }
    meta_path = out_dir / "lite_robustness_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Copy to evidence directory for matrix builder fallback.
    csv_evidence = EVIDENCE_DIR / "lite_robustness_windcn.csv"
    csv_evidence.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(csv_path).to_csv(csv_evidence, index=False)

    return {
        "lite_robustness_csv": str(csv_path),
        "lite_robustness_meta_json": str(meta_path),
    }


def write_summary(
    summary_path: Path,
    step_results: List[StepResult],
    artifact_map: Dict[str, str],
    args: argparse.Namespace,
) -> None:
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "script": "run_submission_oneclick_4090.py",
        "args": vars(args),
        "steps": [
            {
                "name": s.name,
                "ok": s.ok,
                "return_code": s.return_code,
                "elapsed_sec": s.elapsed_sec,
                "log_file": s.log_file,
                "note": s.note,
            }
            for s in step_results
        ],
        "artifacts": artifact_map,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="One-click submission experiment bundle on RTX 4090.")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default="42,123,2025")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--skip-full-backfill", action="store_true")
    ap.add_argument("--skip-cross-domain", action="store_true")
    ap.add_argument("--skip-seed-significance", action="store_true")
    ap.add_argument("--skip-universal-significance", action="store_true")
    ap.add_argument("--skip-lite-robustness", action="store_true")
    ap.add_argument("--skip-rebuild", action="store_true")

    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--pred-len", type=int, default=24)

    ap.add_argument("--ssa-pop-size", type=int, default=8)
    ap.add_argument("--ssa-iters", type=int, default=8)

    ap.add_argument("--skip-arima-svr-crossdomain", action="store_true")
    ap.add_argument("--skip-mamba-crossdomain", action="store_true")
    ap.add_argument("--mamba-epochs", type=int, default=4)
    ap.add_argument("--mamba-batch-size", type=int, default=256)

    ap.add_argument("--lite-epochs", type=int, default=3)
    ap.add_argument("--lite-batch-size", type=int, default=256)
    ap.add_argument("--lite-lr", type=float, default=1e-3)
    return ap.parse_args()


def main() -> int:
    _force_utf8_stdio()
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"submission_oneclick_4090_{ts}"
    out_dir = ROOT / "submission_oneclick_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    step_results: List[StepResult] = []
    artifacts: Dict[str, str] = {}

    try:
        if not args.skip_full_backfill:
            cmd = [args.python, "run_full_backfill_4090.py"]
            if args.continue_on_error:
                cmd.append("--continue-on-error")
            if args.dry_run:
                cmd.append("--dry-run")
            res = run_subprocess_step("phase1_full_backfill", cmd, log_dir, dry_run=args.dry_run)
            append_or_raise(res, step_results, args.continue_on_error)

        if not args.skip_cross_domain:
            t0 = time.time()
            try:
                out = run_cross_domain_completion(
                    seed=args.seed,
                    seq_len=args.seq_len,
                    pred_len=args.pred_len,
                    ssa_pop_size=args.ssa_pop_size,
                    ssa_iters=args.ssa_iters,
                    mamba_epochs=args.mamba_epochs,
                    mamba_batch_size=args.mamba_batch_size,
                    run_arima_svr=(not args.skip_arima_svr_crossdomain),
                    run_mamba=(not args.skip_mamba_crossdomain),
                    out_dir=out_dir,
                    dry_run=args.dry_run,
                )
                artifacts.update(out)
                step_results.append(
                    StepResult(
                        name="phase2_cross_domain_completion",
                        ok=True,
                        return_code=0,
                        elapsed_sec=time.time() - t0,
                        log_file=str(log_dir / "phase2_cross_domain_completion.log"),
                    )
                )
            except Exception as e:
                step_results.append(
                    StepResult(
                        name="phase2_cross_domain_completion",
                        ok=False,
                        return_code=1,
                        elapsed_sec=time.time() - t0,
                        log_file=str(log_dir / "phase2_cross_domain_completion.log"),
                        note=str(e),
                    )
                )
                if not args.continue_on_error:
                    raise

        if not args.skip_seed_significance:
            t0 = time.time()
            try:
                out = run_seed_stability_significance(
                    seeds=seeds,
                    device_name=args.device,
                    out_dir=out_dir,
                    dry_run=args.dry_run,
                )
                artifacts.update(out)
                step_results.append(
                    StepResult(
                        name="phase3_seed_stability_significance",
                        ok=True,
                        return_code=0,
                        elapsed_sec=time.time() - t0,
                        log_file=str(log_dir / "phase3_seed_stability_significance.log"),
                    )
                )
            except Exception as e:
                step_results.append(
                    StepResult(
                        name="phase3_seed_stability_significance",
                        ok=False,
                        return_code=1,
                        elapsed_sec=time.time() - t0,
                        log_file=str(log_dir / "phase3_seed_stability_significance.log"),
                        note=str(e),
                    )
                )
                if not args.continue_on_error:
                    raise

        if not args.skip_universal_significance:
            cmd = [args.python, "run_universal_significance.py"]
            res = run_subprocess_step(
                "phase4_universal_significance",
                cmd,
                log_dir,
                dry_run=args.dry_run,
            )
            append_or_raise(res, step_results, args.continue_on_error)
            artifacts["universal_significance_fig_glob"] = str(ROOT / "fig_dm_matrix_*.png")
            artifacts["universal_significance_bundle_dir"] = str(EVIDENCE_DIR / "universal_significance_bundle")

        if not args.skip_lite_robustness:
            t0 = time.time()
            try:
                out = run_lite_robustness_completion(
                    seed=args.seed,
                    seq_len=args.seq_len,
                    pred_len=args.pred_len,
                    epochs=args.lite_epochs,
                    batch_size=args.lite_batch_size,
                    lr=args.lite_lr,
                    device_name=args.device,
                    out_dir=out_dir,
                    dry_run=args.dry_run,
                )
                artifacts.update(out)
                step_results.append(
                    StepResult(
                        name="phase5_lite_robustness_completion",
                        ok=True,
                        return_code=0,
                        elapsed_sec=time.time() - t0,
                        log_file=str(log_dir / "phase5_lite_robustness_completion.log"),
                    )
                )
            except Exception as e:
                step_results.append(
                    StepResult(
                        name="phase5_lite_robustness_completion",
                        ok=False,
                        return_code=1,
                        elapsed_sec=time.time() - t0,
                        log_file=str(log_dir / "phase5_lite_robustness_completion.log"),
                        note=str(e),
                    )
                )
                if not args.continue_on_error:
                    raise

        if not args.skip_rebuild:
            cmd = [
                args.python,
                "build_main4_matrices_from_evidence.py",
                "--out_dir",
                str(EVIDENCE_DIR),
                "--audit_tag",
                "main4_matrix_build_submission_oneclick",
            ]
            res = run_subprocess_step("phase6_rebuild_main4_matrices", cmd, log_dir, dry_run=args.dry_run)
            append_or_raise(res, step_results, args.continue_on_error)

    except Exception as e:
        print(f"[ERROR] {e}")
        summary_path = log_dir / "run_summary.json"
        write_summary(summary_path, step_results, artifacts, args)
        print(f"[summary] {summary_path}")
        return 1

    summary_path = log_dir / "run_summary.json"
    write_summary(summary_path, step_results, artifacts, args)
    print(f"\n[done] logs: {log_dir}")
    print(f"[done] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
