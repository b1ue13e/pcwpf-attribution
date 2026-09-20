import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ultra_lsnt_timeseries import (
    LSNTConfig,
    TimeSeriesConfig,
    TrainConfig,
    UltraLSNTForecaster,
    compute_metrics,
    create_dataloaders,
    load_csv_data,
)


@dataclass
class SoaConfig:
    seed: int = 42
    seq_len: int = 96
    pred_len: int = 24
    sigma_eff: float = 0.60
    k_grid: Tuple[int, ...] = (1, 2, 4)
    tau_grid: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    lambda2_grid: Tuple[float, ...] = (0.0, 0.01, 0.1, 1.0)
    eval_batches: int = 24
    latency_runs: int = 60


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inject_gaussian(x: torch.Tensor, sigma_eff: float) -> torch.Tensor:
    if sigma_eff <= 0:
        return x
    std = x.std(dim=1, keepdim=True)
    return x + torch.randn_like(x) * sigma_eff * std


def power_envelope_pu(ws: np.ndarray, p_rated: float, v_in: float, v_rated: float, v_out: float) -> np.ndarray:
    out = np.zeros_like(ws, dtype=np.float64)
    m1 = (ws >= v_in) & (ws < v_rated)
    m2 = (ws >= v_rated) & (ws <= v_out)
    denom = max(v_rated - v_in, 1e-6)
    out[m1] = np.clip((ws[m1] - v_in) / denom, 0.0, 1.0) ** 3
    out[m2] = 1.0
    return out


def apply_lambda2_guard(
    pred_kw: np.ndarray,
    ws_proxy_mps: np.ndarray,
    p_rated: float,
    v_in: float,
    v_rated: float,
    v_out: float,
    lambda2: float,
) -> np.ndarray:
    pred_pu = pred_kw / max(p_rated, 1e-6)
    env_pu = power_envelope_pu(ws_proxy_mps, p_rated, v_in, v_rated, v_out)

    if lambda2 <= 0:
        return pred_kw

    neg = np.maximum(-pred_pu, 0.0)
    over = np.maximum(pred_pu - env_pu, 0.0)
    pred_pu = pred_pu + lambda2 * neg - lambda2 * over

    # prevent extreme corrections from causing numerical artifacts
    pred_pu = np.clip(pred_pu, -0.2, 1.5)
    return pred_pu * p_rated


def load_model_with_overrides(top_k: int, tau: float, device: torch.device):
    cfgj = json.load(open("checkpoints_ts/main/model_config.json", "r", encoding="utf-8"))
    cfg = LSNTConfig(
        input_dim=cfgj["input_dim"],
        hidden_dim=cfgj["hidden_dim"],
        output_dim=1,
        num_blocks=cfgj["num_blocks"],
        num_experts=cfgj["num_experts"],
        top_k=top_k,
        dropout=cfgj.get("dropout", 0.1),
        router_z_loss_coef=cfgj.get("router_z_loss_coef", 0.01),
        router_aux_loss_coef=cfgj.get("router_aux_loss_coef", 0.01),
        router_jitter_noise=cfgj.get("router_jitter_noise", 0.1),
        skip_threshold=tau,
        dual_mode_expansion=cfgj.get("dual_mode_expansion", 4),
        probabilistic_mode="none",
        heteroscedastic_moe=False,
    )
    ts_cfg = TimeSeriesConfig(seq_len=96, pred_len=24, target="power")
    model = UltraLSNTForecaster(cfg, ts_cfg).to(device)

    ckpt = torch.load("checkpoints_ts/main/best_model.pth", map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def measure_latency(model: torch.nn.Module, sample_x: torch.Tensor, runs: int = 60) -> float:
    with torch.no_grad():
        for _ in range(20):
            _ = model(sample_x, temperature=0.1)
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(sample_x, temperature=0.1)
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    # maximize R2, minimize latency and over_rated_pct
    vals = df[["R2", "latency_ms", "over_rated_pct"]].to_numpy()
    n = len(vals)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            dominates = (
                vals[j, 0] >= vals[i, 0]
                and vals[j, 1] <= vals[i, 1]
                and vals[j, 2] <= vals[i, 2]
                and (
                    vals[j, 0] > vals[i, 0]
                    or vals[j, 1] < vals[i, 1]
                    or vals[j, 2] < vals[i, 2]
                )
            )
            if dominates:
                keep[i] = False
                break
    out = df.loc[keep].copy()
    out = out.sort_values(["latency_ms", "over_rated_pct", "R2"], ascending=[True, True, False])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Joint sensitivity SOA grid for Ultra-LSNT")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=24)
    ap.add_argument("--out_dir", type=str, default=os.path.join("results", "supplementary_evidence"))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--audit_tag", type=str, default="joint_sensitivity_soa")
    args = ap.parse_args()

    cfg = SoaConfig(seed=args.seed, seq_len=args.seq_len, pred_len=args.pred_len)
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    data, feature_names = load_csv_data("wind_final.csv", "power")
    ws_idx = feature_names.index("windspeed")

    ts_cfg = TimeSeriesConfig(seq_len=cfg.seq_len, pred_len=cfg.pred_len, target="power")
    tr_cfg = TrainConfig(batch_size=128, epochs=1, num_workers=0)
    # 80/20 chronological split, val_ratio=0
    # Keep the primary protocol as 80/20 chronological, but allocate a tiny validation slice
    # to satisfy dataloader construction constraints in the shared utility.
    train_loader, val_loader, test_loader, scaler = create_dataloaders(data, ts_cfg, tr_cfg, train_ratio=0.79, val_ratio=0.01)

    p_rated = float(np.max(data[: int(len(data) * 0.8), -1]))
    v_in, v_rated, v_out = 0.6, 7.55, 25.0

    rows = []
    for k in cfg.k_grid:
        for tau in cfg.tau_grid:
            model = load_model_with_overrides(top_k=k, tau=tau, device=device)

            # single-sample latency under current compute path
            sample_x, _ = next(iter(test_loader))
            sample_x = sample_x[:1].to(device)
            latency_ms = measure_latency(model, sample_x, runs=cfg.latency_runs)

            # run noisy evaluation once per (K,tau), then apply lambda2 post-guard variants
            pred_all, true_all, ws_all = [], [], []
            with torch.no_grad():
                for bi, (xb, yb) in enumerate(test_loader):
                    if bi >= cfg.eval_batches:
                        break
                    xb = xb.to(device)
                    yb = yb.to(device)
                    xb_noisy = inject_gaussian(xb, cfg.sigma_eff)
                    pred, _ = model(xb_noisy, temperature=0.1)

                    # inverse scale target
                    t_mean = float(scaler.mean[-1])
                    t_std = float(scaler.std[-1])
                    pred_kw = pred.cpu().numpy() * t_std + t_mean
                    true_kw = yb.cpu().numpy() * t_std + t_mean

                    ws = xb_noisy[:, -1, ws_idx].cpu().numpy() * float(scaler.std[ws_idx]) + float(scaler.mean[ws_idx])
                    ws = np.repeat(ws[:, None], pred_kw.shape[1], axis=1)

                    pred_all.append(pred_kw)
                    true_all.append(true_kw)
                    ws_all.append(ws)

            pred_np = np.concatenate(pred_all, axis=0)
            true_np = np.concatenate(true_all, axis=0)
            ws_np = np.concatenate(ws_all, axis=0)

            for lambda2 in cfg.lambda2_grid:
                pred_adj = apply_lambda2_guard(pred_np, ws_np, p_rated, v_in, v_rated, v_out, lambda2=lambda2)

                m = compute_metrics(pred_adj, true_np)
                over_rated = 100.0 * float((pred_adj > p_rated).mean())
                negative = 100.0 * float((pred_adj < 0).mean())
                cut_mask = ((ws_np < v_in) | (ws_np > v_out))
                cut_region_nonzero = 100.0 * float(((cut_mask) & (pred_adj > 0)).mean())

                rows.append(
                    {
                        "K": int(k),
                        "tau": float(tau),
                        "lambda2": float(lambda2),
                        "sigma_eff": float(cfg.sigma_eff),
                        "R2": float(m["R2"]),
                        "latency_ms": float(latency_ms),
                        "over_rated_pct": float(over_rated),
                        "negative_pct": float(negative),
                        "cut_region_nonzero_pct": float(cut_region_nonzero),
                    }
                )

    df = pd.DataFrame(rows)
    soa_path = os.path.join(args.out_dir, "joint_sensitivity_soa_windcn.csv")
    df.to_csv(soa_path, index=False)

    pf = pareto_front(df)
    pf_path = os.path.join(args.out_dir, "joint_sensitivity_pareto_frontier.csv")
    pf.to_csv(pf_path, index=False)

    # 3D figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(df["latency_ms"], df["R2"], df["over_rated_pct"], c=df["lambda2"], cmap="viridis", s=28)
    ax.scatter(pf["latency_ms"], pf["R2"], pf["over_rated_pct"], c="red", s=60, marker="^", label="Pareto frontier")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("R2")
    ax.set_zlabel("Over-rated (%)")
    ax.set_title("SOA Pareto under sigma_eff=0.60")
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1, label="lambda2")
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, "fig_soa_3d_pareto.pdf")
    fig.savefig(fig_path)
    plt.close(fig)

    meta = {
        "script": "run_joint_sensitivity_soa.py",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset": "wind_final.csv",
        "split": "80_20_chronological",
        "window": {"seq_len": cfg.seq_len, "pred_len": cfg.pred_len},
        "device": str(device),
        "seed": cfg.seed,
        "grid": {
            "K": list(cfg.k_grid),
            "tau": list(cfg.tau_grid),
            "lambda2": list(cfg.lambda2_grid),
            "sigma_eff": cfg.sigma_eff,
        },
        "physics": {
            "p_rated_kw": p_rated,
            "v_in": v_in,
            "v_rated": v_rated,
            "v_out": v_out,
        },
        "outputs": {
            "soa_csv": soa_path,
            "pareto_csv": pf_path,
            "figure": fig_path,
        },
        "audit_tag": args.audit_tag,
    }

    with open(os.path.join(args.out_dir, "joint_sensitivity_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Do not overwrite global split manifests from sensitivity runs.


if __name__ == "__main__":
    main()
