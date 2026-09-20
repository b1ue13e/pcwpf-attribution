#!/usr/bin/env python3
"""
Joint parameter analysis for Ultra-LSNT under deployment-oriented constraints.

Grid:
- N (num experts)
- Top-K
- tau (jump-gate threshold)
- lambda2 (physics guard weight)

Outputs:
- joint_parameter_full_windcn.csv
- joint_structural_nkt_windcn.csv
- joint_structural_pareto_windcn.csv
- joint_lambda2_tradeoff_anchor.csv
- fig_joint_nkt_pareto_3d.pdf
- fig_lambda2_tradeoff_anchor.pdf
- joint_parameter_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from ultra_lsnt_timeseries import (
    LSNTConfig,
    TimeSeriesConfig,
    TrainConfig,
    UltraLSNTForecaster,
    compute_metrics,
    create_dataloaders,
    load_csv_data,
)


matplotlib.use("Agg")


@dataclass
class JointConfig:
    seed: int = 42
    seq_len: int = 96
    pred_len: int = 24
    sigma_eff: float = 0.60
    n_grid: Tuple[int, ...] = (2, 4, 8)
    topk_grid: Tuple[int, ...] = (1, 2)
    tau_grid: Tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)
    lambda2_grid: Tuple[float, ...] = (0.0, 0.01, 0.1, 1.0)
    eval_batches: int = 24
    latency_runs: int = 80


def parse_int_grid(raw: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def parse_float_grid(raw: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inject_gaussian(x: torch.Tensor, sigma_eff: float) -> torch.Tensor:
    if sigma_eff <= 0:
        return x
    std = x.std(dim=1, keepdim=True)
    return x + torch.randn_like(x) * sigma_eff * std


def power_envelope_pu(ws: np.ndarray, v_in: float, v_rated: float, v_out: float) -> np.ndarray:
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
    env_pu = power_envelope_pu(ws_proxy_mps, v_in=v_in, v_rated=v_rated, v_out=v_out)
    if lambda2 <= 0:
        return pred_kw

    neg = np.maximum(-pred_pu, 0.0)
    over = np.maximum(pred_pu - env_pu, 0.0)
    pred_pu = pred_pu + lambda2 * neg - lambda2 * over
    pred_pu = np.clip(pred_pu, -0.2, 1.5)
    return pred_pu * p_rated


def partial_load_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    own = model.state_dict()
    keep = {}
    skipped = 0
    for k, v in state_dict.items():
        if k in own and own[k].shape == v.shape:
            keep[k] = v
        else:
            skipped += 1
    missing, unexpected = model.load_state_dict(keep, strict=False)
    return {
        "loaded_tensors": len(keep),
        "skipped_tensors": skipped,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


def load_model_with_overrides(
    num_experts: int,
    top_k: int,
    tau: float,
    seq_len: int,
    pred_len: int,
    device: torch.device,
) -> Tuple[UltraLSNTForecaster, Dict[str, int]]:
    cfgj = json.load(open("checkpoints_ts/main/model_config.json", "r", encoding="utf-8"))
    cfg = LSNTConfig(
        input_dim=cfgj["input_dim"],
        hidden_dim=cfgj["hidden_dim"],
        output_dim=1,
        num_blocks=cfgj["num_blocks"],
        num_experts=int(num_experts),
        top_k=int(min(top_k, num_experts)),
        dropout=cfgj.get("dropout", 0.1),
        router_z_loss_coef=cfgj.get("router_z_loss_coef", 0.01),
        router_aux_loss_coef=cfgj.get("router_aux_loss_coef", 0.01),
        router_jitter_noise=cfgj.get("router_jitter_noise", 0.1),
        skip_threshold=float(tau),
        dual_mode_expansion=cfgj.get("dual_mode_expansion", 4),
        probabilistic_mode="none",
        heteroscedastic_moe=False,
    )
    ts_cfg = TimeSeriesConfig(seq_len=seq_len, pred_len=pred_len, target="power")
    model = UltraLSNTForecaster(cfg, ts_cfg).to(device)

    ckpt = torch.load("checkpoints_ts/main/best_model.pth", map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    load_info = partial_load_state_dict(model, state)
    model.eval()
    return model, load_info


def measure_latency_ms(model: torch.nn.Module, sample_x: torch.Tensor, runs: int = 80) -> float:
    with torch.no_grad():
        for _ in range(25):
            _ = model(sample_x, temperature=0.1)
    ts = []
    with torch.no_grad():
        for _ in range(runs):
            if sample_x.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(sample_x, temperature=0.1)
            if sample_x.device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(np.asarray(ts, dtype=np.float64)))


def pareto_structural(df: pd.DataFrame) -> pd.DataFrame:
    # maximize R2, minimize latency and model size
    vals = df[["R2", "latency_ms", "model_size_mib"]].to_numpy(dtype=np.float64)
    keep = np.ones(len(vals), dtype=bool)
    for i in range(len(vals)):
        if not keep[i]:
            continue
        for j in range(len(vals)):
            if i == j:
                continue
            dominates = (
                vals[j, 0] >= vals[i, 0]
                and vals[j, 1] <= vals[i, 1]
                and vals[j, 2] <= vals[i, 2]
                and (vals[j, 0] > vals[i, 0] or vals[j, 1] < vals[i, 1] or vals[j, 2] < vals[i, 2])
            )
            if dominates:
                keep[i] = False
                break
    out = df.loc[keep].copy()
    out = out.sort_values(["latency_ms", "model_size_mib", "R2"], ascending=[True, True, False])
    return out


def nearest_value(candidates: Sequence[float], target: float) -> float:
    arr = np.asarray(list(candidates), dtype=np.float64)
    return float(arr[np.argmin(np.abs(arr - target))])


def save_pdf_and_png(fig: plt.Figure, pdf_path: str, dpi: int = 600) -> str:
    fig.savefig(pdf_path, bbox_inches="tight")
    png_path = os.path.splitext(pdf_path)[0] + ".png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    return png_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Joint N-K-tau-lambda2 analysis for Ultra-LSNT (RTX 4090)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=24)
    ap.add_argument("--sigma_eff", type=float, default=0.60)
    ap.add_argument("--n_grid", type=str, default="2,4,8")
    ap.add_argument("--topk_grid", type=str, default="1,2")
    ap.add_argument("--tau_grid", type=str, default="0.3,0.5,0.7,0.9")
    ap.add_argument("--lambda2_grid", type=str, default="0.0,0.01,0.1,1.0")
    ap.add_argument("--eval_batches", type=int, default=24)
    ap.add_argument("--latency_runs", type=int, default=80)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out_dir", type=str, default=os.path.join("results", "supplementary_evidence"))
    ap.add_argument("--audit_tag", type=str, default="joint_parameter_4090")
    args = ap.parse_args()

    cfg = JointConfig(
        seed=args.seed,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        sigma_eff=args.sigma_eff,
        n_grid=parse_int_grid(args.n_grid),
        topk_grid=parse_int_grid(args.topk_grid),
        tau_grid=parse_float_grid(args.tau_grid),
        lambda2_grid=parse_float_grid(args.lambda2_grid),
        eval_batches=args.eval_batches,
        latency_runs=args.latency_runs,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data, feature_names = load_csv_data("wind_final.csv", "power")
    ws_idx = feature_names.index("windspeed") if "windspeed" in feature_names else 0

    ts_cfg = TimeSeriesConfig(seq_len=cfg.seq_len, pred_len=cfg.pred_len, target="power")
    tr_cfg = TrainConfig(batch_size=128, epochs=1, num_workers=0)
    _, _, test_loader, scaler = create_dataloaders(data, ts_cfg, tr_cfg, train_ratio=0.79, val_ratio=0.01)

    p_rated = float(np.max(data[: int(len(data) * 0.8), -1]))
    v_in, v_rated, v_out = 0.6, 7.55, 25.0

    rows: List[Dict] = []
    for n_exp in cfg.n_grid:
        for top_k in cfg.topk_grid:
            if top_k > n_exp:
                continue
            for tau in cfg.tau_grid:
                model, load_info = load_model_with_overrides(
                    num_experts=n_exp,
                    top_k=top_k,
                    tau=tau,
                    seq_len=cfg.seq_len,
                    pred_len=cfg.pred_len,
                    device=device,
                )

                sample_x, _ = next(iter(test_loader))
                sample_x = sample_x[:1].to(device)
                latency_ms = measure_latency_ms(model, sample_x, runs=cfg.latency_runs)
                model_size_mib = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024.0 * 1024.0)
                active_params_m = sum(p.numel() for p in model.parameters()) / 1e6

                pred_all, true_all, ws_all = [], [], []
                with torch.no_grad():
                    for bi, (xb, yb) in enumerate(test_loader):
                        if bi >= cfg.eval_batches:
                            break
                        xb = xb.to(device)
                        yb = yb.to(device)
                        xb_noisy = inject_gaussian(xb, cfg.sigma_eff)
                        pred, _ = model(xb_noisy, temperature=0.1)

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
                    pred_adj = apply_lambda2_guard(
                        pred_kw=pred_np,
                        ws_proxy_mps=ws_np,
                        p_rated=p_rated,
                        v_in=v_in,
                        v_rated=v_rated,
                        v_out=v_out,
                        lambda2=lambda2,
                    )
                    m = compute_metrics(pred_adj, true_np)
                    over_rated = 100.0 * float((pred_adj > p_rated).mean())
                    negative = 100.0 * float((pred_adj < 0).mean())
                    rows.append(
                        {
                            "N": int(n_exp),
                            "K": int(top_k),
                            "tau": float(tau),
                            "lambda2": float(lambda2),
                            "sigma_eff": float(cfg.sigma_eff),
                            "R2": float(m["R2"]),
                            "RMSE": float(m["RMSE"]),
                            "MAE": float(m["MAE"]),
                            "latency_ms": float(latency_ms),
                            "model_size_mib": float(model_size_mib),
                            "active_params_m": float(active_params_m),
                            "compute_ratio": float(top_k / max(n_exp, 1)),
                            "over_rated_pct": float(over_rated),
                            "negative_pct": float(negative),
                            "loaded_tensors": int(load_info["loaded_tensors"]),
                            "skipped_tensors": int(load_info["skipped_tensors"]),
                        }
                    )

    df = pd.DataFrame(rows)
    full_csv = os.path.join(args.out_dir, "joint_parameter_full_windcn.csv")
    df.to_csv(full_csv, index=False)

    lambda2_base = float(sorted(cfg.lambda2_grid)[0])
    struct_df = df[df["lambda2"] == lambda2_base].copy()
    struct_csv = os.path.join(args.out_dir, "joint_structural_nkt_windcn.csv")
    struct_df.to_csv(struct_csv, index=False)

    pareto_df = pareto_structural(struct_df)
    pareto_csv = os.path.join(args.out_dir, "joint_structural_pareto_windcn.csv")
    pareto_df.to_csv(pareto_csv, index=False)

    anchor_n = int(nearest_value(cfg.n_grid, 8))
    anchor_k = int(nearest_value([k for k in cfg.topk_grid if k <= anchor_n], 2))
    anchor_tau = float(nearest_value(cfg.tau_grid, 0.5))
    anchor = df[(df["N"] == anchor_n) & (df["K"] == anchor_k) & (df["tau"] == anchor_tau)].sort_values("lambda2")
    if anchor.empty:
        anchor = df.groupby("lambda2", as_index=False)[["over_rated_pct", "R2"]].mean()
        anchor["N"] = anchor_n
        anchor["K"] = anchor_k
        anchor["tau"] = anchor_tau
    tradeoff_csv = os.path.join(args.out_dir, "joint_lambda2_tradeoff_anchor.csv")
    anchor.to_csv(tradeoff_csv, index=False)

    # Figure 1: 3D structural Pareto (latency-size-R2)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        struct_df["latency_ms"],
        struct_df["model_size_mib"],
        struct_df["R2"],
        c=struct_df["tau"],
        cmap="viridis",
        s=32,
    )
    ax.scatter(
        pareto_df["latency_ms"],
        pareto_df["model_size_mib"],
        pareto_df["R2"],
        c="red",
        marker="^",
        s=70,
        label="Pareto",
    )
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Model Size (MiB)")
    ax.set_zlabel("R2")
    ax.set_title(f"N-K-tau Joint Analysis (lambda2={lambda2_base})")
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.1, label="tau")
    fig.tight_layout()
    fig1 = os.path.join(args.out_dir, "fig_joint_nkt_pareto_3d.pdf")
    fig.savefig(fig1, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: lambda2 tradeoff at anchor setting, synced with the manuscript figure.
    anchor_plot = anchor.groupby("lambda2", as_index=False)[["R2", "over_rated_pct"]].mean().sort_values("lambda2")
    x = anchor_plot["lambda2"].to_numpy(dtype=np.float64)
    y_violation = anchor_plot["over_rated_pct"].to_numpy(dtype=np.float64)
    y_r2 = anchor_plot["R2"].to_numpy(dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(7.1, 4.3))
    ax1.set_facecolor("#F2F2F2")
    ax1.grid(True, color="#D0D0D0", linestyle="-", linewidth=0.8, alpha=0.75)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.axvspan(0.01, 0.10, color="#CFE8CF", alpha=0.45, zorder=0)
    ax1.text(
        0.055,
        0.93,
        "recommended operating region",
        transform=ax1.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.4,
        color="#1F5D38",
    )

    l1, = ax1.plot(
        x,
        y_violation,
        color="#C81D25",
        marker="o",
        linewidth=2.1,
        label="Rated-power exceedance (%)",
    )
    ax1.set_xlabel(r"$\lambda_2$")
    ax1.set_ylabel("Rated-power exceedance (%)", color="#C81D25")
    ax1.tick_params(axis="y", labelcolor="#C81D25")
    ax1.set_xlim(float(np.min(x)) - 0.01, float(np.max(x)) + 0.03)

    ax2 = ax1.twinx()
    l2, = ax2.plot(
        x,
        y_r2,
        color="#0D67B5",
        marker="s",
        linestyle=(0, (5, 3)),
        linewidth=2.0,
        label=r"$R^2@0.60$",
    )
    ax2.set_ylabel(r"$R^2@\sigma_{\mathrm{eff}}=0.60$", color="#0D67B5")
    ax2.tick_params(axis="y", labelcolor="#0D67B5")
    ax2.set_ylim(float(np.min(y_r2)) - 0.06, float(np.max(y_r2)) + 0.05)

    ax1.set_title(r"$\lambda_2$ trade-off under severe slice ($\sigma_{\mathrm{eff}}=0.60$)")
    ax1.legend(handles=[l1, l2], loc="upper right", frameon=False, fontsize=8.0)
    fig.tight_layout()
    fig2 = os.path.join(args.out_dir, "fig_lambda2_tradeoff_anchor.pdf")
    fig2_png = save_pdf_and_png(fig, fig2)
    plt.close(fig)

    meta = {
        "script": "run_joint_parameter_analysis_4090.py",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "audit_tag": args.audit_tag,
        "dataset": "wind_final.csv",
        "split": "80_20_chronological",
        "window": {"seq_len": cfg.seq_len, "pred_len": cfg.pred_len},
        "device": str(device),
        "grid": {
            "N": list(cfg.n_grid),
            "K": list(cfg.topk_grid),
            "tau": list(cfg.tau_grid),
            "lambda2": list(cfg.lambda2_grid),
            "sigma_eff": cfg.sigma_eff,
        },
        "anchor": {"N": anchor_n, "K": anchor_k, "tau": anchor_tau},
        "physics": {
            "p_rated_kw": p_rated,
            "v_in_mps": v_in,
            "v_rated_mps": v_rated,
            "v_out_mps": v_out,
        },
        "outputs": {
            "full_csv": full_csv,
            "struct_csv": struct_csv,
            "pareto_csv": pareto_csv,
            "tradeoff_csv": tradeoff_csv,
            "fig_joint_3d": fig1,
            "fig_lambda2": fig2,
            "fig_lambda2_png": fig2_png,
        },
    }
    meta_json = os.path.join(args.out_dir, "joint_parameter_meta.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[done] outputs:")
    print(f"- {full_csv}")
    print(f"- {struct_csv}")
    print(f"- {pareto_csv}")
    print(f"- {tradeoff_csv}")
    print(f"- {fig1}")
    print(f"- {fig2}")
    print(f"- {meta_json}")


if __name__ == "__main__":
    main()
