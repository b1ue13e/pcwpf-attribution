"""Route A cross-site runner: PhysGATE vs FG-MoE vs raw, same evaluation protocol.

Same split/corruption/metrics protocol as run_pi_moe_full_training_evidence.py.
One command per (site, backbone, variant, seed); results append to a single
summary CSV (manifest-traceable).

Usage (from 05_实验代码与脚本):
    ../.venv/Scripts/python.exe run_physgate_cross_site.py \
        --site wind_cn --backbone ultra --guardrail physgate --seed 42 \
        --epochs 8 --max-train-batches 20 --max-eval-batches 10
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from physgate import (
    DEFAULT_Q,
    TurbineEnvelope,
    physgate_loss,
    rolling_turbulence_stats,
)
from ultra_lsnt_timeseries import TimeSeriesConfig, TrainConfig, compute_metrics, create_dataloaders, load_csv_data

# Reuse the FG-MoE harness's protocol functions unchanged (same split/corruption).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guardrail_backbones import build_backbone_model, model_forward  # noqa: E402
from robustness_faults import apply_fault_tensor as vectorized_fault  # noqa: E402
from run_pi_moe_full_training_evidence import (  # noqa: E402
    PhysicsConfig,
    estimate_curve_params,
    inject_gaussian_protocol,
    load_main_data,
    positive_tail_loss,
    ramp_alignment_loss,
    scheduled_regularization_weights,
    set_seed,
)


def inject_mixed_vectorized(
    x: torch.Tensor,
    generator: torch.Generator,
    *,
    sigma_min: float,
    sigma_max: float,
    drift_prob: float,
    quantization_prob: float,
) -> torch.Tensor:
    """Vectorized mixed SCADA corruption (gaussian + drift + quantization), same
    family as inject_mixed_scada_corruption but per-batch tensor ops only."""
    sigma = float(torch.empty(1).uniform_(sigma_min, sigma_max, generator=generator).item())
    out = inject_gaussian_protocol(x, sigma_cfg=sigma, alpha=1.5)
    if float(torch.rand(1, generator=generator).item()) < float(drift_prob):
        sev = 0.15 + 0.30 * float(torch.rand(1, generator=generator).item())
        out = vectorized_fault(out, "drift", severity=sev, alpha=1.5)
    if float(torch.rand(1, generator=generator).item()) < float(quantization_prob):
        sev = 0.15 + 0.30 * float(torch.rand(1, generator=generator).item())
        out = vectorized_fault(out, "quantization", severity=sev)
    return out

OUT_DIR = os.path.join("results", "physgate_cross_site")
SUMMARY_CSV = os.path.join(OUT_DIR, "summary.csv")
SITE_CONFIG_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site_configs.json")


def load_sites() -> Dict[str, dict]:
    with open(SITE_CONFIG_JSON, "r", encoding="utf-8") as f:
        return json.load(f)["sites"]


SITES = load_sites()


def resolve_backbone_checkpoint(args_base: str | None, site_cfg: dict, backbone: str = "ultra") -> str | None:
    cands = [args_base, site_cfg.get("backbone_checkpoint")]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backbone == "itransformer":
        cands += [
            os.path.join(repo_root, "10_模型检查点与权重", "checkpoints_ts", "iTransformer_best.pth"),
            os.path.join(repo_root, "checkpoints_ts", "iTransformer_best.pth"),
        ]
    else:
        cands += [
            os.path.join(repo_root, "10_模型检查点与权重", "checkpoints_ts", "main", "best_model.pth"),
            os.path.join(repo_root, "checkpoints_ts", "main", "best_model.pth"),
        ]
    return _resolve_existing(*cands)


def resolve_site_cfg(site: str, data: str | None) -> dict:
    cfg = dict(SITES[site])
    if data:
        cfg["data"] = data
    return cfg


def _resolve_existing(*candidates: str) -> str | None:
    for c in candidates:
        if not c:
            continue
        if os.path.exists(c):
            return c
    return None


def resolve_backbone_config(args_base: str | None, site_cfg: dict) -> str | None:
    """Resolve a model config JSON across local (Chinese-dir wrapper) and cloud (flat) layouts."""
    cands = [args_base, site_cfg.get("backbone_config")]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands += [
        os.path.join(repo_root, "10_模型检查点与权重", "checkpoints_ts", "main", "model_config.json"),
        os.path.join(repo_root, "checkpoints_ts", "main", "model_config.json"),
        os.path.join(repo_root, "05_实验代码与脚本", "..", "checkpoints_ts", "main", "model_config.json"),
    ]
    return _resolve_existing(*cands)


def resolve_envelope(site_cfg: dict, data: np.ndarray, feature_names: List[str]) -> TurbineEnvelope:
    """Prefer explicit nameplate constants; otherwise IEC-style identification of theta only."""
    t = site_cfg.get("turbine", {}) or {}
    p_r = t.get("p_rated_kw")
    v_in = t.get("v_in_mps")
    v_rated = t.get("v_rated_mps")
    v_out = t.get("v_out_mps")
    identified = False
    if None in (p_r, v_in, v_rated, v_out):
        phy = estimate_curve_params(data, feature_names, target=site_cfg["target"], train_ratio=0.7)
        p_r = phy.p_rated_kw if p_r is None else p_r
        v_in = phy.v_in_mps if v_in is None else v_in
        v_rated = phy.v_rated_mps if v_rated is None else v_rated
        v_out = phy.v_out_mps if v_out is None else v_out
        identified = True
    env = TurbineEnvelope(p_rated=float(p_r), v_in=float(v_in), v_rated=float(v_rated), v_out=float(v_out)).validate()
    return env, identified


# ---------------------------------------------------------------------------
# Guardrail-aware training-terms builder
# ---------------------------------------------------------------------------
def wind_speed_denorm(x: torch.Tensor, scaler, ws_idx: int) -> torch.Tensor:
    mean = float(scaler.mean[ws_idx])
    std = float(scaler.std[ws_idx])
    return x[:, -1, ws_idx] * std + mean  # last-step wind (de-normalized)


def build_guardrail_fn(
    guardrail: str,
    feature_names: List[str],
    scaler,
    phy: PhysicsConfig,
    wind_proxy_spec,
    q: float = DEFAULT_Q,
    dace_half_width: float = 1.0,
    gate_gamma: float = 1.0,
    gate_hard_slope: float = 0.0,
):
    """Return a callable (pred_kw, x) -> additional feasibility loss.

    wind_proxy_spec carries the FG-MoE proxy coefficients:
      - mode "auto"  -> per-site calibrated (fgmoe)
      - mode "fixed" -> Wind-CN frozen coefficients zero-shot transfer (fgmoe_zs)
    """
    ws_idx = feature_names.index(wind_proxy_spec.feature_names[0])
    theta = TurbineEnvelope(phy.p_rated_kw, phy.v_in_mps, phy.v_rated_mps, phy.v_out_mps)

    def fgmoe(pred_kw, x):
        from guardrail_proxy_utils import build_horizon_wind_proxy
        ws_proxy = build_horizon_wind_proxy(x, wind_proxy_spec).to(dtype=pred_kw.dtype)
        from run_pi_moe_full_training_evidence import physics_loss
        return physics_loss(pred_kw, ws_proxy, phy)

    def fgmoe_zs(pred_kw, x):
        from guardrail_proxy_utils import build_horizon_wind_proxy
        ws_proxy = build_horizon_wind_proxy(x, wind_proxy_spec).to(dtype=pred_kw.dtype)
        from run_pi_moe_full_training_evidence import physics_loss
        return physics_loss(pred_kw, ws_proxy, phy)

    def physgate(pred_kw, x):
        pred_pu = pred_kw / max(phy.p_rated_kw, 1e-6)
        ws = wind_speed_denorm(x, scaler, ws_idx)
        sigma, rho = rolling_turbulence_stats(
            x[:, :, ws_idx] * scaler.std[ws_idx] + scaler.mean[ws_idx]
        )
        return physgate_loss(pred_pu, ws, sigma, rho, theta, q=q, symmetric=False)

    def physgate_sym(pred_kw, x):
        pred_pu = pred_kw / max(phy.p_rated_kw, 1e-6)
        ws = wind_speed_denorm(x, scaler, ws_idx)
        sigma, rho = rolling_turbulence_stats(
            x[:, :, ws_idx] * scaler.std[ws_idx] + scaler.mean[ws_idx]
        )
        return physgate_loss(pred_pu, ws, sigma, rho, theta, q=q, symmetric=True)

    def none(pred_kw, x):
        return torch.zeros((), device=pred_kw.device, dtype=pred_kw.dtype)

    def dace_gate(pred_kw, x):
        from guardrail_dace import DACESpec, compute_ramp_gate, dace_penalty
        spec = DACESpec(
            p_rated_kw=phy.p_rated_kw, v_in_mps=phy.v_in_mps,
            v_rated_mps=phy.v_rated_mps, v_out_mps=phy.v_out_mps,
            half_width_mps=dace_half_width, gate_mode="ramp", gate_gamma=1.0,
        )
        v_hist = x[:, :, ws_idx] * scaler.std[ws_idx] + scaler.mean[ws_idx]
        gate = compute_ramp_gate(v_hist, gamma=gate_gamma, hard_slope=gate_hard_slope)
        v_last = v_hist[:, -1]
        proxy = v_last.unsqueeze(1).expand(-1, pred_kw.size(1))
        loss, _ = dace_penalty(
            pred_kw, proxy, spec, dace_half_width,
            torch.tensor(1.0, device=pred_kw.device, dtype=pred_kw.dtype),
            gate_weight=gate,
        )
        return loss

    def dace_ratedcap(pred_kw, x):
        # G1: on ramp windows, raise upper envelope to P_rated instead of the
        # lagged-proxy envelope (fix the systematic underestimate), keep full
        # penalty weight (gate_weight=None).
        from guardrail_dace import DACESpec, compute_ramp_gate, dace_penalty
        spec = DACESpec(
            p_rated_kw=phy.p_rated_kw, v_in_mps=phy.v_in_mps,
            v_rated_mps=phy.v_rated_mps, v_out_mps=phy.v_out_mps,
            half_width_mps=dace_half_width, gate_mode="ramp", gate_gamma=1.0,
        )
        v_hist = x[:, :, ws_idx] * scaler.std[ws_idx] + scaler.mean[ws_idx]
        ramp_mask = compute_ramp_gate(v_hist, gamma=0.0, hard_slope=gate_hard_slope)
        ramp_mask = (ramp_mask == 0.0).float()  # 1 where ramping
        v_last = v_hist[:, -1]
        proxy = v_last.unsqueeze(1).expand(-1, pred_kw.size(1))
        loss, _ = dace_penalty(
            pred_kw, proxy, spec, dace_half_width,
            torch.tensor(1.0, device=pred_kw.device, dtype=pred_kw.dtype),
            gate_weight=None, rated_cap=ramp_mask,
        )
        return loss

    def x_sym(pred_kw, x):
        # Option X (minimal): output-endogenous symmetric feasible band
        # (no wind-speed proxy in the soft term) + hard rated/nonneg constraints.
        from physgate import output_endogenous_sym_loss
        pred_pu = pred_kw / max(phy.p_rated_kw, 1e-6)
        soft = output_endogenous_sym_loss(pred_pu, k=2.0)
        hard = F.relu(pred_pu - 1.0).pow(2.0).mean() + F.relu(-pred_pu).pow(2.0).mean()
        return soft + hard

    return {
        "physgate": physgate, "fgmoe": fgmoe, "fgmoe_zs": fgmoe_zs,
        "none": none, "physgate_sym": physgate_sym,
        "dace_gate": dace_gate, "dace_ratedcap": dace_ratedcap, "x_sym": x_sym,
    }[guardrail]


def compute_training_terms_guard(
    model,
    backbone: str,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    temperature: float,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    guard_fn: Callable,
    tail_quantile: float,
) -> Dict[str, torch.Tensor]:
    pred, aux = model_forward(model, backbone, x, temperature=temperature)
    mse = F.mse_loss(pred, y)
    pred_kw = pred * target_std + target_mean
    true_kw = y * target_std + target_mean
    phy_l = guard_fn(pred_kw, x)
    tail_l = positive_tail_loss(pred, y, quantile=tail_quantile)
    ramp_l = ramp_alignment_loss(pred, y)
    return {
        "pred_norm": pred, "true_norm": y, "pred_kw": pred_kw, "true_kw": true_kw,
        "aux": aux, "mse": mse, "phy": phy_l, "tail": tail_l, "ramp": ramp_l,
    }


def train_guard_model(
    model,
    backbone: str,
    train_loader,
    val_loader,
    scaler,
    guard_fn: Callable,
    phy: PhysicsConfig,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    seed: int,
    max_train_batches: int = 0,
    max_eval_batches: int = 0,
    impute_mode: str = "none",
    impute_p: float = 0.1,
    **kw,
):
    if impute_mode == "ffill":
        from run_impute_vs_guardrail import mask_and_ffill
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
    target_mean = torch.tensor(float(scaler.mean[-1]), device=device, dtype=torch.float32)
    target_std = torch.tensor(float(scaler.std[-1]), device=device, dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 20260901)
    best_state, best_select = None, float("inf")
    history = []
    corruption_weight = kw.get("corruption_weight", 0.20)
    tail_weight = kw.get("tail_weight", 0.005)
    ramp_weight = kw.get("ramp_weight", 0.002)
    consistency_weight = kw.get("consistency_weight", 0.005)
    aux_weight = kw.get("aux_weight", 0.005)
    tail_quantile = kw.get("tail_quantile", 0.95)

    for ep in range(1, epochs + 1):
        w = scheduled_regularization_weights(
            ep, epochs, corruption_weight=corruption_weight, tail_weight=tail_weight,
            ramp_weight=ramp_weight, consistency_weight=consistency_weight,
        )
        corr_w, tail_w, ramp_w, cons_w = w["corruption_weight"], w["tail_weight"], w["ramp_weight"], w["consistency_weight"]
        model.train()
        tr = {"loss": 0.0, "phy": 0.0, "mse": 0.0}
        n_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break
            x, y = x.to(device), y.to(device)
            if impute_mode == "ffill":
                x = mask_and_ffill(x, impute_p, generator)
            optimizer.zero_grad(set_to_none=True)
            x_corr = inject_mixed_vectorized(
                x,
                generator=generator,
                sigma_min=kw.get("corruption_sigma_min", 0.04),
                sigma_max=kw.get("corruption_sigma_max", 0.18),
                drift_prob=kw.get("drift_prob", 0.25),
                quantization_prob=kw.get("quantization_prob", 0.20),
            )
            clean = compute_training_terms_guard(
                model, backbone, x, y, temperature=1.0,
                target_mean=target_mean, target_std=target_std, guard_fn=guard_fn, tail_quantile=tail_quantile,
            )
            corr = compute_training_terms_guard(
                model, backbone, x_corr, y, temperature=1.0,
                target_mean=target_mean, target_std=target_std, guard_fn=guard_fn, tail_quantile=tail_quantile,
            )
            clean_obj = clean["mse"] + phy.lambda2 * clean["phy"] + tail_w * clean["tail"] + ramp_w * clean["ramp"]
            corr_obj = corr["mse"] + phy.lambda2 * corr["phy"] + tail_w * corr["tail"] + ramp_w * corr["ramp"]
            consistency = F.smooth_l1_loss(corr["pred_norm"], clean["pred_norm"].detach())
            aux_term = 0.5 * (clean["aux"] + corr["aux"])
            loss = clean_obj + corr_w * corr_obj + cons_w * consistency + aux_weight * aux_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            tr["loss"] += float(loss.item())
            tr["phy"] += float(0.5 * (clean["phy"].item() + corr["phy"].item()))
            tr["mse"] += float(0.5 * (clean["mse"].item() + corr["mse"].item()))
            n_batches += 1

        # validation
        model.eval()
        val_select = float("inf")
        with torch.no_grad():
            vb = 0
            vs = 0.0
            for batch_idx, (x, y) in enumerate(val_loader):
                if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                    break
                x, y = x.to(device), y.to(device)
                terms = compute_training_terms_guard(
                    model, backbone, x, y, temperature=0.1,
                    target_mean=target_mean, target_std=target_std, guard_fn=guard_fn, tail_quantile=tail_quantile,
                )
                vs += float(terms["mse"].item() + phy.lambda2 * terms["phy"].item() + 0.10 * terms["tail"].item())
                vb += 1
            val_select = vs / max(vb, 1)
        ep_row = {
            "epoch": ep, "train_loss": tr["loss"] / max(n_batches, 1),
            "train_phy": tr["phy"] / max(n_batches, 1), "train_mse": tr["mse"] / max(n_batches, 1),
            "val_select": val_select,
        }
        history.append(ep_row)
        print(f"[epoch] {ep_row}", flush=True)
        if val_select < best_select:
            best_select = val_select
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return model, pd.DataFrame(history)


def eval_guard_model(
    model,
    backbone: str,
    test_loader,
    scaler,
    guard_fn: Callable,
    phy: PhysicsConfig,
    device: torch.device,
    *,
    sigma_cfg: float,
    max_eval_batches: int = 0,
    impute_mode: str = "none",
    impute_p: float = 0.1,
):
    """RMSE/R2 + feasibility violations + ramp/tail/error-exposure diagnostics."""
    if impute_mode == "ffill":
        from run_impute_vs_guardrail import mask_and_ffill
    model.eval()
    target_mean = float(scaler.mean[-1])
    target_std = float(scaler.std[-1])
    all_pred, all_true = [], []
    n_neg = n_over = n_all = 0
    for batch_idx, (x, y) in enumerate(test_loader):
        if max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break
        x, y = x.to(device), y.to(device)
        if impute_mode == "ffill":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(20260000 + batch_idx)
            x = mask_and_ffill(x, impute_p, gen)
        x_noisy = inject_gaussian_protocol(x, sigma_cfg=sigma_cfg, alpha=1.5)
        with torch.no_grad():
            pred, _ = model_forward(model, backbone, x_noisy, temperature=0.1)
        pred_kw = pred * target_std + target_mean
        true_kw = y * target_std + target_mean
        n_neg += int((pred_kw < 0).sum().item())
        n_over += int((pred_kw > phy.p_rated_kw).sum().item())
        n_all += int(pred_kw.numel())
        all_pred.append(pred_kw.detach().cpu().numpy())
        all_true.append(true_kw.detach().cpu().numpy())

    pred_np = np.concatenate(all_pred, 0)
    true_np = np.concatenate(all_true, 0)
    metrics = compute_metrics(pred_np, true_np)
    err = pred_np - true_np
    # ramp F1: windowed ramp event detection (> 10% rated per step) on hourly (4-step) horizons
    pred_ramp = np.abs(np.diff(pred_np, axis=1)).max(axis=1)
    true_ramp = np.abs(np.diff(true_np, axis=1)).max(axis=1)
    thr = 0.10 * phy.p_rated_kw
    tp = int(((true_ramp > thr) & (pred_ramp > thr)).sum())
    fp = int(((true_ramp <= thr) & (pred_ramp > thr)).sum())
    fn = int(((true_ramp > thr) & (pred_ramp <= thr)).sum())
    metrics.update({
        "neg_rate_pct": 100.0 * n_neg / max(n_all, 1),
        "over_rated_rate_pct": 100.0 * n_over / max(n_all, 1),
        "p95_abs_err": float(np.percentile(np.abs(err), 95)),
        "pos_error_exposure": float(np.mean(np.maximum(err, 0))),
        "neg_error_exposure": float(np.mean(np.maximum(-err, 0))),
        "ramp_f1": 2.0 * tp / max(2 * tp + fp + fn, 1),
    })
    return metrics


def append_summary_row(summary_csv: str, row: dict) -> None:
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if os.path.exists(summary_csv):
        df_row.to_csv(summary_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(summary_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Route A PhysGATE cross-site runner")
    parser.add_argument("--site", type=str, choices=list(SITES.keys()), required=True)
    parser.add_argument("--backbone", type=str, default="ultra", choices=["ultra", "itransformer"])
    parser.add_argument("--guardrail", type=str, default="physgate", choices=["physgate", "fgmoe", "fgmoe_zs", "none", "physgate_sym", "dace_gate", "dace_ratedcap", "x_sym"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lambda2", type=float, default=0.03)
    parser.add_argument("--q", type=float, default=2.0, help="PhysGATE safety factor (band width knob)")
    parser.add_argument("--impute-mode", type=str, default="none", choices=["none", "ffill"], help="G3 repair-first: forward-fill missingness before the model")
    parser.add_argument("--impute-p", type=float, default=0.1)
    parser.add_argument("--gate-gamma", type=float, default=1.0, help="DACE gate soft constant")
    parser.add_argument("--gate-hard-slope", type=float, default=0.0, help="DACE gate hard threshold (0=soft mode)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--fresh-base", action="store_true", help="Force random-init base model (ignore auto checkpoint)")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--summary-csv", type=str, default=SUMMARY_CSV)
    parser.add_argument("--base-config", type=str, default=None, help="Backbone config JSON (required when default path is not on cwd).")
    parser.add_argument("--base-checkpoint", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    site_cfg = resolve_site_cfg(args.site, args.data)
    target = args.target or site_cfg["target"]

    data, feature_names, ts_cfg, train_loader, val_loader, test_loader, scaler = load_main_data(
        site_cfg["data"], target, seq_len=args.seq_len, pred_len=args.pred_len,
        batch_size=args.batch_size, num_workers=0,
    )
    env, theta_identified = resolve_envelope(site_cfg, data, feature_names)
    phy = PhysicsConfig(env.p_rated, env.v_in, env.v_rated, env.v_out, lambda2=args.lambda2)

    from guardrail_proxy_utils import build_wind_proxy_spec, infer_wind_proxy_columns
    ws_cols = infer_wind_proxy_columns(feature_names, max_candidates=1)
    cal_subset = data[: int(len(data) * 0.7)]
    if args.guardrail == "fgmoe_zs":
        # Wind-CN frozen proxy coefficients, zero-shot transfer to any site.
        wind_proxy_spec = build_wind_proxy_spec(
            feature_names, scaler, ts_cfg.pred_len,
            calibration_data=None,
            calibration_mode="fixed", proxy_columns=ws_cols,
        )
    else:
        wind_proxy_spec = build_wind_proxy_spec(
            feature_names, scaler, ts_cfg.pred_len,
            calibration_data=cal_subset,
            calibration_mode="auto", proxy_columns=ws_cols,
        )
    dace_half_width = 1.0
    if args.guardrail == "dace_gate":
        from guardrail_dace import conformal_half_width
        ws_idx = feature_names.index(ws_cols[0])
        ws_train = data[: int(len(data) * 0.7), ws_idx]
        dace_half_width = conformal_half_width(ws_train[1:] - ws_train[:-1], alpha=0.1)
    guard_fn = build_guardrail_fn(
        args.guardrail, feature_names, scaler, phy, wind_proxy_spec,
        q=args.q, dace_half_width=dace_half_width,
        gate_gamma=args.gate_gamma, gate_hard_slope=args.gate_hard_slope,
    )

    base_config = resolve_backbone_config(args.base_config, site_cfg)
    if args.fresh_base:
        base_checkpoint = "NONE_FRESH"  # truthy non-existent path -> random init, no default checkpoint
    else:
        base_checkpoint = resolve_backbone_checkpoint(args.base_checkpoint, site_cfg, args.backbone)
    base_model = build_backbone_model(
        args.backbone, ts_cfg, device,
        input_dim=len(feature_names),
        data_ref=site_cfg["data"], dataset_name=site_cfg["name"],
        checkpoint_path=base_checkpoint,
        config_path=base_config,
    )

    run_tag = f"{args.site}_{args.backbone}_{args.guardrail}_q{args.q}_seed{args.seed}"
    ckpt_path = os.path.join(args.out_dir, f"{run_tag}.pth")
    hist_path = os.path.join(args.out_dir, f"{run_tag}_history.csv")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    base_loaded_from = getattr(base_model, "_load_report", {}).get("loaded_from") or ""
    base_from_checkpoint = bool(base_loaded_from) and os.path.exists(base_loaded_from)
    # raw (guardrail==none) = base checkpoint evaluated directly, no training.
    skip_training = (args.guardrail == "none") and base_from_checkpoint

    if os.path.exists(ckpt_path) and not args.force_train:
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict):
            base_model.load_state_dict(state, strict=False)
        hist_df = pd.read_csv(hist_path) if os.path.exists(hist_path) else pd.DataFrame()
        print(f"[load] reused checkpoint {ckpt_path}")
    elif skip_training:
        hist_df = pd.DataFrame()
        print(f"[skip-train] raw variant uses base checkpoint directly: {base_loaded_from}")
    else:
        model_trained, hist_df = train_guard_model(
            base_model, args.backbone, train_loader, val_loader, scaler, guard_fn, phy, device,
            epochs=args.epochs, lr=args.lr, seed=args.seed,
            max_train_batches=args.max_train_batches, max_eval_batches=args.max_eval_batches,
            impute_mode=args.impute_mode, impute_p=args.impute_p,
        )
        torch.save(model_trained.state_dict(), ckpt_path)
        hist_df.to_csv(hist_path, index=False)

    for scenario, sigma_cfg, sigma_eff in [("clean", 0.0, 0.0), ("severe", 0.4, 0.6)]:
        met = eval_guard_model(
            base_model, args.backbone, test_loader, scaler, guard_fn, phy, device,
            sigma_cfg=sigma_cfg, max_eval_batches=args.max_eval_batches,
            impute_mode=args.impute_mode, impute_p=args.impute_p,
        )
        row = {
            "site": args.site, "backbone": args.backbone, "guardrail": args.guardrail,
            "seed": args.seed, "scenario": scenario, "sigma_eff": sigma_eff,
            "theta_identified": theta_identified,
            **{k: met[k] for k in ["R2", "RMSE", "MAE", "neg_rate_pct", "over_rated_rate_pct",
                                   "p95_abs_err", "pos_error_exposure", "neg_error_exposure", "ramp_f1"]},
            "q": args.q, "lambda2": args.lambda2, "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        append_summary_row(args.summary_csv, row)
        print(json.dumps(row, ensure_ascii=False))

    print(f"\n[done] {run_tag} -> {args.summary_csv}")


if __name__ == "__main__":
    main()
