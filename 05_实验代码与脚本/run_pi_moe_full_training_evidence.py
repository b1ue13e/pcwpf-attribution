import argparse
import copy
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from guardrail_backbones import (
    BACKBONE_CHOICES,
    DEFAULT_ULTRA_WINDCN_CHECKPOINT,
    DEFAULT_ULTRA_WINDCN_CONFIG,
    backbone_display_name,
    build_backbone_model,
    full_guardrail_name,
    model_forward,
    partial_load_state_dict,
    save_backbone_checkpoint,
    train_base_backbone,
)
from guardrail_proxy_utils import (
    add_wind_proxy_arguments,
    build_horizon_wind_proxy,
    build_wind_proxy_spec,
    infer_wind_proxy_columns,
    proxy_spec_to_dict,
)
from ultra_lsnt_timeseries import TimeSeriesConfig, TrainConfig, compute_metrics, create_dataloaders, load_csv_data


OUT_DIR = os.path.join("results", "supplementary_evidence")
DEFAULT_DATA = "wind_final.csv"
DEFAULT_TARGET = "power"
DEFAULT_DATASET_NAME = "Wind_CN"
DEFAULT_OUTPUT_TAG = "windcn"
DEFAULT_BASE_CONFIG = DEFAULT_ULTRA_WINDCN_CONFIG
DEFAULT_BASE_CHECKPOINT = DEFAULT_ULTRA_WINDCN_CHECKPOINT


@dataclass
class PhysicsConfig:
    p_rated_kw: float
    v_in_mps: float
    v_rated_mps: float
    v_out_mps: float
    lambda2: float = 0.1


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_output_tag(raw: str) -> str:
    tag = re.sub(r"[^0-9A-Za-z]+", "_", str(raw).strip()).strip("_").lower()
    return tag or "dataset"


def infer_output_tag(data_ref: str, dataset_name: str | None = None) -> str:
    ref = str(data_ref).replace("\\", "/").lower()
    if ref in {"wind_final.csv", "./wind_final.csv", "wind_final"}:
        return DEFAULT_OUTPUT_TAG
    if dataset_name and sanitize_output_tag(dataset_name) in {DEFAULT_OUTPUT_TAG, "wind_cn"}:
        return DEFAULT_OUTPUT_TAG
    basis = dataset_name if dataset_name else Path(str(data_ref)).stem
    return sanitize_output_tag(basis)


def default_dataset_name(data_ref: str, dataset_name: str | None = None) -> str:
    if dataset_name:
        return dataset_name
    return DEFAULT_DATASET_NAME if infer_output_tag(data_ref) == DEFAULT_OUTPUT_TAG else Path(str(data_ref)).stem


def tagged_output_path(out_dir: str, stem: str, suffix: str, output_tag: str) -> str:
    filename = f"{stem}{suffix}" if output_tag == DEFAULT_OUTPUT_TAG else f"{stem}_{output_tag}{suffix}"
    return os.path.join(out_dir, filename)


def build_fgmoe_output_paths(out_dir: str, output_tag: str) -> Dict[str, str]:
    return {
        "history_csv": tagged_output_path(out_dir, "pi_moe_full_training_history", ".csv", output_tag),
        "checkpoint": tagged_output_path(out_dir, "pi_moe_full_checkpoint", ".pth", output_tag),
        "compare_csv": tagged_output_path(out_dir, "pi_moe_full_vs_ultra_clean_sigma06", ".csv", output_tag),
        "structured_csv": tagged_output_path(
            out_dir,
            "pi_moe_full_structured_scada_fault_robustness",
            ".csv",
            output_tag,
        ),
        "meta_json": tagged_output_path(out_dir, "pi_moe_full_training_meta", ".json", output_tag),
    }


def load_main_data(
    data_ref: str = DEFAULT_DATA,
    target: str = DEFAULT_TARGET,
    *,
    seq_len: int = 96,
    pred_len: int = 24,
    batch_size: int = 128,
    num_workers: int = 0,
):
    data, feature_names = load_csv_data(str(data_ref), target)
    ts_cfg = TimeSeriesConfig(seq_len=seq_len, pred_len=pred_len, target=target)
    train_cfg = TrainConfig(batch_size=batch_size, epochs=1, num_workers=num_workers)
    train_loader, val_loader, test_loader, scaler = create_dataloaders(data, ts_cfg, train_cfg)
    return data, feature_names, ts_cfg, train_loader, val_loader, test_loader, scaler


def estimate_curve_params(
    data: np.ndarray,
    feature_names: List[str],
    target: str = DEFAULT_TARGET,
    train_ratio: float = 0.7,
) -> PhysicsConfig:
    n_train = int(len(data) * train_ratio)
    train = data[:n_train]
    ws_col = infer_wind_proxy_columns(feature_names, max_candidates=1)[0]
    ws_idx = feature_names.index(ws_col)
    p_idx = feature_names.index(target)

    ws = train[:, ws_idx]
    p = train[:, p_idx]

    p_rated = float(np.max(p))
    active_mask = p >= 0.05 * p_rated
    if active_mask.any():
        v_in = float(np.quantile(ws[active_mask], 0.01))
    else:
        v_in = 3.0

    bins = np.arange(np.floor(ws.min() * 10) / 10, np.ceil(ws.max() * 10) / 10 + 0.11, 0.1)
    idx = np.digitize(ws, bins) - 1
    centers = []
    med = []
    for i in range(len(bins) - 1):
        mask = idx == i
        if mask.sum() >= 30:
            centers.append((bins[i] + bins[i + 1]) / 2)
            med.append(float(np.median(p[mask])))
    centers = np.array(centers)
    med = np.array(med)
    rated_mask = med >= 0.95 * p_rated
    if rated_mask.any():
        v_rated = float(centers[rated_mask][0])
    else:
        v_rated = max(v_in + 1.0, 12.0)

    return PhysicsConfig(
        p_rated_kw=p_rated,
        v_in_mps=v_in,
        v_rated_mps=v_rated,
        v_out_mps=25.0,
        lambda2=0.1,
    )


def inject_gaussian_protocol(x: torch.Tensor, sigma_cfg: float, alpha: float = 1.5) -> torch.Tensor:
    if sigma_cfg <= 0:
        return x
    std = x.std(dim=1, keepdim=True)
    noise = torch.randn_like(x) * (alpha * sigma_cfg) * std
    return x + noise


def power_envelope_pu(v: torch.Tensor, phy: PhysicsConfig) -> torch.Tensor:
    out = torch.zeros_like(v)
    mask_mid = (v >= phy.v_in_mps) & (v < phy.v_rated_mps)
    mask_hi = (v >= phy.v_rated_mps) & (v <= phy.v_out_mps)
    denom = max(phy.v_rated_mps - phy.v_in_mps, 1e-6)
    out = torch.where(mask_mid, ((v - phy.v_in_mps) / denom).clamp(min=0.0, max=1.0) ** 3, out)
    out = torch.where(mask_hi, torch.ones_like(out), out)
    return out


def physics_loss(pred_kw: torch.Tensor, ws_proxy_mps: torch.Tensor, phy: PhysicsConfig) -> torch.Tensor:
    pred_pu = pred_kw / max(phy.p_rated_kw, 1e-6)
    ws_env = power_envelope_pu(ws_proxy_mps, phy)
    neg_pen = F.relu(-pred_pu) ** 2
    over_pen = F.relu(pred_pu - 1.0) ** 2
    env_pen = F.relu(pred_pu - ws_env) ** 2
    cut_mask = (ws_proxy_mps < phy.v_in_mps) | (ws_proxy_mps > phy.v_out_mps)
    cut_pen = torch.where(cut_mask, pred_pu ** 2, torch.zeros_like(pred_pu))
    return (neg_pen + over_pen + env_pen + cut_pen).mean()


def positive_tail_loss(pred_kw: torch.Tensor, true_kw: torch.Tensor, quantile: float = 0.9) -> torch.Tensor:
    pos_err = F.relu(pred_kw - true_kw).reshape(-1)
    if pos_err.numel() == 0:
        return pred_kw.new_tensor(0.0)
    q = float(min(max(quantile, 0.5), 0.995))
    thr = torch.quantile(pos_err.detach(), q)
    tail = pos_err[pos_err >= thr]
    if tail.numel() == 0:
        return pos_err.mean()
    return tail.mean()


def ramp_alignment_loss(pred_kw: torch.Tensor, true_kw: torch.Tensor) -> torch.Tensor:
    if pred_kw.size(1) < 2 or true_kw.size(1) < 2:
        return pred_kw.new_tensor(0.0)
    pred_ramp = pred_kw[:, 1:] - pred_kw[:, :-1]
    true_ramp = true_kw[:, 1:] - true_kw[:, :-1]
    return F.l1_loss(pred_ramp, true_ramp)


def scheduled_regularization_weights(
    epoch: int,
    total_epochs: int,
    *,
    corruption_weight: float,
    tail_weight: float,
    ramp_weight: float,
    consistency_weight: float,
) -> Dict[str, float]:
    if epoch <= min(2, total_epochs):
        return {
            "corruption_weight": min(float(corruption_weight), 0.10),
            "tail_weight": 0.0,
            "ramp_weight": 0.0,
            "consistency_weight": 0.0,
        }
    if epoch <= min(4, total_epochs):
        return {
            "corruption_weight": min(float(corruption_weight), 0.15),
            "tail_weight": min(float(tail_weight), 0.002),
            "ramp_weight": min(float(ramp_weight), 0.001),
            "consistency_weight": min(float(consistency_weight), 0.002),
        }
    return {
        "corruption_weight": float(corruption_weight),
        "tail_weight": float(tail_weight),
        "ramp_weight": float(ramp_weight),
        "consistency_weight": float(consistency_weight),
    }


def inject_linear_drift(x: torch.Tensor, severity: float, generator: torch.Generator) -> torch.Tensor:
    x = x.clone()
    bsz, seq_len, n_feat = x.shape
    time_axis = torch.linspace(0.0, 1.0, seq_len, device=x.device, dtype=x.dtype).view(seq_len, 1)
    std = x.std(dim=1, keepdim=False) + 1e-6
    for b in range(bsz):
        n_channels = max(1, int(round(n_feat * (0.20 + 0.40 * severity))))
        channels = torch.randperm(n_feat, generator=generator)[:n_channels]
        slope = (torch.rand(n_channels, generator=generator) * 2.0 - 1.0) * (0.20 + 0.80 * severity)
        slope = slope.to(device=x.device, dtype=x.dtype)
        scale = std[b, channels]
        drift = time_axis * (slope * scale).unsqueeze(0)
        x[b, :, channels] = x[b, :, channels] + drift
    return x


def inject_quantization(x: torch.Tensor, severity: float) -> torch.Tensor:
    x = x.clone()
    scale = x.std(dim=1, keepdim=True).mean(dim=2, keepdim=True) + 1e-6
    step = (0.05 + 0.15 * float(severity)) * scale
    return torch.round(x / step) * step


def inject_mixed_scada_corruption(
    x: torch.Tensor,
    *,
    generator: torch.Generator,
    sigma_min: float,
    sigma_max: float,
    structured_fault_prob: float,
    drift_prob: float,
    quantization_prob: float,
) -> Tuple[torch.Tensor, Dict[str, float | str]]:
    severity = float(torch.empty(1).uniform_(0.15, 0.75, generator=generator).item())
    sigma = float(torch.empty(1).uniform_(sigma_min, sigma_max, generator=generator).item())
    out = inject_gaussian_protocol(x, sigma_cfg=sigma, alpha=1.5)

    applied_fault = "gaussian"
    if float(torch.rand(1, generator=generator).item()) < float(drift_prob):
        out = inject_linear_drift(out, severity=severity, generator=generator)
        applied_fault += "+drift"
    if float(torch.rand(1, generator=generator).item()) < float(quantization_prob):
        out = inject_quantization(out, severity=severity)
        applied_fault += "+quant"
    if float(torch.rand(1, generator=generator).item()) < float(structured_fault_prob):
        fault_bank = ["flatline", "spike", "dropout"]
        fault_idx = int(torch.randint(0, len(fault_bank), (1,), generator=generator).item())
        fault_type = fault_bank[fault_idx]
        out = inject_structured_fault(out, fault_type=fault_type, severity=severity, generator=generator)
        applied_fault += f"+{fault_type}"

    return out, {
        "severity": severity,
        "sigma_cfg": sigma,
        "fault_recipe": applied_fault,
    }


def compute_training_terms(
    model,
    backbone: str,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    temperature: float,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    wind_proxy_spec,
    phy: PhysicsConfig,
    tail_quantile: float,
) -> Dict[str, torch.Tensor]:
    pred, aux = model_forward(model, backbone, x, temperature=temperature)
    mse = F.mse_loss(pred, y)
    pred_kw = pred * target_std + target_mean
    true_kw = y * target_std + target_mean
    ws_proxy = build_horizon_wind_proxy(x, wind_proxy_spec).to(dtype=pred_kw.dtype)
    phy_l = physics_loss(pred_kw, ws_proxy, phy)
    tail_l = positive_tail_loss(pred, y, quantile=tail_quantile)
    ramp_l = ramp_alignment_loss(pred, y)
    return {
        "pred_norm": pred,
        "true_norm": y,
        "pred_kw": pred_kw,
        "true_kw": true_kw,
        "aux": aux,
        "mse": mse,
        "phy": phy_l,
        "tail": tail_l,
        "ramp": ramp_l,
    }


def eval_model_with_noise(
    model,
    backbone: str,
    test_loader: torch.utils.data.DataLoader,
    scaler,
    wind_proxy_spec,
    phy: PhysicsConfig,
    device: torch.device,
    sigma_cfg: float,
    *,
    max_eval_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    target_mean = float(scaler.mean[-1])
    target_std = float(scaler.std[-1])

    all_pred = []
    all_true = []
    n_neg = 0
    n_over = 0
    n_cut_viol = 0
    n_all = 0

    for batch_idx, (x, y) in enumerate(test_loader):
        if max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break
        x = x.to(device)
        y = y.to(device)
        x_noisy = inject_gaussian_protocol(x, sigma_cfg=sigma_cfg, alpha=1.5)

        with torch.no_grad():
            pred, _ = model_forward(model, backbone, x_noisy, temperature=0.1)

        pred_kw = pred * target_std + target_mean
        true_kw = y * target_std + target_mean
        ws_proxy = build_horizon_wind_proxy(x_noisy, wind_proxy_spec).to(dtype=pred_kw.dtype)

        n_neg += int((pred_kw < 0).sum().item())
        n_over += int((pred_kw > phy.p_rated_kw).sum().item())
        cut_mask = (ws_proxy < phy.v_in_mps) | (ws_proxy > phy.v_out_mps)
        n_cut_viol += int(((pred_kw > 0) & cut_mask).sum().item())
        n_all += int(pred_kw.numel())

        all_pred.append(pred_kw.detach().cpu().numpy())
        all_true.append(true_kw.detach().cpu().numpy())

    pred_np = np.concatenate(all_pred, axis=0)
    true_np = np.concatenate(all_true, axis=0)
    metrics = compute_metrics(pred_np, true_np)
    metrics["neg_rate_pct"] = 100.0 * n_neg / max(n_all, 1)
    metrics["over_rated_rate_pct"] = 100.0 * n_over / max(n_all, 1)
    metrics["cut_region_nonzero_rate_pct"] = 100.0 * n_cut_viol / max(n_all, 1)
    return metrics


def train_pi_moe_full(
    model,
    backbone: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    scaler,
    wind_proxy_spec,
    phy: PhysicsConfig,
    device: torch.device,
    *,
    epochs: int = 8,
    lr: float = 2e-5,
    weight_decay: float = 0.005,
    aux_weight: float = 0.005,
    corruption_weight: float = 0.20,
    tail_weight: float = 0.005,
    ramp_weight: float = 0.002,
    consistency_weight: float = 0.005,
    tail_quantile: float = 0.95,
    corruption_sigma_min: float = 0.04,
    corruption_sigma_max: float = 0.18,
    structured_fault_prob: float = 0.35,
    drift_prob: float = 0.25,
    quantization_prob: float = 0.20,
    seed: int = 42,
    max_train_batches: int = 0,
    max_eval_batches: int = 0,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    target_mean = torch.tensor(float(scaler.mean[-1]), device=device, dtype=torch.float32)
    target_std = torch.tensor(float(scaler.std[-1]), device=device, dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 20260327)

    best_state = None
    best_select = float("inf")
    history = []

    for ep in range(1, epochs + 1):
        epoch_weights = scheduled_regularization_weights(
            ep,
            epochs,
            corruption_weight=corruption_weight,
            tail_weight=tail_weight,
            ramp_weight=ramp_weight,
            consistency_weight=consistency_weight,
        )
        corr_w = float(epoch_weights["corruption_weight"])
        tail_w = float(epoch_weights["tail_weight"])
        ramp_w = float(epoch_weights["ramp_weight"])
        cons_w = float(epoch_weights["consistency_weight"])
        model.train()
        tr_loss = 0.0
        tr_clean_mse = 0.0
        tr_corr_mse = 0.0
        tr_phy = 0.0
        tr_tail = 0.0
        tr_ramp = 0.0
        tr_cons = 0.0
        tr_aux = 0.0
        tr_severity = 0.0
        n_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            x_corr, corr_meta = inject_mixed_scada_corruption(
                x,
                generator=generator,
                sigma_min=corruption_sigma_min,
                sigma_max=corruption_sigma_max,
                structured_fault_prob=structured_fault_prob,
                drift_prob=drift_prob,
                quantization_prob=quantization_prob,
            )
            clean_terms = compute_training_terms(
                model,
                backbone,
                x,
                y,
                temperature=1.0,
                target_mean=target_mean,
                target_std=target_std,
                wind_proxy_spec=wind_proxy_spec,
                phy=phy,
                tail_quantile=tail_quantile,
            )
            corr_terms = compute_training_terms(
                model,
                backbone,
                x_corr,
                y,
                temperature=1.0,
                target_mean=target_mean,
                target_std=target_std,
                wind_proxy_spec=wind_proxy_spec,
                phy=phy,
                tail_quantile=tail_quantile,
            )
            clean_obj = clean_terms["mse"] + phy.lambda2 * clean_terms["phy"] + tail_w * clean_terms["tail"] + ramp_w * clean_terms["ramp"]
            corr_obj = corr_terms["mse"] + phy.lambda2 * corr_terms["phy"] + tail_w * corr_terms["tail"] + ramp_w * corr_terms["ramp"]
            consistency = F.smooth_l1_loss(corr_terms["pred_norm"], clean_terms["pred_norm"].detach())
            aux_term = 0.5 * (clean_terms["aux"] + corr_terms["aux"])
            loss = clean_obj + corr_w * corr_obj + cons_w * consistency + aux_weight * aux_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            tr_loss += float(loss.item())
            tr_clean_mse += float(clean_terms["mse"].item())
            tr_corr_mse += float(corr_terms["mse"].item())
            tr_phy += float(0.5 * (clean_terms["phy"].item() + corr_terms["phy"].item()))
            tr_tail += float(0.5 * (clean_terms["tail"].item() + corr_terms["tail"].item()))
            tr_ramp += float(0.5 * (clean_terms["ramp"].item() + corr_terms["ramp"].item()))
            tr_cons += float(consistency.item())
            tr_aux += float(aux_term.item())
            tr_severity += float(corr_meta["severity"])
            n_batches += 1

        model.eval()
        val_clean_mse = 0.0
        val_corr_mse = 0.0
        val_phy = 0.0
        val_tail = 0.0
        val_ramp = 0.0
        val_cons = 0.0
        val_aux = 0.0
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                    break
                x = x.to(device)
                y = y.to(device)
                x_corr, _ = inject_mixed_scada_corruption(
                    x,
                    generator=generator,
                    sigma_min=corruption_sigma_min,
                    sigma_max=corruption_sigma_max,
                    structured_fault_prob=structured_fault_prob,
                    drift_prob=drift_prob,
                    quantization_prob=quantization_prob,
                )
                clean_terms = compute_training_terms(
                    model,
                    backbone,
                    x,
                    y,
                    temperature=0.1,
                    target_mean=target_mean,
                    target_std=target_std,
                    wind_proxy_spec=wind_proxy_spec,
                    phy=phy,
                    tail_quantile=tail_quantile,
                )
                corr_terms = compute_training_terms(
                    model,
                    backbone,
                    x_corr,
                    y,
                    temperature=0.1,
                    target_mean=target_mean,
                    target_std=target_std,
                    wind_proxy_spec=wind_proxy_spec,
                    phy=phy,
                    tail_quantile=tail_quantile,
                )
                clean_obj = clean_terms["mse"] + phy.lambda2 * clean_terms["phy"] + tail_w * clean_terms["tail"] + ramp_w * clean_terms["ramp"]
                corr_obj = corr_terms["mse"] + phy.lambda2 * corr_terms["phy"] + tail_w * corr_terms["tail"] + ramp_w * corr_terms["ramp"]
                consistency = F.smooth_l1_loss(corr_terms["pred_norm"], clean_terms["pred_norm"].detach())
                aux_term = 0.5 * (clean_terms["aux"] + corr_terms["aux"])
                loss = clean_obj + corr_w * corr_obj + cons_w * consistency + aux_weight * aux_term
                val_loss += float(loss.item())
                val_clean_mse += float(clean_terms["mse"].item())
                val_corr_mse += float(corr_terms["mse"].item())
                val_phy += float(0.5 * (clean_terms["phy"].item() + corr_terms["phy"].item()))
                val_tail += float(0.5 * (clean_terms["tail"].item() + corr_terms["tail"].item()))
                val_ramp += float(0.5 * (clean_terms["ramp"].item() + corr_terms["ramp"].item()))
                val_cons += float(consistency.item())
                val_aux += float(aux_term.item())
                val_batches += 1

        val_clean_mse_avg = val_clean_mse / max(val_batches, 1)
        val_corr_mse_avg = val_corr_mse / max(val_batches, 1)
        val_phy_avg = val_phy / max(val_batches, 1)
        val_tail_avg = val_tail / max(val_batches, 1)
        val_select_score = val_clean_mse_avg + 0.25 * val_corr_mse_avg + 0.05 * val_phy_avg + 0.10 * val_tail_avg
        ep_row = {
            "epoch": ep,
            "train_loss": tr_loss / max(n_batches, 1),
            "train_clean_mse": tr_clean_mse / max(n_batches, 1),
            "train_corr_mse": tr_corr_mse / max(n_batches, 1),
            "train_phy": tr_phy / max(n_batches, 1),
            "train_tail": tr_tail / max(n_batches, 1),
            "train_ramp": tr_ramp / max(n_batches, 1),
            "train_consistency": tr_cons / max(n_batches, 1),
            "train_aux": tr_aux / max(n_batches, 1),
            "train_corruption_severity": tr_severity / max(n_batches, 1),
            "eff_corruption_weight": corr_w,
            "eff_tail_weight": tail_w,
            "eff_ramp_weight": ramp_w,
            "eff_consistency_weight": cons_w,
            "val_loss": val_loss / max(val_batches, 1),
            "val_clean_mse": val_clean_mse_avg,
            "val_corr_mse": val_corr_mse_avg,
            "val_phy": val_phy_avg,
            "val_tail": val_tail_avg,
            "val_ramp": val_ramp / max(val_batches, 1),
            "val_consistency": val_cons / max(val_batches, 1),
            "val_aux": val_aux / max(val_batches, 1),
            "val_select_score": val_select_score,
        }
        history.append(ep_row)
        print(f"[epoch] backbone={backbone} {ep_row}", flush=True)

        if ep_row["val_select_score"] < best_select:
            best_select = ep_row["val_select_score"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return model, pd.DataFrame(history)


def inject_structured_fault(
    x: torch.Tensor,
    fault_type: str,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    x = x.clone()
    bsz, seq_len, n_feat = x.shape

    if fault_type == "flatline":
        n_segments = max(1, int(round(1 + 2 * severity)))
        seg_len = max(2, int(round(seq_len * (0.08 + 0.35 * severity))))
        n_channels = max(1, int(round(n_feat * (0.2 + 0.5 * severity))))
        for b in range(bsz):
            for _ in range(n_segments):
                start = int(torch.randint(0, max(1, seq_len - seg_len + 1), (1,), generator=generator).item())
                channels = torch.randperm(n_feat, generator=generator)[:n_channels].tolist()
                for c in channels:
                    x[b, start : start + seg_len, c] = x[b, start, c]
        return x

    if fault_type == "spike":
        n_spikes = max(1, int(round(seq_len * (0.02 + 0.10 * severity))))
        amp = 4.0 + 8.0 * severity
        std = x.std(dim=1, keepdim=True) + 1e-6
        for b in range(bsz):
            t_idx = torch.randint(0, seq_len, (n_spikes,), generator=generator)
            c_idx = torch.randint(0, n_feat, (n_spikes,), generator=generator)
            sign = torch.where(
                torch.rand(n_spikes, generator=generator) < 0.5,
                torch.full((n_spikes,), -1.0),
                torch.full((n_spikes,), 1.0),
            ).to(device=x.device, dtype=x.dtype)
            x[b, t_idx, c_idx] = x[b, t_idx, c_idx] + sign * amp * std[b, 0, c_idx]
        return x

    if fault_type == "dropout":
        x_nan = x.clone()
        n_events = max(1, int(round(1 + 3 * severity)))
        base_len = max(2, int(round(seq_len * (0.05 + 0.20 * severity))))
        n_channels = max(1, int(round(n_feat * (0.3 + 0.4 * severity))))

        for b in range(bsz):
            for _ in range(n_events):
                scale = float(0.5 + torch.rand(1, generator=generator).item())
                dur = max(1, int(round(base_len * scale)))
                start = int(torch.randint(0, max(1, seq_len - dur + 1), (1,), generator=generator).item())
                channels = torch.randperm(n_feat, generator=generator)[:n_channels]
                x_nan[b, start : start + dur, channels] = float("nan")

        for t in range(1, seq_len):
            missing = torch.isnan(x_nan[:, t, :])
            x_nan[:, t, :] = torch.where(missing, x_nan[:, t - 1, :], x_nan[:, t, :])

        x_nan[torch.isnan(x_nan)] = 0.0
        return x_nan

    raise ValueError(f"Unsupported fault type: {fault_type}")


def eval_structured_faults(
    predictors: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
    test_loader: torch.utils.data.DataLoader,
    scaler,
    device: torch.device,
    *,
    seed: int = 42,
    max_eval_batches: int = 0,
) -> pd.DataFrame:
    target_mean = float(scaler.mean[-1])
    target_std = float(scaler.std[-1])

    scenarios = [{"fault_type": "clean", "severity": 0.0}]
    for fault_type in ["flatline", "spike", "dropout"]:
        for severity in [0.2, 0.6]:
            scenarios.append({"fault_type": fault_type, "severity": severity})

    rows = []
    for sc_idx, sc in enumerate(scenarios):
        pred_map = {name: [] for name in predictors}
        trues_all = []
        for batch_idx, (x, y) in enumerate(test_loader):
            if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                break
            if sc["fault_type"] == "clean":
                x_fault = x
            else:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed + sc_idx * 100000 + batch_idx)
                x_fault = inject_structured_fault(x, sc["fault_type"], float(sc["severity"]), generator)

            x_dev = x_fault.to(device)
            y_dev = y.to(device)
            trues_all.append((y_dev * target_std + target_mean).detach().cpu().numpy())
            for name, fn in predictors.items():
                with torch.no_grad():
                    pred = fn(x_dev)
                pred_map[name].append((pred * target_std + target_mean).detach().cpu().numpy())

        true_np = np.concatenate(trues_all, axis=0)
        for name, pred_list in pred_map.items():
            pred_np = np.concatenate(pred_list, axis=0)
            metrics = compute_metrics(pred_np, true_np)
            rows.append(
                {
                    "fault_type": sc["fault_type"],
                    "severity": float(sc["severity"]),
                    "model": name,
                    "R2": float(metrics["R2"]),
                    "RMSE": float(metrics["RMSE"]),
                    "MAE": float(metrics["MAE"]),
                }
            )

    out_df = pd.DataFrame(rows)
    clean_ref = (
        out_df[out_df["fault_type"] == "clean"][["model", "R2"]]
        .rename(columns={"R2": "R2_clean"})
        .reset_index(drop=True)
    )
    out_df = out_df.merge(clean_ref, on="model", how="left")
    out_df["R2_abs_drop_vs_clean"] = out_df["R2_clean"] - out_df["R2"]
    out_df["R2_rel_drop_pct_vs_clean"] = 100.0 * out_df["R2_abs_drop_vs_clean"] / np.maximum(out_df["R2_clean"], 1e-12)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Full guardrail training evidence with backbone-aware checkpoint handling.")
    parser.add_argument("--backbone", type=str, default="ultra", choices=list(BACKBONE_CHOICES))
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Dataset alias or CSV path.")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target column name.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Display name for this dataset.")
    parser.add_argument("--output-tag", type=str, default=None, help="Output tag; auto-derived when omitted.")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR, help="Directory for outputs.")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-config", type=str, default=None, help="Base backbone config JSON when required.")
    parser.add_argument("--base-checkpoint", type=str, default=None, help="Optional base checkpoint used for initialization.")
    parser.add_argument("--save-base-checkpoint", type=str, default=None, help="Optional output path for a newly trained base checkpoint.")
    parser.add_argument("--force-train", action="store_true", help="Ignore existing full-guardrail checkpoint and retrain.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs for full guardrail fine-tuning.")
    parser.add_argument("--base-train-epochs", type=int, default=10, help="Training epochs when the requested base checkpoint is missing.")
    parser.add_argument("--base-lr", type=float, default=1e-4, help="Learning rate used when a base checkpoint must be trained.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate used for full guardrail fine-tuning.")
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--aux-weight", type=float, default=0.005)
    parser.add_argument("--lambda2", type=float, default=0.03, help="Physics-envelope loss weight.")
    parser.add_argument("--corruption-weight", type=float, default=0.20, help="Weight on corrupted-input objective.")
    parser.add_argument("--tail-weight", type=float, default=0.005, help="Weight on positive-tail overforecast penalty.")
    parser.add_argument("--ramp-weight", type=float, default=0.002, help="Weight on horizon ramp-alignment penalty.")
    parser.add_argument("--consistency-weight", type=float, default=0.005, help="Weight on clean-vs-corrupted consistency.")
    parser.add_argument("--tail-quantile", type=float, default=0.95, help="Quantile used by the positive-tail loss.")
    parser.add_argument("--corruption-sigma-min", type=float, default=0.04)
    parser.add_argument("--corruption-sigma-max", type=float, default=0.18)
    parser.add_argument("--structured-fault-prob", type=float, default=0.35)
    parser.add_argument("--drift-prob", type=float, default=0.25)
    parser.add_argument("--quantization-prob", type=float, default=0.20)
    parser.add_argument("--max-train-batches", type=int, default=0, help="Optional cap for smoke runs; 0 means full loader.")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="Optional cap for smoke runs; 0 means full loader.")
    add_wind_proxy_arguments(parser)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = default_dataset_name(args.data, args.dataset_name)
    output_tag = sanitize_output_tag(args.output_tag) if args.output_tag else infer_output_tag(args.data, dataset_name)
    outputs = build_fgmoe_output_paths(args.out_dir, output_tag)

    print(f"[info] device={device}")
    print(f"[info] backbone={args.backbone} dataset={dataset_name} output_tag={output_tag}")

    data, feature_names, ts_cfg, train_loader, val_loader, test_loader, scaler = load_main_data(
        args.data,
        args.target,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_cut = max(1, int(len(data) * 0.7))
    calibration_data = data[:train_cut]
    phy = estimate_curve_params(data, feature_names, target=args.target, train_ratio=0.7)
    phy.lambda2 = float(args.lambda2)
    wind_proxy_spec = build_wind_proxy_spec(
        feature_names,
        scaler,
        ts_cfg.pred_len,
        calibration_data=calibration_data,
        calibration_mode=args.proxy_calibration_mode,
        history_len=args.proxy_history_len,
        max_wind_mps=args.proxy_max_wind_mps,
        max_step_delta_mps=args.proxy_max_step_delta_mps,
        volatility_scale_mps=args.proxy_volatility_scale_mps,
        trend_weight_min=args.proxy_trend_weight_min,
        trend_weight_max=args.proxy_trend_weight_max,
        damping_denominator=args.proxy_damping_denominator,
    )
    print(f"[physics] {phy}")
    print(f"[proxy] using horizon-aware wind proxy with {wind_proxy_spec.feature_names}")

    base_model = build_backbone_model(
        args.backbone,
        ts_cfg,
        device,
        input_dim=len(feature_names),
        data_ref=args.data,
        dataset_name=dataset_name,
        checkpoint_path=args.base_checkpoint,
        config_path=args.base_config,
    )
    base_load_report = getattr(base_model, "_load_report", {})
    resolved_base_checkpoint = str(base_load_report.get("loaded_from") or "")
    resolved_base_config = str(base_load_report.get("config_path") or "")
    base_checkpoint_exists = bool(resolved_base_checkpoint) and os.path.exists(resolved_base_checkpoint)
    base_training_performed = False
    base_history_rows: List[Dict[str, float]] = []
    base_checkpoint_saved_to = None

    if not base_checkpoint_exists:
        print(f"[train] base checkpoint missing for backbone={args.backbone}; start base training ...")
        base_model, base_history_rows = train_base_backbone(
            args.backbone,
            base_model,
            train_loader,
            val_loader,
            device=device,
            epochs=args.base_train_epochs,
            lr=args.base_lr,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
        )
        base_training_performed = True
        save_path = args.save_base_checkpoint or args.base_checkpoint
        if save_path:
            base_checkpoint_saved_to = save_backbone_checkpoint(base_model, save_path)
            base_load_report = getattr(base_model, "_load_report", {})
            base_load_report["checkpoint_saved_to"] = base_checkpoint_saved_to
            base_model._load_report = base_load_report
            print(f"[save] base checkpoint saved to {base_checkpoint_saved_to}")

    raw_model = copy.deepcopy(base_model)
    guardrail_model = copy.deepcopy(base_model)
    hist_csv = outputs["history_csv"]
    guardrail_ckpt = outputs["checkpoint"]
    reused_checkpoint = False
    guardrail_reuse_load_report = None

    if os.path.exists(guardrail_ckpt) and not args.force_train:
        print(f"[load] existing full-guardrail checkpoint found, reuse without retraining: {guardrail_ckpt}")
        state = torch.load(guardrail_ckpt, map_location=device)
        guardrail_reuse_load_report = partial_load_state_dict(guardrail_model, state if not isinstance(state, dict) else state)
        reused_checkpoint = True
        hist_df = pd.read_csv(hist_csv) if os.path.exists(hist_csv) else pd.DataFrame()
    else:
        print(f"[train] start full guardrail training for backbone={args.backbone} ...")
        guardrail_model, hist_df = train_pi_moe_full(
            guardrail_model,
            args.backbone,
            train_loader,
            val_loader,
            scaler,
            wind_proxy_spec,
            phy,
            device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            aux_weight=args.aux_weight,
            corruption_weight=args.corruption_weight,
            tail_weight=args.tail_weight,
            ramp_weight=args.ramp_weight,
            consistency_weight=args.consistency_weight,
            tail_quantile=args.tail_quantile,
            corruption_sigma_min=args.corruption_sigma_min,
            corruption_sigma_max=args.corruption_sigma_max,
            structured_fault_prob=args.structured_fault_prob,
            drift_prob=args.drift_prob,
            quantization_prob=args.quantization_prob,
            seed=args.seed,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
        )
        hist_df.to_csv(hist_csv, index=False)
        torch.save(guardrail_model.state_dict(), guardrail_ckpt)

    raw_label = backbone_display_name(args.backbone)
    guardrail_label = full_guardrail_name(args.backbone)
    rows = []
    for scenario, sigma_cfg, sigma_eff in [("clean", 0.0, 0.0), ("gaussian_60pct", 0.4, 0.6)]:
        raw_metrics = eval_model_with_noise(
            raw_model,
            args.backbone,
            test_loader,
            scaler,
            wind_proxy_spec,
            phy,
            device,
            sigma_cfg=sigma_cfg,
            max_eval_batches=args.max_eval_batches,
        )
        guard_metrics = eval_model_with_noise(
            guardrail_model,
            args.backbone,
            test_loader,
            scaler,
            wind_proxy_spec,
            phy,
            device,
            sigma_cfg=sigma_cfg,
            max_eval_batches=args.max_eval_batches,
        )

        rows.append(
            {
                "scenario": scenario,
                "sigma_cfg": sigma_cfg,
                "sigma_eff": sigma_eff,
                "backbone": args.backbone,
                "variant": "raw",
                "seed": args.seed,
                "model": raw_label,
                "R2": raw_metrics["R2"],
                "RMSE": raw_metrics["RMSE"],
                "MAE": raw_metrics["MAE"],
                "neg_rate_pct": raw_metrics["neg_rate_pct"],
                "over_rated_rate_pct": raw_metrics["over_rated_rate_pct"],
                "cut_region_nonzero_rate_pct": raw_metrics["cut_region_nonzero_rate_pct"],
            }
        )
        rows.append(
            {
                "scenario": scenario,
                "sigma_cfg": sigma_cfg,
                "sigma_eff": sigma_eff,
                "backbone": args.backbone,
                "variant": "full_guardrail",
                "seed": args.seed,
                "model": guardrail_label,
                "R2": guard_metrics["R2"],
                "RMSE": guard_metrics["RMSE"],
                "MAE": guard_metrics["MAE"],
                "neg_rate_pct": guard_metrics["neg_rate_pct"],
                "over_rated_rate_pct": guard_metrics["over_rated_rate_pct"],
                "cut_region_nonzero_rate_pct": guard_metrics["cut_region_nonzero_rate_pct"],
            }
        )

    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(outputs["compare_csv"], index=False)

    predictors = {
        raw_label: lambda x: model_forward(raw_model, args.backbone, x, temperature=0.1)[0],
        guardrail_label: lambda x: model_forward(guardrail_model, args.backbone, x, temperature=0.1)[0],
    }
    struct_df = eval_structured_faults(
        predictors,
        test_loader,
        scaler,
        device,
        seed=args.seed,
        max_eval_batches=args.max_eval_batches,
    )
    struct_df.to_csv(outputs["structured_csv"], index=False)

    meta = {
        "script": "run_pi_moe_full_training_evidence.py",
        "dataset": args.data,
        "dataset_name": dataset_name,
        "target": args.target,
        "output_tag": output_tag,
        "backbone": args.backbone,
        "seed": args.seed,
        "split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
        "window": {"seq_len": ts_cfg.seq_len, "pred_len": ts_cfg.pred_len},
        "feature_count": int(len(feature_names)),
        "feature_names": list(feature_names),
        "training": {
            "init": resolved_base_checkpoint or args.base_checkpoint,
            "config": resolved_base_config or args.base_config,
            "epochs": int(args.epochs),
            "force_train": bool(args.force_train),
            "reused_checkpoint": reused_checkpoint,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "aux_weight": float(args.aux_weight),
            "lambda2": phy.lambda2,
            "corruption_weight": float(args.corruption_weight),
            "tail_weight": float(args.tail_weight),
            "ramp_weight": float(args.ramp_weight),
            "consistency_weight": float(args.consistency_weight),
            "tail_quantile": float(args.tail_quantile),
            "corruption_sigma_min": float(args.corruption_sigma_min),
            "corruption_sigma_max": float(args.corruption_sigma_max),
            "structured_fault_prob": float(args.structured_fault_prob),
            "drift_prob": float(args.drift_prob),
            "quantization_prob": float(args.quantization_prob),
        },
        "base_training": {
            "performed": base_training_performed,
            "epochs": int(args.base_train_epochs),
            "lr": float(args.base_lr),
            "checkpoint_saved_to": base_checkpoint_saved_to,
            "history": base_history_rows,
        },
        "physics_params": {
            "P_rated_kw": phy.p_rated_kw,
            "v_in_mps": phy.v_in_mps,
            "v_rated_mps": phy.v_rated_mps,
            "v_out_mps": phy.v_out_mps,
        },
        "wind_proxy": proxy_spec_to_dict(wind_proxy_spec),
        "base_model_load_report": base_load_report,
        "reused_guardrail_checkpoint_load_report": guardrail_reuse_load_report,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "outputs": outputs,
    }
    with open(outputs["meta_json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n[done] outputs:")
    for path in outputs.values():
        print(f"- {path}")
    print("\n[preview] raw vs full guardrail:")
    print(compare_df.to_string(index=False))
    print("\n[preview] structured faults:")
    print(struct_df.to_string(index=False))


if __name__ == "__main__":
    main()
