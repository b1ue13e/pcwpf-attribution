#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from robustness_faults import FAULT_CLEAN, apply_fault_tensor, available_faults, parse_faults
from run_pi_moe_full_training_evidence import (
    build_ultra_model,
    estimate_curve_params,
    load_main_data,
    power_envelope_pu,
    set_seed,
)
from ultra_lsnt_timeseries import compute_metrics


OUT_DIR = Path("results") / "supplementary_evidence"
OUT_CSV = OUT_DIR / "physics_closure.csv"
OUT_META = OUT_DIR / "physics_closure.meta.json"


def pick_device(flag: str) -> torch.device:
    if flag == "cuda":
        return torch.device("cuda")
    if flag == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_state_dict(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def load_fgmoe_model(ts_cfg, device: torch.device, ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"FG-MoE checkpoint was not found: {ckpt_path}. "
            "Run `python run_pi_moe_full_training_evidence.py` first."
        )
    model = build_ultra_model(ts_cfg, device)
    state = normalize_state_dict(torch.load(ckpt_path, map_location=device))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def model_predict_norm(model, x: torch.Tensor) -> torch.Tensor:
    pred, _ = model(x, temperature=0.1)
    return pred


def apply_clip_projection(pred_kw: torch.Tensor, ws_proxy: torch.Tensor, phy) -> torch.Tensor:
    out = pred_kw.clamp(min=0.0, max=float(phy.p_rated_kw))
    env_hi = power_envelope_pu(ws_proxy, phy) * float(phy.p_rated_kw)
    out = torch.minimum(out, env_hi)
    cut_mask = (ws_proxy < float(phy.v_in_mps)) | (ws_proxy > float(phy.v_out_mps))
    out = torch.where(cut_mask, torch.zeros_like(out), out)
    return out


def compute_ramp_limit_kw(data: np.ndarray, feature_names: List[str], q: float, train_ratio: float = 0.7) -> float:
    p_idx = feature_names.index("power")
    n_train = max(2, int(len(data) * train_ratio))
    p = np.asarray(data[:n_train, p_idx], dtype=np.float64).reshape(-1)
    if p.size < 2:
        return 0.0
    ramp = np.abs(np.diff(p))
    if ramp.size == 0:
        return 0.0
    qv = float(np.clip(q, 0.0, 1.0))
    return float(np.quantile(ramp, qv))


def scenario_name(fault: str, severity: float) -> str:
    if fault == FAULT_CLEAN:
        return "clean"
    pct = int(round(float(severity) * 100.0))
    return f"{fault}_{pct}pct"


def evaluate_variant(
    *,
    model_name: str,
    model,
    mode: str,
    fault: str,
    severity: float,
    alpha: float,
    scaler,
    ws_idx: int,
    phy,
    ramp_limit_kw: float,
    under_rated_frac: float,
    test_loader,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    model.eval()
    t_mean = float(scaler.mean[-1])
    t_std = float(scaler.std[-1])
    ws_mean = float(scaler.mean[ws_idx])
    ws_std = float(scaler.std[ws_idx])

    all_pred: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    env_dev_chunks: List[np.ndarray] = []

    n_total = 0
    n_neg = 0
    n_over = 0
    n_cut_nonzero = 0

    n_under = 0
    n_under_denom = 0

    n_ramp_vio = 0
    n_ramp_total = 0

    for bi, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        gen = torch.Generator(device="cpu")
        fault_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(fault))
        gen.manual_seed(int(seed) * 100_000 + int(bi) + int(fault_seed))
        x_fault = apply_fault_tensor(
            x,
            fault=fault,
            severity=float(severity) if fault != FAULT_CLEAN else 0.0,
            alpha=float(alpha),
            generator=gen,
        )

        with torch.no_grad():
            pred_norm = model_predict_norm(model, x_fault)

        pred_kw = pred_norm * t_std + t_mean
        true_kw = y * t_std + t_mean

        ws_last = x_fault[:, -1, ws_idx] * ws_std + ws_mean
        ws_proxy = ws_last.unsqueeze(1).expand_as(pred_kw)

        if mode == "clip_only":
            pred_kw = apply_clip_projection(pred_kw, ws_proxy, phy)

        n_total += int(pred_kw.numel())
        n_neg += int((pred_kw < 0.0).sum().item())
        n_over += int((pred_kw > float(phy.p_rated_kw)).sum().item())

        cut_mask = (ws_proxy < float(phy.v_in_mps)) | (ws_proxy > float(phy.v_out_mps))
        n_cut_nonzero += int(((pred_kw > 0.0) & cut_mask).sum().item())

        high_wind = (ws_proxy >= float(phy.v_rated_mps)) & (ws_proxy <= float(phy.v_out_mps))
        n_under += int(((pred_kw < float(under_rated_frac) * float(phy.p_rated_kw)) & high_wind).sum().item())
        n_under_denom += int(high_wind.sum().item())

        if pred_kw.shape[1] >= 2:
            ramp = torch.abs(pred_kw[:, 1:] - pred_kw[:, :-1])
            n_ramp_vio += int((ramp > float(ramp_limit_kw)).sum().item())
            n_ramp_total += int(ramp.numel())

        env_hi = power_envelope_pu(ws_proxy, phy) * float(phy.p_rated_kw)
        env_dev = torch.relu(pred_kw - env_hi) + torch.relu(-pred_kw)
        env_dev_chunks.append(env_dev.detach().cpu().numpy().reshape(-1))

        all_pred.append(pred_kw.detach().cpu().numpy())
        all_true.append(true_kw.detach().cpu().numpy())

    pred_np = np.concatenate(all_pred, axis=0)
    true_np = np.concatenate(all_true, axis=0)
    env_np = np.concatenate(env_dev_chunks, axis=0) if env_dev_chunks else np.zeros((0,), dtype=np.float64)
    m = compute_metrics(pred_np, true_np)

    env_mean = float(np.mean(env_np)) if env_np.size else 0.0
    env_p95 = float(np.quantile(env_np, 0.95)) if env_np.size else 0.0
    env_sum = float(np.sum(env_np)) if env_np.size else 0.0

    return {
        "model": model_name,
        "mode": mode,
        "R2": float(m["R2"]),
        "RMSE": float(m["RMSE"]),
        "MAE": float(m["MAE"]),
        "neg_rate_pct": 100.0 * float(n_neg) / max(float(n_total), 1.0),
        "over_rated_rate_pct": 100.0 * float(n_over) / max(float(n_total), 1.0),
        "cut_region_nonzero_rate_pct": 100.0 * float(n_cut_nonzero) / max(float(n_total), 1.0),
        "under_rated_low_output_rate_pct": 100.0 * float(n_under) / max(float(n_under_denom), 1.0),
        "under_rated_denom_points": int(n_under_denom),
        "ramp_rate_violation_pct": 100.0 * float(n_ramp_vio) / max(float(n_ramp_total), 1.0),
        "ramp_rate_violation_limit_kw": float(ramp_limit_kw),
        "envelope_deviation_area_mean_kw": env_mean,
        "envelope_deviation_area_p95_kw": env_p95,
        "envelope_deviation_area_sum_kw": env_sum,
        "envelope_deviation_area_pct_of_rated": 100.0 * env_mean / max(float(phy.p_rated_kw), 1e-12),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Physics closure: no guardrail vs clip-only vs FG-MoE (full-trained) under multi-fault scenarios."
    )
    parser.add_argument("--faults", type=str, default="gaussian,drift,quantization,missingness")
    parser.add_argument("--severity", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--under-rated-frac", type=float, default=0.5)
    parser.add_argument("--ramp-q", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--fg-checkpoint",
        type=str,
        default=str(OUT_DIR / "pi_moe_full_checkpoint.pth"),
    )
    parser.add_argument("--out-csv", type=str, default=str(OUT_CSV))
    parser.add_argument("--out-meta", type=str, default=str(OUT_META))
    args = parser.parse_args()

    faults = [f for f in parse_faults(args.faults) if f != FAULT_CLEAN]
    for f in faults:
        if f not in available_faults():
            raise ValueError(f"Unsupported fault: {f}")

    set_seed(int(args.seed))
    device = pick_device(args.device)
    out_csv = Path(args.out_csv)
    out_meta = Path(args.out_meta)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    data, feature_names, ts_cfg, _, _, test_loader, scaler = load_main_data()
    ws_idx = feature_names.index("windspeed")
    phy = estimate_curve_params(data, feature_names, train_ratio=0.7)
    ramp_limit_kw = compute_ramp_limit_kw(data, feature_names, q=float(args.ramp_q), train_ratio=0.7)

    ultra = build_ultra_model(ts_cfg, device)
    fg_moe = load_fgmoe_model(ts_cfg, device, Path(args.fg_checkpoint))
    model_pack = {
        "Ultra-LSNT (no guardrail)": ("no_guardrail", ultra),
        "Ultra-LSNT + clip-only (posthoc)": ("clip_only", ultra),
        "FG-MoE (full-trained)": ("full_guardrail", fg_moe),
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_guardrail_physics_closure"
    rows: List[Dict] = []

    eval_faults = [FAULT_CLEAN] + faults
    for f in eval_faults:
        sc = scenario_name(f, float(args.severity))
        print(f"[scenario] {sc}")
        for model_label, (mode, model) in model_pack.items():
            metrics = evaluate_variant(
                model_name=model_label,
                model=model,
                mode=mode,
                fault=f,
                severity=float(args.severity),
                alpha=float(args.alpha),
                scaler=scaler,
                ws_idx=ws_idx,
                phy=phy,
                ramp_limit_kw=ramp_limit_kw,
                under_rated_frac=float(args.under_rated_frac),
                test_loader=test_loader,
                device=device,
                seed=int(args.seed),
            )
            metrics.update(
                {
                    "run_id": run_id,
                    "scenario": sc,
                    "fault": f,
                    "severity": 0.0 if f == FAULT_CLEAN else float(args.severity),
                    "seed": int(args.seed),
                }
            )
            rows.append(metrics)
            print(
                f"  [{model_label}] R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.3f}, "
                f"over={metrics['over_rated_rate_pct']:.4f}%, under={metrics['under_rated_low_output_rate_pct']:.4f}%"
            )

    out_df = pd.DataFrame(rows).sort_values(["scenario", "model"], ascending=[True, True]).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)

    meta = {
        "script": "run_guardrail_physics_closure_4090.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "device": str(device),
        "dataset": "wind_final.csv",
        "window": {"seq_len": 96, "pred_len": 24},
        "seed": int(args.seed),
        "faults": faults,
        "severity": float(args.severity),
        "fault_injection_alpha": float(args.alpha),
        "under_rated_definition": f"prediction < {float(args.under_rated_frac):.2f} * P_rated when v in [v_rated, v_out]",
        "ramp_limit_definition": f"abs(delta prediction) > q{float(args.ramp_q):.2f} of train abs(delta power)",
        "ramp_limit_kw": float(ramp_limit_kw),
        "envelope_deviation_definition": "relu(pred - envelope_upper) + relu(-pred), mean over all points",
        "physics_params": {
            "P_rated_kw": float(phy.p_rated_kw),
            "v_in_mps": float(phy.v_in_mps),
            "v_rated_mps": float(phy.v_rated_mps),
            "v_out_mps": float(phy.v_out_mps),
            "lambda2": float(phy.lambda2),
        },
        "checkpoint_paths": {
            "ultra_no_guardrail": "checkpoints_ts/main/best_model.pth",
            "fg_moe_full_trained": str(Path(args.fg_checkpoint)),
        },
        "outputs": {
            "physics_closure_csv": str(out_csv),
            "meta_json": str(out_meta),
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
