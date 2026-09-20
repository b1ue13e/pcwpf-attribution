#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from robustness_faults import FAULT_CLEAN, apply_fault_tensor, parse_faults
from run_pi_moe_full_training_evidence import (
    build_ultra_model,
    estimate_curve_params,
    power_envelope_pu,
    set_seed,
)
from ultra_lsnt_timeseries import (
    TimeSeriesConfig,
    TrainConfig,
    compute_metrics,
    create_dataloaders,
    load_csv_data,
)


OUT_DIR = Path("results") / "supplementary_evidence"
OUT_LATENCY = OUT_DIR / "arm_revision_latency_energy.csv"
OUT_SMOKE = OUT_DIR / "arm_revision_robustness_smoke.csv"
OUT_META = OUT_DIR / "arm_revision_eval.meta.json"


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
            f"Missing FG-MoE checkpoint: {ckpt_path}. "
            "Run 4090 training first: `python run_pi_moe_full_training_evidence.py`."
        )
    model = build_ultra_model(ts_cfg, device)
    state = normalize_state_dict(torch.load(ckpt_path, map_location=device))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def apply_clip_projection(pred_kw: torch.Tensor, ws_proxy: torch.Tensor, phy) -> torch.Tensor:
    out = pred_kw.clamp(min=0.0, max=float(phy.p_rated_kw))
    env_hi = power_envelope_pu(ws_proxy, phy) * float(phy.p_rated_kw)
    out = torch.minimum(out, env_hi)
    cut_mask = (ws_proxy < float(phy.v_in_mps)) | (ws_proxy > float(phy.v_out_mps))
    out = torch.where(cut_mask, torch.zeros_like(out), out)
    return out


def build_model_bundle(ts_cfg, device: torch.device, fg_ckpt: Path):
    ultra = build_ultra_model(ts_cfg, device)
    fg = load_fgmoe_model(ts_cfg, device, fg_ckpt)
    return {
        "Ultra-LSNT (no guardrail)": {
            "mode": "no_guardrail",
            "model": ultra,
            "checkpoint": "checkpoints_ts/main/best_model.pth",
        },
        "Ultra-LSNT + clip-only (posthoc)": {
            "mode": "clip_only",
            "model": ultra,
            "checkpoint": "checkpoints_ts/main/best_model.pth",
        },
        "FG-MoE (full-trained)": {
            "mode": "full_guardrail",
            "model": fg,
            "checkpoint": str(fg_ckpt),
        },
    }


def parse_scenarios(raw: str) -> List[str]:
    vals = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No scenarios were provided.")
    out: List[str] = []
    for v in vals:
        if v in ("clean", FAULT_CLEAN):
            out.append(FAULT_CLEAN)
        else:
            out.append(v)
    return out


def find_energy_counter() -> Path | None:
    candidates = [
        "/sys/class/powercap/dtpm/energy_uj",
        "/sys/class/powercap/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl:1/energy_uj",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return p
    return None


def read_energy_uj(counter: Path | None) -> float | None:
    if counter is None:
        return None
    try:
        return float(counter.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def model_forward_pred_kw(
    entry: Dict,
    x: torch.Tensor,
    *,
    scaler,
    ws_idx: int,
    phy,
) -> torch.Tensor:
    model = entry["model"]
    pred_norm, _ = model(x, temperature=0.1)
    t_mean = float(scaler.mean[-1])
    t_std = float(scaler.std[-1])
    ws_mean = float(scaler.mean[ws_idx])
    ws_std = float(scaler.std[ws_idx])
    pred_kw = pred_norm * t_std + t_mean
    if entry["mode"] == "clip_only":
        ws_last = x[:, -1, ws_idx] * ws_std + ws_mean
        ws_proxy = ws_last.unsqueeze(1).expand_as(pred_kw)
        pred_kw = apply_clip_projection(pred_kw, ws_proxy, phy)
    return pred_kw


def run_latency_energy(
    *,
    entry: Dict,
    x_ref: torch.Tensor,
    scaler,
    ws_idx: int,
    phy,
    warmup: int,
    iters: int,
    assumed_power_w: float,
) -> Dict[str, float]:
    with torch.no_grad():
        for _ in range(max(1, warmup)):
            _ = model_forward_pred_kw(entry, x_ref, scaler=scaler, ws_idx=ws_idx, phy=phy)

    lat_ms: List[float] = []
    counter = find_energy_counter()
    e0 = read_energy_uj(counter)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max(1, iters)):
            st = time.perf_counter()
            _ = model_forward_pred_kw(entry, x_ref, scaler=scaler, ws_idx=ws_idx, phy=phy)
            ed = time.perf_counter()
            lat_ms.append((ed - st) * 1000.0)
    t1 = time.perf_counter()
    e1 = read_energy_uj(counter)

    total_sec = max(t1 - t0, 1e-12)
    lat = np.asarray(lat_ms, dtype=np.float64)
    mean_ms = float(np.mean(lat)) if lat.size else np.nan
    p95_ms = float(np.quantile(lat, 0.95)) if lat.size else np.nan
    std_ms = float(np.std(lat)) if lat.size else np.nan
    throughput_hz = float(1000.0 / mean_ms) if np.isfinite(mean_ms) and mean_ms > 0 else np.nan

    if e0 is not None and e1 is not None and e1 >= e0:
        energy_j = (e1 - e0) / 1e6
        energy_mj = float(energy_j / max(1, iters) * 1000.0)
        power_w = float(energy_j / total_sec)
        method = "counter"
    else:
        energy_j = float(assumed_power_w) * total_sec
        energy_mj = float(energy_j / max(1, iters) * 1000.0)
        power_w = float(assumed_power_w)
        method = "estimated_constant_power"

    return {
        "latency_mean_ms": mean_ms,
        "latency_p95_ms": p95_ms,
        "latency_std_ms": std_ms,
        "throughput_hz": throughput_hz,
        "energy_per_inference_mj": energy_mj,
        "avg_power_w": power_w,
        "energy_method": method,
    }


def run_robustness_smoke(
    *,
    model_name: str,
    entry: Dict,
    scenario_fault: str,
    severity: float,
    alpha: float,
    scaler,
    ws_idx: int,
    phy,
    test_loader,
    max_batches: int,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    all_pred: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    n_total = 0
    n_over = 0

    for bi, (x, y) in enumerate(test_loader):
        if bi >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        gen = torch.Generator(device="cpu")
        fault_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(scenario_fault))
        gen.manual_seed(int(seed) * 100_000 + int(bi) + int(fault_seed))
        x_fault = apply_fault_tensor(
            x,
            fault=scenario_fault,
            severity=float(severity) if scenario_fault != FAULT_CLEAN else 0.0,
            alpha=float(alpha),
            generator=gen,
        )

        with torch.no_grad():
            pred_kw = model_forward_pred_kw(entry, x_fault, scaler=scaler, ws_idx=ws_idx, phy=phy)

        t_mean = float(scaler.mean[-1])
        t_std = float(scaler.std[-1])
        true_kw = y * t_std + t_mean

        n_total += int(pred_kw.numel())
        n_over += int((pred_kw > float(phy.p_rated_kw)).sum().item())

        all_pred.append(pred_kw.detach().cpu().numpy())
        all_true.append(true_kw.detach().cpu().numpy())

    pred_np = np.concatenate(all_pred, axis=0)
    true_np = np.concatenate(all_true, axis=0)
    m = compute_metrics(pred_np, true_np)

    return {
        "model": model_name,
        "scenario": "clean" if scenario_fault == FAULT_CLEAN else f"{scenario_fault}_{int(round(severity * 100.0))}pct",
        "fault": scenario_fault,
        "severity": 0.0 if scenario_fault == FAULT_CLEAN else float(severity),
        "R2": float(m["R2"]),
        "RMSE": float(m["RMSE"]),
        "MAE": float(m["MAE"]),
        "over_rated_rate_pct": 100.0 * float(n_over) / max(float(n_total), 1.0),
        "checkpoint": str(entry["checkpoint"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ARM deployment-only revision evaluation (no training).")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--scenarios", type=str, default="clean,gaussian,drift,quantization,missingness")
    parser.add_argument("--severity", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--assumed-power-w", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fg-checkpoint", type=str, default=str(OUT_DIR / "pi_moe_full_checkpoint.pth"))
    parser.add_argument("--out-latency", type=str, default=str(OUT_LATENCY))
    parser.add_argument("--out-smoke", type=str, default=str(OUT_SMOKE))
    parser.add_argument("--out-meta", type=str, default=str(OUT_META))
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = pick_device(args.device)
    scenarios = parse_scenarios(args.scenarios)

    data, feature_names = load_csv_data("wind_final.csv", "power")
    if data is None:
        raise RuntimeError("Failed to load wind_final.csv")
    ts_cfg = TimeSeriesConfig(seq_len=96, pred_len=24, target="power")
    tr_cfg = TrainConfig(batch_size=int(args.batch_size), epochs=1, num_workers=0)
    _, _, test_loader, scaler = create_dataloaders(data, ts_cfg, tr_cfg)
    ws_idx = feature_names.index("windspeed")
    phy = estimate_curve_params(data, feature_names, train_ratio=0.7)

    model_bundle = build_model_bundle(ts_cfg, device, Path(args.fg_checkpoint))

    out_latency = Path(args.out_latency)
    out_smoke = Path(args.out_smoke)
    out_meta = Path(args.out_meta)
    out_latency.parent.mkdir(parents=True, exist_ok=True)

    x_ref, _ = next(iter(test_loader))
    x_ref = x_ref.to(device)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_arm_revision_eval"

    lat_rows: List[Dict] = []
    smoke_rows: List[Dict] = []
    for model_name, entry in model_bundle.items():
        print(f"[latency] {model_name}")
        lat = run_latency_energy(
            entry=entry,
            x_ref=x_ref,
            scaler=scaler,
            ws_idx=ws_idx,
            phy=phy,
            warmup=int(args.warmup),
            iters=int(args.iters),
            assumed_power_w=float(args.assumed_power_w),
        )
        lat_rows.append(
            {
                "run_id": run_id,
                "model": model_name,
                "device": str(device),
                "batch_size": int(args.batch_size),
                "warmup": int(args.warmup),
                "iters": int(args.iters),
                "checkpoint": str(entry["checkpoint"]),
                **lat,
            }
        )

        for sc in scenarios:
            print(f"[smoke] {model_name} @ {sc}")
            row = run_robustness_smoke(
                model_name=model_name,
                entry=entry,
                scenario_fault=sc,
                severity=float(args.severity),
                alpha=float(args.alpha),
                scaler=scaler,
                ws_idx=ws_idx,
                phy=phy,
                test_loader=test_loader,
                max_batches=int(args.max_batches),
                seed=int(args.seed),
                device=device,
            )
            row["run_id"] = run_id
            row["device"] = str(device)
            row["batch_size"] = int(args.batch_size)
            smoke_rows.append(row)

    lat_df = pd.DataFrame(lat_rows)
    smoke_df = pd.DataFrame(smoke_rows)
    lat_df.to_csv(out_latency, index=False)
    smoke_df.to_csv(out_smoke, index=False)

    meta = {
        "script": "run_revision_arm_eval.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "dataset": "wind_final.csv",
        "window": {"seq_len": 96, "pred_len": 24},
        "device": str(device),
        "batch_size": int(args.batch_size),
        "scenarios": scenarios,
        "severity": float(args.severity),
        "max_batches_smoke": int(args.max_batches),
        "latency_iters": int(args.iters),
        "checkpoints": {k: str(v["checkpoint"]) for k, v in model_bundle.items()},
        "note": "ARM script performs inference-only deployment evaluation; no training occurs here.",
        "outputs": {
            "latency_energy_csv": str(out_latency),
            "robustness_smoke_csv": str(out_smoke),
            "meta_json": str(out_meta),
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] wrote {out_latency}")
    print(f"[ok] wrote {out_smoke}")
    print(f"[ok] wrote {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
