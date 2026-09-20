#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from robustness_faults import (
    FAULT_CLEAN,
    available_faults,
    apply_fault_tensor,
    parse_faults,
    sample_fault_uniform,
)
from run_dlinear import DLinear
from run_pi_moe_full_training_evidence import (
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_BASE_CONFIG,
    DEFAULT_DATA,
    DEFAULT_TARGET,
    build_ultra_model,
    default_dataset_name,
    infer_output_tag,
    partial_load_state_dict,
    sanitize_output_tag,
    tagged_output_path,
)
from ultra_lsnt_timeseries import (
    TimeSeriesConfig,
    TrainConfig,
    compute_metrics,
    create_dataloaders,
    load_csv_data,
)


OUT_DIR = Path("results") / "supplementary_evidence"


def build_fairness_output_paths(out_dir: Path, output_tag: str) -> Dict[str, Path]:
    out_dir_str = str(out_dir)
    ckpt_name = "fairness_closure_ckpts" if output_tag == "windcn" else f"fairness_closure_ckpts_{output_tag}"
    return {
        "raw_csv": Path(tagged_output_path(out_dir_str, "fairness_closure_fullfault_raw", ".csv", output_tag)),
        "summary_csv": Path(tagged_output_path(out_dir_str, "fairness_closure_fullfault_summary", ".csv", output_tag)),
        "hard_table_csv": Path(tagged_output_path(out_dir_str, "fairness_closure_fullfault_hard_table", ".csv", output_tag)),
        "significance_csv": Path(
            tagged_output_path(out_dir_str, "fairness_closure_fullfault_significance", ".csv", output_tag)
        ),
        "meta_json": Path(tagged_output_path(out_dir_str, "fairness_closure_fullfault_meta", ".json", output_tag)),
        "checkpoints_dir": out_dir / ckpt_name,
    }


class TimeMixer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        self.input_dim = config["input_dim"]
        self.d_model = config.get("d_model", 128)
        self.scales = config.get("scales", [1, 2, 4])

        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.multiscale_convs = nn.ModuleList()
        for scale in self.scales:
            kernel_size = max(3, scale * 2)
            padding = kernel_size // 2
            conv = nn.Sequential(
                nn.Conv1d(self.d_model, self.d_model, kernel_size=kernel_size, padding=padding),
                nn.GELU(),
                nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            )
            self.multiscale_convs.append(conv)

        self.scale_fusion = nn.Sequential(
            nn.Linear(self.d_model * len(self.scales), self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.frequency_decomp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.d_model * 2),
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=config.get("nhead", 4),
            dropout=config.get("dropout", 0.1),
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.pred_len),
        )
        self.skip = nn.Linear(self.seq_len, self.pred_len) if self.seq_len != self.pred_len else None

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.input_proj(x_enc)
        x_t = x.transpose(1, 2)

        multiscale_features = []
        for conv in self.multiscale_convs:
            scale_feat = conv(x_t)
            if scale_feat.size(-1) != self.seq_len:
                diff = scale_feat.size(-1) - self.seq_len
                if diff > 0:
                    start = diff // 2
                    scale_feat = scale_feat[:, :, start : start + self.seq_len]
                else:
                    pad = -diff
                    scale_feat = F.pad(scale_feat, (pad // 2, pad - pad // 2))
            multiscale_features.append(scale_feat.transpose(1, 2))

        fused = torch.cat(multiscale_features, dim=-1)
        fused = self.scale_fusion(fused)
        freq_components = self.frequency_decomp(fused)
        low_freq, high_freq = torch.chunk(freq_components, 2, dim=-1)
        trend = low_freq.mean(dim=1, keepdim=True)
        seasonal = high_freq
        attn_out, _ = self.temporal_attn(seasonal, seasonal, seasonal)
        seasonal = self.attn_norm(seasonal + attn_out)
        combined = seasonal + trend.expand_as(seasonal)
        aggregated = combined.mean(dim=1)
        output = self.output_proj(aggregated)

        if self.skip is not None:
            output = output + self.skip(x_enc[:, :, -1]) * 0.1
        return output.unsqueeze(-1), torch.tensor(0.0, device=x_enc.device)


class iTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        self.input_dim = config["input_dim"]
        self.d_model = config.get("d_model", 128)

        self.var_embedding = nn.Linear(self.seq_len, self.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.input_dim, self.d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.get("nhead", 4),
            dim_feedforward=self.d_model * 4,
            dropout=config.get("dropout", 0.1),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.get("num_layers", 2))
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model * self.input_dim, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.pred_len),
        )
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x_inverted = x_enc.transpose(1, 2)
        var_emb = self.var_embedding(x_inverted)
        var_emb = self.norm(var_emb + self.pos_embedding)
        encoded = self.encoder(var_emb)
        output = self.decoder(encoded.reshape(encoded.size(0), -1))
        return output.unsqueeze(-1), torch.tensor(0.0, device=x_enc.device)


@dataclass
class RunConfig:
    models: str = "dlinear,timemixer,itransformer,ultra"
    seeds: str = "42,123,2025"
    epochs: int = 2
    faults: str = "gaussian,drift,quantization,missingness"
    severity: float = 0.6
    batch_size: int = 128
    lr: float = 1e-4
    alpha: float = 1.5
    device: str = "auto"
    bootstrap_iters: int = 2000
    data: str = DEFAULT_DATA
    target: str = DEFAULT_TARGET
    dataset_name: str | None = None
    output_tag: str | None = None
    out_dir: str = str(OUT_DIR)
    seq_len: int = 96
    pred_len: int = 24
    num_workers: int = 0
    base_config: str = DEFAULT_BASE_CONFIG
    base_checkpoint: str = DEFAULT_BASE_CHECKPOINT


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(flag: str) -> torch.device:
    if flag == "cuda":
        return torch.device("cuda")
    if flag == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_models(raw: str) -> List[str]:
    allowed = {"dlinear", "timemixer", "itransformer", "ultra"}
    out = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("No model was provided.")
    unknown = [x for x in out if x not in allowed]
    if unknown:
        raise ValueError(f"Unsupported model(s): {unknown}")
    return out


def parse_seeds(raw: str) -> List[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("No valid seed was provided.")
    return out


def _load_state(path: Path, device: torch.device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def build_model(
    model: str,
    input_dim: int,
    ts_cfg: TimeSeriesConfig,
    device: torch.device,
    *,
    base_config: str,
    base_checkpoint: str,
) -> nn.Module:
    if model == "dlinear":
        ckpt = torch.load("dlinear_weights/dlinear_trained.pth", map_location=device)
        m = DLinear(ts_cfg.seq_len, ts_cfg.pred_len, input_dim).to(device)
        partial_load_state_dict(m, ckpt["model_state_dict"])
        return m

    if model == "timemixer":
        m = TimeMixer(
            {
                "seq_len": ts_cfg.seq_len,
                "pred_len": ts_cfg.pred_len,
                "input_dim": input_dim,
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.1,
                "scales": [1, 2, 4],
            }
        ).to(device)
        partial_load_state_dict(m, _load_state(Path("checkpoints_ts/TimeMixer_best.pth"), device))
        return m

    if model == "itransformer":
        m = iTransformer(
            {
                "seq_len": ts_cfg.seq_len,
                "pred_len": ts_cfg.pred_len,
                "input_dim": input_dim,
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.1,
            }
        ).to(device)
        partial_load_state_dict(m, _load_state(Path("checkpoints_ts/iTransformer_best.pth"), device))
        return m

    if model == "ultra":
        return build_ultra_model(
            ts_cfg,
            device,
            input_dim=input_dim,
            checkpoint_path=base_checkpoint,
            config_path=base_config,
        )

    raise ValueError(f"Unsupported model: {model}")


def forward_model(model_name: str, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if model_name == "dlinear":
        return model(x)
    pred, _ = model(x)
    if pred.ndim == 3 and pred.shape[-1] == 1:
        return pred.squeeze(-1)
    return pred


def display_name(model_name: str) -> str:
    return {
        "dlinear": "DLinear",
        "timemixer": "TimeMixer",
        "itransformer": "iTransformer",
        "ultra": "Ultra-LSNT",
    }[model_name]


def finetune_model(
    model_name: str,
    model: nn.Module,
    train_loader,
    *,
    mode: str,
    faults: Iterable[str],
    severity: float,
    epochs: int,
    lr: float,
    alpha: float,
    seed: int,
    device: torch.device,
) -> nn.Module:
    out_model = copy.deepcopy(model).to(device)
    out_model.train()
    opt = torch.optim.AdamW(out_model.parameters(), lr=lr, weight_decay=0.01)
    mse = nn.MSELoss()
    rng = np.random.default_rng(seed + 10_000)

    for ep in range(epochs):
        for bi, (bx, by) in enumerate(train_loader):
            bx = bx.to(device)
            by = by.to(device)

            if mode == "clean_trained":
                bx_mod = bx
                sampled_fault = FAULT_CLEAN
            else:
                sampled_fault = sample_fault_uniform(faults, rng)
                gen = torch.Generator(device="cpu")
                gen.manual_seed(seed * 1_000_000 + ep * 10_000 + bi)
                bx_mod = apply_fault_tensor(
                    bx,
                    fault=sampled_fault,
                    severity=severity,
                    alpha=alpha,
                    generator=gen,
                )

            opt.zero_grad(set_to_none=True)
            pred = forward_model(model_name, out_model, bx_mod)
            loss = mse(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(out_model.parameters(), max_norm=1.0)
            opt.step()

    out_model.eval()
    return out_model


def evaluate_model(
    model_name: str,
    model: nn.Module,
    test_loader,
    scaler,
    *,
    seed: int,
    fault: str,
    severity: float,
    alpha: float,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_preds: List[np.ndarray] = []
    all_trues: List[np.ndarray] = []
    abs_errors: List[np.ndarray] = []

    with torch.no_grad():
        for bi, (bx, by) in enumerate(test_loader):
            bx = bx.to(device)
            by = by.to(device)
            gen = torch.Generator(device="cpu")
            fault_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(fault))
            gen.manual_seed(seed * 1_000_000 + bi + fault_seed)
            bx_mod = apply_fault_tensor(
                bx,
                fault=fault,
                severity=severity if fault != FAULT_CLEAN else 0.0,
                alpha=alpha,
                generator=gen,
            )
            pred = forward_model(model_name, model, bx_mod)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(by.cpu().numpy())

    pred_np = np.concatenate(all_preds, axis=0)
    true_np = np.concatenate(all_trues, axis=0)
    t_mean = float(scaler.mean[-1])
    t_std = float(scaler.std[-1])
    pred_kw = pred_np * t_std + t_mean
    true_kw = true_np * t_std + t_mean
    abs_err = np.abs(pred_kw - true_kw).reshape(-1)
    abs_errors.append(abs_err)

    m = compute_metrics(pred_kw, true_kw)
    return {
        "R2": float(m["R2"]),
        "RMSE": float(m["RMSE"]),
        "MAE": float(m["MAE"]),
        "AE_Q90": float(np.quantile(abs_err, 0.90)),
        "AE_P95": float(np.quantile(abs_err, 0.95)),
    }


def _bootstrap_ci_mean(x: np.ndarray, y: np.ndarray, iters: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        vals.append(float(np.mean(x[idx] - y[idx])))
    lo, hi = np.quantile(np.asarray(vals, dtype=np.float64), [0.025, 0.975])
    return float(lo), float(hi)


def compute_significance(df_raw: pd.DataFrame, bootstrap_iters: int) -> pd.DataFrame:
    try:
        from scipy.stats import wilcoxon
    except Exception as exc:
        raise RuntimeError(
            "Significance testing requires scipy. Install scipy before running this script."
        ) from exc

    d = df_raw[
        (df_raw["training_mode"] == "fault_augmented")
        & (df_raw["test_fault"] != FAULT_CLEAN)
    ].copy()
    keep_cols = ["seed", "test_fault", "model", "R2", "RMSE"]
    d = d[keep_cols]

    rows = []
    ultra = d[d["model"] == "Ultra-LSNT"]
    baselines = [m for m in sorted(d["model"].unique()) if m != "Ultra-LSNT"]
    for b in baselines:
        base = d[d["model"] == b]
        merged = ultra.merge(base, on=["seed", "test_fault"], how="inner", suffixes=("_ultra", "_base"))
        if len(merged) < 2:
            rows.append(
                {
                    "baseline": b,
                    "metric": "R2",
                    "n_pairs": int(len(merged)),
                    "mean_diff_ultra_minus_base": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "wilcoxon_stat": np.nan,
                    "pvalue_one_sided_ultra_better": np.nan,
                }
            )
            rows.append(
                {
                    "baseline": b,
                    "metric": "RMSE",
                    "n_pairs": int(len(merged)),
                    "mean_diff_ultra_minus_base": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "wilcoxon_stat": np.nan,
                    "pvalue_one_sided_ultra_better": np.nan,
                }
            )
            continue

        # R2: higher is better -> ultra-base positive is better.
        r2_u = merged["R2_ultra"].to_numpy(dtype=np.float64)
        r2_b = merged["R2_base"].to_numpy(dtype=np.float64)
        r2_diff = r2_u - r2_b
        try:
            wr2 = wilcoxon(r2_diff, alternative="greater", zero_method="wilcox")
            wr2_stat, wr2_p = float(wr2.statistic), float(wr2.pvalue)
        except Exception:
            wr2_stat, wr2_p = np.nan, np.nan
        r2_lo, r2_hi = _bootstrap_ci_mean(r2_u, r2_b, bootstrap_iters, seed=123)
        rows.append(
            {
                "baseline": b,
                "metric": "R2",
                "n_pairs": int(len(merged)),
                "mean_diff_ultra_minus_base": float(np.mean(r2_diff)),
                "ci95_low": r2_lo,
                "ci95_high": r2_hi,
                "wilcoxon_stat": wr2_stat,
                "pvalue_one_sided_ultra_better": wr2_p,
            }
        )

        # RMSE: lower is better -> base-ultra positive indicates ultra better.
        rmse_u = merged["RMSE_ultra"].to_numpy(dtype=np.float64)
        rmse_b = merged["RMSE_base"].to_numpy(dtype=np.float64)
        rmse_gain = rmse_b - rmse_u
        try:
            wrmse = wilcoxon(rmse_gain, alternative="greater", zero_method="wilcox")
            wrmse_stat, wrmse_p = float(wrmse.statistic), float(wrmse.pvalue)
        except Exception:
            wrmse_stat, wrmse_p = np.nan, np.nan
        rmse_lo, rmse_hi = _bootstrap_ci_mean(rmse_b, rmse_u, bootstrap_iters, seed=456)
        rows.append(
            {
                "baseline": b,
                "metric": "RMSE_gain(base-ultra)",
                "n_pairs": int(len(merged)),
                "mean_diff_ultra_minus_base": float(np.mean(rmse_gain)),
                "ci95_low": rmse_lo,
                "ci95_high": rmse_hi,
                "wilcoxon_stat": wrmse_stat,
                "pvalue_one_sided_ultra_better": wrmse_p,
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fairness closure with matched full-fault training/evaluation on RTX 4090.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Dataset alias or CSV path.")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target column name.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Display name for metadata.")
    parser.add_argument("--output-tag", type=str, default=None, help="Output tag; auto-derived when omitted.")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR), help="Output directory.")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--models", type=str, default="dlinear,timemixer,itransformer,ultra")
    parser.add_argument("--seeds", type=str, default="42,123,2025")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--faults", type=str, default="gaussian,drift,quantization,missingness")
    parser.add_argument("--severity", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--base-config", type=str, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base-checkpoint", type=str, default=DEFAULT_BASE_CHECKPOINT)
    args = parser.parse_args()

    cfg = RunConfig(
        data=args.data,
        target=args.target,
        dataset_name=args.dataset_name,
        output_tag=args.output_tag,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_workers=args.num_workers,
        models=args.models,
        seeds=args.seeds,
        epochs=args.epochs,
        faults=args.faults,
        severity=args.severity,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        bootstrap_iters=args.bootstrap_iters,
        device=args.device,
        base_config=args.base_config,
        base_checkpoint=args.base_checkpoint,
    )

    models = parse_models(cfg.models)
    seeds = parse_seeds(cfg.seeds)
    faults = parse_faults(cfg.faults)
    for f in faults:
        if f not in available_faults():
            raise ValueError(f"Unsupported fault: {f}")

    device = pick_device(cfg.device)
    print(f"[info] device={device}")
    print(f"[info] models={models}")
    print(f"[info] seeds={seeds}")
    print(f"[info] faults={faults}")
    dataset_name = default_dataset_name(cfg.data, cfg.dataset_name)
    output_tag = sanitize_output_tag(cfg.output_tag) if cfg.output_tag else infer_output_tag(cfg.data, dataset_name)
    output_paths = build_fairness_output_paths(Path(cfg.out_dir), output_tag)

    data, _ = load_csv_data(cfg.data, cfg.target)
    ts_cfg = TimeSeriesConfig(seq_len=cfg.seq_len, pred_len=cfg.pred_len, target=cfg.target)
    tr_cfg = TrainConfig(batch_size=cfg.batch_size, epochs=1, num_workers=cfg.num_workers)
    train_loader, _, test_loader, scaler = create_dataloaders(data, ts_cfg, tr_cfg)
    input_dim = int(data.shape[1])

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    output_paths["checkpoints_dir"].mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_fairness_closure_fullfault_4090"

    rows: List[Dict] = []
    eval_faults = [FAULT_CLEAN] + faults
    for seed in seeds:
        set_seed(seed)
        print(f"\n[seed] {seed}")
        for model_name in models:
            disp = display_name(model_name)
            print(f"  [model] {disp}")
            base = build_model(
                model_name,
                input_dim=input_dim,
                ts_cfg=ts_cfg,
                device=device,
                base_config=cfg.base_config,
                base_checkpoint=cfg.base_checkpoint,
            )

            for train_mode in ["clean_trained", "fault_augmented"]:
                print(f"    [train_mode] {train_mode}")
                trained = finetune_model(
                    model_name,
                    base,
                    train_loader,
                    mode=train_mode,
                    faults=faults,
                    severity=cfg.severity,
                    epochs=cfg.epochs,
                    lr=cfg.lr,
                    alpha=cfg.alpha,
                    seed=seed,
                    device=device,
                )
                ckpt_name = f"{model_name}_{train_mode}_seed{seed}.pth"
                torch.save(trained.state_dict(), output_paths["checkpoints_dir"] / ckpt_name)

                clean_ref_rmse = None
                clean_ref_r2 = None
                temp_rows: List[Dict] = []
                for tf in eval_faults:
                    m = evaluate_model(
                        model_name,
                        trained,
                        test_loader,
                        scaler,
                        seed=seed,
                        fault=tf,
                        severity=cfg.severity,
                        alpha=cfg.alpha,
                        device=device,
                    )
                    rec = {
                        "run_id": run_id,
                        "seed": int(seed),
                        "model": disp,
                        "training_mode": train_mode,
                        "train_fault_protocol": "none" if train_mode == "clean_trained" else "uniform_fullfault",
                        "test_fault": tf,
                        "severity": 0.0 if tf == FAULT_CLEAN else float(cfg.severity),
                        "R2": float(m["R2"]),
                        "RMSE": float(m["RMSE"]),
                        "MAE": float(m["MAE"]),
                        "AE_Q90": float(m["AE_Q90"]),
                        "AE_P95": float(m["AE_P95"]),
                        "degradation_rmse_pct": np.nan,
                        "degradation_r2_drop_pct": np.nan,
                    }
                    if tf == FAULT_CLEAN:
                        clean_ref_rmse = float(m["RMSE"])
                        clean_ref_r2 = float(m["R2"])
                    temp_rows.append(rec)

                for rec in temp_rows:
                    if rec["test_fault"] != FAULT_CLEAN and clean_ref_rmse is not None and clean_ref_r2 is not None:
                        rec["degradation_rmse_pct"] = 100.0 * (float(rec["RMSE"]) - clean_ref_rmse) / max(clean_ref_rmse, 1e-12)
                        rec["degradation_r2_drop_pct"] = 100.0 * (clean_ref_r2 - float(rec["R2"])) / max(abs(clean_ref_r2), 1e-12)
                    rows.append(rec)

    df_raw = pd.DataFrame(rows).sort_values(
        ["model", "training_mode", "seed", "test_fault"], ascending=[True, True, True, True]
    )
    df_raw.to_csv(output_paths["raw_csv"], index=False)

    summary = (
        df_raw.groupby(["model", "training_mode", "test_fault"], as_index=False)
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            AE_Q90_mean=("AE_Q90", "mean"),
            AE_Q90_std=("AE_Q90", "std"),
            degradation_rmse_pct_mean=("degradation_rmse_pct", "mean"),
            degradation_r2_drop_pct_mean=("degradation_r2_drop_pct", "mean"),
            n_seed=("seed", "nunique"),
        )
        .reset_index(drop=True)
    )
    summary.to_csv(output_paths["summary_csv"], index=False)

    hard_rows: List[Dict] = []
    for (model, train_mode), g in summary.groupby(["model", "training_mode"]):
        row = {
            "model": model,
            "training_mode": train_mode,
            "n_seed": int(g["n_seed"].max()),
        }
        for fault in [FAULT_CLEAN] + faults:
            gf = g[g["test_fault"] == fault]
            row[f"{fault}_R2_mean"] = float(gf["R2_mean"].iloc[0]) if not gf.empty else np.nan
            row[f"{fault}_RMSE_mean"] = float(gf["RMSE_mean"].iloc[0]) if not gf.empty else np.nan
            row[f"{fault}_AE_Q90_mean"] = float(gf["AE_Q90_mean"].iloc[0]) if not gf.empty else np.nan
        corr = g[g["test_fault"] != FAULT_CLEAN]
        row["avg_degradation_rmse_pct"] = float(corr["degradation_rmse_pct_mean"].mean()) if not corr.empty else np.nan
        row["avg_degradation_r2_drop_pct"] = float(corr["degradation_r2_drop_pct_mean"].mean()) if not corr.empty else np.nan
        row["worst_fault_AE_Q90_mean"] = float(corr["AE_Q90_mean"].max()) if not corr.empty else np.nan
        hard_rows.append(row)

    hard = pd.DataFrame(hard_rows).sort_values(["training_mode", "model"])
    hard.to_csv(output_paths["hard_table_csv"], index=False)

    sig = compute_significance(df_raw, bootstrap_iters=cfg.bootstrap_iters)
    sig.to_csv(output_paths["significance_csv"], index=False)

    meta = {
        "script": "run_fairness_closure_fullfault_4090.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "config": asdict(cfg),
        "dataset": cfg.data,
        "dataset_name": dataset_name,
        "target": cfg.target,
        "output_tag": output_tag,
        "window": {"seq_len": cfg.seq_len, "pred_len": cfg.pred_len},
        "training_modes": ["clean_trained", "fault_augmented"],
        "fault_protocol": {
            "faults": faults,
            "severity": cfg.severity,
            "missingness_imputation": "forward-fill then zero-fill",
            "fault_aug_sampling": "uniform over configured faults per batch",
        },
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }
    output_paths["meta_json"].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[ok] wrote {output_paths['raw_csv']}")
    print(f"[ok] wrote {output_paths['summary_csv']}")
    print(f"[ok] wrote {output_paths['hard_table_csv']}")
    print(f"[ok] wrote {output_paths['significance_csv']}")
    print(f"[ok] wrote {output_paths['meta_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
