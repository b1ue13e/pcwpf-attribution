import json
import os
from datetime import datetime
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from run_dlinear import DLinear
from run_latest_sota import TimeMixer
from ultra_lsnt_timeseries import (
    LSNTConfig,
    TimeSeriesConfig,
    TrainConfig,
    UltraLSNTForecaster,
    compute_metrics,
    create_dataloaders,
    load_csv_data,
)


OUT_DIR = os.path.join("results", "supplementary_evidence")


class iTransformer(nn.Module):
    """Minimal iTransformer definition matching checkpoints_ts/iTransformer_best.pth."""

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


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_main_data():
    data, feature_names = load_csv_data("wind_final.csv", "power")
    ts_cfg = TimeSeriesConfig(seq_len=96, pred_len=24, target="power")
    train_cfg = TrainConfig(batch_size=128, epochs=1, num_workers=0)
    train_loader, val_loader, test_loader, scaler = create_dataloaders(data, ts_cfg, train_cfg)
    return data, feature_names, ts_cfg, train_loader, val_loader, test_loader, scaler


def estimate_power_curve_params(data: np.ndarray, feature_names: List[str], train_ratio: float = 0.7) -> Dict[str, float]:
    n_train = int(len(data) * train_ratio)
    train = data[:n_train]
    ws_idx = feature_names.index("windspeed")
    p_idx = feature_names.index("power")

    ws = train[:, ws_idx]
    power = train[:, p_idx]

    p_rated_train = float(np.max(power))
    p_rated_all = float(np.max(data[:, p_idx]))

    active_mask = power >= 0.05 * p_rated_train
    if active_mask.any():
        v_in_empirical = float(np.quantile(ws[active_mask], 0.01))
    else:
        v_in_empirical = float("nan")

    # In this farm-level dataset, cut-out shutdown is not clearly observed; keep an engineering default.
    v_out_assumed = 25.0
    v_obs_max_train = float(np.max(ws))

    return {
        "train_ratio": train_ratio,
        "n_train": n_train,
        "P_rated_train_max_kw": p_rated_train,
        "P_rated_all_max_kw": p_rated_all,
        "v_in_empirical_mps": v_in_empirical,
        "v_out_assumed_mps": v_out_assumed,
        "v_obs_max_train_mps": v_obs_max_train,
    }


def build_ultra_model(ts_cfg: TimeSeriesConfig, device: torch.device) -> UltraLSNTForecaster:
    cfgj = json.load(open("checkpoints_ts/main/model_config.json", "r", encoding="utf-8"))
    cfg = LSNTConfig(
        input_dim=cfgj["input_dim"],
        hidden_dim=cfgj["hidden_dim"],
        output_dim=1,
        num_blocks=cfgj["num_blocks"],
        num_experts=cfgj["num_experts"],
        top_k=cfgj["top_k"],
        dropout=cfgj.get("dropout", 0.1),
        router_z_loss_coef=cfgj.get("router_z_loss_coef", 0.01),
        router_aux_loss_coef=cfgj.get("router_aux_loss_coef", 0.01),
        router_jitter_noise=cfgj.get("router_jitter_noise", 0.1),
        skip_threshold=cfgj.get("skip_threshold", 0.5),
        dual_mode_expansion=cfgj.get("dual_mode_expansion", 4),
        probabilistic_mode="none",
        heteroscedastic_moe=False,
    )
    model = UltraLSNTForecaster(cfg, ts_cfg).to(device)
    ckpt = torch.load("checkpoints_ts/main/best_model.pth", map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def build_dlinear_model(device: torch.device) -> DLinear:
    ckpt = torch.load("dlinear_weights/dlinear_trained.pth", map_location=device)
    cfg = ckpt["config"]
    model = DLinear(cfg["seq_len"], cfg["pred_len"], cfg["input_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_itransformer_model(device: torch.device, input_dim: int = 8) -> iTransformer:
    cfg = {
        "seq_len": 96,
        "pred_len": 24,
        "input_dim": input_dim,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
    }
    model = iTransformer(cfg).to(device)
    state = torch.load("checkpoints_ts/iTransformer_best.pth", map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def build_timemixer_model(device: torch.device, input_dim: int = 8) -> TimeMixer:
    cfg = {
        "seq_len": 96,
        "pred_len": 24,
        "input_dim": input_dim,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "scales": [1, 2, 4],
    }
    model = TimeMixer(cfg).to(device)
    state = torch.load("checkpoints_ts/TimeMixer_best.pth", map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict_ultra(model: UltraLSNTForecaster, x: torch.Tensor) -> torch.Tensor:
    pred, _ = model(x, temperature=0.1)
    return pred


def predict_dlinear(model: DLinear, x: torch.Tensor) -> torch.Tensor:
    return model(x)


def predict_itrans(model: iTransformer, x: torch.Tensor) -> torch.Tensor:
    pred, _ = model(x)
    return pred.squeeze(-1)


def predict_timemixer(model: TimeMixer, x: torch.Tensor) -> torch.Tensor:
    pred, _ = model(x)
    return pred.squeeze(-1)


def inject_gaussian_protocol(x: torch.Tensor, sigma_cfg: float, alpha: float = 1.5) -> torch.Tensor:
    if sigma_cfg <= 0:
        return x
    std = x.std(dim=1, keepdim=True)
    noise = torch.randn_like(x) * (alpha * sigma_cfg) * std
    return x + noise


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
                    x[b, start:start + seg_len, c] = x[b, start, c]
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
            )
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
                x_nan[b, start:start + dur, channels] = float("nan")

        for t in range(1, seq_len):
            missing = torch.isnan(x_nan[:, t, :])
            x_nan[:, t, :] = torch.where(missing, x_nan[:, t - 1, :], x_nan[:, t, :])

        x_nan[torch.isnan(x_nan)] = 0.0
        return x_nan

    raise ValueError(f"Unsupported fault type: {fault_type}")


def evaluate_ultra_vs_pi(
    ultra_model: UltraLSNTForecaster,
    test_loader: torch.utils.data.DataLoader,
    scaler,
    ws_idx: int,
    p_rated_kw: float,
    v_in_mps: float,
    v_out_mps: float,
    device: torch.device,
) -> pd.DataFrame:
    target_mean = float(scaler.mean[-1])
    target_std = float(scaler.std[-1])
    ws_mean = float(scaler.mean[ws_idx])
    ws_std = float(scaler.std[ws_idx])

    scenarios = [
        {"scenario": "clean", "sigma_cfg": 0.0, "sigma_eff": 0.0},
        {"scenario": "gaussian_60pct", "sigma_cfg": 0.4, "sigma_eff": 0.6},
    ]
    rows = []

    for sc in scenarios:
        ultra_preds = []
        pi_preds = []
        trues = []

        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            x_noisy = inject_gaussian_protocol(x, sigma_cfg=sc["sigma_cfg"], alpha=1.5)

            with torch.no_grad():
                pred_norm = predict_ultra(ultra_model, x_noisy)

            pred_ultra = pred_norm * target_std + target_mean
            true_phys = y * target_std + target_mean

            ws_last_phys = x_noisy[:, -1, ws_idx] * ws_std + ws_mean
            ws_proxy = ws_last_phys.unsqueeze(1).expand_as(pred_ultra)

            pred_pi = torch.clamp(pred_ultra, min=0.0, max=p_rated_kw)
            cut_mask = (ws_proxy < v_in_mps) | (ws_proxy > v_out_mps)
            pred_pi = torch.where(cut_mask, torch.zeros_like(pred_pi), pred_pi)

            ultra_preds.append(pred_ultra.cpu().numpy())
            pi_preds.append(pred_pi.cpu().numpy())
            trues.append(true_phys.cpu().numpy())

        ultra_preds = np.concatenate(ultra_preds, axis=0)
        pi_preds = np.concatenate(pi_preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        ultra_m = compute_metrics(ultra_preds, trues)
        pi_m = compute_metrics(pi_preds, trues)

        rows.append(
            {
                "scenario": sc["scenario"],
                "sigma_cfg": sc["sigma_cfg"],
                "sigma_eff": sc["sigma_eff"],
                "model": "Ultra-LSNT",
                "R2": float(ultra_m["R2"]),
                "RMSE": float(ultra_m["RMSE"]),
                "MAE": float(ultra_m["MAE"]),
            }
        )
        rows.append(
            {
                "scenario": sc["scenario"],
                "sigma_cfg": sc["sigma_cfg"],
                "sigma_eff": sc["sigma_eff"],
                "model": "PI-MoE (projection-only)",
                "R2": float(pi_m["R2"]),
                "RMSE": float(pi_m["RMSE"]),
                "MAE": float(pi_m["MAE"]),
            }
        )

    return pd.DataFrame(rows)


def evaluate_structured_faults(
    model_predictors: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
    test_loader: torch.utils.data.DataLoader,
    scaler,
    device: torch.device,
    seed: int = 42,
) -> pd.DataFrame:
    target_mean = float(scaler.mean[-1])
    target_std = float(scaler.std[-1])

    scenarios = [{"fault_type": "clean", "severity": 0.0}]
    for f in ["flatline", "spike", "dropout"]:
        for sev in [0.2, 0.6]:
            scenarios.append({"fault_type": f, "severity": sev})

    rows = []

    for sc_idx, sc in enumerate(scenarios):
        preds_per_model = {k: [] for k in model_predictors.keys()}
        trues_all = []

        for batch_idx, (x, y) in enumerate(test_loader):
            if sc["fault_type"] == "clean":
                x_fault = x
            else:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(seed + sc_idx * 100000 + batch_idx)
                x_fault = inject_structured_fault(
                    x,
                    fault_type=sc["fault_type"],
                    severity=float(sc["severity"]),
                    generator=gen,
                )

            x_dev = x_fault.to(device)
            y_dev = y.to(device)
            true_phys = y_dev * target_std + target_mean
            trues_all.append(true_phys.cpu().numpy())

            for model_name, predict_fn in model_predictors.items():
                with torch.no_grad():
                    pred_norm = predict_fn(x_dev)
                pred_phys = pred_norm * target_std + target_mean
                preds_per_model[model_name].append(pred_phys.cpu().numpy())

        trues = np.concatenate(trues_all, axis=0)
        for model_name, pred_list in preds_per_model.items():
            preds = np.concatenate(pred_list, axis=0)
            m = compute_metrics(preds, trues)
            rows.append(
                {
                    "fault_type": sc["fault_type"],
                    "severity": float(sc["severity"]),
                    "model": model_name,
                    "R2": float(m["R2"]),
                    "RMSE": float(m["RMSE"]),
                    "MAE": float(m["MAE"]),
                }
            )

    out_df = pd.DataFrame(rows)

    # Relative drop against each model's clean baseline.
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
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    data, feature_names, ts_cfg, _, _, test_loader, scaler = load_main_data()
    ws_idx = feature_names.index("windspeed")

    curve = estimate_power_curve_params(data, feature_names, train_ratio=0.7)
    curve_df = pd.DataFrame([curve])
    curve_csv = os.path.join(OUT_DIR, "pi_moe_curve_params_windcn.csv")
    curve_df.to_csv(curve_csv, index=False)

    availability_df = pd.DataFrame(
        [
            {
                "item": "P_rated",
                "available": True,
                "definition_or_proxy": "train-set max power (kW)",
                "value_or_note": curve["P_rated_train_max_kw"],
            },
            {
                "item": "v_in",
                "available": True,
                "definition_or_proxy": "empirical threshold from train set",
                "value_or_note": curve["v_in_empirical_mps"],
            },
            {
                "item": "v_out",
                "available": "partially",
                "definition_or_proxy": "engineering default due no clear cut-out in data",
                "value_or_note": curve["v_out_assumed_mps"],
            },
            {
                "item": "v_hat_(t+h)",
                "available": "proxy",
                "definition_or_proxy": "persistence nowcast from last observed windspeed",
                "value_or_note": "used in PI projection",
            },
            {
                "item": "beta_hat_(t+h)",
                "available": False,
                "definition_or_proxy": "pitch-angle channel not present in wind_final.csv",
                "value_or_note": "not used",
            },
        ]
    )
    availability_csv = os.path.join(OUT_DIR, "pi_moe_physics_variable_availability_windcn.csv")
    availability_df.to_csv(availability_csv, index=False)

    ultra_model = build_ultra_model(ts_cfg, device)
    dlinear_model = build_dlinear_model(device)
    itrans_model = build_itransformer_model(device, input_dim=data.shape[1])
    timemixer_model = build_timemixer_model(device, input_dim=data.shape[1])

    compare_df = evaluate_ultra_vs_pi(
        ultra_model=ultra_model,
        test_loader=test_loader,
        scaler=scaler,
        ws_idx=ws_idx,
        p_rated_kw=float(curve["P_rated_train_max_kw"]),
        v_in_mps=float(curve["v_in_empirical_mps"]),
        v_out_mps=float(curve["v_out_assumed_mps"]),
        device=device,
    )
    compare_csv = os.path.join(OUT_DIR, "pi_moe_vs_ultra_clean_sigma06_windcn.csv")
    compare_df.to_csv(compare_csv, index=False)

    predictors = {
        "Ultra-LSNT": lambda x: predict_ultra(ultra_model, x),
        "iTransformer": lambda x: predict_itrans(itrans_model, x),
        "TimeMixer": lambda x: predict_timemixer(timemixer_model, x),
        "DLinear": lambda x: predict_dlinear(dlinear_model, x),
    }
    struct_df = evaluate_structured_faults(
        model_predictors=predictors,
        test_loader=test_loader,
        scaler=scaler,
        device=device,
        seed=42,
    )
    struct_csv = os.path.join(OUT_DIR, "structured_scada_fault_robustness_windcn.csv")
    struct_df.to_csv(struct_csv, index=False)

    meta = {
        "script": "run_pi_moe_evidence_bundle.py",
        "dataset": "wind_final.csv",
        "target": "power",
        "split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
        "seq_len": 96,
        "pred_len": 24,
        "noise_protocol": {
            "gaussian_sigma_cfg_points": [0.0, 0.4],
            "sigma_eff_formula": "sigma_eff = 1.5 * sigma_cfg",
            "sigma_eff_points": [0.0, 0.6],
        },
        "structured_fault_protocol": {
            "fault_types": ["flatline", "spike", "dropout"],
            "severities": [0.2, 0.6],
            "dropout_imputation": "forward-fill then zero-fill",
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "outputs": {
            "curve_params_csv": curve_csv,
            "availability_csv": availability_csv,
            "pi_vs_ultra_csv": compare_csv,
            "structured_fault_csv": struct_csv,
        },
    }
    meta_json = os.path.join(OUT_DIR, "pi_moe_evidence_meta.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n[done] generated files:")
    print(f"- {curve_csv}")
    print(f"- {availability_csv}")
    print(f"- {compare_csv}")
    print(f"- {struct_csv}")
    print(f"- {meta_json}")
    print("\n[preview] PI-MoE vs Ultra (clean + 60% noise)")
    print(compare_df.to_string(index=False))
    print("\n[preview] Structured SCADA robustness")
    print(struct_df.to_string(index=False))


if __name__ == "__main__":
    main()
