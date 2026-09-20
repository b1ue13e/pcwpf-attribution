from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from battle_dlinear import DLinear
from ultra_lsnt_timeseries import (
    LSNTConfig,
    TimeSeriesConfig,
    TrainConfig,
    UltraLSNTForecaster,
    create_dataloaders,
    load_csv_data,
)
from paper_figure_style import ensure_output_dir, save_figure, set_paper_style
from paper_figure_style import OKABE_ITO


set_paper_style()
OUT_DIR = ensure_output_dir("figure")


def _load_state_dict(path: Path, device: torch.device) -> dict[str, Any]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def _to_2d_pred(arr: np.ndarray) -> np.ndarray:
    # Accept (B, pred_len), (B, pred_len, 1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"Unexpected prediction shape: {arr.shape}")


def penalty_cost_usd(
    preds_power: np.ndarray,
    trues_power: np.ndarray,
    *,
    step_hours: float = 0.25,
    penalty_rate_usd_per_mwh: float = 50.0,
) -> float:
    """
    Convert forecast deviation into an accumulated penalty cost.

    Assumptions:
    - Data sampling interval: 15 minutes (0.25h), consistent with the rest of this repo.
    - Power is in kW for wind_final.csv (heuristic: if max > 2000, treat as kW).
    """
    preds_power = np.asarray(preds_power)
    trues_power = np.asarray(trues_power)
    if preds_power.shape != trues_power.shape:
        raise ValueError(f"Shape mismatch preds={preds_power.shape} trues={trues_power.shape}")

    scale = 1000.0 if float(np.nanmax(trues_power)) > 2000.0 else 1.0  # kW -> MW
    preds_mw = preds_power / scale
    trues_mw = trues_power / scale

    deviation_mw = np.abs(preds_mw - trues_mw)
    deviation_mwh = deviation_mw * step_hours
    return float(np.sum(deviation_mwh) * penalty_rate_usd_per_mwh)


def run_economic_battle(
    *,
    data_path: str | Path = "wind_final.csv",
    ckpt_ultra: str | Path = "checkpoints_ts/main/best_model.pth",
    ckpt_ultra_config: str | Path = "checkpoints_ts/main/model_config.json",
    ckpt_dlinear: str | Path = "dlinear_weights/dlinear_trained.pth",
    max_batches: int = 160,
    noise_levels: list[float] | None = None,
) -> None:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]

    # Reproducibility for paper figures
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target="power")
    data, _ = load_csv_data(str(data_path), ts_config.target)
    input_dim = int(data.shape[1])

    train_config = TrainConfig(batch_size=128, num_workers=0)
    _, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)

    target_std = float(scaler.std[-1])
    target_mean = float(scaler.mean[-1])

    # --- Ultra-LSNT ---
    # Load state_dict first to infer whether the checkpoint uses heteroscedastic routing (dim+1).
    ckpt_ultra_path = Path(ckpt_ultra)
    if not ckpt_ultra_path.exists():
        raise FileNotFoundError(f"Missing Ultra-LSNT checkpoint: {ckpt_ultra_path}")
    ultra_state = _load_state_dict(ckpt_ultra_path, device)

    router_w = ultra_state.get("blocks.0.moe_router.router.weight", None)
    router_in_dim = int(router_w.shape[1]) if hasattr(router_w, "shape") else None

    model_config = LSNTConfig(input_dim=input_dim, heteroscedastic_moe=False)
    config_path = Path(ckpt_ultra_config)
    if config_path.exists():
        import json

        raw = json.loads(config_path.read_text(encoding="utf-8", errors="ignore"))
        allowed = set(LSNTConfig.__dataclass_fields__.keys())
        filtered = {k: v for k, v in raw.items() if k in allowed}
        filtered["input_dim"] = input_dim
        filtered.setdefault("heteroscedastic_moe", False)
        if router_in_dim is not None:
            hidden_dim = int(filtered.get("hidden_dim", router_in_dim))
            if router_in_dim == hidden_dim + 1:
                filtered["heteroscedastic_moe"] = True
            elif router_in_dim == hidden_dim:
                filtered["heteroscedastic_moe"] = False
            else:
                # Fallback: trust checkpoint shape more than config file.
                filtered["hidden_dim"] = router_in_dim
                filtered["heteroscedastic_moe"] = False
        model_config = LSNTConfig(**filtered)

    model_ultra = UltraLSNTForecaster(model_config, ts_config).to(device)
    model_ultra.load_state_dict(ultra_state, strict=False)
    model_ultra.eval()

    # --- DLinear (load trained weights if available) ---
    model_dlinear = DLinear(ts_config.seq_len, ts_config.pred_len, input_dim).to(device)
    ckpt_dlinear_path = Path(ckpt_dlinear)
    if ckpt_dlinear_path.exists():
        model_dlinear.load_state_dict(_load_state_dict(ckpt_dlinear_path, device), strict=False)
    else:
        raise FileNotFoundError(f"Missing DLinear checkpoint: {ckpt_dlinear_path}")
    model_dlinear.eval()

    costs_ultra: list[float] = []
    costs_dlinear: list[float] = []

    for noise in noise_levels:
        preds_u: list[np.ndarray] = []
        preds_d: list[np.ndarray] = []
        trues: list[np.ndarray] = []

        with torch.no_grad():
            for batch_idx, (bx, by) in enumerate(test_loader):
                if batch_idx >= max_batches:
                    break

                bx = bx.to(device)
                by = by.to(device)

                bx_noisy = bx + torch.randn_like(bx) * float(noise)

                out_u, _ = model_ultra(bx_noisy)
                out_d = model_dlinear(bx_noisy)

                preds_u.append(_to_2d_pred(out_u.detach().cpu().numpy()))
                preds_d.append(_to_2d_pred(out_d.detach().cpu().numpy()))
                trues.append(_to_2d_pred(by.detach().cpu().numpy()))

        p_u = np.concatenate(preds_u, axis=0) * target_std + target_mean
        p_d = np.concatenate(preds_d, axis=0) * target_std + target_mean
        y = np.concatenate(trues, axis=0) * target_std + target_mean

        cost_u = penalty_cost_usd(p_u, y, penalty_rate_usd_per_mwh=50.0)
        cost_d = penalty_cost_usd(p_d, y, penalty_rate_usd_per_mwh=50.0)
        costs_ultra.append(cost_u)
        costs_dlinear.append(cost_d)

        print(f"Noise {noise:.2f} -> Ultra-LSNT cost: ${cost_u:,.0f} | DLinear cost: ${cost_d:,.0f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7.1, 4.8), dpi=300)
    ax.plot(
        noise_levels,
        np.array(costs_ultra) / 1e6,
        marker="o",
        color=OKABE_ITO["blue"],
        linewidth=2.6,
        label="Ultra-LSNT (Ours)",
    )
    ax.plot(
        noise_levels,
        np.array(costs_dlinear) / 1e6,
        marker="x",
        color=OKABE_ITO["vermillion"],
        linestyle="--",
        linewidth=2.2,
        label="DLinear (Baseline)",
    )

    # Value of robustness (savings)
    u = np.array(costs_ultra) / 1e6
    d = np.array(costs_dlinear) / 1e6
    ax.fill_between(
        noise_levels,
        u,
        d,
        where=(d > u),
        color=OKABE_ITO["bluish_green"],
        alpha=0.12,
        label="Avoided Penalty (Value of Robustness)",
    )

    ax.set_xlabel(r"Sensor Noise Level ($\sigma$)", fontweight="bold")
    ax.set_ylabel("Accumulated Penalty Cost (Million USD)", fontweight="bold")
    ax.set_title("Economic Risk Assessment: Penalty Analysis")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")

    fig.tight_layout()
    save_figure(fig, OUT_DIR / "fig_economic_impact")
    plt.close(fig)


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    run_economic_battle()
