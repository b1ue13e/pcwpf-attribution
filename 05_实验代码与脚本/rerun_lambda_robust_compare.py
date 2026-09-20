import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ultra_lsnt_timeseries import (
    load_csv_data,
    create_dataloaders,
    TimeSeriesConfig,
    TrainConfig,
    LSNTConfig,
    UltraLSNTForecaster,
    compute_metrics,
)


OUT_DIR = os.path.join("results", "supplementary_evidence")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_OUT = os.path.join(OUT_DIR, "lambda_robustness_sigma_eff_compare.csv")
META_OUT = os.path.join(OUT_DIR, "lambda_robustness_sigma_eff_compare.meta.json")
CKPT_OUT = os.path.join(OUT_DIR, "ultra_lambda005_finetuned_from_main.pth")

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(aux_coef: float, ts_config: TimeSeriesConfig, device: torch.device) -> UltraLSNTForecaster:
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
        router_aux_loss_coef=float(aux_coef),
        router_jitter_noise=cfgj.get("router_jitter_noise", 0.1),
        skip_threshold=cfgj.get("skip_threshold", 0.5),
        dual_mode_expansion=cfgj.get("dual_mode_expansion", 4),
        probabilistic_mode="none",
        heteroscedastic_moe=False,
    )
    model = UltraLSNTForecaster(cfg, ts_config).to(device)

    ckpt = torch.load("checkpoints_ts/main/best_model.pth", map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    for blk in model.blocks:
        blk.moe_router.aux_loss_coef = float(aux_coef)
    return model


@torch.no_grad()
def eval_with_noise(
    model: UltraLSNTForecaster,
    test_loader,
    scaler,
    device: torch.device,
    noise_levels,
):
    model.eval()
    out = {}
    for sigma in noise_levels:
        preds, trues = [], []
        for bx, by in test_loader:
            bx = bx.to(device)
            by = by.to(device)
            if sigma > 0:
                bx = bx + torch.randn_like(bx) * float(sigma)
            pred, _ = model(bx, temperature=0.1)
            preds.append(pred.cpu().numpy())
            trues.append(by.cpu().numpy())

        p = np.concatenate(preds, axis=0)
        y = np.concatenate(trues, axis=0)
        p = p * scaler.std[-1] + scaler.mean[-1]
        y = y * scaler.std[-1] + scaler.mean[-1]
        m = compute_metrics(p, y)
        out[sigma] = {
            "R2": float(m["R2"]),
            "RMSE": float(m["RMSE"]),
            "MAE": float(m["MAE"]),
        }
    return out


def train_lambda005(
    model: UltraLSNTForecaster,
    train_loader,
    val_loader,
    scaler,
    device: torch.device,
    epochs: int = 5,
):
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    mse = nn.MSELoss()
    best_val_mse = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad()
            pred, aux = model(bx, temperature=1.0)
            loss = mse(pred, by) + 0.05 * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        vp, vy = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)
                pred, _ = model(bx, temperature=0.1)
                vp.append(pred.cpu().numpy())
                vy.append(by.cpu().numpy())
        vp = np.concatenate(vp, axis=0) * scaler.std[-1] + scaler.mean[-1]
        vy = np.concatenate(vy, axis=0) * scaler.std[-1] + scaler.mean[-1]
        vm = compute_metrics(vp, vy)
        val_mse = float(vm["MSE"])

        print(
            f"epoch={ep} train_loss={np.mean(losses):.6f} "
            f"val_R2={vm['R2']:.6f} val_MSE={val_mse:.3f}",
            flush=True,
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    torch.save(
        {"model_state_dict": model.state_dict(), "best_val_mse": best_val_mse, "lambda": 0.05},
        CKPT_OUT,
    )
    return best_val_mse


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device, flush=True)

    data, _ = load_csv_data("wind_final.csv", "power")
    ts = TimeSeriesConfig(seq_len=96, pred_len=24, target="power")
    tc = TrainConfig(batch_size=128, num_workers=0, epochs=1)
    train_loader, val_loader, test_loader, scaler = create_dataloaders(data, ts, tc)

    m001 = build_model(0.01, ts, device)
    res001 = eval_with_noise(m001, test_loader, scaler, device, noise_levels=[0.0, 0.2, 0.6])

    m005 = build_model(0.05, ts, device)
    best_val_mse = train_lambda005(m005, train_loader, val_loader, scaler, device, epochs=5)
    res005 = eval_with_noise(m005, test_loader, scaler, device, noise_levels=[0.0, 0.2, 0.6])

    mapping = [
        (0.2, 0.30, "legacy-anchor-from-robust-sweep"),
        (0.6, 0.60, "legacy-anchor-from-robust-sweep"),
    ]
    rows = []
    for lam, res in [(0.01, res001), (0.05, res005)]:
        for noise_std, sigma_eff, note in mapping:
            r = res[noise_std]
            rows.append(
                {
                    "lambda": lam,
                    "noise_std_normalized": noise_std,
                    "reported_sigma_eff": sigma_eff,
                    "R2": r["R2"],
                    "RMSE": r["RMSE"],
                    "MAE": r["MAE"],
                    "note": note,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)

    meta = {
        "seed": SEED,
        "device": str(device),
        "data": "wind_final.csv",
        "target": "power",
        "split": "create_dataloaders default (train/val/test)",
        "inverse_scaling": True,
        "noise_injection": "x_noisy = x + N(0, noise_std_normalized) in normalized input space",
        "lambda_005_training": {
            "init": "load checkpoints_ts/main/best_model.pth",
            "epochs": 5,
            "optimizer": "AdamW(lr=5e-5, wd=0.01)",
            "loss": "MSE + 0.05 * aux",
            "best_val_mse": best_val_mse,
            "checkpoint_out": CKPT_OUT,
        },
        "lambda_001_clean_r2": res001[0.0]["R2"],
        "lambda_001_r2_noise_0.2": res001[0.2]["R2"],
        "lambda_001_r2_noise_0.6": res001[0.6]["R2"],
        "lambda_005_clean_r2": res005[0.0]["R2"],
        "lambda_005_r2_noise_0.2": res005[0.2]["R2"],
        "lambda_005_r2_noise_0.6": res005[0.6]["R2"],
    }
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nSaved:", CSV_OUT)
    print(df.to_string(index=False))
    print("\nMeta:", META_OUT)


if __name__ == "__main__":
    main()
