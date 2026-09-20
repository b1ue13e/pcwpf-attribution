import json
import os
import time
import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


OUT_DIR = os.path.join("results", "supplementary_evidence")
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_windows(features: np.ndarray, target: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    n = len(target)
    for i in range(n - seq_len - pred_len + 1):
        x.append(features[i : i + seq_len])
        y.append(target[i + seq_len : i + seq_len + pred_len])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def _mirror_extend(signal: np.ndarray) -> np.ndarray:
    return np.concatenate([signal[::-1], signal, signal[::-1]])


def vmd_minimal(
    signal: np.ndarray,
    alpha: float = 2000.0,
    tau: float = 0.0,
    k_modes: int = 3,
    dc: int = 0,
    init: int = 1,
    tol: float = 1e-7,
    max_iter: int = 250,
) -> np.ndarray:
    """
    Minimal VMD implementation adapted for auditable lightweight experiments.
    Returns modes with shape [k_modes, len(signal)].
    """
    f = signal.astype(np.float64)
    n = f.shape[0]
    f_mirror = _mirror_extend(f)
    # Keep an even FFT grid to avoid half-split shape mismatch.
    if f_mirror.shape[0] % 2 == 1:
        f_mirror = f_mirror[:-1]
    t = np.arange(1, len(f_mirror) + 1) / len(f_mirror)
    freqs = t - 0.5 - (1.0 / len(f_mirror))

    f_hat = np.fft.fftshift(np.fft.fft(f_mirror))
    f_hat_plus = f_hat.copy()
    f_hat_plus[: len(f_hat) // 2] = 0

    u_hat_plus = np.zeros((max_iter, len(freqs), k_modes), dtype=np.complex128)
    omega_plus = np.zeros((max_iter, k_modes), dtype=np.float64)

    if init == 1:
        for k in range(k_modes):
            omega_plus[0, k] = (0.5 / k_modes) * k
    elif init == 2:
        omega_plus[0, :] = np.sort(
            np.exp(np.log(1.0 / len(f_mirror)) + (np.log(0.5) - np.log(1.0 / len(f_mirror))) * np.random.rand(k_modes))
        )
    else:
        omega_plus[0, :] = 0
    if dc:
        omega_plus[0, 0] = 0

    lambda_hat = np.zeros((max_iter, len(freqs)), dtype=np.complex128)
    u_diff = tol + np.finfo(float).eps
    it = 0

    while u_diff > tol and it < max_iter - 1:
        k = 0
        sum_uk = u_hat_plus[it, :, k_modes - 1] - u_hat_plus[it, :, 0]
        u_hat_plus[it + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[it, :] / 2) / (
            1 + alpha * (freqs - omega_plus[it, k]) ** 2
        )
        if not dc:
            denom = np.sum(np.abs(u_hat_plus[it + 1, len(freqs) // 2 :, k]) ** 2) + 1e-12
            omega_plus[it + 1, k] = np.sum(
                freqs[len(freqs) // 2 :] * np.abs(u_hat_plus[it + 1, len(freqs) // 2 :, k]) ** 2
            ) / denom

        for k in range(1, k_modes):
            sum_uk = u_hat_plus[it + 1, :, k - 1] + sum_uk - u_hat_plus[it, :, k]
            u_hat_plus[it + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[it, :] / 2) / (
                1 + alpha * (freqs - omega_plus[it, k]) ** 2
            )
            denom = np.sum(np.abs(u_hat_plus[it + 1, len(freqs) // 2 :, k]) ** 2) + 1e-12
            omega_plus[it + 1, k] = np.sum(
                freqs[len(freqs) // 2 :] * np.abs(u_hat_plus[it + 1, len(freqs) // 2 :, k]) ** 2
            ) / denom

        lambda_hat[it + 1, :] = lambda_hat[it, :] + tau * (np.sum(u_hat_plus[it + 1, :, :], axis=1) - f_hat_plus)

        it += 1
        u_diff = np.finfo(float).eps
        for k in range(k_modes):
            num = u_hat_plus[it, :, k] - u_hat_plus[it - 1, :, k]
            u_diff += (1.0 / len(freqs)) * np.dot(num, np.conj(num)).real

    niter = min(max_iter, it + 1)
    u_hat = np.zeros((len(freqs), k_modes), dtype=np.complex128)
    u_hat[len(freqs) // 2 :, :] = u_hat_plus[niter - 1, len(freqs) // 2 :, :]
    u_hat[: len(freqs) // 2, :] = np.conj(u_hat[len(freqs) // 2 :, :][::-1, :])
    u_hat[0, :] = np.conj(u_hat[-1, :])

    u = np.zeros((k_modes, len(freqs)), dtype=np.float64)
    for k in range(k_modes):
        u[k, :] = np.fft.ifft(np.fft.ifftshift(u_hat[:, k])).real

    t_len = len(freqs) // 4
    u = u[:, t_len : t_len + n]
    return u


def _extract_imf(signal: np.ndarray, max_sift: int = 8, stop_tol: float = 1e-3) -> np.ndarray:
    h = signal.copy()
    n = len(h)
    x = np.arange(n)
    for _ in range(max_sift):
        max_idx = argrelextrema(h, np.greater)[0]
        min_idx = argrelextrema(h, np.less)[0]
        if len(max_idx) < 2 or len(min_idx) < 2:
            break

        max_idx = np.concatenate(([0], max_idx, [n - 1]))
        min_idx = np.concatenate(([0], min_idx, [n - 1]))
        max_vals = h[max_idx]
        min_vals = h[min_idx]

        upper = CubicSpline(max_idx, max_vals, bc_type="natural")(x)
        lower = CubicSpline(min_idx, min_vals, bc_type="natural")(x)
        mean_env = 0.5 * (upper + lower)
        h_new = h - mean_env

        if np.mean((h_new - h) ** 2) < stop_tol:
            h = h_new
            break
        h = h_new
    return h


def ceemdan_like_minimal(
    signal: np.ndarray,
    n_imfs: int = 3,
    trials: int = 4,
    noise_width: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """
    Lightweight CEEMDAN-like decomposition for auditable baseline prototyping.
    Returns modes with shape [n_imfs + 1, len(signal)] (last one is residual).
    """
    rng = np.random.default_rng(seed)
    residual = signal.astype(np.float64).copy()
    modes: List[np.ndarray] = []
    base_std = np.std(signal) + 1e-12

    for i in range(n_imfs):
        imf_ens = []
        sigma = noise_width * (np.std(residual) / base_std) * base_std
        for _ in range(trials):
            noisy = residual + rng.normal(0.0, sigma, size=residual.shape)
            imf_ens.append(_extract_imf(noisy))
        imf = np.mean(np.vstack(imf_ens), axis=0)
        modes.append(imf)
        residual = residual - imf
        if len(argrelextrema(residual, np.greater)[0]) + len(argrelextrema(residual, np.less)[0]) < 4:
            break

    modes.append(residual)
    return np.vstack(modes)


def decompose_with_downsample(
    signal: np.ndarray,
    method: str,
    ds_factor: int = 8,
    vmd_k: int = 3,
    ceemdan_k: int = 3,
) -> np.ndarray:
    n = len(signal)
    idx = np.arange(0, n, ds_factor, dtype=np.int64)
    sig_ds = signal[idx]
    if method == "vmd":
        modes_ds = vmd_minimal(sig_ds, k_modes=vmd_k)
    elif method == "ceemdan":
        modes_ds = ceemdan_like_minimal(sig_ds, n_imfs=ceemdan_k, trials=4, noise_width=0.15, seed=42)
    else:
        raise ValueError(method)

    all_modes = []
    x_full = np.arange(n, dtype=np.int64)
    for m in modes_ds:
        all_modes.append(np.interp(x_full, idx, m))
    return np.vstack(all_modes)


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, pred_len: int, dropout: float):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, pred_len: int, dropout: float):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


@dataclass
class ModelConfig:
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    weight_decay: float


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ModelConfig,
    device: torch.device,
    epochs: int = 4,
    batch_size: int = 128,
    patience: int = 2,
) -> Tuple[nn.Module, float]:
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.MSELoss()

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    best = float("inf")
    best_state = None
    stall = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                l = crit(pred, yb).item()
                val_loss += l * xb.size(0)
                n += xb.size(0)
        val_loss = val_loss / max(n, 1)

        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def infer(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            out.append(pred)
    return np.concatenate(out, axis=0)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    return {
        "R2": float(r2_score(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
    }


def inject_gaussian(x: np.ndarray, sigma_eff: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    std = np.std(x, axis=1, keepdims=True) + 1e-12
    y = x + rng.normal(0.0, sigma_eff, size=x.shape) * std
    return y.astype(np.float32)


def inject_spike(x: np.ndarray, severity: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = x.copy()
    p = 0.02 + 0.03 * severity
    feat_std = np.std(x, axis=(0, 1), keepdims=True) + 1e-12
    mask = rng.random(size=x.shape) < p
    spike = rng.laplace(0.0, (0.5 + severity) * feat_std, size=x.shape)
    z[mask] = z[mask] + spike[mask]
    return z.astype(np.float32)


def measure_latency_ms(model: nn.Module, sample: np.ndarray, device: torch.device, runs: int = 120) -> Dict[str, float]:
    x = torch.from_numpy(sample[:1]).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "latency_mean_ms": float(np.mean(arr)),
        "latency_std_ms": float(np.std(arr)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
    }


def state_size_mb(model: nn.Module) -> float:
    total = 0
    for v in model.state_dict().values():
        total += v.numel() * v.element_size()
    return float(total / (1024 ** 2))


def pso_optimize(objective_fn, bounds: List[Tuple[float, float]], pop_size: int = 5, iters: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds], dtype=np.float64)
    ub = np.array([b[1] for b in bounds], dtype=np.float64)
    pos = rng.uniform(lb, ub, size=(pop_size, dim))
    vel = np.zeros_like(pos)
    pbest = pos.copy()
    pbest_score = np.array([objective_fn(p) for p in pos], dtype=np.float64)
    g_idx = int(np.argmin(pbest_score))
    gbest = pbest[g_idx].copy()
    gscore = float(pbest_score[g_idx])
    trace = []
    for it in range(iters):
        for i in range(pop_size):
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            vel[i] = 0.6 * vel[i] + 1.2 * r1 * (pbest[i] - pos[i]) + 1.2 * r2 * (gbest - pos[i])
            pos[i] = np.clip(pos[i] + vel[i], lb, ub)
            s = float(objective_fn(pos[i]))
            if s < pbest_score[i]:
                pbest_score[i] = s
                pbest[i] = pos[i].copy()
                if s < gscore:
                    gscore = s
                    gbest = pos[i].copy()
        trace.append({"iter": it, "best_score": gscore, "best_params": gbest.tolist()})
    return gbest, gscore, trace


def woa_optimize(objective_fn, bounds: List[Tuple[float, float]], pop_size: int = 5, iters: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds], dtype=np.float64)
    ub = np.array([b[1] for b in bounds], dtype=np.float64)
    whales = rng.uniform(lb, ub, size=(pop_size, dim))
    scores = np.array([objective_fn(w) for w in whales], dtype=np.float64)
    b_idx = int(np.argmin(scores))
    best = whales[b_idx].copy()
    best_score = float(scores[b_idx])
    trace = []
    for t in range(iters):
        a = 2 - 2 * (t / max(iters - 1, 1))
        for i in range(pop_size):
            r1 = rng.random()
            r2 = rng.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = rng.random()
            l = rng.uniform(-1, 1)
            if p < 0.5:
                if abs(A) < 1:
                    D = np.abs(C * best - whales[i])
                    whales[i] = best - A * D
                else:
                    rand_idx = rng.integers(0, pop_size)
                    X_rand = whales[rand_idx]
                    D = np.abs(C * X_rand - whales[i])
                    whales[i] = X_rand - A * D
            else:
                D = np.abs(best - whales[i])
                whales[i] = D * np.exp(1.0 * l) * np.cos(2 * np.pi * l) + best
            whales[i] = np.clip(whales[i], lb, ub)
            s = float(objective_fn(whales[i]))
            if s < scores[i]:
                scores[i] = s
            if s < best_score:
                best_score = s
                best = whales[i].copy()
        trace.append({"iter": t, "best_score": best_score, "best_params": best.tolist()})
    return best, best_score, trace


def decode_cfg(params: np.ndarray) -> ModelConfig:
    hidden_dim = int(np.clip(np.round(params[0]), 16, 96))
    num_layers = int(np.clip(np.round(params[1]), 1, 2))
    dropout = float(np.clip(params[2], 0.0, 0.35))
    lr = float(np.clip(params[3], 1e-4, 3e-3))
    wd = float(np.clip(params[4], 1e-7, 1e-3))
    return ModelConfig(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, lr=lr, weight_decay=wd)


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    df = pd.read_csv("wind_final.csv")
    numeric = df.select_dtypes(include=[np.number]).copy()
    if "power" not in numeric.columns:
        raise RuntimeError("power column not found in wind_final.csv")
    cols = [c for c in numeric.columns if c != "power"] + ["power"]
    numeric = numeric[cols]
    arr = numeric.values.astype(np.float64)

    n = len(arr)
    n_train = int(n * 0.8)
    train_raw = arr[:n_train]
    test_raw = arr[n_train:]
    target_idx = len(cols) - 1

    print("[info] running decomposition ...")
    power_all = arr[:, target_idx]
    vmd_modes = decompose_with_downsample(power_all, method="vmd", ds_factor=8, vmd_k=3, ceemdan_k=3)
    ceemdan_modes = decompose_with_downsample(power_all, method="ceemdan", ds_factor=8, vmd_k=3, ceemdan_k=3)

    # Use first 3 components for each decomposition as additive features.
    vmd_feats_all = np.hstack([arr, vmd_modes[:3].T])
    ceemdan_feats_all = np.hstack([arr, ceemdan_modes[:3].T])

    seq_len = 96
    pred_len = 24

    def build_xy(feats_all: np.ndarray):
        feats_train = feats_all[:n_train]
        feats_test = feats_all[n_train:]
        scaler_x = StandardScaler().fit(feats_train)
        x_train_scaled = scaler_x.transform(feats_train)
        x_test_scaled = scaler_x.transform(feats_test)

        y_train = train_raw[:, target_idx]
        y_test = test_raw[:, target_idx]
        scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        xtr, ytr = create_windows(x_train_scaled, y_train_scaled, seq_len, pred_len)
        xte, yte = create_windows(x_test_scaled, y_test_scaled, seq_len, pred_len)
        return xtr, ytr, xte, yte, scaler_y

    xtr_vmd, ytr_vmd, xte_vmd, yte_vmd, sy_vmd = build_xy(vmd_feats_all)
    xtr_cee, ytr_cee, xte_cee, yte_cee, sy_cee = build_xy(ceemdan_feats_all)

    # Objective subsets for fast metaheuristic search.
    tr_obj = 3000
    va_obj = 800
    xtr_vmd_obj = xtr_vmd[:tr_obj]
    ytr_vmd_obj = ytr_vmd[:tr_obj]
    xva_vmd_obj = xtr_vmd[tr_obj : tr_obj + va_obj]
    yva_vmd_obj = ytr_vmd[tr_obj : tr_obj + va_obj]
    xtr_cee_obj = xtr_cee[:tr_obj]
    ytr_cee_obj = ytr_cee[:tr_obj]
    xva_cee_obj = xtr_cee[tr_obj : tr_obj + va_obj]
    yva_cee_obj = ytr_cee[tr_obj : tr_obj + va_obj]

    search_bounds = [
        (16, 96),      # hidden_dim
        (1, 2.2),      # num_layers
        (0.0, 0.35),   # dropout
        (1e-4, 3e-3),  # lr
        (1e-7, 1e-3),  # wd
    ]

    trace_rows = []

    def vmd_obj(params: np.ndarray) -> float:
        cfg = decode_cfg(params)
        model = LSTMForecaster(
            input_dim=xtr_vmd_obj.shape[-1],
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            pred_len=pred_len,
            dropout=cfg.dropout,
        )
        _, loss = train_model(
            model,
            xtr_vmd_obj,
            ytr_vmd_obj,
            xva_vmd_obj,
            yva_vmd_obj,
            cfg,
            device,
            epochs=2,
            batch_size=128,
            patience=1,
        )
        return float(loss)

    def cee_obj(params: np.ndarray) -> float:
        cfg = decode_cfg(params)
        model = GRUForecaster(
            input_dim=xtr_cee_obj.shape[-1],
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            pred_len=pred_len,
            dropout=cfg.dropout,
        )
        _, loss = train_model(
            model,
            xtr_cee_obj,
            ytr_cee_obj,
            xva_cee_obj,
            yva_cee_obj,
            cfg,
            device,
            epochs=2,
            batch_size=128,
            patience=1,
        )
        return float(loss)

    print("[info] PSO search for VMD-PSO-LSTM ...")
    best_vmd, best_vmd_score, pso_trace = pso_optimize(vmd_obj, search_bounds, pop_size=5, iters=5, seed=42)
    for row in pso_trace:
        trace_rows.append(
            {
                "model": "VMD-PSO-LSTM",
                "iter": row["iter"],
                "best_score": row["best_score"],
                "best_params": json.dumps(row["best_params"]),
            }
        )

    print("[info] WOA search for CEEMDAN-WOA-GRU ...")
    best_cee, best_cee_score, woa_trace = woa_optimize(cee_obj, search_bounds, pop_size=5, iters=5, seed=42)
    for row in woa_trace:
        trace_rows.append(
            {
                "model": "CEEMDAN-WOA-GRU",
                "iter": row["iter"],
                "best_score": row["best_score"],
                "best_params": json.dumps(row["best_params"]),
            }
        )

    best_cfg_vmd = decode_cfg(best_vmd)
    best_cfg_cee = decode_cfg(best_cee)

    # Final training with a larger but still lightweight subset.
    tr_final = min(12000, xtr_vmd.shape[0] - 1000)
    xtr_vmd_f = xtr_vmd[:tr_final]
    ytr_vmd_f = ytr_vmd[:tr_final]
    xva_vmd_f = xtr_vmd[tr_final : tr_final + 1000]
    yva_vmd_f = ytr_vmd[tr_final : tr_final + 1000]
    xtr_cee_f = xtr_cee[:tr_final]
    ytr_cee_f = ytr_cee[:tr_final]
    xva_cee_f = xtr_cee[tr_final : tr_final + 1000]
    yva_cee_f = ytr_cee[tr_final : tr_final + 1000]

    print("[info] final training VMD-PSO-LSTM ...")
    t0 = time.time()
    model_vmd = LSTMForecaster(
        input_dim=xtr_vmd.shape[-1],
        hidden_dim=best_cfg_vmd.hidden_dim,
        num_layers=best_cfg_vmd.num_layers,
        pred_len=pred_len,
        dropout=best_cfg_vmd.dropout,
    )
    model_vmd, _ = train_model(
        model_vmd,
        xtr_vmd_f,
        ytr_vmd_f,
        xva_vmd_f,
        yva_vmd_f,
        best_cfg_vmd,
        device,
        epochs=5,
        batch_size=128,
        patience=2,
    )
    train_time_vmd = time.time() - t0

    print("[info] final training CEEMDAN-WOA-GRU ...")
    t0 = time.time()
    model_cee = GRUForecaster(
        input_dim=xtr_cee.shape[-1],
        hidden_dim=best_cfg_cee.hidden_dim,
        num_layers=best_cfg_cee.num_layers,
        pred_len=pred_len,
        dropout=best_cfg_cee.dropout,
    )
    model_cee, _ = train_model(
        model_cee,
        xtr_cee_f,
        ytr_cee_f,
        xva_cee_f,
        yva_cee_f,
        best_cfg_cee,
        device,
        epochs=5,
        batch_size=128,
        patience=2,
    )
    train_time_cee = time.time() - t0

    # Clean metrics.
    pred_vmd_s = infer(model_vmd, xte_vmd, device)
    pred_cee_s = infer(model_cee, xte_cee, device)
    true_vmd = sy_vmd.inverse_transform(yte_vmd.reshape(-1, 1)).reshape(yte_vmd.shape)
    pred_vmd = sy_vmd.inverse_transform(pred_vmd_s.reshape(-1, 1)).reshape(pred_vmd_s.shape)
    true_cee = sy_cee.inverse_transform(yte_cee.reshape(-1, 1)).reshape(yte_cee.shape)
    pred_cee = sy_cee.inverse_transform(pred_cee_s.reshape(-1, 1)).reshape(pred_cee_s.shape)
    clean_vmd = metric_dict(true_vmd, pred_vmd)
    clean_cee = metric_dict(true_cee, pred_cee)

    # Robustness metrics.
    xte_vmd_g = inject_gaussian(xte_vmd, sigma_eff=0.6, seed=42)
    xte_cee_g = inject_gaussian(xte_cee, sigma_eff=0.6, seed=42)
    xte_vmd_s = inject_spike(xte_vmd, severity=0.6, seed=42)
    xte_cee_s = inject_spike(xte_cee, severity=0.6, seed=42)

    pred_vmd_g = sy_vmd.inverse_transform(infer(model_vmd, xte_vmd_g, device).reshape(-1, 1)).reshape(yte_vmd.shape)
    pred_cee_g = sy_cee.inverse_transform(infer(model_cee, xte_cee_g, device).reshape(-1, 1)).reshape(yte_cee.shape)
    pred_vmd_sp = sy_vmd.inverse_transform(infer(model_vmd, xte_vmd_s, device).reshape(-1, 1)).reshape(yte_vmd.shape)
    pred_cee_sp = sy_cee.inverse_transform(infer(model_cee, xte_cee_s, device).reshape(-1, 1)).reshape(yte_cee.shape)

    rob_vmd_g = metric_dict(true_vmd, pred_vmd_g)
    rob_cee_g = metric_dict(true_cee, pred_cee_g)
    rob_vmd_sp = metric_dict(true_vmd, pred_vmd_sp)
    rob_cee_sp = metric_dict(true_cee, pred_cee_sp)

    # Latency.
    cpu_device = torch.device("cpu")
    model_vmd_cpu = copy.deepcopy(model_vmd).to(cpu_device)
    model_cee_cpu = copy.deepcopy(model_cee).to(cpu_device)
    lat_vmd = measure_latency_ms(model_vmd_cpu, xte_vmd.astype(np.float32), cpu_device)
    lat_cee = measure_latency_ms(model_cee_cpu, xte_cee.astype(np.float32), cpu_device)

    clean_rows = [
        {
            "dataset": "wind_cn",
            "model": "VMD-PSO-LSTM",
            "R2": clean_vmd["R2"],
            "RMSE": clean_vmd["RMSE"],
            "MAE": clean_vmd["MAE"],
            "train_time_s": train_time_vmd,
        },
        {
            "dataset": "wind_cn",
            "model": "CEEMDAN-WOA-GRU",
            "R2": clean_cee["R2"],
            "RMSE": clean_cee["RMSE"],
            "MAE": clean_cee["MAE"],
            "train_time_s": train_time_cee,
        },
    ]

    robust_rows = [
        {
            "dataset": "wind_cn",
            "model": "VMD-PSO-LSTM",
            "fault_type": "gaussian",
            "severity": 0.6,
            "sigma_eff": 0.6,
            "R2": rob_vmd_g["R2"],
            "relative_drop_pct": 100.0 * (clean_vmd["R2"] - rob_vmd_g["R2"]) / max(abs(clean_vmd["R2"]), 1e-8),
        },
        {
            "dataset": "wind_cn",
            "model": "CEEMDAN-WOA-GRU",
            "fault_type": "gaussian",
            "severity": 0.6,
            "sigma_eff": 0.6,
            "R2": rob_cee_g["R2"],
            "relative_drop_pct": 100.0 * (clean_cee["R2"] - rob_cee_g["R2"]) / max(abs(clean_cee["R2"]), 1e-8),
        },
        {
            "dataset": "wind_cn",
            "model": "VMD-PSO-LSTM",
            "fault_type": "spike",
            "severity": 0.6,
            "sigma_eff": np.nan,
            "R2": rob_vmd_sp["R2"],
            "relative_drop_pct": 100.0 * (clean_vmd["R2"] - rob_vmd_sp["R2"]) / max(abs(clean_vmd["R2"]), 1e-8),
        },
        {
            "dataset": "wind_cn",
            "model": "CEEMDAN-WOA-GRU",
            "fault_type": "spike",
            "severity": 0.6,
            "sigma_eff": np.nan,
            "R2": rob_cee_sp["R2"],
            "relative_drop_pct": 100.0 * (clean_cee["R2"] - rob_cee_sp["R2"]) / max(abs(clean_cee["R2"]), 1e-8),
        },
    ]

    latency_rows = [
        {
            "dataset": "wind_cn",
            "model": "VMD-PSO-LSTM",
            "latency_mean_ms": lat_vmd["latency_mean_ms"],
            "latency_std_ms": lat_vmd["latency_std_ms"],
            "latency_p95_ms": lat_vmd["latency_p95_ms"],
            "active_params_m": sum(p.numel() for p in model_vmd.parameters()) / 1e6,
            "model_size_mib": state_size_mb(model_vmd),
        },
        {
            "dataset": "wind_cn",
            "model": "CEEMDAN-WOA-GRU",
            "latency_mean_ms": lat_cee["latency_mean_ms"],
            "latency_std_ms": lat_cee["latency_std_ms"],
            "latency_p95_ms": lat_cee["latency_p95_ms"],
            "active_params_m": sum(p.numel() for p in model_cee.parameters()) / 1e6,
            "model_size_mib": state_size_mb(model_cee),
        },
    ]

    clean_csv = os.path.join(OUT_DIR, "minimal_vmd_ceemdan_clean_windcn.csv")
    robust_csv = os.path.join(OUT_DIR, "minimal_vmd_ceemdan_robustness_windcn.csv")
    latency_csv = os.path.join(OUT_DIR, "minimal_vmd_ceemdan_latency_windcn.csv")
    trace_csv = os.path.join(OUT_DIR, "minimal_vmd_ceemdan_search_trace.csv")
    meta_json = os.path.join(OUT_DIR, "minimal_vmd_ceemdan_meta.json")

    pd.DataFrame(clean_rows).to_csv(clean_csv, index=False)
    pd.DataFrame(robust_rows).to_csv(robust_csv, index=False)
    pd.DataFrame(latency_rows).to_csv(latency_csv, index=False)
    pd.DataFrame(trace_rows).to_csv(trace_csv, index=False)

    meta = {
        "script": "run_minimal_vmd_pso_ceemdan_woa.py",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset": "wind_final.csv",
        "split": "80_20_chronological",
        "window": {"seq_len": seq_len, "pred_len": pred_len},
        "decomposition": {
            "vmd": {"k_modes": 3, "downsample_factor": 8, "alpha": 2000.0},
            "ceemdan_like": {"n_imfs": 3, "trials": 4, "noise_width": 0.15, "downsample_factor": 8},
        },
        "search": {
            "pso": {"population": 5, "iterations": 5, "best_score": best_vmd_score, "best_cfg": vars(best_cfg_vmd)},
            "woa": {"population": 5, "iterations": 5, "best_score": best_cee_score, "best_cfg": vars(best_cfg_cee)},
        },
        "outputs": {
            "clean_csv": clean_csv,
            "robust_csv": robust_csv,
            "latency_csv": latency_csv,
            "trace_csv": trace_csv,
        },
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[done] outputs:")
    print(f"- {clean_csv}")
    print(f"- {robust_csv}")
    print(f"- {latency_csv}")
    print(f"- {trace_csv}")
    print(f"- {meta_json}")


if __name__ == "__main__":
    main()
