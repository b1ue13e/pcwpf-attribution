import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA as SMARIMA


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def build_anchor_indices(n: int, split: float, seq_len: int, pred_len: int) -> np.ndarray:
    start = max(int(n * split), seq_len)
    end = n - pred_len
    if end <= start:
        return np.array([], dtype=np.int64)
    return np.arange(start, end, dtype=np.int64)


def inject_gaussian(x: np.ndarray, sigma_eff: float, rng: np.random.Generator) -> np.ndarray:
    if sigma_eff <= 0:
        return x.copy()
    std = float(np.std(x) + 1e-12)
    return x + rng.normal(0.0, sigma_eff * std, size=x.shape)


def inject_spike(x: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    n = len(y)
    if n == 0:
        return y
    n_spikes = max(1, int(0.05 * n))
    idx = rng.choice(n, size=n_spikes, replace=False)
    amp = severity * (np.std(y) + 1e-12) * 3.0
    y[idx] = y[idx] + rng.choice([-1.0, 1.0], size=n_spikes) * amp
    return y


@dataclass
class ArimaAuditable:
    order: Tuple[int, int, int] = (2, 1, 1)
    candidate_orders: Tuple[Tuple[int, int, int], ...] = (
        (1, 1, 0),
        (2, 1, 0),
        (3, 1, 0),
        (2, 1, 1),
        (3, 1, 1),
    )
    trend: str = "n"
    params: np.ndarray = None
    fit_aic: float = np.inf
    nobs: int = 0
    aic_trace: Dict[str, float] = None
    _fit_res = None

    def fit(self, y_scaled: np.ndarray) -> "ArimaAuditable":
        y = np.asarray(y_scaled, dtype=np.float64)
        best_res = None
        best_order = None
        trace = {}
        for order in self.candidate_orders:
            try:
                mdl = SMARIMA(
                    y,
                    order=order,
                    trend=self.trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = mdl.fit()
                aic = float(res.aic)
                trace[str(order)] = aic
                if np.isfinite(aic) and (best_res is None or aic < float(best_res.aic)):
                    best_res = res
                    best_order = order
            except Exception:
                trace[str(order)] = float("inf")
                continue
        if best_res is None:
            raise RuntimeError("ARIMA fit failed for all candidate orders.")
        self.order = best_order
        self._fit_res = best_res
        self.params = np.asarray(best_res.params, dtype=np.float64)
        self.fit_aic = float(best_res.aic)
        self.nobs = int(best_res.nobs)
        self.aic_trace = trace
        return self

    def forecast_scaled(self, history_scaled: np.ndarray, pred_len: int) -> np.ndarray:
        if self._fit_res is None:
            raise RuntimeError("ARIMA model must be fit before forecasting.")
        h = np.asarray(history_scaled, dtype=np.float64)
        applied = self._fit_res.apply(h, refit=False)
        pred = applied.forecast(pred_len)
        return np.asarray(pred, dtype=np.float64)

    @property
    def active_params_m(self) -> float:
        if self.params is None:
            return 0.0
        return float(len(self.params) / 1_000_000.0)

    @property
    def model_size_mib(self) -> float:
        if self.params is None:
            return 0.0
        nbytes = 8 * len(self.params)
        return float(nbytes / (1024.0 * 1024.0))


def fit_svr_one_step(
    y_scaled: np.ndarray,
    seq_len: int,
    seed: int,
    max_train_samples: int = 6000,
    max_val_samples: int = 1200,
) -> Tuple[SVR, Dict[str, float]]:
    X, yy = [], []
    for i in range(seq_len, len(y_scaled)):
        X.append(y_scaled[i - seq_len : i])
        yy.append(y_scaled[i])
    X = np.asarray(X, dtype=np.float64)
    yy = np.asarray(yy, dtype=np.float64)

    n = len(X)
    if n == 0:
        raise RuntimeError("Insufficient samples to fit SVR.")
    split = max(1, int(0.9 * n))
    X_tr, y_tr = X[:split], yy[:split]
    X_va, y_va = X[split:], yy[split:]
    rng = np.random.default_rng(seed)
    if len(X_tr) > max_train_samples:
        idx = rng.choice(len(X_tr), size=max_train_samples, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    if len(X_va) > max_val_samples:
        idx = rng.choice(len(X_va), size=max_val_samples, replace=False)
        X_va, y_va = X_va[idx], y_va[idx]

    grid = [
        {"C": 1.0, "epsilon": 0.05, "gamma": "scale"},
        {"C": 2.0, "epsilon": 0.05, "gamma": "scale"},
        {"C": 5.0, "epsilon": 0.10, "gamma": "scale"},
        {"C": 10.0, "epsilon": 0.10, "gamma": "scale"},
    ]
    best = None
    best_model = None
    for p in grid:
        m = SVR(kernel="rbf", C=p["C"], epsilon=p["epsilon"], gamma=p["gamma"])
        m.fit(X_tr, y_tr)
        pred = m.predict(X_va) if len(X_va) > 0 else m.predict(X_tr[: min(200, len(X_tr))])
        true = y_va if len(X_va) > 0 else y_tr[: len(pred)]
        score = r2_score(true, pred)
        if (best is None) or (score > best):
            best = score
            best_model = m
            best_params = p
    return best_model, {"val_r2": float(best), **best_params}


def forecast_svr_scaled(model: SVR, history_scaled: np.ndarray, pred_len: int) -> np.ndarray:
    h = history_scaled.astype(np.float64).copy()
    out = []
    for _ in range(pred_len):
        y_next = float(model.predict(h.reshape(1, -1))[0])
        out.append(y_next)
        h = np.roll(h, -1)
        h[-1] = y_next
    return np.asarray(out, dtype=np.float64)


def evaluate_model(
    y_raw: np.ndarray,
    scaler: StandardScaler,
    anchors: np.ndarray,
    seq_len: int,
    pred_len: int,
    mode: str,
    model_obj,
    rng_seed: int,
    max_eval: int = 2000,
) -> Dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    if len(anchors) > max_eval:
        anchors_eval = anchors[:max_eval]
    else:
        anchors_eval = anchors

    preds = []
    trues = []
    for a in anchors_eval:
        hist_raw = y_raw[a - seq_len : a].copy()
        if mode.startswith("gaussian:"):
            sigma_eff = float(mode.split(":")[1])
            hist_raw = inject_gaussian(hist_raw, sigma_eff=sigma_eff, rng=rng)
        elif mode == "spike:0.6":
            hist_raw = inject_spike(hist_raw, severity=0.6, rng=rng)

        hist_scaled = scaler.transform(hist_raw.reshape(-1, 1)).reshape(-1)
        if isinstance(model_obj, ArimaAuditable):
            pred_scaled = model_obj.forecast_scaled(hist_scaled, pred_len)
        else:
            pred_scaled = forecast_svr_scaled(model_obj, hist_scaled, pred_len)

        pred_raw = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
        true_raw = y_raw[a : a + pred_len]
        preds.append(pred_raw)
        trues.append(true_raw)

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    return metrics(y_true, y_pred)


def profile_latency(model_name: str, model_obj, sample_history_scaled: np.ndarray, pred_len: int, repeats: int = 200) -> Dict[str, float]:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        if isinstance(model_obj, ArimaAuditable):
            _ = model_obj.forecast_scaled(sample_history_scaled, pred_len)
            params_m = model_obj.active_params_m
            model_size_mib = model_obj.model_size_mib
        else:
            _ = forecast_svr_scaled(model_obj, sample_history_scaled, pred_len)
            # Approximate param count via support vectors and dual coefficients.
            sv = getattr(model_obj, "support_vectors_", np.empty((0, sample_history_scaled.shape[0])))
            dual = getattr(model_obj, "dual_coef_", np.empty((1, 0)))
            n_params = sv.size + dual.size + 1
            params_m = float(n_params / 1_000_000.0)
            model_size_mib = float((8 * n_params) / (1024.0 * 1024.0))
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    return {
        "platform": "x86_cpu",
        "model": model_name,
        "latency_mean_ms": mean_ms,
        "latency_std_ms": float(np.std(arr)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "throughput_hz": float(1000.0 / max(mean_ms, 1e-12)),
        "active_params_m": float(params_m),
        "model_size_mib": float(model_size_mib),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run auditable ARIMA/SVR suite on Wind(CN)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_manifest", type=str, default="split_manifest_80_20_unified.json")
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=24)
    ap.add_argument("--out_dir", type=str, default=os.path.join("results", "supplementary_evidence"))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--audit_tag", type=str, default="arima_svr_auditable_suite")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.audit_tag

    df = pd.read_csv("wind_final.csv")
    y_raw = df["power"].astype(np.float64).values
    n = len(y_raw)
    split = 0.8
    n_train = int(n * split)
    y_train = y_raw[:n_train]
    scaler = StandardScaler()
    scaler.fit(y_train.reshape(-1, 1))
    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
    anchors = build_anchor_indices(n=n, split=split, seq_len=args.seq_len, pred_len=args.pred_len)

    # Fit models
    arima = ArimaAuditable().fit(y_train_scaled)
    svr, svr_fit_meta = fit_svr_one_step(y_train_scaled, seq_len=args.seq_len, seed=args.seed)

    # Clean + robustness
    modes = ["clean", "gaussian:0.15", "gaussian:0.30", "gaussian:0.60", "spike:0.6"]
    arima_rows = []
    svr_rows = []
    arima_clean_r2 = None
    svr_clean_r2 = None
    for mode in modes:
        m_ar = evaluate_model(
            y_raw=y_raw,
            scaler=scaler,
            anchors=anchors,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            mode=mode,
            model_obj=arima,
            rng_seed=args.seed,
        )
        m_sv = evaluate_model(
            y_raw=y_raw,
            scaler=scaler,
            anchors=anchors,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            mode=mode,
            model_obj=svr,
            rng_seed=args.seed + 1,
        )

        if mode == "clean":
            arima_clean_r2 = m_ar["R2"]
            svr_clean_r2 = m_sv["R2"]
        else:
            pass

        if mode == "clean":
            arima_fault, arima_sev, arima_sigma = "clean", 0.0, np.nan
            svr_fault, svr_sev, svr_sigma = "clean", 0.0, np.nan
        elif mode.startswith("gaussian:"):
            sev = float(mode.split(":")[1])
            arima_fault, arima_sev, arima_sigma = "gaussian", sev, sev
            svr_fault, svr_sev, svr_sigma = "gaussian", sev, sev
        else:
            arima_fault, arima_sev, arima_sigma = "spike", 0.6, np.nan
            svr_fault, svr_sev, svr_sigma = "spike", 0.6, np.nan

        arima_rows.append(
            {
                "fault_type": arima_fault,
                "severity": arima_sev,
                "sigma_eff": arima_sigma,
                "model": "ARIMA (auditable)",
                "R2": m_ar["R2"],
                "RMSE": m_ar["RMSE"],
                "MAE": m_ar["MAE"],
                "R2_clean": arima_clean_r2,
                "relative_drop_pct": 0.0 if mode == "clean" else float(100.0 * (arima_clean_r2 - m_ar["R2"]) / (abs(arima_clean_r2) + 1e-12)),
                "source_file": "arima_robustness_windcn.csv",
                "run_id": run_id,
            }
        )
        svr_rows.append(
            {
                "fault_type": svr_fault,
                "severity": svr_sev,
                "sigma_eff": svr_sigma,
                "model": "SVR (auditable)",
                "R2": m_sv["R2"],
                "RMSE": m_sv["RMSE"],
                "MAE": m_sv["MAE"],
                "R2_clean": svr_clean_r2,
                "relative_drop_pct": 0.0 if mode == "clean" else float(100.0 * (svr_clean_r2 - m_sv["R2"]) / (abs(svr_clean_r2) + 1e-12)),
                "source_file": "svr_robustness_windcn.csv",
                "run_id": run_id,
            }
        )

    arima_rob = pd.DataFrame(arima_rows)
    svr_rob = pd.DataFrame(svr_rows)
    arima_clean = arima_rob[arima_rob["fault_type"] == "clean"][["model", "R2", "RMSE", "MAE"]].copy()
    svr_clean = svr_rob[svr_rob["fault_type"] == "clean"][["model", "R2", "RMSE", "MAE"]].copy()
    arima_clean.insert(0, "dataset", "wind_cn")
    svr_clean.insert(0, "dataset", "wind_cn")
    arima_clean["source_file"] = "arima_clean_windcn.csv"
    svr_clean["source_file"] = "svr_clean_windcn.csv"
    arima_clean["run_id"] = run_id
    svr_clean["run_id"] = run_id

    sample_hist = scaler.transform(y_raw[max(int(0.8 * n) - args.seq_len, 0) : int(0.8 * n)].reshape(-1, 1)).reshape(-1)
    arima_lat = pd.DataFrame([profile_latency("ARIMA (auditable)", arima, sample_hist, args.pred_len)])
    svr_lat = pd.DataFrame([profile_latency("SVR (auditable)", svr, sample_hist, args.pred_len)])
    arima_lat["source_file"] = "arima_latency_windcn.csv"
    arima_lat["run_id"] = run_id
    svr_lat["source_file"] = "svr_latency_windcn.csv"
    svr_lat["run_id"] = run_id

    # Save outputs
    arima_clean.to_csv(os.path.join(args.out_dir, "arima_clean_windcn.csv"), index=False)
    svr_clean.to_csv(os.path.join(args.out_dir, "svr_clean_windcn.csv"), index=False)
    arima_rob.to_csv(os.path.join(args.out_dir, "arima_robustness_windcn.csv"), index=False)
    svr_rob.to_csv(os.path.join(args.out_dir, "svr_robustness_windcn.csv"), index=False)
    arima_lat.to_csv(os.path.join(args.out_dir, "arima_latency_windcn.csv"), index=False)
    svr_lat.to_csv(os.path.join(args.out_dir, "svr_latency_windcn.csv"), index=False)

    meta = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "wind_final.csv",
        "split_manifest": args.split_manifest,
        "split": "80_20_chronological",
        "train_ratio": 0.8,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "seed": args.seed,
        "models": {
            "ARIMA (auditable)": {
                "selected_order": list(arima.order),
                "candidate_orders": [list(x) for x in arima.candidate_orders],
                "aic_trace": arima.aic_trace,
                "fit_aic": arima.fit_aic,
                "nobs": arima.nobs,
                "active_params_m": arima.active_params_m,
                "model_size_mib": arima.model_size_mib,
            },
            "SVR (auditable)": svr_fit_meta,
        },
        "notes": "ARIMA uses statsmodels.tsa.arima.model.ARIMA with lightweight AIC order selection over a fixed candidate grid.",
    }
    with open(os.path.join(args.out_dir, "arima_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.out_dir, "svr_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("ARIMA/SVR auditable suite completed.")
    print("run_id:", run_id)


if __name__ == "__main__":
    main()
