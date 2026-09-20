from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ae_plot_style import save_ae, set_ae_style


def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_metrics_json(path: str) -> pd.DataFrame:
    """
    Load model metrics from JSON and normalize metric names to:
    ["R^2", "RMSE", "MAE"].
    Supports both uppercase and lowercase keys, e.g. R2/r2, RMSE/rmse.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON at {path}, got {type(obj)}")

    df = pd.DataFrame(obj).T
    rename_map = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in {"r2", "r^2"}:
            rename_map[c] = "R^2"
        elif lc == "rmse":
            rename_map[c] = "RMSE"
        elif lc == "mae":
            rename_map[c] = "MAE"
    df = df.rename(columns=rename_map)

    need = ["R^2", "RMSE", "MAE"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"{path} missing required metrics columns: {miss}")

    out = df[need].copy()
    out["R^2"] = _to_float_series(out["R^2"])
    out["RMSE"] = _to_float_series(out["RMSE"])
    out["MAE"] = _to_float_series(out["MAE"])
    out = out.dropna(subset=need)
    return out


def fig3_performance_bars(df: pd.DataFrame, model_order: list[str] | None = None) -> plt.Figure:
    set_ae_style()
    if model_order is not None:
        model_order = [m for m in model_order if m in df.index]
        df = df.loc[model_order]

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.6), constrained_layout=True)
    metrics = ["R^2", "RMSE", "MAE"]

    for ax, metric in zip(axes, metrics, strict=True):
        vals = df[metric].to_numpy(dtype=float)
        colors = ["#d95f02" if m == "Ultra-LSNT" else "#4c78a8" for m in df.index]
        ax.bar(df.index, vals, color=colors, alpha=0.9)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        if metric == "R^2":
            ax.set_ylim(max(0.0, np.nanmin(vals) - 0.05), min(1.0, np.nanmax(vals) + 0.05))
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}" if metric == "R^2" else f"{v:.0f}", ha="center", va="bottom", fontsize=6)

    return fig


def load_robustness_curves(
    gbdt_json: str = "gbdt_results_complete.json",
    dlinear_json: str = "dlinear_noise_robustness_results.json",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Build robustness curves from real experiment files.
    Returns:
    - noise_levels_pct: e.g., [0, 10, 20, 30, 40]
    - curves: dict model -> R^2 array
    """
    with open(gbdt_json, "r", encoding="utf-8") as f:
        gbdt_obj = json.load(f)

    nl = np.asarray(gbdt_obj["noise_levels"], dtype=float)
    curves: dict[str, np.ndarray] = {
        "Ultra-LSNT": np.asarray(gbdt_obj["ultra_lsnt_scores"], dtype=float),
        "LightGBM": np.asarray(gbdt_obj["lightgbm_scores"], dtype=float),
    }

    dlinear_path = Path(dlinear_json)
    if dlinear_path.exists():
        with open(dlinear_path, "r", encoding="utf-8") as f:
            d_obj = json.load(f)
        d_nl = np.asarray(d_obj["noise_levels"], dtype=float)
        d_r2 = np.asarray(d_obj["r2_scores"], dtype=float)
        if np.allclose(d_nl, nl):
            curves["DLinear"] = d_r2

    return nl * 100.0, curves


def fig4_robustness_curve(noise_levels_pct: np.ndarray, curves: dict[str, np.ndarray]) -> plt.Figure:
    set_ae_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.0), constrained_layout=True)
    for name, y in curves.items():
        ax.plot(noise_levels_pct, y, marker="o", label=name)
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel("$R^2$")
    ax.set_xticks(noise_levels_pct)
    ax.legend(frameon=False, ncol=1, fontsize=7)
    ax.grid(alpha=0.25)
    return fig


def _load_r2_lookup(
    full_json: str = "checkpoints_ts/full_sota_comparison.json",
    metrics_json: str = "results/metrics_corrected.json",
) -> dict[str, float]:
    out: dict[str, float] = {}
    if Path(full_json).exists():
        with open(full_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    if "R2" in v:
                        out[k] = float(v["R2"])
                    elif "r2" in v:
                        out[k] = float(v["r2"])
    if Path(metrics_json).exists():
        with open(metrics_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    if "R2" in v:
                        out[k] = float(v["R2"])
                    elif "r2" in v:
                        out[k] = float(v["r2"])
    return out


def build_efficiency_df() -> pd.DataFrame:
    """
    Build efficiency-accuracy table from real experiment CSV files.
    Output columns:
    ["Model", "R^2", "Avg_latency_ms", "Energy_mJ", "Size_MiB"].
    """
    r2_lookup = _load_r2_lookup()
    rows: list[dict[str, float | str]] = []

    def add_row(model: str, latency_ms: float, size_mib: float, energy_mj: float | None, r2: float | None) -> None:
        rows.append(
            {
                "Model": model,
                "R^2": np.nan if r2 is None else float(r2),
                "Avg_latency_ms": float(latency_ms),
                "Energy_mJ": np.nan if energy_mj is None else float(energy_mj),
                "Size_MiB": float(size_mib),
            }
        )

    p = Path("dlinear_latency_results.csv")
    if p.exists():
        d = pd.read_csv(p).iloc[0]
        add_row(
            model="DLinear",
            latency_ms=float(d["avg_inference_time_ms"]),
            size_mib=float(d["model_size_mb"]),
            energy_mj=float(d["raspberry_pi_energy_mj"]) if "raspberry_pi_energy_mj" in d else None,
            r2=r2_lookup.get("DLinear"),
        )

    p = Path("lightgbm_latency_results.csv")
    if p.exists():
        d = pd.read_csv(p).iloc[0]
        add_row(
            model="LightGBM",
            latency_ms=float(d["avg_inference_time_ms"]),
            size_mib=float(d["model_size_mb"]),
            energy_mj=float(d["raspberry_pi_energy_mj"]) if "raspberry_pi_energy_mj" in d else None,
            r2=r2_lookup.get("LightGBM"),
        )

    p = Path("timemixer_efficiency_results.csv")
    if p.exists():
        d = pd.read_csv(p).iloc[0]
        add_row(
            model="TimeMixer",
            latency_ms=float(d["avg_inference_time_ms"]),
            size_mib=float(d["model_size_mb"]),
            energy_mj=None,
            r2=r2_lookup.get("TimeMixer"),
        )

    p = Path("ultra_lsnt_lite_optimized_results.csv")
    if p.exists():
        d = pd.read_csv(p).iloc[0]
        r2_lite = None
        perf_path = Path("ultra_lsnt_lite_optimized_performance.csv")
        if perf_path.exists():
            perf = pd.read_csv(perf_path).iloc[0]
            r2_lite = float(perf["R2"])
        add_row(
            model="Ultra-LSNT-Lite-Optimized",
            latency_ms=float(d["avg_inference_time_ms"]),
            size_mib=float(d["model_size_mb"]),
            energy_mj=float(d["estimated_energy_per_inference_mj"])
            if "estimated_energy_per_inference_mj" in d
            else None,
            r2=r2_lite,
        )

    p = Path("efficiency_benchmark_results_enhanced.csv")
    if p.exists():
        d = pd.read_csv(p).iloc[0]
        model_size = float(d["model_parameters"]) * 4.0 / (1024.0 * 1024.0)
        add_row(
            model="Ultra-LSNT",
            latency_ms=float(d["avg_inference_time_ms"]),
            size_mib=model_size,
            energy_mj=None,
            r2=r2_lookup.get("Ultra-LSNT"),
        )

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["R^2", "Avg_latency_ms", "Size_MiB"], how="any")
    return out


def fig_efficiency_tradeoff(df: pd.DataFrame) -> plt.Figure:
    """
    df columns expected: ["Model", "R^2", "Avg_latency_ms", "Energy_mJ", "Size_MiB"]
    """
    set_ae_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.0), constrained_layout=True)

    sizes = np.clip(df["Size_MiB"].to_numpy(dtype=float) * 8.0, 35.0, 260.0)
    ax.scatter(df["Avg_latency_ms"], df["R^2"], s=sizes, alpha=0.8)
    for _, r in df.iterrows():
        ax.text(float(r["Avg_latency_ms"]), float(r["R^2"]), str(r["Model"]), fontsize=6, ha="left", va="bottom")

    ax.set_xlabel("Avg. latency (ms)")
    ax.set_ylabel("$R^2$")
    ax.grid(alpha=0.25)
    return fig


def load_expert_hour_matrix(path: str = "wind_expert_activation.csv") -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load expert activation profile (real run output) and return:
    - matrix [n_experts, n_hours]
    - expert names
    - hour labels
    """
    df = pd.read_csv(path)
    expert_cols = [c for c in df.columns if c.lower().startswith("expert_")]
    if not expert_cols or "Hour" not in df.columns:
        raise ValueError(f"{path} missing expected columns. got {list(df.columns)}")

    df = df.sort_values("Hour")
    mat = df[expert_cols].to_numpy(dtype=float).T
    expert_names = [c.replace("_", " ") for c in expert_cols]
    feature_names = [str(int(h)) for h in df["Hour"].to_numpy()]
    return mat, expert_names, feature_names


def fig_expert_heatmap(expert_feature: np.ndarray, expert_names: list[str], feature_names: list[str]) -> plt.Figure:
    """
    expert_feature: shape [n_experts, n_features], e.g. activation/correlation matrix.
    """
    set_ae_style()
    fig, ax = plt.subplots(figsize=(6.0, 2.6), constrained_layout=True)
    im = ax.imshow(expert_feature, aspect="auto", origin="lower")
    ax.set_yticks(range(len(expert_names)))
    ax.set_yticklabels(expert_names)
    tick_ids = np.arange(0, len(feature_names), max(1, len(feature_names) // 8))
    ax.set_xticks(tick_ids)
    ax.set_xticklabels([feature_names[i] for i in tick_ids], rotation=0)
    ax.set_xlabel("Hour of day")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return fig


def main() -> None:
    print("Generating figures from real experiment tables...")

    metrics_path = "checkpoints_ts/full_sota_comparison.json"
    df_metrics = load_metrics_json(metrics_path)
    order = [
        "MLP",
        "LSTM",
        "Transformer",
        "Autoformer",
        "Informer",
        "PatchTST",
        "TimeMixer",
        "iTransformer",
        "DLinear",
        "Ultra-LSNT",
    ]
    fig = fig3_performance_bars(df_metrics, model_order=order)
    save_ae(fig, "Figure_3_FromTables_real", out_dir="figures_out")
    plt.close(fig)
    print("Saved Figure_3_FromTables_real")

    noise_levels_pct, curves = load_robustness_curves()
    fig = fig4_robustness_curve(noise_levels_pct, curves)
    save_ae(fig, "Figure_4_FromTables_real", out_dir="figures_out")
    plt.close(fig)
    print("Saved Figure_4_FromTables_real")

    df_eff = build_efficiency_df()
    if not df_eff.empty:
        fig = fig_efficiency_tradeoff(df_eff)
        save_ae(fig, "Figure_Efficiency_FromTables_real", out_dir="figures_out")
        plt.close(fig)
        print("Saved Figure_Efficiency_FromTables_real")
        print(df_eff.to_string(index=False))
    else:
        print("Skip efficiency figure: no valid rows loaded")

    mat, experts, hours = load_expert_hour_matrix("wind_expert_activation.csv")
    fig = fig_expert_heatmap(mat, experts, hours)
    save_ae(fig, "Figure_ExpertHeatmap_FromTables_real", out_dir="figures_out")
    plt.close(fig)
    print("Saved Figure_ExpertHeatmap_FromTables_real")


if __name__ == "__main__":
    main()

