from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ae_plot_style import save_ae, set_ae_style


def _normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {c.lower(): c for c in df.columns}
    sigma_col = col_map.get("sigma")
    model_col = col_map.get("model")
    r2_col = col_map.get("r2")

    if sigma_col is None or model_col is None or r2_col is None:
        raise ValueError("Input data must contain columns: sigma, model, R2")

    out = df[[sigma_col, model_col, r2_col]].copy()
    out.columns = ["sigma", "model", "R2"]
    out["sigma"] = pd.to_numeric(out["sigma"], errors="coerce")
    out["R2"] = pd.to_numeric(out["R2"], errors="coerce")
    out = out.dropna(subset=["sigma", "model", "R2"])

    # Accept both fraction (0.20) and percent (20) input.
    if float(out["sigma"].max()) > 1.0:
        out["sigma"] = out["sigma"] / 100.0

    return out.sort_values(["model", "sigma"]).reset_index(drop=True)


def _append_curve(
    rows: list[dict[str, float | str]],
    *,
    model: str,
    sigma: np.ndarray,
    r2: np.ndarray,
    max_sigma: float,
) -> None:
    keep = sigma <= max_sigma + 1e-12
    sigma_keep = sigma[keep]
    r2_keep = r2[keep]
    for s, v in zip(sigma_keep, r2_keep, strict=True):
        rows.append({"sigma": float(s), "model": model, "R2": float(v)})


def _build_from_experiment_files(max_sigma: float = 0.40) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    # Source A: GBDT comparison (Ultra-LSNT + LightGBM), typically sigma=[0,0.1,0.2,0.3,0.4]
    gbdt_path = Path("gbdt_results_complete.json")
    if gbdt_path.exists():
        obj = json.loads(gbdt_path.read_text(encoding="utf-8"))
        sigma_all = np.asarray(obj["noise_levels"], dtype=float)
        ultra_all = np.asarray(obj["ultra_lsnt_scores"], dtype=float)
        lgbm_all = np.asarray(obj["lightgbm_scores"], dtype=float)
        _append_curve(rows, model="Ultra-LSNT", sigma=sigma_all, r2=ultra_all, max_sigma=max_sigma)
        _append_curve(rows, model="LightGBM", sigma=sigma_all, r2=lgbm_all, max_sigma=max_sigma)

    # Source B: DLinear robustness, same Gaussian protocol (if available)
    dlinear_path = Path("dlinear_noise_robustness_results.json")
    if dlinear_path.exists():
        d_obj = json.loads(dlinear_path.read_text(encoding="utf-8"))
        sigma_d = np.asarray(d_obj["noise_levels"], dtype=float)
        r2_d = np.asarray(d_obj["r2_scores"], dtype=float)
        _append_curve(rows, model="DLinear", sigma=sigma_d, r2=r2_d, max_sigma=max_sigma)

    # Source C: fine-grained Ultra-LSNT curve at 0/5/10/15/20 (fallback only).
    if not any(str(r["model"]) == "Ultra-LSNT" for r in rows):
        npy_path = Path("robustness_scores_ultra_lsnt.npy")
        if npy_path.exists():
            r2 = np.asarray(np.load(npy_path), dtype=float).reshape(-1)
            sigma = np.asarray([0.00, 0.05, 0.10, 0.15, 0.20], dtype=float)
            if r2.size != sigma.size:
                raise ValueError(
                    f"{npy_path} expected {sigma.size} points for sigma=[0,0.05,0.10,0.15,0.20], got {r2.size}"
                )
            _append_curve(rows, model="Ultra-LSNT", sigma=sigma, r2=r2, max_sigma=max_sigma)

    if not rows:
        raise FileNotFoundError(
            "No robustness source found. Please provide results/robust_gaussian_windcn.csv "
            "or experiment files (gbdt_results_complete.json / dlinear_noise_robustness_results.json)."
        )

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["model", "sigma"], keep="last")
    return out.sort_values(["model", "sigma"]).reset_index(drop=True)


def load_robustness_df(csv_path: str = "results/robust_gaussian_windcn.csv") -> pd.DataFrame:
    auto_df = _build_from_experiment_files(max_sigma=0.40)

    p = Path(csv_path)
    if p.exists():
        user_df = _normalize_input_df(pd.read_csv(p))
        df = pd.concat([auto_df, user_df], ignore_index=True)
        # User CSV entries override auto entries at same (model, sigma).
        df = df.drop_duplicates(subset=["model", "sigma"], keep="last")
    else:
        df = auto_df

    p.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["model", "sigma"]).reset_index(drop=True)
    df.to_csv(p, index=False)
    return df


def plot_curve(
    df: pd.DataFrame,
    value_col: str,
    stem: str,
    ylabel: str,
    title: str,
) -> None:
    set_ae_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)

    for model, g in df.groupby("model"):
        g = g.sort_values("sigma")
        is_ours = model == "Ultra-LSNT"
        ax.plot(
            g["sigma"],
            g[value_col],
            marker="o",
            linewidth=1.8 if is_ours else 1.2,
            markersize=4.8 if is_ours else 4.0,
            label=model,
        )

    xt = np.sort(df["sigma"].unique())
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{int(round(v * 100))}%" for v in xt])
    ax.set_xlabel("Noise injection level $\\sigma$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best", frameon=False, ncol=1)

    save_ae(fig, stem, out_dir="figures_out", dpi_mixed=600)
    plt.close(fig)


def main() -> None:
    # Target theme:
    # Robustness under progressively stronger Gaussian noise on Wind (CN).
    df = load_robustness_df("results/robust_gaussian_windcn.csv")

    plot_curve(
        df=df,
        value_col="R2",
        stem="FIG_robust_gaussian_abs",
        ylabel="$R^2$",
        title="Robustness under progressively stronger Gaussian noise on Wind (CN)",
    )

    base = (
        df.sort_values("sigma")
        .groupby("model", as_index=False)
        .first()[["model", "R2"]]
        .rename(columns={"R2": "R2_0"})
    )
    d2 = df.merge(base, on="model", how="left")
    d2["R2_rel"] = d2["R2"] / d2["R2_0"]

    plot_curve(
        df=d2,
        value_col="R2_rel",
        stem="FIG_robust_gaussian_rel",
        ylabel=r"Relative retention ($R^2(\sigma)/R^2(0)$)",
        title="Relative degradation under Gaussian noise on Wind (CN)",
    )


if __name__ == "__main__":
    main()
