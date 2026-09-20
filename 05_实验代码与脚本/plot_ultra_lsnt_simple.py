#!/usr/bin/env python3
"""
Figure 3 generator: main performance comparison for wind power point forecasting.

This script uses real local experiment outputs:
- Ultra-LSNT / DLinear / Informer from result JSON files
- LightGBM from real prediction arrays (preds_lgbm.npy, trues_lgbm.npy)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "figure"
MODEL_ORDER = ["Ultra-LSNT", "DLinear", "Informer", "LightGBM"]
METRIC_ORDER = ["R2", "RMSE", "MAE"]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_metrics_from_arrays(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    pred_flat = pred.reshape(-1).astype(np.float64)
    true_flat = true.reshape(-1).astype(np.float64)

    mse = float(np.mean((pred_flat - true_flat) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred_flat - true_flat)))
    ss_res = float(np.sum((true_flat - pred_flat) ** 2))
    ss_tot = float(np.sum((true_flat - np.mean(true_flat)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def _find_latest_file(pattern: str) -> Path:
    candidates = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return candidates[-1]


def load_real_metrics() -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    metrics: Dict[str, Dict[str, float]] = {}
    sources: Dict[str, str] = {}

    full_sota_path = ROOT / "checkpoints_ts" / "full_sota_comparison.json"
    if full_sota_path.exists():
        full_sota = _read_json(full_sota_path)
        for name in ("Ultra-LSNT", "DLinear", "Informer"):
            if name not in full_sota:
                raise KeyError(f"{name} is missing in {full_sota_path}")
            metrics[name] = {
                "R2": float(full_sota[name]["R2"]),
                "RMSE": float(full_sota[name]["RMSE"]),
                "MAE": float(full_sota[name]["MAE"]),
            }
            sources[name] = str(full_sota_path.relative_to(ROOT))
    else:
        ultra_path = ROOT / "checkpoints_ts" / "main" / "final_results.json"
        ultra = _read_json(ultra_path)["test_metrics"]
        metrics["Ultra-LSNT"] = {
            "R2": float(ultra["R2"]),
            "RMSE": float(ultra["RMSE"]),
            "MAE": float(ultra["MAE"]),
        }
        sources["Ultra-LSNT"] = str(ultra_path.relative_to(ROOT))

        dlinear_path = _find_latest_file("dlinear_results/dlinear_results_*.json")
        dlinear = _read_json(dlinear_path)["metrics"]
        metrics["DLinear"] = {
            "R2": float(dlinear["R2"]),
            "RMSE": float(dlinear["RMSE"]),
            "MAE": float(dlinear["MAE"]),
        }
        sources["DLinear"] = str(dlinear_path.relative_to(ROOT))

        informer_path = ROOT / "checkpoints_ts" / "Informer_results.json"
        informer = _read_json(informer_path)
        metrics["Informer"] = {
            "R2": float(informer["R2"]),
            "RMSE": float(informer["RMSE"]),
            "MAE": float(informer["MAE"]),
        }
        sources["Informer"] = str(informer_path.relative_to(ROOT))

    pred_path = ROOT / "preds_lgbm.npy"
    true_path = ROOT / "trues_lgbm.npy"
    if not pred_path.exists() or not true_path.exists():
        raise FileNotFoundError("Missing LightGBM arrays: preds_lgbm.npy or trues_lgbm.npy")
    pred_lgbm = np.load(pred_path)
    true_lgbm = np.load(true_path)
    metrics["LightGBM"] = _compute_metrics_from_arrays(pred_lgbm, true_lgbm)
    sources["LightGBM"] = f"{pred_path.relative_to(ROOT)} + {true_path.relative_to(ROOT)}"

    for model in MODEL_ORDER:
        if model not in metrics:
            raise KeyError(f"Missing model metrics: {model}")

    return metrics, sources


def normalize_scores(raw_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {m: {} for m in raw_metrics}

    for metric in METRIC_ORDER:
        values = np.array([raw_metrics[m][metric] for m in MODEL_ORDER], dtype=np.float64)
        v_min = float(values.min())
        v_max = float(values.max())
        span = v_max - v_min

        if span < 1e-12:
            norm = np.ones_like(values)
        elif metric == "R2":
            norm = (values - v_min) / span
        else:
            norm = (v_max - values) / span

        for i, model in enumerate(MODEL_ORDER):
            scores[model][metric] = float(norm[i])

    return scores


def _format_raw_value(metric: str, value: float) -> str:
    if metric == "R2":
        return f"{value:.3f}"
    return f"{value:,.0f}"


def plot_grouped_bar(
    raw_metrics: Dict[str, Dict[str, float]],
    score_metrics: Dict[str, Dict[str, float]],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "Ultra-LSNT": "#D62828",
        "DLinear": "#457B9D",
        "Informer": "#2A9D8F",
        "LightGBM": "#8D99AE",
    }
    hatches = {"Ultra-LSNT": "///", "DLinear": "", "Informer": "", "LightGBM": ""}

    x = np.arange(len(METRIC_ORDER), dtype=np.float64)
    width = 0.18

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)

    for idx, model in enumerate(MODEL_ORDER):
        scores = [score_metrics[model][metric] for metric in METRIC_ORDER]
        centers = x + (idx - (len(MODEL_ORDER) - 1) / 2.0) * width

        bars = ax.bar(
            centers,
            scores,
            width=width,
            color=colors[model],
            label=model,
            alpha=0.9 if model == "Ultra-LSNT" else 0.8,
            edgecolor="black",
            linewidth=1.0,
            hatch=hatches[model],
            zorder=3,
        )

        for j, bar in enumerate(bars):
            metric = METRIC_ORDER[j]
            raw_v = raw_metrics[model][metric]
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.02,
                _format_raw_value(metric, raw_v),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([r"$R^2$", "RMSE", "MAE"], fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Normalized score (higher is better)", fontsize=11)
    ax.set_title(
        "Figure 3. Main Performance Comparison (Wind Power Point Forecasting)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.legend(loc="upper center", ncol=4, frameon=True, fontsize=10)
    ax.text(
        0.5,
        -0.14,
        "Note: R^2 higher is better. RMSE/MAE are converted to higher-better scores; labels show real values.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    fig.tight_layout()
    png_path = out_dir / "Figure3_Main_Performance_GroupedBar.png"
    pdf_path = out_dir / "Figure3_Main_Performance_GroupedBar.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_radar(
    raw_metrics: Dict[str, Dict[str, float]],
    score_metrics: Dict[str, Dict[str, float]],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [r"$R^2$", "RMSE", "MAE"]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    colors = {
        "Ultra-LSNT": "#D62828",
        "DLinear": "#457B9D",
        "Informer": "#2A9D8F",
        "LightGBM": "#8D99AE",
    }

    fig = plt.figure(figsize=(8.2, 7.2), dpi=300)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.grid(alpha=0.35)

    for model in MODEL_ORDER:
        values = [score_metrics[model][m] for m in METRIC_ORDER]
        values += values[:1]
        ax.plot(
            angles,
            values,
            color=colors[model],
            linewidth=2.6 if model == "Ultra-LSNT" else 1.9,
            linestyle="-" if model == "Ultra-LSNT" else "--",
            label=model,
        )
        ax.fill(
            angles,
            values,
            color=colors[model],
            alpha=0.16 if model == "Ultra-LSNT" else 0.08,
        )

    ax.set_title(
        "Figure 3 (Alternative). Radar Comparison with Unified Higher-Better Scoring",
        y=1.09,
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), frameon=True, fontsize=9)

    summary = (
        f"Raw values | Ultra-LSNT: R2={raw_metrics['Ultra-LSNT']['R2']:.3f}, "
        f"RMSE={raw_metrics['Ultra-LSNT']['RMSE']:.0f}, MAE={raw_metrics['Ultra-LSNT']['MAE']:.0f}"
    )
    fig.text(0.5, 0.03, summary, ha="center", fontsize=8.8)

    fig.tight_layout()
    png_path = out_dir / "Figure3_Main_Performance_Radar.png"
    pdf_path = out_dir / "Figure3_Main_Performance_Radar.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def save_metrics_record(
    raw_metrics: Dict[str, Dict[str, float]],
    normalized_scores: Dict[str, Dict[str, float]],
    sources: Dict[str, str],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Figure3_real_metrics_record.json"
    payload = {
        "dataset": "wind_final.csv",
        "models": MODEL_ORDER,
        "metrics_raw": raw_metrics,
        "metrics_normalized_higher_better": normalized_scores,
        "sources": sources,
        "notes": {
            "R2": "higher is better",
            "RMSE": "lower is better",
            "MAE": "lower is better",
            "normalization": "R2: min-max, RMSE/MAE: reversed min-max",
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 3 main model performance plots.")
    parser.add_argument(
        "--chart",
        choices=["bar", "radar", "both"],
        default="both",
        help="Choose which plot type to generate.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for figures and metric record.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_metrics, sources = load_real_metrics()
    score_metrics = normalize_scores(raw_metrics)
    record_path = save_metrics_record(raw_metrics, score_metrics, sources, out_dir)

    generated = []
    if args.chart in ("bar", "both"):
        generated.append(plot_grouped_bar(raw_metrics, score_metrics, out_dir))
    if args.chart in ("radar", "both"):
        generated.append(plot_radar(raw_metrics, score_metrics, out_dir))

    print("Real metrics loaded:")
    for model in MODEL_ORDER:
        m = raw_metrics[model]
        print(f"  {model:10s} | R2={m['R2']:.6f} | RMSE={m['RMSE']:.4f} | MAE={m['MAE']:.4f}")
    print(f"Metric record saved: {record_path}")
    for path in generated:
        print(f"Figure saved: {path}")


if __name__ == "__main__":
    main()
