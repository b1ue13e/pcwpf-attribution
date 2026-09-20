#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_safety_latency(table_path: Path, out_path: Path) -> None:
    df = pd.read_csv(table_path)
    if df.empty:
        raise SystemExit("table_safety_coupling.csv is empty")

    severe = df[df["corruption_type"].str.lower().str.contains("severe")].copy()
    if severe.empty:
        severe = df.copy()

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = {
        "Ultra-LSNT Raw Output": "#B22222",
        "Ultra-LSNT + Clip-only": "#1F77B4",
        "Ultra-LSNT + FG-MoE": "#2E8B57",
    }
    for _, row in severe.iterrows():
        label = row["model_name"]
        ax.scatter(
            row["p95_latency_ms"],
            row["exceedance_rate"],
            s=80,
            color=colors.get(label, "#444444"),
            label=label,
        )
        ax.annotate(
            label.replace("Ultra-LSNT + ", ""),
            (row["p95_latency_ms"], row["exceedance_rate"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    ax.set_xlabel("P95 latency (ms)")
    ax.set_ylabel("Rated-power exceedance (%)")
    ax.set_title("Safety-latency trade-off under severe Gaussian corruption")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate deployment evidence figures from CSV tables.")
    ap.add_argument("--outdir", type=Path, default=Path("results") / "deployment_evidence")
    args = ap.parse_args()

    table_path = args.outdir / "table_safety_coupling.csv"
    out_path = args.outdir / "fig3_safety_latency_tradeoff.pdf"
    plot_safety_latency(table_path, out_path)
    print(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
