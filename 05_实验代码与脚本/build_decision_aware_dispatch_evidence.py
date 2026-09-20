#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEGACY_MODEL_ORDER = ["DLinear", "TimeMixer", "iTransformer", "Ultra-LSNT"]
LEGACY_MAPPING_ORDER = ["affine", "isotonic", "quantile"]
DEFAULT_OPERATION_MODEL_ORDER = [
    "DLinear",
    "TimeMixer",
    "iTransformer raw",
    "Ultra-LSNT raw",
    "Ultra-LSNT + clip-only",
    "Ultra-LSNT + FG-MoE",
]

PALETTE = {
    "base_light": "#D5E1EF",
    "dlinear": "#9DC2D5",
    "timemixer": "#5A94B9",
    "itransformer": "#2A6398",
    "ultra": "#7EB87B",
    "ultra_raw": "#5FA86A",
    "ultra_clip": "#C47A5A",
    "ultra_fg": "#2F7D44",
}

MODEL_COLOR = {
    "DLinear": PALETTE["dlinear"],
    "TimeMixer": PALETTE["timemixer"],
    "iTransformer": PALETTE["itransformer"],
    "iTransformer raw": PALETTE["itransformer"],
    "iTransformer + FG-MoE": "#4C85B8",
    "Ultra-LSNT": PALETTE["ultra"],
    "Ultra-LSNT raw": PALETTE["ultra_raw"],
    "Ultra-LSNT + clip-only": PALETTE["ultra_clip"],
    "Ultra-LSNT + FG-MoE": PALETTE["ultra_fg"],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build decision-aware dispatch evidence tables for either legacy or operation-aware AE workflows."
    )
    ap.add_argument("--mode", type=str, default="operation", choices=["operation", "operation_uq", "legacy"])

    ap.add_argument(
        "--dispatch-agg-csv",
        type=str,
        default="results/supplementary_evidence/dispatch_closure_mapping_decision_aggregate.csv",
    )
    ap.add_argument(
        "--dispatch-meta-json",
        type=str,
        default="results/supplementary_evidence/dispatch_closure_mapping_decision.meta.json",
    )
    ap.add_argument("--focus-mapping", type=str, default="risk_conditional")
    ap.add_argument("--focus-subset", type=str, default="high_risk_shared")
    ap.add_argument(
        "--model-order",
        type=str,
        default=",".join(DEFAULT_OPERATION_MODEL_ORDER),
        help="Comma-separated display order for operation-aware summary.",
    )
    ap.add_argument(
        "--out-summary-csv",
        type=str,
        default="results/ae_operation_summary/operation_high_risk_summary.csv",
    )
    ap.add_argument(
        "--out-figure-prefix",
        type=str,
        default="results/ae_operation_summary/fig_operation_high_risk_metrics",
    )
    ap.add_argument(
        "--out-meta",
        type=str,
        default="",
    )
    ap.add_argument(
        "--out-latex-table",
        type=str,
        default="",
        help="Optional LaTeX table snippet path. Currently used for operation_uq summaries.",
    )

    ap.add_argument(
        "--fairness-csv",
        type=str,
        default="results/supplementary_evidence/fairness_closure_fullfault_hard_table.csv",
    )
    ap.add_argument(
        "--dispatch-l1-csv",
        type=str,
        default="results/supplementary_evidence/dispatch_raw_vs_mapped_summary.csv",
    )
    ap.add_argument(
        "--matrix4-csv",
        type=str,
        default="results/supplementary_evidence/main4_matrix4_dispatch_value.csv",
    )
    ap.add_argument(
        "--rank-csv",
        type=str,
        default="results/supplementary_evidence/dispatch_closure_mapping_decision_ranking_stability.csv",
    )
    return ap.parse_args()


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _require_columns(df: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def _assert_models(df: pd.DataFrame, model_order: List[str], source: str) -> None:
    got = set(df["model"].astype(str))
    miss = [m for m in model_order if m not in got]
    if miss:
        raise ValueError(f"Missing model rows in {source}: {miss}")


def _parse_model_order(raw: str) -> List[str]:
    items = [part.strip() for part in str(raw).split(",") if part.strip()]
    return items or DEFAULT_OPERATION_MODEL_ORDER


def _build_operation_summary(
    dispatch_agg_csv: Path,
    dispatch_meta_json: Path,
    *,
    focus_mapping: str,
    focus_subset: str,
    model_order: List[str],
) -> tuple[pd.DataFrame, Dict[str, object]]:
    agg = pd.read_csv(dispatch_agg_csv)
    _require_columns(
        agg,
        [
            "mapping_method",
            "subset",
            "model",
            "success_rate",
            "feasibility_fail_rate",
            "backfire_rate_vs_dlinear",
            "total_slack_mwh",
            "total_positive_error_mwh",
            "p95_positive_error_mw",
            "total_curtailment_mwh",
            "total_rt_cost",
            "forecast_r2",
            "mean_forecast_error_mw",
        ],
        str(dispatch_agg_csv),
    )
    focus = agg[
        (agg["mapping_method"].astype(str) == str(focus_mapping))
        & (agg["subset"].astype(str) == str(focus_subset))
    ].copy()
    if focus.empty:
        raise ValueError(
            f"No rows found in {dispatch_agg_csv} for mapping={focus_mapping}, subset={focus_subset}."
        )

    focus["model"] = focus["model"].astype(str)
    ordered_models = [model for model in model_order if model in set(focus["model"])]
    extra_models = [model for model in sorted(set(focus["model"])) if model not in ordered_models]
    final_order = ordered_models + extra_models
    focus["model"] = pd.Categorical(focus["model"], categories=final_order, ordered=True)
    focus = focus.sort_values("model").reset_index(drop=True)

    meta = _load_json(dispatch_meta_json)
    reserve_margin = {}
    mapping_params = meta.get("mapping_params", {}).get(str(focus_mapping), {})
    if isinstance(mapping_params, dict):
        for model_name, params in mapping_params.items():
            if isinstance(params, dict):
                reserve_margin[str(model_name)] = float(params.get("reserve_margin_mw", 0.0) or 0.0)

    focus["reserve_margin_mw"] = focus["model"].astype(str).map(lambda m: reserve_margin.get(m, np.nan))
    focus["high_risk_success_rate"] = focus["success_rate"].astype(float)
    focus["high_risk_feasibility_fail_rate"] = focus["feasibility_fail_rate"].astype(float)
    focus["high_risk_backfire_risk_vs_dlinear"] = focus["backfire_rate_vs_dlinear"].astype(float)
    focus["high_risk_total_slack_mwh"] = focus["total_slack_mwh"].astype(float)
    focus["reserve_call_proxy_mwh"] = focus["total_positive_error_mwh"].astype(float)
    focus["p95_positive_error_mw"] = focus["p95_positive_error_mw"].astype(float)
    focus["high_risk_total_curtailment_mwh"] = focus["total_curtailment_mwh"].astype(float)
    focus["high_risk_total_rt_cost"] = focus["total_rt_cost"].astype(float)
    focus["high_risk_forecast_r2"] = focus["forecast_r2"].astype(float)
    focus["high_risk_mean_forecast_error_mw"] = focus["mean_forecast_error_mw"].astype(float)

    cols = [
        "model",
        "mapping_method",
        "subset",
        "high_risk_success_rate",
        "high_risk_feasibility_fail_rate",
        "high_risk_backfire_risk_vs_dlinear",
        "reserve_margin_mw",
        "high_risk_total_slack_mwh",
        "reserve_call_proxy_mwh",
        "p95_positive_error_mw",
        "high_risk_total_curtailment_mwh",
        "high_risk_total_rt_cost",
        "high_risk_forecast_r2",
        "high_risk_mean_forecast_error_mw",
    ]
    summary = focus[cols].copy()

    meta_payload = {
        "dispatch_agg_csv": str(dispatch_agg_csv).replace("\\", "/"),
        "dispatch_meta_json": str(dispatch_meta_json).replace("\\", "/"),
        "focus_mapping": str(focus_mapping),
        "focus_subset": str(focus_subset),
        "model_order": final_order,
        "reserve_margin_available": bool(reserve_margin),
    }
    return summary, meta_payload


def _build_operation_uq_summary(
    dispatch_agg_csv: Path,
    dispatch_meta_json: Path,
    *,
    focus_mapping: str,
    focus_subset: str,
    model_order: List[str],
) -> tuple[pd.DataFrame, Dict[str, object]]:
    agg = pd.read_csv(dispatch_agg_csv)
    _require_columns(
        agg,
        [
            "mapping_method",
            "subset",
            "model",
            "total_slack_mwh",
            "total_positive_error_mwh",
            "p95_positive_error_mw",
            "total_rt_cost",
            "total_curtailment_mwh",
            "backfire_rate_vs_dlinear",
            "coverage_rate",
            "coverage_target",
            "mean_margin_mw",
            "forecast_r2",
            "hours",
            "holdout_rate",
        ],
        str(dispatch_agg_csv),
    )
    focus = agg[
        (agg["mapping_method"].astype(str) == str(focus_mapping))
        & (agg["subset"].astype(str) == str(focus_subset))
    ].copy()
    if focus.empty:
        raise ValueError(
            f"No rows found in {dispatch_agg_csv} for mapping={focus_mapping}, subset={focus_subset}."
        )

    focus["model"] = focus["model"].astype(str)
    ordered_models = [model for model in model_order if model in set(focus["model"])]
    extra_models = [model for model in sorted(set(focus["model"])) if model not in ordered_models]
    final_order = ordered_models + extra_models
    focus["model"] = pd.Categorical(focus["model"], categories=final_order, ordered=True)
    focus = focus.sort_values("model").reset_index(drop=True)

    meta = _load_json(dispatch_meta_json)
    uq_meta = meta.get("uq", {}) if isinstance(meta, dict) else {}

    summary = focus[
        [
            "model",
            "mapping_method",
            "subset",
            "hours",
            "total_slack_mwh",
            "total_positive_error_mwh",
            "p95_positive_error_mw",
            "total_rt_cost",
            "total_curtailment_mwh",
            "backfire_rate_vs_dlinear",
            "coverage_rate",
            "coverage_target",
            "mean_margin_mw",
            "forecast_r2",
            "holdout_rate",
        ]
    ].copy()
    summary = summary.rename(
        columns={
            "hours": "holdout_hours",
            "total_slack_mwh": "uq_holdout_total_slack_mwh",
            "total_positive_error_mwh": "uq_holdout_positive_error_mwh",
            "p95_positive_error_mw": "uq_holdout_p95_positive_error_mw",
            "total_rt_cost": "uq_holdout_total_rt_cost",
            "total_curtailment_mwh": "uq_holdout_total_curtailment_mwh",
            "backfire_rate_vs_dlinear": "uq_holdout_backfire_risk_vs_dlinear",
            "coverage_rate": "uq_holdout_coverage_rate",
            "coverage_target": "uq_holdout_coverage_target",
            "mean_margin_mw": "uq_holdout_mean_margin_mw",
            "forecast_r2": "uq_holdout_forecast_r2",
            "holdout_rate": "uq_holdout_rate",
        }
    )

    meta_payload = {
        "dispatch_agg_csv": str(dispatch_agg_csv).replace("\\", "/"),
        "dispatch_meta_json": str(dispatch_meta_json).replace("\\", "/"),
        "focus_mapping": str(focus_mapping),
        "focus_subset": str(focus_subset),
        "model_order": final_order,
        "decision_mode": str(meta.get("decision_mode", "")),
        "uq_enabled": bool(uq_meta.get("enabled", False)),
        "uq_alpha": uq_meta.get("uq_alpha"),
        "coverage_target": uq_meta.get("coverage_target"),
        "holdout_start_day": uq_meta.get("holdout_start_day"),
    }
    return summary, meta_payload


def _build_operation_figure(summary: pd.DataFrame, out_figure_prefix: Path) -> List[str]:
    out_figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    models = summary["model"].astype(str).tolist()
    colors = [MODEL_COLOR.get(model, "#4B5563") for model in models]

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.edgecolor": "#6B7280",
            "axes.linewidth": 0.8,
            "grid.color": "#CBD5E1",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    metrics = [
        ("high_risk_total_slack_mwh", "High-risk slack burden (MWh)"),
        ("high_risk_backfire_risk_vs_dlinear", "Backfire risk vs DLinear"),
        ("reserve_call_proxy_mwh", "Reserve-call proxy (MWh)"),
        ("reserve_margin_mw", "Reserve-like margin (MW)"),
    ]

    for ax, (col, title) in zip(axes.ravel(), metrics):
        vals = summary[col].astype(float).to_numpy()
        ypos = np.arange(len(models), dtype=float)
        ax.barh(ypos, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(models)
        ax.set_title(title, pad=8)
        ax.grid(True, axis="x", alpha=0.85)
        for yi, val in zip(ypos, vals):
            if np.isfinite(val):
                ax.text(val, yi, f" {val:.3f}", va="center", ha="left", fontsize=8)

    fig.suptitle("Operation-aware evidence on the high-risk subset", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    pdf_path = out_figure_prefix.with_suffix(".pdf")
    png_path = out_figure_prefix.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf_path).replace("\\", "/"), str(png_path).replace("\\", "/")]


def _build_operation_uq_figure(summary: pd.DataFrame, out_figure_prefix: Path) -> List[str]:
    out_figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    models = summary["model"].astype(str).tolist()
    colors = [MODEL_COLOR.get(model, "#4B5563") for model in models]

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.edgecolor": "#6B7280",
            "axes.linewidth": 0.8,
            "grid.color": "#CBD5E1",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    metrics = [
        ("uq_holdout_total_slack_mwh", "Holdout slack burden (MWh)"),
        ("uq_holdout_positive_error_mwh", "Holdout positive-error exposure (MWh)"),
        ("uq_holdout_coverage_rate", "One-sided coverage rate"),
        ("uq_holdout_mean_margin_mw", "Mean calibrated margin (MW)"),
    ]

    for ax, (col, title) in zip(axes.ravel(), metrics):
        vals = summary[col].astype(float).to_numpy()
        ypos = np.arange(len(models), dtype=float)
        ax.barh(ypos, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(models)
        ax.set_title(title, pad=8)
        ax.grid(True, axis="x", alpha=0.85)
        for yi, val in zip(ypos, vals):
            if np.isfinite(val):
                ax.text(val, yi, f" {val:.3f}", va="center", ha="left", fontsize=8)

    fig.suptitle("Calibrated UQ-aware holdout evidence", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    pdf_path = out_figure_prefix.with_suffix(".pdf")
    png_path = out_figure_prefix.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf_path).replace("\\", "/"), str(png_path).replace("\\", "/")]


def _fmt_num(value: float, digits: int = 2, *, scale: float = 1.0) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value) / float(scale):.{digits}f}"


def _write_operation_uq_latex_table(summary: pd.DataFrame, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_hours = int(summary["holdout_hours"].max()) if "holdout_hours" in summary.columns else 0
    focus_subset = (
        str(summary["subset"].iloc[0])
        if "subset" in summary.columns and not summary.empty
        else "holdout"
    )
    subset_label = {
        "all_holdout": "chronological all-holdout readout",
        "high_risk_shared_holdout": "shared high-risk holdout readout",
        "high_risk_model_specific_holdout": "model-specific high-risk holdout readout",
        "high_risk_holdout": "high-risk holdout readout",
    }.get(focus_subset, f"{focus_subset.replace('_', ' ')} readout")
    coverage_target = (
        float(summary["uq_holdout_coverage_target"].iloc[0])
        if "uq_holdout_coverage_target" in summary.columns and not summary.empty
        else float("nan")
    )

    panel_a_rows: List[str] = []
    panel_b_rows: List[str] = []
    for _, row in summary.iterrows():
        model = str(row["model"])
        panel_a_rows.append(
            "        "
            + " & ".join(
                [
                    model,
                    _fmt_num(row["uq_holdout_total_slack_mwh"]),
                    _fmt_num(row["uq_holdout_positive_error_mwh"]),
                    _fmt_num(row["uq_holdout_p95_positive_error_mw"]),
                    _fmt_num(row["uq_holdout_total_rt_cost"], scale=1e6),
                ]
            )
            + r" \\"
        )
        panel_b_rows.append(
            "        "
            + " & ".join(
                [
                    model,
                    _fmt_num(row["uq_holdout_total_curtailment_mwh"]),
                    _fmt_num(100.0 * float(row["uq_holdout_backfire_risk_vs_dlinear"]), digits=1),
                    _fmt_num(100.0 * float(row["uq_holdout_coverage_rate"]), digits=1),
                    _fmt_num(row["uq_holdout_mean_margin_mw"]),
                    _fmt_num(row["uq_holdout_forecast_r2"], digits=3),
                ]
            )
            + r" \\"
        )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Calibrated UQ-aware {subset_label} under the declared controller-side interface.}}",
        r"\label{tab:master_dispatch_uq}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\aePanelHeading{Panel A}{Holdout burden-type endpoints}",
        r"\begin{tabular*}{0.98\linewidth}{@{\extracolsep{\fill}}lcccc@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Slack (MWh)} & \textbf{Pos. err. (MWh)} & \textbf{P95+ err. (MW)} & \textbf{RT cost (M CNY)} \\",
        r"\midrule",
        *panel_a_rows,
        r"\bottomrule",
        r"\end{tabular*}",
        r"\vspace{4pt}",
        r"\aePanelHeading{Panel B}{Coverage and conservatism endpoints}",
        r"\begin{tabular*}{0.98\linewidth}{@{\extracolsep{\fill}}lccccc@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Curt. (MWh)} & \textbf{Backfire (\%)} & \textbf{Coverage (\%)} & \textbf{Mean margin (MW)} & \textbf{$R^2$} \\",
        r"\midrule",
        *panel_b_rows,
        r"\bottomrule",
        r"\end{tabular*}",
        (
            r"\aeTableNote{The holdout layer is evaluated only after chronological calibration on the earlier block of the same "
            + f"{holdout_hours}-hour subset. "
            + "The deployed day-ahead input is the mapped point forecast minus a calibrated one-sided residual margin. "
            + f"Coverage is reported against the target one-sided rate of {_fmt_num(100.0 * coverage_target, digits=1)}\\%, "
            + f"and backfire is measured relative to DLinear on the same {focus_subset.replace('_', ' ')} slice.}}"
        ),
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path).replace("\\", "/")


def _build_legacy_summary(
    fairness_csv: Path,
    dispatch_l1_csv: Path,
    matrix4_csv: Path,
    dispatch_agg_csv: Path,
    rank_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    fairness = pd.read_csv(fairness_csv)
    _require_columns(
        fairness,
        ["model", "training_mode", "gaussian_R2_mean", "missingness_R2_mean"],
        str(fairness_csv),
    )
    fair = fairness[
        (fairness["training_mode"].astype(str) == "fault_augmented")
        & (fairness["model"].astype(str).isin(LEGACY_MODEL_ORDER))
    ][["model", "gaussian_R2_mean", "missingness_R2_mean"]].copy()
    _assert_models(fair, LEGACY_MODEL_ORDER, str(fairness_csv))

    l1 = pd.read_csv(dispatch_l1_csv)
    _require_columns(l1, ["model", "mapped_fallback_pct"], str(dispatch_l1_csv))
    l1 = l1[l1["model"].astype(str).isin(LEGACY_MODEL_ORDER)][["model", "mapped_fallback_pct"]].copy()
    _assert_models(l1, LEGACY_MODEL_ORDER, str(dispatch_l1_csv))

    matrix4 = pd.read_csv(matrix4_csv)
    _require_columns(matrix4, ["layer", "model", "operating_cost_million"], str(matrix4_csv))
    l1_cost = matrix4[
        (matrix4["layer"].astype(str) == "L1_CopperPlate_120d")
        & (matrix4["model"].astype(str).isin(LEGACY_MODEL_ORDER))
    ][["model", "operating_cost_million"]].copy()
    _assert_models(l1_cost, LEGACY_MODEL_ORDER, str(matrix4_csv))

    agg = pd.read_csv(dispatch_agg_csv)
    _require_columns(
        agg,
        [
            "mapping_method",
            "subset",
            "model",
            "total_rt_cost",
            "total_curtailment_mwh",
            "success_rate",
            "backfire_rate_vs_dlinear",
        ],
        str(dispatch_agg_csv),
    )
    l2 = agg[
        (agg["mapping_method"].astype(str) == "affine")
        & (agg["subset"].astype(str) == "all")
        & (agg["model"].astype(str).isin(LEGACY_MODEL_ORDER))
    ][["model", "total_curtailment_mwh", "success_rate"]].copy()
    _assert_models(l2, LEGACY_MODEL_ORDER, str(dispatch_agg_csv))
    l2["l2_success_pct"] = 100.0 * l2["success_rate"].astype(float)

    risk = agg[
        (agg["subset"].astype(str) == "all")
        & (agg["model"].astype(str).isin(LEGACY_MODEL_ORDER))
    ][["model", "backfire_rate_vs_dlinear"]].copy()
    _assert_models(risk, LEGACY_MODEL_ORDER, str(dispatch_agg_csv))
    risk = (
        risk.groupby("model", as_index=False)["backfire_rate_vs_dlinear"]
        .mean()
        .rename(columns={"backfire_rate_vs_dlinear": "backfire_risk_ratio"})
    )
    risk["backfire_risk_pct"] = 100.0 * risk["backfire_risk_ratio"].astype(float)

    cost_rows = agg[
        (agg["subset"].astype(str) == "all")
        & (agg["mapping_method"].astype(str).isin(LEGACY_MAPPING_ORDER))
        & (agg["model"].astype(str).isin(LEGACY_MODEL_ORDER))
    ][["mapping_method", "model", "total_rt_cost"]].copy()
    cost_rows = cost_rows.groupby(["mapping_method", "model"], as_index=False)["total_rt_cost"].mean()

    rank = pd.read_csv(rank_csv)
    _require_columns(rank, ["record_type", "mapping_method", "model", "rank_by_total_rt_cost"], str(rank_csv))
    rank_rows = rank[
        (rank["record_type"].astype(str) == "rank")
        & (rank["mapping_method"].astype(str).isin(LEGACY_MAPPING_ORDER))
        & (rank["model"].astype(str).isin(LEGACY_MODEL_ORDER))
    ][["mapping_method", "model", "rank_by_total_rt_cost"]].copy()
    _assert_models(rank_rows, LEGACY_MODEL_ORDER, str(rank_csv))

    rank_stats = (
        rank_rows.groupby("model", as_index=False)["rank_by_total_rt_cost"]
        .agg(rank_mean="mean", rank_worst="max")
        .copy()
    )
    rank_var = (
        rank_rows.groupby("model")["rank_by_total_rt_cost"]
        .apply(lambda s: float(np.var(s.to_numpy(dtype=float), ddof=0)))
        .reset_index(name="rank_var")
    )
    rank_stats = rank_stats.merge(rank_var, on="model", how="left")

    summary = (
        fair.merge(l1, on="model", how="left")
        .merge(l1_cost, on="model", how="left")
        .merge(l2[["model", "total_curtailment_mwh", "l2_success_pct"]], on="model", how="left")
        .merge(rank_stats, on="model", how="left")
        .merge(risk[["model", "backfire_risk_pct"]], on="model", how="left")
    )
    summary = summary.rename(
        columns={
            "gaussian_R2_mean": "r2_at_060",
            "missingness_R2_mean": "missingness_r2_at_060",
            "mapped_fallback_pct": "l1_fallback_pct",
            "operating_cost_million": "l1_operating_cost_m_cny",
            "total_curtailment_mwh": "l2_curtailment_mwh",
        }
    )
    summary["rank_worst"] = summary["rank_worst"].astype(int)
    summary["rank_avg_var_worst"] = summary.apply(
        lambda r: f"{float(r['rank_mean']):.2f}/{float(r['rank_var']):.2f}/{int(r['rank_worst'])}",
        axis=1,
    )
    summary["model"] = pd.Categorical(summary["model"], categories=LEGACY_MODEL_ORDER, ordered=True)
    summary = summary.sort_values("model").reset_index(drop=True)

    meta_metrics = {"spearman_mean": np.nan, "top1_consistency": np.nan}
    if {"record_type", "spearman_rank_corr", "top1_consistency"}.issubset(rank.columns):
        s = rank[rank["record_type"].astype(str) == "summary"]
        if not s.empty:
            meta_metrics["spearman_mean"] = float(s.iloc[0]["spearman_rank_corr"])
            meta_metrics["top1_consistency"] = float(s.iloc[0]["top1_consistency"])

    return summary, rank_rows, cost_rows, meta_metrics


def _build_legacy_bump_chart(rank_rows: pd.DataFrame, cost_rows: pd.DataFrame, out_figure_prefix: Path) -> List[str]:
    out_figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    x_labels = LEGACY_MAPPING_ORDER
    x_pos = np.arange(len(x_labels), dtype=float)
    x_map = {m: i for i, m in enumerate(x_labels)}

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.edgecolor": "#6B7280",
            "axes.linewidth": 0.8,
            "grid.color": "#CBD5E1",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(8.5, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.15], hspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[1, 0])
    ax.set_facecolor("#F8FAFC")
    axb.set_facecolor("#F8FAFC")
    cost_map = {
        (str(r["mapping_method"]), str(r["model"])): float(r["total_rt_cost"])
        for _, r in cost_rows.iterrows()
    }

    for model in LEGACY_MODEL_ORDER:
        sub = rank_rows[rank_rows["model"].astype(str) == model].copy()
        sub["x"] = sub["mapping_method"].map(x_map)
        sub = sub.sort_values("x")
        y = sub["rank_by_total_rt_cost"].astype(float).to_numpy()
        x = sub["x"].astype(float).to_numpy()
        color = MODEL_COLOR[model]
        ax.plot(x, y, color=color, linewidth=2.6, marker="o", markersize=7.0, label=model, zorder=3)
        for xi, yi, mm in zip(x, y, sub["mapping_method"].tolist()):
            cost_m = cost_map.get((str(mm), model), np.nan) / 1e6
            ax.text(xi, yi - 0.16, f"{int(yi)}", ha="center", va="center", fontsize=8.5, color=color, weight="bold")
            if np.isfinite(cost_m):
                ax.text(xi, yi + 0.20, f"{cost_m:.2f}M", ha="center", va="center", fontsize=7.2, color="#1F2937")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Affine", "Isotonic", "Quantile"])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(4.4, 0.6)
    ax.set_ylabel("Dispatch Rank (1 = best)")
    ax.set_xlabel("Mapping interface")
    ax.set_title("Ranking sensitivity under mapping interfaces", pad=9)
    ax.grid(True, axis="y", alpha=0.85)
    ax.text(0.02, 0.02, "Point labels show total RT cost (M CNY)", transform=ax.transAxes, fontsize=7.5, color="#374151")

    width = 0.18
    x_idx = np.arange(len(LEGACY_MAPPING_ORDER), dtype=float)
    for j, model in enumerate(LEGACY_MODEL_ORDER):
        ys = []
        for mm in LEGACY_MAPPING_ORDER:
            ys.append(cost_map.get((mm, model), np.nan) / 1e6)
        offset = (j - 1.5) * width
        axb.bar(x_idx + offset, ys, width=width, color=MODEL_COLOR[model], edgecolor="white", linewidth=0.8, label=model)
    axb.set_xticks(x_idx)
    axb.set_xticklabels(["Affine", "Isotonic", "Quantile"])
    axb.set_ylabel("Total RT cost (M CNY)")
    axb.set_xlabel("Mapping interface")
    axb.set_title("Absolute cost under each mapping interface", pad=6)
    axb.grid(True, axis="y", alpha=0.85)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    pdf_path = out_figure_prefix.with_suffix(".pdf")
    png_path = out_figure_prefix.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf_path).replace("\\", "/"), str(png_path).replace("\\", "/")]


def main() -> None:
    args = parse_args()
    out_summary_csv = Path(args.out_summary_csv)
    out_figure_prefix = Path(args.out_figure_prefix)
    out_meta = Path(args.out_meta) if str(args.out_meta).strip() else out_summary_csv.with_suffix(".meta.json")
    out_latex_table = Path(args.out_latex_table) if str(args.out_latex_table).strip() else None
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    if out_latex_table is not None:
        out_latex_table.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "operation":
        summary, meta_payload = _build_operation_summary(
            dispatch_agg_csv=Path(args.dispatch_agg_csv),
            dispatch_meta_json=Path(args.dispatch_meta_json),
            focus_mapping=args.focus_mapping,
            focus_subset=args.focus_subset,
            model_order=_parse_model_order(args.model_order),
        )
        summary.to_csv(out_summary_csv, index=False)
        fig_paths = _build_operation_figure(summary, out_figure_prefix)
        meta = {
            "script": "build_decision_aware_dispatch_evidence.py",
            "mode": "operation",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": meta_payload,
            "outputs": {
                "summary_csv": str(out_summary_csv).replace("\\", "/"),
                "figure_pdf": fig_paths[0],
                "figure_png": fig_paths[1],
                "meta_json": str(out_meta).replace("\\", "/"),
            },
        }
        out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    elif args.mode == "operation_uq":
        focus_subset = args.focus_subset
        if str(focus_subset).strip() == "high_risk_shared":
            focus_subset = "high_risk_shared_holdout"
        summary, meta_payload = _build_operation_uq_summary(
            dispatch_agg_csv=Path(args.dispatch_agg_csv),
            dispatch_meta_json=Path(args.dispatch_meta_json),
            focus_mapping=args.focus_mapping,
            focus_subset=focus_subset,
            model_order=_parse_model_order(args.model_order),
        )
        summary.to_csv(out_summary_csv, index=False)
        fig_paths = _build_operation_uq_figure(summary, out_figure_prefix)
        latex_table_path = ""
        if out_latex_table is not None:
            latex_table_path = _write_operation_uq_latex_table(summary, out_latex_table)
        meta = {
            "script": "build_decision_aware_dispatch_evidence.py",
            "mode": "operation_uq",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": meta_payload,
            "outputs": {
                "summary_csv": str(out_summary_csv).replace("\\", "/"),
                "figure_pdf": fig_paths[0],
                "figure_png": fig_paths[1],
                "meta_json": str(out_meta).replace("\\", "/"),
                "latex_table": latex_table_path,
            },
        }
        out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        summary, rank_rows, cost_rows, meta_metrics = _build_legacy_summary(
            fairness_csv=Path(args.fairness_csv),
            dispatch_l1_csv=Path(args.dispatch_l1_csv),
            matrix4_csv=Path(args.matrix4_csv),
            dispatch_agg_csv=Path(args.dispatch_agg_csv),
            rank_csv=Path(args.rank_csv),
        )
        summary.to_csv(out_summary_csv, index=False)
        fig_paths = _build_legacy_bump_chart(rank_rows, cost_rows, out_figure_prefix)
        meta = {
            "script": "build_decision_aware_dispatch_evidence.py",
            "mode": "legacy",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "models": LEGACY_MODEL_ORDER,
            "mapping_order": LEGACY_MAPPING_ORDER,
            "sources": {
                "fairness_csv": str(Path(args.fairness_csv)).replace("\\", "/"),
                "dispatch_l1_csv": str(Path(args.dispatch_l1_csv)).replace("\\", "/"),
                "matrix4_csv": str(Path(args.matrix4_csv)).replace("\\", "/"),
                "dispatch_agg_csv": str(Path(args.dispatch_agg_csv)).replace("\\", "/"),
                "rank_csv": str(Path(args.rank_csv)).replace("\\", "/"),
            },
            "outputs": {
                "summary_csv": str(out_summary_csv).replace("\\", "/"),
                "figure_pdf": fig_paths[0],
                "figure_png": fig_paths[1],
                "meta_json": str(out_meta).replace("\\", "/"),
            },
            "metrics": {
                "spearman_mean": float(meta_metrics["spearman_mean"]),
                "top1_consistency": float(meta_metrics["top1_consistency"]),
            },
        }
        out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote: {out_summary_csv}")
    print(f"Wrote: {out_figure_prefix.with_suffix('.pdf')}")
    print(f"Wrote: {out_figure_prefix.with_suffix('.png')}")
    if out_latex_table is not None and args.mode == "operation_uq":
        print(f"Wrote: {out_latex_table}")
    print(f"Wrote: {out_meta}")


if __name__ == "__main__":
    main()
