import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def csv_rows(path: str) -> int:
    df = safe_read_csv(path)
    return int(len(df)) if not df.empty else 0


def first_row(df: pd.DataFrame, **conds):
    if df.empty:
        return None
    m = np.ones(len(df), dtype=bool)
    for k, v in conds.items():
        if k not in df.columns:
            return None
        m = m & (df[k] == v)
    out = df[m]
    if out.empty:
        return None
    return out.iloc[0]


def build_core2(evi_dir: str, run_id: str) -> pd.DataFrame:
    deep = safe_read_csv(os.path.join(evi_dir, "deep_sota_gaussian_windcn.csv"))
    robust = safe_read_csv(os.path.join(evi_dir, "structured_scada_fault_robustness_windcn.csv"))
    vmd_clean = safe_read_csv(os.path.join(evi_dir, "minimal_vmd_ceemdan_clean_windcn.csv"))
    mamba_clean = safe_read_csv(os.path.join(evi_dir, "mamba_clean_windcn.csv"))
    arima_clean = safe_read_csv(os.path.join(evi_dir, "arima_clean_windcn.csv"))
    svr_clean = safe_read_csv(os.path.join(evi_dir, "svr_clean_windcn.csv"))

    rows: List[Dict] = []

    # Deep models from deep_sota clean slice
    for m in ["DLinear", "iTransformer", "TimeMixer", "PatchTST"]:
        r = None
        if m == "DLinear":
            r = first_row(robust, fault_type="clean", severity=0.0, model="DLinear")
            src = "structured_scada_fault_robustness_windcn.csv"
        else:
            r = first_row(deep, sigma=0.0, model=m)
            src = "deep_sota_gaussian_windcn.csv"
        if r is not None:
            rows.append(
                {
                    "model_category": "A. General DL",
                    "model": m,
                    "R2": float(r["R2"]),
                    "RMSE_kW": float(r["RMSE"]),
                    "MAE_kW": float(r["MAE"]),
                    "source_file": src,
                    "run_id": run_id,
                }
            )

    # Edge-native
    # ultra_metrics_main_fulltest is json; parse separately
    uj = json.load(open(os.path.join(evi_dir, "ultra_metrics_main_fulltest.json"), "r", encoding="utf-8"))
    rows.append(
        {
            "model_category": "C. Edge-Native",
            "model": "Ultra-LSNT (Reliability, tau=0.5)",
            "R2": float(uj["metrics"]["R2"]),
            "RMSE_kW": float(uj["metrics"]["RMSE"]),
            "MAE_kW": float(uj["metrics"]["MAE"]),
            "source_file": "ultra_metrics_main_fulltest.json",
            "run_id": run_id,
        }
    )

    tau = safe_read_csv(os.path.join(evi_dir, "tau_sweep_ultra_pareto.csv"))
    r_tau = first_row(tau, tau=0.3)
    if r_tau is not None:
        rows.append(
            {
                "model_category": "C. Edge-Native",
                "model": "Ultra-LSNT (Edge-Frugal, tau=0.3)",
                "R2": float(r_tau["R2"]),
                "RMSE_kW": float(r_tau["RMSE"]),
                "MAE_kW": float(r_tau["MAE"]),
                "source_file": "tau_sweep_ultra_pareto.csv",
                "run_id": run_id,
            }
        )

    # Hybrid baselines
    for m in ["VMD-PSO-LSTM", "CEEMDAN-WOA-GRU"]:
        r = first_row(vmd_clean, model=m)
        if r is not None:
            rows.append(
                {
                    "model_category": "D. Hybrid Opt.",
                    "model": m,
                    "R2": float(r["R2"]),
                    "RMSE_kW": float(r["RMSE"]),
                    "MAE_kW": float(r["MAE"]),
                    "source_file": "minimal_vmd_ceemdan_clean_windcn.csv",
                    "run_id": run_id,
                }
            )

    # Mamba
    if not mamba_clean.empty:
        r = mamba_clean.iloc[0]
        rows.append(
            {
                "model_category": "A. General DL",
                "model": "Mamba (auditable)",
                "R2": float(r["R2"]),
                "RMSE_kW": float(r["RMSE"]),
                "MAE_kW": float(r["MAE"]),
                "source_file": "mamba_clean_windcn.csv",
                "run_id": run_id,
            }
        )

    # ARIMA / SVR
    if not arima_clean.empty:
        r = arima_clean.iloc[0]
        rows.append(
            {
                "model_category": "E. Traditional",
                "model": "ARIMA (auditable)",
                "R2": float(r["R2"]),
                "RMSE_kW": float(r["RMSE"]),
                "MAE_kW": float(r["MAE"]),
                "source_file": "arima_clean_windcn.csv",
                "run_id": run_id,
            }
        )
    if not svr_clean.empty:
        r = svr_clean.iloc[0]
        rows.append(
            {
                "model_category": "E. Traditional",
                "model": "SVR (auditable)",
                "R2": float(r["R2"]),
                "RMSE_kW": float(r["RMSE"]),
                "MAE_kW": float(r["MAE"]),
                "source_file": "svr_clean_windcn.csv",
                "run_id": run_id,
            }
        )

    return pd.DataFrame(rows)


def build_core3(evi_dir: str, run_id: str) -> pd.DataFrame:
    deep = safe_read_csv(os.path.join(evi_dir, "deep_sota_gaussian_windcn.csv"))
    robust = safe_read_csv(os.path.join(evi_dir, "structured_scada_fault_robustness_windcn.csv"))
    two = safe_read_csv(os.path.join(evi_dir, "robust_two_point_windcn.csv"))
    hybrid = safe_read_csv(os.path.join(evi_dir, "minimal_vmd_ceemdan_robustness_windcn.csv"))
    mamba = safe_read_csv(os.path.join(evi_dir, "mamba_robustness_windcn.csv"))
    arima = safe_read_csv(os.path.join(evi_dir, "arima_robustness_windcn.csv"))
    svr = safe_read_csv(os.path.join(evi_dir, "svr_robustness_windcn.csv"))

    rows: List[Dict] = []

    # clean references
    clean_models = ["Ultra-LSNT", "iTransformer", "DLinear"]
    for m in clean_models:
        r = first_row(robust, fault_type="clean", severity=0.0, model=m)
        if r is not None:
            rows.append(
                {
                    "fault_type": "clean_reference",
                    "severity": 0.0,
                    "sigma_eff": 0.0,
                    "model": m,
                    "R2": float(r["R2"]),
                    "rel_drop_pct": 0.0,
                    "status": "check",
                    "source_file": "structured_scada_fault_robustness_windcn.csv",
                    "run_id": run_id,
                }
            )

    for m in ["PatchTST", "TimeMixer"]:
        r = first_row(deep, sigma=0.0, model=m)
        if r is not None:
            rows.append(
                {
                    "fault_type": "clean_reference",
                    "severity": 0.0,
                    "sigma_eff": 0.0,
                    "model": m,
                    "R2": float(r["R2"]),
                    "rel_drop_pct": 0.0,
                    "status": "check",
                    "source_file": "deep_sota_gaussian_windcn.csv",
                    "run_id": run_id,
                }
            )

    # severe gaussian sigma_eff=0.60
    # deep_sota sigma cfg 0.4 corresponds to sigma_eff 0.6 with alpha=1.5
    for m in ["PatchTST", "TimeMixer", "iTransformer"]:
        rc = first_row(deep, sigma=0.0, model=m)
        rs = first_row(deep, sigma=0.4, model=m)
        if rc is not None and rs is not None:
            rel = 100.0 * (float(rc["R2"]) - float(rs["R2"])) / (abs(float(rc["R2"])) + 1e-12)
            rows.append(
                {
                    "fault_type": "gaussian",
                    "severity": 0.6,
                    "sigma_eff": 0.6,
                    "model": m,
                    "R2": float(rs["R2"]),
                    "rel_drop_pct": float(rel),
                    "status": "check",
                    "source_file": "deep_sota_gaussian_windcn.csv",
                    "run_id": run_id,
                }
            )

    for m in ["DLinear", "LightGBM", "Persistence", "Ultra-LSNT"]:
        r = first_row(two, model=m)
        if r is not None:
            rows.append(
                {
                    "fault_type": "gaussian",
                    "severity": 0.6,
                    "sigma_eff": 0.6,
                    "model": m,
                    "R2": float(r["r2_sigma_0_4"]),
                    "rel_drop_pct": float(100.0 * (float(r["r2_sigma_0_0"]) - float(r["r2_sigma_0_4"])) / (abs(float(r["r2_sigma_0_0"])) + 1e-12)),
                    "status": "check" if float(r["r2_sigma_0_4"]) > 0 else "fail",
                    "source_file": "robust_two_point_windcn.csv",
                    "run_id": run_id,
                }
            )

    for m in ["VMD-PSO-LSTM", "CEEMDAN-WOA-GRU"]:
        r = first_row(hybrid, fault_type="gaussian", severity=0.6, model=m)
        if r is not None:
            rows.append(
                {
                    "fault_type": "gaussian",
                    "severity": 0.6,
                    "sigma_eff": 0.6,
                    "model": m,
                    "R2": float(r["R2"]),
                    "rel_drop_pct": float(r["relative_drop_pct"]),
                    "status": "check",
                    "source_file": "minimal_vmd_ceemdan_robustness_windcn.csv",
                    "run_id": run_id,
                }
            )

    if not mamba.empty:
        rc = first_row(mamba, fault_type="gaussian", severity=0.15, model="Mamba (auditable)")
        rg = first_row(mamba, fault_type="gaussian", severity=0.6, model="Mamba (auditable)")
        rs = first_row(mamba, fault_type="spike", severity=0.6, model="Mamba (auditable)")
        clean_r2 = float(mamba.iloc[0]["R2_clean"]) if "R2_clean" in mamba.columns else (float(rc["R2_clean"]) if rc is not None else np.nan)
        if rg is not None:
            rows.append(
                {
                    "fault_type": "gaussian",
                    "severity": 0.6,
                    "sigma_eff": 0.6,
                    "model": "Mamba (auditable)",
                    "R2": float(rg["R2"]),
                    "rel_drop_pct": float(rg["relative_drop_pct"]),
                    "status": "check",
                    "source_file": "mamba_robustness_windcn.csv",
                    "run_id": run_id,
                }
            )
        if rs is not None:
            rows.append(
                {
                    "fault_type": "spike",
                    "severity": 0.6,
                    "sigma_eff": np.nan,
                    "model": "Mamba (auditable)",
                    "R2": float(rs["R2"]),
                    "rel_drop_pct": float(rs["relative_drop_pct"]),
                    "status": "check",
                    "source_file": "mamba_robustness_windcn.csv",
                    "run_id": run_id,
                }
            )

    for df, model_name, src in [
        (arima, "ARIMA (auditable)", "arima_robustness_windcn.csv"),
        (svr, "SVR (auditable)", "svr_robustness_windcn.csv"),
    ]:
        if df.empty:
            continue
        rg = first_row(df, fault_type="gaussian", severity=0.6, model=model_name)
        rs = first_row(df, fault_type="spike", severity=0.6, model=model_name)
        if rg is not None:
            rows.append(
                {
                    "fault_type": "gaussian",
                    "severity": 0.6,
                    "sigma_eff": 0.6,
                    "model": model_name,
                    "R2": float(rg["R2"]),
                    "rel_drop_pct": float(rg["relative_drop_pct"]),
                    "status": "check" if float(rg["R2"]) > 0 else "fail",
                    "source_file": src,
                    "run_id": run_id,
                }
            )
        if rs is not None:
            rows.append(
                {
                    "fault_type": "spike",
                    "severity": 0.6,
                    "sigma_eff": np.nan,
                    "model": model_name,
                    "R2": float(rs["R2"]),
                    "rel_drop_pct": float(rs["relative_drop_pct"]),
                    "status": "check" if float(rs["R2"]) > 0 else "fail",
                    "source_file": src,
                    "run_id": run_id,
                }
            )

    # spike severity 0.6 from structured/hybrid
    for m in ["Ultra-LSNT", "iTransformer", "DLinear"]:
        r = first_row(robust, fault_type="spike", severity=0.6, model=m)
        if r is not None:
            rows.append(
                {
                    "fault_type": "spike",
                    "severity": 0.6,
                    "sigma_eff": np.nan,
                    "model": m,
                    "R2": float(r["R2"]),
                    "rel_drop_pct": float(r["R2_rel_drop_pct_vs_clean"]),
                    "status": "check" if float(r["R2"]) > 0 else "fail",
                    "source_file": "structured_scada_fault_robustness_windcn.csv",
                    "run_id": run_id,
                }
            )

    for m in ["VMD-PSO-LSTM", "CEEMDAN-WOA-GRU"]:
        r = first_row(hybrid, fault_type="spike", severity=0.6, model=m)
        if r is not None:
            rows.append(
                {
                    "fault_type": "spike",
                    "severity": 0.6,
                    "sigma_eff": np.nan,
                    "model": m,
                    "R2": float(r["R2"]),
                    "rel_drop_pct": float(r["relative_drop_pct"]),
                    "status": "check",
                    "source_file": "minimal_vmd_ceemdan_robustness_windcn.csv",
                    "run_id": run_id,
                }
            )

    return pd.DataFrame(rows)


def build_core4(evi_dir: str, run_id: str) -> pd.DataFrame:
    cpu_main = safe_read_csv(os.path.join(evi_dir, "cpu_batch1_latency_table.csv"))
    cpu_ultra = safe_read_csv(os.path.join(evi_dir, "cpu_batch1_ultra_family_latency.csv"))
    hybrid_lat = safe_read_csv(os.path.join(evi_dir, "minimal_vmd_ceemdan_latency_windcn.csv"))
    mamba_lat = safe_read_csv(os.path.join(evi_dir, "mamba_latency_windcn.csv"))
    arima_lat = safe_read_csv(os.path.join(evi_dir, "arima_latency_windcn.csv"))
    svr_lat = safe_read_csv(os.path.join(evi_dir, "svr_latency_windcn.csv"))

    rows: List[Dict] = []

    for _, r in cpu_main.iterrows():
        rows.append(
            {
                "platform": "x86_cpu",
                "model": str(r["model"]),
                "batch1_mean_ms": float(r["batch1_latency_ms"]),
                "batch1_std_ms": float(r["batch1_latency_std_ms"]),
                "p95_ms": np.nan,
                "throughput_hz": float(r["batch1_freq_hz"]),
                "params_m": np.nan,
                "model_size_mib": np.nan,
                "source_file": "cpu_batch1_latency_table.csv",
                "run_id": run_id,
            }
        )

    for _, r in cpu_ultra.iterrows():
        rows.append(
            {
                "platform": "x86_cpu",
                "model": str(r["model"]),
                "batch1_mean_ms": float(r["latency_mean_ms"]),
                "batch1_std_ms": float(r["latency_std_ms"]),
                "p95_ms": float(r["latency_p95_ms"]),
                "throughput_hz": float(1000.0 / max(float(r["latency_mean_ms"]), 1e-12)),
                "params_m": np.nan,
                "model_size_mib": np.nan,
                "source_file": "cpu_batch1_ultra_family_latency.csv",
                "run_id": run_id,
            }
        )

    for _, r in hybrid_lat.iterrows():
        rows.append(
            {
                "platform": "x86_cpu",
                "model": str(r["model"]),
                "batch1_mean_ms": float(r["latency_mean_ms"]),
                "batch1_std_ms": float(r["latency_std_ms"]),
                "p95_ms": float(r["latency_p95_ms"]),
                "throughput_hz": float(1000.0 / max(float(r["latency_mean_ms"]), 1e-12)),
                "params_m": float(r["active_params_m"]),
                "model_size_mib": float(r["model_size_mib"]),
                "source_file": "minimal_vmd_ceemdan_latency_windcn.csv",
                "run_id": run_id,
            }
        )

    if not mamba_lat.empty:
        r = mamba_lat.iloc[0]
        rows.append(
            {
                "platform": "x86_cpu",
                "model": str(r["model"]),
                "batch1_mean_ms": float(r["latency_mean_ms"]),
                "batch1_std_ms": float(r["latency_std_ms"]),
                "p95_ms": float(r["latency_p95_ms"]),
                "throughput_hz": float(r["throughput_hz"]),
                "params_m": float(r["active_params_m"]),
                "model_size_mib": float(r["model_size_mib"]),
                "source_file": "mamba_latency_windcn.csv",
                "run_id": run_id,
            }
        )

    for df, src in [(arima_lat, "arima_latency_windcn.csv"), (svr_lat, "svr_latency_windcn.csv")]:
        if df.empty:
            continue
        r = df.iloc[0]
        rows.append(
            {
                "platform": str(r["platform"]),
                "model": str(r["model"]),
                "batch1_mean_ms": float(r["latency_mean_ms"]),
                "batch1_std_ms": float(r["latency_std_ms"]),
                "p95_ms": float(r["latency_p95_ms"]),
                "throughput_hz": float(r["throughput_hz"]),
                "params_m": float(r["active_params_m"]),
                "model_size_mib": float(r["model_size_mib"]),
                "source_file": src,
                "run_id": run_id,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build standardized core matrices from auditable evidence")
    ap.add_argument("--out_dir", type=str, default=os.path.join("results", "supplementary_evidence"))
    ap.add_argument("--audit_tag", type=str, default="core_matrix_build")
    args = ap.parse_args()

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + args.audit_tag

    core2 = build_core2(args.out_dir, run_id)
    core3 = build_core3(args.out_dir, run_id)
    core4 = build_core4(args.out_dir, run_id)

    core2_path = os.path.join(args.out_dir, "core2_clean.csv")
    core3_path = os.path.join(args.out_dir, "core3_robustness.csv")
    core4_path = os.path.join(args.out_dir, "core4_hardware.csv")

    core2.to_csv(core2_path, index=False)
    core3.to_csv(core3_path, index=False)
    core4.to_csv(core4_path, index=False)

    manifest_rows = [
        {
            "claim_id": "core1_protocol",
            "file": "split_manifest_80_20_unified.json;deep_sota_gaussian_windcn.csv;structured_scada_fault_robustness_windcn.csv",
            "rows": 1
            + csv_rows(os.path.join(args.out_dir, "deep_sota_gaussian_windcn.csv"))
            + csv_rows(os.path.join(args.out_dir, "structured_scada_fault_robustness_windcn.csv")),
            "generated_run_id": run_id,
        },
        {"claim_id": "core2_clean", "file": "core2_clean.csv", "rows": len(core2), "generated_run_id": run_id},
        {"claim_id": "core3_robustness", "file": "core3_robustness.csv", "rows": len(core3), "generated_run_id": run_id},
        {"claim_id": "core4_hardware", "file": "core4_hardware.csv", "rows": len(core4), "generated_run_id": run_id},
        {
            "claim_id": "core5_ablation_physics",
            "file": "pi_moe_full_vs_ultra_clean_sigma06_windcn.csv;lambda_robustness_sigma_eff_compare.csv;tau_sweep_ultra_pareto.csv;joint_sensitivity_soa_windcn.csv;joint_sensitivity_pareto_frontier.csv",
            "rows": csv_rows(os.path.join(args.out_dir, "pi_moe_full_vs_ultra_clean_sigma06_windcn.csv"))
            + csv_rows(os.path.join(args.out_dir, "lambda_robustness_sigma_eff_compare.csv"))
            + csv_rows(os.path.join(args.out_dir, "tau_sweep_ultra_pareto.csv"))
            + csv_rows(os.path.join(args.out_dir, "joint_sensitivity_soa_windcn.csv"))
            + csv_rows(os.path.join(args.out_dir, "joint_sensitivity_pareto_frontier.csv")),
            "generated_run_id": run_id,
        },
        {
            "claim_id": "core6_dispatch",
            "file": "dispatch_rolling_with_ultra_scaled_summary.csv;dispatch_network_ieee24_dcopf_wmax260_aggregate.csv;dispatch_network_ieee24_dcopf_wmax280_aggregate.csv",
            "rows": csv_rows(os.path.join(args.out_dir, "dispatch_rolling_with_ultra_scaled_summary.csv"))
            + csv_rows(os.path.join(args.out_dir, "dispatch_network_ieee24_dcopf_wmax260_aggregate.csv"))
            + csv_rows(os.path.join(args.out_dir, "dispatch_network_ieee24_dcopf_wmax280_aggregate.csv")),
            "generated_run_id": run_id,
        },
        {
            "claim_id": "core7_transfer",
            "file": "table_multi_domain.csv",
            "rows": csv_rows("table_multi_domain.csv"),
            "generated_run_id": run_id,
        },
    ]
    pd.DataFrame(manifest_rows).to_csv(os.path.join(args.out_dir, "evidence_manifest.csv"), index=False)


if __name__ == "__main__":
    main()
