import argparse
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def read_csv_any(*paths: str) -> pd.DataFrame:
    for path in paths:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


def find_row(df: pd.DataFrame, **conds):
    if df.empty:
        return None
    mask = np.ones(len(df), dtype=bool)
    for k, v in conds.items():
        if k not in df.columns:
            return None
        mask = mask & (df[k] == v)
    out = df[mask]
    return None if out.empty else out.iloc[0]


def to_million(x: float) -> float:
    return float(x) / 1_000_000.0


def _safe_float(v, default=np.nan) -> float:
    try:
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _normalize_model_name(name: str) -> str:
    key = str(name).strip().lower()
    mapping = {
        "ultra-lsnt": "Ultra-LSNT",
        "ultra": "Ultra-LSNT",
        "dlinear": "DLinear",
        "timemixer": "TimeMixer",
        "itransformer": "iTransformer",
        "fg-moe (full-trained)": "FG-MoE (full-trained)",
    }
    return mapping.get(key, str(name).strip())


def _infer_fairness_r2_at_060(row: pd.Series) -> float:
    if "gaussian_R2_mean" in row.index and not pd.isna(row["gaussian_R2_mean"]):
        return float(row["gaussian_R2_mean"])
    cols = [c for c in row.index if c.endswith("_R2_mean") and c != "clean_R2_mean"]
    vals = [_safe_float(row[c], np.nan) for c in cols]
    vals = [v for v in vals if np.isfinite(v)]
    if vals:
        return float(np.mean(vals))
    return np.nan


def _apply_fairness_closure_overrides(
    matrix2: pd.DataFrame,
    fairness_hard: pd.DataFrame,
    fairness_sig: pd.DataFrame,
) -> pd.DataFrame:
    if matrix2.empty or fairness_hard.empty:
        return matrix2

    if "training_mode" not in fairness_hard.columns:
        return matrix2
    fair = fairness_hard[fairness_hard["training_mode"] == "fault_augmented"].copy()
    if fair.empty:
        return matrix2

    out = matrix2.copy()
    sig_map: Dict[str, Dict[str, float]] = {}
    if not fairness_sig.empty and {"baseline", "metric", "pvalue_one_sided_ultra_better"}.issubset(set(fairness_sig.columns)):
        sig_r = fairness_sig[fairness_sig["metric"] == "RMSE_gain(base-ultra)"]
        for _, r in sig_r.iterrows():
            sig_map[_normalize_model_name(str(r["baseline"]))] = {
                "pvalue": _safe_float(r["pvalue_one_sided_ultra_better"], np.nan),
                "ci95_low": _safe_float(r.get("ci95_low", np.nan), np.nan),
                "ci95_high": _safe_float(r.get("ci95_high", np.nan), np.nan),
            }

    existing = {str(m): i for i, m in enumerate(out["model"].astype(str).tolist())}
    for _, fr in fair.iterrows():
        model = _normalize_model_name(fr["model"])
        row_update = {
            "clean_R2": _safe_float(fr.get("clean_R2_mean", np.nan), np.nan),
            "R2_sigma_eff_060": _infer_fairness_r2_at_060(fr),
            "R2_spike_06": _safe_float(fr.get("drift_R2_mean", np.nan), np.nan),
            "drop_pct_sigma060": _safe_float(fr.get("avg_degradation_r2_drop_pct", np.nan), np.nan),
            "fair_train_mode": "fault_augmented",
            "fair_n_seed": _safe_float(fr.get("n_seed", np.nan), np.nan),
            "fair_avg_degradation_rmse_pct": _safe_float(fr.get("avg_degradation_rmse_pct", np.nan), np.nan),
            "fair_avg_degradation_r2_drop_pct": _safe_float(fr.get("avg_degradation_r2_drop_pct", np.nan), np.nan),
            "fair_worst_q90_ae_kw": _safe_float(fr.get("worst_fault_AE_Q90_mean", np.nan), np.nan),
            "fair_source": "fairness_closure_fullfault_hard_table.csv",
        }
        if model in sig_map:
            row_update["fair_pvalue_vs_ultra_one_sided"] = sig_map[model]["pvalue"]
            row_update["fair_ci95_low_vs_ultra"] = sig_map[model]["ci95_low"]
            row_update["fair_ci95_high_vs_ultra"] = sig_map[model]["ci95_high"]

        if model in existing:
            idx = existing[model]
            for k, v in row_update.items():
                out.at[idx, k] = v
        else:
            # In case matrix2 currently does not include one of the four fairness models.
            payload = {c: np.nan for c in out.columns}
            payload.update(
                {
                    "category": "Fairness Closure",
                    "model": model,
                    "run_id": out["run_id"].iloc[0] if "run_id" in out.columns and not out.empty else "",
                }
            )
            payload.update(row_update)
            out = pd.concat([out, pd.DataFrame([payload])], axis=0, ignore_index=True)

    return out.reset_index(drop=True)


def build_matrix1(out_dir: str, run_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "Wind (CN)",
                "split": "80/20 chronological",
                "seq_len": 96,
                "pred_len": 24,
                "gaussian_sigma_eff": "0.15,0.30,0.45,0.60",
                "structured_faults": "flatline,spike,dropout@0.2/0.6",
                "dispatch_layers": "copper_plate_120d + RTS24_DCOPF",
                "source_files": "split_manifest_80_20_unified.json;deep_sota_gaussian_windcn.csv;structured_scada_fault_robustness_windcn.csv;lightgbm_gaussian_audit_windcn.csv;chronos_family_windcn.csv;moirai_family_windcn.csv;windcn_context_audit_summary.csv",
                "run_id": run_id,
            }
        ]
    )


def build_matrix2(out_dir: str, run_id: str) -> pd.DataFrame:
    core2 = read_csv(os.path.join(out_dir, "core2_clean.csv"))
    core3 = read_csv(os.path.join(out_dir, "core3_robustness.csv"))
    arima_clean = read_csv(os.path.join(out_dir, "arima_clean_windcn.csv"))
    svr_clean = read_csv(os.path.join(out_dir, "svr_clean_windcn.csv"))
    arima_rob = read_csv(os.path.join(out_dir, "arima_robustness_windcn.csv"))
    svr_rob = read_csv(os.path.join(out_dir, "svr_robustness_windcn.csv"))
    deep = read_csv(os.path.join(out_dir, "deep_sota_gaussian_windcn.csv"))
    robust_two = read_csv(os.path.join(out_dir, "robust_two_point_windcn.csv"))
    robust_struct = read_csv(os.path.join(out_dir, "structured_scada_fault_robustness_windcn.csv"))
    vmd_rob = read_csv(os.path.join(out_dir, "minimal_vmd_ceemdan_robustness_windcn.csv"))
    ssa_clean = read_csv(os.path.join(out_dir, "ssa_elm_clean_windcn.csv"))
    ssa_rob = read_csv(os.path.join(out_dir, "ssa_elm_robustness_windcn.csv"))
    context_audit = read_csv_any(
        os.path.join(out_dir, "windcn_context_audit_summary.csv"),
        os.path.join("results", "supplementary_evidence", "windcn_context_audit_summary.csv"),
    )
    multi = read_csv("table_multi_domain.csv")
    transfer_ext = read_csv(os.path.join(out_dir, "multi_domain_transfer_matrix.csv"))
    fairness_hard = read_csv(os.path.join(out_dir, "fairness_closure_fullfault_hard_table.csv"))
    fairness_sig = read_csv(os.path.join(out_dir, "fairness_closure_fullfault_significance.csv"))

    cd_map: Dict[str, Dict[str, float]] = {}
    if not transfer_ext.empty and "model" in transfer_ext.columns:
        for _, r in transfer_ext.iterrows():
            m = str(r["model"]).strip()
            cd_map[m] = {
                "Wind (US)": float(r["R2_wind_us"]) if "R2_wind_us" in r.index and not pd.isna(r["R2_wind_us"]) else np.nan,
                "Air Quality": float(r["R2_air_quality"]) if "R2_air_quality" in r.index and not pd.isna(r["R2_air_quality"]) else np.nan,
                "GEFCom Load": float(r["R2_gefcom_load"]) if "R2_gefcom_load" in r.index and not pd.isna(r["R2_gefcom_load"]) else np.nan,
            }
    elif not multi.empty:
        # Backward-compatible fallback: only Ultra-LSNT row from table_multi_domain.csv.
        tmp = {}
        for _, r in multi.iterrows():
            ds = str(r["Dataset"]).strip()
            tmp[ds] = float(r["R2 Score"])
        cd_map["Ultra-LSNT"] = {
            "Wind (US)": tmp.get("Wind (US)", np.nan),
            "Air Quality": tmp.get("Air Quality", np.nan),
            "GEFCom Load": tmp.get("GEFCom Load", np.nan),
        }

    def get_clean(model_name: str) -> Optional[Dict[str, float]]:
        if model_name == "ARIMA (auditable)" and not arima_clean.empty:
            r = arima_clean.iloc[0]
            return {"R2": float(r["R2"]), "RMSE": float(r["RMSE"]), "MAE": float(r["MAE"])}
        if model_name == "SVR (auditable)" and not svr_clean.empty:
            r = svr_clean.iloc[0]
            return {"R2": float(r["R2"]), "RMSE": float(r["RMSE"]), "MAE": float(r["MAE"])}
        if model_name == "SSA-ELM" and not ssa_clean.empty:
            r = ssa_clean.iloc[0]
            return {"R2": float(r["R2"]), "RMSE": float(r["RMSE"]), "MAE": float(r["MAE"])}
        r = find_row(core2, model=model_name)
        if r is not None:
            return {"R2": float(r["R2"]), "RMSE": float(r["RMSE_kW"]), "MAE": float(r["MAE_kW"])}
        if model_name == "Ultra-LSNT":
            r = find_row(core2, model="Ultra-LSNT (Reliability, tau=0.5)")
            if r is not None:
                return {"R2": float(r["R2"]), "RMSE": float(r["RMSE_kW"]), "MAE": float(r["MAE_kW"])}
        return None

    def get_rob(model_name: str, fault: str) -> Optional[Dict[str, float]]:
        if model_name in ["ARIMA (auditable)", "SVR (auditable)"]:
            df = arima_rob if "ARIMA" in model_name else svr_rob
            if df.empty:
                return None
            if fault == "gaussian":
                r = find_row(df, fault_type="gaussian", sigma_eff=0.6)
            else:
                r = find_row(df, fault_type="spike", severity=0.6)
            if r is None:
                return None
            return {"R2": float(r["R2"]), "drop": float(r["relative_drop_pct"])}
        if model_name == "SSA-ELM":
            if ssa_rob.empty:
                return None
            if fault == "gaussian":
                r = find_row(ssa_rob, fault_type="gaussian", sigma_eff=0.6)
            else:
                r = find_row(ssa_rob, fault_type="spike", severity=0.6)
            if r is None:
                return None
            return {"R2": float(r["R2"]), "drop": float(r["relative_drop_pct"])}
        if model_name in ["VMD-PSO-LSTM", "CEEMDAN-WOA-GRU"]:
            if vmd_rob.empty:
                return None
            if fault == "gaussian":
                r = find_row(vmd_rob, model=model_name, fault_type="gaussian", severity=0.6)
            else:
                r = find_row(vmd_rob, model=model_name, fault_type="spike", severity=0.6)
            if r is None:
                return None
            return {"R2": float(r["R2"]), "drop": float(r["relative_drop_pct"])}

        if model_name in ["PatchTST", "TimeMixer", "iTransformer"]:
            if deep.empty:
                return None
            clean = find_row(deep, model=model_name, sigma=0.0)
            if fault == "gaussian":
                r = find_row(deep, model=model_name, sigma=0.4)
                if clean is not None and r is not None:
                    drop = 100.0 * (float(clean["R2"]) - float(r["R2"])) / (abs(float(clean["R2"])) + 1e-12)
                    return {"R2": float(r["R2"]), "drop": float(drop)}
                return None
            r = find_row(core3, fault_type="spike", severity=0.6, model=model_name)
            if r is not None:
                return {"R2": float(r["R2"]), "drop": float(r["rel_drop_pct"])}
            # Fallback to structured robustness evidence if core3 does not contain this model.
            rs = find_row(robust_struct, fault_type="spike", severity=0.6, model=model_name)
            if rs is not None:
                drop = (
                    float(rs["R2_rel_drop_pct_vs_clean"])
                    if "R2_rel_drop_pct_vs_clean" in rs.index and not pd.isna(rs["R2_rel_drop_pct_vs_clean"])
                    else 100.0
                    * (float(rs["R2_clean"]) - float(rs["R2"]))
                    / (abs(float(rs["R2_clean"])) + 1e-12)
                )
                return {"R2": float(rs["R2"]), "drop": float(drop)}
            return None

        if model_name in ["Ultra-LSNT", "DLinear", "LightGBM", "Persistence"]:
            if robust_two.empty:
                return None
            r = find_row(robust_two, model=model_name)
            if r is None:
                return None
            if fault == "gaussian":
                return {"R2": float(r["r2_sigma_0_4"]), "drop": float(100.0 * (float(r["r2_sigma_0_0"]) - float(r["r2_sigma_0_4"])) / (abs(float(r["r2_sigma_0_0"])) + 1e-12))}
            sp = find_row(core3, fault_type="spike", severity=0.6, model=model_name)
            if sp is None:
                return None
            return {"R2": float(sp["R2"]), "drop": float(sp["rel_drop_pct"])}

        r = find_row(core3, fault_type="gaussian", severity=0.6, model=model_name) if fault == "gaussian" else find_row(core3, fault_type="spike", severity=0.6, model=model_name)
        if r is not None:
            return {"R2": float(r["R2"]), "drop": float(r["rel_drop_pct"])}
        return None

    models = [
        ("General DL", "DLinear"),
        ("General DL", "iTransformer"),
        ("General DL", "TimeMixer"),
        ("General DL", "Mamba (auditable)"),
        ("Hybrid Opt.", "VMD-PSO-LSTM"),
        ("Hybrid Opt.", "CEEMDAN-WOA-GRU"),
        ("Traditional", "ARIMA (auditable)"),
        ("Traditional", "SVR (auditable)"),
        ("Traditional", "SSA-ELM"),
        ("Edge-Native", "Ultra-LSNT"),
    ]

    rows: List[Dict] = []
    for cat, m in models:
        clean = get_clean(m)
        if clean is None:
            continue
        g = get_rob(m, "gaussian")
        s = get_rob(m, "spike")
        rows.append(
            {
                "category": cat,
                "model": m,
                "clean_R2": float(clean["R2"]),
                "clean_RMSE_kW": float(clean["RMSE"]),
                "clean_MAE_kW": float(clean["MAE"]),
                "R2_sigma_eff_015": np.nan,
                "RMSE_sigma_eff_015_kW": np.nan,
                "MAE_sigma_eff_015_kW": np.nan,
                "R2_sigma_eff_060": float(g["R2"]) if g is not None else np.nan,
                "RMSE_sigma_eff_060_kW": np.nan,
                "MAE_sigma_eff_060_kW": np.nan,
                "rel_rmse_increase_pct_060": np.nan,
                "R2_spike_06": float(s["R2"]) if s is not None else np.nan,
                "drop_pct_sigma060": float(g["drop"]) if g is not None else np.nan,
                "R2_wind_us": float(cd_map[m]["Wind (US)"]) if m in cd_map and not pd.isna(cd_map[m]["Wind (US)"]) else np.nan,
                "R2_air_quality": float(cd_map[m]["Air Quality"]) if m in cd_map and not pd.isna(cd_map[m]["Air Quality"]) else np.nan,
                "R2_gefcom_load": float(cd_map[m]["GEFCom Load"]) if m in cd_map and not pd.isna(cd_map[m]["GEFCom Load"]) else np.nan,
                "source_clean": "",
                "source_015": "",
                "source_060": "",
                "run_id": run_id,
            }
        )

    if rows:
        df_rows = pd.DataFrame(rows)
        df_rows = df_rows.drop_duplicates(subset=["model"], keep="last")
        df_rows = _apply_fairness_closure_overrides(df_rows.reset_index(drop=True), fairness_hard, fairness_sig)

        context_cols = [
            "category",
            "clean_R2",
            "clean_RMSE_kW",
            "clean_MAE_kW",
            "R2_sigma_eff_015",
            "RMSE_sigma_eff_015_kW",
            "MAE_sigma_eff_015_kW",
            "R2_sigma_eff_060",
            "RMSE_sigma_eff_060_kW",
            "MAE_sigma_eff_060_kW",
            "rel_rmse_increase_pct_060",
            "source_clean",
            "source_015",
            "source_060",
        ]
        for col in context_cols:
            if col not in df_rows.columns:
                df_rows[col] = "" if col.startswith("source_") else np.nan

        if not context_audit.empty and "model" in context_audit.columns:
            existing = {str(m).strip(): i for i, m in enumerate(df_rows["model"].astype(str).tolist())}
            for _, cr in context_audit.iterrows():
                model_name = str(cr["model"]).strip()
                if not model_name:
                    continue
                payload = {
                    "category": str(cr["category"]).strip() if "category" in cr.index and not pd.isna(cr["category"]) else np.nan,
                    "clean_R2": _safe_float(cr.get("clean_R2", np.nan), np.nan),
                    "clean_RMSE_kW": _safe_float(cr.get("clean_RMSE_kW", np.nan), np.nan),
                    "clean_MAE_kW": _safe_float(cr.get("clean_MAE_kW", np.nan), np.nan),
                    "R2_sigma_eff_015": _safe_float(cr.get("R2_sigma_eff_015", np.nan), np.nan),
                    "RMSE_sigma_eff_015_kW": _safe_float(cr.get("RMSE_sigma_eff_015_kW", np.nan), np.nan),
                    "MAE_sigma_eff_015_kW": _safe_float(cr.get("MAE_sigma_eff_015_kW", np.nan), np.nan),
                    "R2_sigma_eff_060": _safe_float(cr.get("R2_sigma_eff_060", np.nan), np.nan),
                    "RMSE_sigma_eff_060_kW": _safe_float(cr.get("RMSE_sigma_eff_060_kW", np.nan), np.nan),
                    "MAE_sigma_eff_060_kW": _safe_float(cr.get("MAE_sigma_eff_060_kW", np.nan), np.nan),
                    "rel_rmse_increase_pct_060": _safe_float(cr.get("rel_rmse_increase_pct_060", np.nan), np.nan),
                    "source_clean": str(cr.get("source_clean", "")) if "source_clean" in cr.index and not pd.isna(cr["source_clean"]) else "",
                    "source_015": str(cr.get("source_015", "")) if "source_015" in cr.index and not pd.isna(cr["source_015"]) else "",
                    "source_060": str(cr.get("source_060", "")) if "source_060" in cr.index and not pd.isna(cr["source_060"]) else "",
                }
                if model_name in existing:
                    idx = existing[model_name]
                    for key, value in payload.items():
                        if key == "category":
                            if isinstance(value, str) and value:
                                df_rows.at[idx, key] = value
                            continue
                        if key.startswith("source_"):
                            if value:
                                df_rows.at[idx, key] = value
                            continue
                        if not pd.isna(value):
                            df_rows.at[idx, key] = value
                else:
                    base = {c: ("" if c.startswith("source_") else np.nan) for c in df_rows.columns}
                    base.update(
                        {
                            "category": payload["category"] if isinstance(payload["category"], str) and payload["category"] else "Context Audit",
                            "model": model_name,
                            "run_id": run_id,
                        }
                    )
                    for key, value in payload.items():
                        if key == "category":
                            continue
                        base[key] = value
                    df_rows = pd.concat([df_rows, pd.DataFrame([base])], axis=0, ignore_index=True)
                    existing[model_name] = len(df_rows) - 1
        return df_rows.reset_index(drop=True)
    return pd.DataFrame(rows)


def build_matrix3(out_dir: str, run_id: str, matrix2: pd.DataFrame) -> pd.DataFrame:
    core4 = read_csv(os.path.join(out_dir, "core4_hardware.csv"))
    m3_old = read_csv(os.path.join(out_dir, "main6_matrix3_hardware_ablation.csv"))
    arima_lat = read_csv(os.path.join(out_dir, "arima_latency_windcn.csv"))
    svr_lat = read_csv(os.path.join(out_dir, "svr_latency_windcn.csv"))
    ssa_lat = read_csv(os.path.join(out_dir, "ssa_elm_latency_windcn.csv"))
    soa = read_csv(os.path.join(out_dir, "joint_sensitivity_soa_windcn.csv"))
    pi_cmp = read_csv(os.path.join(out_dir, "pi_moe_full_vs_ultra_clean_sigma06_windcn.csv"))
    clip_cmp = read_csv(os.path.join(out_dir, "fgmoe_clip_only_control_windcn.csv"))
    physics_cmp = read_csv(os.path.join(out_dir, "physics_closure.csv"))
    tau = read_csv(os.path.join(out_dir, "tau_sweep_ultra_pareto.csv"))
    lite_rob = read_csv(os.path.join(out_dir, "lite_robustness_windcn.csv"))

    perf_map = {}
    for _, r in matrix2.iterrows():
        perf_map[str(r["model"])] = {
            "clean_R2": float(r["clean_R2"]),
            "r2_sigma060": float(r["R2_sigma_eff_060"]) if not pd.isna(r["R2_sigma_eff_060"]) else np.nan,
        }

    lite_clean_r2 = np.nan
    lite_sigma060_r2 = np.nan
    if not lite_rob.empty:
        rc = find_row(lite_rob, model="Ultra-LSNT-Lite(opt)", scenario="clean")
        rg = find_row(lite_rob, model="Ultra-LSNT-Lite(opt)", scenario="gaussian_60pct")
        if rc is not None and "R2" in rc.index and not pd.isna(rc["R2"]):
            lite_clean_r2 = float(rc["R2"])
        if rg is not None and "R2" in rg.index and not pd.isna(rg["R2"]):
            lite_sigma060_r2 = float(rg["R2"])

    rows: List[Dict] = []
    hardware_models = [
        "Ultra-LSNT(full)",
        "Ultra-LSNT-Lite(opt)",
        "Mamba (auditable)",
        "DLinear",
        "VMD-PSO-LSTM",
        "CEEMDAN-WOA-GRU",
    ]
    for m in hardware_models:
        r = find_row(m3_old, panel="hardware", model_or_setting=m)
        if r is None:
            continue
        key = "Ultra-LSNT" if "Ultra-LSNT(full)" in m else m
        clean_r2 = perf_map.get(key, {}).get("clean_R2", np.nan)
        sigma_r2 = perf_map.get(key, {}).get("r2_sigma060", np.nan)
        if m == "Ultra-LSNT-Lite(opt)":
            if pd.isna(clean_r2):
                clean_r2 = lite_clean_r2
            if pd.isna(sigma_r2):
                sigma_r2 = lite_sigma060_r2
        rows.append(
            {
                "panel": "A_hardware",
                "item": m,
                "latency_ms": float(r["latency_ms"]),
                "throughput_hz": float(r["throughput_hz"]),
                "p95_ms": float(r["p95_ms"]) if not pd.isna(r["p95_ms"]) else np.nan,
                "model_size_mib": np.nan,
                "clean_R2": clean_r2,
                "R2_sigma_eff_060": sigma_r2,
                "safety_or_control": "",
                "run_id": run_id,
            }
        )

    for df, name in [(arima_lat, "ARIMA (auditable)"), (svr_lat, "SVR (auditable)"), (ssa_lat, "SSA-ELM")]:
        if df.empty:
            continue
        r = df.iloc[0]
        rows.append(
            {
                "panel": "A_hardware",
                "item": name,
                "latency_ms": float(r["latency_mean_ms"]),
                "throughput_hz": float(r["throughput_hz"]) if "throughput_hz" in r.index else float(1000.0 / max(float(r["latency_mean_ms"]), 1e-12)),
                "p95_ms": float(r["latency_p95_ms"]),
                "model_size_mib": float(r["model_size_mib"]) if "model_size_mib" in r.index else np.nan,
                "clean_R2": perf_map.get(name, {}).get("clean_R2", np.nan),
                "R2_sigma_eff_060": perf_map.get(name, {}).get("r2_sigma060", np.nan),
                "safety_or_control": "",
                "run_id": run_id,
            }
        )

    guardrail_added = 0
    if not physics_cmp.empty:
        guardrail_models = [
            "Ultra-LSNT (no guardrail)",
            "Ultra-LSNT + clip-only (posthoc)",
            "FG-MoE (full-trained)",
        ]
        model_col = "model" if "model" in physics_cmp.columns else None
        if model_col is not None:
            for mname in guardrail_models:
                sub = physics_cmp[physics_cmp["model"].astype(str) == mname].copy()
                if sub.empty:
                    continue
                clean = sub[sub.get("scenario", pd.Series([""] * len(sub))).astype(str) == "clean"]
                clean_row = clean.iloc[0] if not clean.empty else sub.iloc[0]

                gauss = sub[sub.get("fault", pd.Series([""] * len(sub))).astype(str) == "gaussian"]
                if gauss.empty:
                    gauss = sub[sub.get("scenario", pd.Series([""] * len(sub))).astype(str) == "gaussian_60pct"]
                stress = sub[sub.get("scenario", pd.Series([""] * len(sub))).astype(str) != "clean"]
                if stress.empty:
                    stress = gauss
                if stress.empty:
                    stress_row = clean_row
                else:
                    key_col = "envelope_deviation_area_mean_kw"
                    if key_col in stress.columns:
                        stress_row = stress.sort_values(key_col, ascending=False).iloc[0]
                    else:
                        stress_row = stress.iloc[0]
                gauss_row = gauss.iloc[0] if not gauss.empty else stress_row

                over = _safe_float(stress_row.get("over_rated_rate_pct", np.nan), np.nan)
                under = _safe_float(stress_row.get("under_rated_low_output_rate_pct", np.nan), np.nan)
                ramp = _safe_float(stress_row.get("ramp_rate_violation_pct", np.nan), np.nan)
                env = _safe_float(stress_row.get("envelope_deviation_area_mean_kw", np.nan), np.nan)
                rows.append(
                    {
                        "panel": "B_safety_ablation",
                        "item": mname,
                        "latency_ms": np.nan,
                        "throughput_hz": np.nan,
                        "p95_ms": np.nan,
                        "model_size_mib": np.nan,
                        "clean_R2": _safe_float(clean_row.get("R2", np.nan), np.nan),
                        "R2_sigma_eff_060": _safe_float(gauss_row.get("R2", np.nan), np.nan),
                        "safety_or_control": f"Over-rated {over:.4f}% | Under-rated {under:.4f}% | Ramp-viol {ramp:.4f}% | Env-area {env:.3f} kW",
                        "under_rated_rate_pct": under,
                        "ramp_rate_violation_pct": ramp,
                        "envelope_deviation_area_mean_kw": env,
                        "source": "physics_closure.csv",
                        "run_id": run_id,
                    }
                )
                guardrail_added += 1

    if guardrail_added == 0 and not clip_cmp.empty:
        guardrail_models = [
            "Ultra-LSNT (no guardrail)",
            "Ultra-LSNT + clip-only (posthoc)",
            "FG-MoE (full-trained)",
        ]
        for mname in guardrail_models:
            c = find_row(clip_cmp, scenario="clean", model=mname)
            n = find_row(clip_cmp, scenario="gaussian_60pct", model=mname)
            if c is None and n is None:
                continue
            c_use = c if c is not None else n
            n_use = n if n is not None else c
            rows.append(
                {
                    "panel": "B_safety_ablation",
                    "item": mname,
                    "latency_ms": np.nan,
                    "throughput_hz": np.nan,
                    "p95_ms": np.nan,
                    "model_size_mib": np.nan,
                    "clean_R2": float(c_use["R2"]) if c_use is not None and "R2" in c_use.index else np.nan,
                    "R2_sigma_eff_060": float(n_use["R2"]) if n_use is not None and "R2" in n_use.index else np.nan,
                    "safety_or_control": f"Over-rated {float(n_use['over_rated_rate_pct']) if n_use is not None and 'over_rated_rate_pct' in n_use.index else np.nan:.4f}%",
                    "run_id": run_id,
                }
            )
            guardrail_added += 1

    if guardrail_added == 0 and not pi_cmp.empty:
        print("[warn] clip-only guardrail CSV missing or incomplete; falling back to legacy two-row guardrail ablation.")
        c1 = find_row(pi_cmp, scenario="clean", model="Ultra-LSNT")
        c2 = find_row(pi_cmp, scenario="clean", model="PI-MoE (full-trained)")
        n1 = find_row(pi_cmp, scenario="gaussian_60pct", model="Ultra-LSNT")
        n2 = find_row(pi_cmp, scenario="gaussian_60pct", model="PI-MoE (full-trained)")
        if c1 is not None:
            rows.append(
                {
                    "panel": "B_safety_ablation",
                    "item": "Ultra-LSNT (no guardrail)",
                    "latency_ms": np.nan,
                    "throughput_hz": np.nan,
                    "p95_ms": np.nan,
                    "model_size_mib": np.nan,
                    "clean_R2": float(c1["R2"]),
                    "R2_sigma_eff_060": float(n1["R2"]) if n1 is not None else np.nan,
                    "safety_or_control": f"Over-rated {float(n1['over_rated_rate_pct']) if n1 is not None else float(c1['over_rated_rate_pct']):.4f}%",
                    "run_id": run_id,
                }
            )
        if c2 is not None:
            rows.append(
                {
                    "panel": "B_safety_ablation",
                    "item": "FG-MoE (full-trained)",
                    "latency_ms": np.nan,
                    "throughput_hz": np.nan,
                    "p95_ms": np.nan,
                    "model_size_mib": np.nan,
                    "clean_R2": float(c2["R2"]),
                    "R2_sigma_eff_060": float(n2["R2"]) if n2 is not None else np.nan,
                    "safety_or_control": f"Over-rated {float(n2['over_rated_rate_pct']) if n2 is not None else float(c2['over_rated_rate_pct']):.4f}%",
                    "run_id": run_id,
                }
            )

    if not tau.empty:
        for tau_v in [0.3, 0.5]:
            r = find_row(tau, tau=tau_v)
            if r is None:
                continue
            rows.append(
                {
                    "panel": "B_safety_ablation",
                    "item": f"Jump Gate tau={tau_v:.1f}",
                    "latency_ms": float(r["cpu_batch1_latency_mean_ms"]),
                    "throughput_hz": float(1000.0 / max(float(r["cpu_batch1_latency_mean_ms"]), 1e-12)),
                    "p95_ms": np.nan,
                    "model_size_mib": np.nan,
                    "clean_R2": float(r["R2"]),
                    "R2_sigma_eff_060": np.nan,
                    "safety_or_control": f"Skip {float(r['skip_rate']) * 100.0:.2f}%",
                    "run_id": run_id,
                }
            )

    if not soa.empty:
        # Pick one representative point per K at lambda2=0.01 and highest R2.
        rep = []
        for k in [1, 2, 4]:
            sub = soa[(soa["K"] == k) & (np.isclose(soa["lambda2"], 0.01))]
            if sub.empty:
                continue
            rep.append(sub.sort_values("R2", ascending=False).iloc[0])
        # Add one safest frugal point.
        safe = soa[np.isclose(soa["over_rated_pct"], 0.0)]
        if not safe.empty:
            rep.append(safe.sort_values("latency_ms", ascending=True).iloc[0])
        seen = set()
        for r in rep:
            key = (int(r["K"]), float(r["tau"]), float(r["lambda2"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "panel": "C_joint_param",
                    "item": f"K={int(r['K'])}, tau={float(r['tau']):.1f}, lambda2={float(r['lambda2']):.2f}",
                    "latency_ms": float(r["latency_ms"]),
                    "throughput_hz": float(1000.0 / max(float(r["latency_ms"]), 1e-12)),
                    "p95_ms": np.nan,
                    "model_size_mib": np.nan,
                    "clean_R2": np.nan,
                    "R2_sigma_eff_060": float(r["R2"]),
                    "safety_or_control": f"Over-rated {float(r['over_rated_pct']):.4f}%",
                    "run_id": run_id,
                }
            )

    # backfill model size from core4 if available
    size_map = {}
    if not core4.empty:
        for _, r in core4.iterrows():
            size_map[str(r["model"])] = float(r["model_size_mib"]) if not pd.isna(r["model_size_mib"]) else np.nan
    for row in rows:
        if row["panel"] != "A_hardware":
            continue
        if pd.isna(row["model_size_mib"]):
            row["model_size_mib"] = size_map.get(row["item"], row["model_size_mib"])

    return pd.DataFrame(rows)


def build_matrix4(out_dir: str, run_id: str) -> pd.DataFrame:
    disp = read_csv(os.path.join(out_dir, "dispatch_rolling_with_ultra_scaled_summary.csv"))
    rts260 = read_csv(os.path.join(out_dir, "dispatch_network_ieee24_dcopf_wmax260_aggregate.csv"))
    dispatch_closure_agg = read_csv(os.path.join(out_dir, "dispatch_closure_mapping_decision_aggregate.csv"))
    dispatch_closure_rank = read_csv(os.path.join(out_dir, "dispatch_closure_mapping_decision_ranking_stability.csv"))
    rows: List[Dict] = []

    if not disp.empty:
        baseline = find_row(disp, model="DLinear")
        for m in ["iTransformer", "TimeMixer", "Ultra-LSNT", "DLinear"]:
            r = find_row(disp, model=m)
            if r is None:
                continue
            total_cost = float(r["total_cost_cny_mean"]) * float(r["days"])
            total_curt = float(r["total_curtailment_kwh_mean"]) * float(r["days"])
            fallback = 100.0 * (1.0 - float(r["solver_optimized_ratio"]))
            improve = np.nan
            if baseline is not None and m != "DLinear":
                bcost = float(baseline["total_cost_cny_mean"]) * float(baseline["days"])
                improve = 100.0 * (bcost - total_cost) / (abs(bcost) + 1e-12)
            rows.append(
                {
                    "layer": "L1_CopperPlate_120d",
                    "model": m,
                    "operating_cost_million": to_million(total_cost),
                    "curtailment_million": to_million(total_curt),
                    "fallback_pct": fallback,
                    "improvement_vs_dlinear_pct": improve,
                    "run_id": run_id,
                }
            )

    if not rts260.empty and dispatch_closure_agg.empty:
        baseline = find_row(rts260, model="DLinear")
        for m in ["iTransformer", "TimeMixer", "Ultra-LSNT", "DLinear"]:
            r = find_row(rts260, model=m)
            if r is None:
                continue
            improve = np.nan
            if baseline is not None and m != "DLinear":
                bcurt = float(baseline["total_curtailment_mwh"])
                improve = 100.0 * (bcurt - float(r["total_curtailment_mwh"])) / (abs(bcurt) + 1e-12)
            rows.append(
                {
                    "layer": "L2_RTS24_DCOPF_wmax260",
                    "model": m,
                    "operating_cost_million": to_million(float(r["total_rt_cost"])),
                    "curtailment_mwh": float(r["total_curtailment_mwh"]),
                    "opf_success_pct": 100.0 * float(r["opf_success_ratio"]),
                    "curtailment_improve_vs_dlinear_pct": improve,
                    "run_id": run_id,
                }
            )

    if not dispatch_closure_agg.empty:
        rank_summary = find_row(dispatch_closure_rank, record_type="summary") if not dispatch_closure_rank.empty else None
        rank_spearman_mean = _safe_float(rank_summary["spearman_rank_corr"], np.nan) if rank_summary is not None and "spearman_rank_corr" in rank_summary.index else np.nan
        rank_top1_consistency = _safe_float(rank_summary["top1_consistency"], np.nan) if rank_summary is not None and "top1_consistency" in rank_summary.index else np.nan

        use_models = ["iTransformer", "TimeMixer", "Ultra-LSNT", "DLinear"]
        for mapping in sorted(dispatch_closure_agg["mapping_method"].astype(str).unique().tolist()):
            for subset in ["all", "extreme_ramp", "high_risk"]:
                sub = dispatch_closure_agg[
                    (dispatch_closure_agg["mapping_method"].astype(str) == mapping)
                    & (dispatch_closure_agg["subset"].astype(str) == subset)
                ].copy()
                if sub.empty:
                    continue
                baseline = find_row(sub, model="DLinear")
                for m in use_models:
                    r = find_row(sub, model=m)
                    if r is None:
                        continue
                    improve = np.nan
                    if baseline is not None and m != "DLinear":
                        bcost = _safe_float(baseline.get("total_rt_cost", np.nan), np.nan)
                        cost = _safe_float(r.get("total_rt_cost", np.nan), np.nan)
                        if np.isfinite(bcost) and np.isfinite(cost):
                            improve = 100.0 * (bcost - cost) / (abs(bcost) + 1e-12)
                    layer = (
                        "L2_RTS24_DCOPF_wmax260"
                        if mapping == "affine" and subset == "all"
                        else f"L2_RTS24_DCOPF_{mapping}_{subset}"
                    )
                    rows.append(
                        {
                            "layer": layer,
                            "model": m,
                            "operating_cost_million": to_million(_safe_float(r.get("total_rt_cost", np.nan), np.nan)),
                            "curtailment_mwh": _safe_float(r.get("total_curtailment_mwh", np.nan), np.nan),
                            "opf_success_pct": 100.0 * (1.0 - _safe_float(r.get("feasibility_fail_rate", np.nan), np.nan))
                            if np.isfinite(_safe_float(r.get("feasibility_fail_rate", np.nan), np.nan))
                            else np.nan,
                            "curtailment_improve_vs_dlinear_pct": np.nan,
                            "improvement_vs_dlinear_pct": improve,
                            "feasibility_fail_pct": 100.0 * _safe_float(r.get("feasibility_fail_rate", np.nan), np.nan)
                            if np.isfinite(_safe_float(r.get("feasibility_fail_rate", np.nan), np.nan))
                            else np.nan,
                            "backfire_rate_pct": 100.0 * _safe_float(r.get("backfire_rate_vs_dlinear", np.nan), np.nan)
                            if np.isfinite(_safe_float(r.get("backfire_rate_vs_dlinear", np.nan), np.nan))
                            else np.nan,
                            "mapping_method": mapping,
                            "risk_subset": subset,
                            "ranking_spearman_mean": rank_spearman_mean,
                            "top1_consistency": rank_top1_consistency,
                            "source": "dispatch_closure_mapping_decision_aggregate.csv",
                            "run_id": run_id,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 4 main matrices from auditable evidence")
    ap.add_argument("--out_dir", type=str, default=os.path.join("results", "supplementary_evidence"))
    ap.add_argument("--audit_tag", type=str, default="main4_matrix_build")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.audit_tag

    m1 = build_matrix1(args.out_dir, run_id)
    m2 = build_matrix2(args.out_dir, run_id)
    m3 = build_matrix3(args.out_dir, run_id, m2)
    m4 = build_matrix4(args.out_dir, run_id)

    p1 = os.path.join(args.out_dir, "main4_matrix1_protocol.csv")
    p2 = os.path.join(args.out_dir, "main4_matrix2_perf_robust_transfer.csv")
    p3 = os.path.join(args.out_dir, "main4_matrix3_hardware_safety_joint.csv")
    p4 = os.path.join(args.out_dir, "main4_matrix4_dispatch_value.csv")
    m1.to_csv(p1, index=False)
    m2.to_csv(p2, index=False)
    m3.to_csv(p3, index=False)
    m4.to_csv(p4, index=False)

    manifest = pd.DataFrame(
        [
            {"matrix_id": "matrix1_protocol", "file": "main4_matrix1_protocol.csv", "rows": len(m1), "generated_run_id": run_id},
            {"matrix_id": "matrix2_perf_robust_transfer", "file": "main4_matrix2_perf_robust_transfer.csv", "rows": len(m2), "generated_run_id": run_id},
            {"matrix_id": "matrix3_hardware_safety_joint", "file": "main4_matrix3_hardware_safety_joint.csv", "rows": len(m3), "generated_run_id": run_id},
            {"matrix_id": "matrix4_dispatch_value", "file": "main4_matrix4_dispatch_value.csv", "rows": len(m4), "generated_run_id": run_id},
        ]
    )
    manifest.to_csv(os.path.join(args.out_dir, "main4_matrix_manifest.csv"), index=False)

    print("Built main4 matrix evidence.")
    print("run_id:", run_id)
    print(p1)
    print(p2)
    print(p3)
    print(p4)


if __name__ == "__main__":
    main()
