from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent if (ROOT.parent / "04_数据集").exists() else ROOT
DATA_ROOT = PROJECT_ROOT / "04_数据集" if (PROJECT_ROOT / "04_数据集").exists() else ROOT
PREPARED_DIR = DATA_ROOT / "prepared_priority_datasets"


def _resolve_source_path(source_path: str) -> Path:
    path = Path(source_path)
    if path.is_absolute():
        return path
    candidates = [
        ROOT / source_path,
        DATA_ROOT / source_path,
        PROJECT_ROOT / source_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass
class PriorityDatasetSpec:
    alias: str
    source_path: str
    default_target: str
    role: str
    description: str
    builder_name: str
    prepared_freq_minutes: int


def _read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read CSV {path}: {last_error}")


def _add_time_features(df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    ts = pd.to_datetime(df[time_col], errors="coerce")
    out = df.copy()
    hours = ts.dt.hour.fillna(0) + ts.dt.minute.fillna(0) / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    out["dayofweek_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek.fillna(0) / 7.0)
    out["dayofweek_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek.fillna(0) / 7.0)
    return out


def _finalize_numeric_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    keep_cols: List[str] = []
    for col in out.columns:
        if col == target or col == "datetime":
            keep_cols.append(col)
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        if float(numeric.notna().mean()) >= 0.5:
            out[col] = numeric
            keep_cols.append(col)
    out = out[keep_cols]
    out[target] = pd.to_numeric(out[target], errors="coerce")
    out = out.sort_values("datetime").drop_duplicates("datetime")
    out = out.interpolate(method="linear", limit_direction="both", axis=0)
    out = out.ffill().bfill()
    cols = [c for c in out.columns if c not in ("datetime", target)] + [target]
    out = out[["datetime"] + cols]
    return out


def _build_xinjiang_wind_2019(spec: PriorityDatasetSpec) -> tuple[pd.DataFrame, Dict]:
    path = _resolve_source_path(spec.source_path)
    df = _read_csv_flexible(path)
    rename = {
        "时间": "datetime",
        "测风塔10m风速(m/s)": "wind_speed_10m",
        "测风塔30m风速(m/s)": "wind_speed_30m",
        "测风塔50m风速(m/s)": "wind_speed_50m",
        "测风塔70m风速(m/s)": "wind_speed_70m",
        "轮毂高度风速(m/s)": "hub_wind_speed",
        "测风塔10m风向(°)": "wind_direction_10m",
        "测风塔30m风向(°)": "wind_direction_30m",
        "测风塔50m风向(°)": "wind_direction_50m",
        "测风塔70m风向(°)": "wind_direction_70m",
        "轮毂高度风向(°)": "hub_wind_direction",
        "温度(°)": "temperature",
        "气压(hPa)": "pressure",
        "湿度(%)": "humidity",
        "实际发电功率（mw）": "power",
    }
    out = df.rename(columns=rename)
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = _add_time_features(out, "datetime")
    out = _finalize_numeric_frame(out, target="power")
    meta = {"source_target": "实际发电功率（mw）", "prepared_target": "power"}
    return out, meta


def _choose_turbine_target(df: pd.DataFrame) -> str:
    candidates = ["YD15", "ROUND(A.POWER,0)", "PREPOWER"]
    best = candidates[0]
    best_ratio = -1.0
    for col in candidates:
        if col not in df.columns:
            continue
        ratio = float(pd.to_numeric(df[col], errors="coerce").notna().mean())
        if ratio > best_ratio:
            best = col
            best_ratio = ratio
    return best


def _build_turbine_file(spec: PriorityDatasetSpec) -> tuple[pd.DataFrame, Dict]:
    path = _resolve_source_path(spec.source_path)
    df = _read_csv_flexible(path)
    target_col = _choose_turbine_target(df)
    rename = {
        "DATATIME": "datetime",
        "WINDSPEED": "windspeed",
        "PREPOWER": "prepower",
        "WINDDIRECTION": "winddirection",
        "TEMPERATURE": "temperature",
        "HUMIDITY": "humidity",
        "PRESSURE": "pressure",
        "ROUND(A.WS,1)": "round_ws",
        "ROUND(A.POWER,0)": "round_power",
        "YD15": "yd15",
    }
    out = df.rename(columns=rename)
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", dayfirst=False)
    out = _add_time_features(out, "datetime")
    prepared_target_col = rename.get(target_col, target_col)
    out["power"] = out[prepared_target_col]
    out = _finalize_numeric_frame(out, target="power")
    meta = {"source_target": target_col, "prepared_target": "power"}
    return out, meta


def _build_us_wind_2012_15min(spec: PriorityDatasetSpec) -> tuple[pd.DataFrame, Dict]:
    path = _resolve_source_path(spec.source_path)
    df = _read_csv_flexible(path, skiprows=3)
    for col in df.columns:
        if isinstance(col, str):
            df.rename(columns={col: col.strip()}, inplace=True)
    df["datetime"] = pd.to_datetime(
        {
            "year": pd.to_numeric(df["Year"], errors="coerce"),
            "month": pd.to_numeric(df["Month"], errors="coerce"),
            "day": pd.to_numeric(df["Day"], errors="coerce"),
            "hour": pd.to_numeric(df["Hour"], errors="coerce"),
            "minute": pd.to_numeric(df["Minute"], errors="coerce"),
        },
        errors="coerce",
    )
    out = df[["datetime", "power (MW)", "wind speed at 100m (m/s)"]].rename(
        columns={"power (MW)": "power", "wind speed at 100m (m/s)": "wind_speed_100m"}
    )
    out = out.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime")
    out = out.set_index("datetime").resample("15min").mean().dropna().reset_index()
    out = _add_time_features(out, "datetime")
    out = _finalize_numeric_frame(out, target="power")
    meta = {
        "source_target": "power (MW)",
        "prepared_target": "power",
        "source_frequency_minutes": 5,
        "prepared_frequency_minutes": 15,
    }
    return out, meta


def _build_core_15min_file(spec: PriorityDatasetSpec) -> tuple[pd.DataFrame, Dict]:
    path = _resolve_source_path(spec.source_path)
    df = _read_csv_flexible(path)
    required = ["datetime", "wind_speed", "wind_direction", "temperature", "power"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required core columns: {missing}")
    out = df[required].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime")
    for col in required[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = _add_time_features(out, "datetime")
    out = _finalize_numeric_frame(out, target="power")
    meta = {
        "source_target": "power",
        "prepared_target": "power",
        "schema": "core_wind_15min",
        "core_columns": required,
    }
    return out, meta


BUILDERS: Dict[str, Callable[[PriorityDatasetSpec], tuple[pd.DataFrame, Dict]]] = {
    "xinjiang_wind_2019": _build_xinjiang_wind_2019,
    "wind_turbine_file": _build_turbine_file,
    "us_wind_2012_15min": _build_us_wind_2012_15min,
    "core_15min_file": _build_core_15min_file,
}


def priority_dataset_catalog() -> Dict[str, PriorityDatasetSpec]:
    return {
        "xinjiang_wind_2019": PriorityDatasetSpec(
            alias="xinjiang_wind_2019",
            source_path="（新疆）风电场风速/新疆_风电一场_2019.csv",
            default_target="power",
            role="domestic_external_wind_site",
            description="Xinjiang wind farm 2019 with multi-height wind, meteorology, and power.",
            builder_name="xinjiang_wind_2019",
            prepared_freq_minutes=15,
        ),
        "us_wind_2012": PriorityDatasetSpec(
            alias="us_wind_2012",
            source_path="（美国）风电场数据-2012/55567-2012.csv",
            default_target="power",
            role="international_external_wind_site",
            description="US wind farm 2012 resampled from 5-minute data to 15-minute protocol.",
            builder_name="us_wind_2012_15min",
            prepared_freq_minutes=15,
        ),
        "wind_turbine_primary": PriorityDatasetSpec(
            alias="wind_turbine_primary",
            source_path="风电机组运行数据集/17.csv",
            default_target="power",
            role="observed_fault_bank_and_safety_diagnostics",
            description="Highest-coverage turbine run file with dense YD15 target availability.",
            builder_name="wind_turbine_file",
            prepared_freq_minutes=15,
        ),
        "wind_cn_core_15min": PriorityDatasetSpec(
            alias="wind_cn_core_15min",
            source_path="prepared_priority_datasets/wind_cn_core_15min.csv",
            default_target="power",
            role="cross_site_primary_arid_inland_wind",
            description="Wind CN reduced to the unified five-column 15-minute cross-site schema.",
            builder_name="core_15min_file",
            prepared_freq_minutes=15,
        ),
        "wind_us_core_15min": PriorityDatasetSpec(
            alias="wind_us_core_15min",
            source_path="prepared_priority_datasets/wind_us_core_15min.csv",
            default_target="power",
            role="cross_site_us_external_wind",
            description="Wind US reduced to the unified five-column 15-minute cross-site schema.",
            builder_name="core_15min_file",
            prepared_freq_minutes=15,
        ),
        "kelmarsh_uk_2016": PriorityDatasetSpec(
            alias="kelmarsh_uk_2016",
            source_path="prepared_priority_datasets/kelmarsh_uk_2016.csv",
            default_target="power",
            role="cross_site_uk_temperate_onshore_wind",
            description="Kelmarsh UK 2016 aggregated farm-level 15-minute SCADA in the unified cross-site schema.",
            builder_name="core_15min_file",
            prepared_freq_minutes=15,
        ),
        "penmanshiel_scotland_2016": PriorityDatasetSpec(
            alias="penmanshiel_scotland_2016",
            source_path="prepared_priority_datasets/penmanshiel_scotland_2016.csv",
            default_target="power",
            role="cross_site_scotland_high_latitude_onshore_wind",
            description="Penmanshiel Scotland 2016 aggregated farm-level 15-minute SCADA in the unified cross-site schema.",
            builder_name="core_15min_file",
            prepared_freq_minutes=15,
        ),
        "smarteole_france_2020": PriorityDatasetSpec(
            alias="smarteole_france_2020",
            source_path="prepared_priority_datasets/smarteole_france_2020.csv",
            default_target="power",
            role="cross_site_france_wind_farm_control",
            description="SMARTEOLE France 2020 aggregated farm-level 15-minute SCADA in the unified cross-site schema.",
            builder_name="core_15min_file",
            prepared_freq_minutes=15,
        ),
        "wind_turbine_11": PriorityDatasetSpec(
            alias="wind_turbine_11",
            source_path="风电机组运行数据集/11.csv",
            default_target="power",
            role="turbine_run_training_candidate",
            description="Wind turbine run file 11.",
            builder_name="wind_turbine_file",
            prepared_freq_minutes=15,
        ),
        "wind_turbine_12": PriorityDatasetSpec(
            alias="wind_turbine_12",
            source_path="风电机组运行数据集/12.csv",
            default_target="power",
            role="turbine_run_training_candidate",
            description="Wind turbine run file 12.",
            builder_name="wind_turbine_file",
            prepared_freq_minutes=15,
        ),
        "wind_turbine_13": PriorityDatasetSpec(
            alias="wind_turbine_13",
            source_path="风电机组运行数据集/13.csv",
            default_target="power",
            role="turbine_run_training_candidate",
            description="Wind turbine run file 13.",
            builder_name="wind_turbine_file",
            prepared_freq_minutes=15,
        ),
        "wind_turbine_14": PriorityDatasetSpec(
            alias="wind_turbine_14",
            source_path="风电机组运行数据集/14.csv",
            default_target="power",
            role="turbine_run_training_candidate",
            description="Wind turbine run file 14.",
            builder_name="wind_turbine_file",
            prepared_freq_minutes=15,
        ),
        "wind_turbine_17": PriorityDatasetSpec(
            alias="wind_turbine_17",
            source_path="风电机组运行数据集/17.csv",
            default_target="power",
            role="turbine_run_training_candidate",
            description="Wind turbine run file 17.",
            builder_name="wind_turbine_file",
            prepared_freq_minutes=15,
        ),
    }


def list_priority_datasets() -> List[Dict]:
    return [asdict(spec) for spec in priority_dataset_catalog().values()]


def default_target_for_reference(path_or_alias: str, fallback: str = "power") -> str:
    catalog = priority_dataset_catalog()
    if path_or_alias in catalog:
        return catalog[path_or_alias].default_target
    return fallback


def _write_prepared_dataset(alias: str, df: pd.DataFrame, metadata: Dict) -> Path:
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREPARED_DIR / f"{alias}.csv"
    meta_path = PREPARED_DIR / f"{alias}.json"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _ensure_existing_prepared_metadata(spec: PriorityDatasetSpec, source_path: Path, out_path: Path, meta_path: Path) -> None:
    if meta_path.exists():
        return
    df = _read_csv_flexible(out_path)
    metadata = {
        **asdict(spec),
        "source_path_abs": str(source_path),
        "prepared_path_abs": str(out_path),
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": list(df.columns),
        "prepared_only_bundle": True,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_dataset_reference(path_or_alias: str, target: str = "") -> Dict[str, object]:
    catalog = priority_dataset_catalog()
    if path_or_alias not in catalog:
        path = Path(path_or_alias)
        return {
            "reference": path_or_alias,
            "alias": "",
            "source_path": str(path),
            "resolved_path": str(path),
            "target": target or "power",
            "prepared": False,
            "role": "direct_path",
            "description": "Direct user-provided dataset path.",
        }

    spec = catalog[path_or_alias]
    source_path = _resolve_source_path(spec.source_path)
    out_path = PREPARED_DIR / f"{spec.alias}.csv"
    meta_path = PREPARED_DIR / f"{spec.alias}.json"
    source_exists = source_path.exists()

    if out_path.exists() and not source_exists:
        _ensure_existing_prepared_metadata(spec, source_path, out_path, meta_path)
    elif (not out_path.exists()) and (not source_exists):
        raise FileNotFoundError(
            f"Dataset alias `{spec.alias}` requires either prepared file `{out_path}` "
            f"or source file `{source_path}`, but neither exists."
        )
    elif (not out_path.exists()) or out_path.stat().st_mtime < source_path.stat().st_mtime:
        builder = BUILDERS[spec.builder_name]
        df, extra_meta = builder(spec)
        metadata = {
            **asdict(spec),
            **extra_meta,
            "source_path_abs": str(source_path),
            "prepared_path_abs": str(out_path),
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "columns": list(df.columns),
        }
        _write_prepared_dataset(spec.alias, df, metadata)
    elif not meta_path.exists():
        builder = BUILDERS[spec.builder_name]
        df, extra_meta = builder(spec)
        metadata = {
            **asdict(spec),
            **extra_meta,
            "source_path_abs": str(source_path),
            "prepared_path_abs": str(out_path),
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "columns": list(df.columns),
        }
        _write_prepared_dataset(spec.alias, df, metadata)

    resolved_target = target or spec.default_target
    return {
        "reference": path_or_alias,
        "alias": spec.alias,
        "source_path": str(source_path),
        "resolved_path": str(out_path),
        "target": resolved_target,
        "prepared": True,
        "role": spec.role,
        "description": spec.description,
    }
