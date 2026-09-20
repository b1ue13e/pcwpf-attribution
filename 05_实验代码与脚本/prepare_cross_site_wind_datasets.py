#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "04_数据集"
OPEN_ROOT = DATA_ROOT / "open_cross_site_wind"
PREPARED_DIR = DATA_ROOT / "prepared_priority_datasets"
AUDIT_DIR = PROJECT_ROOT / "06_实验结果与日志" / "results" / "cross_site_wind"

CORE_COLUMNS = ["datetime", "wind_speed", "wind_direction", "temperature", "power"]


@dataclass
class SiteMeta:
    alias: str
    display_name: str
    country: str
    climate_terrain_label: str
    source: str
    doi: str
    license: str


SITE_META = {
    "wind_cn_core_15min": SiteMeta(
        alias="wind_cn_core_15min",
        display_name="Wind CN",
        country="China",
        climate_terrain_label="arid_or_semi_arid_inland",
        source="local Xinjiang wind farm dataset",
        doi="",
        license="proprietary/anonymized",
    ),
    "wind_us_core_15min": SiteMeta(
        alias="wind_us_core_15min",
        display_name="Wind US",
        country="United States",
        climate_terrain_label="external_us_wind_site",
        source="local US wind 2012 dataset",
        doi="",
        license="open/local",
    ),
    "kelmarsh_uk_2016": SiteMeta(
        alias="kelmarsh_uk_2016",
        display_name="Kelmarsh UK",
        country="UK",
        climate_terrain_label="temperate_onshore",
        source="Kelmarsh wind farm data",
        doi="10.5281/zenodo.5841834",
        license="CC-BY-4.0",
    ),
    "penmanshiel_scotland_2016": SiteMeta(
        alias="penmanshiel_scotland_2016",
        display_name="Penmanshiel Scotland",
        country="UK/Scotland",
        climate_terrain_label="high_latitude_onshore",
        source="Penmanshiel Wind Farm Data",
        doi="10.5281/zenodo.5946808",
        license="CC-BY-4.0",
    ),
    "smarteole_france_2020": SiteMeta(
        alias="smarteole_france_2020",
        display_name="SMARTEOLE France",
        country="France",
        climate_terrain_label="temperate_wind_farm_control_site",
        source="SMARTEOLE Wind Farm Control open dataset",
        doi="10.5281/zenodo.7342466",
        license="Etalab-2.0",
    ),
}


def read_csv_flexible(path_or_file, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path_or_file, encoding=encoding, **kwargs)
        except Exception as exc:
            last_error = exc
            try:
                path_or_file.seek(0)
            except Exception:
                pass
    raise RuntimeError(f"failed to read CSV: {last_error}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def circular_mean_deg(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return pd.Series(np.nan, index=frame.index)
    values = frame[cols].apply(pd.to_numeric, errors="coerce")
    radians = np.deg2rad(values)
    sin_mean = np.nanmean(np.sin(radians), axis=1)
    cos_mean = np.nanmean(np.cos(radians), axis=1)
    out = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0
    invalid = np.isnan(sin_mean) | np.isnan(cos_mean)
    out[invalid] = np.nan
    return pd.Series(out, index=frame.index)


def circular_resample_deg(series: pd.Series, rule: str = "15min") -> pd.Series:
    radians = np.deg2rad(pd.to_numeric(series, errors="coerce"))
    sin_mean = np.sin(radians).resample(rule).mean()
    cos_mean = np.cos(radians).resample(rule).mean()
    out = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0
    out[(sin_mean.isna()) | (cos_mean.isna())] = np.nan
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["datetime"], errors="coerce")
    hours = ts.dt.hour.fillna(0) + ts.dt.minute.fillna(0) / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    out["dayofweek_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek.fillna(0) / 7.0)
    out["dayofweek_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek.fillna(0) / 7.0)
    return out


def finalize_core_frame(df: pd.DataFrame, *, alias: str, source_frequency_minutes: int | None) -> tuple[pd.DataFrame, dict]:
    out = df[CORE_COLUMNS].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime")
    for col in CORE_COLUMNS[1:]:
        out[col] = numeric(out[col])
    out.loc[out["power"] < 0, "power"] = np.nan
    raw_rows = int(len(out))
    raw_missing = {col: float(out[col].isna().mean()) for col in CORE_COLUMNS[1:]}

    out = out.set_index("datetime")
    resampled = pd.DataFrame(index=out.resample("15min").mean().index)
    resampled["wind_speed"] = out["wind_speed"].resample("15min").mean()
    resampled["wind_direction"] = circular_resample_deg(out["wind_direction"], "15min")
    resampled["temperature"] = out["temperature"].resample("15min").mean()
    resampled["power"] = out["power"].resample("15min").mean()
    resampled = resampled.reset_index()

    missing_before_interp = {col: float(resampled[col].isna().mean()) for col in CORE_COLUMNS[1:]}
    resampled = resampled.dropna(subset=["power"])
    for col in ["wind_speed", "wind_direction", "temperature", "power"]:
        resampled[col] = numeric(resampled[col])
    resampled[["wind_speed", "wind_direction", "temperature", "power"]] = (
        resampled[["wind_speed", "wind_direction", "temperature", "power"]]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    all_missing_imputed = []
    for col in ["wind_speed", "wind_direction", "temperature"]:
        if resampled[col].isna().all():
            resampled[col] = 0.0
            all_missing_imputed.append(col)
    resampled = resampled.dropna(subset=CORE_COLUMNS)
    resampled = add_time_features(resampled)
    ordered = ["datetime", "wind_speed", "wind_direction", "temperature", "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "power"]
    resampled = resampled[ordered]

    if len(resampled) > 1:
        diffs = pd.to_datetime(resampled["datetime"]).diff().dropna().dt.total_seconds() / 60.0
        dominant_freq = float(diffs.mode().iloc[0]) if not diffs.empty else math.nan
        freq_15_ratio = float((diffs == 15).mean()) if not diffs.empty else math.nan
    else:
        dominant_freq = math.nan
        freq_15_ratio = math.nan

    meta = {
        **asdict(SITE_META[alias]),
        "schema": "core_wind_15min",
        "core_columns": CORE_COLUMNS,
        "source_frequency_minutes": source_frequency_minutes,
        "prepared_frequency_minutes": 15,
        "raw_rows": raw_rows,
        "rows": int(len(resampled)),
        "cols": int(len(resampled.columns)),
        "start": str(resampled["datetime"].iloc[0]) if len(resampled) else "",
        "end": str(resampled["datetime"].iloc[-1]) if len(resampled) else "",
        "raw_missing_ratio": raw_missing,
        "resampled_missing_ratio_before_interp": missing_before_interp,
        "all_missing_columns_imputed_to_zero": all_missing_imputed,
        "dominant_freq_minutes": dominant_freq,
        "freq_15min_ratio": freq_15_ratio,
        "power_min": float(resampled["power"].min()) if len(resampled) else math.nan,
        "power_max": float(resampled["power"].max()) if len(resampled) else math.nan,
        "power_mean": float(resampled["power"].mean()) if len(resampled) else math.nan,
        "columns": list(resampled.columns),
    }
    return resampled, meta


def write_prepared(alias: str, df: pd.DataFrame, meta: dict) -> None:
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PREPARED_DIR / f"{alias}.csv"
    json_path = PREPARED_DIR / f"{alias}.json"
    meta = {
        **meta,
        "prepared_path_abs": str(csv_path),
    }
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def build_wind_cn_core() -> tuple[pd.DataFrame, dict]:
    path = PREPARED_DIR / "xinjiang_wind_2019.csv"
    df = read_csv_flexible(path)
    out = pd.DataFrame(
        {
            "datetime": df["datetime"],
            "wind_speed": df["hub_wind_speed"] if "hub_wind_speed" in df.columns else df.filter(like="wind_speed").mean(axis=1),
            "wind_direction": df["hub_wind_direction"] if "hub_wind_direction" in df.columns else circular_mean_deg(df, [c for c in df.columns if "wind_direction" in c]),
            "temperature": df["temperature"],
            "power": df["power"],
        }
    )
    return finalize_core_frame(out, alias="wind_cn_core_15min", source_frequency_minutes=15)


def build_wind_us_core() -> tuple[pd.DataFrame, dict]:
    path = PREPARED_DIR / "us_wind_2012.csv"
    df = read_csv_flexible(path)
    speed_col = "wind_speed_100m" if "wind_speed_100m" in df.columns else "wind_speed"
    out = pd.DataFrame(
        {
            "datetime": df["datetime"],
            "wind_speed": df[speed_col],
            "wind_direction": np.nan,
            "temperature": np.nan,
            "power": df["power"],
        }
    )
    return finalize_core_frame(out, alias="wind_us_core_15min", source_frequency_minutes=15)


def list_zip_members(zip_path: Path, prefix: str) -> List[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return sorted([name for name in zf.namelist() if name.startswith(prefix) and name.lower().endswith(".csv")])


def read_greenbyte_member(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            lines = fh.read().decode("utf-8-sig", errors="replace").splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("# Date and time,"):
            header_idx = idx
            break
    if header_idx is None:
        raise RuntimeError(f"Could not find Greenbyte header in {member}")
    from io import StringIO

    return read_csv_flexible(StringIO("\n".join(lines[header_idx:])))


def aggregate_greenbyte_site(zip_paths: List[Path], alias: str, source_frequency_minutes: int = 10) -> tuple[pd.DataFrame, dict]:
    frames: List[pd.DataFrame] = []
    turbine_count = 0
    for zip_path in zip_paths:
        members = list_zip_members(zip_path, "Turbine_Data_")
        for member in members:
            raw = read_greenbyte_member(zip_path, member)
            required = ["# Date and time", "Wind speed (m/s)", "Wind direction (°)", "Power (kW)"]
            if not all(col in raw.columns for col in required):
                continue
            temp_col = "Nacelle ambient temperature (°C)" if "Nacelle ambient temperature (°C)" in raw.columns else "Nacelle temperature (°C)"
            frame = pd.DataFrame(
                {
                    "datetime": raw["# Date and time"],
                    "wind_speed": raw["Wind speed (m/s)"],
                    "wind_direction": raw["Wind direction (°)"],
                    "temperature": raw[temp_col] if temp_col in raw.columns else np.nan,
                    "power": raw["Power (kW)"],
                }
            )
            frame["turbine_member"] = member
            frames.append(frame)
            turbine_count += 1
    if not frames:
        raise RuntimeError(f"No Greenbyte turbine data found for {alias}")
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["datetime"] = pd.to_datetime(all_rows["datetime"], errors="coerce")
    for col in ["wind_speed", "wind_direction", "temperature", "power"]:
        all_rows[col] = numeric(all_rows[col])
    direction_rad = np.deg2rad(all_rows["wind_direction"])
    all_rows["_direction_sin"] = np.sin(direction_rad)
    all_rows["_direction_cos"] = np.cos(direction_rad)
    grouped = all_rows.groupby("datetime", sort=True)
    sin_mean = grouped["_direction_sin"].mean()
    cos_mean = grouped["_direction_cos"].mean()
    direction_mean = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0
    direction_mean[(sin_mean.isna()) | (cos_mean.isna())] = np.nan
    site = pd.DataFrame(
        {
            "datetime": grouped.size().index,
            "wind_speed": grouped["wind_speed"].mean().values,
            "wind_direction": direction_mean.reindex(grouped.size().index).values,
            "temperature": grouped["temperature"].mean().values,
            "power": grouped["power"].sum(min_count=1).values,
        }
    )
    prepared, meta = finalize_core_frame(site, alias=alias, source_frequency_minutes=source_frequency_minutes)
    meta["turbine_files_used"] = turbine_count
    meta["raw_zip_files"] = [str(path) for path in zip_paths]
    return prepared, meta


def build_kelmarsh() -> tuple[pd.DataFrame, dict]:
    return aggregate_greenbyte_site(
        [OPEN_ROOT / "kelmarsh_uk" / "raw" / "Kelmarsh_SCADA_2016_3082.zip"],
        alias="kelmarsh_uk_2016",
    )


def build_penmanshiel() -> tuple[pd.DataFrame, dict]:
    return aggregate_greenbyte_site(
        [
            OPEN_ROOT / "penmanshiel_scotland" / "raw" / "Penmanshiel_SCADA_2016_WT01-10_3107.zip",
            OPEN_ROOT / "penmanshiel_scotland" / "raw" / "Penmanshiel_SCADA_2016_WT11-15_3107.zip",
        ],
        alias="penmanshiel_scotland_2016",
    )


def build_smarteole() -> tuple[pd.DataFrame, dict]:
    zip_path = OPEN_ROOT / "smarteole_france" / "raw" / "SMARTEOLE-WFC-open-dataset.zip"
    member = "SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_SCADA_1minData.csv"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            raw = read_csv_flexible(fh)
    power_cols = [c for c in raw.columns if c.startswith("active_power_") and c.endswith("_avg")]
    speed_cols = [c for c in raw.columns if c.startswith("wind_speed_") and c.endswith("_avg")]
    temp_cols = [c for c in raw.columns if c.startswith("temperature_") and c.endswith("_avg")]
    direction_cols = [c for c in raw.columns if c.startswith("wind_direction_") and c.endswith("_avg")]
    if not power_cols or not speed_cols:
        raise RuntimeError("SMARTEOLE SCADA file is missing expected power or wind speed columns")
    out = pd.DataFrame(
        {
            "datetime": raw["time"],
            "wind_speed": raw[speed_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1),
            "wind_direction": circular_mean_deg(raw, direction_cols),
            "temperature": raw[temp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) if temp_cols else np.nan,
            "power": raw[power_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1),
        }
    )
    prepared, meta = finalize_core_frame(out, alias="smarteole_france_2020", source_frequency_minutes=1)
    meta["raw_zip_files"] = [str(zip_path)]
    meta["power_columns_used"] = power_cols
    meta["wind_speed_columns_used"] = speed_cols
    meta["temperature_columns_used"] = temp_cols
    meta["wind_direction_columns_used"] = direction_cols
    return prepared, meta


BUILDERS = {
    "wind_cn_core_15min": build_wind_cn_core,
    "wind_us_core_15min": build_wind_us_core,
    "kelmarsh_uk_2016": build_kelmarsh,
    "penmanshiel_scotland_2016": build_penmanshiel,
    "smarteole_france_2020": build_smarteole,
}


def build_audit(rows: List[dict]) -> pd.DataFrame:
    audit_rows = []
    for meta in rows:
        audit_rows.append(
            {
                "alias": meta["alias"],
                "display_name": meta["display_name"],
                "country": meta["country"],
                "climate_terrain_label": meta["climate_terrain_label"],
                "rows": meta["rows"],
                "start": meta["start"],
                "end": meta["end"],
                "dominant_freq_minutes": meta["dominant_freq_minutes"],
                "freq_15min_ratio": meta["freq_15min_ratio"],
                "power_min": meta["power_min"],
                "power_max": meta["power_max"],
                "power_mean": meta["power_mean"],
                "wind_speed_missing_before_interp": meta["resampled_missing_ratio_before_interp"].get("wind_speed", np.nan),
                "wind_direction_missing_before_interp": meta["resampled_missing_ratio_before_interp"].get("wind_direction", np.nan),
                "temperature_missing_before_interp": meta["resampled_missing_ratio_before_interp"].get("temperature", np.nan),
                "power_missing_before_interp": meta["resampled_missing_ratio_before_interp"].get("power", np.nan),
                "doi": meta["doi"],
                "license": meta["license"],
                "prepared_path_abs": meta["prepared_path_abs"],
            }
        )
    return pd.DataFrame(audit_rows).sort_values("alias")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare unified 15-minute cross-site wind datasets.")
    parser.add_argument("--datasets", nargs="*", default=list(BUILDERS.keys()), choices=list(BUILDERS.keys()))
    parser.add_argument("--audit-out", type=str, default=str(AUDIT_DIR / "cross_site_dataset_audit.csv"))
    return parser.parse_args()


def load_all_available_metadata() -> List[dict]:
    metas: List[dict] = []
    for alias in BUILDERS:
        meta_path = PREPARED_DIR / f"{alias}.json"
        if meta_path.exists():
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
    return metas


def main() -> int:
    args = parse_args()
    metas: List[dict] = []
    for alias in args.datasets:
        print(f"[prepare] {alias}")
        df, meta = BUILDERS[alias]()
        write_prepared(alias, df, meta)
        meta = json.loads((PREPARED_DIR / f"{alias}.json").read_text(encoding="utf-8"))
        metas.append(meta)
        print(f"  rows={len(df)} path={PREPARED_DIR / (alias + '.csv')}")
    all_metas = load_all_available_metadata()
    if all_metas:
        audit_path = Path(args.audit_out)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        build_audit(all_metas).to_csv(audit_path, index=False, encoding="utf-8-sig")
        print(f"[audit] {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
