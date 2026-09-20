#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "results" / "dataset_audit"
DEFAULT_DATASETS = [
    "风电机组运行数据集",
    "（新疆）风电场风速",
    "（美国）风电场数据-2012",
    "内蒙古风速50425",
    "GEFCom2012年全球能源预测大赛电力负荷预测数据集",
    "负荷预测数据集（一）",
    "电力负荷数据集（二）",
    "电力系统短期负荷预测数据集",
    "全国空气质量监测数据集",
]
SUPPORTED_EXTS = {".csv", ".xlsx", ".xls"}


@dataclass
class FileAudit:
    dataset_name: str
    dataset_path: str
    file_path: str
    relative_path: str
    extension: str
    parse_status: str
    rows: int
    cols: int
    time_col: str
    start_time: str
    end_time: str
    dominant_freq_minutes: float
    regularity_ratio: float
    regular_sampling: bool
    overall_missing_pct: float
    numeric_missing_pct: float
    multivariate: bool
    candidate_target: str
    power_columns: str
    wind_columns: str
    direction_columns: str
    temperature_columns: str
    pressure_columns: str
    humidity_columns: str
    unit_tag_columns: str
    power_ready: bool
    lh_96_24_compatible: bool
    missing_burst_candidate: bool
    flatline_candidate: bool
    spike_candidate: bool
    drift_candidate: bool
    example_columns: str
    notes: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit multiple local datasets and build audit_table.csv.")
    p.add_argument("--root", type=str, default=str(ROOT))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    p.add_argument("--max-parse-files", type=int, default=12)
    return p.parse_args()


def norm_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name).strip() if ch.isalnum())


def join_limited(items: Iterable[str], limit: int = 8) -> str:
    values = [str(x) for x in items if str(x)]
    if len(values) <= limit:
        return ", ".join(values)
    return ", ".join(values[:limit]) + f", ... (+{len(values) - limit})"


def detect_csv_header_row(path: Path, encoding: str) -> int:
    try:
        lines = path.read_text(encoding=encoding, errors="ignore").splitlines()[:12]
    except Exception:
        return 0
    for idx, line in enumerate(lines):
        cells = [cell.strip().strip("\ufeff") for cell in line.split(",")]
        normalized = {norm_name(cell) for cell in cells if cell}
        if {"year", "month", "day"}.issubset(normalized):
            return idx
        if any(key in normalized for key in {"time", "datetime", "timestamp", "datatime"}):
            return idx
        if any("时间" in cell or "日期" in cell for cell in cells):
            return idx
    return 0


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
        last_error: Exception | None = None
        for enc in encodings:
            try:
                header = detect_csv_header_row(path, enc)
                return pd.read_csv(path, header=header, encoding=enc, low_memory=False)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Failed to read CSV {path}: {last_error}")
    if ext in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read Excel {path}: {exc}") from exc
    raise RuntimeError(f"Unsupported file type: {path.suffix}")


def parse_datetime_series(series: pd.Series) -> pd.Series:
    if str(series.dtype).startswith("datetime64"):
        return pd.to_datetime(series, errors="coerce")

    attempts = [
        lambda s: pd.to_datetime(s, errors="coerce", format="%Y-%m-%d %H:%M:%S"),
        lambda s: pd.to_datetime(s, errors="coerce", format="%Y/%m/%d %H:%M:%S"),
        lambda s: pd.to_datetime(s, errors="coerce", format="%m/%d/%Y %H:%M:%S"),
        lambda s: pd.to_datetime(s, errors="coerce", format="%d/%m/%Y %H:%M:%S"),
        lambda s: pd.to_datetime(s, errors="coerce", format="%Y-%m-%d"),
        lambda s: pd.to_datetime(s, errors="coerce", dayfirst=False),
        lambda s: pd.to_datetime(s, errors="coerce", dayfirst=True),
    ]
    best = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    best_ratio = -1.0
    for parser in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = parser(series)
        except Exception:
            continue
        ratio = float(parsed.notna().mean())
        if ratio > best_ratio:
            best = parsed
            best_ratio = ratio
        if ratio >= 0.99:
            break
    return best


def detect_time_series(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    for col in df.columns:
        col_str = str(col)
        key = norm_name(col_str)
        if any(token in key for token in ("time", "datetime", "timestamp", "datatime", "date")) or "时间" in col_str or "日期" in col_str:
            ts = parse_datetime_series(df[col])
            if ts.notna().mean() >= 0.7:
                return col_str, ts

    norm_map = {norm_name(col): str(col) for col in df.columns}
    year_col = next((norm_map[key] for key in norm_map if key == "year"), "")
    month_col = next((norm_map[key] for key in norm_map if key == "month"), "")
    day_col = next((norm_map[key] for key in norm_map if key == "day"), "")
    hour_col = next((norm_map[key] for key in norm_map if key == "hour"), "")
    minute_col = next((norm_map[key] for key in norm_map if key == "minute"), "")
    if year_col and month_col and day_col:
        parts = {
            "year": pd.to_numeric(df[year_col], errors="coerce"),
            "month": pd.to_numeric(df[month_col], errors="coerce"),
            "day": pd.to_numeric(df[day_col], errors="coerce"),
            "hour": pd.to_numeric(df[hour_col], errors="coerce").fillna(0) if hour_col else 0,
            "minute": pd.to_numeric(df[minute_col], errors="coerce").fillna(0) if minute_col else 0,
        }
        ts = pd.to_datetime(parts, errors="coerce")
        if ts.notna().mean() >= 0.7:
            return "derived_year_month_day", ts
    return "", pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def find_matching_columns(df: pd.DataFrame, keywords: Sequence[str]) -> List[str]:
    matched = []
    for col in df.columns:
        col_str = str(col)
        col_norm = norm_name(col_str)
        if any(keyword in col_norm or keyword in col_str.lower() or keyword in col_str for keyword in keywords):
            matched.append(col_str)
    return matched


def choose_target_column(df: pd.DataFrame, numeric_df: pd.DataFrame) -> str:
    best_col = ""
    best_score = -10**9
    for col in df.columns:
        col_str = str(col)
        key = norm_name(col_str)
        valid = float(numeric_df[col_str].notna().mean()) if col_str in numeric_df.columns else 0.0
        score = int(valid * 100)
        if "实际发电功率" in col_str:
            score += 1000
        if "powermw" in key or key == "power":
            score += 900
        if "yd15" in key:
            score += 850
        if "负荷" in col_str or "load" in key:
            score += 800
        if "roundapower0" in key:
            score += 650
        if "prepower" in key:
            score += 450
        if "风速" in col_str or "windspeed" in key:
            score += 300
        if "温度" in col_str or "temperature" in key:
            score += 100
        if score > best_score:
            best_col = col_str
            best_score = score
    return best_col


def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in mask.astype(bool):
        if value:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def anomaly_flags(series: pd.Series) -> Dict[str, bool]:
    values = pd.to_numeric(series, errors="coerce")
    missing_mask = values.isna().to_numpy()
    missing_burst = longest_true_run(missing_mask) >= 4

    arr = values.dropna().to_numpy(dtype=float)
    if arr.size < 16:
        return {
            "missing_burst_candidate": missing_burst,
            "flatline_candidate": False,
            "spike_candidate": False,
            "drift_candidate": False,
        }

    rounded = np.round(arr, 6)
    equal_run = np.isclose(rounded[1:], rounded[:-1], atol=1e-6)
    flatline = longest_true_run(equal_run) + 1 >= 12

    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad > 1e-12:
        robust_z = np.abs(arr - median) / (1.4826 * mad + 1e-12)
        spike = bool(np.sum(robust_z > 10.0) >= max(3, int(0.001 * arr.size)))
    else:
        spike = False

    quarter = max(8, arr.size // 4)
    first_mean = float(np.mean(arr[:quarter]))
    last_mean = float(np.mean(arr[-quarter:]))
    std = float(np.std(arr)) + 1e-9
    drift = abs(last_mean - first_mean) / std >= 1.5

    return {
        "missing_burst_candidate": missing_burst,
        "flatline_candidate": flatline,
        "spike_candidate": spike,
        "drift_candidate": drift,
    }


def bool_from_any(items: Iterable[bool]) -> bool:
    return bool(any(bool(x) for x in items))


def priority_and_role(dataset_name: str, has_power: bool, has_wind: bool, power_ready_files: int) -> Tuple[str, str]:
    name = dataset_name
    if "机组" in name:
        return "high", "observed_fault_bank_and_safety_diagnostics"
    if "新疆" in name:
        return "high", "domestic_external_wind_site"
    if "美国" in name:
        return "high", "international_external_wind_site"
    if "内蒙古" in name:
        return "medium", "lightweight_external_wind_sanity_check"
    if "GEFCom" in name:
        return "medium", "secondary_boundary_benchmark"
    if "负荷" in name:
        return "medium", "single_auxiliary_load_benchmark_only"
    if "空气" in name:
        return "low", "optional_non_power_boundary_test"
    if has_power and has_wind and power_ready_files > 0:
        return "high", "wind_site_candidate"
    return "medium", "needs_manual_review"


def select_files_to_parse(files: Sequence[Path], dataset_path: Path, max_parse_files: int) -> List[Path]:
    def sort_key(path: Path) -> Tuple[int, int, int, str]:
        depth = len(path.relative_to(dataset_path).parts)
        ext_priority = {".csv": 0, ".xlsx": 1, ".xls": 2}.get(path.suffix.lower(), 9)
        size_rank = -int(path.stat().st_size)
        return (depth, ext_priority, size_rank, str(path).lower())

    ordered = sorted(files, key=sort_key)
    return ordered[: max(1, max_parse_files)]


def audit_file(dataset_name: str, dataset_path: Path, file_path: Path) -> FileAudit:
    relative_path = str(file_path.relative_to(dataset_path))
    try:
        df = read_table(file_path)
        df = df.loc[:, [str(c).strip() != "" for c in df.columns]]
        num_df = numeric_frame(df)
        time_col, ts = detect_time_series(df)
        ts_valid = ts.dropna().sort_values()
        if len(ts_valid) >= 2:
            deltas = ts_valid.diff().dropna().dt.total_seconds().div(60)
            dominant = float(deltas.mode().iloc[0]) if not deltas.empty else float("nan")
            regularity = float((np.isclose(deltas, dominant, atol=1e-6)).mean()) if not deltas.empty else float("nan")
        else:
            dominant = float("nan")
            regularity = float("nan")

        power_cols = find_matching_columns(df, ("实际发电功率", "power", "功率", "yd15", "load", "负荷", "prepower", "roundapower0"))
        wind_cols = find_matching_columns(df, ("风速", "windspeed", "windspeedat100m", "windspeedatm"))
        direction_cols = find_matching_columns(df, ("风向", "winddirection", "direction"))
        temperature_cols = find_matching_columns(df, ("温度", "temperature"))
        pressure_cols = find_matching_columns(df, ("气压", "pressure"))
        humidity_cols = find_matching_columns(df, ("湿度", "humidity"))
        unit_tag_cols = [str(col) for col in df.columns if any(tag in str(col) for tag in ("(", ")", "（", "）", "%", "°"))]

        target = choose_target_column(df, num_df)
        target_series = num_df[target] if target and target in num_df.columns else pd.Series(dtype=float)
        flags = anomaly_flags(target_series if not target_series.empty else num_df.iloc[:, 0] if not num_df.empty else pd.Series(dtype=float))

        overall_missing = float(df.isna().mean().mean() * 100.0)
        usable_numeric = num_df.loc[:, num_df.notna().any(axis=0)]
        numeric_missing = float(usable_numeric.isna().mean().mean() * 100.0) if not usable_numeric.empty else float("nan")
        power_ready = bool(power_cols) and any(float(num_df[col].notna().mean()) >= 0.5 for col in power_cols if col in num_df.columns)

        return FileAudit(
            dataset_name=dataset_name,
            dataset_path=str(dataset_path),
            file_path=str(file_path),
            relative_path=relative_path,
            extension=file_path.suffix.lower(),
            parse_status="ok",
            rows=int(len(df)),
            cols=int(len(df.columns)),
            time_col=str(time_col),
            start_time=str(ts_valid.min()) if not ts_valid.empty else "",
            end_time=str(ts_valid.max()) if not ts_valid.empty else "",
            dominant_freq_minutes=dominant,
            regularity_ratio=regularity,
            regular_sampling=bool(np.isfinite(regularity) and regularity >= 0.95),
            overall_missing_pct=overall_missing,
            numeric_missing_pct=numeric_missing,
            multivariate=bool(len(df.columns) >= 3 and len(usable_numeric.columns) >= 3),
            candidate_target=str(target),
            power_columns=join_limited(power_cols),
            wind_columns=join_limited(wind_cols),
            direction_columns=join_limited(direction_cols),
            temperature_columns=join_limited(temperature_cols),
            pressure_columns=join_limited(pressure_cols),
            humidity_columns=join_limited(humidity_cols),
            unit_tag_columns=join_limited(unit_tag_cols),
            power_ready=power_ready,
            lh_96_24_compatible=bool(len(df) >= 120 and np.isfinite(dominant)),
            missing_burst_candidate=flags["missing_burst_candidate"],
            flatline_candidate=flags["flatline_candidate"],
            spike_candidate=flags["spike_candidate"],
            drift_candidate=flags["drift_candidate"],
            example_columns=join_limited([str(col) for col in df.columns]),
            notes="",
        )
    except Exception as exc:
        return FileAudit(
            dataset_name=dataset_name,
            dataset_path=str(dataset_path),
            file_path=str(file_path),
            relative_path=relative_path,
            extension=file_path.suffix.lower(),
            parse_status="error",
            rows=0,
            cols=0,
            time_col="",
            start_time="",
            end_time="",
            dominant_freq_minutes=float("nan"),
            regularity_ratio=float("nan"),
            regular_sampling=False,
            overall_missing_pct=float("nan"),
            numeric_missing_pct=float("nan"),
            multivariate=False,
            candidate_target="",
            power_columns="",
            wind_columns="",
            direction_columns="",
            temperature_columns="",
            pressure_columns="",
            humidity_columns="",
            unit_tag_columns="",
            power_ready=False,
            lh_96_24_compatible=False,
            missing_burst_candidate=False,
            flatline_candidate=False,
            spike_candidate=False,
            drift_candidate=False,
            example_columns="",
            notes=str(exc),
        )


def collect_supported_files(dataset_path: Path) -> List[Path]:
    return [
        path
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS and not path.name.startswith("~$")
    ]


def aggregate_dataset(dataset_name: str, dataset_path: Path, file_audits: Sequence[FileAudit], total_supported_files: int, total_files: int) -> Dict[str, object]:
    ok_files = [fa for fa in file_audits if fa.parse_status == "ok"]
    primary = max(ok_files, key=lambda fa: (fa.rows * max(1, fa.cols), fa.rows, fa.cols), default=None)

    valid_starts = [pd.to_datetime(fa.start_time, errors="coerce") for fa in ok_files if fa.start_time]
    valid_ends = [pd.to_datetime(fa.end_time, errors="coerce") for fa in ok_files if fa.end_time]
    valid_freqs = [fa.dominant_freq_minutes for fa in ok_files if np.isfinite(fa.dominant_freq_minutes)]
    dominant_dataset_freq = float(pd.Series(valid_freqs).mode().iloc[0]) if valid_freqs else float("nan")
    power_ready_files = sum(int(fa.power_ready) for fa in ok_files)

    has_power = bool(any(fa.power_columns for fa in ok_files))
    has_wind = bool(any(fa.wind_columns for fa in ok_files))
    priority, role = priority_and_role(dataset_name, has_power, has_wind, power_ready_files)

    return {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "dataset_exists": dataset_path.exists(),
        "total_files": total_files,
        "supported_table_files": total_supported_files,
        "parsed_files": len(file_audits),
        "parsed_ok_files": len(ok_files),
        "primary_file": primary.relative_path if primary else "",
        "primary_rows": primary.rows if primary else 0,
        "primary_cols": primary.cols if primary else 0,
        "dataset_start": str(min(valid_starts)) if valid_starts else "",
        "dataset_end": str(max(valid_ends)) if valid_ends else "",
        "dominant_freq_minutes": dominant_dataset_freq,
        "regular_sampling_files": sum(int(fa.regular_sampling) for fa in ok_files),
        "regular_sampling_flag": bool_from_any(fa.regular_sampling for fa in ok_files),
        "overall_missing_pct_mean": float(np.nanmean([fa.overall_missing_pct for fa in ok_files])) if ok_files else float("nan"),
        "numeric_missing_pct_mean": float(np.nanmean([fa.numeric_missing_pct for fa in ok_files])) if ok_files else float("nan"),
        "multivariate_flag": bool_from_any(fa.multivariate for fa in ok_files),
        "candidate_target_example": primary.candidate_target if primary else "",
        "has_power": has_power,
        "has_wind": has_wind,
        "has_direction": bool(any(fa.direction_columns for fa in ok_files)),
        "has_temperature": bool(any(fa.temperature_columns for fa in ok_files)),
        "has_pressure": bool(any(fa.pressure_columns for fa in ok_files)),
        "has_humidity": bool(any(fa.humidity_columns for fa in ok_files)),
        "power_ready_files": power_ready_files,
        "lh_96_24_compatible": bool_from_any(fa.lh_96_24_compatible for fa in ok_files),
        "missing_burst_candidate": bool_from_any(fa.missing_burst_candidate for fa in ok_files),
        "flatline_candidate": bool_from_any(fa.flatline_candidate for fa in ok_files),
        "spike_candidate": bool_from_any(fa.spike_candidate for fa in ok_files),
        "drift_candidate": bool_from_any(fa.drift_candidate for fa in ok_files),
        "unit_tags_found": bool(any(fa.unit_tag_columns for fa in ok_files)),
        "example_columns": primary.example_columns if primary else "",
        "paper_priority": priority,
        "recommended_role": role,
        "notes": "" if all(fa.parse_status == "ok" for fa in file_audits) else join_limited(
            [f"{fa.relative_path}: {fa.notes}" for fa in file_audits if fa.parse_status != "ok"],
            limit=3,
        ),
    }


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows: List[Dict[str, object]] = []
    file_rows: List[Dict[str, object]] = []
    manifest: List[Dict[str, object]] = []

    for dataset_name in args.datasets:
        dataset_path = root / dataset_name
        if not dataset_path.exists():
            dataset_rows.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_path": str(dataset_path),
                    "dataset_exists": False,
                    "total_files": 0,
                    "supported_table_files": 0,
                    "parsed_files": 0,
                    "parsed_ok_files": 0,
                    "primary_file": "",
                    "primary_rows": 0,
                    "primary_cols": 0,
                    "dataset_start": "",
                    "dataset_end": "",
                    "dominant_freq_minutes": float("nan"),
                    "regular_sampling_files": 0,
                    "regular_sampling_flag": False,
                    "overall_missing_pct_mean": float("nan"),
                    "numeric_missing_pct_mean": float("nan"),
                    "multivariate_flag": False,
                    "candidate_target_example": "",
                    "has_power": False,
                    "has_wind": False,
                    "has_direction": False,
                    "has_temperature": False,
                    "has_pressure": False,
                    "has_humidity": False,
                    "power_ready_files": 0,
                    "lh_96_24_compatible": False,
                    "missing_burst_candidate": False,
                    "flatline_candidate": False,
                    "spike_candidate": False,
                    "drift_candidate": False,
                    "unit_tags_found": False,
                    "example_columns": "",
                    "paper_priority": "missing",
                    "recommended_role": "dataset_not_found",
                    "notes": "dataset path missing",
                }
            )
            continue

        all_files = [path for path in dataset_path.rglob("*") if path.is_file()]
        supported = collect_supported_files(dataset_path)
        to_parse = select_files_to_parse(supported, dataset_path, args.max_parse_files)
        audits = [audit_file(dataset_name, dataset_path, file_path) for file_path in to_parse]

        dataset_rows.append(
            aggregate_dataset(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                file_audits=audits,
                total_supported_files=len(supported),
                total_files=len(all_files),
            )
        )
        file_rows.extend(asdict(audit) for audit in audits)
        manifest.append(
            {
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path),
                "total_files": len(all_files),
                "supported_files": len(supported),
                "parsed_files": [str(path) for path in to_parse],
            }
        )

    dataset_df = pd.DataFrame(dataset_rows).sort_values(["paper_priority", "dataset_name"], ascending=[True, True])
    file_df = pd.DataFrame(file_rows).sort_values(["dataset_name", "relative_path"])

    dataset_df.to_csv(out_dir / "audit_table.csv", index=False, encoding="utf-8-sig")
    file_df.to_csv(out_dir / "audit_file_details.csv", index=False, encoding="utf-8-sig")
    (out_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] wrote {(out_dir / 'audit_table.csv')}")
    print(f"[done] wrote {(out_dir / 'audit_file_details.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
