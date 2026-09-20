from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SUMMARY = ROOT / "results" / "supplementary_evidence" / "windcn_context_audit_summary.csv"
DEFAULT_OUT = ROOT / "results" / "supplementary_evidence" / "table2_panel_a_rows.tex"

MODEL_ORDER = [
    "DLinear",
    "TimeMixer",
    "iTransformer",
    "Ultra-LSNT",
    "LightGBM",
    "Persistence",
    "Chronos-Bolt-Small",
    "Moirai-2.0-R-Small",
]

PLACEHOLDERS = {
    "Chronos-Bolt-Small": "cloud-side upper-bound row populated by the audited Chronos rerun CSV",
    "Moirai-2.0-R-Small": "cloud-side upper-bound row populated by the audited Moirai rerun CSV",
}


def fmt(value: float, digits: int) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Render Table 2 Panel A LaTeX rows from the Wind (CN) context audit summary.")
    ap.add_argument("--summary-csv", type=str, default=str(DEFAULT_SUMMARY))
    ap.add_argument("--out-tex", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    summary_path = Path(args.summary_csv)
    out_path = Path(args.out_tex)
    if not summary_path.exists():
        raise FileNotFoundError(f"summary CSV not found: {summary_path}")

    df = pd.read_csv(summary_path)
    model_col = df["model"].astype(str).str.strip()

    lines: list[str] = []

    for model in MODEL_ORDER:
        sub = df[model_col == model]
        row = None if sub.empty else sub.iloc[0]
        if row is None:
            if model in PLACEHOLDERS:
                lines.append(f"{model} & \\multicolumn{{5}}{{c}}{{{PLACEHOLDERS[model]}}} \\\\")
                continue
            lines.append(f"{model} & -- & -- & -- & -- & -- \\\\")
            continue

        numeric_fields = [
            "clean_R2",
            "clean_RMSE_kW",
            "R2_sigma_eff_060",
            "RMSE_sigma_eff_060_kW",
            "rel_rmse_increase_pct_060",
        ]
        if model in PLACEHOLDERS and all(pd.isna(row.get(field, pd.NA)) for field in numeric_fields):
            lines.append(f"{model} & \\multicolumn{{5}}{{c}}{{{PLACEHOLDERS[model]}}} \\\\")
            continue

        lines.append(
            " & ".join(
                [
                    model,
                    fmt(row.get("clean_R2", pd.NA), 4),
                    fmt(row.get("clean_RMSE_kW", pd.NA), 2),
                    fmt(row.get("R2_sigma_eff_060", pd.NA), 4),
                    fmt(row.get("RMSE_sigma_eff_060_kW", pd.NA), 2),
                    fmt(row.get("rel_rmse_increase_pct_060", pd.NA), 2),
                ]
            )
            + r" \\"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
