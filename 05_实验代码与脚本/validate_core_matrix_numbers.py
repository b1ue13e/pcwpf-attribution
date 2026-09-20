import argparse
import math
import os
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _find_row_line(text: str, model_pattern: str):
    for ln in text.splitlines():
        if "&" in ln and re.search(model_pattern, ln):
            return ln
    return None


def _extract_table_block(tex: str, label: str):
    pattern = re.compile(r"\\begin\{table\}.*?\\end\{table\}", re.S)
    for m in pattern.finditer(tex):
        block = m.group(0)
        if f"\\label{{{label}}}" in block:
            return block
    return None


def _extract_first_three_floats(line: str):
    if line is None:
        return None
    nums = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", line)
    if len(nums) < 3:
        return None
    return [float(nums[0]), float(nums[1]), float(nums[2])]


def _extract_core2_triplet(line: str):
    if line is None:
        return None
    parts = [p.strip() for p in line.split("&")]
    if len(parts) < 5:
        return None
    vals = []
    for p in parts[2:5]:
        nums = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", p)
        if not nums:
            return None
        vals.append(float(nums[0]))
    return vals


def validate_core2(tex_path: str, core2_csv: str, tol: float):
    tex = open(tex_path, "r", encoding="utf-8").read()
    block = _extract_table_block(tex, "tab:main_wind_cn")
    if block is None:
        return ["tex missing table block: tab:main_wind_cn"]
    df = pd.read_csv(core2_csv)
    issues = []

    # validate a representative subset that must match exactly in main table
    mapping = {
        "Ultra-LSNT (Reliability, tau=0.5)": r"Ultra-LSNT \(Reliability, .*?\)",
        "VMD-PSO-LSTM": r"VMD-PSO-LSTM",
        "CEEMDAN-WOA-GRU": r"CEEMDAN-WOA-GRU",
        "Mamba (auditable)": r"Mamba \(auditable\)",
    }

    for model, pat in mapping.items():
        row = df[df["model"].str.contains(model, regex=False, na=False)]
        if row.empty:
            issues.append(f"core2 missing model: {model}")
            continue
        r = row.iloc[0]
        line = _find_row_line(block, pat)
        if line is None:
            issues.append(f"tex missing row for model: {model}")
            continue
        vals = _extract_core2_triplet(line)
        if vals is None:
            issues.append(f"cannot parse numeric row in tex for model: {model}")
            continue
        exp = [float(r["R2"]), float(r["RMSE_kW"]), float(r["MAE_kW"])]
        # tex row typically has formatted precision, compare after rounding to 4/2/2 style
        chk = [abs(vals[0] - round(exp[0], 4)), abs(vals[1] - round(exp[1], 2)), abs(vals[2] - round(exp[2], 2))]
        if any(c > tol for c in chk):
            issues.append(f"core2 mismatch {model}: tex={vals}, expected~={[round(exp[0],4),round(exp[1],2),round(exp[2],2)]}")

    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate core matrix numbers against evidence")
    ap.add_argument("--tex", type=str, default="main_PI-MoE_final_AE_ready.tex")
    ap.add_argument("--evidence_dir", type=str, default=os.path.join("results", "supplementary_evidence"))
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args()

    report = []

    core2 = os.path.join(args.evidence_dir, "core2_clean.csv")
    if os.path.exists(core2):
        report.extend(validate_core2(args.tex, core2, tol=args.tolerance))
    else:
        report.append("missing core2_clean.csv")

    # Core3/Core4 are checked for file existence and non-empty rows.
    for fn in ["core3_robustness.csv", "core4_hardware.csv"]:
        p = os.path.join(args.evidence_dir, fn)
        if not os.path.exists(p):
            report.append(f"missing {fn}")
        else:
            df = pd.read_csv(p)
            if df.empty:
                report.append(f"empty {fn}")

    out_path = os.path.join(args.evidence_dir, "validate_core_matrix_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        if report:
            f.write("FAILED\n")
            for r in report:
                f.write(f"- {r}\n")
        else:
            f.write("PASSED\n")

    if report:
        print("Validation failed")
        for r in report:
            print(r)
        raise SystemExit(1)
    print("Validation passed")


if __name__ == "__main__":
    main()
