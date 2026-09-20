import argparse
import os
import re
from typing import List, Tuple

import pandas as pd


def fail(msg: str) -> None:
    raise SystemExit(msg)


def count_tables(tex: str) -> int:
    return tex.count("\\begin{table}")


def find_float_in_tex(tex: str, value: float, tol: float = 1e-3) -> bool:
    # Check typical printed formats used in manuscript.
    candidates = [
        f"{value:.4f}",
        f"{value:.3f}",
        f"{value:.2f}",
        f"{value:.1f}",
    ]
    for c in candidates:
        if c in tex:
            return True
    # fallback numeric regex approximate
    pattern = re.compile(r"-?\d+\.\d+")
    for m in pattern.finditer(tex):
        try:
            x = float(m.group(0))
            if abs(x - value) <= tol:
                return True
        except Exception:
            pass
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate main-6 matrix layout and key numbers")
    ap.add_argument("--tex", type=str, required=True)
    ap.add_argument("--evidence_dir", type=str, default=os.path.join("results", "supplementary_evidence"))
    ap.add_argument("--tolerance", type=float, default=1e-3)
    args = ap.parse_args()

    if not os.path.exists(args.tex):
        fail(f"Missing tex file: {args.tex}")
    tex = open(args.tex, "r", encoding="utf-8", errors="ignore").read()

    n_tables = count_tables(tex)
    if n_tables != 6:
        fail(f"Expected 6 tables in main text, found {n_tables}")

    required_labels = [
        "tab:data_summary_windcn",
        "tab:main_wind_cn",
        "tab:master_robustness",
        "tab:master_hardware",
        "tab:matrix_ablation_physics",
        "tab:master_dispatch",
        "tab:multi_domain",
        "tab:joint_parameter_analysis",
    ]
    missing = [x for x in required_labels if x not in tex]
    if missing:
        fail("Missing required labels/aliases: " + ", ".join(missing))

    # verify key ARIMA/SVR values are surfaced in merged matrix text
    arima_clean = pd.read_csv(os.path.join(args.evidence_dir, "arima_clean_windcn.csv")).iloc[0]
    svr_clean = pd.read_csv(os.path.join(args.evidence_dir, "svr_clean_windcn.csv")).iloc[0]
    arima_r060 = pd.read_csv(os.path.join(args.evidence_dir, "arima_robustness_windcn.csv"))
    arima_r060 = arima_r060[(arima_r060["fault_type"] == "gaussian") & (arima_r060["sigma_eff"] == 0.6)].iloc[0]
    svr_r060 = pd.read_csv(os.path.join(args.evidence_dir, "svr_robustness_windcn.csv"))
    svr_r060 = svr_r060[(svr_r060["fault_type"] == "gaussian") & (svr_r060["sigma_eff"] == 0.6)].iloc[0]

    checks: List[Tuple[str, float]] = [
        ("ARIMA clean R2", float(arima_clean["R2"])),
        ("SVR clean R2", float(svr_clean["R2"])),
        ("ARIMA sigma0.60 R2", float(arima_r060["R2"])),
        ("SVR sigma0.60 R2", float(svr_r060["R2"])),
    ]
    missing_vals = []
    for name, v in checks:
        if not find_float_in_tex(tex, v, tol=args.tolerance):
            missing_vals.append(f"{name}={v:.6f}")
    if missing_vals:
        fail("Missing key evidence values in tex: " + "; ".join(missing_vals))

    print("Validation passed")


if __name__ == "__main__":
    main()
