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
    tex_norm = tex.replace("{,}", "").replace(",", "")
    candidates = [
        f"{value:.4f}",
        f"{value:.3f}",
        f"{value:.2f}",
        f"{value:.1f}",
    ]
    for c in candidates:
        if c in tex or c in tex_norm:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate main-4 matrix layout and key numbers")
    ap.add_argument("--tex", type=str, required=True)
    ap.add_argument("--evidence_dir", type=str, default=os.path.join("results", "supplementary_evidence"))
    ap.add_argument("--tolerance", type=float, default=1e-3)
    args = ap.parse_args()

    if not os.path.exists(args.tex):
        fail(f"Missing tex file: {args.tex}")
    tex = open(args.tex, "r", encoding="utf-8", errors="ignore").read()

    n_tables = count_tables(tex)
    if n_tables != 4:
        fail(f"Expected 4 tables in main text, found {n_tables}")

    required_labels = [
        "tab:data_summary_windcn",
        "tab:main_wind_cn",
        "tab:master_robustness",
        "tab:multi_domain",
        "tab:master_hardware",
        "tab:matrix_ablation_physics",
        "tab:joint_parameter_analysis",
        "tab:master_dispatch",
    ]
    missing = [x for x in required_labels if x not in tex]
    if missing:
        fail("Missing required labels/aliases: " + ", ".join(missing))

    m2_path = os.path.join(args.evidence_dir, "main4_matrix2_perf_robust_transfer.csv")
    m3_path = os.path.join(args.evidence_dir, "main4_matrix3_hardware_safety_joint.csv")
    m4_path = os.path.join(args.evidence_dir, "main4_matrix4_dispatch_value.csv")
    for p in [m2_path, m3_path, m4_path]:
        if not os.path.exists(p):
            fail(f"Missing evidence CSV: {p}")

    m2 = pd.read_csv(m2_path)
    m3 = pd.read_csv(m3_path)
    m4 = pd.read_csv(m4_path)

    def r2_row(model: str):
        out = m2[m2["model"] == model]
        if out.empty:
            fail(f"Missing model in matrix2: {model}")
        return out.iloc[0]

    def row3(panel: str, item: str):
        out = m3[(m3["panel"] == panel) & (m3["item"] == item)]
        if out.empty:
            fail(f"Missing row in matrix3: {panel}/{item}")
        return out.iloc[0]

    def row4(layer: str, model: str):
        out = m4[(m4["layer"] == layer) & (m4["model"] == model)]
        if out.empty:
            fail(f"Missing row in matrix4: {layer}/{model}")
        return out.iloc[0]

    dlinear = r2_row("DLinear")
    ultra = r2_row("Ultra-LSNT")
    ssa = r2_row("SSA-ELM")
    arima = r2_row("ARIMA (auditable)")
    jump = row3("B_safety_ablation", "Jump Gate tau=0.3")
    joint = row3("C_joint_param", "K=1, tau=0.9, lambda2=1.00")
    fg = row3("B_safety_ablation", "FG-MoE (full-trained)")
    l1_ultra = row4("L1_CopperPlate_120d", "Ultra-LSNT")
    l2_ultra = row4("L2_RTS24_DCOPF_wmax260", "Ultra-LSNT")

    checks: List[Tuple[str, float]] = [
        ("DLinear clean R2", float(dlinear["clean_R2"])),
        ("DLinear R2@0.60", float(dlinear["R2_sigma_eff_060"])),
        ("Ultra clean R2", float(ultra["clean_R2"])),
        ("Ultra R2@0.60", float(ultra["R2_sigma_eff_060"])),
        ("Ultra WindUS R2", float(ultra["R2_wind_us"])),
        ("SSA clean R2", float(ssa["clean_R2"])),
        ("ARIMA clean R2", float(arima["clean_R2"])),
        ("Jump tau0.3 latency", float(jump["latency_ms"])),
        ("Joint safest R2", float(joint["R2_sigma_eff_060"])),
        ("L1 Ultra cost million", float(l1_ultra["operating_cost_million"])),
        ("L2 Ultra curtailment MWh", float(l2_ultra["curtailment_mwh"])),
    ]

    missing_vals = []
    for name, v in checks:
        if not find_float_in_tex(tex, v, tol=args.tolerance):
            missing_vals.append(f"{name}={v:.6f}")

    fg_safety = str(fg["safety_or_control"])
    if fg_safety:
        fg_safety_latex = fg_safety.replace("%", "\\%")
        if (fg_safety not in tex) and (fg_safety_latex not in tex):
            missing_vals.append(f"FG-MoE safety string missing: {fg_safety}")

    if missing_vals:
        fail("Missing key evidence values in tex: " + "; ".join(missing_vals))

    print("Validation passed")


if __name__ == "__main__":
    main()
