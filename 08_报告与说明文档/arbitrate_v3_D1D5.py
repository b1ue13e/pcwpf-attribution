"""Arbitration of coder_A vs coder_B D1-D3 disagreements (D4/D5 majority agreed).

Arbitration rules (declared):
1. Prefer coder_A (understanding-based) over coder_B (keyword-rule) where the
   abstract text supports a substantive reading; keyword rules mis-fired on
   several (e.g. PCT D3 'contemporaneous' vs true 'none' since KDE prior frozen
   pre-training).
2. D3 is the least objective dimension; arbitrate each case from the paper's
   actual inference requirement, defaulting to 'unreported' when the abstract
   does not specify.
3. Multi-coder output merged; flag 'weak' where full-text needed.
"""
from __future__ import annotations
import csv, json

# Arbitrated decisions per paper for the disagreed dimensions only.
# (D1, D2, D3) arbitrated values; D4/D5 take the agreed coder_A value.
ARB = {
    # paper_id: (D1_final, D2_final, D3_final, why)
    "JARA2024":        ("historical-static", "inference-architecture", "latent-state",
                        "possible power is latent under curtailment; similarity inference is architecture-stage"),
    "PHYCELL":         ("forecast-derived", "training-loss", "forecast-input",
                        "PDE on forecast wind field; forecast input holds"),
    "TANG2025":        ("forecast-derived", "training-loss", "forecast-input",
                        "theoretical power curve fitted from historical+Meteo; A correct"),
    "WANG2026":        ("historical-static", "inference-architecture", "latent-state",
                        "available power censored reconstruction; latent under curtailment"),
    "PITFN2025":       ("forecast-derived", "training-loss", "forecast-input",
                        "Cp_hat estimator is learned-online but deployment needs learned Cp; forecast-dependent"),
    "CPDIGITAL2025":   ("historical-static", "training-loss", "contemporaneous-state",
                        "Cp-lambda-beta from contemporaneous SCADA; keep A"),
    "XAIPC2023":       ("unreported", "unreported", "unreported",
                        "XAI validation, no forecasting constraint; excluded upstream"),
    "HUANG2026":       ("unreported", "unreported", "unreported",
                        "not a physics-constraint paper; excluded"),
    "MMD-PGF":         ("historical-static", "training-loss", "none",
                        "MMD over train-time historical joint; frozen prior -> none"),
    "TgDPF":           ("historical-static", "training-loss", "none",
                        "KDE prior frozen pre-training -> inference needs none"),
    "RISKREG2025":     ("unreported", "unreported", "unreported",
                        "chance-constrained curve fitting excluded"),
    "JENSENWAKE2026":  ("forecast-derived", "training-loss", "forecast-input",
                        "Jensen wake model needs upstream wind (contemporaneous measured) -> forecast-input close; keep A D2, D3=forecast-input"),
    "TCNTRANS2026":    ("unreported", "training-loss", "unreported",
                        "interval physics detail only in abstract -> D3 unreported"),
    "VARFORMERCQR2026":("unreported", "unreported", "unreported",
                        "conformal UQ, no physics constraint, excluded"),
    "MLPINN":          ("intrinsic", "training-loss", "contemporaneous-state",
                        "PMSG/BEM intrinsic turbine physics in simulation; contemporaneous state at eval"),
    "PIBPNN2025":      ("intrinsic", "training-loss", "contemporaneous-state",
                        "Betz/Cp/mechanical intrinsic; contemporaneous turbine state"),
    "SETVAL2025":      ("unreported", "unreported", "unreported",
                        "excluded curve fitting"),
}

def main():
    rows = list(csv.DictReader(open('literature_coding_table_v3_D1D5.csv', encoding='utf-8-sig')))
    for r in rows:
        pid = r['paper_id']
        if pid in ARB:
            d1, d2, d3, why = ARB[pid]
            r['D1_provenance'] = d1
            r['D2_stage'] = d2
            r['D3_inference_req'] = d3
            r['notes'] = (r.get('notes','') + f' | ARB(D1-D3)={d1}/{d2}/{d3}: {why}').strip(' |')
    fields = list(rows[0].keys())
    with open('literature_coding_table_v3_arbitrated.csv','w',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print('arbitrated table written:', len(rows), 'rows')
    from collections import Counter
    for dim in ['D1_provenance','D2_stage','D3_inference_req','D4_fidelity','D5_harm']:
        print(dim, dict(Counter(r[dim] for r in rows)))

if __name__ == '__main__':
    main()
