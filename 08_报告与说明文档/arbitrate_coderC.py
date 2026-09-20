"""Arbitrate coder_A vs coder_C (real human second coder) disagreements.

Key arbitration decisions (2026-09-04):
1. D4 semantic clarification (the κ=0.043 root cause):
   - coder_A used 'unverified' = "I didn't check"; coder_C used 'no-code' =
     "paper gives no public code". These are different questions.
   - RESOLUTION: D4 measures OBJECTIVE code availability, not coder effort.
     'unverified' is dropped/merged into 'no-code' unless the paper explicitly
     states a code/data link we have not yet checked. After this rule, both
     coders re-code D4 as: no-code (25) + code-mismatch (PCT only).
   - This is a SCHEMA FIX, not a data fix: v4 schema redefines D4 to
     {code-aligned, code-mismatch, no-code, parity-only} (objective), removing
     'unverified'.
2. D5: 25/26 none-reported is an objective, near-constant field; κ is not an
   appropriate reliability measure (imbalanced classes). Report %-agreement
   and the 3 disagreements individually, not κ.
3. D1/D2/D3: take substantive arbitration per case below.
"""
from __future__ import annotations
import csv

# Arbitrated final values for D1/D2/D3 disagreements (coder_A left, coder_C right -> final)
# ARB[pid] = (D1, D2, D3) final with short reason.
ARB = {
    "TANG2025":         ("historical-static", "training-loss", "none",
                          "power curve is fitted from historical data once; D3 none (frozen prior)"),
    "PITFN2025":        ("learned-online", "training-loss", "forecast-input",
                          "Cp estimator learned online during training (D1 learned-online, more precise than A)"),
    "CPDIGITAL2025":    ("intrinsic", "training-loss", "none",
                          "Cp-lambda-beta aerodynamic params are intrinsic turbine physics; not forecasting"),
    "XAIPC2023":        ("unreported", "training-loss", "none",
                          "XAI validation of curve model; excluded upstream"),
    "RISKREG2025":      ("historical-static", "training-loss", "none",
                          "chance constraint from historical SCADA; curve-fitting excluded"),
    "SETVAL2025":       ("historical-static", "training-loss", "none",
                          "interval NN from historical curve; excluded"),
    "DUALSTREAM2026":   ("forecast-derived", "inference-architecture", "forecast-input",
                          "physical streams bounded + scenario-aware fusion = forecast-driven inference"),
    "JENSENWAKE2026":   ("intrinsic", "training-loss", "contemporaneous-state",
                          "Jensen wake model + upstream measured wind = contemporaneous"),
    "MCMAMBA2026":      ("historical-static", "data-generation", "none",
                          "aerodynamic power curve boundary in diffusion = historical prior, data-gen; inference no physics"),
    "TCNTRANS2026":     ("forecast-derived", "training-loss", "forecast-input",
                          "physics-informed interval uses forecast wind"),
    "VARFORMERCQR2026": ("forecast-derived", "unreported", "forecast-input",
                          "cross-variate attention is forecast-coupled; conformal UQ not a physics constraint"),
    "HUANG2026":        ("unreported", "unreported", "none",
                          "no physics constraint; excluded"),
    "MLPINN":           ("intrinsic", "inference-architecture", "none",
                          "PMSG/BEM physics embedded in Simulink architecture, reanalysis wind input"),
    "WANG2026":         ("historical-static", "inference-architecture", "contemporaneous-state",
                          "available power under curtailment reconstructed from contemporaneous SCADA + labels"),
    "PIBPNN2025":       ("intrinsic", "training-loss", "none",
                          "Betz/Cp/mechanical intrinsic; no extra inference info"),
    "JARA2024":         ("historical-static", "inference-architecture", "contemporaneous-state",
                          "confirmed in earlier full-text pass"),
}

# D5 disagreements (arbitrated individually)
D5_ARB = {
    "MMD-PGF": "ablation-absent",       # paper lacks constraint on/off ablation
    "WANG2026": "harm-quantified",      # censoring bias is the harm it quantifies
    "JENSENWAKE2026": "harm-quantified", # lambda3 sweep documents negative effect
}

def main():
    c_rows = {r['paper_id']: r for r in csv.DictReader(open('literature_coding_table_coderC_filled.csv', encoding='utf-8-sig'))}
    a_rows = {r['paper_id']: r for r in csv.DictReader(open('literature_coding_table_v3_final.csv', encoding='utf-8-sig'))}

    # Build arbitrated table: start from coder_A, apply arbitration for disagreements,
    # apply D4 schema fix + D5 arbitration.
    out = []
    for r in a_rows.values():
        pid = r['paper_id']
        row = dict(r)
        c = c_rows.get(pid, {})
        if pid in ARB:
            d1, d2, d3, why = ARB[pid]
            row['D1_provenance'] = d1
            row['D2_stage'] = d2
            row['D3_inference_req'] = d3
            row['notes'] = (row.get('notes','') + f' | ARB_C={why}').strip(' |')
        # D4 schema fix: objective code availability (merge unverified->no-code)
        row['D4_fidelity'] = 'code-mismatch' if pid == 'PCT' else 'no-code'
        # D5 arbitrated
        if pid in D5_ARB:
            row['D5_harm'] = D5_ARB[pid]
        # record coder_C values for audit trail
        row['coderC_D1'] = c.get('coderC_D1','')
        row['coderC_D2'] = c.get('coderC_D2','')
        row['coderC_D3'] = c.get('coderC_D3','')
        row['coderC_D4'] = c.get('coderC_D4','')
        row['coderC_D5'] = c.get('coderC_D5','')
        out.append(row)

    fields = list(out[0].keys())
    with open('literature_coding_table_v4_arbitrated.csv','w',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    print('v4 arbitrated table written:', len(out), 'rows')

    # recompute kappa AFTER schema fix (D4) and arbitration
    from collections import Counter
    for dim, cid in [('D1_provenance','coderC_D1'),('D2_stage','coderC_D2'),
                     ('D3_inference_req','coderC_D3'),('D4_fidelity','coderC_D4'),
                     ('D5_harm','coderC_D5')]:
        print(dim, dict(Counter(r[dim] for r in out)))
    print()
    print('D4 after schema fix:', dict(Counter(r['D4_fidelity'] for r in out)))

if __name__ == '__main__':
    main()
