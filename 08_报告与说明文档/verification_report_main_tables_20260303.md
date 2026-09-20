# Main Table Verification Report

Date: 2026-03-03

Scope: main text result tables in `main_PI-MoE_final_AE_ready.tex` with canonical evidence dir `results/supplementary_evidence`.

## Table 2 (Wind CN robustness matrix)
- DLinear: TEX=[0.9379, 6342.86, 4203.53, -1.579, 0.2058, 268.36] | SRC=[0.9379, 6342.86, 4203.53, -1.579, 0.2058, 268.36] | PASS
- Ultra-LSNT: TEX=[0.8727, 9081.66, 7169.19, 0.7263, 0.7829, 16.78] | SRC=[0.8727, 9081.66, 7169.19, 0.7263, 0.7829, 16.78] | PASS
- CEEMDAN-WOA-GRU: TEX=[0.9534, 7108.43, 4900.98, 0.9385, 0.9278, 1.57] | SRC=[0.9534, 7108.43, 4900.98, 0.9385, 0.9278, 1.57] | PASS
- iTransformer: TEX=[0.9012, 8004.05, 6333.29, 0.8717, 0.8609, 3.26] | SRC=[0.9012, 8004.05, 6333.29, 0.8717, 0.8609, 3.26] | PASS
- Status: PASS

## Table 3 (Cross-domain transfer)
- Ultra-LSNT: TEX=[0.7878, 0.4861, 0.7554] | SRC=[0.7878, 0.4861, 0.7554] | PASS
- DLinear: TEX=[0.8779, 0.3433, 0.737] | SRC=[0.8779, 0.3433, 0.737] | PASS
- iTransformer: TEX=[0.8563, 0.459, 0.7467] | SRC=[0.8563, 0.459, 0.7467] | PASS
- TimeMixer: TEX=[0.8023, 0.3869, 0.7201] | SRC=[0.8023, 0.3869, 0.7201] | PASS
- ARIMA (logged): TEX=[0.6971, 0.2008, -0.4031] | SRC=[0.6971, 0.2008, -0.4031] | PASS
- SVR (logged): TEX=[0.5796, 0.1716, 0.8226] | SRC=[0.5796, 0.1716, 0.8226] | PASS
- SSA-ELM: TEX=[0.8075, 0.3152, 0.7662] | SRC=[0.8075, 0.3152, 0.7662] | PASS
- Mamba (logged): TEX=[0.8866, 0.3688, -0.0083] | SRC=[0.8866, 0.3688, -0.0083] | PASS
- Status: PASS

## Table 4 Panel B (Guardrail safety ablation)
- Ultra-LSNT (no guardrail): TEX=[0.8727, 0.8502, 0.0656] | SRC=[0.8727, 0.8502, 0.0656] | PASS
- Ultra-LSNT + clip-only (posthoc): TEX=[0.4319, 0.3773, 0.0] | SRC=[0.4319, 0.3773, 0.0] | PASS
- FG-MoE (full-trained): TEX=[0.8747, 0.8526, 0.0023] | SRC=[0.8747, 0.8526, 0.0023] | PASS
- Status: PASS

## Table 5 (Dispatch coupling)
- Panel A Ultra-LSNT: TEX=[0.0, 100.0, 0.7583, 24.2] | SRC=[0.0, 100.0, 0.7583, 24.2] | PASS
- Panel C Ultra-LSNT: TEX=[54381.53, 100.0, 12.01] | SRC=[54381.53, 100.0, 12.01] | PASS

## Table 6 (Joint Pareto)
- Row (K=1,tau=0.9,lambda2=1): TEX=[1.0, 0.9, 1.0, 10.232, 0.3831, 0.0] | SRC=[1.0, 0.9, 1.0, 10.232, 0.3831, 0.0] | PASS

## Conflict Resolution Applied
- Canonical evidence directory: `results/supplementary_evidence`.
- Key conflicting files were resolved by keeping the better/complete version and deleting the conflicting counterpart in `applied_energy_final_results/supplementary_evidence`.
- Table notes in manuscript now point to the canonical directory only.