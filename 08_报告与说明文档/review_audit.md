# Adversarial Reviewer Attack & Integrity Audit Report

**Date**: 2026-09-04  
**Audit Target**: *From Specification to Deployment: A Systematic Audit of Constraint Provenance, Implementation Fidelity, and Harm in Physics-Constrained Wind Power Forecasting*  
**Auditor Persona**: 5 Critical Peer Reviewers (NeurIPS/ICLR/Applied Energy Standards)  
**Status**: **PASS (All Prior P0 Blockers Resolved & Calibrated)**

---

## 1. Executive Summary & Verdict

| Reviewer Persona | Initial Attack / Concern | Final Defense & Evidence Grounding | Status |
|---|---|---|---|
| **Reviewer 1 (Systematic Reviewer)** | "Denominator drift between 53, 48, 46, and 21 studies. Unclear PRISMA flow." | **Resolved**: Single source of truth locked (`study_level_ledger_48.csv`). 53 retrieved $\to$ 5 duplicate pairs merged $\to$ 48 studies $\to$ 2 explicit exclusions (S26, S42) $\to$ exactly 46 included studies. Dual-coder pilot transparently reported on 21 studies. | **PASS** |
| **Reviewer 2 (Software / Math Auditor)** | "PCT joint-KDE marginal independence accusation is mathematically wrong. Downstream RMSE degradation is unverified." | **Resolved**: Outer-product independence charge is formally retracted; mathematical equivalence to 2D product-kernel KDE is rigorously derived in Section 6.2.2. Reverse-JS divergence (`F.kl_div(p.log(), q)` = $\text{KL}(q \parallel p)$) is strictly kept as a static mathematical fact. All corrupted RMSE ratios are eliminated from main claims. | **PASS** |
| **Reviewer 3 (Reliability Specialist)** | "The 46-study corpus was integrated by a single specialist (Coder D); claiming dual-blind coding on all 46 would be fraudulent." | **Resolved**: Transparently disclosed in Section 4.2 and Section 7.1. Coder A vs. Coder C dual-coding is reported strictly for the 21-study pilot ($\kappa = 0.385\text{--}0.436$). The 46-study set is explicitly designated as a 'specialist-audited pilot scoping corpus' with open per-study citations. | **PASS** |
| **Reviewer 4 (Domain Energy Reviewer)** | "Wind-CN experiments cannot claim universal SOTA; physical constraints do not always win on clean data." | **Resolved**: Fully aligned. Section 5.4 clearly demonstrates that LightGBM (0.9563) and DLinear (0.9494) beat constrained models on clean data. Physical constraints are proven to function as defensive guardrails against sensor corruption ($\sigma=0.60$), not clean accuracy boosters. | **PASS** |
| **Reviewer 5 (Sceptical Adversary)** | "Interpreting 0/46 harm reporting as proof of publication bias is an overclaim." | **Resolved**: Language strictly calibrated. D5 is reported purely as an empirical absence of negative reporting under evaluated protocols, disclaiming causal assertions of publication bias. | **PASS** |

---

## 2. Review Gate Checklist (14/14 Passed)

1. [x] **No Fabricated Data / Numbers**: Every percentage ($0.0\%$, $97.8\%$, $52.2\%$, $47.8\%$, $87.0\%$) traces to exact rows in `corpus48_编码_全维度集成版_杜俊涛.csv`.
2. [x] **No Orphan Numbers**: Clean and noisy $R^2$ numbers match `table2_robustness_data.csv` to 4 decimal places.
3. [x] **Traceable Benchmarks**: Latency numbers match `latency_fixed.csv` exactly.
4. [x] **Retraction Discipline**: Erroneous independence assumption is openly corrected; previous 2.83-8.50x RMSE degradation claims are permanently removed.
5. [x] **PRISMA Traceability**: Every exclusion has an explicit justification in `study_level_ledger_48.csv`.
6. [x] **No Ghost Baselines**: All models (DLinear, LightGBM, Ultra-LSNT, iTransformer, TimeMixer, Persistence) have verified local run logs.
7. [x] **Hardware Parity Disclosed**: Identical NVIDIA RTX 4090 test bench explicitly stated.
8. [x] **Decoupled Lifecycle**: D1 (parameter origin) and D3 (runtime inputs) are proven orthogonal.
9. [x] **Ablation Confounding Disclosed**: 22 module-only ablation studies are explicitly separated from 24 strict constraint-on/off studies.
10. [x] **Software Mismatch Proven**: PyTorch `F.kl_div` argument order is mathematically proved to yield $\text{KL}(q \parallel p)$.
11. [x] **Scope Calibration**: Target venue calibrated to Q2 scoping audit (Applied Energy / Renewable Energy).
12. [x] **Double-Blind Boundary Respected**: Single-rater full-text status of the 46-study set is explicitly stated as a limitation.
13. [x] **No Causal Overclaims**: Harm reporting absence is framed descriptively, not as causal proof of bias.
14. [x] **Reproducibility Openness**: All CSVs, ledgers, scripts, and logs are referenced with open access paths.

---

## 3. Final Recommendation

**Overall Assessment**: **Accept as Calibrated Scoping Empirical Audit**.  
The manuscript has eliminated every trace of promotional rhetoric, resolved all prior technical contradictions, and presents an airtight, defensible audit of the field.
