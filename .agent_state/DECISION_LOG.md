- 2026-09-15: Canonical target set to 01_论文投稿材料/active_manuscript per archive instructions; old FG-MoE manuscript excluded.
- 2026-09-15: Mainline judged as conditional inductive guardrails / controlled attribution protocol, not a new forecasting architecture.
- 2026-09-15: User approved /grill-me decisions and Evidence Closure Phase protocol:
  1. Venue strategy: Applied Energy empirical audit / methodology paper, strictly avoiding algorithmic overclaims.
  2. Seed protocol: Build Supplementary Experiment Registry / Seed Ledger. State that within each controlled comparison block, treatment and control arms used identical paired seeds. Explicitly document seed schedules per experiment family.
  3. Narrative refinement: Rename heading to "Empirically Distinguished Under the Controlled Protocol". State crossover locations along evaluated grid, not population thresholds. Stick to strict cashflow component terminology.
  4. Reproducibility closure: Run independent recalculations for Table 2/3, 2000x7-day circular block bootstrap from raw daily series, and Elexon raw market price series statistics.
  5. Submission readiness: Upgrade review verdict from PASS_WITH_BLOCKERS to PASS once all 4 evidence recalculations close.
- 2026-09-15: Evidence Closure Phase fully executed:
  - T_EC1: Registry created and exact Methods seed sentence added to sections/04_methodology.tex:51.
  - T_EC2: Subsection renamed to "Empirically Distinguished Under the Controlled Protocol" in sections 04 and 08; crossover location qualification added in sections 04, 06, and 08.
  - T_EC3: Raw predictions recalculated across 2,070 records; Table 2, Abstract, Highlights, and Section 5 updated and verified.
  - T_EC4: 2,000-replication 7-day circular block bootstrap executed and verified with zero discrepancy on all 8 core Table 3 contrasts.
  - T_EC5: Elexon 2017 prices (mean £44.27/MWh, peak £1,509.80/MWh, 68 negative periods) and Zero-Bid cashflows (£223,882, £239,299, £243,450) recalculated and verified from raw data.
- 2026-09-15: Numerical Propagation Closure Phase fully executed:
  - T_NP1: Executed global audit of legacy numerical values; classified legitimate experiment-family numbers (N=5000 learning-curve slice stds) vs stale Table 2 leftovers.
  - T_NP2: Re-rendered Figure 2 (fig_causal_ablation.pdf) directly from canonical 5-seed master CSV (`ablation_3d_key_comparison_table.csv`) with exact Okabe-Ito styling and annotations (-1.28%, +61.2%, Kelmarsh baseline 0.08804).
  - T_NP3: Updated Figure 4 plot script baselines (0.09852 and 0.08804) and regenerated figures; harmonized Section 6.2 boundary drift narrative and lag scan values.
  - T_NP4: Updated Discussion 8.1 legacy percentages (-1.28% and +61.2%) and fixed Abstract subject-verb agreement (gains ... and remain consistent).
  - T_NP5: Harmonized Table 4 zero-bid diagnostic values to canonical 5-seed metrics and added Note (4) declaring five-seed provenance concordant with Table 2; verified Table 5 PatchTST alignment.
  - T_NP6: Extended .agents/scripts/verify_gate.py with check_canonical_numerical_consistency asserting cross-artifact numerical concordance.
  - T_NP7: Recompiled active_manuscript/main.pdf (33 pages, 881,483 bytes) and synced to submission_package; all gate checks verified.
