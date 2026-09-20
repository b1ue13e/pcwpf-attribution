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

- 2026-09-21: GitHub integration authorized by user (account b1ue13e, email 3211049186@qq.com, PAT provided in chat). Decisions:
  1. Private repo `b1ue13e/pcwpf-attribution`; local `revision` branch pushes to remote default branch `main`.
  2. Gitignore policy: exclude raw SCADA zips (public on Zenodo/BMRS), heavy run caches under 06_实验结果与日志, TIFF/PNG production masters under 03_图表与绘图素材, cloud delivery packs (07_云端压缩包与交付包, 07_审稿Rebuttal备用包), third-party journal PDFs (01_论文投稿材料/对标顶刊论文), personal/third-party/legacy dirs (09_, 90_, 99_), and scratch/tool state (tmp*, .zcode, .reasonix, .workbuddy, .experiment-to-paper). Kept: canonical CSVs (04_数据集/constraint_rebuild_v2/*.csv, elexon_system_prices_2016_2024.csv), analysis-level evidence, vector figure sources, all scripts.
  3. CI = verification gate (state + canonical numerical consistency) followed by tectonic compile of main.tex with PDF artifact upload.
  4. Ledger-honesty rule: in-flight GitHub tasks are excluded from pushed state until VERIFIED with evidence, so every CI gate run is genuinely green (no fake-verified tasks).
  5. Security: PAT pasted in plaintext chat is treated as compromised; it is never written to repo files or git config; user advised to revoke/rotate after the session. Local gh keyring already held b1ue13e credentials, so pushes did not require embedding the token in git URLs.
