# Evidence Ledger

## Task-ID: T_EC1
- Status: VERIFIED
- Claim: Build Supplementary Experiment Registry / Seed Ledger and align Methods seed phrasing.
- Direct Evidence:
  - `01_论文投稿材料/SUPPLEMENTARY_EXPERIMENT_REGISTRY.md`: Comprehensive registry covering all 7 empirical blocks, detailing site, arms, seeds, N_train, lambda, checkpoint rule, calibration split, and test split.
  - `01_论文投稿材料/active_manuscript/tables/supplementary_experiment_registry.csv`: Machine-readable registry table.
  - `01_论文投稿材料/active_manuscript/sections/04_methodology.tex:51`: Exact phrasing adopted: "Within each controlled comparison block, treatment and control arms used identical paired random seeds. Seed sets differed across distinct experiment families and are reported explicitly in the corresponding tables and supplementary experiment registry."

## Task-ID: T_EC2
- Status: VERIFIED
- Claim: Refine attribution headings and precision boundary wording in manuscript.
- Direct Evidence:
  - `01_论文投稿材料/active_manuscript/sections/04_methodology.tex:76` & `sections/08_discussion.tex:19`: Heading updated to "Empirically Distinguished Under the Controlled Protocol".
  - `01_论文投稿材料/active_manuscript/sections/04_methodology.tex:78` & `sections/08_discussion.tex:21`: Exact bounded phrasing: "Under the tested matched-control protocol, divergence between authentic and pseudo-scrambled priors provides evidence that the observed performance shift is specific to the empirical aerodynamic mapping, rather than being explained by penalty form and weight alone."
  - `01_论文投稿材料/active_manuscript/sections/04_methodology.tex:80`, `sections/06_boundaries.tex:41`, & `sections/08_discussion.tex:24`: Added exact qualification: "These values are empirical crossover locations along the evaluated perturbation grid, not population-level physical thresholds."

## Task-ID: T_EC3
- Status: VERIFIED
- Claim: Independent raw predictions recalculation for Table 2 and Table 3.
- Direct Evidence:
  - Command: `python 05_实验代码与脚本/audit_raw_predictions_recalculation.py` exited with code 0.
  - Report artifact: `06_实验结果与日志/audit_raw_predictions_recalculation_report.json` with status PASS (audited 2,070 records across seeds {42, 123, 456, 789, 1024} across Kelmarsh, Hill of Towie, Penmanshiel).
  - Main manuscript numerical concordance: Table 2 (`tables/table2_dynamic_settlement.tex`), Abstract, Highlights, and Section 5 centerpiece verified with exact 5-seed metrics (Penmanshiel authentic prior: $0.09726$, $-1.28\%$; scrambled prior: $0.15884$, $+61.2\%$).

## Task-ID: T_EC4
- Status: VERIFIED
- Claim: Independent 2,000x7-day circular block bootstrap reproduction from raw daily time series.
- Direct Evidence:
  - Command: `python 05_实验代码与脚本/constraint_round2/evaluate_dynamic_price.py --bootstrap_reps 2000` executed and exited with code 0.
  - Output artifacts: `06_实验结果与日志/constraint_round2/analysis/dynamic_price_paired_intervals.csv` (38 contrasts) and `dynamic_price_settlement_table2.csv`.
  - Audit script: `python 05_实验代码与脚本/audit_circular_bootstrap_reproduction.py` exited with code 0.
  - Report artifact: `06_实验结果与日志/audit_circular_bootstrap_report.json` with status PASS, verifying zero discrepancy on all 8 core Table 3 contrasts (Hill of Towie and Kelmarsh for Bounds, Curve, Distribution, Ridge).

## Task-ID: T_EC5
- Status: VERIFIED
- Claim: Independent Elexon raw price series recalculation (mean, peak, 68 negative periods, Zero-Bid diagnostic).
- Direct Evidence:
  - Command: `python 05_实验代码与脚本/audit_elexon_market_recalculation.py` exited with code 0.
  - Report artifact: `06_实验结果与日志/audit_elexon_market_report.json` with status PASS.
  - Verified 2017 System Buy Price parameters: 17,520 settlement periods, mean = £44.27/MWh, peak = £1,509.80/MWh, negative periods = 68.
  - Verified Zero-Bid unhedged imbalance cashflow diagnostic: Kelmarsh = +£223,882.14/(turb·yr), Hill of Towie = +£239,299.30/(turb·yr), Penmanshiel = +£243,449.80/(turb·yr), all with zero shortfall penalty (£0.00).

## Task-ID: T_EC6
- Status: VERIFIED
- Claim: Recompile active manuscript LaTeX, upgrade REVIEW_REPORT to PASS, and close verify_gate.py.
- Direct Evidence:
  - Command: `bibtex main; pdflatex -interaction=nonstopmode main.tex; pdflatex -interaction=nonstopmode main.tex` in `01_论文投稿材料/active_manuscript` exited with code 0.
  - Output PDF: `01_论文投稿材料/active_manuscript/main.pdf` (33 pages, 1,973,215 bytes).
  - Forbidden pattern "Penmanshiel: Specificity in Data Scarcity" checked and confirmed absent.
  - `.agent_state/REVIEW_REPORT.md` upgraded to PASS (SUBMISSION-READY).
  - `.agent_state/VERIFICATION_REPORT.md` generated with PASS.
  - Verification script `.agents/scripts/verify_gate.py` passes all checks with exit code 0.

## Task-ID: T_NP1
- Status: VERIFIED
- Claim: Global search, classification, and elimination of legacy numerical values (1.22%, 68.1%, 0.09881, 0.09760, 0.08812, 0.08924, 0.10473, 0.10461) across manuscript and figures.
- Direct Evidence:
  - Ripgrep pattern audits across `01_论文投稿材料/active_manuscript/` and `01_论文投稿材料/chinese_manuscript/`.
  - Stale occurrences classified: legacy 3-seed Table 2 leftovers vs legitimate subsampled $N=5000$ learning curve slice means ($0.08812 \pm 0.00022$ and $0.08936 \pm 0.00023$).
  - Eliminated stale values across all occurrences; canonical 5-seed values verified in `06_实验结果与日志/ablation_3d_results/ablation_3d_key_comparison_table.csv`.

## Task-ID: T_NP2
- Status: VERIFIED
- Claim: Regenerate Figure 2 directly from canonical 5-seed master CSV with matching annotations.
- Direct Evidence:
  - Generator script: `05_实验代码与脚本/generate_figure2_causal_ablation.py` executed with exit code 0.
  - Direct read from `06_实验结果与日志/ablation_3d_results/ablation_3d_key_comparison_table.csv`.
  - Output vector PDF: `01_论文投稿材料/active_manuscript/figures/fig_causal_ablation.pdf` regenerated with Penmanshiel authentic prior delta $-1.28\%$, scrambled prior $+61.2\%$, and Kelmarsh unconstrained baseline $0.08804 \pm 0.00006$.

## Task-ID: T_NP3
- Status: VERIFIED
- Claim: Audit Figure 4 and Section 6.2 boundary drift narrative for canonical baseline consistency.
- Direct Evidence:
  - Generator script `05_实验代码与脚本/plot_learning_and_corruption.py` updated (axhline $0.09852$ and $0.08804$) and executed with exit code 0, regenerating `active_manuscript/figures/fig_prior_corruption_gradient.pdf`.
  - Text updated: `01_论文投稿材料/active_manuscript/sections/06_boundaries.tex:47` ($0.09833$ vs.\ $0.09852$).
  - Lag scan numbers in `sections/06_boundaries.tex:52` updated to canonical 5-seed: HoT reference $0.08913$, $k=6$ ($0.09011 \pm 0.00008$), $k=36$ ($0.09246 \pm 0.00006$); Kelmarsh $k=6$ ($0.08867$), $k=36$ ($0.09034$); Penmanshiel $k=6$ ($0.09996$), $k=36$ ($0.10425$).

## Task-ID: T_NP4
- Status: VERIFIED
- Claim: Update Discussion 8.1 legacy percentages and fix Abstract subject-verb agreement (gains ... remain).
- Direct Evidence:
  - `01_论文投稿材料/active_manuscript/sections/08_discussion.tex:9`: replaced $-1.22\%$ with $-1.28\%$ on Penmanshiel, and $+68.1\%$ with $+61.2\%$.
  - `01_论文投稿材料/active_manuscript/main.tex:86`: corrected `observed gains cannot be specifically attributed to physical mapping and remains consistent...` to `...and remain consistent...`.

## Task-ID: T_NP5
- Status: VERIFIED
- Claim: Harmonize Table 4 and Table 5 Penmanshiel values with canonical baseline and document explicit provenance.
- Direct Evidence:
  - `01_论文投稿材料/active_manuscript/tables/table4_zero_bid_diagnostic.tex`: Kelmarsh DLinear updated to $0.08804$, shortfall \pounds 5,764.03, cashflow +\pounds 6,584.74; HoT DLinear updated to $0.08913$, shortfall \pounds 5,941.22, cashflow +\pounds 8,678.09; Penmanshiel DLinear updated to $0.09852$, shortfall \pounds 6,658.86, cashflow +\pounds 6,377.76; Penmanshiel Calibrated Power Curve updated to $0.09726$, shortfall \pounds 6,682.66, cashflow +\pounds 6,382.43.
  - Added Note (4) to Table 4 explicitly establishing 5-seed provenance: "DLinear and physical prior rows report multi-seed means across five synchronized random seeds (42, 123, 456, 789, 1024), directly concordant with the canonical benchmarks in Table 2."
  - Verified Table 5 (`table5_patchtst_and_rules.tex`) and Section 6.4 PatchTST cost ($0.11134$) consistency.

## Task-ID: T_NP6
- Status: VERIFIED
- Claim: Extend verify_gate.py with automated cross-artifact canonical numerical consistency check.
- Direct Evidence:
  - `.agents/scripts/verify_gate.py`: added `check_canonical_numerical_consistency(root, failures)` asserting concordance across Abstract, Table 4, Figure 4 script, Figure 2 script, Section 6 boundaries, Section 7 forensics, and Section 8 discussion.

## Task-ID: T_NP7
- Status: VERIFIED
- Claim: Recompile main.pdf, pass extended verify_gate.py, update ledgers, and return line-by-line audit.
- Direct Evidence:
  - `pdflatex -interaction=nonstopmode main.tex` executed twice with exit code 0.
  - Output PDF: `01_论文投稿材料/active_manuscript/main.pdf` (33 pages, 881,483 bytes).
  - `.agents/scripts/verify_gate.py` executed and exited with code 0 (VERIFICATION_GATE: PASS).


## Task-ID: T_GH1
- Status: VERIFIED
- Claim: PAT authenticated; private repo b1ue13e/pcwpf-attribution created.
- Direct Evidence:
  - `gh api user --jq ...` returned `{"login":"b1ue13e","name":"Junyu Li","plan":"free"}`.
  - `gh repo view b1ue13e/pcwpf-attribution --json ...` returned `{"name":"pcwpf-attribution","visibility":"PRIVATE","default":"main"}`.
  - URL: https://github.com/b1ue13e/pcwpf-attribution

## Task-ID: T_GH2
- Status: VERIFIED
- Claim: Local git identity set to GitHub account.
- Direct Evidence:
  - `git config --local user.name` -> `b1ue13e`; `git config --local user.email` -> `3211049186@qq.com` (command output captured).

## Task-ID: T_GH3
- Status: VERIFIED
- Claim: .gitignore rewritten (UTF-8) with heavy/personal exclusions; addable set within GitHub limits.
- Direct Evidence:
  - `.gitignore` written via Python explicit UTF-8 (2228 bytes); earlier Write-tool attempt produced mojibake and was replaced (detected via byte inspection).
  - `git check-ignore -v` matched all 8 probe paths (10_模型检查点与权重/, 06_实验结果与日志/results/, constraint_round2/*, 04_数据集/*, 09_项目申请_简历_个人材料/, 99_待确认归档/, 07_审稿Rebuttal备用包/, tmp_*/).
  - `git ls-files --others --exclude-standard`: 3704 files, 193.2 MB total, max file 14.2 MB (`04_数据集/constraint_rebuild_v2/hill_of_towie.csv`) — below the 50 MB GitHub warning threshold.

## Task-ID: T_GH4
- Status: VERIFIED
- Claim: Full project snapshot committed and pushed to origin/main.
- Direct Evidence:
  - Commit `f80855f` "chore(repo): snapshot revision working tree with gitignore hygiene".
  - `git push -u origin revision:main` output: `* [new branch] revision -> main` (https://github.com/b1ue13e/pcwpf-attribution.git).
  - `gh repo view` pushedAt 2026-09-20T21:36:12Z.

## Task-ID: T_GH5
- Status: VERIFIED
- Claim: CI workflow and README added.
- Direct Evidence:
  - `.github/workflows/verify.yml` (1121 bytes) with verification-gate job (python verify_gate.py) and build-manuscript-pdf job (tectonic compile + upload-artifact).
  - `README.md` (3183 bytes) with structure table, verification/build instructions, data DOIs, CI badge.
  - Committed in `ci(repo): add GitHub Actions verify pipeline and project README`.

## Task-ID: T_GH6
- Status: VERIFIED
- Claim: CI green on main.
- Direct Evidence:
  - First run 35539217497 failed: tectonic installer unpacks binary to cwd (`tectonic: command not found`, exit 127) — gate job had already passed.
  - Fix commit moved binary to /usr/local/bin; run 35539277539 `{"conclusion":"success"}` with jobs verification-gate=success and build-manuscript-pdf=success.
  - Artifact `main-pdf` present: 609,874 bytes, not expired.

## Task-ID: T_GH7
- Status: VERIFIED
- Claim: Four tracking issues seeded from the adversarial/first-principles review.
- Direct Evidence:
  - https://github.com/b1ue13e/pcwpf-attribution/issues/1 .. /4 (titles: smooth-but-wrong 安慰剂对照; r 敏感性扫描; 经验包络判据化; 审稿后公开方案).

## Task-ID: T_GH8
- Status: VERIFIED
- Claim: Release snapshot with main.pdf created.
- Direct Evidence:
  - `gh release view v2026.09-r1` -> name "修订快照 v2026.09-r1（minor revision）", assets ["main.pdf"].
  - URL: https://github.com/b1ue13e/pcwpf-attribution/releases/tag/v2026.09-r1

## Task-ID: T_GH9
- Status: VERIFIED
- Claim: Local verification gate passes after restructure.
- Direct Evidence:
  - Command `python .agents/scripts/verify_gate.py` printed `VERIFICATION_GATE: PASS`, exit code 0.

## Task-ID: T_GH10
- Status: VERIFIED
- Claim: State ledgers closed and security advisory issued.
- Direct Evidence:
  - This EVIDENCE_LEDGER section plus DECISION_LOG 2026-09-21 entries and RUN_STATE.json update.
  - Final response carries token-rotation advisory (PAT `ghp_...` was pasted in plaintext chat; never written to any repo file).
