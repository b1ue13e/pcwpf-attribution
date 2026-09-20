# Methods/statistical adversarial audit (active manuscript)

Scope: `01_论文投稿材料/active_manuscript/` sections 03/04/05/06/08/09 and linked raw artifacts in `06_实验结果与日志/ablation_3d_results`, `learning_curve`, `information_budget`. No manuscript edits made.

## Verified evidence

1. **Protocol controls are explicitly specified.** `sections/04_methodology.tex:47-53` states backbone/parameter parity, synchronized splits, form/weight-matched scrambled priors, and validation-year (2016) scalar calibration before 2017 test. `:59-67` enumerates lambda grid, 500-sample budget matching, learning curves, corruption/drift, lag scans, and PatchTST probe. `:70` claims 2,000-replication 7-day circular block bootstrap; the claim is present in manuscript, but executable bootstrap code was not located in the restricted artifact paths (see UNKNOWN).

2. **Learning-curve and information-budget raw CSVs genuinely contain five seeds.** `06_实验结果与日志/learning_curve/within_site_learning_curve_results.csv` has 300 rows, seeds {42,123,456,789,1024}; `within_site_learning_curve_summary.csv` reproduces manuscript values: at N=500 calibrated cost Kelmarsh unconstrained 0.096476±0.003136, genuine 0.094550±0.002854; Hill of Towie 0.104439±0.003123 vs 0.100660±0.003072. At N=5000, unconstrained 0.088121 vs genuine 0.088411 (Kelmarsh), and 0.089361 vs 0.090909 (Hill). This supports the reported attenuation/reversal numerically for these two tables.

3. **Information-budget CSV also has five seeds and reproduces table values.** `information_budget_experiment_results.csv` has 100 rows and seeds {42,123,456,789,1024}; summary means/std match Table information_budget values, including Kelmarsh A/B/C/D/E and Hill A/B/C/D/E.

4. **Raw ablation_3d master coverage is not five seeds uniformly.** `ablation_3d_master_summary.csv` has 1,380 rows with seeds 42,43,44,45 overall; per-site files show Hill of Towie seeds 42–44 only, Penmanshiel seeds 42–44 only, Kelmarsh seeds 42–45 only. Thus the main table note claiming five seeds (Table `table_information_budget.tex:6,31`) is supported by the separate information-budget/learning-curve CSVs, but not by the ablation_3d export set.

5. **The manuscript itself has a seed-schedule contradiction.** `sections/04_methodology.tex:51` says synchronized seeds “42, 43, 44, 45”, while `tables/table_information_budget.tex:31` says five seeds “42, 123, 456, 789, 1024”. This must be resolved before claiming a single canonical protocol.

6. **Theoretical claims are appropriately bounded in places.** `sections/03_theory.tex:101` calls Term III a diagnostic caveat and says it is not empirically estimated; `:119` labels formal counterexamples as limits of scalar calibration. `sections/06_boundaries.tex:25,28` explicitly says bias–variance interpretation and joint N×lambda tendency are conceptual/not formally identified. `sections/08_discussion.tex:29-35` disclaims cross-site topographic causation, universal physics effects, and universal calibration dominance. These are VERIFIED claim-boundary safeguards.

## First-principles/adversarial findings

### Strongly supported (within tested scope)

- Five-seed means/std for the learning-curve and information-budget claims are directly reproducible from raw CSVs. The negative control is severe and consistent at lambda=1.0 across tested N in the learning-curve summary.
- The strongest “domain-specific mapping under sample restriction” result is only a conditional empirical effect: one-dimensional empirical priors, specific sites, DLinear/PatchTST settings, lambda choices, and 2017 test timeline. The manuscript mostly states this scope, so universal physics language would be an overclaim.

### Blockers / high risks

1. **Canonical seed inconsistency (HIGH, VERIFIED).** Method section and information-budget notes name different seed sets. This undermines exact reproducibility and makes pooled standard deviations/paired comparisons ambiguous.

2. **Unequal seed coverage in ablation_3d exports (HIGH, VERIFIED).** Hill/Penmanshiel have 3 seeds and Kelmarsh 4 in `ablation_3d_results`, despite methods/table language often implying 5. Any claim relying on those exports needs explicit n and cannot inherit five-seed uncertainty.

3. **Bootstrap provenance is UNKNOWN.** The manuscript states 2,000 circular 7-day paired bootstrap replications (`04_methodology.tex:70`, `tables/table3_paired_bootstrap.tex:30`), but no bootstrap script/raw replicate archive was found in the audited paths. Table CIs therefore cannot be independently recomputed from the provided evidence here. This is a provenance gap, not proof of fabrication.

4. **Parameter-selection/data-adaptation risk (UNKNOWN→audit required).** The manuscript evaluates lambda ∈ {0.01,0.1,1.0} (`04_methodology.tex:60-66`) and highlights lambda=1.0 for centerpiece claims (`05_centerpiece.tex:6+`; `06_boundaries.tex:28`). No inspected file establishes whether lambda, lag grid, corruption crossover, or calibration policy were selected without looking at the 2017 test timeline. Validation calibration is specified, but selection provenance for the highlighted operating points is not demonstrated.

5. **Leakage cannot be closed from manuscript text alone (UNKNOWN).** `04_methodology.tex:53` asserts calibration on 2016 validation before 2017 test, and `:63` defines full-history priors (7–10 months), but the audited CSVs lack timestamps and prior-fitting manifests. We cannot verify that “full-history” excludes future/test records or that all preprocessing is fit only on train/validation.

6. **Statistical unit mismatch risk (INFERRED).** Seed-level SDs quantify optimization variance (n=5 in learning/info tables), whereas 7-day block bootstrap CIs quantify temporal uncertainty over one calendar year. The manuscript presents both but does not show a pre-specified hierarchy/multiple-comparison policy across many arms, sites, lambdas, lags, and stress scans. Nominal CIs should not be read as global confirmatory significance.

7. **Cross-backbone evidence is exploratory (VERIFIED).** `sections/06_boundaries.tex:57` says 60 PatchTST models at nominal lambda=0.01; `:61-63` reports mixed cashflow effects and Ridge/Persistence outperforming constrained deep models on Penmanshiel. This supports boundary heterogeneity, not a universal architecture-independent mechanism.

## Bottom-line judgment for parent audit

The central conditional claim (“under restricted samples and strong regularization, an authentic empirical prior can reduce asymmetric dispatch loss; scrambled form-matched controls can be harmful; advantage attenuates with more data”) is numerically supported by the learning-curve and information-budget CSVs, but the evidence package is not yet fully audit-closed. Seed schedules conflict, ablation exports have unequal n, and bootstrap/selection/leakage provenance is not directly recoverable from the inspected artifacts. Treat universal or causal ‘physics improves forecasting’ language as unsupported; retain tested-scope wording and expose exact n/seed sets per experiment.
