# Conflict Resolution Log (2026-03-03)

Canonical directory: `results/supplementary_evidence`

## Decision rule
- Select the better or more complete version for each conflicting main-table experiment file.
- Copy winner into canonical directory.
- Delete conflicting counterpart from `applied_energy_final_results/supplementary_evidence`.

## File decisions
- `main4_matrix2_perf_robust_transfer.csv`: kept `results` version.
- `main4_matrix3_hardware_safety_joint.csv`: selected `applied_energy_final_results` version and copied into `results`.
- `main4_matrix4_dispatch_value.csv`: selected `applied_energy_final_results` version and copied into `results`.
- `multi_domain_transfer_matrix_all_models.csv`: selected complete version (restored canonical values matching main table) in `results`.
- `joint_parameter_full_windcn.csv`: selected `applied_energy_final_results` version (full grid) and copied into `results`.
- `joint_sensitivity_pareto_frontier.csv`: kept `results` version.
- `dispatch_rolling_with_ultra_summary.csv`: kept `results` version.
- `dispatch_raw_vs_mapped_summary.csv`: copied into `results` as canonical source.

## Deleted conflicting files from applied pack
- `applied_energy_final_results/supplementary_evidence/main4_matrix2_perf_robust_transfer.csv`
- `applied_energy_final_results/supplementary_evidence/main4_matrix3_hardware_safety_joint.csv`
- `applied_energy_final_results/supplementary_evidence/main4_matrix4_dispatch_value.csv`
- `applied_energy_final_results/supplementary_evidence/multi_domain_transfer_matrix_all_models.csv`
- `applied_energy_final_results/supplementary_evidence/joint_parameter_full_windcn.csv`
- `applied_energy_final_results/supplementary_evidence/joint_sensitivity_pareto_frontier.csv`
- `applied_energy_final_results/supplementary_evidence/dispatch_rolling_with_ultra_summary.csv`
- `applied_energy_final_results/supplementary_evidence/dispatch_raw_vs_mapped_summary.csv`

## Verification
- Main-table numeric checks: PASS (see `verification_report_main_tables_20260303.md`).
- Main-table source notes now point to `results/supplementary_evidence` only.
