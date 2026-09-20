# Fix1 Pack - 2026-03-07 (DLinear/Ultra-LSNT RMSE Audit)

This folder contains the specific fixes for the missing DLinear/Ultra-LSNT RMSE audit data in Table 2 Panel A.

## Directory Structure

### `scripts/`
- `run_dlinear_ultra_audit_windcn.py`: The new script that trains and audits DLinear and Ultra-LSNT under the exact same protocol (Clean vs Severe 0.60), generating RMSE/R2/MAE.
- `build_windcn_context_audit_summary.py`: Updated summary builder that now consumes `dlinear_ultra_audit_windcn.csv` to populate the main table rows.

### `tex/`
- `main_PI-MoE_final_AE_ready.tex`: The final revised manuscript. **Table 2 Panel A** is now fully populated with consistent, real-world audit data for all 8 models (including DLinear and Ultra-LSNT's severe-noise RMSE).

### `results/`
- `dlinear_ultra_audit_windcn.csv`: The raw audit log showing DLinear's performance collapse (R2 ~ -0.99, RMSE ~46k) and Ultra-LSNT's stability (R2 ~0.79, RMSE ~15k) under severe noise.
- `windcn_context_audit_summary.csv`: The consolidated evidence table used for the paper.
- `table2_panel_a_rows.tex`: The generated LaTeX fragment verifying the exact numbers in the TeX file.

## Verification
You can verify the data flow by checking `fix1/results/dlinear_ultra_audit_windcn.csv` against the rows in `fix1/tex/main_PI-MoE_final_AE_ready.tex`.
