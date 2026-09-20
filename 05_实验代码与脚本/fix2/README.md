# Fix2 Pack - 2026-03-07 (Chronos/Moirai Context Fix)

This folder contains the fixes for the "unfair baseline" issue where Chronos and Moirai showed negative R2 in zero-shot settings due to insufficient context length.

## Directory Structure

### `scripts/`
- `run_chronos_family_windcn.py`: Updated script that now uses `model_seq_len=512` (Extended Context) while strictly aligning the prediction target window with the standard 96-step protocol.
- `run_moirai_family_windcn.py`: Updated script implementing the same "Long Context Input, Standard Target Output" logic for Moirai, including input shape fixes for GluonTS.
- `build_windcn_context_audit_summary.py`: The aggregator script.

### `tex/`
- `main_PI-MoE_final_AE_ready.tex`: The final manuscript with **Table 2 Panel A** updated. Chronos and Moirai now show positive, high R2 (>0.96) on Clean data, correctly reflecting their status as "Cloud Upper Bound" models.

### `results/`
- `chronos_family_windcn.csv`: New audit logs showing R2 ~0.96 (Clean) and ~0.88 (Severe).
- `moirai_family_windcn.csv`: New audit logs showing R2 ~0.97 (Clean) and ~0.89 (Severe).
- `windcn_context_audit_summary.csv`: The consolidated table.
- `table2_panel_a_rows.tex`: The LaTeX rows generated from the above data.

## Key Changes
- **Context Length**: Increased from 96 to 512 for TSFMs to allow proper zero-shot pattern matching.
- **Target Alignment**: Strictly maintained the original 24-step target window corresponding to the 96-step protocol, ensuring fairness.
- **Result**: "Cloud Upper Bound" baselines now actually look like upper bounds (positive R2), eliminating the "unfair comparison" critique.
