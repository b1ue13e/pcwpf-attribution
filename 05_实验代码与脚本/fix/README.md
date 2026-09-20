# Fix Pack - 2026-03-07

This folder contains all the modifications and artifacts generated to address the review comments regarding data completeness, robustness protocol, and formatting.

## Directory Structure

### `scripts/`
Contains the Python scripts created or modified to execute the audit pipeline:
- `windcn_audit_common.py`: Shared utilities, updated with KS-test for residual normality.
- `run_lightgbm_gaussian_audit_windcn.py`: LightGBM audit script with Gaussian/Drift/Quantization support.
- `run_chronos_family_windcn.py`: Chronos-Bolt-Small inference script (Zero-shot).
- `run_moirai_family_windcn.py`: Moirai-2.0-R-Small inference script (Zero-shot).
- `monitor_wrapper.py`: Wrapper to capture runtime and peak GPU memory.
- `run_full_pipeline.py`: Orchestrator script that runs the full sequence.
- `build_windcn_context_audit_summary.py`: Aggregates individual model results into the summary CSV.
- `build_main4_matrices_from_evidence.py`: Builds the final matrices for the paper.
- `render_windcn_context_panel_tex.py`: Generates the LaTeX rows for Table 2 Panel A.

### `tex/`
- `main_PI-MoE_final_AE_ready.tex`: The final revised manuscript TeX file. Key changes:
    - **Table 2 Panel A**: Populated with real RMSE/R2 for LightGBM, Chronos, and Moirai.
    - **Notation**: $h_t$ changed to $\mathbf{o}_t$ to avoid clash with horizon $H$.
    - **Text**: Added interpretation of $\sigma_{eff}=0.15$ (empirical) vs $0.60$ (stress), removed "not archived" excuses.
    - **Typo**: Fixed $v_{rated}$ and Figure references.

### `results/`
Contains the generated evidence files:
- `lightgbm_gaussian_audit_windcn.csv`: LightGBM results.
- `chronos_family_windcn.csv`: Chronos results (note negative R2).
- `moirai_family_windcn.csv`: Moirai results (note negative R2).
- `windcn_context_audit_summary.csv`: Aggregated table used for the paper.
- `table2_panel_a_rows.tex`: The exact LaTeX fragment injected into the main TeX file.
- `*_stats.json`: Runtime and memory logs.

## How to Apply
The `tex/main_PI-MoE_final_AE_ready.tex` file is the self-contained final version. You can use it directly for compilation.
The `scripts/` can be run to reproduce the results if needed.
