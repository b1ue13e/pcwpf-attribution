## Cloud 4090 AE Revision Bundle v3

This bundle is for the revised `Applied Energy` operation-aware study on `Wind (CN)`.

The revised workflow does four things:

1. retrains `Ultra-LSNT + FG-MoE` with corruption-aware, tail-aware, and ramp-aware objectives;
2. exports `raw / clip-only / full_guardrail` prediction archives;
3. evaluates them under a `shared high-risk` dispatch screen instead of the older model-specific high-risk slice.
4. replays the same seed-42 operation stream under a `calibrated uq_holdout` readout based on chronological one-sided residual margins, now using a rolling bias-corrected residual envelope rather than a static regime-level margin.

## Quick Start

```bash
unzip cloud_4090_ae_revision_operation_bundle_v3.zip
cd cloud_4090_ae_revision_operation_bundle_v3
pip install -r requirements_cloud_4090.txt
bash run_ae_operation_full_suite_4090.sh
```

For a small environment check first:

```bash
bash run_ae_operation_quick_smoke.sh
```

## What Changes In This Revision

- `run_pi_moe_full_training_evidence.py` now trains with:
  - mixed SCADA corruption augmentation
  - positive-tail overforecast penalty in the same normalized output space as the forecasting MSE
  - horizon ramp-alignment penalty in the same normalized output space as the forecasting MSE
  - clean-vs-corrupted consistency regularization on normalized forecasts
  - staged warmup before the full regularization weights are activated
- `run_dispatch_closure_mapping_decision_4090.py` now supports:
  - `--risk-subset-mode shared|model_specific`
  - `high_risk_shared` aggregation
  - `--decision-mode point|uq_holdout`
  - chronological holdout calibration with rolling bias-corrected one-sided residual margins
  - extra burden metrics such as positive-error proxy
- `build_decision_aware_dispatch_evidence.py` now supports both `operation` and `operation_uq` summaries, and can emit a manuscript-ready LaTeX table snippet for the calibrated holdout readout
- `aggregate_ae_operation_results.py` now excludes decoupling directories from the main seed-stability summary

## Included Assets

- `wind_final.csv`
- `checkpoints_ts/main/*`
- `checkpoints_ts/iTransformer_best.pth`
- `checkpoints_ts/TimeMixer_preds.npy`
- `results/supplementary_evidence/ultra_predictions_main_fulltest.npz`
- `results/supplementary_evidence/dlinear_predictions_full.npz`
- `pretrained_guardrails/windcn_ultra_fg_seed42.pth`
- `pretrained_guardrails/windcn_itransformer_fg_seed42.pth`
- `pretrained_guardrails/windcn_itransformer_fg_seed123.pth`
- `pretrained_guardrails/windcn_itransformer_fg_seed2025.pth`

## Main Outputs

- `results/ae_operation_guardrails/`
  - revised Ultra-LSNT + FG-MoE checkpoints and training logs
- `results/ae_operation_exports/`
  - exported prediction archives
- `results/ae_operation_dispatch/`
  - hourly/daily/aggregate dispatch results
- `results/ae_operation_summary/`
  - main summary, calibrated UQ holdout summary, manuscript-ready UQ table snippet, decoupling summary, and seed stability summary

The main manuscript-oriented files are:

- `results/ae_operation_summary/operation_high_risk_summary_seed42.csv`
- `results/ae_operation_summary/operation_uq_holdout_summary_seed42.csv`
- `results/ae_operation_summary/operation_uq_holdout_summary_seed42_table.tex`
- `results/ae_operation_summary/fig_operation_high_risk_seed42.pdf`
- `results/ae_operation_summary/fig_operation_uq_holdout_seed42.pdf`
- `results/ae_operation_summary/operation_seed_stability_mean_std.csv`

The supplementary decoupling file is:

- `results/ae_operation_summary/operation_decoupling_summary_seed42.csv`

## GPU Layout

`run_ae_operation_full_suite_4090.sh` assumes 10 single-card processes:

- `GPU0`: quick smoke and final summaries
- `GPU1-3`: revised Ultra-LSNT guardrail training plus export for seeds `42 / 123 / 2025`
- `GPU4-6`: iTransformer export for seeds `42 / 123 / 2025`
- `GPU7`: `risk_conditional` main run, decoupling supplementary, then `risk_conditional_uq_holdout`
- `GPU8`: `quantile` sensitivity plus `risk_conditional` seed `123`
- `GPU9`: `affine` baseline plus `risk_conditional` seed `2025`

## Important Notes

- The main operation-aware table now uses `high_risk_shared`, not the old model-specific high-risk slice.
- The calibrated UQ holdout table now uses `all_holdout` as the primary readout, while `high_risk_shared_holdout` is treated as a stress slice.
- The default UQ rerun now uses `rolling_bias_scaled` margins with `holdout_start_day=80`, `rolling_window_days=14`, `min_history_samples=6`, `hour_band_radius=1`, `bias_quantile=0.50`, `bias_shrink=0.75`, `score_floor_quantile=0.25`, `score_floor_min_mw=5`, `horizon_smooth_window=3`, and `max_relative_margin=0.75`.
- Dispatch remains secondary evidence; this revision is designed to surface controller-side burden metrics more clearly.
- `Wind (US)` is still treated as boundary evidence and is not part of this bundle.
- `requirements_cloud_4090.txt` does not install PyTorch; the cloud machine is expected to have CUDA and PyTorch ready.
