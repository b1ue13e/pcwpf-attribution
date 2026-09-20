## Cloud 4090 AE Fast Revision Bundle

This bundle is the fast-revision package for the `Applied Energy` submission.

It is designed to do four things:

1. rerun the `Wind (CN)` `FG-MoE proxy` sensitivity cases on top of the audited `Ultra-LSNT` base checkpoint;
2. export `raw / clip-only / full_guardrail` archives for each proxy setting;
3. rerun the shared high-risk dispatch screen for each proxy case and aggregate one-row manuscript summaries;
4. run one lightweight supplementary `dispatch parameter sensitivity` sweep over reserve cost and storage capacity using the canonical `iTransformer` and `Ultra-LSNT` archives.

## Quick Start

```bash
unzip cloud_4090_ae_fast_revision_bundle_v1.zip
cd cloud_4090_ae_fast_revision_bundle_v1
pip install -r requirements_cloud_4090.txt
bash run_ae_fast_revision_full_suite_4090.sh
```

For a minimal environment check:

```bash
bash run_ae_fast_revision_quick_smoke.sh
```

## Included Assets

- `wind_final.csv`
- `checkpoints_ts/main/*`
- `results/supplementary_evidence/dlinear_predictions_full.npz`
- `checkpoints_ts/TimeMixer_preds.npy`
- canonical `Wind (CN)` archives
  - `upload_results_ae_operation_20260328_104442/ae_operation_exports/windcn_ultra/seed_42/*`
  - `upload_results_ae_operation_20260328_104442/ae_operation_exports/windcn_itransformer/seed_42/prediction_archive_raw_*`
  - `upload_results_ae_operation_20260328_104442/ae_operation_exports/windcn_itransformer/seed_42/prediction_archive_full_guardrail_*`
- canonical revision summaries
  - `results/supplementary_evidence/fgmoe_clip_only_control_windcn.csv`
  - `results_export_cfg_lower_corr_20260329_012420/stage1_summary_cfg_lower_corr/operation_high_risk_summary_seed42.csv`
  - `results_export_cfg_lower_corr_20260329_012420/stage1_summary_cfg_lower_corr/operation_decoupling_summary_seed42.csv`
- canonical manifests
  - `manifests/ae_fast_revision_seed42_main_manifest.json`
  - `manifests/ae_fast_revision_seed42_decoupling_manifest.json`

## Main Commands

- `run_ae_proxy_sensitivity_case.py`
  - one proxy configuration at a time
  - phases: `prep`, `dispatch`, `all`
- `aggregate_ae_proxy_sensitivity_results.py`
  - merges all case summaries into `results/ae_revision_summary/ae_proxy_sensitivity_summary.csv`
- `run_ae_dispatch_param_sensitivity.py`
  - supplementary reserve-cost/storage-capacity sweep on the canonical `iTransformer` and `Ultra-LSNT` streams

## Output Layout

- `results/ae_proxy_sensitivity/`
  - per-case training outputs, archives, dispatch reruns, and one-row summaries
- `results/ae_dispatch_param_sensitivity/`
  - long-format and pairwise reserve/storage sensitivity summaries
- `results/ae_revision_summary/`
  - aggregated proxy sensitivity summary for the manuscript revision

## GPU Layout

`run_ae_fast_revision_full_suite_4090.sh` assumes 10 single-card workers:

- `GPU0`: smoke, `baseline_fixed` prep, final aggregation
- `GPU1-6`: the six perturbation prep runs
- `GPU7-8`: shared high-risk dispatch readouts for the proxy cases
- `GPU9`: lightweight dispatch parameter sensitivity

## Important Notes

- The main manuscript keeps the `FG-MoE-first` framing.
- The proxy sweep is the only mandatory new training experiment in this bundle.
- The reserve/storage sweep is supplementary support; it does not replace the main fixed-interface Table 5 readout.
- The canonical `iTransformer + FG-MoE` archive is reused as existing evidence and is not retrained here.
- `Wind (US)` and UQ-holdout reruns are not rerun by this package; they remain existing boundary evidence.
- `requirements_cloud_4090.txt` is expected to be used in an environment where CUDA and PyTorch are already available.
