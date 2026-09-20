# Cloud 4090 Missing-AE Bundle

这个上传包只覆盖 Applied Energy 当前最关键的缺失实验补强：

- `Wind_CN + iTransformer` 的训练态 guardrail 对比
- `Wind_US + Ultra-LSNT` 的跨站点 guardrail 对比
- `Wind_US + iTransformer` 的 base backbone 训练 + guardrail 对比
- `Wind_US + Ultra-LSNT` 的 safety-deployment audit

包内是自包含结构，解压后直接在根目录运行。

## Quick Start

```bash
unzip cloud_4090_missing_ae_bundle_v1.zip
cd cloud_4090_missing_ae_bundle_v1
pip install -r requirements_cloud_4090.txt
bash run_missing_ae_full_suite_4090.sh
```

如果只想先验环境：

```bash
bash run_missing_ae_quick_smoke.sh
```

## Assumptions

- 远端已安装 CUDA 版 PyTorch。
- `requirements_cloud_4090.txt` 只补充本包缺少的 Python 依赖，不主动覆盖现有 GPU PyTorch。
- 所有脚本默认在包根目录运行。

## Included Data

- `wind_final.csv`
- `wind_us.csv`

## Included Checkpoints

- `checkpoints_ts/main/best_model.pth`
- `checkpoints_ts/main/model_config.json`
- `checkpoints_ts/main/final_results.json`
- `checkpoints_ts/exp_wind_us/best_model.pth`
- `checkpoints_ts/exp_wind_us/model_config.json`
- `checkpoints_ts/exp_wind_us/final_results.json`
- `checkpoints_ts/iTransformer_best.pth`

## Included Code

- `guardrail_backbones.py`
- `guardrail_proxy_utils.py`
- `priority_dataset_hub.py`
- `ultra_lsnt_timeseries.py`
- `run_pi_moe_full_training_evidence.py`
- `run_clip_only_guardrail_control.py`
- `run_safety_deployment.py`
- `aggregate_missing_ae_results.py`
- `run_missing_ae_quick_smoke.sh`
- `run_missing_ae_full_suite_4090.sh`

## GPU Mapping

`run_missing_ae_full_suite_4090.sh` 默认按 10 张卡分配：

- GPU0: quick smoke 后跑 `Wind_US + Ultra-LSNT` safety audit
- GPU1-3: `Wind_CN + iTransformer` seeds `42/123/2025`
- GPU4-6: `Wind_US + iTransformer` seeds `42/123/2025`
- GPU7-9: `Wind_US + Ultra-LSNT` seeds `42/123/2025`

## Outputs

主结果目录：

- `results/missing_ae_guardrails/`
- `results/deployment_evidence_missing_ae/`

最终汇总文件：

- `results/missing_ae_guardrails/ae_missing_experiments_summary.csv`
- `results/missing_ae_guardrails/ae_missing_experiments_summary.json`

## Integrity

包根目录会包含：

- `bundle_manifest.txt`
- `SHA256SUMS.txt`

可用来检查是否漏传或损坏。
