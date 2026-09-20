# Cross-Site Wind Generalization Runbook

## Prepared datasets

Unified 15-minute wind datasets are generated under:

`E:\论文1\04_数据集\prepared_priority_datasets`

Aliases:

- `wind_cn_core_15min`
- `wind_us_core_15min`
- `kelmarsh_uk_2016`
- `penmanshiel_scotland_2016`
- `smarteole_france_2020`

Each CSV uses:

`datetime, wind_speed, wind_direction, temperature, hour_sin, hour_cos, dayofweek_sin, dayofweek_cos, power`

## Rebuild preprocessing

```powershell
python E:\论文1\05_实验代码与脚本\prepare_cross_site_wind_datasets.py
```

Audit output:

`E:\论文1\06_实验结果与日志\results\cross_site_wind\cross_site_dataset_audit.csv`

## Check data and windows without training

```powershell
python E:\论文1\05_实验代码与脚本\run_cross_site_recalibration_experiments.py --check-only
```

## Smoke training run

Requires a Python environment with PyTorch.

```powershell
python E:\论文1\05_实验代码与脚本\run_cross_site_recalibration_experiments.py --smoke --device cuda --batch-size 32 --max-train-batches 2 --max-eval-batches 2
```

## Full experiment

```powershell
python E:\论文1\05_实验代码与脚本\run_cross_site_recalibration_experiments.py --device cuda --batch-size 64 --epochs 20 --recalibration-epochs 8 --resume
```

Main outputs:

- `cross_site_zero_shot_matrix.csv`
- `cross_site_recalibration_curve.csv`
- `cross_site_summary_for_paper.csv`
- `cross_site_experiment_manifest.json`

Use wording such as "cross-site boundary and recalibration validation" rather than claiming universal wind-farm generalization.
