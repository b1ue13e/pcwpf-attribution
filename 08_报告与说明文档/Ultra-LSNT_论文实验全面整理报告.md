# Ultra-LSNT 论文实验结果整理

更新时间：2026-02-09  
项目路径：`c:\Users\32110\Desktop\论文`

---

## 1. Wind (CN) 主任务结果

数据来源：`results/metrics_corrected.json`

| 模型 | R2 | RMSE | MAE | MAPE |
|---|---:|---:|---:|---:|
| DLinear | 0.939990 | 6236.3862 | 4085.0898 | 12.2565 |
| Ultra-LSNT | 0.872773 | 9080.5547 | 7171.4165 | 27.7434 |
| Informer | 0.816659 | 10900.6514 | 8765.9189 | 32.5842 |
| Transformer | 0.743890 | 2916.8264 | 2074.5305 | 10.5305 |
| Autoformer | 0.721941 | 13424.2764 | 10646.7725 | 37.9394 |
| LSTM | 0.697142 | 3171.8770 | 2273.7424 | 11.9365 |
| MLP | 0.664953 | 3336.1848 | 2307.6277 | 11.7230 |

### 新 SOTA 对比

数据来源：  
`checkpoints_ts/PatchTST_latest_results.json`  
`checkpoints_ts/TimeMixer_latest_results.json`  
`checkpoints_ts/iTransformer_latest_results.json`  
`mamba_results.json`

| 模型 | R2 | RMSE | MAE | MAPE | 参数量 |
|---|---:|---:|---:|---:|---:|
| Mamba(S6) | 0.920769 | 7165.8765 | 4875.1675 | - | 331352 |
| iTransformer | 0.901151 | 8004.0488 | 6333.2939 | 24.8834 | 678808 |
| TimeMixer | 0.888563 | 8498.3838 | 6661.3589 | 25.8586 | 621936 |
| Ultra-LSNT | 0.872773 | 9080.5547 | 7171.4165 | 27.7434 | 34491317 |
| PatchTST | 0.866604 | 9298.0840 | 7342.1587 | 29.3740 | 781592 |

---

## 2. 多数据集泛化结果

数据来源：`table_multi_domain.csv`

| 数据集 | R2 | RMSE | MAE | Skip Rate |
|---|---:|---:|---:|---:|
| Wind (CN) | 0.8728 | 9080.5547 | 7171.4165 | 0.0% |
| Wind (US) | 0.7878 | 0.4556 | 0.2891 | 11.4% |
| Air Quality | 0.4861 | 32.2145 | 19.8663 | 0.0% |
| GEFCom Load | 0.7554 | 2836.9788 | 2049.4832 | 0.0% |

---

## 3. 鲁棒性结果（高斯噪声）

数据来源：`results/robust_gaussian_windcn.csv`

### 统一噪声点（0/0.1/0.2/0.3/0.4）

| sigma | DLinear | LightGBM | Ultra-LSNT |
|---:|---:|---:|---:|
| 0.0 | 0.937924 | 0.906939 | 0.872775 |
| 0.1 | 0.781369 | 0.885954 | 0.868680 |
| 0.2 | 0.295150 | 0.822163 | 0.855232 |
| 0.3 | -0.465927 | 0.730472 | 0.786563 |
| 0.4 | -1.579048 | 0.615562 | 0.726290 |

### Ultra-LSNT 额外噪声点

| sigma | R2 |
|---:|---:|
| 0.05 | 0.871884 |
| 0.15 | 0.862899 |

20%噪声专项对比（`noise_20_comparison.txt`）：  
DLinear = 0.2952，Ultra-LSNT = 0.8318，LightGBM = 0.8222

---

## 4. 消融实验

数据来源：`universal_ablation_results.csv`

| 数据集 | Kernel=3 | Kernel=7 | Kernel=15 |
|---|---:|---:|---:|
| Wind (CN) | 0.850532 | 0.859058 | 0.821409 |
| Air Quality | 0.277348 | 0.252772 | 0.353228 |

---

## 5. 效率与部署结果

### 主模型效率

数据来源：`efficiency_benchmark_results_enhanced.csv`

| 模型 | 设备 | avg inference (ms) | std (ms) | 参数量 |
|---|---|---:|---:|---:|
| Ultra-LSNT | CUDA | 15.2937 | 3.6130 | 34491317 |

### 轻量模型效率

数据来源：`timemixer_efficiency_results.csv`、`ultra_lsnt_lite_efficiency_results.csv`、`ultra_lsnt_lite_optimized_results.csv`

| 模型 | 参数量 | 模型大小(MB) | 单次推理(ms) | 平均推理(ms) |
|---|---:|---:|---:|---:|
| TimeMixer | 621936 | 2.3725 | 3.3583 | 7.2651 |
| Ultra-LSNT-Lite | 1342744 | 5.1222 | 8.6706 | 8.9449 |
| Ultra-LSNT-Lite-Optimized | 349368 | 1.3323 | 5.2100 | 6.5300 |

轻量优化模型精度（`ultra_lsnt_lite_optimized_performance.csv`）：  
R2 = 0.856429，RMSE = 9646.1963，MAE = 7611.0439，MAPE = 30.9535

### CPU 延迟（边缘侧参考）

数据来源：`dlinear_latency_results.csv`、`lightgbm_latency_results.csv`

| 模型 | 单次推理(ms) | 平均推理(ms) | 模型大小(MB) |
|---|---:|---:|---:|
| DLinear (CPU) | 0.2654 | 0.4596 | 0.0178 |
| LightGBM (CPU) | 50.3970 | 50.5903 | 27.1757 |

---

## 6. 统计显著性（DM 检验）

数据来源：`dm_complete_table.tex`、`dm_results/*.json`

| 比较 | DM统计量 | p值 | 结论 |
|---|---:|---:|---|
| Ultra-LSNT vs PatchTST | -5.8998 | 1.82e-09 | Ultra-LSNT 显著更优 |
| Ultra-LSNT vs TimeMixer | 3.45 | 2.83e-04 | TimeMixer 显著更优 |
| Ultra-LSNT vs iTransformer | 5.12 | 3.07e-07 | iTransformer 显著更优 |
| Ultra-LSNT vs Transformer (fair-1) | -1.9504 | 2.56e-02 | Ultra-LSNT 显著更优 |
| Ultra-LSNT vs Transformer (fair-2) | -2.9603 | 1.54e-03 | Ultra-LSNT 显著更优 |

---

## 7. 调度成本结果

数据来源：`dispatch_comparison_results.json`

| 模型 | 总成本 |
|---|---:|
| Ultra-LSNT | 583200.0000 |
| Informer | 621000.0000 |
| Autoformer | 626400.0000 |
| DLinear | 668667.7191 |

---

## 8. 基线结果（GEFCom）

### 深度学习基线（最新汇总）

数据来源：`baseline_results/baseline_summary.json`

| 模型 | R2 |
|---|---:|
| Transformer | 0.741983 |
| LSTM | 0.705399 |
| MLP | 0.665303 |
| Ultra-LSNT 参考值 | 0.751300 |

### 经典基线

数据来源：`baseline_results/classical_baselines_optimized.csv`

| 模型 | R2 | RMSE | MAE |
|---|---:|---:|---:|
| RandomForest | 0.5325 | 16212.6505 | 11033.6646 |
| Persistence | -26.3596 | 6360.7183 | 6243.3932 |
| Seasonal_Naive | -281.3579 | 20433.9121 | 20307.0040 |
| PowerCurve | -3.9060 | 52448.2672 | 46800.7495 |
| ARIMA | - | - | - |

---

## 9. 结果文件索引

- `results/metrics_corrected.json`
- `checkpoints_ts/PatchTST_latest_results.json`
- `checkpoints_ts/TimeMixer_latest_results.json`
- `checkpoints_ts/iTransformer_latest_results.json`
- `mamba_results.json`
- `table_multi_domain.csv`
- `results/robust_gaussian_windcn.csv`
- `noise_20_comparison.txt`
- `universal_ablation_results.csv`
- `efficiency_benchmark_results_enhanced.csv`
- `timemixer_efficiency_results.csv`
- `ultra_lsnt_lite_efficiency_results.csv`
- `ultra_lsnt_lite_optimized_results.csv`
- `ultra_lsnt_lite_optimized_performance.csv`
- `dlinear_latency_results.csv`
- `lightgbm_latency_results.csv`
- `dm_complete_table.tex`
- `dm_results/dm_patchtst_20260128_054441.json`
- `dm_results/dm_transformer_fair_20260128_051106.json`
- `dm_results/dm_transformer_fair_20260128_052455.json`
- `dispatch_comparison_results.json`
- `baseline_results/baseline_summary.json`
- `baseline_results/classical_baselines_optimized.csv`
