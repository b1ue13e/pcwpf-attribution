# ARM 评测结果汇总

本文件夹包含 ARM 端部署评测的所有结果文件。

## 执行时间
- 日期: 2026-03-03
- 执行时长: ~64秒

## 评测配置
- 设备: CPU
- 批次大小: 1
- 测试场景: clean, gaussian, drift, quantization, missingness
- 延迟迭代次数: 100 (warmup 20)
- 鲁棒性测试批次: 64

## 测试模型
1. Ultra-LSNT (no guardrail)
2. Ultra-LSNT + clip-only (posthoc)
3. FG-MoE (full-trained)

## 文件说明

### 核心结果文件
| 文件名 | 说明 |
|--------|------|
| `arm_revision_latency_energy.csv` | ARM端延迟和能量消耗测量结果 |
| `arm_revision_robustness_smoke.csv` | 鲁棒性烟雾测试结果 (5种场景 x 3个模型) |
| `arm_revision_eval.meta.json` | 评测元数据和配置信息 |

### 汇总数据
| 文件名 | 说明 |
|--------|------|
| `latency_fixed.csv` | 所有模型的延迟汇总表 (含12个模型) |
| `latency_arm_full_backfill.csv` | ARM延迟回退审计文件 |

### 日志文件
| 文件名 | 说明 |
|--------|------|
| `arm_eval_*.log` | ARM评测执行日志 |
| `arm_backfill_*.log` | 数据汇总执行日志 |

## 关键结果摘要

### 延迟性能 (CPU, batch_size=1)
| 模型 | 平均延迟 (ms) | P95延迟 (ms) | 吞吐量 (Hz) |
|------|---------------|--------------|-------------|
| FG-MoE (full-trained) | 37.29 | 40.89 | 26.82 |
| Ultra-LSNT (no guardrail) | 40.31 | 44.26 | 24.81 |
| Ultra-LSNT + clip-only | 43.15 | 52.97 | 23.18 |

### 能量消耗
- 能量测量方法: estimated_constant_power (30W)
- FG-MoE: 1118.88 mJ/推理
- Ultra-LSNT (no guardrail): 1209.49 mJ/推理
- Ultra-LSNT + clip-only: 1294.61 mJ/推理

### 鲁棒性指标 (FG-MoE在clean场景)
- R²: 0.360
- RMSE: 7210.54
- MAE: 5803.86
- 超额定功率率: 0%
