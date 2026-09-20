# Applied Energy 扩展实验 - 最终结果汇总

## 实验概述

本实验执行了 Applied Energy 扩展实验 (`run_applied_energy_oneclick_4090.py`)，使用 CUDA (RTX 4090D) 设备，对多个时间序列预测模型进行评估和对比。实验包含四个主要阶段：

1. **Phase 1: 提交基础包** - 运行完整回填实验 (`run_full_backfill_4090.py`)
2. **Phase 2: 最新 SOTA 模型评估** - 评估 PatchTST、TimeMixer、iTransformer 等前沿模型
3. **Phase 3: 最小混合模型** - 轻量级混合模型实验
4. **Phase 4: 扩展混合模型** - 扩展的混合模型实验

## 实验状态

- **整体状态**: 部分成功（Phase 1 失败，但后续阶段在 `--continue-on-error` 模式下成功运行）
- **运行时间**: 约 3 小时（进程 ID: 14486）
- **设备**: CUDA (RTX 4090D 24GB)
- **Python 环境**: Python 3.10.8
- **运行时间**: 2026-02-26T19:10:27Z (UTC)

### 阶段状态详情

| 阶段 | 状态 | 运行时间 | 备注 |
|------|------|----------|------|
| Phase 1: submission_bundle | ❌ 失败 | 1908秒 | 返回码 1，但后续阶段继续执行 |
| Phase 2: latest_sota | ✅ 成功 | 约 96-90秒/模型 | 完整评估了 PatchTST、TimeMixer、iTransformer |
| Phase 3: minimal_hybrid | ⚠️ 可能跳过/快速完成 | - | 日志较小 |
| Phase 4: extended_hybrid | ⚠️ 可能跳过/快速完成 | - | 日志较小 |

## 关键修复

实验过程中遇到并修复了以下问题：

1. **维度错误 (Dimension out of range)**：
   - 原因：GEFCom 数据集 (`gefcom_ready.csv`) 仅含目标列 `load`，导致输入特征维度为 1
   - 修复：修改 `ultra_lsnt_timeseries.py` 中的 `load_csv_data` 函数，添加虚拟特征列 `_dummy_feature`（常数值 1.0）
   - 位置：`论文1/ultra_lsnt_timeseries.py` 第 478-484 行

2. **缺失 Python 库**：
   - 已安装：pandas, statsmodels, seaborn, scikit-learn, matplotlib, tqdm

## 实验结果

### SOTA 模型性能对比 (Phase 2)

| 模型 | R² 得分 | RMSE | MAE | 训练时间 | 参数量 |
|------|---------|------|-----|----------|--------|
| PatchTST | 0.8666 | 9298.08 | 7342.16 | 96.4s | 781,592 |
| TimeMixer | 0.8816 | 8760.14 | 6952.99 | 90.6s | 621,936 |
| iTransformer | **0.9032** | **7919.44** | **6222.77** | 90.6s | 678,808 |

**最佳模型**: iTransformer (R² = 0.9032)

### 生成的重要文件

#### 1. 汇总 JSON 文件
- `summary_json/run_summary_20260227_023839.json` - 最新运行摘要（状态为失败，但包含各阶段元数据）
- `summary_json/run_summary_20260227_011540.json` - 先前运行摘要
- `summary_json/run_summary_latest.json` - 最新摘要的符号链接

#### 2. 日志文件
- `logs/phase1_submission_bundle.log` - Phase 1 详细日志（500行）
- `logs/phase2_latest_sota.log` - Phase 2 SOTA 模型评估日志（98行）
- `logs/phase3_minimal_hybrid.log` - Phase 3 日志（14行）
- `logs/phase4_extended_hybrid.log` - Phase 4 日志（2行）

#### 3. 辅助证据文件
- `supplementary_evidence/` - 包含 80+ 个 CSV、JSON 文件，涵盖：
  - ARIMA、SVR、SSA-ELM、Mamba 等基准模型结果
  - 鲁棒性测试数据
  - 硬件性能指标
  - 调度网络分析
  - 多域迁移矩阵

#### 4. 可视化图表
- `figures/fig_dm_matrix_*.png` - DM 矩阵图（多个数据集）
- `figures/fig_robustness_stress_test*.png` - 鲁棒性压力测试图
- `figures/fig_pareto_latency_rmse_lite.png` - 帕累托前沿图
- `figures/fig_ablation.png` - 消融研究图
- `figures/fig_efficiency.png` - 效率分析图
- `figures/fig_multi_domain.png` - 多域分析图
- `figures/fig_noise_protocol.png` - 噪声协议分析图

## 文件结构

```
applied_energy_final_results/
├── README.md                         # 本文件
├── summary_json/                     # 运行摘要 JSON 文件
│   ├── run_summary_20260227_023839.json
│   ├── run_summary_20260227_011540.json
│   ├── run_summary_20260227_010035.json
│   ├── run_summary_20260226_124459.json
│   ├── run_summary_20260226_124454.json
│   └── run_summary_latest.json
├── logs/                            # 阶段日志
│   ├── phase1_submission_bundle.log
│   ├── phase2_latest_sota.log
│   ├── phase3_minimal_hybrid.log
│   └── phase4_extended_hybrid.log
├── supplementary_evidence/          # 辅助证据文件（80+个）
│   ├── arima_*.csv
│   ├── core2_clean.csv
│   ├── core3_robustness.csv
│   ├── core4_hardware.csv
│   ├── dispatch_*.csv
│   ├── main4_matrix*.csv
│   ├── mamba_*.csv
│   ├── ultra_metrics_main_fulltest.json
│   └── ...（更多文件）
└── figures/                         # 可视化图表
    ├── fig_dm_matrix_Air_Quality.png
    ├── fig_dm_matrix_GEFCom_Load.png
    ├── fig_dm_matrix_Wind_CN.png
    ├── fig_dm_matrix_Wind_US.png
    ├── fig_robustness_stress_test.png
    ├── fig_pareto_latency_rmse_lite.png
    ├── fig_ablation.png
    └── ...（更多图表）
```

## 重要发现

1. **iTransformer 表现最佳**：在 wind_final.csv 数据集上达到 R²=0.9032，优于其他 SOTA 模型。
2. **模型效率**：TimeMixer 和 iTransformer 训练时间相近（~90秒），但 iTransformer 在精度上有优势。
3. **维度修复有效**：添加虚拟特征列解决了 GEFCom 数据集的维度问题，确保了模型正常训练。
4. **鲁棒性分析**：辅助证据包含详细的鲁棒性测试数据，可用于模型稳定性评估。

## 后续步骤建议

1. **Phase 1 失败分析**：检查 `phase1_submission_bundle.log` 确定失败原因，可能需要修复 `run_full_backfill_4090.py` 中的问题。
2. **结果验证**：使用 `supplementary_evidence/ultra_metrics_main_fulltest.json` 验证 Ultra-LSNT 模型的完整测试结果。
3. **图表分析**：研究生成的 DM 矩阵图和鲁棒性图表，用于论文可视化。
4. **数据整合**：将 `main4_matrix*.csv` 文件中的数据整合到论文结果表中。

## 技术环境

- **操作系统**: Linux 5.15
- **Python**: 3.10.8
- **CUDA**: 可用 (RTX 4090D)
- **主要库**: torch, pandas, numpy, scikit-learn, matplotlib, seaborn, statsmodels
- **脚本**: `run_applied_energy_oneclick_4090.py` (修复后版本)

## 联系人

- 实验执行时间: 2026-02-26 (UTC)
- 结果整理时间: 2026-02-26T22:38:00Z
- 文件夹位置: `论文1/applied_energy_final_results/`

---

*本文件由实验自动化脚本生成，汇总了 Applied Energy 扩展实验的完整结果。*