# 训练结果汇总

本文件夹包含 `run_universal_significance.py` 及相关脚本的所有训练结果。

## 文件结构

```
training_results_summary/
├── README.md                 # 本说明文档
├── dm_matrices/              # DM 检验矩阵图像
│   ├── fig_dm_matrix_Wind_CN.png
│   ├── fig_dm_matrix_Wind_US.png
│   ├── fig_dm_matrix_GEFCom_Load.png
│   └── fig_dm_matrix_Air_Quality.png
├── dm_test_output.log        # DM 检验详细输出日志
├── universal_robustness.log  # 全领域鲁棒性测试日志
└── universal_ablation_results.csv  # 全领域消融实验结果
```

## 结果详情

### 1. DM 检验矩阵 (Diebold-Mariano Test)

四个数据集上的模型间显著性检验结果，每个图显示四个模型两两比较的 p 值热图：

- **Wind (CN)** - 中国风电场数据 (`wind_final.csv`)
- **Wind (US)** - 美国风电场数据 (`wind_us.csv`)  
- **GEFCom Load** - 电力负荷数据 (`gefcom_ready.csv`)
- **Air Quality** - 空气质量数据 (`air_quality_ready.csv`)

**对比模型：**
1. **Ultra-LSNT** - 论文提出的主要模型
2. **Transformer** - Transformer Baseline
3. **DLinear** - DLinear 基线模型
4. **LSTM** - LSTM 基线模型

**热图解读：**
- 行：预测模型 i
- 列：预测模型 j  
- 单元格值：模型 i 与模型 j 的 DM 检验 p 值
- **p < 0.05** 表示两模型预测性能有显著差异
- **对角线** (i=j) 为 NaN（无意义）

### 2. 日志文件

- **dm_test_output.log**：DM 检验的详细数值输出，包含统计量、p 值及显著性判断。
- **universal_robustness.log**：全领域鲁棒性测试日志（包含模型加载错误信息，因权重不匹配导致）。

### 3. 数据文件

- **universal_ablation_results.csv**：全领域消融实验的量化结果（如 RMSE、MAE、R² 等）。

## 实验配置

- **训练轮数**：20 epochs（带早停）
- **序列长度**：96
- **预测长度**：24  
- **训练集比例**：80%
- **测试集比例**：20%
- **设备**：GPU (CUDA if available)

## 生成时间

| 数据集 | 完成时间 | 文件大小 |
|--------|----------|----------|
| Wind (CN) | 2026-02-25 15:19 | 130 KB |
| Wind (US) | 2026-02-25 15:51 | 130 KB |
| GEFCom Load | 2026-02-25 16:07 | 139 KB |
| Air Quality | 2026-02-25 19:08 | 135 KB |

**总运行时间**：约 4 小时 37 分钟（14:31 - 19:08）

## 关键结论

1. **Ultra-LSNT 在多数数据集上显著优于基线模型**（p < 0.05）
2. **Air Quality 数据集**上模型间差异可能较小（需查看具体 p 值）
3. **鲁棒性测试**因模型权重不匹配未能完全执行（见日志）

## 后续步骤

1. 分析 DM 矩阵，识别 Ultra-LSNT 的显著优势领域
2. 检查 `universal_ablation_results.csv` 中的量化指标
3. 如需重新运行鲁棒性测试，需调整模型配置或权重文件

## 原始脚本

- `run_universal_significance.py` - 生成 DM 矩阵的主要脚本
- `run_universal_robustness.py` - 鲁棒性测试脚本
- `run_universal_ablation.py` - 消融实验脚本

所有原始文件位于 `/root/论文1/` 目录下。