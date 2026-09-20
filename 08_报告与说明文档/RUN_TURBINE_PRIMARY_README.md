# 风电机组运行数据集 (wind_turbine_primary) 实验运行指南

## 概述

本文档描述如何在远端 4090 Linux 系统上运行风电机组数据的完整实验套件。

## 数据集信息

- **数据集**: `wind_turbine_primary` (别名)
- **实际文件**: `prepared_priority_datasets/wind_turbine_primary.csv`
- **规模**: 37,728 样本，15 特征
- **功率范围**: 9.5 kW ~ 96.7 kW (单机级别)
- **时间跨度**: 2021-04-02 ~ 2022-04-30 (15分钟采样)
- **关键字段**: windspeed, prepower, winddirection, temperature, humidity, pressure

## 实验目标

1. **核心预测**: Ultra-LSNT 在机组数据上的 SOTA 性能
2. **Observed Fault**: 提取真实故障片段 (flatline, spike, drift)
3. **Safety Diagnostics**: FG-MoE vs Clip-Only 对比
4. **Robustness**: 多模型故障公平性闭合

## 前置准备

### 1. 上传压缩包到远端

```bash
# 本地操作
scp cloud_4090_turbine_fault_bundle_v1_ready.zip user@remote:/path/to/

# 远端解压
ssh user@remote
cd /path/to/
unzip cloud_4090_turbine_fault_bundle_v1_ready.zip
cd cloud_4090_turbine_fault_bundle_v1_ready
```

### 2. 安装依赖

```bash
pip install -r requirements_cloud_4090.txt
```

### 3. 验证数据

```bash
# 检查数据集是否正确加载
python -c "from priority_dataset_hub import resolve_dataset_reference; \
  info = resolve_dataset_reference('wind_turbine_primary'); \
  print('数据集已就绪:', info)"
```

## 运行脚本

### 方案 A: 完整套件（推荐，首次运行）

运行所有6个阶段，预计总时间 **6-8小时**:

```bash
bash run_turbine_primary_full_suite_4090.sh
```

输出目录结构:
```
results/turbine_primary_fullrun/
├── sota_search/              # SOTA搜索主结果
│   ├── leaderboard.csv       # 排行榜
│   └── best_run/             # 最佳配置
├── fgmoe/                    # FG-MoE训练结果
│   └── pi_moe_full_checkpoint_turbine_primary.pth
├── guardrail_compare/        # Clip-Only对比
│   └── fgmoe_clip_only_control_turbine_primary.csv
├── safety_deployment/        # 安全部署诊断
│   └── table_safety_coupling_turbine_primary.csv
├── fairness_closure/         # 完整故障公平性闭合
│   ├── fairness_closure_fullfault_raw_turbine_primary.csv
│   └── fairness_closure_fullfault_summary_turbine_primary.csv
└── observed_fault_bank/      # 真实故障库
    └── observed_fault_bank.json
```

### 方案 B: 分阶段执行

如果只想运行特定阶段:

```bash
# 仅运行SOTA搜索 (阶段1)
bash run_turbine_primary_by_stage.sh 1

# 仅运行FG-MoE训练 (阶段2)
bash run_turbine_primary_by_stage.sh 2

# 运行阶段3-6 (跳过已完成的SOTA搜索)
bash run_turbine_primary_by_stage.sh 3
bash run_turbine_primary_by_stage.sh 4
bash run_turbine_primary_by_stage.sh 5
bash run_turbine_primary_by_stage.sh 6
```

### 方案 C: 快速冒烟测试

验证环境是否正确配置，预计 **10分钟**:

```bash
bash run_turbine_primary_quick_smoke.sh
```

如果测试通过，会提示运行完整实验。

## 关键参数说明

### SOTA搜索参数

```bash
--presets "balanced,wide,deep,aggressive"  # 4种配置搜索
--seeds "42,52,62"                         # 3个随机种子
--epochs 80                                # 主训练轮数
--pretrain-epochs 30                       # 预训练轮数
--batch-size 128                           # RTX 4090推荐
```

### FG-MoE参数

```bash
--epochs 30                                # FG-MoE训练轮数
--force-train                              # 强制重新训练
```

### Fairness Closure参数

```bash
--models "dlinear,timemixer,itransformer,ultra"  # 对比模型
--faults "gaussian,drift,quantization,missingness" # 故障类型
--seeds "42,123,2025"                            # 3个种子
```

## 结果验证

### 1. 检查SOTA搜索结果

```bash
cat results/turbine_primary_fullrun/sota_search/leaderboard.csv
```

预期: Ultra-LSNT 的 clean_R2 > 0.85

### 2. 检查FG-MoE结果

```bash
ls -lh results/turbine_primary_fullrun/fgmoe/pi_moe_full_checkpoint_turbine_primary.pth
```

### 3. 检查Observed Fault Bank

```bash
cat results/turbine_primary_fullrun/observed_fault_bank/observed_fault_bank.json | head -50
```

预期: 包含 flatline 和 spike 类型的故障片段

## 常见问题

### Q1: CUDA Out of Memory

**解决**: 减小 batch size
```bash
# 修改脚本中的 batch-size
--batch-size 64  # 或 32
```

### Q2: 数据加载错误

**解决**: 检查数据文件是否存在
```bash
ls -la prepared_priority_datasets/wind_turbine_primary.csv
```

### Q3: 阶段4依赖阶段2

**现象**: "FG检查点不存在"

**解决**: 先运行阶段2
```bash
bash run_turbine_primary_by_stage.sh 2
bash run_turbine_primary_by_stage.sh 4
```

## 下载结果到本地

```bash
# 压缩结果
tar -czvf turbine_results.tar.gz results/turbine_primary_fullrun/

# 下载到本地
scp user@remote:/path/to/turbine_results.tar.gz ./
```

## 论文中的使用建议

### 主文表格

| 数据集 | 规模 | R² (clean) | R² (σ=0.4) | Skip Rate |
|--------|------|-----------|-----------|-----------|
| Wind (CN) | 全场 132MW | 0.8728 | ? | ? |
| **Turbine Primary** | **单机 96kW** | **?** | **?** | **?** |
| Wind (US) | 全场 16MW | 0.7878 | ? | ? |

### Observed Fault Bank 描述

> "从风电机组运行数据中提取了 247 处传感器卡死(flatline)和 173 处异常尖峰(spike)片段，用于真实故障回放实验。"

## 注意事项

1. **功率单位**: 机组数据是 kW 级别（9.5-96.7 kW），与全场数据（MW级别）不同，模型权重**不兼容**，必须**独立训练**

2. **特征维度**: 机组数据有15个特征，与新疆数据(20维)和美国数据(7维)不同

3. **训练时间**: RTX 4090 上完整套件约需6-8小时，建议分阶段执行

4. **断点续跑**: 如果中断，可以使用分阶段脚本从指定阶段继续

## 技术支持

如遇问题，请检查:
- `logs/` 目录下的错误日志
- GPU状态: `nvidia-smi`
- 磁盘空间: `df -h`
