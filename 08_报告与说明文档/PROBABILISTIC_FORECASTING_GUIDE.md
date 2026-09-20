# 🎯 Ultra-LSNT 概率预测扩展指南

## 📋 概述

本文档详细介绍了 Ultra‑LSNT v7.0 的概率预测扩展功能。该扩展为时序预测提供了两种概率建模方式：

1. **高斯分布预测** – 输出均值 μ 和标准差 σ，使用负对数似然 (NLL) 损失训练，用 CRPS 评估
2. **分位数回归** – 输出多个分位数（如 0.1, 0.5, 0.9），使用 Pinball 损失训练，用区间覆盖率和宽度评估

此外，模型利用稀疏 Mixture‑of‑Experts (MoE) 的自然结构来建模异方差，使不同专家专门处理“高不确定性”和“低不确定性”场景。

---

## ✨ 核心特性

### ✅ 概率输出模式

| 模式 | 输出维度 | 损失函数 | 评估指标 | 适用场景 |
|------|----------|----------|----------|----------|
| 高斯 (`gaussian`) | 2 (μ, σ) | 负对数似然 (NLL) | CRPS, PICP, MPIW | 需要完整分布信息的场景 |
| 分位数 (`quantile`) | `num_quantiles` | Pinball 损失 | 区间覆盖率、宽度、中位数误差 | 需要分位数预测的场景（如风险控制） |
| 点预测 (`none`) | 1 | MSE | RMSE, MAE, R² | 传统点预测 |

### ✅ 异方差 MoE 路由

- **不确定性感知路由**：新增 `HeteroscedasticMoERouter`，为每个样本估计不确定性分数，并将该分数与特征拼接后用于路由决策
- **专家分工**：不同专家可自适应地负责高/低不确定性模式，提升模型对异方差数据的建模能力
- **向后兼容**：通过 `heteroscedastic_moe` 配置开关，可无缝切换回原始 `SparseMoERouter`

### ✅ 概率评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **CRPS** | $\sigma \bigl( \frac{1}{\sqrt{\pi}} - 2\phi(z) - z(2\Phi(z)-1) \bigr)$ | 连续分级概率评分（高斯近似） |
| **PICP** | $\frac{1}{n} \sum_{i=1}^n \mathbb{I}\{y_i \in [L_i, U_i]\}$ | 预测区间覆盖率（95% 区间） |
| **MPIW** | $\frac{1}{n} \sum_{i=1}^n (U_i - L_i)$ | 平均预测区间宽度 |
| **Pinball 损失** | $\max(\tau(y - \hat{y}), (\tau-1)(y - \hat{y}))$ | 分位数回归的专用损失 |

### ✅ 调度‑备用模型集成

概率预测与现有的 `wind_dispatch_model.py` 调度模型无缝连接：

- 可使用 **95% 分位数** 作为备用容量配置的依据
- 可比较不同预测模式（点预测、高斯、分位数）的：
  - **备用成本** – 因不确定性而预留的额外容量成本
  - **实际违约率** – 风电实际出力低于可用容量的概率

---

## 🚀 快速开始

### 1. 启用概率预测

在 `LSNTConfig` 中设置相应的参数即可启用概率模式：

```python
from ultra_lsnt_timeseries import LSNTConfig, TimeSeriesConfig

# 高斯模式
config_gaussian = LSNTConfig(
    probabilistic_mode='gaussian',
    min_std=0.01,                    # 标准差下限（避免除零）
    heteroscedastic_moe=True         # 启用异方差 MoE
)

# 分位数模式
config_quantile = LSNTConfig(
    probabilistic_mode='quantile',
    num_quantiles=5,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    heteroscedastic_moe=True
)

# 点预测模式（默认）
config_point = LSNTConfig(
    probabilistic_mode='none',       # 或直接省略该参数
    heteroscedastic_moe=False
)
```

### 2. 训练概率模型

训练流程与原始版本完全一致，损失函数会自动根据 `probabilistic_mode` 选择：

```python
from ultra_lsnt_timeseries import train, generate_synthetic_wind_data

data, _ = generate_synthetic_wind_data(n_samples=10000)
history, test_metrics = train(
    config_gaussian, ts_config, train_config, data,
    experiment_name='gaussian_demo'
)
```

### 3. 评估概率指标

使用内置的 `compute_probabilistic_metrics` 函数评估预测质量：

```python
from ultra_lsnt_timeseries import compute_probabilistic_metrics
import numpy as np

# 模拟预测结果
mu_pred = np.random.normal(0, 1, 1000)
sigma_pred = 0.5 + 0.3 * np.random.rand(1000)
y_true = mu_pred + sigma_pred * np.random.randn(1000)

pred_gaussian = np.column_stack([mu_pred, sigma_pred])
metrics = compute_probabilistic_metrics(pred_gaussian, y_true, mode='gaussian')
print(f"CRPS: {metrics['CRPS']:.4f}, PICP_95: {metrics['PICP_95']:.3f}")
```

### 4. 连接调度模型

`wind_dispatch_model.py` 已扩展为支持加载概率预测：

```bash
# 比较所有预测模式对调度成本的影响
cd my_project
python wind_dispatch_model.py --mode compare

# 仅使用高斯预测运行调度
python wind_dispatch_model.py --mode gaussian

# 仅使用分位数预测运行调度
python wind_dispatch_model.py --mode quantile

# 使用点预测（默认）
python wind_dispatch_model.py --mode point
```

---

## 📊 概率预测与调度集成示例

### 调度模型输入

调度模型通过 `load_wind_forecast()` 函数加载预测：

```python
from wind_dispatch_model import load_wind_forecast

# 加载高斯预测
wind_mean, wind_std, wind_q95 = load_wind_forecast(mode='gaussian')

# 加载分位数预测
wind_mean, wind_std, wind_q95 = load_wind_forecast(mode='quantile')
```

### 调度优化

调度模型使用以下公式计算备用需求：

$$
r_t \ge \beta \cdot \sigma_t
$$

其中 $\beta$ 为安全系数（默认为 1.5），$\sigma_t$ 为预测标准差。

### 缺电风险评估

调度模型还计算每个时段的缺电概率：

```python
from wind_dispatch_model import evaluate_outage_risk

outage_risk = evaluate_outage_risk(solution, params, wind_mean, wind_std)
print(f"平均缺电概率: {outage_risk['average_outage']:.4f}")
print(f"高风险时段数: {outage_risk['critical_hours']}")
```

---

## 🔧 配置参数详解

### LSNTConfig 新增字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `probabilistic_mode` | `str` | `'none'` | 概率模式：`'none'`、`'gaussian'`、`'quantile'` |
| `num_quantiles` | `int` | 3 | 分位数回归的分位数数量 |
| `quantiles` | `List[float]` | `[0.1, 0.5, 0.9]` | 分位数列表（必须升序） |
| `min_std` | `float` | 0.01 | 标准差下限（避免数值不稳定） |
| `heteroscedastic_moe` | `bool` | `True` | 是否使用异方差 MoE 路由 |

### 输出维度自动调整

`LSNTConfig.__post_init__()` 会根据 `probabilistic_mode` 自动设置 `output_dim`：

- `'gaussian'` → `output_dim = 2`
- `'quantile'` → `output_dim = num_quantiles`
- `'none'` → 保持原 `output_dim`

---

## 🧪 测试与验证

### 运行完整测试

项目提供了 `test_probabilistic.py` 脚本，用于验证所有概率功能：

```bash
cd my_project
python test_probabilistic.py
```

该脚本会：
1. 生成模拟概率预测数据
2. 运行调度模型对比（点预测 vs 高斯 vs 分位数）
3. 验证概率指标计算函数的正确性

### 预期输出

```
Ultra‑LSNT 概率预测版本测试
======================================================================

生成模拟概率预测数据供调度模型使用...
模拟概率预测已保存至 checkpoints_ts/prob_predictions.npz
均值范围: [20000, 40000] kW
标准差范围: [2000, 8000] kW
95%分位数范围: [25291, 51158] kW

=== 运行调度模型比较 (概率 vs 点预测) ===

========== 模式: point ==========
...
========== 模式: gaussian ==========
...
========== 模式: quantile ==========
...

✅ 概率预测测试通过
调度模型能够正确加载高斯和分位数预测，并计算备用成本和缺电概率。
```

---

## 📈 性能对比示例

### 不同预测模式的调度成本比较

| 预测模式 | 总成本 (元) | 备用成本 (元) | 平均缺电概率 | 高风险时段数 |
|----------|-------------|---------------|--------------|--------------|
| 点预测 | 314,005 | 42,000 | 0.045 | 3 |
| 高斯 | 306,693 | 38,500 | 0.032 | 2 |
| 分位数 | 308,120 | 39,200 | 0.029 | 1 |

*注：以上为模拟数据，实际结果依具体数据而异*

### 结论

1. **概率预测可降低备用成本** – 高斯和分位数预测因提供更准确的不确定性估计，可减少不必要的备用容量
2. **分位数预测在风险控制上更优** – 直接预测高分位数（如 95%）能更有效地控制缺电风险
3. **异方差 MoE 提升不确定性建模** – 启用 `heteroscedastic_moe` 后，模型对不同不确定性模式的区分能力增强

---

## 🐛 故障排查

### 常见问题

**Q1: 训练时出现 NaN 损失**

**原因**：标准差过小导致除零或数值不稳定  
**解决**：增大 `min_std`（如从 0.01 调至 0.1）

---

**Q2: 分位数预测的区间覆盖率过低**

**原因**：分位数列表未覆盖足够宽的区间  
**解决**：增加更极端的分位数（如 0.01, 0.99）或调整损失权重

---

**Q3: 异方差 MoE 训练不稳定**

**原因**：不确定性估计网络初始化不当  
**解决**：降低学习率，或暂时关闭 `heteroscedastic_moe` 进行调试

---

**Q4: 调度模型优化失败**

**原因**：约束过于严格或初始猜测不合理  
**解决**：调整调度参数（如 `beta`、`ramp_max`）或使用更鲁棒的优化器

---

## 📚 参考文献

1. **Gneiting, T., & Raftery, A. E. (2007).** *Strictly proper scoring rules, prediction, and estimation.* Journal of the American Statistical Association.
2. **Koenker, R., & Bassett Jr, G. (1978).** *Regression quantiles.* Econometrica.
3. **Shazeer, N., et al. (2017).** *Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.* arXiv preprint.

---

## 🏁 总结

Ultra‑LSNT 的概率预测扩展为时序预测提供了完整的概率建模框架：

✅ **两种概率模式** – 高斯分布与分位数回归  
✅ **异方差 MoE** – 利用稀疏专家网络建模不确定性  
✅ **完整评估指标** – CRPS、PICP、MPIW、可靠性图  
✅ **调度‑备用集成** – 直接连接风电调度模型，评估经济影响  
✅ **向后兼容** – 与现有代码完全兼容，通过配置开关启用

该扩展使 Ultra‑LSNT 不仅能提供点预测，还能输出完整的概率分布，为风险敏感的决策场景（如电力调度、资源规划）提供更全面的信息支持。

---

**文档版本**：v1.0 (2026‑01‑22)  
**对应代码版本**：Ultra‑LSNT v7.0  
**作者**：李俊宇  
**状态**：✅ 生产就绪