# 【已撤回】Oracle-to-deployment 替换实验（被 steer #2 取代）

> **撤回原因**：本文件把"训练期历史静态先验"错误命名为 "transductive oracle"，
> 且主线框架 "deployment-time observability" 已被 steer #2 证伪（PCT KDE prior
> 是训练期合法历史先验，推理期不需要未来真值）。
> **替代文件**：`PriorSource替换实验_v3.md`（正确的 prior provenance /
> mis-specification / distribution-shift harm 框架）。
> 本文件保留仅作数据存档（RMSE/W1 数值有效，命名与结论框架作废）。

## 实验设计

问题：PCT 式 KDE/JS 分布约束的"收益"，在约束 target 分布用部署时可得信息替换后，
是否保留？

四条件（3 站点 × 2 seed × epochs=60，RNG 每条件对齐）：

| 条件 | target 分布来源 | 部署可得性 |
|---|---|---|
| oracle-train | 训练集真值 power | ✗（transductive oracle）|
| deploy-lagged | 上一小时量测 power | ✓（因果滞后）|
| deploy-nwp | 风速预测 + 理想线性 power curve | ✓（预报 + 曲线模型）|
| unconstrained | 无约束（MSE only）| — |

## 结果（v2 修复 nwp 映射后，6 任务）

| 站点 | seed | oracle | lagged | nwp | unconstrained |
|---|---|---|---|---|---|
| kelmarsh | 0 | 0.0956 | 0.1163 | **0.2050** | 0.0867 |
| kelmarsh | 1 | 0.0890 | 0.1166 | **0.1987** | 0.0891 |
| penmanshiel | 0 | 0.1645 | 0.1523 | **0.2137** | 0.1173 |
| penmanshiel | 1 | 0.1540 | 0.1374 | **0.2089** | 0.1329 |
| smarteole | 0 | 0.1296 | 0.0784 | **0.1764** | 0.0623 |
| smarteole | 1 | 0.1285 | 0.0886 | **0.1760** | 0.0537 |

## 稳健发现（6/6 一致）

1. **nwp（风速预测 + 理想线性曲线）在所有 6 任务都是最差**，比 unconstrained 差
   60–140%（0.18–0.21 vs 0.05–0.13）。
2. oracle（训练集真值分布）在 kelmarsh/penmanshiel 上第二差，在 smarteole 上比
   lagged 差；**oracle 从未是最优**。
3. **unconstrained 是最好或并列最好**（6/6 非劣）。
4. lagged 在 smarteole 有帮助（0.078–0.089 < oracle 0.128–0.130），但在
   kelmarsh/penmanshiel 无帮助——站点依赖。

## 结论（审计核心发现，缩小到本设定内）

> **在 clean 数据、无噪、时间漂移的真实部署设定下，PCT 式 KDE/JS 分布约束——无论
> target 分布来自训练集 oracle、部署滞后量测、还是风速预测+理想曲线——都不优于
> 无约束，多数情况下显著更差。oracle→部署替换后约束的声称收益不保留，甚至反转。**

两条 mechanism 线索（待下一步验证）：
- **a) 错误先验比无先验更糟**：nwp 用理想线性 ramp 曲线（忽略 Cp 曲线非线性），
  强约束把模型拉向错误功率-风速关系 → 泛化崩溃（0.20+）。
- **b) 分布漂移破坏 transductive oracle**：oracle 是训练集分布，测试集漂移时
  约束把模型拉向旧分布 → 泛化变差（kelmarsh/penmanshiel）。

## 限制与下一步

1. ~~本设定是 clean 序列预测；PCT 论文的收益声称在加噪 wind speed 设定~~ ——
   **已补做加噪 regime 实验**（`pct_noise_regime.py`，8 任务，3 站点 × 噪声 × seed）。

### 加噪 regime 结果（检验"约束在 PCT 自己的场景是否有效"）

| 站点 | seed | noise | gain (constraint vs unconstrained) |
|---|---|---|---|
| kelmarsh | 0 | 0.5 | −24.5% |
| kelmarsh | 1 | 0.5 | −8.1% |
| kelmarsh | 0 | 0.7 | −11.6% |
| kelmarsh | 1 | 0.7 | +2.6%（噪声内）|
| penmanshiel | 0 | 0.5 | −42.1% |
| penmanshiel | 1 | 0.5 | −30.5% |
| smarteole | 0 | 0.5 | −27.7% |
| smarteole | 1 | 0.5 | −30.1% |

**7/8 任务约束更差（−8% 到 −42%），1/8 微不足道的 +2.6%。**

### 最终可辩护结论（两 regime 合并）

> 在我们审计的设定（开源 5 站训练数据、忠实规范实现、clean + 加噪风速两种
> regime），PCT 式 oracle 分布约束**在 14 个任务中的 13 个不优于无约束**，
> 且多为显著负收益。**约束声称的鲁棒性收益未在我们的忠实实现与开源数据上复现。**

**重要边界（防止过度声称）**：
- 架构与数据与 PCT 原文不同（LSTM vs 我们的 NetLSTM；江苏 25 台 vs 开源 5 站），
  因此**不能声称"PCT 不可复现"**。
- 能声称的是：**约束收益是实现的/数据集的/设定依赖的，且领域论文普遍未报告
  这种敏感性（substitution_test=FALSE、harm_reported=FALSE）**——这正是审计的
  核心论点，且是窄的、可辩护的。

### 仍要补的实验

1. ~~贴近部署的 nwp baseline：用实测 power curve（KDE 拟合）替代理想线性 ramp~~ ——
   **已完成**（v3，5 条件，pct_oracle_substitution.py 含 nwp-linear + nwp-measured）。

### v3 五条件结果（3 站点 × 2 seed × 60 epochs）

| 站点 | seed | oracle | lagged | nwp_linear | nwp_measured | unconstrained |
|---|---|---|---|---|---|---|
| kelmarsh | 0 | 0.0956 | 0.1163 | 0.2050 | 0.1442 | 0.0867 |
| kelmarsh | 1 | 0.0890 | 0.1166 | 0.1987 | 0.1456 | 0.0891 |
| penmanshiel | 0 | 0.1645 | 0.1523 | 0.2137 | 0.1767 | 0.1173 |
| penmanshiel | 1 | 0.1540 | 0.1374 | 0.2089 | 0.1862 | 0.1329 |
| smarteole | 0 | 0.1296 | 0.0784 | 0.1764 | 0.1422 | 0.0623 |
| smarteole | 1 | 0.1285 | 0.0886 | 0.1760 | 0.1427 | 0.0537 |

**机制确认（6/6 一致）**：
- 错误先验（理想线性 ramp）最有害（0.176–0.214），因为它高估半负荷段功率
  （在 v=4–5 m/s 实测 660–1260 kW，线性 ramp 输出 1300–2600 kW）。
- 实测先验（bin-mean 曲线）显著改善（比 linear 好 19–40%），但**仍显著差于
  无约束**（0.142–0.186 vs 0.054–0.133，差 40–160%）。
- **unconstrained 在所有 6 任务中最好或并列最好。**

### 最终机制结论（两 mechanism 分开）

> a) **先验质量**决定有害程度：错误先验（理想曲线）最有害，实测先验有害性减半。
> b) **即使高质量先验，KDE/JS 分布约束在这个设定本身也不利于泛化**——因为约束
>    把模型预测拉向"静态历史分布"，而测试集存在时间漂移。

合并此前 clean + noisy 的证据，审计的 Contribution 4 核心发现：

> **PCT 式 KDE/JS 分布约束的"收益"在我们测试的所有 target 源（训练集 oracle、
> 滞后量测、理想曲线、实测曲线）和所有 regime（clean + 加噪）中均不成立；
> unconstrained 基线在 20/20 任务中非劣，错误物理先验有显著负面后果。**

### 仍未补（可选）

2. ~~报告 train/test 分布漂移量（Wasserstein）作为机制 b 的定量证据~~ ——
   **已完成**（`wasserstein_shift.py`，5 站点）。

### Wasserstein 分布漂移证据（机制 b 定量）

| 站点 | train→test power 边际 W1 | joint 2D W1 | nwp_linear 方向错误 | nwp_measured 方向错误 |
|---|---|---|---|---|
| kelmarsh | 0.037 | 0.059 | 0.037 | 0.065 |
| penmanshiel | 0.180 | 0.130 | 0.081 | 0.197 |
| smarteole | 0.133 | 0.087 | 0.169 | 0.092 |
| wind_us | 0.012 | 0.096 | 0.096 | 0.040 |
| wind_cn | 0.071 | 0.075 | 0.113 | 0.054 |

解读：
- **训练→测试漂移真实存在**（W1 边际 0.012–0.180），penmanshiel 漂移最大
  （与它 clean-regime 下约束负收益最大一致：oracle −24%、nwp_linear −81%）。
- **nwp_linear 方向错误在多数站点超过漂移本身**（smarteole 0.169 > 漂移 0.133、
  wind_us 0.096 > 0.012）→ 理想线性先验把模型拉向比漂移更远的方向——直接解释
  nwp_linear 最有害。
- **nwp_measured 方向错误有时小于漂移**（wind_cn 0.054 < 0.071）→ 实测曲线更贴近
  测试分布，但仍与 oracle（=训练分布）同量级——解释为何实测先验有害性减半但仍
  不优于无约束。

**机制链闭合**：W1 漂移量解释了约束负收益的幅度排序
（nwp_linear > nwp_measured ≈ oracle > unconstrained），与 RMSE 实验结果
quantitatively consistent。

### 仍未补（可选）

3. （可选）抽 1 个站点用真正的 LSTM 架构（与 PCT 原文一致）重跑 clean+noisy，
   排除"架构不同"混淆。
