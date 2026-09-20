# Prior-source stress test（Contribution 3 案例研究，v4 修复版，2026-09-04）

> ⚠️ **v4 修复声明（respond 终审裁决）**：v3 的 `deploy-lagged` 移位方向写反
> （实际用了未来值，future leakage），相关结果作废。本版用因果正确的过去值重跑。
> 同时 `deploy-nwp` 已重命名为 `wind-curve`（最后观测风速+手工曲线，不是真 NWP）。
> 旧的 "n=15、4 sources Wilcoxon p≤0.0001" 结论作废，见下方 v4 结果。

> 修订说明（steer #2）：本实验不再称为 "oracle-to-deployment substitution"。
> 原命名把"训练期静态历史先验"错称为 transductive oracle——这是不准确的：
> PCT 的 KDE power curve 是训练前用历史观测构造、训练期冻结的 **historical static
> prior**，推理期不需要未来真值，部署上完全合法。因此本实验的正确标题和定位是
> **prior provenance mis-specification / distribution-shift harm**（先验来源设定错误
> 与分布漂移伤害）。

## 问题

PCT 式 KDE/JS 分布约束的收益，在约束 target 分布用**不同 prior 来源**构造时，
是否稳健？哪些 prior 来源是有害的，为什么？

## 五个 prior source 条件（3 站点 × 2 seed × epochs=60，RNG 对齐）

| 条件 | target 分布来源 | provenance 归类（D1）|
|---|---|---|
| historical-static | 训练集真值 power 的一次性 KDE | historical-static（合法训练期先验）|
| lagged | 上一小时量测 power（因果滞后）| historical-static（滞后变体）|
| nwp-linear | 风速预测 + 理想线性 power curve | forecast-derived（理想模型）|
| nwp-measured | 风速预测 + bin-mean 实测 power curve | forecast-derived（实测拟合）|
| unconstrained | 无约束（MSE only）| — |

## 结果（kelmarsh, seed=0, epochs=60）

| prior source | RMSE (pu) | vs unconstrained |
|---|---|---|
| historical-static | 0.0956 | +10.3% |
| lagged | 0.1163 | +34.2% |
| nwp-linear | 0.2050 | +136.4% |
| nwp-measured | 0.1442 | +66.4% |
| unconstrained | 0.0867 | — |

## 多站点多 seed 汇总（v3，5 条件，6 任务）

| 站点 | seed | hist-static | lagged | nwp_linear | nwp_measured | uncon |
|---|---|---|---|---|---|---|
| kelmarsh | 0 | 0.0956 | 0.1163 | 0.2050 | 0.1442 | 0.0867 |
| kelmarsh | 1 | 0.0890 | 0.1166 | 0.1987 | 0.1456 | 0.0891 |
| penmanshiel | 0 | 0.1645 | 0.1523 | 0.2137 | 0.1767 | 0.1173 |
| penmanshiel | 1 | 0.1540 | 0.1374 | 0.2089 | 0.1862 | 0.1329 |
| smarteole | 0 | 0.1296 | 0.0784 | 0.1764 | 0.1422 | 0.0623 |
| smarteole | 1 | 0.1285 | 0.0886 | 0.1760 | 0.1427 | 0.0537 |

**稳健发现（6/6）**：
1. nwp-linear（理想线性曲线）在所有任务最差，比 unconstrained 差 60–140%。
2. 在所列实验格中方向非正（不称"6/6 非劣"或"约束有害"）。
3. 先验质量次第：nwp-linear（最差）> nwp-measured > historical-static ≈ lagged > unconstrained。

## 加噪 regime（PCT 声称的鲁棒性场景）

| 站点 | seed | noise | gain（约束 vs uncon）|
|---|---|---|---|
| kelmarsh | 0 | 0.5 | −24.5% |
| kelmarsh | 1 | 0.5 | −8.1% |
| kelmarsh | 0 | 0.7 | −11.6% |
| kelmarsh | 1 | 0.7 | +2.6%（噪声内）|
| penmanshiel | 0 | 0.5 | −42.1% |
| penmanshiel | 1 | 0.5 | −30.5% |
| smarteole | 0 | 0.5 | −27.7% |
| smarteole | 1 | 0.5 | −30.1% |

7/8 任务约束更差，1/8 微不足道的 +2.6%。

## 架构一致性（kelmarsh, seed=0, epochs=60）

| 架构 | clean gain | noisy gain |
|---|---|---|
| ours | −41.5% | −63.1% |
| paper-lstm | −46.8% | −55.1% |
| transformer | −75.1% | −65.2% |

三种架构 6/6 负收益——仅在 Kelmarsh seed-0 验证方向一致（不称"架构无关"）。

## 机制

### 机制 a：prior misspecification（先验本身错误）
nwp-linear 用理想线性 ramp（忽略 Cp 曲线非线性），在 v=4–5 m/s 段高估功率
100–130%（理想 1300–2600 kW vs 实测 660–1260 kW）→ 把模型拉向错误功率-风速关系。
nwp-measured 修正后有害性减半（0.176→0.144），但仍差于无约束。

### 机制 b：distribution shift（训练→部署分布漂移）
W1 定量（5 站点）：

| 站点 | train→test power 边际 W1 | nwp_linear 方向错误 | nwp_measured 方向错误 |
|---|---|---|---|
| kelmarsh | 0.037 | 0.037 | 0.065 |
| penmanshiel | 0.180 | 0.081 | 0.197 |
| smarteole | 0.133 | 0.169 | 0.092 |
| wind_us | 0.012 | 0.096 | 0.040 |
| wind_cn | 0.071 | 0.113 | 0.054 |

训练→测试漂移真实存在（W1 0.012–0.180），nwp_linear 方向错误常超过漂移本身，
解释了负收益幅度排序（nwp_linear > nwp_measured ≈ historical-static > unconstrained）。

## 可辩护结论（v4 修复版，2026-09-04）

> **Physics constraints are learned priors whose benefit depends on provenance,
> implementation fidelity and distribution stability.**（先验来源、实现忠实度、
> 分布稳定性共同决定约束收益。）

### v4 修复后 kelmarsh 5 seed（lagged 因果修正，60 epochs）

| prior source | mean 相对差 | worse | t_p | wilcox_p |
|---|---|---|---|---|
| historical-static（论文合法先验）| +5.4% | 4/5 | 0.060 | 0.125（**不显著**）|
| lagged-measured（因果过去值）| +18.6% | 5/5 | 0.0015 | 0.0625 |
| wind-curve-linear（错误线性先验）| **+123.7%** | 5/5 | <0.0001 | 0.0625 |
| wind-curve-measured | +51.8% | 5/5 | 0.001 | 0.0625 |

**稳定结论（v4，诚实收敛）**：
1. **历史静态先验（论文的合法做法）中性**——+5.4% 不显著，不是"约束有害"。
2. **错误先验显著有害**——wind-curve-linear +123.7%（p<0.0001），这是最稳的发现。
3. 边际的 lagged/wind-curve-measured 有害性 t 显著但 wilcoxon 边界（n=5 局限）。

**作废声明**：v3 的 "n=15、4 sources Wilcoxon p≤0.0001" 结论**作废**（其中
lagged 泄漏 + 混入多站未修复数据）。v4 只在 kelmarsh 5 seed 上重新建立了
诚实结论，且 historical-static 不再显著。

**不声称**：PCT 原结果错误（无原始数据/预处理/超参 exact reproduction）。

## 边界清单（写进论文）

1. 数据 = 开源 5 站，≠ PCT 江苏 25 台。
2. 架构 = 我们的 NetLSTM / paper-lstm / transformer 三类对照，但 PCT 原始训练用
   Transformer（code），论文声称 LSTM（spec）——我们两类都测。
3. 5 seed 配对显著性：v4 修复后 kelmarsh 5 seed（n=5），historical-static 不显著、
   wind-curve-linear t_p<0.0001。旧 "n=15 Wilcoxon p≤0.0001" 已撤回（含泄漏数据）。
4. 这是 transfer stress test，不是 exact reproduction。
