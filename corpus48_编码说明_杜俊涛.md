# 48 项扩展语料独立编码说明与技术审计报告 (第一批交付)

**复核人**：杜俊涛  
**完成时间**：2026-09-04  
**交付文件**：
- 编码总表：`e:\djt\corpus48_编码_杜俊涛.csv`
- 详细说明文档：`e:\djt\corpus48_编码说明_杜俊涛.md`（本文档）

---

### 一、 执行概况与数据看板

本轮复核将系统审计语料从初始的 21 篇扩展至 Task D 宽筛选去重后的 **48 项独立研究（S01 至 S48）**。依照《48 项语料扩展编码任务书》及《Constraint Lifecycle Taxonomy (schema v4)》，本人完成了第一批核心指标（`inclusion_check`、`D4_fidelity` 以及 **D5 拆解的四个二元/多分类判定字段**）的全量编码，并结合前期全文调研对 `D1/D2/D3` 进行了前置校准与预编。

#### 1. 核心维度统计看板 (N = 48)

| 审计维度 | 字段名称 | 统计分布 | 核心发现与学术定性 |
|---|---|---|---|
| **0. 纳入复核** | `inclusion_check` | `include`: 46 项 (95.8%)<br>`exclude`: 2 项 (4.2%) | 46 项具备明确的解析物理方程、功率曲线、尾流动力学或分布散度约束；2 项（S26, S42）判定为排除。 |
| **1. 代码忠实度** | `D4_fidelity` | `not-found`: 47 项 (97.9%)<br>`code-mismatch`: 1 项 (2.1%) | 全库仅 S02 (PCT) 开源了官方代码（但存在 Reverse-JS 实现偏差）；其余 47 篇经检索均未提供公开可执行代码仓库。 |
| **2. 消融完备性** | `d5_ablation_present` | `yes`: 46 项 (95.8%)<br>`unknown`: 2 项 (4.2%) | 46 篇纳入文献均包含物理模块“开 vs 关”或与无物理基准的消融对比；2 篇排除文献标记为 unknown。 |
| **3. 负面效应报告** | `d5_negative_effect_reported` | `no`: 48 项 (**100.0%**) | **全库 0 篇报告物理约束的负面效应**。所有文献均为典型的方法增益倡导型论文，清一色声称物理约束提升了预测表现。 |
| **4. 伤害指标量化** | `d5_harm_metric_reported` | `no`: 48 项 (**100.0%**) | 全库文献均报告常规统计误差（RMSE, MAE, MAPE, $R^2$），无一量化报告电网调度损失、弃风经济惩罚等真实系统级伤害指标。 |
| **5. 先验替换测试** | `d5_prior_substitution_tested` | `no`: 45 项 (93.8%)<br>`yes`: 3 项 (6.2%) | 仅 3 篇（S27 UniWind, S34 PRX Energy, S44 Bayesian MAP）测试了先验模型替换或超参数敏感性；其余均采用单一硬编码先验。 |

---

### 二、 排除文献（exclude）判定专项说明

在 48 项候选文献中，本人严格执行任务书防伪红线（*“标题带 physics 但实则纯数据驱动 / 仅物理启发设计（无任何显式约束）”*），甄别并排除了以下 **2 篇文献**：

#### 1. S42 (`10.1016/j.epsr.2026.112925`)
- **论文题目**：*A Physics-Guided Feature-Subspace ensemble framework for forecasting photovoltaic and wind power generation* (*Electric Power Systems Research* 2026)
- **判定结果**：`exclude`
- **排除原因**：`no-explicit-physics-constraint`
- **判定深度剖析**：
  该文在标题中冠以“Physics-Guided”，但通读其方法设计，作者仅仅是将输入的气象特征（温度、湿度、辐照度、风速）依据物理常识人为切分成了若干个子空间（Feature Subspaces），然后分别训练集成学习回归器。  
  **病理诊断**：文中**没有任何显式数学物理方程**（无气动方程、无 Betz 极限、无尾流衰减公式、无 Navier-Stokes PDE）、**无功率特性曲线解析模型**、亦**无物理惩罚项或分布损失约束**。这属于典型的“物理启发式特征分组工程（Heuristic Feature Grouping）”，在数学与算法本质上是 100% 的纯数据驱动集成学习。严格符合排除标准。

#### 2. S26 (`10.1016/j.energy.2026.142114`)
- **论文题目**：*Cross-scale regime-aware physics-informed forecasting for reliable wind-speed prediction in risk-aware wind power operation* (*Energy* 2026)
- **判定结果**：`exclude`
- **排除原因**：`wind-speed-only / no-explicit-power-constraint`
- **判定深度剖析**：
  该文研究目标为风电场入流风速预测（Wind Speed Forecasting），而非风电功率预测；文中所谓“物理”是指利用大气边界层相似理论与中微尺度天气模态对风速时序进行分区段聚类（Regime-aware）。虽然其对风电场运行有风险参考价值，但其模型既不预测功率输出，也未对发电机能量转换机理施加显式物理约束。为了维护“风电功率预测（Wind Power Forecasting）物理约束”语料的严格纯洁性，予以排除。

---

### 三、 D5 拆分为四个二元字段的统计学价值分析

上几轮评审中出现的所谓“18/19/20 口径漂移”，其根源在于旧版单一 `D5_harm` 标签将两个完全不同的学术行为混淆在一个字段中：
1. “作者有没有做消融实验证明约束有效”；
2. “约束有没有在某些工况下导致预测劣化，作者是否如实报告了伤害”。

本次任务书将 D5 彻底拆解为 **4 个布尔/分类字段**，彻底解决了这一学术口径模糊：

1. **`d5_ablation_present = yes` (46/48) 与 `d5_negative_effect_reported = no` (48/48) 的鲜明反差**：
   - 绝大多数作者（95.8%）都在论文中精心设计了“带物理 vs 不带物理”的对比实验，但其目的纯粹是为了论证自身算法的优越性；
   - 所有的消融结果无一例外都是“加上物理约束后误差降低、去掉物理后误差上升”；
   - 没有任何一篇公开发表的正刊论文承认或报告了物理约束在噪声数据下可能引发的刚性拟合误差（例如我们在 Task E 真实 GPU 重跑中所观测到的 `constrained` RMSE 显著高于 `unconstrained` +20%~+33% 的客观物理代偿现象）。
2. **`d5_harm_metric_reported = no` (48/48)**：
   - 证明当前风电预测领域的学术论文普遍缺乏电力系统实际运行视角的“伤害审计意识”；
   - 论文仅关注标准学术指标（RMSE/MAE/MAPE），完全没有评估错误物理约束导致调度日前弃风、过充或电网备用考核罚款等系统级量化惩罚。
3. **`d5_prior_substitution_tested` 的高度稀缺性 (仅 3/48 篇，6.2%)**：
   - 仅有 S27 (UniWind 针对不同场群测试先验扭曲敏感性)、S34 (PRX Energy 探索了高斯解析尾流模型的参数鲁棒性) 和 S44 (Bayesian MAP 测试了 CFD 空间先验方差) 进行了先验替换与敏感性测试；
   - 其余 93.8% 的论文均将物理先验视为不可更改的绝对真理（Ground Truth），缺乏对先验偏差（Prior Misspecification）的模型鲁棒性压力测试。

---

### 四、 把握度说明（Confidence Notes）

1. **关于 S30 (DSTC-Net) 与 S46 (Federated Multi-Farm)**：
   - 此两篇工作利用了空间机组坐标、距离与风向构建动态图邻接矩阵或联邦聚合加权矩阵。虽然其物理先验以拓扑图权重形式体现（相比直接在 loss 里加方程稍显温和），但其图权重由解析流体力学距离与尾流几何公式计算得出，而非自由参数自注意力，故认定其具备显式物理先验，判定为 `include`。
2. **关于第二批 D1/D2/D3 全文级编码**：
   - 本表中已根据摘要与公开方法预先填报了 D1/D2/D3 的初步基准（`intrinsic`: 40, `historical-static`: 5, `learned-online`: 1; `inference-architecture`: 22, `training-loss`: 22, `data-generation`: 1, `post-hoc-correction`: 1）；
   - 待第一批纳入语料范围经您与团队最终锁定后，本人可无缝启动第二批全文级深读核查，确保每个字段的证据链达到顶会/期刊发表级的抗辩严密性。

数据表 [corpus48_编码_杜俊涛.csv](file:///e:/djt/corpus48_%E7%BC%96%E7%A0%81_%E6%9D%9C%E4%BF%8A%E6%B6%9B.csv) 已在本地保存完毕，请协调人老师审阅指导！
