# 05. 全文数值冲突与数据真实性审计报告 (Numeric Conflict Report)

本报告详细记录当前稿件中所有数值冲突、历史笔误、列错位及其根因追踪，并给出基于原始仿真结果账本的重新计算与归正记录。

---

## 冲突 1: Table 4 (原Table 5) Penmanshiel Zero-Bid 决策成本严重失真与列错位

### 1. 发现问题
- **原稿数值**：Table 4（原Table 5）中 Penmanshiel 的 Zero-Bid 申报策略对应之“Decision Cost”列填报为 **`0.20584`**。
- **疑点分析**：
  在 Table 2 中，Penmanshiel 的 Vanilla Unconstrained DLinear 模型之测试集 **RMSE 恰好为 `0.2058`**（多种子均值 `0.205846`，Seed 42 值为 `0.2058426`）。
  这一极其反常的高度重合，强烈提示存在表格排版时的复制粘贴错误或列错位。

### 2. 代码回溯与机理推导
回溯 `05_实验代码与脚本/constraint_round2/run_cluster_ablation.py` 第 214--230 行：
- Zero-Bid 的定义为预测值恒等于零：$\hat{y}_t \equiv 0$。
- 在风电功率标幺值 $y_t \in [0, 1]$ 下，实际功率必有 $y_t \ge 0$。
- 非对称调度损失计算公式为：
  $$\text{Cost}_{\text{static}} = \frac{1}{N} \sum_{t} \left[ r \cdot \max(\hat{y}_t - y_t, 0) + \max(y_t - \hat{y}_t, 0) \right]$$
  当 $\hat{y}_t = 0$ 时，$\max(\hat{y}_t - y_t, 0) = 0$（欠发 Shortfall 严格为 0）。
  而 $\max(y_t - \hat{y}_t, 0) = y_t$（少报电量即为全部发电量）。
  因此：
  $$\text{Cost}_{\text{static}}(\text{Zero-Bid}) \equiv \frac{1}{N} \sum_{t} y_t = \bar{y} \quad (\text{测试集平均标幺出力})$$
  对应 $q = \frac{1}{r+1} = \frac{1}{3}$ 下的 Pinball Loss 为：
  $$\text{PinballLoss}(q=1/3) = \frac{1}{3} \bar{y}$$

### 3. 原始 CSV 重新计算结果 (Source of Truth)
查验 `06_实验结果与日志/ablation_3d_results/ablation_results_penmanshiel_seed42.csv` 第 135 行：
```csv
penmanshiel,42,zero_bid,baseline,0.0,0,uncalibrated,2,0.3333333333333333,0.33689355850219727,0.11229784786701202,0.46125394105911255,0.3367663323879242,-0.3367663323879242,120373.29548605366,0.0
```
- **真实 Decision Cost ($\text{Cost}_{\text{static}}$)**: **`0.33689`**（即 Penmanshiel 测试期平均出力 $\bar{y} = 0.3369$）！
- **真实 Pinball Loss**: **`0.11230`**！
- **真实 RMSE**: **`0.46125`**！
- **真实 Shortfall**: **`£0.00`**！
- **原稿中填入的 `0.20584` 确系将 Vanilla Unconstrained 的 RMSE 错贴进了 Decision Cost 列！**

### 4. 连带检查 Kelmarsh 与 Hill of Towie Zero-Bid
连带审查发现，原稿中 Kelmarsh 与 Hill of Towie 的 Zero-Bid 决策损失亦存在同样复制粘帖列错位：
- **Kelmarsh 原填 `0.17181`** $\rightarrow$ 实为 Vanilla Unconstrained 的 RMSE（`0.17181`）！  
  **归正后真实值**：$\text{Cost}_{\text{static}} = \mathbf{0.30428}$, $\text{Pinball} = \mathbf{0.10143}$, $\text{RMSE} = \mathbf{0.41131}$, $\text{Shortfall} = \mathbf{\pounds 0.00}$。
- **Hill of Towie 原填 `0.17354`** $\rightarrow$ 实为 Vanilla Unconstrained 的 RMSE（`0.1740` 单种子值 `0.17354`）！  
  **归正后真实值**：$\text{Cost}_{\text{static}} = \mathbf{0.30473}$, $\text{Pinball} = \mathbf{0.10158}$, $\text{RMSE} = \mathbf{0.42383}$, $\text{Shortfall} = \mathbf{\pounds 0.00}$。

---

## 冲突 2: $\lambda$ 正则化权重跨章节与跨站点口径混淆

### 1. 发现问题
正文各节、图表注、Table 2 中，在讨论“物理曲线与伪先验对照”时，频繁穿插 $\lambda=0.01$、$\lambda=0.1$ 与 $\lambda=1.0$：
- Kelmarsh Genuine Curve 列为 $\lambda=0.1$，而同表同栏的 Pseudo-Scrambled 却列为 $\lambda=0.01$；
- 读者误以为是用 $\lambda=0.1$ 的真实先验去和 $\lambda=0.01$ 的打乱先验进行对比，从而质疑比较的不对等性。

### 2. 原始账本核查与归正方案
查验 `ablation_3d_key_comparison_table.csv`：
- **在 Kelmarsh 上**：
  - $\lambda=0.01$ 下：`genuine_curve_l0.01` 损失为 `0.08810`，`pseudo_scrambled_curve_l0.01` 损失为 `0.08801`，无约束基准为 `0.08812`。两者差异仅为 0.00009，统计上完全无法区分。这是证明“**低 $\lambda$ 数值正则化拟态（Regularization Mimicry）**”的严格对称对照！
  - $\lambda=0.1$ 下：`genuine_curve_l0.1` 达到局部最优 `0.08791`（$-0.24\%$）。
- **在 Penmanshiel 上**：
  - $\lambda=1.0$ 下：`genuine_curve_l1.0` 损失为 `0.09760`（$-1.22\%$），而 `pseudo_scrambled_curve_l1.0` 损失恶化至 `0.16611`（$+68.1\%$）。这是证明“**强约束下物理先验特异性（Prior Specificity）**”的严格同权重对照！
- **归正原则**：
  Table 2 及正文必须显式标注每个对照所处的严格 $\lambda$ 实验条件。严禁将不同 $\lambda$ 的结果并置而忽略权重差异。

---

## 冲突 3: 随机种子与方差报告口径统一

### 1. 发现问题
原稿方法节（Section 4.4）统一声称“所有模型均基于 5 个随机初始化种子进行评估”。

### 2. 原始实验日志核实
- 查验 `ablation_3d_results/`：
  - Penmanshiel 实际完成种子：`42, 43, 44`（共 3 个种子）；
  - Hill of Towie 实际完成种子：`42, 43, 44`（共 3 个种子）；
  - Kelmarsh 实际完成种子：`42, 43, 44, 45`（共 4 个种子）。
- 查验 `learning_curve/`（学习曲线）与 `patchtst_settlement_table.csv`：
  - 学习曲线与 PatchTST 敏感性实验确实运行了完整的 5 个种子（`42, 43, 44, 45, 46`）。
- **归正方案**：
  实事求是。在 Table 2 的表注和方法节中精确披露：主 3D 消融基准在 Kelmarsh 上采用 4 个种子，在 Hill of Towie 和 Penmanshiel 上采用 3 个种子；学习曲线与 PatchTST 探针采用 5 个种子。严禁虚夸为“全量统一 5 种子”。

---

## 冲突 4: 货币与单位符号乱码

### 1. 发现问题
在部分旧 LaTeX 输出和表格源文件中，英镑符号 `£` 存在编译为 `\pounds` 缺失、显示为乱码或被吞掉的情况；部分列头单位错位。

### 2. 归正方案
- LaTeX 导言区统一引入 `\usepackage{textcomp}`，正文与表格中货币符号统一采用 `\pounds` 宏命令或 UTF-8 标准英镑符号；
- 统一量纲写法：
  - 发电功率标幺值：$\text{pu}$
  - 归一化静态调度损失：$\text{Cost}_{\text{static}} \quad (r=2)$（无量纲或 pu）
  - 单边欠发惩罚现金流：$\pounds/(\text{turb}\cdot\text{yr})$
  - 净不平衡结算现金流分量：$\pounds/(\text{turb}\cdot\text{yr})$
  - 现货与平衡清算电价：$\pounds/\text{MWh}$。

---

## 冲突 5: 英国电力市场结算符号与 FPN 概念混淆归正

### 1. 发现问题
1. 在正文 Section 8 旧版中残留有集中日前市场符号 $P_{\text{DA}}$；
2. 在原稿与初步审计中，曾将结算量 $q$ 误写为“申报物理通知（FPN）”。

### 2. 制度审计与归正
根据英国 BSC（Balancing and Settlement Code post-P305）规则：
- Energy Imbalance 比较的是账户层面的合同量（通过 ECVN 通知）与实际计量物理发电量（Metered Volume）；
- FPN（Final Physical Notification）是机组向系统调度机构（National Grid ESO）提交的物理运行出力曲线，用于系统实时平衡操作，并非直接用于结算双边财务头寸；
- 因此将结算公式 $R_{\text{total}} = P_{\text{F}} \cdot q + P_{\text{imb}} \cdot (y - q)$ 中的 $q$ 统一定义为“forward-contracted / nominated energy position used in the stylized accounting model”，彻底杜绝 $q = \text{FPN}$ 的制度性概念瑕疵；
- 将“computed strictly under Elexon BSC rules”降格定性为“a stylized BSC-price-based imbalance cashflow diagnostic”，承认当前建模未涉及账户级双边合同通知与 BM 接受量细则。

---

## 冲突 6: 正文内部修改痕迹与元注释清理

### 1. 发现问题
在 Section 5.5 及 Table 4 旧版表注中，残留有内部协作修改痕迹（如“earlier drafts reported Penmanshiel Zero-Bid Decision Cost as 0.20584...”），带有工作草稿性质。

### 2. 归正方案
将内部审校性陈述完全剥离至 `revision_outputs/` 归档报告中，正文彻底重写为成熟学术论文语言：“Under asymmetric operational loss ($r=2, q=1/3$), the Zero-Bid strategy yields a Decision Cost of 0.33689 on Penmanshiel (Table 4), correctly penalizing the policy for severe under-commitment ($\text{Cost}_{\text{static}} = \bar{y} \approx 0.337$, matching the site's mean capacity factor)...”，杜绝任何元注释。

---

## 冲突 7: 数据稀缺与先验信息预算（Information Budget）混淆

### 1. 发现问题
原学习曲线在 $N_{\text{train}}=500$（约 21 天数据）时，真实先验是利用前序完整 7–10 个月历史数据拟合的。审稿人质疑：此时先验模型的优势可能来源于额外喂入的数月历史季节性信息，而非物理归纳偏置本身。原稿声称“we causally identify that physical constraints deliver measurable inductive protection primarily under restricted sample volumes”存在因果过推。

### 2. 实证隔离与归正
- 编写并执行 `run_information_budget_experiment.py`，在 $N_{\text{train}}=500$ 下补全严格信息预算对齐的 5 臂消融：
  - Arm A: 无先验 (500 样本)
  - Arm B: 严格预算对齐真实先验（先验仅由相同的 500 样本拟合）
  - Arm C: 跨期大先验（先验由前序 7–10 个月拟合）
  - Arm D: 严格预算对齐乱序对照（先验仅由相同的 500 样本拟合并置换）
  - Arm E: 跨期大乱序对照
- 实证结果证明：在完全禁止访问外部历史数据的严格预算下（Arm B vs Arm A），真实物理先验仍带来 $-1.54\%$（Kelmarsh）与 $-3.34\%$（Hill of Towie）的风险降幅，解释了总保护效应的 $77\%$ 至 $92\%$；扩充至数月历史数据（Arm C）仅带来额外 $-0.46\%$ 与 $-0.28\%$ 的微幅边际增益；
- 论文正文将所有“causally identify”彻底降调为“an advantage observed under restricted model-training windows when the prior retains a larger historical information budget”，实现科学论断的完全诚实与严谨闭环。

---

## 冲突 8: 负对照“梯度尺度匹配（Gradient-Scale Matched）”表述过强

### 1. 发现问题
原稿称伪乱序先验对照（Pseudo-scrambled controls）“preserve identical loss formulas, parameter counts, and gradient scales”。然而，仅保持同一公式和置换映射，在数学上并不保证反向传播的梯度范数完全相同。

### 2. 归正方案
全文统一删除“gradient-scale matched”，精准修改为：
“form- and weight-matched pseudo-scrambled controls preserving identical loss formulations, network architectures, and regularization weights $\lambda$”。
杜绝无显式梯度归一化算法下的过强声称。

