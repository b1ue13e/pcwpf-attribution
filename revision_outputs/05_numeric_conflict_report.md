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

## 冲突 5: Section 8 英国电力市场远期结算价格符号不一致

### 1. 发现问题
在正文 Section 8（Discussion）第 17 行及第 72 行（Gate M4）中，两部制市场结算公式残留旧版集中日前市场符号：$R_{\text{total}} = P_{\text{DA}} \cdot q + P_{\text{imb}} \cdot (y - q)$。而在 Section 4（Protocol）第 40 行与 Section 5（Results）第 92 行中，统一定义为 $R_{\text{total}} = P_{\text{F}} \cdot q + P_{\text{imb}} \cdot (y - q)$。

### 2. 制度审计与归正
在英国（GB BETTA / BSC）电力市场制度下，发电商主要通过场外双边合同（OTC Forward）与电力交易所自主申报物理通知（FPN），而非强制性日前集中调度池（Mandatory Day-Ahead Pool）。因此：
- 采用通用远期合同电价符号 $P_{\text{F}}$（Forward Price），准确覆盖双边远期协议与日前交易；
- 已将 Section 8 中所有残留的 $P_{\text{DA}}$ 统一修正为 $P_{\text{F}}$，保持全篇制度与数学表达 100% 严谨闭环。

---

## 冲突 6: 正文内部修改痕迹与元注释清理

### 1. 发现问题
在 Section 5.5 第 97 行及 Table 4 表注中，残留有内部协作修改痕迹（如“earlier drafts reported Penmanshiel Zero-Bid Decision Cost as 0.20584, which was an erroneous transcription that copy-pasted...”），带有工作草稿与评审注记性质。

### 2. 归正方案
将内部审校性陈述完全剥离至 `revision_outputs/` 归档报告中，正文彻底重写为成熟学术论文语言：“Under asymmetric operational loss ($r=2, q=1/3$), the Zero-Bid strategy yields a Decision Cost of 0.33689 on Penmanshiel (Table 4), correctly penalizing the policy for severe under-commitment ($\text{Cost}_{\text{static}} = \bar{y} \approx 0.337$, matching the site's mean capacity factor)...”，杜绝任何元注释。

