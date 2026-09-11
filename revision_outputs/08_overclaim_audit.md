# 08. 全文过度主张与越界词汇审计报告 (Overclaim Audit)

本报告对当前稿件中所有高风险动词、绝对化副词、因果断言及外推范围进行拉网式排查与修剪，确保论文所有论述严格受制于实际经验证据。

---

## 一、 四大核心归纳外推越界（Generalization Overclaims）拦截

| 序号 | 原文高风险越界主张 | 事实边界与反例 | 审定修订方案 |
|:---:|---|---|---|
| **1** | **“3 座风场 $\rightarrow$ 全球风电工业界”**<br>*“Our results demonstrate that physics constraints universally fail across the wind industry.”* | 本文仅测试了英国 3 座风电场（Kelmarsh、Hill of Towie、Penmanshiel），其机组为 Senvion MM82/MM92 及 Siemens 2.3MW，地形涵盖苏格兰高地与低地平原。不能代表全球复杂山地、海上超大型机组或干旱戈壁风电场。 | **修改为**：*“Across the three commercial UK wind farms tested under our controlled evaluation framework...”* 严禁推及全行业。 |
| **2** | **“46 篇范围审查语料 $\rightarrow$ 全体科学文献”**<br>*“The forecasting literature contains zero documented negative trade-offs.”* | 本文仅对预先指定的 PRISMA-ScR 检索字符串下纳入的 46 篇 2023--2026 年风电功率显式约束研究进行了编码；即使加上 14 篇敏感性检索，亦不代表已穷尽所有未纳入数据库的灰色文献或历史经典。 | **修改为**：*“Within the 46 scoping studies audited under our prespecified protocol, none reported operational failure regimes or negative trade-offs (0/46, 0.0%).”* |
| **3** | **“PCT 单一代码缺陷 $\rightarrow$ 科学软件普遍不可信”**<br>*“Software inspection proves that deep physics-informed networks suffer from catastrophic gradient explosions.”* | 本文仅对 S02 PCT 官方 PyTorch 仓库的 `loss.py` 进行了参数方向核查，发现其反向调用了 `F.kl_div`。这属于单一代码库的实现偏差，且网络反向传播受梯度裁剪（gradient clipping）保护。 | **修改为**：*“Software code inspection of one public repository (PCT, S02) reveals an inverted parameter argument altering backpropagated gradient scales...”* 明确区分局部算子缺陷与端到端优化表现。 |
| **4** | **“2 个反例构造 $\rightarrow$ 三风场的经验因果证明”**<br>*“Counterexamples 1 and 2 prove that Term III caused the metric inversion observed on Hill of Towie.”* | 构造的反例属于“存在性证明（Possibility / Existence Results）”，在逻辑上证明了“点误差下降但决策损失上升”这一现象在数学上可以发生。它绝不构成“三座风场中的具体现象一定由同一协方差机制所主导”的经验因果判定。 | **修改为**：*“Counterexamples 1 and 2 illustrate the analytical possibility of prediction-decision inversions, providing an explanatory diagnostic rather than an isolated empirical causal proof.”* |

---

## 二、 典型因果动词与绝对化副词拉网式清剿

### 1. 动词级替换
- `proves` $\rightarrow$ *shows* / *is consistent with* / *indicates under our test conditions*
- `confirms` $\rightarrow$ *supports* / *provides empirical evidence for*
- `causes` $\rightarrow$ *is associated with* / *leads to (only under matched backbone ablation)*
- `validates` $\rightarrow$ *evaluates* / *tests*
- `guarantees` $\rightarrow$ *provides inductive protection under*
- `uncovers the true mechanism` $\rightarrow$ *provides a diagnostic framework consistent with*

### 2. 副词与形容词修剪
- `fundamentally` $\rightarrow$ 删除或替换为 *primarily*, *systematically*
- `unambiguously` / `definitively` $\rightarrow$ 删除或替换为 *consistently*
- `catastrophic` $\rightarrow$ 替换为具体的量化数字 *inflates decision loss by +68.1%*
- `severe` $\rightarrow$ 替换为客观的实测幅度 *a +44.2% increase in shortfall penalty*
- `pathological` $\rightarrow$ 替换为 *degenerate* / *diagnostic*

---

## 三、 理论节与机制分解定性降调 (Section 7)

1. **Proposition 1 降调为 Lemma 1**：
   - 彻底删除“发明全新决策理论”的任何自我暗示；
   - 明确标注其为 Koenker (2001) 与 Pinson (2007) 经典分位数最优性事实在本文受控审计基准中的**操作化引理（Decision-Theoretic Operational Baseline）**。

2. **三项风险分解（Three-term Decomposition）定性**：
   - 将 Term I（Point MSE）、Term II（Quantile Mismatch）、Term III（Interaction Covariance）的阐述定位为**“用于定性诊断误差与决策风险解耦的一致性解释框架（Consistent Explanatory Framework）”**；
   - 明确承认：在当前的经验测试中，未对条件协方差项进行独立的因果隔离识别（Term III is analytically possible but not separately identified as an independent empirical mechanism in the current test bench）。
