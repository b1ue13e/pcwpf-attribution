# 全新论文科学叙事架构蓝图
# (New Manuscript Master Narrative Architecture Blueprint)

> **版本**：v4.0 证据校准与严谨实证叙事重构最终蓝图 (Phase 4 Calibrated Master Architecture Blueprint)  
> **基准定位**：*Conditional Inductive Guardrails: Identifying When Physical Priors Enhance, Mimic, or Impair Decisions in Wind Power Forecasting*  
> **状态**：已执行 13 项强制性证据校准更正（Evidence-Calibration Corrections），完全受控于现有真实实验数据（Zero Data Fabrication），严禁过度因果推断与机制断言。

---

## 一、论文唯一主问题与中心科学命题 (The Master Scientific Anchor)

### 1.1 唯一科学主问题 (The Single Research Question)
> **When a physical prior is introduced into wind power forecasting, does it provide genuine decision value because it encodes valid domain knowledge, and under what conditions does that value persist, disappear, or reverse?**  
> （核心科学问题在于识别：物理先验何时提供真实的决策价值，其表观收益何时是非特异性的，以及何时会反过来损害运行决策？）

### 1.2 冻结主张层级体系 (Frozen Claim Hierarchy)
1. **中心命题 (Central Claim)**:  
   *In the evaluated wind-power forecasting settings, physical priors exhibit conditional rather than universal decision value.*（在所评估的风电预测设置中，物理先验表现为条件性归纳护栏，而非普遍适用的性能增强器。）
2. **主要实证命题 (Primary Empirical Claim)**:  
   *Under sample restriction and strong regularization, the genuine empirical power-curve prior improves asymmetric decision risk while a form- and weight-matched scrambled prior strongly degrades it, providing evidence of prior-specific domain structure.*（在样本受限与强正则化设置下，真实经验功率曲线先验改善非对称调度风险，而形式与权重完全匹配的伪打乱先验导致严重恶化，为先验特异性领域结构提供了实证证据。）
3. **有效性边界命题 (Boundary Claim)**:  
   *This advantage attenuates with increasing training data and is vulnerable to prior corruption, engineering drift, excessive constraint strength, and temporal proxy misalignment.*（该先验优势随着训练样本扩充而衰减，并对先验失配、工程漂移、过度约束强度以及气象代理时间错位表现出脆弱性。）
4. **弱惩罚解释命题 (Weak-Penalty Claim)**:  
   *When genuine and scrambled priors yield closely matched performance, improvement cannot be specifically attributed to physical knowledge and remains consistent with non-specific numerical regularization.*（当真实先验与打乱伪先验表现紧密匹配、无明显实证分离时，所观测到的收益不能明确归因于物理知识，而与非特异性数值正则化保持一致。）
5. **决策对齐命题 (Decision-Alignment Claim)**:  
   *Symmetric prediction accuracy and standalone balancing cashflow are insufficient proxies for the asymmetric operational objective studied here.*（对称预测精度与单边平衡结算现金流均不足以代表本文所关注的非对称电网运行决策目标。）
6. **支撑性取证命题 (Supporting Forensic Claim)**:  
   *The literature and software audits explain why conventional positive comparisons frequently lack the controls necessary to establish these claims.*（文献与软件审计解释了为何常规的正面文献对比往往缺乏确立这些结论所必需的受控实验与负对照。）

### 1.3 概念性机制解释：方差–偏置概念模型 (The Conceptual Bias–Variance Interpretation)
*注：本篇论文推导了非对称调度风险的决策论二阶三项分解（表示误差、校准弥散、交叉耦合项），并未直接从经验数据中估计古典统计方差与代数偏置分量。因此，严禁将偏置–方差权衡表述为数学恒等式、定理或已证实的恒等式。*

- **概念性阐释 (Conceptual Interpretation)**：  
  观测到的学习曲线模式与以下机制假说保持一致（*is consistent with*）：在固定物理先验下，随着可用训练数据的增加，先验所提供的数值与归纳正则化价值逐步减退，而固定近似代数先验所固有的结构性逼近误差（Structural Approximation Error）对预测的影响则变得愈发显著。
- **在评估样本受限工况下 ($N_{\text{train}} \in \{500, 1{,}000\}$)**：纯数据模型的估计方差相对较高，形态合理的物理先验约束了假设搜索空间，提供了可观的下行决策风险保护；
- **在评估最大训练规模下 ($N_{\text{train}} = 5{,}000$)**：纯数据模型已有充分样本逼近真实非线性功率响应，先验优势衰减；在评估的强约束设定下，纯数据基准模型的表现略优于固定先验；
- **数据量与惩罚强度的联合作用**：学习曲线实验与独立的 $\lambda$ 敏感性实验共同表明，数据可用性与正则化惩罚强度共同决定了物理先验的有效运行区间（而非断言经过析因检验的全面 $N \times \lambda$ 交互效应）；
- **先验特异性与失配**：在强约束 ($\lambda=1.0$) 下，颠倒气动响应的伪打乱先验施加了错误的梯度引导，导致决策成本在受测工况下剧烈上升（$+68.1\%$）；
- **气象代理时间错位**：在 Hill of Towie 滞后扫描中，决策风险在滞后步数 $k \approx 12$ 附近穿透无约束基准，并在更长受试时滞下持续恶化。

---

## 二、核心元数据重构设计 (Metadata Master Specifications)

### 2.1 标题确定 (Title)
**Conditional Inductive Guardrails: Identifying When Physical Priors Enhance, Mimic, or Impair Decisions in Wind Power Forecasting**

### 2.2 Highlights 重构方案
```latex
\begin{highlights}
\item Physical priors exhibit conditional rather than universal decision value in wind power forecasting.
\item Under sample restriction, genuine empirical priors improve dispatch risk while scrambled controls inflate loss by 68.1\%.
\item As training data expands to $N=5{,}000$, unconstrained forecasters match fixed priors, consistent with bias--variance trade-offs.
\item Under weak regularization, genuine and scrambled priors yield closely matched outcomes, reflecting non-specific smoothing.
\item Info-symmetric and zero-bid evaluations show that point accuracy and standalone balancing cashflows decouple from dispatch risk.
\end{highlights}
```

### 2.3 Abstract 重构方案 (Problem -> Challenge -> Controlled Design -> Findings -> Decision Implication)
```latex
\begin{abstract}
Physical constraints are widely credited with improving wind power forecasting, but reported reductions in statistical error do not establish that empirical gains stem from valid domain knowledge or translate into operational decision value. The central scientific problem is to identify when a physical prior provides genuine decision value, when its apparent benefit is non-specific, and when it becomes harmful. Here, we present an attributional identification study across three commercial UK wind farms holding forecasting backbones, network capacity, data splits, and post-hoc quantile calibration strictly constant. By contrasting authentic empirical power curve priors against form- and weight-matched pseudo-scrambled negative controls across varied sample regimes, constraint weights, and proxy delays, we delineate the conditional operating envelope of physical priors. In the tested sample-restricted regimes ($N_{\text{train}}=1{,}583$, Penmanshiel), strong empirical curve constraints ($\lambda=1.0$) reduce normalized asymmetric dispatch loss by 1.22\%, whereas permuting the aerodynamic mapping triggers a 68.1\% surge in dispatch loss, demonstrating prior domain specificity under data scarcity. A strictly budget-matched information control ($N_{\text{prior}}=N_{\text{train}}=500$) confirms that domain-structured empirical prior effects retain 77\% to 92\% of the full-history benefit, demonstrating that the majority of the observed benefit cannot be explained by additional historical data volume alone. However, at the largest evaluated training budget ($N_{\text{train}}=5{,}000$), unconstrained data-driven models match or slightly outperform fixed priors, consistent with a transition in which the regularizing value of a fixed prior diminishes while structural approximation error becomes increasingly consequential. Under weak regularization ($\lambda=0.01$), authentic and scrambled priors yield closely matched outcomes ($0.08810$ vs.\ $0.08801$), indicating that the observed effect cannot be specifically attributed to physical mapping and remains consistent with non-specific numerical regularization. In continuous stress testing on Penmanshiel, the evaluated corruption trajectory crosses the unconstrained baseline around $\alpha \approx 0.07$, while in the Hill of Towie lag scan, decision risk crosses the unconstrained reference around $k \approx 12$ and continues to deteriorate at larger tested lags. Downstream, an info-symmetric evaluation reveals that adding wind information improves RMSE by 1.16\% yet worsens asymmetric decision cost by 1.11\%, and a degenerate zero-bid baseline mechanically generates six-figure apparent balancing cashflows without forecasting skill. Systematic literature and software audits explain why conventional evaluations rarely identify these boundaries. Physical priors should therefore be treated as conditional inductive guardrails requiring controlled attribution protocols, boundary stress tests, and decision-aligned evaluation.
\end{abstract}
```

---

## 三、Introduction 与 Contributions 叙事重构逻辑

### 3.1 六段式 Introduction 逻辑流
1. **第一段：风电预测的最终价值由非对称电网运行决策风险决定**  
   - 强调日前申报、备用容量预留与实时不平衡考核中的成本非对称性（$c_u \ne c_o$）；
   - 对称点误差（RMSE/MAE）无法充分表征极端短缺带来的非线性决策风险。
2. **第二段：物理约束预测的核心隐含假设与传统信念**  
   - 文献普遍将功率曲线、机理限值或分布先验嵌入神经网络，并将其误差下降直接归功于“模型吸收了真实物理领域知识”。
3. **第三段：归因挑战与潜在替代解释 (The Attribution Dilemma)**  
   - 仅观察到先验模型优于无约束基准，不足以断言物理有效性；
   - 梳理替代解释：网络容量变动、先验包含额外历史数据带来的信息泄露、单纯数值平滑/L2 正则化、代理变量滞后、评价指标背离等。
4. **第四段：提出本文核心科学问题与中心命题**  
   - 核心问题：物理先验何时提供真实决策价值，何时仅为非特异性拟态，何时演变为运行危害？
   - 核心命题：物理先验是具有明确运行边界的“条件性归纳护栏（Conditional Inductive Guardrails）”。
5. **第五段：受控归因识别实验体系 (Controlled Attribution Protocol)**  
   - 引入 Authentic vs. Scrambled vs. None 阴性对照设计；
   - 样本量梯队实验、500 样本信息预算对齐、代理滞后扫描、连续失配应力测试；
   - 冻结 Primary Endpoint 为非对称调度风险 $\mathcal{L}_{\text{disp}}$。
6. **第六段：文献与软件审计的支撑性定位**  
   - 将 46 篇 Scoping 综述作为“证据缺口审计（Evidence-Gap Audit）”，置于动机与相关工作部分，解释为何以往文献未能揭示失效边界；
   - 将 PCT 软件 AST 逆向与梯度分析作为支撑性计算图取证。

### 3.2 三大互锁贡献 (The Three Interlocked Contributions)
- **Contribution 1 — 受控归因协议 (Controlled Attribution Protocol for Physical Priors)**  
  建立严格固定 Backbone、网络容量、数据划分、随机种子与后验分位数校准的 Authentic vs. Scrambled vs. None 负对照实验协议，为甄别物理领域特异性指导与非特异性数值平滑提供实证判别基准。
- **Contribution 2 — 实证有效性边界测绘 (Empirical Operating Envelope Mapping)**  
  在受试风电场与实验设置下测绘物理先验的运行边界：揭示了样本受限工况到大样本工况下先验优势的减退；通过 500 样本预算对齐排除了主要收益纯粹源于历史数据泄露的假说；测绘了 Penmanshiel 上连续失配经验交叉区（$\alpha \approx 0.07$）及 Hill of Towie 气象代理时滞交叉区（$k \approx 12$）。
- **Contribution 3 — 决策对齐评价准则与治理框架 (Decision-Aligned Evaluation & Governance)**  
  通过决策论二阶展开与实证证据（Info-Symmetric 臂点误差改善但决策成本恶化、Zero-Bid 机械获利归谬），揭示对称点指标与单边现金流的评价局限；确立非对称调度风险为主终点，并输出包含 C1--C5 清单的方法学操作规范。

---

## 四、全论文 9 章节架构与内容重构安排

```
+------------------------------------------------------------------------+
| Section 1: Introduction                                                |
| 决策风险痛点 -> 物理先验隐含假设 -> 归因困境与替代机制                  |
| 核心科学问题 -> 条件性归纳护栏命题 -> 三大互锁贡献                      |
+-----------------------------------┬------------------------------------+
                                    |
+-----------------------------------v------------------------------------+
| Section 2: Related Work & Evidence-Gap Literature Audit                |
| 物理约束风电预测综述 -> 决策导向评估现状 -> 可复现性与审计规范          |
| 嵌入精炼版 46 篇 Scoping 证据缺口审计 (0/46 报告失败边界, 44/46 缺代码, |
| 23/46 模块混淆消融)，解释现有文献评估为何难以确立归因                  |
| [Table 1: 文献对比表; Table 2: 审计生命周期分布精简摘要]              |
+-----------------------------------┬------------------------------------+
                                    |
+-----------------------------------v------------------------------------+
| Section 3: Mathematical Formulation & Decision-Theoretic Foundation    |
| 日前调度与非对称损失形式化 -> Lemma 1 决策论超额风险精确积分恒等式      |
| 二阶局部近似与三项分解 (Term I 预测误差, Term II 校准弥散, Term III 耦合)|
| 理论反例 1 (平移偏差吸收) 与 反例 2 (空间异方差下点误差与决策损失解耦) |
| 双结算电力市场与 BSC 结算机制                                         |
+-----------------------------------┬------------------------------------+
                                    |
+-----------------------------------v------------------------------------+
| Section 4: Controlled Empirical Testbed & Attributional Protocol       |
| 5D 约束生命周期形式化定义 (D1--D5)                                    |
| 三风场运维 SCADA 测试平台与数据清洗                                   |
| 受控归因协议: Authentic vs. Scrambled vs. None 阴性对照设计            |
| 样本梯队、500 样本信息预算对齐、代理滞后、连续失配测试方案            |
| DLinear 基准、PatchTST 敏感性探针与分位数校准规范                     |
+-----------------------------------┬------------------------------------+
                                    |
+-----------------------------------v------------------------------------+
| Section 5: Core Controlled Ablations: When Does Physics Actually Help? |
| Penmanshiel 样本受限下真实先验与打乱先验鲜明对比 (-1.22% vs +68.1%)    |
| 弱约束 (λ=0.01) 真实与打乱紧密重合，表现为非特异性数值平滑            |
| 500 样本信息预算对齐实验: 保留 77%--92% 收益，证实收益非纯历史数据泄漏 |
| [Figure 1: 核心受控消融全景图; Table 3: 核心消融主表; Table 4: 预算表] |
+-----------------------------------┬------------------------------------+
                                    |
+-----------------------------------v------------------------------------+
| Section 6: Boundary Stress Tests: Where Does the Advantage Break Down? |
| 样本量学习曲线: N=500 收益收缩至 N=5000 优势消失/略反转 (偏置-方差假说)|
| 连续先验扰动应力测试: Penmanshiel 上 α≈0.07 经验交叉区与机组老化降额   |
| 气象代理时间滞后测试: Hill of Towie 上 k≈12 穿透无约束基准            |
| PatchTST 跨架构敏感性探针分析                                          |
| [Figure 2: 学习曲线图; Figure 3: 连续失配与漂移图; Table 5: PatchTST]  |
+-----------------------------------┬------------------------------------+
                                    |
+-----------------------------------v------------------------------------+
| Section 7: Decision Alignment & Supporting Computational Graph Audit   |
| 指标解耦实证: Info-Symmetric 臂 Kelmarsh 上 RMSE 改善 1.16% 成本劣化   |
| Zero-Bid 归谬诊断: 零申报机械获取六位数单边现金流，证实单边核算缺陷    |
| 支撑性计算图审计 (PCT): AST 逆向反向 JS 导致 O(1/ε) 梯度放大实测       |
| 2D Product-Kernel KDE 代数张量展开负对照验证                           |
| [Figure 4: 决策脱钩与 Zero-Bid 图; Figure 5: PCT 梯度与分歧图; Table 6]|
+-----------------------------------┬------------------------------------+
                                    |
+-----------------------------------v------------------------------------+
| Section 8: Discussion, Conceptual Synthesis & Governance Checklist     |
| 偏置–方差机制假说的综合理论讨论 (Conceptual Bias-Variance Synthesis)   |
| 严格界定“受控识别所确立的内容”与“未被识别的混杂因素与边界”            |
| 审计过程中被证伪的两项初始猜想                                         |
| 面向工程部署的“物理先验 5 步判别流程”与 REFORMS C1--C5 方法学清单     |
| [Table 7: C1--C5 方法学治理清单]                                       |
+-----------------------------------┬------------------------------------+
                                    |
+-----------------------------------v------------------------------------+
| Section 9: Conclusion                                                  |
| 总结核心实证发现与边界，重申物理先验为条件性归纳护栏                   |
+------------------------------------------------------------------------+
```

---

## 五、严禁措辞对照与法定许可学术表达库

| 严禁使用的过度宣称 (Prohibited Overclaim) | 法定许可的严谨学术表达 (Permitted Calibrated Wording) |
|:---|:---|
| ❌ *“We prove the bias-variance tradeoff equation $\Delta \text{Risk} = \Delta \text{Variance} - \Delta \text{Bias}$.”* | ✅ *“The observed learning curves are consistent with a transition in which the regularizing value of a fixed prior diminishes as data availability increases, while structural approximation error becomes increasingly consequential.”* |
| ❌ *“$N \le 1{,}000$ is the universal data-scarce regime and $N \ge 3{,}000$ is the universal structural-bias regime.”* | ✅ *“In the tested sample-restricted regimes ($N \le 1{,}000$) ... whereas at the largest evaluated training budget ($N=5{,}000$) under the tested strong-prior setting, unconstrained models match or slightly outperform fixed priors.”* |
| ❌ *“Our Causal Attribution Protocol proves the universal causal effect of physics.”* | ✅ *“Our Controlled Attribution Protocol provides evidence that the empirical physical mapping itself matters beyond generic regularization under the tested configuration, without asserting universal causal effects of physics as an abstract treatment.”* |
| ❌ *“Physical priors universally fail above a catastrophic 7% corruption threshold.”* | ✅ *“On Penmanshiel, the tested corruption trajectory crosses the unconstrained baseline around an empirical crossover region of $\alpha \approx 0.07$.”* |
| ❌ *“Priors are proven safe for $k \le 12$ and unsafe for $k > 12$ as a universal deployment limit.”* | ✅ *“In the Hill of Towie lag scan, decision risk crosses the unconstrained reference around $k \approx 12$ and continues to deteriorate at larger tested lags under the evaluated training loss proxy query.”* |
| ❌ *“At $\lambda=0.01$, genuine and scrambled priors are perfectly identical / proven equivalent.”* | ✅ *“Under weak penalty scaling ($\lambda=0.01$), genuine and scrambled priors yield closely matched outcomes with no clear empirical separation, remaining consistent with non-specific numerical regularization.”* |
| ❌ *“Our experiments verify a full factorial $N \times \lambda$ interaction theorem.”* | ✅ *“Taken together, the learning-curve and penalty-sensitivity experiments suggest that both data availability and regularization strength govern the operating envelope.”* |
| ❌ *“The 500-sample experiment completely eliminates every conceivable form of data leakage.”* | ✅ *“Evaluating budget-matched priors confirms that the majority of the observed benefit (77\% to 92\%) cannot be explained by additional historical data volume alone.”* |
| ❌ *“Zero-Bid generates £240k of genuine commercial trading profit.”* | ✅ *“A degenerate zero-bid baseline mechanically generates six-figure apparent balancing cashflow components by converting generation into unhedged spill, demonstrating the structural pathology of standalone imbalance accounting.”* |

---
*本蓝图严格作为 Phase 4 LaTeX 正文改稿的法定纲领性文件。*
