# 论文叙事实证取证与第一性原理重构审计报告
# (Narrative Reconstruction Forensic Audit)

> **审计基准**：面向第一性原理科研叙事重构（First-Principles Scientific Narrative Reconstruction）  
> **审计对象**：`01_论文投稿材料/active_manuscript/` 全量 LaTeX 源文件、`06_实验结果与日志/` 原始实验数据总账及历史取证材料  
> **核心军规**：严格保护既有实验事实，严禁任何数据篡改、幽灵指标或无对照因果断言。

---

## 一、论文唯一主问题的确立与偏离诊断

### 1.1 唯一主问题的精确表述
本研究要解决的**不是**：
- *“如何提出一个更强大的物理约束风电预测模型架构？”*（算法/架构发明范式）
- *“现有物理约束预测文献是否存在复现危机？”*（纯元科学/学术打假范式）

本研究真正且唯一的科学主问题是：
> **When a physical prior is introduced into wind power forecasting, does it provide genuine decision value because it encodes valid domain knowledge, and under what conditions does that value persist, disappear, or reverse?**  
> （当物理先验被加入风电预测模型后，它究竟什么时候因为“真实物理领域知识”而改善实际电网运行决策，什么时候只是起到普通数值正则化作用，什么时候又会因为数据充足、先验失配或时间错位而反过来伤害决策？）

### 1.2 论文的核心论点与理论模型
论文最终确立的核心科学命题必须统一收敛至：
> **Physical priors are conditional inductive guardrails rather than universally beneficial performance enhancers.**  
> （物理先验是具有严格有效边界的“条件性归纳护栏”，而非普遍有益的性能增强器。）

其背后的底层理论解释模型为**估计方差与结构偏置的权衡（Bias–Variance Trade-off under Prior Induction）**：
$$\text{Net Decision Benefit} = \Delta \text{Variance Reduction} - \Delta \text{Structural Approximation Bias}$$
- **在数据匮乏（Data Scarcity, $N \le 1{,}000$）时**：无约束数据驱动模型的估计方差占主导，正确设定的物理先验能够有效约束假设空间（Hypothesis Space），其方差削减收益显著大于静态物理近似引入的结构偏置，此时物理先验展现出稳健的下行决策风险保护（Downside Guardrail）；
- **在数据充足（Data Abundance, $N \ge 3{,}000$）时**：纯数据驱动模型已具备足够的经验样本来逼近真实气动响应，估计方差大幅收敛。此时静态物理先验不再能提供边际方差削减，反而是其固化、理想化的先验代数关系退化为刚性结构偏置（Structural Bias），导致物理先验模型的表现被纯数据模型超越甚至反转；
- **在先验失配（Misspecification / Scrambled）时**：在强约束惩罚（$\lambda = 1.0$）下，错误的先验持续向模型参数注入反物理梯度压力（Counter-physical Gradients），造成灾难性的决策损失膨胀；
- **在微弱惩罚（Weak Regularization, $\lambda = 0.01$）时**：物理映射几乎未实质改变表征特征，真实先验与伪打乱先验表现完全重合，所谓“提升”纯属非特异性数值平滑；
- **在代理滞后（Proxy Staleness, $k > 12$）时**：离线损失函数所依赖的气象代理变量与预测目标时序错位，导致“物理知情”蜕变为“物理错位”（Physics-Misaligned），归纳收益单调衰退。

---

## 二、当前 Manuscript 的 Central Claim 与章节 Claim 现状诊断

### 2.1 当前稿件的 Central Claim 诊断
在现行 `01_intro.tex` 与 `08_conclusion.tex` 中，文章被概括为三个并列支柱：
1. 文献与软件可用性审计（46 篇 PRISMA 调查 + PCT 逆向 KL 散度）；
2. 多风场受控实验（Penmanshiel、Kelmarsh、Hill of Towie 的消融与学习曲线）；
3. 决策评价与报告治理（三项分解、Zero-Bid 诊断与 REFORMS 清单）。

**诊断结论**：当前稿件存在严重的**“四重叙事割裂（Parallel-Story Problem）”**。文章试图在同一篇论文中同时扮演四种角色：
- 角色 A：文献计量学/元科学 Scoping Review（PRISMA-ScR、Fleiss $\kappa$ 盲审、0% 负面报告率）；
- 角色 B：开源代码逆向工程审计（PyTorch ATen C++ 内核、AST 解析、$\mathcal{O}(1/\epsilon)$ 逆密度求导）；
- 角色 C：电力市场设计与不平衡结算批判（BSC 单价结算、Zero-Bid 六位数虚假现金流归谬）；
- 角色 D：风电物理约束预测的实证机器学习基准测试（阴性对照、学习曲线、延迟应力测试）。

这四重叙事在当前正文中处于平级陈列状态，审稿人极易陷入认知困惑：*“这到底是一篇 Systematic Review，一份软件 Bug Report，一篇电力市场经济学分析，还是一篇风电预测论文？”*

### 2.2 当前各章节实际承担的 Claim 穿透审计

| 现行章节 | 现行标题 | 实际承担的内容与 Claim | 存在的核心问题与叙事错位 |
|:---|:---|:---|:---|
| **§1** | Introduction | 提出电力市场痛点；概述三大并列贡献（文献/软件审计、消融基准、决策分解/清单）。 | **动机混杂**：将 46 篇文献审计作为第一贡献，将 PCT 软件 bug 写入贡献，冲淡了“物理先验是否有真正决策价值”的核心科学张力。 |
| **§2** | Related Work | 梳理物理约束风电预测、决策评价、科学 AI 复现性三支文献；给出 Table 1（本审计与现有综述的对比）。 | **功能定位模糊**：过度聚焦于证明“为什么我们需要做审计综述”，未能服务于“归因困难（Attribution Failure）的 8 大假象”。 |
| **§3** | Theory | 给出日前调度形式化、引理 1 积分非负性、二阶三项展开（Term I, II, III）、两个反例、双结算市场与 Elexon BSC 模型。 | **未与主问题闭环**：理论分解非常扎实，但目前仅作为静态数学推导存在，未与后文的 Bias-Variance 假说与边界图形成无缝映射。 |
| **§4** | Methodology | 5D 约束全生命周期分类法（$\mathcal{D}_1$--$\mathcal{D}_5$）；PRISMA-ScR 协议；AST 软件审计协议；3 风场 SCADA 与 Elexon 数据；DLinear/PatchTST 实验设计。 | **前置审计权重过大**：将 5D 分类法和 PRISMA 协议作为方法论主体，将受控实验设计（Genuine vs Scrambled）挤在末尾。 |
| **§5** | Audit (Literature & Code) | 汇报 46 篇文献审计结果（0% 负面报告，95.7% 开源赤字，50% 消融混杂）；Fleiss $\kappa$；PCT 逆向 JS 求导与梯度放大 $2.2\times$--$2.7\times$；2D 核负对照证明。 | **严重反客为主**：本节全篇在讲文献缺陷和软件 bug，直接占据了文章核心视觉版面（Table 1, Table 4, Fig 2, Fig 5），推迟了真正的因果实证。 |
| **§6** | Experiments | 汇报 Penmanshiel 先验特异性（$-1.22\%$ vs $+68.1\%$）；Kelmarsh/Hill 学习曲线；500 样本信息预算；低 $\lambda$ 拟态；代理滞后扫描；动态结算与 Zero-Bid；PatchTST 探针。 | **核心实证被包夹**：包含最硬核的科学证据，却被放置在第 6 节，实验小节之间各自独立，缺乏统一的“边界图（Boundary Map）”统领。 |
| **§7** | Discussion | 总结四大边界；厘清识别边界（What is identified vs not）；被证伪的两个猜想；提出 Table 6（C1--C5 验证清单）。 | **机理解释深度不足**：更多是在罗列实验现象，未将 Bias–Variance 理论解释提升为核心概念模型。 |
| **§8** | Conclusion | 总结三点发现：文献无负面报告、PCT 软件实现错位、实证具有条件性。 | **重心偏离**：结论第一点依然在强调文献 0% 报告和 95.7% 开源赤字，使审稿人记住的是审计数字而非物理先验的本质。 |

---

## 三、四类审计材料的证据层级重构（Evidence Hierarchy Realignment）

为了彻底解决 Parallel-Story 缺陷，必须打破四类材料的平级状态，确立绝对清晰的四级证据金字塔：

```
                    ┌──────────────────────────────────────────────┐
                    │   LEVEL 1: 核心因果归因证据 (Main Causal)       │
                    │   Genuine vs. Scrambled vs. None 消融        │
                    │   Penmanshiel: -1.22% vs. +68.1%             │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │   LEVEL 2: 有效性边界应力证据 (Boundary Map)   │
                    │   • 样本量学习曲线 (N=500 -> 5000 优势反转)     │
                    │   • 约束强度 λ 敏感性 (λ=0.01 正则化拟态)      │
                    │   • 先验失配与工程漂移 (α 连续退化, 老化, 漂移) │
                    │   • 气象代理时间滞后 (k=6 -> 36 衰退曲线)       │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │   LEVEL 3: 评价准则对齐证据 (Decision Metric)  │
                    │   • RMSE 与调度损失解耦 (点预测改善 vs 决策恶化) │
                    │   • Zero-Bid 归谬诊断 (单边结算现金流病态脱钩)   │
                    │   • 理论三项分解 (Term I, II, III 与反例)        │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │   LEVEL 4: 支撑性取证证据 (Supporting Forensic)│
                    │   • 46 篇 Scoping: 解释为何领域过去缺乏该控制  │
                    │   • PCT 代码错位: 解释为何只看数学公式会受骗   │
                    │   • REFORMS 清单: 固化为未来研究规范           │
                    └──────────────────────────────────────────────┘
```

### 3.1 每一级的具体定位与降级/升级策略
1. **LEVEL 1 — 核心因果归因证据（提升为全文 Empirical Centerpiece）**：
   - 实验设计：严格保证同一 Backbone（DLinear）、相同容量、相同划分、相同分位数校准下，对比：
     $$\text{Genuine Physical Prior} \quad \text{vs.} \quad \text{Pseudo-Scrambled Prior} \quad \text{vs.} \quad \text{Unconstrained Baseline}$$
   - 核心数据锚点：Penmanshiel（$N=1,583, \lambda=1.0$）下，真实先验降低调度损失 **$1.22\%$**（$0.09760$ vs $0.09881$），而打乱先验暴增 **$+68.1\%$**（$0.16611$ vs $0.09881$）。
   - 核心论证逻辑：这是证明先验收益源于“物理领域知识”而非“非特异性数值平滑”的不可反驳证据。

2. **LEVEL 2 — 边界地图证据（整合成单一的 Physical Prior Validity Envelope）**：
   - 不再分散为孤立的小实验，而是围绕 **“物理先验何时从防御护栏变成结构偏置”** 绘制多维运行包络线：
     - 样本量维度（$N$）：$N \le 1,000$ 收益明显；$N \ge 3,000$ 优势消失并反转（Hill: $+1.73\%$, Kelmarsh: $+0.33\%$）；
     - 强度维度（$\lambda$）：$\lambda=0.01$ 下真实与打乱完全重合（$0.08810$ vs $0.08801$），证明弱惩罚仅为数值平滑；
     - 先验质量维度（$\alpha$）：连续打乱在 $\alpha \approx 0.07$ 即穿透无约束基线；风机老化超过 10% 迅速翻转为惩罚；
     - 代理时序对齐维度（$k$）：历史代理延迟 $k \in \{6, \dots, 36\}$ 导致调度损失单调恶化，在 $k \approx 12$ 失去所有收益。

3. **LEVEL 3 — 评价有效性证据（证明 Primary Endpoint 必须是非对称决策损失）**：
   - 确立唯一的 PRIMARY ENDPOINT：与真实调度成本比率对齐的归一化非对称决策损失 $\text{Cost}_{\text{static}}$（$r=2, q=1/3$）；
   - 将 RMSE/MAE、不平衡结算现金流降级为 SECONDARY DIAGNOSTICS；
   - 证据闭环：
     - RMSE 解耦证据：对等信息输入组（Info-Symmetric）RMSE 降低，但尾部调度成本反弹；
     - Zero-Bid 诊断：$q \equiv 0$ 机械产生 22--24 万英镑年化结算现金流，证明单边现金流存在结构性脱钩，绝不能作为评价指标。

4. **LEVEL 4 — 支撑性取证证据（降级为动机说明与验证层）**：
   - **46 篇 Scoping 审计**：绝不作为第一独立科学发现，而是作为引言和方法论的“现实反思背景”——回答*“为什么当前风电预测领域此前从未可靠回答过 RQ1 和 RQ2？”*（因为 0% 文献测试破坏边界，95.7% 无代码，50% 存在参数混杂消融）；
   - **PCT 代码审计**：回答*“为什么仅阅读论文中的数学公式仍然不足以识别物理归因？”*（因为存在参数颠倒导致的 $\mathcal{O}(1/\epsilon)$ 奇异点实现分歧）；
   - 这两项工作是“为什么需要本文实验协议”的支撑证据，绝不能绑架论文的主线。

---

## 四、全新 Claim Hierarchy 与证据链完备性审查

### 4.1 核心 Claim 树状图
- **CENTRAL CLAIM**：Physical priors in wind power forecasting provide conditional inductive guardrails rather than universally beneficial performance enhancers; their decision value is strictly governed by data scarcity, prior fidelity, penalty strength, and temporal alignment.
  - **SUBCLAIM A (Prior Specificity under Data Scarcity)**：在小样本与强惩罚制度下，正确设定的气动功率曲线先验能提供具有领域特异性的归纳指导，显著降低非对称调度损失。
  - **SUBCLAIM B (Validity Boundary Breakdown)**：随着训练数据规模扩大，纯数据驱动模型可自行学习气动关系，物理先验的方差削减收益消退，静态偏置开始主导；而在代理滞后、先验漂移或过度约束时，先验收益发生不可逆反转。
  - **SUBCLAIM C (Weak-Regularization Mimicry)**：在弱惩罚制度下，打乱映射的伪先验能够完美复现真实先验的表现，此时观测到的边际收益属于非特异性数值平滑，不可归因于物理知识。
  - **SUBCLAIM D (Decision Metric Decoupling)**：物理先验的价值主张必须建立在非对称决策风险上，因为对称点预测误差（RMSE）与单边市场结算现金流存在结构性脱钩与病态归谬。
  - **SUPPORTING CLAIM (Methodological & Software Blindspots)**：现有文献普遍缺乏区分真实物理指导与上述替代解释所需的受控实验协议与开源计算图执行保真度。

---

## 五、现有证据对新主线的支撑力评估与 Evidence Gap 分析

### 5.1 支撑力极强的证据矩阵（Ironclad Evidence）
1. **Penmanshiel 强惩罚特异性**：真实先验 $-1.22\%$ vs 伪打乱 $+68.1\%$（3 种子 [42, 43, 44]，std=0.00044），因果对比极其尖锐，彻底锁死 RQ1。可在 `ablation_3d_key_comparison_table.csv` 与 `table2_dynamic_settlement.tex` 表注中直接核验。
2. **场站内学习曲线**：跨 Kelmarsh 与 Hill of Towie 两个风场，各 5 个样本量阶梯（500 到 5000），各 5 个随机种子。真实先验在 500/1000 时占优，在 3000/5000 时被无约束模型超越（$+0.33\%$ 和 $+1.73\%$），彻底解耦了样本量与地形，完美支撑 Bias-Variance 假说。
3. **信息预算对齐实验**：严格限制在 500 样本的先验保留了全历史先验 77%--92% 的收益，而打乱先验暴增 43%--46% 损失，彻底证伪了“额外历史数据泄漏”的竞争假说。
4. **弱惩罚拟态**：Kelmarsh 与 Hill of Towie 在 $\lambda=0.01$ 下，真实先验与伪打乱先验均值差均在 $0.00003$ 以内，误差棒完全重叠，完美支撑 SUBCLAIM C。
5. **代理延迟单调衰退**：$k=6 \to 36$ 步在三场站均表现出单调成本递增，Hill of Towie 在 $k \approx 12$ 穿透基线，完美支撑时间错位边界。
6. **Zero-Bid 归谬与指标反转**：Hill of Towie 上分布先验使单边现金流虚增 $+42.5\%$，而决策损失恶化 $+6.5\%$，完美支撑 SUBCLAIM D。

### 5.2 潜在的 Evidence Gap 与必须恪守的措辞防御边界（Hedges）
尽管实证矩阵极为完整，但在重构时必须诚实声明以下局限性，严禁过度延伸：
1. **Gap 1: 跨风场地形差异并非受控因果实验**：
   - 现况：Kelmarsh（平坦）、Hill of Towie（复杂丘陵）、Penmanshiel（高地）在机型、容量、轮毂高度和样本量上均不同。
   - 防御边界：**严禁声称“复杂地形必然因果导致物理约束收益”**。必须将地形差异明确降维为“背景工况变化（Contextual Heterogeneity）”，真正的第一驱动因是样本量 $N$ 与先验质量。
2. **Gap 2: 物理先验类型的局限性**：
   - 现况：实证测试的先验主要为离线经验功率曲线与 2D KDE 分布先验（属于 $\mathcal{D}_1=\texttt{historical-static}$）。
   - 防御边界：**严禁声称“全人类所有物理先验（如动态流体 PDE、在线自适应先验）均在大样本下失效”**。必须明确限定在“离线经验功率曲线与静态分布约束”这一工业主流先验范畴内。
3. **Gap 3: 理论三项分解中 Term III 的定性诊断属性**：
   - 现况：在真实 SCADA 数据中，$F_{Y|X}$ 真实条件分布未知，Term III 无法逐点精确数值求积。
   - 防御边界：正文必须保留并加粗 **Diagnostic Caveat**，强调三项分解是解释决策解耦现象的“定性理论框架”，而非可点对点独立测量的经验因果量。
4. **Gap 4: PatchTST 的探索性探针定位**：
   - 现况：PatchTST 仅测试了名义参数 $\lambda=0.01$，未执行跨 $\lambda \in \{0.01, 0.1, 1.0\}$ 的网格搜索。
   - 防御边界：正文必须明确将其定性为“轻量架构敏感性探针”，不可声称完成了跨架构穷尽搜索。

---

## 六、图表与章节重构操作策略

### 6.1 现有 Figures/Tables 重构优先级与裁撤建议（基于当前稿件 `main.aux` 编译事实）

| 编译序号与标签 | 原稿件文件与位置 | 实际内容 | 重构新定位与处置方案 | 优先级 |
|:---|:---|:---|:---|:---:|
| **Figure 5**<br>`fig:ablation_3d` | `fig_causal_ablation.pdf`<br>(06_experiments.tex L13) | 3D 受控消融基准图：(a) Penmanshiel 特异性；(b) Kelmarsh 弱 $\lambda$ 拟态；(c) 代理延迟衰退；(d) RMSE 决策解耦 | **提升为全文视觉核心图（Empirical Centerpiece）**。直接回答 RQ1 与 RQ2，形成物理先验因果与边界的核心证据。 | **P0 (最高)** |
| **Figure 6**<br>`fig:within_site_learning_curve` | `fig_within_site_learning_curve.pdf`<br>(06_experiments.tex L23) | Kelmarsh & Hill of Towie 跨 5 样本量学习曲线 ($N \in [500, 5000]$) | **保留并提升为核心边界图 (Boundary Map: Sample Size Envelope)**。直观展现小样本物理保护与大样本优势反转，支撑 Bias-Variance 理论解释。 | **P0 (最高)** |
| **Figure 7**<br>`fig:prior_corruption_gradient` | `fig_prior_corruption_gradient.pdf`<br>(06_experiments.tex L38) | 连续打乱 $\alpha$、机组老化降额与测风仪漂移衰退曲线 | **保留并作为核心边界图 (Boundary Map: Prior Corruption & Drift)**。证明物理先验存在精确的失效穿越阈值（$\alpha \approx 0.07$）与工程容差窗口。 | **P1 (核心)** |
| **Figure 8**<br>`fig:market_decoupling` | `fig_market_decoupling.pdf`<br>(06_experiments.tex L69) | Zero-Bid 归谬诊断与 Hill of Towie 指标反转对比 | **保留并作为指标有效性诊断图 (Decision Alignment & Metric Decoupling)**。证明单边平衡现金流与真实预测技能脱钩。 | **P1 (核心)** |
| **Figure 4**<br>`fig:pct_loss_empirical` | `fig_pct_loss_empirical_contrast.pdf`<br>(05_audit.tex L90) | PCT 官方代码逆向 JS 梯度放大与标准 JS 训练对比 | **降级并移入支持性取证章节或附录**。作为证明“为什么必须静态审计开源计算图”的典型案例，绝不抢占正文视觉焦点。 | **P2 (支持)** |
| **Figure 1**<br>`fig:constraint_lifecycle` | `fig_constraint_lifecycle.pdf`<br>(04_methodology.tex L18) | 5D 约束全生命周期架构图 ($\mathcal{D}_1$--$\mathcal{D}_5$) | **精炼保留在方法学**。作为先验分类与协议规范的形式化框架。 | **P2 (支持)** |
| **Figure 2**<br>`fig:prisma_flow` | `prisma_flow_diagram.pdf`<br>(04_methodology.tex L43) | PRISMA-ScR 文献筛选流程图 | **压缩至单栏或移入附录**。避免正文前部产生纯文献综述观感。 | **P3 (附录/支持)** |
| **Figure 3**<br>`fig:field_audit` | `fig_field_audit_distribution.pdf`<br>(05_audit.tex L18) | 46 篇文献在 5D 生命周期各阶段的分布统计 | **移入附录或合并**。作为支持性背景证据。 | **P3 (附录/支持)** |
| **Table 4**<br>`tab:controlled_ablation` | `table2_dynamic_settlement.tex`<br>(06_experiments.tex L11) | 三场站受控消融主账本（Genuine, Scrambled, Unconstrained, Info-Symmetric 等） | **保留并重构为 Table 1 (Core Empirical Benchmark)**。是全文最核心的数据总账。 | **P0 (最高)** |
| **Table 5**<br>`tab:information_budget` | `table_information_budget.tex`<br>(06_experiments.tex L27) | 500 样本严格对齐信息预算受控实验 | **保留为 Table 2 (Information Budget Control)**。证明先验收益源于几何结构而非历史季节性泄漏。 | **P0 (最高)** |
| **Table 7**<br>`tab:zero_bid_diagnostic` | `table4_zero_bid_diagnostic.tex`<br>(06_experiments.tex L67) | 三风场 Zero-Bid 归谬核算诊断表 | **保留为 Table 3 (Metric Validity Diagnostic)**。证明单边平衡现金流的病态性。 | **P1 (核心)** |
| **Table 6**<br>`tab:paired_bootstrap` | `table3_paired_bootstrap.tex`<br>(06_experiments.tex L53) | 7-Day 块 Bootstrap 动态结算置信区间表 | **保留**。展示真实电价波动下的结算分布。 | **P1 (核心)** |
| **Table 8**<br>`tab:patchtst_and_rules` | `table5_patchtst_and_rules.tex`<br>(06_experiments.tex L123) | PatchTST 跨架构敏感性探针结果 | **保留作为轻量敏感性探针**。证明发现不仅限于 DLinear 单一架构。 | **P2 (支持)** |
| **Table 9**<br>`tab:verification_checklist` | `table_reforms_checklist.tex`<br>(07_discussion.tex L52) | REFORMS-PCWPF 方法学核验清单 (C1--C5) | **保留并作为 Discussion 总结输出**。形成可操作的方法论操作指南。 | **P1 (核心)** |
| **Table 1**<br>`tab:lit_comparison` | `02_related.tex` L31 | 本文与现有物理约束综述对比表 | **精炼保留在 Related Work**。说明现有综述缺乏因果受控消融。 | **P2 (支持)** |
| **Table 2 & 3**<br>`tab:lifecycle_distribution`, `tab:ablation_semantics` | `table1_lifecycle_distribution.tex`, `table4_ablation_semantics.tex` | 46 篇统计分布与消融语义分类表 | **压缩合并为单表或移入附录**。作为支持性背景证据。 | **P3 (附录/支持)** |

---

## 七、PHASE 1 审计结论

经全面取证，当前代码库、数据 CSV 与实验日志**完全具备**支撑“第一性原理科研叙事重构”的全部必要证据。新主线不仅不会掩盖任何真实数据，反而能将此前被文献审计与代码审计掩盖的、真正具备极高学术震撼力的物理先验“双刃剑”本质与条件性边界推向舞台中央！
