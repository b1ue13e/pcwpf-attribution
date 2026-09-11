# 09. 论文重构前后架构对照与逻辑演进 (Manuscript Structure Before & After)

根据顶刊锚点论文（Applied Energy / PRX Energy / Science REFORMS）的标准范式，对当前手稿进行全面解构与逻辑重塑。

---

## 一、 重构前存在的核心结构与叙事病灶

1. **结构跳跃（Methods/Results Oscillation）**：
   原稿顺序为：`04_protocol` $\rightarrow$ `05_results` $\rightarrow$ `06_code_audit` $\rightarrow$ `07_theory` $\rightarrow$ `08_discussion`。
   读者读完 Section 5 的主实证结果后，突然又跳回 Section 6 的代码反编译细节，紧接着又跳到 Section 7 补推导决策论公式，结构严重破碎。
2. **多头宣称贡献，主线被淹没**：
   文献统计（PRISMA-ScR）、代码缺陷（PCT reverse JS）、理论推导（Proposition 1）、3D 消融、市场零申报、以及规范清单各自为政，缺乏单一主干问题（Central Question）的统摄。
3. **开会式汇报与同一数据反复出现四次**：
   摘要、引言贡献点、结果节、讨论节、结论节中，`0/46`, `95.7%`, `50.0%`, `-1.22%`, `+68.1%` 被逐字逐句无差别完整复述多达 4 次，严重削弱了论文的论证深度。
4. **讨论提前侵入结果节（Discussion Leaking into Results）**：
   在 Results 汇报具体表格数值时，大量混入半页篇幅的认识论批判和文献发表偏倚反思，模糊了客观事实与主观解释的边界。

---

## 二、 重构后顶刊标准八联排架构 (Standard 8-Section Architecture)

```
====================================================================================================
1. Introduction
   - 核心矛盾开场：风电非对称决策风险 vs 传统点误差评价
   - 归因缺口：加入物理后误差下降，能否归因于物理信息？
   - 现有文献的三大方法学盲区：缺乏负对照、模块消融混杂、孤立单边现金流陷阱
   - 本文审计切入点与三大核心发现（克制定量概括）

2. Background & Decision-Theoretic Foundation
   - 物理约束风电预测范式与先验溯源
   - 非对称调度损失与分位数决策基准（Lemma 1: 经典决策论最优分位数基准，杜绝虚假理论首创宣称）
   - 英国 2017 GB 电力市场制度实态（自愿性双边交易 + BSC 单一不平衡清算，澄清两阶段会计解构模型）

3. Systematic Scoping & Auditing Methodology
   - 预注册 PRISMA-ScR 范围审查检索与双盲筛选（N=46 核心语料 vs 14篇敏感性检索）
   - 物理约束全生命周期五维分类审计体系（D1-D5 Schema v4）
   - 多评价员信度评定（Cohen's Kappa & Fleiss Kappa 校验）
   - 软件代码库审查协议（PCT & PINN-UQ）

4. Controlled Experimental Test Bench
   - 跨地形风电场实证基准（Kelmarsh 平原, Hill of Towie 复杂山地, Penmanshiel 数据受限区）
   - 骨干对齐受控消融实验设计（Vanilla vs Info-Symmetric vs Genuine vs Pseudo-Scrambled Controls）
   - 冻结样本与信息预算控制（澄清先验离线拟合窗口与模型训练窗口）
   - 下游非对称决策与 BSC 物理净额结算评价协议

5. Empirical Audit Results
   5.1 Scoping Audit: 纳入文献中负面效应报告的系统性缺失（0/46 报告边界）与代码缺失（95.7%）
   5.2 Ablation Semantics: 模块消融与参数容量混杂（50.0%）
   5.3 Software Fidelity: PCT 官方实现中逆向 JS 散度参数导致的梯度尺度偏移
   5.4 Controlled Ablation: 强约束下物理特异性（-1.22% vs +68.1%）与弱约束数值拟态（lambda=0.01）
   5.5 Sample Window Dynamics: 样本规模扩展下历史物理先验优势的衰减
   5.6 Training Proxy Staleness & Continuous Corruption Curves
   5.7 Market Settlement Decoupling: 零申报（Zero-Bid）极端负对照暴露孤立不平衡现金流的评价陷阱
   5.8 Cross-Backbone Sensitivity: PatchTST Transformer 架构下的敏感性验证

6. Analytical Mechanisms & Diagnostic Explanations
   - 预测-决策解耦的三项方差分解（Point MSE, Quantile Mismatch, Interaction Covariance）
   - 明确标注：Term III 协方差项作为一致性解释诊断，而非完全隔离的经验因果证明
   - 构造反例（Counterexamples 1 & 2）作为可能性的存在性证明

7. Discussion & Methodological Recommendations
   - 实证审计所识别与未识别的边界（What is identified vs what is not）
   - 物理先验在可再生能源预测中的适用边界条件（有限数据保护 vs 大样本隐式学习）
   - 面向研究者与审稿人的规范化报告建议（基于 REFORMS 的物理约束报告清单）

8. Conclusion
   - 提炼最终科学判断，彻底告别数字简单复述
====================================================================================================
```

---

## 三、 各章节内容重组与裁剪台账

| 章节编号与标题 | 承接原章节文件 | 重构与精简动作 |
|---|---|---|
| **Section 1: Introduction** | `01_intro.tex` | 1. 彻底压缩能源转型宏观空话，开门见山切入“收益归因难”；<br>2. 规范贡献段落，剔除“革命性、首个全面”等浮夸词汇；<br>3. 句长严格压至 20--30 词。 |
| **Section 2: Background** | `03_framework.tex` | 1. 将原 Section 3 的数学背景前移至 Section 2；<br>2. 修正英国 2017 GB 市场描述，澄清自愿性双边+BSC 清算体系；<br>3. 公式改写为 $P_{\text{F}}$，说明两阶段收益会计解构属性。 |
| **Section 3: Protocol** | `04_protocol.tex` | 1. 明确区分 46 篇核心 PRISMA 语料与 14 篇敏感性检索；<br>2. 清晰呈现 D1-D5 生命周期分类判据与 Kappa 信度。 |
| **Section 4: Test Bench** | `04_protocol.tex` (后半) | 1. 规范三座风电场参数描述，明确机组型号与样本量；<br>2. 详细说明模型训练窗口与先验离线拟合区间的信息预算关系。 |
| **Section 5: Results** | `05_results.tex`, `06_code_audit.tex` | 1. 将代码审计（Section 6）合并为 5.3 节，实现实证证据连续呈现；<br>2. 全面归正 Table 4（原Table 5）Penmanshiel Zero-Bid 决策成本（由 0.20584 修正为 0.33689）；<br>3. 澄清陈旧度实验为“训练损失风速代理延迟”，杜绝“推理运行时延迟”误读；<br>4. 统一 $\lambda$ 条件。 |
| **Section 6: Mechanisms** | `07_theory.tex` | 1. Proposition 1 降调为 Lemma 1；<br>2. 修正三项分解推导中的符号问题，将 Term III 标定为定性诊断解释；<br>3. 反例定性为存在性证明。 |
| **Section 7: Discussion** | `08_discussion.tex` | 1. 集中讨论本文的真实外推范围（3座风场、46篇语料）；<br>2. 提出面向物理约束深度学习的 5 项可操作性报告建议。 |
| **Section 8: Conclusion** | `09_conclusion.tex` | 给出克制坚定的学术判断，不单纯堆砌实验百分比。 |
