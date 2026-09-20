# 审稿人致命攻击与抗辩风险登记册（Reviewer Risk Register）
## 针对 Applied Energy / Renewable Energy / IEEE TSTE 严苛同行评议的预先防御与自洽性封堵表

> **定位说明**：本文档以最高标准的“对抗式压力测试（Adversarial Stress-Testing）”为原则，主动站在挑剔审稿人（Devil's Advocate）的角度，全面预演可能导致 Rejection 或 Major Revision 的 12 项重大质疑靶点，并为每一项制定不可辩驳的实证与理论防线。

---

# 一、12 项关键风险评级与预先防御总览

| 风险编号 | 风险领域与靶点 | 风险等级 | 模拟审稿人致命提问（Attack Vector） | 预先防御与文本闭环策略（Preemptive Defense） | 落实状态 |
|:---:|:---|:---:|:---|:---|:---:|
| **R01** | **论文体裁与格式合规性** | **Fatal** | *"Section 11 contains Cell Press 'STAR Methods' and Key Resources table with RRIDs, yet it is submitted to Applied Energy. Is this a rejected paper from another journal?"* | **彻底切除 Section 11**，将关键信息融入正文方法论与声明，彻底杜绝转投痕迹。 | **P0 立即执行** |
| **R02** | **理论认知逻辑倒置** | **Fatal** | *"The authors report excess risk and quantile mismatch in Section 5, but only define the mathematical formulation and lemmas in Section 7. Why is the math inverted?"* | **将 Section 7 理论前置为 Section 2**，在实验之前建立完整的优化与分解数学基础。 | **P0 立即执行** |
| **R03** | **跨风场因果混杂 (Confounding)** | **Major** | *"You claim complex terrain causes physical constraints to help while flat terrain causes harm. But Penmanshiel had only 1,583 samples and different turbines. This is a severe confounding error."* | **主张严格降维**：将地形因果降为“机制假说”，并在 §5.2 突出场站内学习曲线，证明样本量是第一主因。 | **已受控闭环** |
| **R04** | **先验信息预算泄漏** | **Major** | *"The empirical power curve was fitted on 10 months of SCADA, giving the constrained model an unfair seasonal advantage over the 500-sample baseline."* | **突出 Table 3 严格预算对齐实验**：$N_{	ext{prior}}=500$ 依然保留 77%--92% 的收益，彻底证伪泄漏假说。 | **已受控闭环** |
| **R05** | **市场单边现金流定性错误** | **Major** | *"The Zero-Bid strategy making £240k/yr is fictitious because real wind farms sign PPAs. Standalone imbalance cashflows do not represent commercial profit."* | **严格定义为“归谬法结算诊断”**，写明式 (40) 双结算模型，确立非对称调度损失为真实评价指标。 | **已受控闭环** |
| **R06** | **代码审计道德化指控风险** | **Major** | *"Auditing PCT code and claiming it 'diverges from published equations' might be considered an aggressive attack on a peer's integrity."* | **措辞彻底中立化**：使用 *implementation divergence*，并在 §6.4 主动给出 2D 核展开数学成立的反例。 | **已受控闭环** |
| **R07** | **PatchTST 超参数搜索不充分** | **Major** | *"PatchTST was only evaluated at nominal $\lambda=0.01$, unlike the multi-scale $\lambda$ grid of DLinear. How can you claim cross-backbone validity?"* | **明确限定为“定性架构敏感性探针”**，坦诚承认未做跨骨干联合网格搜索，避免过度宣称。 | **已受控闭环** |
| **R08** | **46 篇 Scoping 样本外推漂移** | **Moderate** | *"You claim 0% of wind forecasting literature reports failure regimes, which is an overgeneralization."* | **全文严格限定**：加上 *"within the audited 46-study scoping cohort (95% Wilson CI: [0.0%, 7.7%])"*。 | **已受控闭环** |
| **R09** | **Term III 交叉项的实测不可测性** | **Moderate** | *"In real data, the conditional density is unknown, so Term III cannot be empirically measured point-by-point."* | **显式加入 Diagnostic Caveat 段落**，明确三项分解是定性运筹诊断模型，而非实测经验测度。 | **已受控闭环** |
| **R10** | **训练代理延迟与推理遥测混淆** | **Moderate** | *"If the model requires weather data with lag k, doesn't it violate the claim that the model has no runtime telemetry requirements?"* | **清晰界定**：推理时输入仅为历史功率（$\mathcal{D}_3=	exttt{none}$），滞后 $k$ 仅指训练损失中的代理查询敏感性。 | **已受控闭环** |
| **R11** | **简单线性基线超越深度模型的尴尬** | **Moderate** | *"On Penmanshiel, Ridge and Persistence beat all constrained deep models. Does this invalidate your physical constraint study?"* | **逆向转化为核心贡献**：实证揭示“盲目堆砌深度网络会加剧少样本过拟合，物理约束亦无法完全弥补”。 | **已受控闭环** |
| **R12** | **AI 生成腔调与抽象名词堆砌** | **Minor** | *"The text contains dense AI-style vocabulary (groundbreaking, remarkable, multifaceted operationalization)."* | **执行严格的去 AI 化重写**：遵循 Observation-Evidence-Mechanism-Boundary，句子平均控制在 20--28 词。 | **P1 重构落实** |

---

# 二、三大主力审稿人对抗场景预演与实弹防御

## 场景 1：面对 Reviewer #2 (以刁钻挑剔著称的方法学专家)
- **审稿人开火**：
  > *"The authors claim that physical priors act as a 'defensive guardrail' under data scarcity. However, if the prior is fitted on historical data, how do you know the improvement isn't simply because you gave the model more data through the backdoor? This looks like data leakage."*
- **我们的绝杀防御 (Bulletproof Defense)**：
  > “我们在正文 §5.2 和 Table 3 中前瞻性地设计了**严格信息预算匹配实验（Strict Information Budget Ablation）**。我们将经验功率曲线的拟合样本严格限制在与神经网络模型完全一致的 500 个样本时间窗口内（$N_{	ext{prior}} = N_{	ext{train}} = 500$）。实验证实，在完全剥离额外历史季节性数据的条件下，真实先验在 Kelmarsh 和 Hill of Towie 依然保留了全历史先验 77% 和 92% 的下行风险减免（配对差值 95% CI 均显著为负），而形式匹配的打乱先验暴增 43%--46% 的损失。这确凿地证明了收益来源于气动先验本身的结构性归纳偏置，彻底排除了数据预算泄漏假说。”

---

## 场景 2：面对 Reviewer #3 (电力系统与电力市场专家)
- **审稿人开火**：
  > *"The zero-bid anomaly where bidding zero yields £240k/yr is completely divorced from actual wind power commercial operation. Wind operators do not just dump energy into the balancing mechanism; they trade in day-ahead and intra-day markets. Why should Applied Energy publish a paper that uses a flawed economic metric?"*
- **我们的绝杀防御 (Bulletproof Defense)**：
  > “审稿人指出的核心痛点恰恰是本研究要向整个领域敲响的警钟！在式 (40) 中，我们正是给出了真实的双结算模型 $R_{	ext{total}} = P_F \cdot q + P_{	ext{imb}} \cdot (y - q)$。现有大量文献在评估风电预测时，错误地丢弃日前合约项，仅报告单边不平衡现金流 $P_{	ext{imb}} \cdot (y - q)$。我们的 Zero-Bid 实验正是一个**经典的归谬法诊断（Reductio ad Absurdum Diagnostic）**：它用极端数据证明，如果只算单边不平衡，不作预测的零申报者可以通过将全部出力当作溢出电量而机械获利 24 万英镑。这从数学上彻底封死了‘将单边不平衡现金流作为预测质量代理’的错误做法，强有力地证明了运筹评估必须锚定非负非对称调度损失。”

---

## 场景 3：面对 Applied Energy Associate Editor (关注工程真实性与期刊契合度)
- **编辑开火**：
  > *"This paper spends a lot of space discussing PRISMA scoping reviews and Python ATen kernel code bugs. Is this suitable for Applied Energy, or should it be in a software/data journal?"*
- **我们的绝杀防御 (Bulletproof Defense)**：
  > “本文的核心科学问题高度聚焦于《Applied Energy》最核心的关切：**在新型高比例风电并网的非对称备用调度与真实英国 Elexon 市场清算中，物理约束究竟如何影响运行经济性？** 我们发现的‘双刃剑本质’（在平坦风场产生 -£3,264 的年化单机亏损，在复杂高地产生 +£1,481 的增收）直接解答了电网运营商对于物理 AI 可靠性的重大疑虑。46 篇文献审计与代码审计并非孤立的软工报告，而是解释‘为何过去学术界普遍对物理先验产生盲目乐观偏见’的必要实证归因。重构后的全文严格遵循 IMRaD 范式，理论前置、实证扎实，直接服务于安全可靠的电力能源转型。”

---
*审稿人风险登记册已生成并固化为 `e:\论文1eviewer_risk_register.md`。*
