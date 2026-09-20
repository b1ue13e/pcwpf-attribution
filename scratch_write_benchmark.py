# -*- coding: utf-8 -*-
import os

content = """# 顶刊论文写作与论证基准语料库（Literature Benchmark Corpus）
## 面向 Applied Energy / Renewable Energy / IEEE Transactions on Sustainable Energy 的写作范式、论证逻辑与审稿标准深度解构

> **定位说明**：本语料库针对当前稿件《Auditing the Decision Value of Physics-Constrained Wind Power Forecasting》的重构需求，以 **Applied Energy** 为首选目标，**Renewable Energy** 与 **IEEE TSTE** 为核心对标梯队，遴选并深度阅读 30 篇近年（2021–2026，高权重 2023–2026）经典与顶尖文献。  
> 语料库划分为三大功能层次：  
> - **Tier A：领域直系前沿论文（Direct-Domain Papers，12 篇）** —— 揭示物理约束在风电预测中的主流建模套路、宣称模式与常见评价盲区；  
> - **Tier B：方法学邻近与实证审计论文（Methodological-Neighbor Papers，10 篇）** —— 提供机器学习严苛审计、数据泄漏、阴性对照（Negative Control）、代码保真度与决策对齐（Prediction-to-Decision）的硬核方法学底座；  
> - **Tier C：顶刊写作与结构对标论文（Writing-Benchmark Papers，8 篇）** —— 剖析 Applied Energy / IEEE Transactions 中论证最严密、实验叙事最自洽、Discussion 最透彻、Reviewer 阻力最小的顶刊八股文范式。

---

# 第一部分：三层语料库全景结构清单

| 层次类别 | 篇号 | 论文简写与年份 | 目标刊物/会议 | 核心贡献与研究定位 | 对本重构论文的关键借鉴维度 |
|:---|:---|:---|:---:|:---|:---|
| **Tier A: 领域直系前沿 (12 篇)** | A01 | Gao et al. (2025) | *Applied Energy* | 经验功率曲线 2D KDE 分布对齐与 JS 散度损失 (Corpus S01) | 揭示未校准点预测评价缺陷；作为经验分布先验基准 |
| | A02 | Wang et al. (2026) | *Scientific Reports* | 物理约束 Transformer (PCT, Corpus S02) | 核心被审计对象；AST 揭示反向求导奇异性与核外积数学验证 |
| | A03 | Xu et al. (2025) | *Eng. App. Artif. Intell.* | PINN 与证据不确定性量化 (PINN-UQ, S10) | 代码保真度开源基准；机电代数等式约束建模规范 |
| | A04 | Gijon et al. (2025) | *Applied Energy* | 物理气动与数据驱动混合双流风电预测 | 学习其在 Discussion 中主动披露物理模型失效边界的诚实风格 |
| | A05 | Xu et al. (2024/2026) | *Applied Energy Preprint* | UniWind: 状态路由统一日前风电预测 (S27) | 剖析其模块消融混杂（Module-ablation）之方法学教训 |
| | A06 | Zehtabiyan-Rezaie (2023) | *PRX Energy* | 流体物理引导机器学习风电场预测 (S34) | 学习其在引言中对复杂流体力学与工程先验的克制权衡叙事 |
| | A07 | Guo et al. (2026) | *Renewable Energy* | NCF: 气动功率分支预训练大模型 (S07) | 剖析其气动分支移除引入的参数量混杂问题 |
| | A08 | Nie et al. (2026) | *Energy* | 极端天气下物理生成增强与分层分解 (S03) | 辨析数据生成阶段（D2=data-generation）的物理边界约束 |
| | A09 | Zhang et al. (2026) | *Energy* | 复杂地形非量纲 NS 方程 PINN 尾流预测 (S11) | 复杂地形微气象挑战背景引用；偏微分方程先验的计算开销 |
| | A10 | Li et al. (2025) | *IEEE TII* | 自适应权重 PI-NBeatsx 物理基函数 (S12) | 剖析网络基函数隐式引入物理偏置时的架构混杂 |
| | A11 | Piao et al. (2024) | *IEEE TSTE* | NWP 傅里叶几何风电功率预测 | 学习 IEEE Transactions 严密的数学符号体系与多场站验证 |
| | A12 | Liu et al. (2024) | *Applied Energy* | 极端气象事件下物理强化学习 (S09) | 学习其在强化学习奖励函数中施加物理惩罚的建模技巧 |
| **Tier B: 方法学邻近与审计 (10 篇)** | B01 | Kapoor & Narayanan (2023) | *Patterns (Cell)* | 科学机器学习中的数据泄漏与可复现性危机 | 建立全生命周期泄漏审计指标（D1--D5）与信息预算审查 |
| | B02 | Kapoor et al. (2024) | *Science* | REFORMS 机器学习科研报告黄金标准清单 | 本文 REFORMS-PCWPF 五大门禁清单的法理与结构标杆 |
| | B03 | Lipton & Steinhardt (2019) | *CACM* | 机器学习学术中的浮夸趋势与装饰性数学批判 | 清除浮夸词汇（groundbreaking, novel）；杜绝数学装饰化 |
| | B04 | Pineau et al. (2021) | *JMLR* | 提升 ML 可复现性：NeurIPS 检查表与方差评估 | 确立 4--5 随机种子均值/标准差与块 Bootstrap 方差规范 |
| | B05 | Varoquaux et al. (2022) | *CACM* | 科学人工智能方法学失效与审稿避坑指南 | 学习审稿人视角：如何一眼识别假增益与无对照消融 |
| | B06 | Henderson et al. (2018) | *AAAI* | Deep Reinforcement Learning that Matters | 学习超参数扰动与随机性剥离实验设计 |
| | B07 | D'Amour et al. (2022) | *JMLR* | 欠定性（Underspecification）对 ML 可信度的挑战 | 解释模型在测试分布偏移下相同精度却决策崩溃的根源 |
| | B08 | Pinson et al. (2007) | *IEEE TPS* | 市场惩罚下风电申报出力的最优分位数决策 | 本文核心数学基石：非对称调度损失与关键分位数证明 |
| | B09 | Weron (2014) | *Int. J. Forecasting* | 电力价格预测前沿综述与市场结算特征 | 支撑英国 Elexon 市场极端电价（34.1倍尖峰）结算乘子效应 |
| | B10 | Optis et al. (2022) | *Wind Energy Sci.* | NREL WE-Validate 风能模型开源验证基准平台 | 学习风能专业领域公开基准集构建与多指标无偏评估 |
| **Tier C: 顶刊写作与论证标杆 (8 篇)** | C1 | Yan et al. (2016) | *Applied Energy* | 新能源预测与综合评估指标体系奠基综述 | Applied Energy 经典叙事节奏：从电网运行非对称损失切入 |
| | C2 | Yan et al. (2025) | *Applied Energy* | 现货电力市场中风电预测经济价值与调度实测 | 双结算模型（日前合约 + 实时平衡）与商业损益严格核算 |
| | C3 | Bessa et al. (2012) | *IEEE TPS* | 非对称非平稳条件下的时变分位数风电预测 | 学习电力期刊中条件分位数校准与决策风险建模表达 |
| | C4 | Zhang et al. (2014) | *IEEE TSTE* | 风电预测到电力系统调度的决策链全流程综述 | 统一专业学术术语：日前计划、平衡机制、备用短缺、弃风 |
| | C5 | Ding et al. (2023) | *Applied Energy* | 多场站时空联合概率风电预测受控实证研究 | 学习 Applied Energy 多风场对比实验的多面板图表设计 |
| | C6 | Wang et al. (2024) | *Applied Energy* | 高风电渗透下日前与实时平衡鲁棒出清机制 | 论证日前申报出力与实时平衡结算之间的互动传导机理 |
| | C7 | Chen et al. (2024) | *Renewable Energy* | 深度学习在新能源系统中的消融实验设计标准 | 学习如何通过严格的单变量开关（On/Off）杜绝模块混杂 |
| | C8 | Morales et al. (2014) | *Applied Energy 专著* | 不确定性电力市场结算与运筹机制解析 | 形式化定义日前-平衡市场结算方程，支撑 Zero-Bid 反常揭示 |

---

# 第二部分：顶刊“共性写作范式”深度提炼（The Gold-Standard Playbook）

## 1. Introduction 的“漏斗形”下沉与研究缺口（Research Gap）塑造

| 维度 | 低水平论文 / AI 生成通病 | Applied Energy / IEEE 顶刊成熟范式 |
|:---|:---|:---|
| **首段切入** | *Recently, with the rapid development of renewable energy, wind power has attracted widespread attention...* (大而空的泛背景) | **从电网运行非对称损失直接切入**：*In power grid operations and wholesale electricity markets, wind forecasting errors incur asymmetric costs—under-delivery triggers expensive spinning reserves, while over-delivery leads to curtailment. Conventional symmetric metrics (RMSE) fail to reflect this downside risk.* |
| **文献梳理** | 按作者逐个罗列（Literature dump：A did X, B did Y, C did Z） | **按解决范式归类对比**：将已有文献归纳为“代数边界截断”、“流体几何图嵌入”、“统计分布对齐”三大技术路线，并提炼其共同假定。 |
| **Gap 提出** | *However, no existing research has studied physics constraints from an audit perspective. To fill this gap, for the first time, we...* (虚假的首创性声称) | **指出方法学归因混淆与评价盲区**：*However, prevailing evaluations leave an attributional question unanswered: do reported gains stem from genuine aerodynamic domain reasoning, or merely from non-specific numerical smoothing? Existing studies confound parameter capacity with physical priors, omit negative controls, and evaluate models exclusively under uncalibrated nominal regimes.* |
| **贡献呈现** | 模糊宣称 *We propose a novel framework that improves accuracy by 10%.* | **拆解为实证发现与方法学贡献（三条明确、可检验的独立贡献）**：<br>1. Scoping 文献与开源代码审计发现盲区；<br>2. 严格受控阴性对照实验确立“双刃剑本质”与样本边界；<br>3. 揭示市场结算解耦病理（Zero-Bid）并确立 REFORMS-PCWPF 准则。 |

## 2. Methodology 的组织哲学：动机先行与数学去装饰化

1. **先讲物理/工程 Motivation，再出公式**：
   - 严禁在没有任何工程背景解释前直接堆砌数学公式。每个公式前必须说明它在电力系统物理或市场结算中对应什么现实实体；
   - 变量符号在第一次出现时完整定义，并在全文严格统一（例如：$p$ 代表日前申报出力，$y$ 代表实测出力，$q_\\tau$ 代表最优分位数，$\\lambda$ 代表物理约束惩罚权重，$\\delta$ 代表后验校准偏移量）。
2. **数学形式化服务于实证解释（No Mathiness）**：
   - 本文 §2 的引理 1（Exact Integral Excess Operational Risk）与三项二阶展开（Term I, Term II, Term III），其唯一目的就是解释实证中的两大疑团：
     - *疑团 1*：为什么未校准时加了物理约束感觉“有收益”？ $\\rightarrow$ 因为 Term III 交叉项偶然吸收了未校准的分位数偏移；
     - *疑团 2*：为什么校准后在平坦地形 Kelmarsh 反而亏损激增？ $\\rightarrow$ 因为分位数被统一校准后，物理约束引入的刚性偏差净增加了 Term I（表征误差）。

## 3. Experiments 与 Baselines 的公平对抗（Fair Benchmark）

顶刊评审专家最痛恨的是“稻草人基准（Strawman Baselines）”和“挑拣参数（Cherry-picking）”。高质量论文必须包含：
1. **朴素基准（Naive Baselines）**：Persistence（持续法）与 Ridge 回归。在极端低样本（Penmanshiel）下，简单的线性模型甚至能够击败复杂的深度模型，如实报告这一点极大提升了审稿人的信任；
2. **强现代基准（Modern Competitive Baselines）**：PatchTST（ICLR 2023 顶会最强架构）作为架构敏感性探针，证明双刃剑现象不仅存在于线性骨干（DLinear），在非线性注意力网络中依然稳健存在；
3. **严格形式匹配的阴性对照（Form-Matched Negative Control）**：
   - 这是整篇论文最硬核的科学创新点！我们设计了 **Pseudo-Scrambled Prior（伪打乱先验）**，保持网络结构、损失公式、参数量和惩罚权重 $\\lambda$ 100% 不变，仅在风速-功率映射关系上进行乱序；
   - *实验反差*：真实先验在少样本下降低 1.22% 损失，而打乱先验暴增 68.1% 损失！这无可辩驳地证明了模型确实感知到了“真实空气动力学知识”，而不是普通正则化。

## 4. Results 的叙事语法：Observation $\\rightarrow$ Evidence $\\rightarrow$ Mechanism $\\rightarrow$ Boundary

顶刊正文绝不能写成流水账（如 *Table 2 shows the results. As can be seen, our method is best*）。顶刊标准的每个实验段落必须严格遵循四步递进结构：
- **Observation (实证观察)**：明确陈述在该实验切片下观察到的核心现象；
- **Evidence (定量证据支撑)**：引用 Table/Figure 的具体数值、均值±标准差、百分比变化与 95% 置信区间；
- **Mechanism (机理归因解释)**：结合气动学特性、采样方差或决策分解理论解释“为什么会发生这种现象”；
- **Boundary (限定适用边界)**：诚实指出该结论的适用前提，不进行无根据的泛化。

## 5. Discussion 的深度与诚实：直面 Negative Results 与识别边界

1. **主动保留并强化负面结果（Negative Results as Core Contributions）**：
   - 在高信噪比平坦风场（Kelmarsh），物理约束导致单机年化结算亏损高达 **-£3,264.86/(台·年)**。我们不仅绝不隐藏，反而将其作为核心标题与发现——**“物理约束的本质是场景依赖的双刃剑防过拟合护栏，而非无脑提升精度的灵丹妙药”**。这种坦诚是征服顶刊审稿人的终极法宝；
2. **形式化归因识别边界（What Is Identified vs. What Is Not）**：
   - 显式声明：跨风场差异混杂了地形、风机型号（2.05MW vs 2.3MW）与局部微气象，本文明确不做“复杂地形因果导致物理约束必然受益”的过度归因，仅将其作为在受控样本规模下成立的“可检验假说”；
   - 46 篇文献的统计结果严格限定在 PRISMA-ScR 纳排范畴内，绝不妄称全人类文献普适。

## 6. Figure / Table 的“独立故事线（Self-Contained Storyline）”

1. **图表自明性（Self-Contained）**：审稿人即便不阅读正文，仅看图和 Caption 也能完整理解实验逻辑与结论；
2. **多面板协同（Multi-Panel Storyline）**：
   - Fig. 1：5D 约束生命周期（从定义到部署）；
   - Fig. 2：46 篇文献实证分布（四联面板，揭示 0% 负面报告与 95.7% 开源赤字）；
   - Fig. 3：三维受控消融（四联面板，(a) 阴性对照暴增, (b) 低 $\\lambda$ 平滑重合, (c) 代理延迟单调恶化, (d) 点误差与决策风险解耦）；
   - Fig. 4：场站内样本学习曲线（跨 5 个样本尺度，解耦地形与数据量）；
   - Fig. 5：PCT 官方代码反向求导缺陷与真实梯度放大；
   - Fig. 6：Zero-Bid 结算反常（揭示日前-实时解耦陷阱）。

## 7. 语言风格与“去 AI 感”军规（Anti-AI Stylistic Rules）

| 严禁词汇 / AI 浮夸套话 | 顶刊规范替换表达 |
|:---|:---|
| *groundbreaking, revolutionary, remarkable* | 彻底删除，用定量对比数字说话 |
| *It is worth noting that / It should be highlighted that* | 彻底删除，直接陈述事实 |
| *Comprehensively, novel framework, paradigm shift* | 替换为具体描述：*systematic scoping audit, controlled multi-site benchmark* |
| *significantly* (无显著性检验统计量时滥用) | 仅在配备 $p < 0.05$ 或 95% Bootstrap 置信区间时使用；无检验时用 *substantially, markedly, by 1.22%* |
| 连续堆砌 4 个抽象名词 (*optimization mechanism parameterization framework*) | 拆解为动宾或介词短语，平均句长控制在 20--28 词 |

---

# 第三部分：重构落地映射表（Reconstruction Traceability Matrix）

| 原始章节与状态 | 重构后归宿与目标章节 | 具体重构操作与增强规范 | 对应的顶刊范式与依据 |
|:---|:---|:---|:---|
| **Abstract** | **Abstract** | 彻底重写。明确点出“物理约束非万灵药，具有双刃剑本质”；提炼三风场受控实验核心定量证据（Penmanshiel -1.22% vs +68.1% 打乱；Kelmarsh +£932 罚款增收）；总结 REFORMS-PCWPF 准则。 | Applied Energy 严谨实证摘要范式 |
| **§1 Introduction** | **§1 Introduction** | 重塑为“漏斗形”下沉结构：非对称电力市场损失 $\\rightarrow$ 现有物理约束研究的盲目乐观共识 $\\rightarrow$ 归因混淆与代码赤字盲区 $\\rightarrow$ 本文实证与理论解耦裁决 $\\rightarrow$ 三项核心贡献。 | 消除大而空，建立精准文献断言 |
| **§2 Related Work** | **§2 Related Work & Positioning** | 废弃流水账，按“气动边界、尾流拓扑、分布对齐”三大范式分类梳理，并直接定位本工作与 5 篇权威综述的对比矩阵（Table 1）。 | 范式归纳取代作者罗列 |
| **§7 Theory** (原后置) | **§3 Mathematical Formulation & Market Decoupling** | **大幅前置！** 移至实证结果之前。建立非对称 Pinball 调度损失、引理 1 积分表示、二阶三项分解（表征误差、校准发散、交叉分量）、反例 1&2、以及日前-实时双结算市场模型。 | 理论与市场机理前置，为后续指标奠定数学自洽性 |
| **§3 Framework** + **§4 Protocol** | **§4 Scoping Audit Framework & Experimental Testbed** | 合并精简。完整阐述 5D 约束生命周期分类法、PRISMA-ScR 纳排协议、3 个商业风场（Kelmarsh, Hill of Towie, Penmanshiel）、DLinear/PatchTST 骨干与伪打乱阴性对照设计。 | 统一实验设计，消除冗余叙述 |
| **§4.3 Pilot** + **§6 PCT Code Audit** | **§5 Systematic Scoping & Software Fidelity Audit** | 呈现 46 篇文献全流程审计结果（0% 负面报告、95.7% 开源赤字、50% 消融混杂）与 Fleiss $\\kappa$ 统计量；深入剖析 PCT 官方计算图逆向 KL 求导缺陷与张量外积 2D 核验证。 | 宏观文献盲区与微观代码求导缺陷形成闭环 |
| **§5 Results** | **§6 Multi-Site Controlled Empirical Benchmark** | 彻底遵循 Observation-Evidence-Mechanism-Boundary 范式。分小节呈现：(1) 低样本防御护栏与打乱先验脆弱性；(2) 样本扩展学习曲线与信息预算解耦；(3) 低 $\\lambda$ 数值平滑混淆；(4) 代理气象延迟单调恶化；(5) Zero-Bid 市场解耦反常；(6) PatchTST 跨骨干架构探针。 | 彻底杜绝流水账，每段具备机理解释与置信区间 |
| **§8 Discussion** | **§7 In-Depth Discussion & Reporting Governance** | 阐明四项实证边界条件；严格界定“何者已因果识别，何者未识别”；正式发布 REFORMS-PCWPF 5 大验证门禁清单。 | 展现顶刊学者的严谨学术声誉与防反驳护城河 |
| **§9 Conclusion** | **§8 Conclusion** | 简明扼要地总结科学结论、对电网调度员与算法研发者的工程警示，以及未来研究方向。 | 紧凑有力，不引入未经证明的新主张 |
| **§11 STAR Methods** | **彻底吸收并移除** | 将关键资源表、计算环境配置、数据可用性声明无缝融入方法论与正文末尾声明，彻底清除 Cell Press 遗留体裁痕迹。 | 满足 Elsevier Applied Energy 格式规范 |
"""

target_path = r"e:\论文1\literature_benchmark.md"
with open(target_path, "w", encoding="utf-8") as f:
    f.write(content.strip() + "\n")

print(f"Successfully written {os.path.getsize(target_path)} bytes to {target_path}")
