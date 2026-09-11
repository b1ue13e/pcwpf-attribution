# 03. 顶刊学术写作与语感重构指南 (Top-Journal Writing Patterns)

基于 *Applied Energy*, *Renewable Energy*, *IEEE TSTE*, *Science*, 以及 *Cell Patterns* 的真实顶刊范式，提炼出针对本论文重构的学术写作与语感规范。

---

## 一、 顶刊核心句式与段落三段论架构

### 1. 单句信息载荷原则 (One Sentence ≈ One Main Claim)
- **控制句长**：平均控制在 **20--30 words / sentence**。
- **杜绝连环从句堆叠**：严禁连续出现三个包含 40+ 词汇的复杂复合长句。
- **清除名词堆叠 (Anti-Noun-Stacking)**：
  - ❌ *“The multi-site prospective empirical physics-informed neural network dispatch optimization performance framework...”* (5个名词连续做定语修饰)
  - ✅ *“The dispatch performance of physics-informed networks across three operational wind farms...”*

### 2. 段落组织的标准三段论架构 (Opening $\rightarrow$ Evidence $\rightarrow$ Interpretation)
- **首句 (Topic / Opening Sentence)**：开门见山，声明本段回答的研究问题或给出的核心实证事实。
- **中段 (Evidence Body)**：提供严谨的量化数据、表格交叉引用或对照实验对比。
- **尾句 (Restrained Interpretation)**：给出克制、受条件约束的科学解释，明确该证据支持什么、无法排除什么。
- **段尾禁忌**：杜绝每个段落千篇一律以 *"These findings conclusively demonstrate that..."* 结尾。

---

## 二、 Claim Calibration：四级证据陈述标准

顶刊评审专家极其反感浮夸与将局部相关性推论为普遍因果的作风。全文执行四级证据阶梯：

| 证据层级 | 允许使用的动词与句式 | 适用场景 | 严厉禁止的表述 |
|---|---|---|---|
| **Level 1: 直接观测** (Direct Observation) | *We find...*<br>*We observe...*<br>*In the audited cohort...*<br>*Across three operational sites...* | 实验测量数据、文献统计计数、代码核查现象 | 严禁使用 *proves*, *establishes*, *demonstrates causation* |
| **Level 2: 获得支持的推论** (Supported Interpretation) | *The results suggest...*<br>*This pattern is consistent with...*<br>*Under the tested configuration...*<br>*Provides one plausible explanation...* | 机制解释、学习曲线趋势推断、消融对比分析 | 严禁使用 *definitively confirms*, *fundamentally reveals* |
| **Level 3: 因果识别** (Causal Identification) | *isolates the effect of...*<br>*holding backbone capacity constant...*<br>*under matched gradient scales...* | 仅在严格满足双盲负对照、同骨干、同超参的实验（如 Pseudo-Scrambled 对照） | 只有控制了所有混杂因子才可使用因果词 |
| **Level 4: 全称/普适性判断** (Universal Claim) | *原则上极少使用*（仅限经典数学公理/定理） | 经典决策论引理（如 Koenker 分位数最优性） | **严禁在实证部分使用**（3座风场 $\neq$ 全球风电工业界；46篇语料 $\neq$ 全体科学文献） |

---

## 三、 去除 AI 模板味与浮夸用词替换清单

### 1. 高频副词与形容词降调表
| 原文高频浮夸词汇 | 顶刊规范替换表达 | 降调理由与学术效果 |
|---|---|---|
| `fundamentally` / `strictly` | *consistently* / *systematically* / *primarily* | 减少绝对化断言，增强经验科学的严谨性 |
| `unambiguously` / `definitively` | *indicates that* / *is consistent with* | 避免过早下不可推翻的论断 |
| `severely` / `dramatically` / `strikingly` | *substantially* / *by 68.1%* / *notably* | 用精确数字与客观幅度代替情绪化修饰词 |
| `forensic` / `pathological` | *controlled* / *systematic* / *diagnostic* | 消除刑事侦查/病理学词汇，回归工程与计量审计语感 |
| `catastrophic` surge | *marked increase* / *loss inflation of +68.1%* | 准确报告百分比，拒绝新闻通稿式修辞 |
| `proves` / `confirms` | *shows* / *demonstrates under tested settings* | 经验观察不能替代数学证明 |

### 2. 清理内部工程流水线与会务汇报语言
- ❌ 抹除所有类似 `Schema v4`, `Task D`, `Boundary Contracts`, `10-GPU Cluster Topology Dispatched Nodes`, `48-study Taskbook` 等内部项目管理术语；
- ✅ 统一使用正式出版学术术语：`PRISMA-ScR scoping protocol`, `controlled 3D experimental bench`, `multi-site operational telemetry`, `multi-seed initialization`。

---

## 四、 制度准确性与经济学术语规范

在电力市场与运行调度评估部分，必须严格遵循英国能源监管机构（Ofgem）与 Elexon BSC 的官方制度定义：

1. **市场结构定性**：
   - 严禁将 2017 年大不列颠（GB）电力市场描述为强制性的“日前面向集中出清 + 实时结算的二阶段市场”（即严禁与 PJM/ERCOT 等强制 ISO 集中池模式混淆）。
   - 必须准确描述为：**基于自愿性远期/双边场外交易与电力交易所（Bilateral forward contracting & voluntary power exchanges），配合由 Elexon 运营的平衡与结算规约（BSC Imbalance Settlement）**。

2. **两阶段收益公式定性**：
   $$R_{\text{total}} = P_{\text{forward}} \cdot q + P_{\text{imb}} \cdot (y - q)$$
   明确界定为：**“用于解释申报风险与失配敞口的程式化两阶段收益会计解构（Stylized two-stage accounting decomposition）”**，并在正文中采用 $P_{\text{F}}$ 或 $P_{\text{forward}}$ 取代易引起制度误解的 $P_{\text{DA}}$。

3. **零申报（Zero-Bid）诊断定性**：
   - 严禁将零申报产生的 £223k--£243k/(turb$\cdot$yr) 现金流描述为“真实可行的商业交易策略或盈利能力（Commercial trading profit）”。
   - 必须定性为：**“单结算不平衡现金流评价的归谬诊断（Reductio ad absurdum diagnostic）”**，证明脱离远期合约对冲的孤立不平衡结算现金流会由于不计前向收益损失和违约责任而严重失真，绝不能作为衡量预测模型优劣的独立代理指标。
   - 全文严格执行：**平衡现金流分量（Imbalance cashflow component） $\neq$ 商业价值（Commercial value） $\neq$ 交易利润（Trading profit）**。

---

## 五、 表格与图表视觉质感标准

1. **Table 规范**：
   - 全面遵循三线表（Booktabs），严格禁止任何垂直竖线；
   - 彻底解决两栏排版（twocolumn）下的溢出、文字挤压与截断；
   - 标题明确区分“主消融误差（Forecasting Accuracy）”与“经济结算诊断（Settlement Diagnostic）”；
   - 表头清晰标注量纲（如 `\pounds/(turb\cdot yr)`, `pu`, `MWh`），所有货币符号统一为标准 `\pounds` 且无缺失。

2. **Figure Caption 规范**：
   - **图注功能回归**：图注仅用于说明图表包含的曲线、定义关键变量符号、指明误差棒定义（如 standard deviation across random seeds 或 95% bootstrap confidence interval）、标注关键实验条件；
   - **杜绝图注承担正文 Discussion**：严禁在图注中铺陈半页篇幅的机制总结与文献大反思。所有推论与讨论均回归正文 Results 与 Discussion 章节。
