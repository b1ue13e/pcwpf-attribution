# -*- coding: utf-8 -*-
import os

content = """# 论文全面学术重构执行蓝图（Reconstruction Plan）
## 面向 *Applied Energy* 顶级期刊录用标准的模块化重构施工路线图

> **重构最高指导原则**：
> 1. **结构合法性（Structural Legitimacy）**：坚决废除 Cell Press 专属的 STAR Methods，将数学推导与市场模型大幅前置，重塑为 Applied Energy 标志性的标准 IMRaD 八股结构；
> 2. **主线唯一性（Unified Spine）**：全篇紧密围绕“物理约束的双刃剑本质与决策解耦”展开，将文献审计与代码审计定位为解释“学术界盲目乐观历史”的实证基石；
> 3. **证据绝对求真（Single Source of Truth）**：严禁修改或虚构任何实验数据，所有定量指标 100% 绑定 `constraint_round2` 原始账本；
> 4. **语言去 AI 质感（Academic Prose Reconstruction）**：彻底剔除套话，严格执行 Observation $\\rightarrow$ Evidence $\\rightarrow$ Mechanism $\\rightarrow$ Boundary 叙事语法。

---

# 一、章节演变全景拓扑图（Structural Topology）

```mermaid
graph TD
    subgraph Old [原稿件旧结构 (11 Sections, 结构倒置且混杂)]
        O1[§1 Introduction]
        O2[§2 Related Work]
        O3[§3 Taxonomic Framework]
        O4[§4 Scoping Protocol & Testbed]
        O5[§5 Empirical Results]
        O6[§6 Software Audit: PCT]
        O7[§7 Prediction-Decision Theory]
        O8[§8 Discussion & Governance]
        O9[§9 Conclusion]
        O11[§11 STAR Methods 违规体裁]
        O10[§10 Appendix]
    end

    subgraph New [重构后法定新结构 (Applied Energy 经典 IMRaD 八股)]
        N1[§1 Introduction: 痛点切入 + 双刃剑主线 + 贡献]
        N2[§2 Mathematical Formulation & Market Decoupling: 调度损失 + 引理1 + 三项分解 + 反例 + 双结算]
        N3[§3 Empirical Audit Framework & Methodology: 5D分类法 + PRISMA协议 + 3风场数据 + 骨干与阴性对照]
        N4[§4 Systematic Literature & Software Fidelity Audit: 46篇文献盲区 + PCT求导反向缺陷与核验证]
        N5[§5 Multi-Site Controlled Empirical Evaluation: 6大实验切片 + 信息预算对齐 + Zero-Bid诊断 + PatchTST探针]
        N6[§6 Discussion, Identification Boundaries & Governance: 4大边界 + 因果识别厘清 + REFORMS清单]
        N7[§7 Conclusion: 紧凑结论 + 工程警示]
        N8[Appendix: 反例完整数学证明 + 20篇共识表 + 补充图表]
    end

    O1 -->|REWRITE & SHARPEN| N1
    O7 -->|MOVE FORWARD & EXPAND| N2
    O3 -->|MERGE & COMPACT| N3
    O4 -->|MERGE| N3
    O6 -->|MOVE & INTEGRATE| N4
    O4 -->|Extract Scoping Stats| N4
    O5 -->|RESTRUCTURE (Obs-Evid-Mech-Bnd)| N5
    O8 -->|REFINE & STRENGTHEN| N6
    O9 -->|COMPACT| N7
    O11 -->|DELETE & ABSORB into N3/Decl| N3
    O10 -->|PRESERVE & ENHANCE| N8
```

---

# 二、全量模块手术清单（Section-by-Section Surgical Operations）

### 1. Abstract
- **Operation**: `REWRITE`
- **重构目标**:
  - 开篇直接指明风电预测在电网非对称调度中的核心痛点与物理约束的“双刃剑本质”；
  - 精炼呈现三风场受控实证数据（Penmanshiel 真实先验 -1.22% vs 打乱先验 +68.1%；Kelmarsh 真实结算单机年化短缺激增 +£932.56 与现金流亏损 -£3,264.86）；
  - 概括 46 篇 Scoping 审计（0% 负面报告，95.7% 开源赤字，50% 消融混杂）与 PCT 代码参数倒置缺陷；
  - 阐明 Zero-Bid 反常与 REFORMS-PCWPF 准则。
- **字数控制**: 240--250 词，零浮夸词汇。

---

### 2. §1 Introduction
- **Operation**: `REWRITE`
- **重构目标**:
  - 第一段：从新型电力系统高比例风电并网的非对称备用调度与实时平衡惩罚切入，指出传统对称指标（RMSE/MAE）的失真；
  - 第二段：梳理嵌入气动与功率曲线先验的主流研究，点出核心认知假定（Premise）；
  - 第三段：指出两大方法学盲区与归因混淆（缺乏阴性对照无法区隔物理偏置与数值平滑；缺乏开源代码与真实调度校准）；
  - 第四段：引出本文对标研究，明确提出“双刃剑本质”的核心科学发现；
  - 第五段（Summary of Contributions）：提炼三大互锁贡献（文献与开源代码审计、多风场严格受控消融实验、预测-决策解耦理论与治理准则）；
  - 第六段：明确交代四项方法学边界（Scope & Boundaries）。

---

### 3. §2 Mathematical Formulation and Decision-Theoretic Foundation (新章节：前置原 §7)
- **Operation**: `MOVE & EXPAND` (原 §7 前置)
- **重构目标**:
  - 2.1 节：日前风电申报出力的运筹调度问题形式化，定义非对称 Pinball 损失函数 $\mathcal{L}_{\text{disp}}(p, Y)$ 与最优分位数 $q_\tau = F^{-1}_{Y|X}(\tau)$，$\tau = 1/(r+1)$；
  - 2.2 节：引理 1（Exact Integral Excess Operational Risk）及其完整积分非负性证明；
  - 2.3 节：二阶泰勒展开与三项误差分解（Term I 表征误差、Term II 校准发散、Term III 交叉项），从数学机理解释“未校准如何掩盖伪收益”与“校准后为何暴露表征劣势”，加入 Diagnostic Caveat；
  - 2.4 节：收录常数平移反例与异方差反例，证伪绝对占优性；
  - 2.5 节：建立真实日前-实时双结算市场模型（Two-Settlement Wholesale Accounting）与英国 Elexon BSC 30分钟净额算子，为后文结算实证奠定法理基础。

---

### 4. §3 Empirical Audit Framework and Methodology (合并原 §3 与 §4 前半部分)
- **Operation**: `MERGE & REORGANIZE`
- **重构目标**:
  - 3.1 节：5D 约束全生命周期分类法（$\mathcal{D}_1$ 参数来源、$\mathcal{D}_2$ 应用阶段、$\mathcal{D}_3$ 推理遥测、$\mathcal{D}_4$ 代码保真度、$\mathcal{D}_5$ 消融语义与负面报告）；
  - 3.2 节：PRISMA-ScR 文献检索与三方独立盲审纳排协议（$89 \rightarrow 72 \rightarrow 53 \rightarrow 48 \rightarrow 46$）；
  - 3.3 节：三个商业风电场实测 SCADA 数据集（Kelmarsh, Hill of Towie, Penmanshiel）与 Elexon 2017 全年动态结算电价特征；
  - 3.4 节：预测模型套件（DLinear、PatchTST、Ridge、Persistence）与训练/验证/校准/测试时间分区隔离协议；
  - 3.5 节：受控消融设计（$\lambda$ 敏感性网格、Pseudo-Scrambled 伪打乱阴性对照设计、严格 500 样本信息预算对齐方案）；
  - 3.6 节：静态代码与计算图审计流程（AST 解析与符号推导规范）。

---

### 5. §4 Systematic Scoping and Software Fidelity Audit (整合文献审计与原 §6 PCT 代码审计)
- **Operation**: `MERGE & DEEPEN`
- **重构目标**:
  - 4.1 节：46 篇 Scoping 文献全流程统计结果（0.0% 负面报告、95.7% 开源赤字、50.0% 模块消融混杂），展示 Table 1 与 Fig. 2；
  - 4.2 节：三方独立盲审信度分析，报告 Fleiss $\kappa$，科学阐释 $\kappa=0.341$ 的患病率悖论；
  - 4.3 节：PCT 代码保真度深度取证，展示 PyTorch C++ ATen 内核反向调用，推导 $\mathcal{O}(1/\epsilon)$ 逆密度梯度奇异点，呈现梯度范数放大 2.20--2.72 倍的实测证据（Fig. 5）；
  - 4.4 节：张量外积 2D 核密度估计代数等价性证明，作为肯定性阴性对照，确立客观中立的学术声誉。

---

### 6. §5 Multi-Site Controlled Empirical Benchmark (重塑原 §5)
- **Operation**: `RESTRUCTURE (Observation-Evidence-Mechanism-Boundary)`
- **重构目标**:
  - 5.1 节：低样本防御护栏与先验打乱脆弱性（Penmanshiel: 真实先验 -1.22% vs 打乱先验 +68.1%）；
  - 5.2 节：场站内样本学习曲线与信息预算严格对齐（前置 Table 3，证明 $N_{\text{prior}}=500$ 下依然保留 77%--92% 收益，化解数据泄漏质疑；展示样本量扩至 5,000 时先验收益消失之实证）；
  - 5.3 节：低 $\lambda=0.01$ 数值平滑混淆验证（真实先验与打乱先验无统计差异）；
  - 5.4 节：离线训练气动代理延迟对损失的单调恶化（$k=6 \rightarrow 36$ 步）；
  - 5.5 节：真实英国 BSC 30分钟净额结算分化与 Zero-Bid 结算反常诊断（Table 2, Table 4, Table 5, Fig. 6）；
  - 5.6 节：PatchTST 现代注意力跨骨干架构敏感性探针（Table 6）。

---

### 7. §6 In-Depth Discussion, Identification Boundaries, and Reporting Governance (重构原 §8)
- **Operation**: `REFINE & STRENGTHEN`
- **重构目标**:
  - 6.1 节：四大实证边界条件系统综合（样本容量饱和、代理对齐、结算诊断包络、代码保真度）；
  - 6.2 节：因果归因识别边界（What Is Identified vs What Is Not），显式承认跨风场混杂，恪守科学谨慎性；
  - 6.3 节：审计中证伪的两大初始猜想（张量外积独立性假设证伪、现金流反转物理归因证伪）；
  - 6.4 节：面向电力能源 AI 领域的 **REFORMS-PCWPF 五大门禁报告准则**（Table 7）。

---

### 8. §7 Conclusion (重构原 §9)
- **Operation**: `COMPACT & FOCUS`
- **重构目标**:
  - 紧凑概括文献盲区、代码实现分歧与多风场受控实证三大发现；
  - 给电网运营商与算法研发者的明确工程启示；
  - 展望未来开放挑战（在线自适应先验、多模态时空气象耦合）。

---

### 9. §11 STAR Methods 模块
- **Operation**: `DELETE & ASSIMILATE`
- **处置原则**:
  - 彻底删除 `11_starmethods.tex`；
  - 将硬件规格、开源数据 DOI、Zenodo 链接、代码仓库地址无缝吸收进 §3 方法论与文末 `Data and Code Availability` 声明。

---

### 10. §10 Appendix
- **Operation**: `PRESERVE & POLISH`
- **处置原则**:
  - 保留反例 1（常数平移）与反例 2（空间异方差）的严密代数推导；
  - 附带 20 篇文献三方盲审与最终仲裁共识表（Table S1）。

---

# 三、图表全量布局与状态对齐表

| 编号 | 图表标题简述 | 在新正文中的章节归宿 | 数据来源与状态 |
|:---:|:---|:---:|:---|
| **Fig. 1** | 5D 约束生命周期架构图 (fig_constraint_lifecycle.pdf) | §3.1 | 矢量高精图，保留 |
| **Fig. 2** | 46 篇 Scoping 文献生命周期四联分布图 (fig_field_audit_distribution.pdf) | §4.1 | 矢量高精图，保留 |
| **Fig. 3** | 三维受控消融四联面板 (fig_causal_ablation.pdf) | §5.1 | 矢量高精图，保留 |
| **Fig. 4** | 场站内跨 5 样本尺度学习曲线 (fig_within_site_learning_curve.pdf) | §5.2 | 矢量高精图，保留 |
| **Fig. 5** | PCT 官方代码反向求导缺陷与真实梯度放大 (fig_pct_loss_empirical_contrast.pdf) | §4.3 | 矢量高精图，移至 §4 |
| **Fig. 6** | Zero-Bid 结算反常与指标脱钩诊断 (fig_market_decoupling.pdf) | §5.5 | 矢量高精图，移至 §5 |
| **Tab. 1** | 46 篇 Scoping 文献 D1--D5 分布与综述定位对比表 | §4.1 | 动态导出，保留 |
| **Tab. 2** | 三风场受控消融基准测试主表 (Normalized Decision Cost) | §5.1 | 原始 CSV 导出，保留 |
| **Tab. 3** | 严格 500 样本信息预算对齐消融表 (Table Information Budget) | §5.2 | **提升为主干表**，保留 |
| **Tab. 4** | 真实英国 BSC 30分钟结算与 2000 次块 Bootstrap 置信区间表 | §5.5 | 真实 Elexon 结算，保留 |
| **Tab. 5** | Zero-Bid 结算诊断表 (The Zero-Bid Settlement Anomaly) | §5.5 | 归谬法诊断，保留 |
| **Tab. 6** | PatchTST 跨骨干架构敏感性探针表 | §5.6 | 60 组实验导出，保留 |
| **Tab. 7** | REFORMS-PCWPF 五大门禁检查清单表 | §6.4 | 方法学报告准则，保留 |

---
*重构执行蓝图生成完毕，已固化为 `e:\论文1\reconstruction_plan.md`。*
"""

target_path = r"e:\论文1\reconstruction_plan.md"
with open(target_path, "w", encoding="utf-8") as f:
    f.write(content.strip() + "\n")

print(f"Successfully written {os.path.getsize(target_path)} bytes to {target_path}")
