# 章节级重构与手术规划书（Section-level Restructuring Plan）

## 1. 结构优化对照表（Before vs. After）

| 现 active_manuscript 章节 | 重构调整动作 | 重构后结构与定位 | 目标字数/篇幅控制 |
|:---|:---:|:---|:---:|
| **Frontmatter & Abstract** | `SURGICAL REWRITE` | 精炼 5 步漏斗，保留 4 个核心硬核数字，剔除冗余修饰词 | 220–250 words |
| **§1 Introduction** | `REWRITE & SHARPEN` | 6 段式黄金下沉逻辑，强化痛点，明确三大互锁贡献与四项边界 | 1,100–1,250 words |
| **§2 Related Work & Positioning** | `MERGE & TIGHTEN` | 将分类综述压缩，保留 Table 1 对标已有综述的空白矩阵，作为立论基石 | 800–900 words |
| **§3 Mathematical Formulation** | `CALIBRATE & DEFINE` | 保留引理1与三项展开，加强 Diagnostic Caveat，澄清反例1-2为存在性定理 | 1,400–1,600 words |
| **§4 Methodology & Testbed** | `STANDARDIZE` | 5D 分类法定义 + PRISMA 协议 + 3 风场数据 + DLinear/PatchTST 骨干设置 | 1,200–1,400 words |
| **§5 Literature & Software Audit** | `FORENSIC PRECISION` | 46 篇统计分布 + PCT 梯度奇异性实测 + 2D 核代数负对照 | 1,400–1,600 words |
| **§6 Controlled Experiments** | `QUESTION-DRIVEN` | 6 大实证切片，严格遵循 Obs $\to$ Evid $\to$ Mech $\to$ Bnd 叙事语法 | 2,200–2,500 words |
| **§7 Discussion & Governance** | `DEEPEN & BOUND` | 区分 5 大认知层级，明确 What is identified vs what is not，REFORMS 门禁 | 1,400–1,600 words |
| **§8 Conclusion** | `SHARP & IMPACTFUL` | 3 大发现紧凑提炼 + 3 条工程部署警示 | 250–300 words |
| **Appendix & Supp Material** | `CLEAN UP & ALIGN` | 补全反例完整证明、48 篇完整提取表、对齐 S10 code-aligned 标注 | 独立附录文档 |

---

## 2. 详细重构要点与重点手术目标

### §1 Introduction 手术要点
- 坚决杜绝“风电是关键清洁能源”这种低幼套话，第一句必须直插电网调度核心：
  *"In electric power systems and wholesale electricity markets, wind power forecasting errors incur asymmetric decision costs governed by reserve procurement and real-time balancing settlement."*
- 明确把被审计文献的核心信仰拎出来作为批驳靶子：“现有范式默认，精度的提升必然源自物理定律所赋予的有效气动归纳偏置”。
- 贡献段落严格结构化为三大互锁贡献，杜绝流水账任务清单。

### §2 Mathematical Formulation 手术要点
- 明确引理 1（Lemma 1）是对经典分位数决策理论（Koenker 2001, Pinson 2007）的严密形式化表述，避免被 Reviewer 抨击过度包装理论创新；
- 强调三项展开（Term I, Term II, Term III）是定性解释工具（Diagnostic Framework），明确声明由于真实条件概率分布未知，无法在实测时序中逐点分离，杜绝方法学漏洞；
- 给出反例 1（常数平移）与反例 2（空间异方差）的物理与直觉诠释。

### §5 & §6 Results 手术要点
- 将每个 Subsection 的标题全部重塑为“科学问题句”或“论点驱动句”，例如：
  - *旧标题*: 6.1 Multi-Site Controlled Empirical Benchmark
  - *新标题*: 6.1 Prior Specificity under Data Scarcity vs. Vulnerability to Prior Misspecification
  - *旧标题*: 6.3 Low-lambda Numerical Regularization Mimicry
  - *新标题*: 6.3 Weak Penalties ($\lambda \le 0.01$) Behave Empirically as Non-Specific Numerical Smoothing
- 严查正文与表格、Figure 的数字一致性，彻底消灭任何浮点数四舍五入引发的冲突。

### §7 Discussion 手术要点
- 建立正规的“因果识别声明（Attributional Identification Statement）”：
  - 清楚区分：什么是已经严密受控识别的（如先验特异性、弱惩罚平滑、计算图梯度放大、Zero-Bid反常）；
  - 什么是存在混杂无法因果认定的（如平原 vs 山地的地貌因果效应、46篇文献对全领域的推断边界）。
