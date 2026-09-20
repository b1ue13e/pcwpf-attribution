# 终审 Reviewer 模拟：Physics-Constrained Forecasting Observability Audit

- 日期：2026-09-04
- 审查对象：当前审计论文研究包，而非旧 FG-MoE 主稿
- 材料：Observability Audit Protocol v1、26篇编码表、PCT 官方代码核查、三版本实验、prior-source/shift 实验、noise-regime 与架构一致性实验
- 目标期刊标尺：Applied Energy / Renewable Energy 级能源预测期刊
- 审查性质：read-only；未修改论文与实验代码

## 0. Editorial Decision

**Decision: Reject / No-Go in the current form.**

当前研究包已经形成一个有潜力的“constraint audit”方向，但冻结标题和核心命题没有被现有证据支持。最严重的问题是把训练期历史先验误判为部署期不可观测约束。PCT 的 target wind-power distribution 在训练前由历史训练数据构造，只用于训练 loss；推理时不需要未来真值。因此，`oracle-train` 不是 deployment oracle，而是合法的 `historical static training prior`。当前 26 篇编码也全部给出 `oracle_leakage=FALSE`，直接否定了“领域方法假设了部署时不可观测约束”这一标题级主张。

此外，oracle-to-deployment 替换实验存在实现性 P0：所谓 lagged power 的移动方向反了，索引 `lag_pw[:-lag] = Y_tr[lag:]` 让较早位置使用未来样本；所谓 NWP 条件也只是最后观测风速经过手工曲线，并非真实或独立训练的 NWP forecast。PCT 三版本实验没有实现论文声明的 dynamic JS switch，且使用自建数据、简化单步模型、小 batch 与不同训练协议，不能称 specification-faithful reproduction。

现阶段可保留的创新核是：

> 对 physics-constrained forecasting 的 constraint provenance、application lifecycle、implementation fidelity、prior misspecification 和 harm reporting 做系统审计。

但需要重构标题、taxonomy 和实验后才能进入正式写稿。

---

## 1. Reviewer Configuration

| 角色 | 独立关注点 | 初步决定 |
|---|---|---|
| EIC：能源预测与期刊适配 | 贡献是否构成领域级审计 | Reject；问题有价值但主标题失真 |
| Reviewer 1：系统综述方法 | PRISMA、双编码、一致性与抽样 | Major methodological failure |
| Reviewer 2：ML复现与软件审计 | paper/code parity、实验公平性 | Reject；当前不能称忠实复现 |
| Reviewer 3：风电与部署 | 信息在训练/推理生命周期中的可用性 | Reject；training prior 与 deployment signal 混淆 |
| Devil's Advocate | 最强替代解释与过度归因 | Critical：论文审计了一个原方法并未提出的问题 |

---

## 2. Claim–Evidence Matrix

| ID | 当前主张 | 证据 | 关键问题 | 裁决 |
|---|---|---|---|---|
| C1 | L1–L5 taxonomy 是新审计框架 | Protocol v1 | 未区分 `application_stage` 与 `required_at_inference`；training-only prior 被错误纳入部署可观测性问题 | 需重构 |
| C2 | 领域方法假设 deployment-unobservable constraints | 26篇编码 | 26/26 `oracle_leakage=FALSE`；PCT 明确预计算训练 target | 被当前证据反驳 |
| C3 | 文献普遍不报告 constraint harm | 25/26 FALSE | 语料尚非完整PRISMA；只有8个新增条目在合并表中保留 coder_B 七字段 | pilot 支持，不能领域化 |
| C4 | 文献没有 oracle→causal substitution test | 26/26 FALSE | 多数方法根本不在 inference 使用该 constraint，未做 substitution 不一定是缺陷 | 需限定适用对象 |
| C5 | PCT paper specification 与官方代码不一致 | 静态代码、loss/gradient parity | 静态数学不一致基本成立；但 downstream 实验省略 dynamic switch，不能归因完整官方方法 | 部分支持 |
| C6 | 官方实现导致 RMSE 差3–8倍 | 5站×2 seed | 自建数据/模型/协议；未匹配 dynamic switch、batch、horizon、原始数据和超参；loss scale 未公平调谐 | 不可作为PCT性能结论 |
| C7 | corrected-parity 等于 paper-spec | 单站、固定RNG | 两者由同一团队按相同函数定义，逐位一致更多是实现自洽测试，不是外部parity | 支持代码自洽，不支持论文复现 |
| C8 | oracle→deployment 后收益消失 | 3站×2 seed | `oracle-train`不是oracle；lagged分支未来泄漏；NWP分支不是NWP | INVALID |
| C9 | constraints 在20/20任务非劣于unconstrained | 多条件RMSE | 条件/seed不是独立研究重复；仅2 seed、无CI/检验；实验命名与原方法生命周期错位 | 描述性线索 |
| C10 | W1漂移解释约束负收益 | 5站Wasserstein | n=5、无预注册模型或显著性；方向并非逐站严格对应；只能称机制线索 | 不可称机制闭环 |
| C11 | 架构混淆已排除 | Kelmarsh单站×1 seed×3架构 | 只排除一个站点/seed下的明显反例；不能证明架构无关 | 有用但不足 |
| C12 | 当前已达到一区创新门槛 | taxonomy+编码+PCT案例 | 核心observability claim失效，只剩单家族软件/shift案例 | 不支持 |

---

## 3. P0 Issues

### P0-1：训练期先验与部署期约束混淆

PCT论文说明 target wind-power distribution 在训练前由历史数据构建，随后作为 MSE+JS 的训练正则。训练结束后，该 target 已编译进模型权重，推理不需要重新获得未来真实风速/功率。因而：

- `train-time historical distribution` 是训练可用信息；
- 它不等于 `deployment-unobservable signal`；
- 只有 inference/post-processing 阶段实时依赖 future truth 才构成 deployment oracle leakage。

当前 taxonomy 必须增加：

1. `constraint_application_stage = training/calibration/inference/postprocess`；
2. `required_at_inference = yes/no`；
3. `prior_fit_data = train-only/train+test/external/unknown`；
4. `frozen_after_training = yes/no`。

在重编码前，标题级 observability 结论冻结。

### P0-2：论文自己的语料结果否定标题

`literature_coding_table_dual.csv` 中26/26 `oracle_leakage=FALSE`。这不是“缺少证据”，而是当前编码没有发现被标题指控的现象。可支持的发现是 `harm_reported` 25/26 FALSE，而不是“assumes constraints it may not observe”。

### P0-3：lagged替换使用未来值

`pct_oracle_substitution.py` 使用：

```python
lag_pw[:-lag] = Y_tr[lag:, 0]
```

对位置 `i`，右侧是 `i+lag` 的未来值。真正的过去滞后应将旧值写到后部，如 `lag_pw[lag:] = Y_tr[:-lag,0]`，并明确处理前 `lag` 个边界样本。当前 `deploy-lagged` 结果全部失效。

### P0-4：NWP替换并非NWP

代码以 `X_tr[:, -1, wind]` 作为 `wf_real`，即最后观测风速，再通过手工 linear/bin-mean curve 映射。它既不是外部NWP，也不是独立风速预测器。应改名为 `last-observation curve prior`，不得称 deployment-NWP substitution。

### P0-5：PCT复现遗漏dynamic switch

PCT论文明确描述：预测分布无重叠或均值/方差不满足条件时，JS ratio设为0，否则为1。当前 `pct_fulltrain_v3_runner.py` 每个batch恒定执行：

```python
loss = mse + js
```

因此 official-code arm 不是完整官方训练逻辑。没有 dynamic switch parity，就不能将3–8倍RMSE差异归因于官方PCT实现。

### P0-6：loss-scale与超参公平性未关闭

reversed KL使JS数值放大6–11.5倍，同时固定系数为1。corrected arm的改善可能部分来自约束权重显著变小，而不仅是“方向修正”。需要至少比较：

- 原始系数；
- gradient-norm matched系数；
- validation-tuned系数；
- 带官方dynamic switch的版本。

### P0-7：系统编码尚未完成

当前合并表26篇，但：

- 只有8行在同一合并表中保留完整 `coderB_*` 七字段；
- pilot 的 `observability_level` κ=0.10，说明核心分类最初不稳定；
- 24/26 `code_available=UNKNOWN`；
- enforcement取值出现 `soft (chance constraint)`、`hard (boundary) + soft (gen)` 等超出冻结schema的值；
- PRISMA日志缺数据库总命中数、去重数、title/abstract/full-text排除数及逐篇排除理由。

不能将该语料称为已完成的系统审计。

### P0-8：当前并非完整论文

活跃材料仍是协议、报告和实验脚本，没有围绕新主线形成统一 manuscript。不同报告还保留“oracle”“部署不可得”“机制闭合”等已失效语言。终审对象实为研究包，不能进入投稿终审。

---

## 4. Independent Reviews

### EIC Review

研究方向有潜在价值：physics-constrained forecasting 需要比“是否提高RMSE”更严格的约束来源、实现和伤害审计。作者在发现错误后多次主动撤回主张，也是优点。然而，当前标题将所有训练期物理先验都要求为部署时实时可观测，这是错误的生命周期标准。文章自己的编码没有发现oracle leakage，反而显示缺少harm reporting。建议拒绝当前定位，允许以“constraint lifecycle and harm audit”完全重构后重投。

评分：Significance 6/10；Novelty 5/10；Readiness 2/10；Decision Reject。

### Reviewer 1：Systematic Review Methodology

Protocol字段设计有潜力，但语料规模和流程尚不足。两轮检索更接近定向snowballing，而非可复现PRISMA；没有数据库级flow counts与完整排除记录。双人编码只对部分条目落表，核心level字段pilot κ极低。26篇可以作为scoping pilot，不能支持领域比例。建议先修订codebook生命周期字段，对全部语料重新独立双编码，再扩大数据库检索。

评分：Review rigor 2/5；Transparency 3/5；Decision Major/Reject。

### Reviewer 2：Reproducibility and Software Audit

paper/code数学不一致和KL方向核查是研究包中最扎实的部分；loss、gradient和下游训练均进行了多层验证。但“corrected parity”等于“paper spec”是同一实现链内部的一致性，不是与作者数据/输出的外部parity。当前runner遗漏论文dynamic switch，并使用不同数据、batch、horizon与模型。结论必须限制为“在移植测试中观察到的implementation sensitivity”，不能写成PCT不可复现或官方代码性能差3–8倍。

评分：Static audit 4/5；Reproduction fidelity 2/5；Decision Reject pending faithful rerun。

### Reviewer 3：Wind Forecasting and Deployment

历史训练分布是合法offline prior，不需要在现场每次预测时重建。部署可观测性应只约束推理时真正使用的输入。对train-time prior更合理的问题是：来源是否匹配目标站点、是否随时间漂移、是否会导致negative transfer。当前结果实际上更支持 prior misspecification / distribution shift audit。这个重定位与风电实践更一致。

评分：Domain motivation 4/5；Current construct validity 1/5；Decision Reject/Reframe。

### Devil's Advocate

最强反论点是：作者并未发现physics-constrained forecasting使用不可观测约束，而是把完全合法的训练集分布先验重新命名为oracle，然后用一个包含未来泄漏的“lagged”分支和一个并非NWP的last-wind映射与之比较。随后，作者省略原方法的dynamic switch，以固定1:1权重比较数值尺度不同的JS实现，并把迁移结果解释为官方代码缺陷的下游后果。即使所有数字均可复现，它们也没有回答标题中的问题。

**CRITICAL**：现标题和Contribution 4必须撤回。按照Reviewer Skill铁律，存在该CRITICAL问题，Editorial Decision不能是Accept。

---

## 5. What Can Be Saved

可以保留并发展：

1. `constraint provenance + lifecycle` taxonomy；
2. 论文公式与官方代码的implementation fidelity审计；
3. 约束先验在站点/时间分布漂移下的negative-transfer stress test；
4. `harm_reported`与constraint-induced harm的系统编码；
5. RNG alignment、paper/code/corrected三版本实验框架。

必须撤回或改名：

1. `oracle-train` → `historical-static target`；
2. `deploy-lagged` 当前结果 → INVALID；
3. `deploy-nwp-*` → `last-observation curve prior`；
4. `oracle-to-deployment substitution` → `prior-source and shift stress test`；
5. “架构无关” → “在Kelmarsh seed-0三种架构中方向一致”；
6. “20/20显著负收益” → “在所列20个相关实验格中方向非正”；
7. “机制闭合” → “Wasserstein shift提供机制线索”。

---

## 6. Recommended New Positioning

建议标题：

> **From Specification to Stress Test: A Systematic Audit of Constraint Provenance, Implementation Fidelity, and Harm in Physics-Constrained Wind Power Forecasting**

核心问题：

> 物理约束来自何种数据、在模型生命周期的哪一阶段生效、发布实现是否忠于论文规格，以及在站点与时间分布漂移下是否产生可测的negative transfer？

这条主线不再预设“约束不可观测”，而是让系统编码和可执行审计决定问题出现在哪个生命周期阶段。

---

## 7. Revision Roadmap

| Gate | 必做项 | 通过标准 |
|---|---|---|
| G0 | 冻结旧标题与observability结论 | 不再出现领域级deployment-unobservable指控 |
| G1 | Protocol v2 | 增加application_stage、required_at_inference、prior_fit_data、frozen_after_training |
| G2 | 全量重新编码 | 全部纳入论文双盲式独立编码；每字段κ/α；冲突逐条仲裁 |
| G3 | 完整PRISMA | 数据库、query、时间、命中、去重、筛选、排除理由完整可复现 |
| G4 | 修复替换实验 | 删除未来泄漏；使用真实风速预测或诚实改名；固定train-only prior |
| G5 | faithful PCT runner | dynamic switch、架构、batch、horizon、标准化、超参全部逐项对齐 |
| G6 | 公平loss比较 | raw、official、corrected、gradient-matched、tuned-lambda均报告 |
| G7 | 统计闭环 | ≥5独立训练seed；paired CI；预注册主contrast；不把格子数当样本量 |
| G8 | 第二独立方法家族 | pointwise或uncertainty-aware，需全文公式/官方代码或可审计parity |
| G9 | 作者核实 | 对官方实现差异先做版本/commit冻结并邀请原作者澄清 |
| G10 | 新稿重审 | 完整manuscript形成后重新full review |

---

## 8. Final Scores

| 维度 | 分数（10） |
|---|---:|
| 问题重要性 | 7 |
| 当前主线创新性 | 5 |
| 潜在重构创新性 | 8 |
| 文献审计严谨性 | 3 |
| 复现忠实度 | 3 |
| 实验统计严谨性 | 3 |
| Claim boundary | 4 |
| 投稿准备度 | 2 |
| **综合** | **3.8/10** |

## 9. One-Sentence Verdict

**当前研究包不能证明physics-constrained wind forecasting依赖部署时不可观测约束；它更有希望证明的是，约束的来源、生命周期、实现忠实度与分布漂移共同决定其收益和伤害，但该新命题仍需Protocol v2、忠实PCT runner、完整系统编码和第二方法家族才能达到一区审计论文门槛。**
