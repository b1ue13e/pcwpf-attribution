# 新主线冻结（steer #2，2026-09-03）

## 标题（冻结）

> **From Specification to Deployment: A Systematic Audit of Constraint Provenance,
> Implementation Fidelity, and Harm in Physics-Constrained Wind Power Forecasting**

## 一句话定位

物理约束风电功率预测方法把"约束有益"当作默认前提，但领域的系统性盲点是：
**约束是 learned prior，其收益取决于来源（provenance）、实现忠实度（fidelity）、
分布稳定性（distribution stability），而这些维度几乎从未被报告或审计。**

## 三条贡献 + 我们的证据映射

### C1. Constraint Lifecycle Taxonomy（5 维）
文件：`ConstraintLifecycle_Taxonomy_schema_v3.md`

| 维度 | 含义 | 状态 |
|---|---|---|
| D1 Provenance | prior 用什么数据估计 | schema 冻结 ✅ + coder_A 首编 ✅ |
| D2 Application stage | 约束在哪个阶段作用 | schema 冻结 ✅ + coder_A 首编 ✅ |
| D3 Inference requirement | 推理期需要什么新信息 | schema 冻结 ✅ + coder_A 首编 ✅ |
| D4 Implementation fidelity | 实现是否忠于规格 | schema 冻结 ✅ + coder_A 首编 ✅ |
| D5 Harm reporting | 是否报告约束伤害 | schema 冻结 ✅ + coder_A 首编 ✅ |

**首次编码（coder_A，26 篇摘要级）结果**，见 `ConstraintLifecycle_Taxonomy_schema_v3.md`：
- D4 `unverified` 21/26、D5 `none-reported` 25/26 —— 正面支持 gap。
- 7 篇弱编码待全文核对、coder_B 独立编码待做。

**击穿防护（核心 distinction）**：训练期合法历史先验（D1=historical-static）≠
推理期不可观测约束（D3）。这是旧主线被 PCT 方法部分击穿的原因，v3 从字段上排除。

### C2. Systematic Field Audit（26 篇，2023–2026）
文件：`literature_coding_table_dual.csv` + `双人编码与仲裁记录_pilot.md`

已做：
- 26 篇编码（2023×1 / 2024×1 / 2025×13 / 2026×11），PRISMA 两轮检索日志。
- 双人编码 observability_level 三批（9+8+9）独立编码一致。

**关键正面结果（支持新 gap）**：
- `harm_reported = FALSE` 25/26 —— **约束潜在伤害系统性缺少审计（这是真正 gap）**
- `substitution_test = FALSE` 26/26 —— 无一篇做过 prior 替换测试
- `code_available = UNKNOWN` 24/26 —— 实现忠实度普遍无法核验

**已证伪、不再用作主张**：
- `oracle_leakage = FALSE` 26/26 —— 这是**反证**：语料没有证明 deployment-unobservable
  oracle，旧"observability 缺口"主张不成立，已撤回。

### C3. Executable PCT-family Case Study
文件：`PCT官方代码核查报告.md` + `PriorSource替换实验_v3.md` + `LSTM架构一致性重跑.md`

三层证据：
1. **spec-implementation mismatch**（D4 的实证）：
   - KL 反参数（`F.kl_div(p.log(), q)`），loss 放大 6–11.5×；
   - official vs corrected 的 raw ratio 2.83–8.50×（已撤回作"性能结论"；
     gradient-matched 后真实幅度 ~2–4×，方向仍显著，见 `PCT官方代码核查报告.md` §3.5f）；
   - paper-spec ≡ corrected-parity（RNG 对齐后逐位一致）→ 缺陷在实现不在规格。
   - **仅声称**：官方实现与论文 specification 不一致，迁移测试中产生显著差异；
     **不声称** PCT 原结果错误（无 original-data exact reproduction）。
2. **prior provenance / mis-specification harm**（D1+D5 的实证）：
   - 修复后（v4）kelmarsh 5 seed：historical-static 中性（+5.4% 不显著）、
     错误线性先验显著有害（+123.7%，t_p<0.0001）；
   - 在所列 20 个实验格中方向非正（不再称"20/20 非劣"）；
   - 加噪 regime 8 任务 7/8 负收益；三种架构仅在 Kelmarsh seed-0 验证方向一致。
   - W1 漂移提供"机制线索"（不称"机制闭环"）。
3. **paper-spec 声称 LSTM、code 实际跑 Transformer**：第二种 spec-code mismatch。
4. **第二方法族（PI-TFN，pointwise penalty）**：见 `PI-TFN第二族审计方案.md`。
   - paper-spec 重建：5/5 劣于无约束（+119%，t_p<0.001）；
   - 错误固定 Betz Cp 比 learned Ĉp 更差（+139% vs +119%）——跨族复现"错误先验有害"；
   - 边界：n=5 wilcoxon p=0.0625（未达正式显著），无官方代码（no-code 重建）。

## 当前打分（据 steer #2，接受）

| 项 | 分 |
|---|---:|
| Constraint lifecycle taxonomy（升级后）| 待重编码后评估 |
| 系统文献编码 | 5/10（需 26 篇统一 κ + code 核查）|
| PCT 文档-实现不一致 | 7/10 |
| Prior misspecification/harm audit | 7/10 |
| 综合创新性 | 6/10 |

## 必须补的证据（按优先级，通往一区的路径）

1. **P0 击穿防护落地**：~~26 篇按 v3 五维重新编码~~ 已完成（coder_A + 规则 coder_B
   + 仲裁，`literature_coding_table_v3_arbitrated.csv`；D5 κ=1.00、D4 21/26 unverified）。
2. **P0 统计显著性**：~~补 5 seed + 配对显著性~~ 已完成（n=15 site-seed 对，
   4 个 prior source Wilcoxon p≤0.0001；见 `PriorSource替换实验_v3.md`）。
3. **P1 code 可得性核查**：24 篇 UNKNOWN 回头核实（GitHub/README），把
   D4 编码从 UNKNOWN 落到 aligned/mismatch/no-code。
4. **P1 PRISMA 达标**：补多数据库检索 + 去重记录 + PRISMA flow diagram。
5. **P2 exact reproduction 尝试**：PCT 原始江苏数据（向通讯作者索取）如不可得，
   书面记录索取失败 → 明确"这是 transfer stress test 不是复现"。
6. **P2 26 篇统一 κ + Krippendorff's α**（7 字段 → 5 维字段）。

## 已撤回的旧主张（不再写）

- "deployment-time observability" 主线标题。
- oracle_leakage 作为支持证据。
- "整个领域都忽视"（保留为 hypothesis，见编码表声明）。
- partial identification 的 minimax-regret 版本（早已撤回）。
