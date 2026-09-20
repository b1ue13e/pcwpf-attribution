# Protocol v2：Constraint Lifecycle Audit（替换 v1 Observability 协议）

冻结日期：2026-09-04。v1（observability）已归档撤回，本版按终审裁决升级。

## 0. 定位与边界

本协议是**系统性实证审计**的可复用执行规范，审计对象是
physics-constrained wind power forecasting 中的约束：
来源（provenance）→ 应用阶段（stage）→ 推理依赖（inference req）→
实现忠实度（fidelity）→ 伤害报告（harm）。

**核心命题**（可检验 formulation）：
> Physics constraints are learned priors whose benefit depends on provenance,
> implementation fidelity, and distribution stability.

**不检验、不声称**：
- ❌ 部署时不可观测约束（observability 旧主线，已击穿）
- ❌ 原论文结果错误（无 original-data exact reproduction）
- ❌ 约束必然有害（只有"错误先验有害"有证据）

## 1. 五维编码（D1–D5）

见 `ConstraintLifecycle_Taxonomy_schema_v3.md`。取值：
- D1 provenance：intrinsic / historical-static / forecast-derived / learned-online / unreported
- D2 stage：training-loss / post-hoc / inference-architecture / data-generation / unreported
- D3 inference req：none / forecast-input / contemporaneous-state / latent-state / unreported
- D4 fidelity：code-aligned / code-mismatch / no-code / parity-only / unverified
- D5 harm：none-reported / ablation-absent / harm-quantified / substitution-tested

**防击穿条款**：训练期合法历史先验（D1=historical-static）≠ 推理期不可观测约束（D3）。

## 2. 检索与语料（PRISMA-style）

- 检索通道：OpenAlex（title.search）+ 搜索引擎 web_search 两独立通道；记录 query、
  日期、命中数、去重比例、PRISMA flow（见 `PRISMA_flow_diagram.md`）。
- 纳入：forecasting + 显式物理约束，2023–2026。
- 排除：curve-fitting-only、非 forecasting、无物理约束。
- 定位：pilot corpus（当前 26 篇），不称 full systematic review。

## 3. 双人编码与一致性

- 双人工独立编码（coder_A + 真人 coder_C；规则 coder_B 只能作为自检参考，不得
  冒充第二人工标注者并据此报告 κ）。
- 分歧仲裁：先区分字段污染 vs 实质分歧，只对实质分歧仲裁。
- 一致性指标：Cohen's κ + Krippendorff's α（5 字段分别报告）。

## 4. 可执行案例审计（第二方法族要求）

对每个被审计方法族，三层对照：
1. **paper-spec**：按论文公式的严格实现；
2. **official-code**（如有）：逐行核验，记录 fidelity（aligned/mismatch）；
3. **stress test**：prior-source 替换 + distribution-shift 敏感性。

**忠实 runner 硬要求**（终审 G5 固化）：
- dynamic switch / 阈值逻辑逐项对齐（如 PCT 的 js_ratio）；
- batch / horizon / 标准化 / 超参逐项对齐，无法对齐的明确记录为"loss-level"或
  "简化"而非"忠实复现"；
- 因果 shift 方向必须写单元测试（lag 泄漏教训固化）。

**公平比较硬要求**（终审 G6 固化）：
- 不同 loss 项用 gradient-matched 系数消除尺度混淆；
- 报告 raw、official、corrected、gradient-matched、tuned-λ 全谱。

## 5. 统计要求

- ≥5 独立 seed；配对检验（t + Wilcoxon）+ effect size（Cohen's d）+ CI；
- 报告"方向非正"而非"非劣"（避免把单点观测当稳健结论）;
- W1 漂移只作"机制线索"，不称"机制闭环"。

## 6. 提交前 checklist（重新终审门槛）

- [ ] Protocol v2（本文）冻结
- [ ] P0 修复全部落地（lag 因果、nwp 命名、dynamic switch、gradient-matched）
- [ ] 完整 PRISMA flow + 多通道检索记录
- [ ] 26 篇全量双编码（真人 coder_C）+ 5 字段 κ/α
- [ ] ≥5 seed 配对显著性或 effect size
- [ ] 第二方法族审计（pointwise penalty / uncertainty-aware）
- [ ] 数据索取失败/成功书面记录
- [ ] 过度表述全部按"方向一致/线索"口径回退
