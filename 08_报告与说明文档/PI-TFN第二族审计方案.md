# 第二方法族忠实审计：PI-TFN（pointwise penalty 族）

> ⚠️ **作废（第二轮终审 2026-09-04）**：P^phys 单位 W/kW 错配（放大 1000×），
> fixed-Cp 爆炸与 n=9 跨族显著结论作废。修复后 fixed_cp 从 ≈88 降至 ≈0.19。
> 详见 `第二轮终审响应_最终状态.md`。


日期：2026-09-04。目标：补齐 steer #3 的最低门槛"至少第二个独立方法家族"。

## 为什么选 PI-TFN（而非 MMD-PGF）

| 候选 | 公式完整度 | 数据 | 与 PCT 族关系 |
|---|---|---|---|
| **PI-TFN** | ✅ 公式 (19)(20) 完整（4 惩罚项 + λ 权重 + P^phys 定义）| ✅ Kelmarsh（我们有）| **pointwise penalty 族，正交** |
| MMD-PGF | ⚠️ MMD 核函数带宽等细节未公开 | 未知 | distributional 族，与 PCT 同族，判别力弱 |

PI-TFN 是 pointwise penalty（hard-constraint-violation 正则），PCT 是 distributional
（KDE/JS 匹配）。**跨族对照**能检验"约束收益依赖 provenance/fidelity/stability"
这一命题是否局限于 distributional 族。

## PI-TFN 关键公式（冻结）

```
L_total = L_MSE + λ1·L_cutin + λ2·L_rated + λ3·L_nonneg + λ4·L_phys
L_cutin = (1/|S|)Σ_{i∈S} max(0, ŷ_i)²         S={i: v_i < v_cutin}        λ1=0.1
L_rated = (1/|R|)Σ_{i∈R} (ŷ_i − P_rated)²    R={i: ŷ_i > P_rated}        λ2=0.1
L_nonneg = (1/|N|)Σ_{i∈N} ŷ_i²               N={i: ŷ_i < 0}              λ3=0.05
L_phys  = (1/|V|)Σ_{i∈V} (ŷ_i − P^phys_i)²   V={i: v_cutin ≤ v_i ≤ v_rated}  λ4=0.05
P^phys_i = ½·ρ·A·Ĉp_i·v_i³    （Betz-limited learned Ĉp estimator, 64→32→1）
```

主体架构：Bi-LSTM backbone + 多尺度时序注意力 + physical 层（风速-功率立方律 +
Ĉp estimator）+ 热状态 GRU + 叶片协调模块。**本审计只忠实重建损失层（公式 20）
+ Ĉp estimator 的约束语义**，不重建完整多模块架构（这是"loss-level parity"，
与 PCT 案例的"loss-level+full-train parity"层级对应）。

## 审计命题（预注册）

P1: PI-TFN pointwise 惩罚项在 clean Kelmarsh 上的收益（vs unconstrained）。
P2: L_phys 的 self-referential Ĉp 依赖：Ĉp 由预测驱动，P^phys 又约束预测，
   是否形成循环（D1=learned-online 的独特风险）。
P3: 错误物理参数（ρ/A/Ĉp 误设）如何传播——与 PCT 的 nwp-linear 先验错误对照。

## 三版本对照设计（与 PCT 案例同构）

- **paper-spec**：公式 (20) 的严格实现（Ĉp 用 estimator 网络输出）
- **official-code**：PI-TFN 无公开官方代码 → 该项为 `no-code`（记录），
  与 PCT 的 code-mismatch 形成对照（PCT 有码但错，PI-TFN 无码不可核验）
- **corrected-parity**：N/A（无官方代码可 corrected）→ 审计层级降为
  "paper-spec 忠实重建 + 变体消融"，不声称 parity

## 边界（诚实，防过度声称）

1. 无官方代码 → 只能做 paper-spec 重建，不能声称"官方实现不一致"。
2. 只重建 loss 层 + Ĉp estimator，不重建完整多模块架构 → 是 loss-level parity，
   非架构级 exact reproduction。
3. 数据=Kelmarsh（PI-TFN 外部验证用的同一数据），但分桶/预处理按我们协议。
4. 若 PI-TFN paper-spec 重建无法显著复现其声称（"物理引导提升性能"），
   只能报告"我们未能验证点约束的收益"，不声称 PI-TFN 结果错误。

## 结果（kelmarsh, 5 seed, 40 epochs, loss-level reconstruction）

### v1（初版，2026-09-04，三处 paper-spec 缺陷未修）

| 变体 | mean 相对差 | worse | t_p |
|---|---|---|---|
| paper-spec（4 惩罚项）| +119.3% | 5/5 | 0.0001 |
| fixed-cp（错误 Betz 16/27）| +139.4% | 5/5 | <0.0001 |

### v2（复核修复：rated mask→pred>1.0；Cp 乘 Betz 上界；物理 A=6×π·46²）

| 变体 | mean 相对差 | worse | t_p |
|---|---|---|---|
| paper-spec | +186.6% | 5/5 | 0.0033 |
| no-phi（去 L_phys）| +60.8% | 5/5 | 0.046 |
| fixed-cp（错误 Betz 16/27）| **+129740%**（约束项冲突致训练爆炸）| 5/5 | <0.0001 |

### v3（补 seed 至 n=9，2026-09-04，正式显著）

| 变体 | mean 相对差 | worse | wilcox_p | Cohen's d |
|---|---|---|---|---|
| paper-spec | +171.6% | 9/9 | **0.0039** | 3.19 |
| no-phi | +85.2% | 9/9 | **0.0039** | 1.66 |
| fixed-cp | +128115%（训练爆炸）| 9/9 | 0.0039 | — |

**v1 结论作废**（rated mask 用 wind>v_rated 而非论文的 pred>P_rated；Cp 无 Betz 界；
A 反推而非物理）。v2/v3 为当前有效版本。

**跨族结论（v3，正式显著）**：
1. paper-spec 9/9 劣于无约束，wilcoxon p=0.0039、Cohen's d=3.19（大效应）——与
   PCT distributional 族方向一致，"约束收益依赖 provenance/stability"命题
   **跨族正式显著成立**。
2. fixed-cp 的 +128115% 是**错误先验 + 约束项冲突**（L_phys target 1.55 pu 与
   L_rated 惩罚 pred>1.0 矛盾）导致的训练爆炸——"错误先验有害"的极端形态，
   定性证据（不作量化）。
3. no-phi +85.2%（p=0.0039），L_phys 单独贡献约 +86pp。

**诚实边界（回应复核"尚非忠实锚点"）**：
- 无官方代码 → 只能 paper-spec 尽力重建；**A 是物理推理假设（6×Senvion MM92），
  非论文值**；架构简化（loss-level + 简化 backbone）。
- 因此本审计定性为 "pointwise-penalty 族的 paper-spec 重建 stress test"，
  **不称忠实复现或官方实现 audit**。

## 边界补强（n=5 局限）

本审计是 5 seed 的 loss-level 重建，statistics 为边界显著（wilcoxon 0.0625）。
要写成正式结论需补到 ≥9 seed（wilcoxon 精确分布能到 p<0.01），或报告 t 检验 + 
effect size（Cohen's d）而非仅 wilcoxon。**当前结论强度：跨族方向一致 + t 显著，
但 wilcoxon 未达 0.05，属"强信号但非正式显著"。**
