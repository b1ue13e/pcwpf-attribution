# Contribution 1：Constraint Observability Gap —— 理论陈述（正式稿 v1）

> 状态：主线理论钉死稿。所有命题必须是可证伪的；任何"必然/普适"字样禁止出现。
> 配套证据：5 站 × 3 horizon × 2 seed 判生死实验（15/15 支配）见 `ccf_full_all.csv`。

---

## 1. 问题设定（Setup）

风电功率预测任务 `(t, h)`：在时刻 `t` 用部署可见历史 `H_t` 预测 `t+h` 的功率 `P̂_{t+h}`。

**物理可行上界** `U*_{t+h}` 是一个**潜变量**，依赖未来运行状态（风速 `v_{t+h}`、限电、尾流、控制状态）。我们假设部署时无法观测 `U*_{t+h}`，只知道：
- realized power `Y_{t+h}`（经 QC、排除明显传感器故障后的实测出力）是 **feasible witness**：
  若 `Y_{t+h}` 真实发生过，则该值未被可行上界排除，即 `Y_{t+h} ≤ U*_{t+h}` 几乎必然成立。

> 关键不可判定性：由 `Y_{t+h} ≤ U*_{t+h}` 只能在下向判断——`Uhat < Y` ⇒ 确定 false exclusion；
> `Uhat ≥ Y` 不能证明安全（因为 `U*` 可能高于 `Uhat` 也高于 `Y`）。

---

## 2. 两类约束（Definition 1）

**Intrinsic constraint**（部署时确定，可硬执行）：
```
0 ≤ P̂ ≤ P_rated
```
**State-contingent constraint**（依赖未来状态，部署时是随机的）：
```
P̂_{t+h} ≤ U*_{t+h}
```
FG-MoE 类方法的核心错误：把第二类约束当作第一类，用确定性 proxy `c(H_t)` 代替随机上界。

---

## 3. Constraint Observability Gap（Definition 2）

定义部署时可见的可行上界分布与点估计之间的差距：

```
Gap(H_t) := U*_{t+h} | H_t   vs   ĉ(H_t)
```

其中 `ĉ(H_t)` 是任意确定性函数（persistence / linear / NWP / empirical curve）。

---

## 4. 三难不可能性（Proposition 1，可证伪的核心）

**假设 (A1)**：`U*_{t+h} | H_t` 非退化（对某个正测度 history 集，条件分布支集非单点）。

**命题**：在 (A1) 下，不存在确定性 ceiling `ĉ(H_t)` 同时满足：

1. **零 false exclusion**：`Pr(ĉ < U* | H_t) = 0`；
2. **零 conservatism**：`ĉ = U*(H_t)` 的某个中心位置（不系统抬高）；
3. **零 false admission 风险**：`Pr(U* < ĉ | H_t)` 不因压低 ĉ 而恶化。

**证明（精确版）**：由（A1），存在两实数 `q₁ < q₂` 与 `ε > 0`，使
`Pr(U* ≤ q₁ | H_t) ≥ ε` 且 `Pr(U* ≥ q₂ | H_t) ≥ ε`（支集非单点）。

对任意确定性 `ĉ = ĉ(H_t)`：

- 若 `ĉ < q₂`，则 `Pr(U* > ĉ | H_t) ≥ Pr(U* ≥ q₂ | H_t) ≥ ε > 0` → 存在正的 false-exclusion 质量（合法上界高于 ceiling）。
- 若 `ĉ ≥ q₂`，则 `Pr(U* ≤ q₁ | H_t) ≥ ε`，即 ceiling 在至少 ε 质量上高于可行上界的中低位 → 正的 conservatism（为覆盖上方而整体抬高 ceiling，牺牲下方紧致性）。

故任意 `ĉ` 都必须落在"非零 FE"或"非零 conservatism"（或两者）中，false admission 风险的降低与二者正交，三难不可兼得。∎

> 这与旧的"点约束必然崩溃"完全不同：旧命题声称**某一族方法一定失败**（有反例），
> 新命题只声称**确定性替代必然承担可量化的权衡**（由分布非退化直接推出，无反例）。

---

## 5. 条件分位数是最优 ceiling（Proposition 2，Bayes-optimal）

**设定非对称代价**：false exclusion 单位代价 `c_FE > 0`（合法出力被切），false admission 单位代价 `c_FA > 0`。

**命题**：在平方/线性代价形状下，Bayes-optimal ceiling 是条件分位数：

```
ĉ*(H_t) = Q_{α*}(U*_{t+h} | H_t),   α* = c_FE / (c_FE + c_FA)
```

**证明**：设可行上界为 `U`，给定 ceiling `q` 的期望代价
`R(q) = c_FE · E[(U − q)⁺] + c_FA · E[(q − U)⁺]`。
subgradient：`∂R(q) = −c_FE·Pr(U>q) + c_FA·Pr(U≤q)`。
令其含 0 ⟺ `c_FE·Pr(U>q) = c_FA·Pr(U<q)`（普连续情形）
⟺ `Pr(U≤q) = c_FE/(c_FE+c_FA) = α*`，即 `q = Q_{α*}`。∎

**推论**：α 的物理意义是风险预算——α→0 是 intrinsic box（FER→0 但最保守），α→1 是无约束（FER 最大但零 conservatism）。CCF 就是显式选择 α。

---

## 6. 离线可检验形式（Proposition 3，与实验对接）

因 `U*` 不可观测，离线只能用 feasible witness `Y`。定义可观测指标：

```
FER(ĉ) := Pr(ĉ(H_t) < Y_{t+h})              # false exclusion rate（下向确定）
LEC(ĉ) := E[(Y_{t+h} − ĉ(H_t))⁺ | ĉ < Y]    # legitimate energy clipped（条件于被误切）
```

**可证伪判据**：若条件分位数 ceiling `Q_α` 由 pinball loss 训练且在独立校准集上校准，则
`FER(Q_α) ≈ α`，且对 calibration 良好的 `Q_α`，在相同 FER 水平下 `LEC(Q_α) < LEC(proxy)`。

**⚠ 单向声明**：FER 只给出 false-exclusion 侧的证伪能力；false-admission 侧在线下
**不可完整测出**（因 `U*` 未知），只能在可行 witness 上做一致下界。这是方法的固有限制，必须写进 Limitations。

---

## 7. 反例与边界（防止重蹈旧主线覆辙）

本理论**不声称**：
1. "点约束一定失败"——只声称确定性 ceiling 必须在 FER/conservatism/FA 间分配质量；
2. "分布约束必然安全"——只声称条件分位数在给定代价下 Bayes-optimal；
3. "proxy lag 是唯一机制"——lag 只是让 `U|H_t` 非退化的一个来源（另有限电、尾流、控制）；
4. "α 校准在一切 site 稳定"——penmanshiel 的 FER 偏高（见实验）正是 α 校准精度差的体现。

---

## 8. 对已发表方法的定位（不抢功，只划界）

- **PCT / TgDPF 的 KDE+JS**：在**预测边际分布**上对齐，属于 distribution matching，不直接估计 `U|H_t` 的条件分位数。
- **CCF 的差异**：直接对**可行上界的条件分布**建模并校准风险水平，得到的是**有 coverage 保证的最优 ceiling**。
- 三者的关系是**正交**而非竞争：可以同时使用（分布对齐做预测质量，CCF 做约束风险）。

---

## 9. 一句话主线（钉死版）

> **部署时不可观测的 state-contingent 物理上界，被确定性 proxy 约束替代时，必然产生可量化的 false-exclusion/conservatism 权衡；将约束重构为可行上界的条件分位数，可在任意给定风险预算下 Bayes-最优地最小化该权衡。**
