# LSTM 架构一致性重跑（Contribution 4 补充）

## 目的

排除"oracle 约束负收益"结论来自架构选择的混淆。我们此前的实验用
`NetLSTM`（2 层 LSTM + Linear(20,1)），但 PCT 论文声称用 LSTM，code 实际跑
Transformer。为严谨，三种架构对照：

| 架构 | 来源 | 结构 |
|---|---|---|
| ours | 我们之前的实验 | 2 层 LSTM + Linear(dim_hidden→1) |
| paper-lstm | PCT repo 注释的 NETLSTM | 2 层 LSTM + fc2(hidden→hidden/2)+ReLU+fc3(hidden/2→1)，dropout 0.3 |
| transformer | PCT repo 实际训练的 | d_model=12, 1 层 encoder, nhead=2 |

## 实验

kelmarsh 单站，seed=0，epochs=60，3 架构 × {clean, noisy(0.5)} × {oracle-constraint,
unconstrained} = 12 runs，RNG 对齐。

## 结果（kelmarsh, seed=0, epochs=60, 2026-09-03）

| 架构 | regime | oracle | unconstrained | gain |
|---|---|---|---|---|
| ours | clean | 0.1073 | 0.0758 | −41.5% |
| ours | noisy | 0.1325 | 0.0812 | −63.1% |
| paper-lstm | clean | 0.1134 | 0.0772 | −46.8% |
| paper-lstm | noisy | 0.1260 | 0.0812 | −55.1% |
| transformer | clean | 0.1191 | 0.0680 | −75.1% |
| transformer | noisy | 0.1159 | 0.0702 | −65.2% |

## 结论

**三种架构 6/6 全部 oracle 约束负收益（−41.5% 到 −75.1%）**，方向完全一致：
- ours：−41.5% / −63.1%
- paper-lstm（PCT 论文声称）：−46.8% / −55.1%
- transformer（PCT 代码实际跑）：−75.1% / −65.2%

**架构混淆的有限排除**：在 Kelmarsh 单站 seed-0 三种架构中，约束负收益方向一致
（不构成"架构无关"的普遍证明）。额外发现：transformer
（PCT 代码实际使用的架构）负收益最大（−65~−75%）——code 实际架构下约束害处
最严重，与"paper 声称 LSTM、code 跑 transformer"的 documentation-implementation
mismatch 相互印证。
