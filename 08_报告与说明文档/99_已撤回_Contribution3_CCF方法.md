# Contribution 3：CCF 方法（Causal Chance-Constrained Feasibility）

日期：2026-09-03

## 模块
- `ccf.py`：`CCF` 类（fit / calibrate / ceiling / audit 四步 API）
- `test_ccf_smoke.py`：三个可证伪性质的自测

## 三步管线（全 slice 隔离，无泄漏）

```
fit(X_tr, Y_tr)                   -> pinball 条件分位模型（train only）
calibrate(X_cal, Y_cal, alphas)   -> split-conformal 单侧非负修正（calib only）
ceiling(X, alpha)                 -> Uhat = Q_{1-alpha}(X) + qhat(alpha)
audit(X_te, Y_te, alpha)          -> FER/LEC/coverage/sharpness（test only）
```

## 三个核心性质（smoke 验证通过）

1. **声明风险兑现（coverage 保证）**
   α=0.1 → calib coverage 0.933（≥0.9）；α=0.2 → 0.852（≥0.8）。
2. **Pareto 支配 deterministic proxy**
   wind_cn h6：proxy FER=0.763 / LEC=0.081 → CCF FER=0.100 / LEC=0.007。
3. **无泄漏四步 API**
   scaler/模型 fit train only；conformal shift 来自 calib only；shift 非负。

## 与其他方法的关系（正交，不竞争）
- PCT/TgDPF（KDE+JS）：对齐预测边际分布，不估计可行上界条件分布。
- CCF：直接估计 U*|H_t 的条件分位并校准风险，与分布对齐正交可叠加。

## 待完善（写方法稿前）
- 更大数据上的多地 seed 平均 coverage 是否稳定（penmanshiel 在标杆中 FER 偏高）。
- α 与 (c_FE, c_FA) 代价的显式映射（当前只用 α 作旋钮，未做代价优化）。
- 非 LSTM 载体（transformer/patchtst）的 CCF 一致性。
