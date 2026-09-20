# PRISMA 式检索日志（扩展语料 v1，2026-09-03）

## 检索记录（query / 结果 / 日期）

| # | query | 关键命中（新增） | 日期 |
|---|---|---|---|
| 1 | physics-informed wind power forecasting constraint loss 2024 2025 | PISTNet (Aslam 2025), PIBPNN (IEEE 2025), TCN-TFT quantile (Mi 2025), PI-TFN | 09-03 |
| 2 | wind power curve constraint deep learning regularization 2023-2025 | XAI power curve (Energy and AI 2023), Risk-regularization (Zhang 2025), Set-valued regression (TSTE 2025), Cp-curve SCADA digital-twin (Energies 2025) | 09-03 |

## 纳入标准（按 ObsAudit Protocol v1 §2）

- 纳入：明确用 power-curve / Betz / JS-KDE / MMD / 物理 residual / Cp boundary
  作为 forecasting 模型的 constraint（hard/soft/distributional）。
- 排除：纯 power-curve fitting（非 forecasting）、纯 XAI 后验、无物理约束的纯数据驱动。

## 新增编码对象（8 篇，待 coder 编码）

1. PISTNet (Aslam et al., Computers & Electrical Engineering, 2025) — advection-equation physics + adaptive penalty loss (APPL)
2. PIBPNN (IEEE conference, 2025) — Betz limit + Cp bounds + mechanical laws 作为 loss 正则
3. PI-TFN (Mi et al., Energy, 2025) — 公式 (20)：cut-in/rated/nonneg/Betz-residual 四惩罚项，含权重 λ
4. TCN-TFT quantile (Mi et al., Energy, 2025) — Bernoulli + ideal gas law physics loss
5. Cp-curve SCADA digital-twin (MDPI Energies, 2025) — penalty terms 保持 Cp 曲线物理合理
6. XAI power curve (Energy and AI, 2023) — physics-informed power curve 正则化验证（可能排除：非 forecasting）
7. Risk-regularization (Zhang et al., IEEE TETCI, 2025) — chance-constrained interval power curve
8. Set-valued regression (IEEE TSTE, 2025) — interval neural network + chance constraints（可能排除：curve fitting）

## 关键锚点更新

- **PI-TFN 是 pointwise power-curve penalty 的理想已发表锚点**：公式 (20) 明确
  `L_total = L_MSE + λ1 L_cutin + λ2 L_rated + λ3 L_nonneg + λ4 L_phys`，
  `L_phys = mean((ŷ − P^phys)²)` with `P^phys = ½·ρ·A·Ĉp·v³`（Betz-limited C_p estimator）。
  这比唐 2025（只有摘要）扎实得多，且明确用了**部署不可得的 C_p estimator**——
  正好是可观测性审计的靶子（C_p 依赖未来/训练集数据）。

## 状态

- 新 8 篇已编码进 `literature_coding_table_extended.csv`（17 行 total）。
- PRISMA 筛选结果（初步，2026-09-03）：
  - **纳入（forecasting + 明确物理约束）13 篇**：TgDPF、PCT、MMD-PGF、PISTNET2025、
    PIBPNN2025、PITFN2025、TCNTFT2025、TANG2025、WANG2026、JARA2024、PHYCELL、MLPINN。
  - **排除 5 篇**：HUANG2026（非物理约束）、XAIPC2023 / RISKREG2025 / SETVAL2025
    （curve fitting 非 forecasting）、CPDIGITAL2025（Cp curve fitting）。
- 下一步：coder_B 对 8 篇新增独立编码 → 与 coder_A 对比 → κ（扩展样本）。
