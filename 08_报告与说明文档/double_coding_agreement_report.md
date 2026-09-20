# 双人独立盲审信度分析与仲裁总表 (Inter-Rater Reliability Report)

- 样本容量: $N = 20$ 篇代表性抽样文献
- 评价编码员: Coder A (杜俊涛) 与 Coder B (独立第二编码员)
- 统计指标: 观测一致率 ($P_o$), 偶然一致率 ($P_e$), Cohen's $\kappa$ 及 95% 置信区间

## 一、 核心维度信度统计汇总
| 审计维度 | 样本数 $N$ | 观测一致率 $P_o$ | 偶然一致率 $P_e$ | Cohen's $\kappa$ | 95% 置信区间 | 一致性评级 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **纳排资格 (Inclusion)** | 20 | 100.0% | 82.0% | **1.000** | [1.000, 1.000] | Almost Perfect (几乎完全一致) |
| **D1: 参数来源 (Provenance)** | 20 | 100.0% | 54.0% | **1.000** | [1.000, 1.000] | Almost Perfect (几乎完全一致) |
| **D2: 施加阶段 (Application Stage)** | 20 | 95.0% | 33.5% | **0.925** | [0.781, 1.000] | Almost Perfect (几乎完全一致) |
| **D3: 推理依赖 (Inference Requirement)** | 20 | 80.0% | 46.0% | **0.630** | [0.305, 0.954] | Substantial (高度一致) |
| **D4: 代码保真度 (Fidelity)** | 20 | 80.0% | 80.8% | **-0.039** | [-0.950, 0.872] | Poor (一致性极低/低于偶然) |
| **D5: 消融类型 (Ablation Type)** | 20 | 85.0% | 31.5% | **0.781** | [0.553, 1.000] | Substantial (高度一致) |
| **D5: 负面效应报告 (Harm Reporting)** | 18 | 100.0% | 100.0% | **nan** | [nan, nan] | Poor (一致性极低/低于偶然) |

---

## 二、 编码分歧与仲裁对照表 (Disagreement Table for Arbitration)
共检测到 **12 处**独立编码分歧，请在仲裁会议中根据原文依据逐项裁定：

| 文献 ID | 分歧维度 | Coder A 判定与依据 | Coder B 判定与依据 | 仲裁决议建议 |
| :---: | :--- | :--- | :--- | :--- |
| **S18** | D2: 施加阶段 (Application Stage) | **training-loss**<br><small>“Jensen wake model velocity deficits define dynamic graph edges (intrinsic); phys...”</small> | **inference-architecture**<br><small>“Jensen wake decay into graph topology; non-crossing penalty in loss (mu>0 vs 0);...”</small> | 待讨论 |
| **S15** | D3: 推理依赖 (Inference Requirement) | **none**<br><small>“Spatial wake decay formula based on turbine distances and wind direction (intrin...”</small> | **contemporaneous-state**<br><small>“A_ij(t) built from wake decay with instantaneous wind direction theta(t); Table ...”</small> | 待讨论 |
| **S18** | D3: 推理依赖 (Inference Requirement) | **none**<br><small>“Jensen wake model velocity deficits define dynamic graph edges (intrinsic); phys...”</small> | **contemporaneous-state**<br><small>“Jensen wake decay into graph topology; non-crossing penalty in loss (mu>0 vs 0);...”</small> | 待讨论 |
| **S27** | D3: 推理依赖 (Inference Requirement) | **none**<br><small>“Physical power curve prior calibrated per wind farm from historical records (his...”</small> | **forecast-input**<br><small>“f_prior(v) = SplineWarping(v; Theta_historical) fitted from historical telemetry...”</small> | 待讨论 |
| **S39** | D3: 推理依赖 (Inference Requirement) | **none**<br><small>“Turbine aerodynamic power conversion relationships (intrinsic); physical convers...”</small> | **forecast-input**<br><small>“P_phys = 1/2 rho pi R^2 Cp(v_WRF) v_WRF^3; P_final = P_phys + residual; Table 3 ...”</small> | 待讨论 |
| **S02** | D4: 代码保真度 (Fidelity) | **code-mismatch**<br><small>“Average power curve joint KDE pre-computed on historical training set (historica...”</small> | **not-found**<br><small>“Target pdf derived from historical SCADA via KDE; L = MSE + alpha*D_JS(P_pred||P...”</small> | 待讨论 |
| **S10** | D4: 代码保真度 (Fidelity) | **not-found**<br><small>“Physical governing differential equations connecting power, aerodynamic torque, ...”</small> | **unverified**<br><small>“J dw/dt = Ta - Tg; L = L_evidential + lambda_ode ||J dw/dt - (Ta-Tg)||^2; Table ...”</small> | 待讨论 |
| **S26** | D4: 代码保真度 (Fidelity) | **not-found**<br><small>“Forecasts wind speed rather than wind power; atmospheric boundary layer regime c...”</small> | **n/a**<br><small>“The objective is to forecast future wind speed series v_{t+1}...v_{t+H} (m/s)....”</small> | 待讨论 |
| **S42** | D4: 代码保真度 (Fidelity) | **not-found**<br><small>“Feature-subspace ensemble using domain heuristics to partition inputs, but lacks...”</small> | **n/a**<br><small>“Features divided into sub-domains (thermal/dynamic/pressure); regression trees t...”</small> | 待讨论 |
| **S19** | D5: 消融类型 (Ablation Type) | **module-ablation**<br><small>“Turbine aerodynamic nonlinear power curve bounds (intrinsic); post-hoc adaptive ...”</small> | **constraint-on-off**<br><small>“Physical constraints enforced post-hoc during conformal interval formation witho...”</small> | 待讨论 |
| **S27** | D5: 消融类型 (Ablation Type) | **both**<br><small>“Physical power curve prior calibrated per wind farm from historical records (his...”</small> | **module-ablation**<br><small>“f_prior(v) = SplineWarping(v; Theta_historical) fitted from historical telemetry...”</small> | 待讨论 |
| **S34** | D5: 消融类型 (Ablation Type) | **both**<br><small>“Analytical Gaussian wake model and aerodynamic efficiency equations (intrinsic);...”</small> | **module-ablation**<br><small>“Analytical Gaussian wake deficit transforms freestream speed/direction into effe...”</small> | 待讨论 |