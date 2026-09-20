# 3-Arm Guardrail 消融实验结果全总账（5 随机种子 $\times$ 4 噪声环境）

> 运行时间：2026-09-06
> 数据源：`04_数据集/wind_final.csv`（真实中国风电 SCADA，121,528 样本，80/20 时间切分）
> 统一 Backbone：标准 DLinear 时序分解网络（AAAI 2023）
> 随机种子：[42, 123, 456, 789, 1024]
> 评测指标：RMSE (kW), MAE (kW), Physical Envelope Violation Rate (%)

---

## 一、主表（Table 3，标准对比：$\lambda=0.1$ vs Post-hoc vs No Guardrail）

| 扰动环境 ($\sigma$) | 评测策略 / Arm | RMSE (kW) | MAE (kW) | 物理越限率 (%) |
|---|---|---|---|---|
| **Clean ($\sigma=0.0$)** | Arm 1: No Guardrail | $6151.3 \pm 48.7$ | $3907.4 \pm 39.1$ | $1.14\% \pm 0.31\%$ |
| | Arm 2: Post-hoc Clipping | $6148.1 \pm 50.9$ | $3937.6 \pm 35.4$ | $0.00\% \pm 0.00\%$ |
| | Arm 3: Joint Guardrail (Raw, $\lambda=0.1$) | $6153.8 \pm 47.5$ | $3910.8 \pm 40.0$ | $1.13\% \pm 0.30\%$ |
| | Arm 3: Joint Guardrail (Clipped, $\lambda=0.1$) | $6151.0 \pm 49.7$ | $3941.0 \pm 36.2$ | $0.00\% \pm 0.00\%$ |
| **Low Noise ($\sigma=0.2$)** | Arm 1: No Guardrail | $15211.7 \pm 92.3$ | $11821.5 \pm 70.9$ | $6.89\% \pm 0.33\%$ |
| | Arm 2: Post-hoc Clipping | $14520.9 \pm 74.8$ | $11274.7 \pm 71.9$ | $0.00\% \pm 0.00\%$ |
| | Arm 3: Joint Guardrail (Raw, $\lambda=0.1$) | $15179.7 \pm 91.4$ | $11797.0 \pm 70.2$ | $6.85\% \pm 0.33\%$ |
| | Arm 3: Joint Guardrail (Clipped, $\lambda=0.1$) | $14494.7 \pm 74.0$ | $11254.5 \pm 71.0$ | $0.00\% \pm 0.00\%$ |
| **Moderate Noise ($\sigma=0.4$)** | Arm 1: No Guardrail | $28381.6 \pm 193.2$ | $22249.6 \pm 148.6$ | $16.64\% \pm 0.48\%$ |
| | Arm 2: Post-hoc Clipping | $24899.8 \pm 321.7$ | $19464.5 \pm 258.5$ | $0.00\% \pm 0.00\%$ |
| | Arm 3: Joint Guardrail (Raw, $\lambda=0.1$) | $28311.8 \pm 191.8$ | $22196.5 \pm 147.9$ | $16.59\% \pm 0.48\%$ |
| | Arm 3: Joint Guardrail (Clipped, $\lambda=0.1$) | $24853.4 \pm 319.5$ | $19429.4 \pm 256.7$ | $0.00\% \pm 0.00\%$ |
| **Severe Noise ($\sigma=0.6$)** | Arm 1: No Guardrail | $41921.2 \pm 345.4$ | $32896.1 \pm 266.4$ | $25.89\% \pm 0.39\%$ |
| | Arm 2: Post-hoc Clipping | $33514.5 \pm 171.7$ | $26302.4 \pm 173.1$ | $0.00\% \pm 0.00\%$ |
| | Arm 3: Joint Guardrail (Raw, $\lambda=0.1$) | $41814.3 \pm 344.8$ | $32815.2 \pm 266.7$ | $25.81\% \pm 0.39\%$ |
| | Arm 3: Joint Guardrail (Clipped, $\lambda=0.1$) | $33459.3 \pm 171.1$ | $26259.3 \pm 172.8$ | $0.00\% \pm 0.00\%$ |

---

## 二、$\lambda$ 扫描与 Pareto 前沿规律（Figure 4）

在强噪声（$\sigma=0.6$）环境下，随着惩罚权重 $\lambda$ 从 0.01 增大至 1.00：
- 未截断 Raw RMSE 从 $41910.4 \pm 345.3$ kW 持续改善至 $40886.3 \pm 340.3$ kW（降低逾 1,024 kW）；
- 结合截断 Clipped RMSE 从 $33509.0 \pm 171.6$ kW 持续优化至 $32969.9 \pm 169.0$ kW（较 Post-hoc Clipping 进一步降低 545 kW）；
- 清洁数据下，$\lambda=1.00$ 的 RMSE 为 $6174.5$ kW，较 $\lambda=0$ 仅有约 $23$ kW（0.37%）的轻微保守代价。

这强力证明：**物理约束的主要工程价值是强噪声扰动下的防御性守门员（Defensive Guardrail），而非清洁基准下的精度刷新器。**
