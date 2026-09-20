# CCF 主线完整结果（Contribution 1+2，2026-09-03）

## 三件套当前状态
1. **Contribution 1（理论）** ✅ ：Constraint Observability Gap 三命题（可证伪，见 Contribution1 文档）
2. **Contribution 2（benchmark）** ✅ ：5 种 ceiling 对照 × 5 指标，300 行结果
3. **Contribution 3（CCF 方法）** 🔄 ：条件分位 + split-conformal，已在 benchmark 内实现，待独立方法稿

## 全站汇总（5 站 × 3 horizon × 2 seed）

| ceiling | FER | LEC | ramp_FER | sharpness |
|---|---|---|---|---|
| intrinsic_box | 0.000 | 0.000 | 0.000 | 1.000 |
| oracle_curve | 0.849 | 0.133 | 0.885 | 0.233 |
| linear_proxy | 0.790 | 0.137 | 0.751 | 0.243 |
| persistence_proxy | 0.804 | 0.138 | 0.766 | 0.240 |
| **ccf_quantile (α=0.1)** | **0.104** | **0.010** | **0.180** | **0.523** |

## 每站 proxy vs CCF 支配判定（15/15 全胜）

| 站点 | proxy FER | CCF FER | 支配 |
|---|---|---|---|
| kelmarsh | 0.902 | 0.151 | ✅ |
| penmanshiel | 0.742 | 0.105 | ✅ |
| smarteole | 0.892 | 0.066 | ✅ |
| wind_cn | 0.731 | 0.105 | ✅ |
| wind_us | 0.752 | 0.093 | ✅ |

## 三个结论（已钉死）

1. **deterministic proxy ceiling 全站 FER 73-90%**——手工 proxy 把 7-9 成合法出力错杀。
2. **oracle_curve FER=0.849**——连"真实未来风速曲线"也错杀 85%，
   证明 observability gap 根源不仅是 proxy lag，更是 power-curve-as-upper-bound 的模型错误。
3. **CCF 条件分位把 FER 校准到 ≈α（0.10），LEC 降 10 倍以上，15/15 支配**。

## 数据文件
- 本地：tmp/ccf_benchmark_all.csv（300 行 = 5站×3h×2seed×(5 ceiling + α网格)）
- 云端：25329 /root/paper1/results/ccf_bench/
- 协议：08_报告与说明文档/Contribution2_CCF审计benchmark协议.md
- 理论：08_报告与说明文档/Contribution1_ConstraintObservabilityGap_理论陈述.md
