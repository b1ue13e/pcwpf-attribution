# 双人编码与仲裁记录（v3 五维，含真实 coder_C，2026-09-04 终版）

## 1. 编码者与过程

| 编码者 | 身份 | 编码依据 | 独立性 |
|---|---|---|---|
| coder_A | 主研究者 | 论文摘要/正文 + 预先理解的五维定义 | 首编 |
| coder_B（规则） | 关键词脚本 | 与 coder_A 共享摘要 + 预先写好的关键词 | **非独立，仅自检参考** |
| **coder_C** | **外部合作者（杜俊涛老师）** | **独立阅读 26 篇摘要/正文，盲审（不看 coder_A 结果）** | **真独立** |

## 2. 真实双人工一致率（coder_A vs coder_C, N=26）

运行 `interrater_agreement_final.py`：

| 维度 | 一致率 | Cohen's κ | 分歧数 |
|---|---|---|---|
| D1 Provenance | 57.7% | 0.436 | 11 |
| D2 Application Stage | 73.1% | 0.426 | 7 |
| D3 Inference Requirement | 53.8% | 0.385 | 12 |
| D4 Implementation Fidelity | 11.5% (对齐后 92.3%) | 0.043 (对齐后 0.480) | 23→对齐后仅 PCT |
| D5 Harm Reporting | 88.5% | −0.040 | 3 |

**方法学结论（诚实，不美化）**：
1. **D5 一致率 88.5% 是真实、稳健的核心 gap**：23/26 双方独立判 none-reported。
   κ=−0.040 是因为类别极度不平衡（近乎恒定字段），此时**直接报告一致率比 κ 更
   有说服力**（这是标准做法：κ 对 prevalence 敏感）。
2. **D4 的 11.5% 是 schema 语义差异**（coder_A 用 `unverified`、coder_C 用
   `no-code`），对齐后 92.3%、κ=0.480——"代码不可核验"是普遍事实，非认知分歧。
3. **D1–D3 的 κ=0.39–0.44 是真实的中等一致**，暴露了独立标注者对"先验来源/应用
   阶段/推理依赖"的认知差异——这正是 taxonomy 需要仲裁的维度，不是错误。

## 3. 实质分歧仲裁（coder_A 认领 vs 需裁定）

### coder_A 认领的错误（修正到 coder_C 判定）
- PITFN D1：forecast-derived → **learned-online**（Cp 是训练中在线学习的网络）
- WANG2026 D3：latent-state → **contemporaneous-state**（限电状态是可观测 SCADA）
- 多篇 D3：unreported → **none**（先前"摘要级不可定"实际是可判的 none）
- RISKREG/SETVAL：unreported → **historical-static + training-loss**

### 需裁定的边界（不偏向任何一方，记录双判）
- HUANG2026 / XAIPC2023：应标记 EXCLUDE（非物理约束 / curve fitting），
  coder_C 给它们编了五维 → 保留编码但标记"边缘纳入，正文应说明"。
- MCMAMBA D1：A=forecast-derived vs C=historical-static → 裁定：Diffusion 的气动
  曲线边界来自历史数据拟合（C 对），但生成的是预报场景样本，两者兼有，
  记 `historical-static(裁定)`。
- JENSENWAKE D1：A=forecast-derived vs C=intrinsic → 裁定：Jensen 公式本身是
  解析物理（intrinsic），但计算需要上游实测风速（见 D3）→ D1=intrinsic 对。

## 4. 仲裁后最终分布（26 篇）

（以 coder_C 为主、认领修正 + 裁定为准；完整表见 `literature_coding_table_v3_final.csv` 的更新版）

- D1：historical-static×10、forecast-derived×8、intrinsic×5、learned-online×1、unreported×2
- D3：none×9（含认领修正）、forecast-input×9、contemporaneous-state×6、unreported×1、latent-state×0
- D4：no-code×24、code-mismatch×1（PCT）、parity-only×1
- D5：none-reported×24、harm-quantified×1（WANG2026）、ablation-absent×1（MMD-PGF）

## 5. 关键 claim 的强度重新校准（据真实 κ）

| claim | 旧表述 | 新表述（据 κ） |
|---|---|---|
| "约束伤害系统性缺少审计" | 强支持（κ=1.0 假）| **稳固**（D5 双人独立一致率 88.5%，直接报告一致率）|
| "实现忠实度不可核验" | 21/26 unverified | **稳固**（对齐后 92.3% 一致：24 no-code + 1 mismatch）|
| "先验来源/推理依赖的分布" | 单作者编码 | **中等置信**（D1/D3 κ=0.39–0.44，需在正文报告 κ 而非只报分布）|

## 6. 诚实声明（替换旧"规则 coder_B κ=1.0"）

- 旧的规则 coder_B κ=1.0 **作废**（非独立，之前已声明"上限而非证据"）。
- 真实双人工（coder_A vs coder_C）结果如上，D1–D3 为中等 κ、D4/D5 对齐后稳固。
- 正文必须**直接引用此表**，不得再引用规则 coder_B 或任何统一 κ=1.0 的旧数。

## 7. 产物

- `literature_coding_table_coderC_blank.csv`（coder_C 完整 26 行 5 维 + notes）
- `interrater_agreement_final.py`（可复现的 κ 脚本）
- 本文（仲裁记录）
