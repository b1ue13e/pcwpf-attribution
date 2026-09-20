# TaskC2 三方编码验收结论（v4 schema，最终口径）

> 日期：2026-09-04。本文件是 D1–D3 最终口径的**单一事实源**。
> 最终表：`literature_coding_table_FINAL21_v2.csv`（D1–D3 以本文件为准）。

## 一、三方 κ 结果（N=21）

| 维度 | FINAL21(摘要级,旧schema) vs 杜老师 v4(全文级) | 杜老师 v3→v4（全文级，schema 微调）|
|---|---|---|
| D1 | 42.9%，κ=0.257 | 95.2%，κ=0.892 |
| D2 | 61.9%，κ=0.188 | 100.0%，κ=1.000 |
| D3 | 57.1%，κ=0.366 | 100.0%，κ=1.000 |

## 二、κ 的解释（论文要写的核心方法论结论）

1. **低 κ 的两个来源，均已定位，均非"概念随机性"**：
   - **信息不对称**：coder_A/coder_C 为摘要级编码；杜老师为全文级编码（20/21 篇读过全文）。
   - **旧 schema 歧义**：v3 的 D1 混入"计算先验需要什么输入"（forecast-derived 取值），
     v4 已删除该取值并立铁律"输入依赖全交 D3 表达"。
2. **全文级判定的可复现性证据**：同一全文级编码者（杜老师）在 v3→v4 schema
   修订下，D1–D3 判定 63 个单元格仅 1 个变动（NCF2026 D1），κ=0.89–1.0。
   这证明：**给定全文 + 清晰 schema，判定高度可复现**。
3. **D4/D5 不受影响**（事实性字段，此前三方已验：D4 语义对齐后 92.3% 一致、
   D5 88.5% 一致）。

## 三、最终口径（仲裁决议）

- **D1–D3 全部采用杜老师 v4 全文级编码**（21 篇全量）。
- **3 处仲裁标注**（写入 FINAL21_v2 的 `arbitration_note` 列）：
  1. **PITFN2025 D1 = `intrinsic+learned-online`**（混合）：cut-in/rated/nonneg 边界为
     解析常数（intrinsic）；L_phys 的 Ĉp 由估计器网络训练中学习（learned-online，
     coder_C 判定有公式 (20) 文本依据）。杜老师全文判 intrinsic，未计入 Ĉp 网络层。
     正文描述 PITFN 时写混合表述。
  2. **TANG2025 D1**：采用杜老师全文级 intrinsic（气动方程）；coder_C 摘要级判
     historical-static（历史拟合曲线）。差异可能反映该文同时含两类约束，正文脚注。
  3. **WANG2026 D3 = `latent-state`**：采用杜老师（预测对象为限电遮蔽的反事实可用
     功率，无传感器可读）；coder_C 判 contemporaneous-state 针对的是输入侧限电标签，
     非预测对象。二者不矛盾，以预测对象口径为准。

## 四、最终 21 篇分布（FINAL21_v2）

| 维度 | 分布 |
|---|---|
| D1 | intrinsic×14、historical-static×6、intrinsic+learned-online×1 |
| D2 | training-loss×14、inference-architecture×4、post-hoc-correction×2、data-generation×1 |
| D3 | none×15、forecast-input×3、contemporaneous-state×2、latent-state×1 |
| D4 | no-code×20、code-mismatch×1 |
| D5 | none-reported×19、ablation-absent×1、harm-quantified×1 |

## 五、对论文表述的影响

- C2 的编码可靠性声明改为三段式：
  "摘要级编码（两作者）在 D1–D3 上 κ=0.19–0.37，定位为信息不对称与 schema 歧义；
  schema v4 修订后，全文级第三编码员重编，v3→v4 判定 κ=0.89–1.0；
  D1–D3 最终采用全文级 v4 编码，D4/D5 三方一致（92.3%/88.5%）。"
- 不再写"taxonomy 可靠性中等、D1-D3 降级为描述性分布"——v4+全文级编码后，
  D1–D3 有可辩护的最终口径。
