# Task C — 21 篇文献五维盲法编码（第三编码员）

> 材料：本文件 + `literature_coding_table_coderD_blank.csv`（空白表）+ 各论文公开摘要/全文。
> 你的任务：对 21 篇论文做五维编码。**这是盲法**：空白表里只有论文 id 和出处信息，
> 没有我们或前两位编码员的任何判定，也不需要 & 不应查看任何既有结论。

## 1. 五维定义（请严格按此，不要自行扩展语义）

对每篇论文，判断其**物理约束（physical constraint / physics prior）**的五个属性：

### D1 Provenance（约束来源）——物理先验用什么数据估计？
| 取值 | 含义 |
|---|---|
| `intrinsic` | 物理常数/固有定律（Betz 极限、额定功率、切入切出风速、PMSG/BEM 方程），**不需从数据估计** |
| `historical-static` | 训练前用**历史观测**一次性估计，训练期冻结（如历史数据拟合的 KDE 功率曲线）|
| `forecast-derived` | 用**预报/NWP 输入 + 物理模型**的输出（如预报风场上的 Jensen 尾流、理想功率曲线）|
| `learned-online` | 训练中随参数**在线更新**的先验（如 C_p 估计器网络）|
| `unreported` | 论文未说明来源 |

### D2 Application stage（应用阶段）
| 取值 | 含义 |
|---|---|
| `training-loss` | 作为训练损失的正则/惩罚项 |
| `inference-architecture` | 作为推理期模型结构的一部分（嵌入网络架构）|
| `data-generation` | 用于生成/扩充训练数据（如 physics-guided diffusion 样本生成）|
| `post-hoc-correction` | 推理后再修正输出 |
| `unreported` | 未说明 |

### D3 Inference requirement（推理期依赖）——推理时是否需要训练集以外的新信息？
| 取值 | 含义 |
|---|---|
| `none` | 推理完全自足（只用已训练模型 + 常规输入）|
| `forecast-input` | 需要预报输入（NWP 等未来时步信息）|
| `contemporaneous-state` | 需要实时 SCADA/机型状态（如实时桨距角、叶尖速比、限电状态）|
| `latent-state` | 需要不可直接观测的隐状态（如"本可发出的功率"在限电遮蔽下）|
| `unreported` | 未说明 |

### D4 Implementation fidelity（实现忠实度）——如实核验后填写
| 取值 | 含义 |
|---|---|
| `code-aligned` | 有公开官方代码，且与论文公式一致 |
| `code-mismatch` | 有公开官方代码，但与论文公式不一致 |
| `no-code` | 论文明确声明无公开代码 |
| `not-found` | **你检索后未找到**公开代码（不代表不存在，只代表未找到）|
| `unverified` | 你未做检索核验 |

### D5 Harm reporting（伤害报告）
| 取值 | 含义 |
|---|---|
| `none-reported` | 未报告约束的任何负面效应 |
| `ablation-absent` | 未做约束有无的消融（收益未经过约束对照）|
| `harm-quantified` | 报告了约束在某些设定下的负面结果 |
| `substitution-tested` | 做了先验替换/oracle 敏感性测试 |

## 2. 核心防混淆条款（务必遵守）

> **训练期用历史数据估计的静态先验（D1=historical-static）≠ 推理期需要不可观测信息
> （D3=latent-state）。** 两者是不同维度，不要因为"先验来自数据"就把它标成 D3 依赖。

## 3. 操作步骤

1. 打开 `literature_coding_table_coderD_blank.csv`（21 行，含 paper_id / paper_short / journal / year）。
2. 对每篇，用 paper_short + 期刊 + 年份**自行检索**到论文（Google Scholar / DOI / 期刊网站），读摘要；D4 需要实际搜一下官方代码（GitHub / 论文 Data Availability）。
3. 在空白表的 `D1_provenance`…`D5_harm` 五列填入取值（用上表英文值），`coderD_notes` 写一句依据（"论文公式/代码哪一处"）。
4. **不要修改** paper_id / paper_short / journal / year 四列。

## 4. 报告模板（正文部分，与表一起交）

```markdown
# TaskC 报告 — <你的名字> — <日期>

## 检索说明
- 每篇论文的检索方式（Scholar/DOI/期刊页）
- 哪些篇只有摘要、哪些篇读了全文

## 个人把握标记
- 列出你认为"只有摘要信息、判定存疑"的 paper_id 及原因

## 编码表
（填写后的 coderD_blank.csv 附上；也可粘贴关键列）

## 声明
我独立完成编码，未参考本项目任何既有编码结果。
```

## 5. 提交

- `literature_coding_table_coderD_blank.csv`（填写后，改名 `TaskC_coderD_你的名字.csv`）
- `TaskC_报告_你的名字.md`
