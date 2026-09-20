# Constraint Lifecycle Taxonomy（schema v4，2026-09-04 修订）

> 修订说明：第三编码员（杜俊涛，读 20/21 全文）独立编码后，D1/D2/D3 的 κ 仅
> 0.19–0.37，暴露 v3 schema 定义歧义。核心歧义：**D1（先验从哪来）被混入了
> "计算先验需要什么输入"这层意思**，导致同一个物理模型（如 Jensen wake）一方
> 判 `intrinsic`（公式是解析物理）、另一方判 `forecast-derived`（计算需预报）。
> v4 把这两层彻底拆分，并把 D2 的判定原则显式化。

## 一句话定位

本文审计的是**约束在完整生命周期中的角色与风险**：prior 的知识形态与参数来源
（D1）→ 应用阶段（D2）→ 推理期计算依赖（D3）→ 实现忠实度（D4）→ 伤害报告（D5）。

---

## 铁律（防歧义，重编前必读）

> **D1 只问"先验的参数从哪来 / 知识形态是什么"；D3 只问"推理时算这个先验需要
> 什么输入"。两者独立，互不牵连。**
>
> - 同一个 Jensen wake：D1=`intrinsic`（解析物理公式）、D3=`forecast-input`
>   （要算亏损需要预报风速）——这是**合法且常见**的组合，不许因为 D3 需要预报
>   就把 D1 也改成 forecast-derived。
> - 判断顺序永远：先"这先验是什么形态"，再"用它要什么输入"。两步分开填。

---

## D1. Provenance（参数来源 / 知识形态）——**只此一问：这个先验的参数值从哪里来？**

- `intrinsic`：先验是**不需从数据估计的物理事实/解析方程**（Betz 极限 16/27、
  额定功率、切入切出风速、Jensen 尾流解析式、PMSG d-q / BEM 方程、理想气体/
  伯努利方程）。**注意**：即使计算它需要预报输入，只要参数本身是解析物理，
  就仍是 `intrinsic`。
- `historical-static`：先验的**参数从历史观测一次性估计**，训练期冻结
  （如从 SCADA 历史拟合的 KDE 功率曲线、历史分位数、经验功率曲线）。
- `learned-online`：先验的**参数在训练中随网络更新**（如 C_p 估计器网络、
  可训练物理模块）。
- `unreported`：论文未说明参数来源。

> 删除 v3 的 `forecast-derived` 作为 D1 取值——它混淆了"参数来源"与"计算输入"。
> "依赖预报输入"的一律改由 D3 表达。

## D2. Application stage（应用阶段）——**只此一问：约束在哪个阶段被施加？**

- `training-loss`：作为**训练目标的正则/惩罚项**（loss 里加一项）。
- `inference-architecture`：作为**推理期网络结构本身**的一部分（约束嵌入前向计算），
  而非训练目标。
- `data-generation`：用于**生成/扩充训练样本**（如 physics-guided diffusion 采样）。
- `post-hoc-correction`：**训练完成后**、推理输出之后再做修正/校准。
- `unreported`：未说明。

> 判定原则（v4 新增，消除 v3 的 MLPINN/DUALSTREAM 等分歧）：
> - 如果它的效果是通过**损失函数的梯度**影响参数 → `training-loss`；
> - 如果它是**前向计算图里不可分割的一环**、去掉就不再有这个预测结果 → `inference-architecture`；
> - 如果它产出的是**额外训练数据**而非直接约束模型 → `data-generation`。

## D3. Inference requirement（推理期计算依赖）——**只此一问：推理时算先验/预测需要什么输入？**

- `none`：推理只需常规输入（已训练模型 + 已冻结先验 + 常规历史/实时观测量），
  不需要未来时步信息，也不需要训练集以外的新信息。
- `forecast-input`：需要**预报/NWP 输入**（未来时步、或独立气象预报产品）。
- `contemporaneous-state`：需要**实时 SCADA / 机型状态**（实时桨距角、叶尖速比、
  上游风机实测风速、实时限电状态等当前时刻可观测量）。
- `latent-state`：需要**不可直接观测的隐状态**（被限电/维护遮蔽的"本可发出功率"）。
- `unreported`：未说明。

> 判定铁律：`contemporaneous-state` 是"当前时刻**可观测**的实时量"；
> `latent-state` 是"当前时刻**不可直接观测**的隐变量"。可观测归前者，不可观测归后者。

## D4 / D5（v4 一并收紧措辞）

### D4. Implementation fidelity
- `code-aligned`：有官方代码且与论文公式一致
- `code-mismatch`：有官方代码但与论文公式不一致（附证据）
- `not-found`：**检索后未找到**官方代码（不作"绝对无代码"断言；`no-code` 旧写法废弃）
- `unverified`：未做代码检索核验

### D5. Harm reporting
- `none-reported`：未报告约束任何负面效应
- `ablation-absent`：未做约束有无消融
- `harm-quantified`：报告了约束在某些设定的负面结果
- `substitution-tested`：做了先验/oarcle 替换敏感性测试

---

## 与 v3 的差异（重编时只需注意这几点）

1. D1 删除 `forecast-derived`（用 intrinsic/historical-static 表达形态，预报依赖→D3）。
2. D2 增加三段判定原则（training-loss vs inference-architecture vs data-generation 的分界）。
3. D3 明确 contemporaneous-state（可观测）vs latent-state（不可观测）的边界。
4. D4 用 `not-found` 替代 `no-code`。
