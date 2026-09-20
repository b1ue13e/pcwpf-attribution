# PCT 官方代码核查报告（contribution 3 前置门禁）

核查日期：2026-09-03
仓库：`https://github.com/daxin007/PCT-WPF`（clone 成功，2 commits，2025-08-09）
论文：Wang, Luo, Gao, Zhang, Chen. "Physics-constrained transformer for wind power
forecasting", Scientific Reports (2025).

## 1. 仓库结构

- `train_clean.py`（43748 bytes）：主训练脚本，含 LSTM 与 Transformer 网络、
  data loading、power-curve/JS 逻辑。
- `tst/`：Transformer 子模块（encoder/decoder/attention/positionwise/transformer）+ `loss.py`。
- `data_compare.py` / `data_statistics.py` / `summary.py`：数据预处理与画图。
- README 明确声明：**数据集需向 corresponding author 索取**
  （"The datasets used and/or analyzed during the current study available from the
  corresponding author on reasonable request."）。

## 2. 运行硬依赖（不可得，需作者数据）

- `江苏十分钟数据/*.csv`：25 台江苏风机 10min SCADA（GBK 编码，列含 时间/平均风速/功率）。
- `synthetic_wind_speed/task1.csv`：合成风速（U/V 分量合成，含 wind_speed 列）。
- `dataset2.pkl`：预处理后的缓存（可选，缺失时从 raw CSV 重建）。
- 结论：**官方代码缺数据集无法直接运行**；但代码本身完整，可用我们自己的
  5 站 15min 数据做**逐式 parity**（数据接口 `read_data`/`make_dataset` 清晰可替换）。

## 3. 核心发现：JS constraint 的数学实现与论文声明不符（数值影响经证伪）

> ⚠️ **本节结论已按 2026-09-03 数值 parity 结果修正**：初始静态代码读数标记为
> "P0 级发现"（independence-joint 破坏物理约束），经逐式 parity 数值验证后
> **证伪**——loss 偏移仅 ~12%、梯度方向几乎一致。教训见第 3.6 节。

### 3.1 论文声明

PCT/TgDPF 声称用 **Jensen-Shannon divergence** 度量「predicted wind power distribution」
与「wind power curve distribution」的差异，作为物理约束 loss。

### 3.2 代码实际实现（train_clean.py:297-341, 182-198）

```python
# marginal_pdf：对单个变量做 Gaussian-kernel 直方图
def marginal_pdf(values, bins, a, b=1e-10):
    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / a).pow(2))
    pdf = torch.mean(kernel_values, dim=1)   # 一维 marginal

# joint_pdf：配对样本 2D product-kernel KDE（已按 Task B 修正，勿再称"外积/独立"）
def joint_pdf(kernel_values1, kernel_values2):
    joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
    # = Σ_s φ(b_i; w_s)·φ(b_j; p_s)，标准配对 2D product-kernel；不等于边际乘积
    pdf = joint_kernel_values / torch.sum(joint_kernel_values, dim=(1,2)).view(-1,1,1)
```

在 `power_curve()` 中调用顺序：
```python
_, kernel_p = marginal_pdf(power,     bins_p, brandwidth_p)   # power 的样本×bin 核矩阵
_, kernel_w = marginal_pdf(windspeed, bins_w, brandwidth_w)   # wind 的样本×bin 核矩阵
pc = joint_pdf(kernel_p, kernel_w)                            # 配对 2D product-kernel KDE
```

### 3.3 静态读数（数学事实，经第三轮终审修正）

> ⚠️ **第三轮终审修正（2026-09-04）**：此前把 `joint_pdf` 的
> `torch.matmul(kv1^T, kv2)` 误判为"两个边际密度的乘积（独立性假设）"。
> 该判断**错误，已撤回**。正确数学事实如下：

`marginal_pdf` 返回的 `kernel_values` 形状是 `(1, n_samples, n_bins)`（样本×bin
矩阵），因此 `matmul(kv1^T, kv2)` = `(n_bins_p, n_samples)@(n_samples, n_bins_w)`
= `(n_bins_p, n_bins_w)`，其中

```
joint[b_i, b_j] ∝ Σ_s φ(b_i; w_s) · φ(b_j; p_s)
```

这是**在配对样本 s 上求和的标准 2D product-kernel（独立核）联合密度估计**，
**并非**两个边际密度的乘积，**不蕴含** power ⊥ wind 独立。它与论文声称的
standard 2D KDE **在数学上等价**（同为 product-kernel joint density）。

| 层面 | 论文声称 | 代码实际 | 静态性质（修正后）|
|---|---|---|---|
| joint 密度形式 | standard KDE on joint (wind, power) | **配对样本 2D product-kernel KDE**（Σ_s φ·φ）| ✅ **数学等价**（非"边际乘积"）|
| divergence | 标准 JS = ½[KL(p‖m)+KL(q‖m)] | `F.kl_div(p.log(), q)` → **reverse-JS-type** | ✅ 成立：**方向不同**（唯一剩余静态差异）|

**撤回**："outer-product 破坏物理依赖 / 独立性假设"——**作废，不再引用**。
**保留**：KL 方向差异（reverse-JS-type divergence）——这是唯一站得住的静态代码差异。

### 3.4 数值验证（作废，第三轮终审后）

> ⚠️ 下表是对比"official joint vs 真 2D joint KDE"，但第三轮终审已证明两者
> **数学等价**（都是配对样本 product-kernel），此对比**无意义，作废**。
> loss_ratio ~1.1 与 grad_cos ~0.99 恰恰印证了等价性（数值实现细节的小差异），
> 而非"独立性破坏"的证据。

| 站 | official_js | true_js | ratio | grad_cos |
|---|---|---|---|---|
| wind_cn | 2.06 | 1.83 | 1.12 | +0.998 |
| …（仅存档，不作为 claim）|

### 3.5 修正后的结论（第三轮终审定稿）

**"independence-joint / outer-product 不等价"指控已撤回。** 官方 `joint_pdf`
与论文 standard 2D KDE **数学等价**（配对样本 product-kernel）。

**唯一剩余静态差异**：`F.kl_div(log p, q) = KL(q‖p)`，即
`½[KL(m‖p)+KL(m‖q)]`，与标准 JS `½[KL(p‖m)+KL(q‖m)]` **方向不同**——
准确名称 **reverse-JS-type divergence**。

### 3.5b 非规范 KL 数值对照（2026-09-03，pct_kldiv_check.py）—— 实质发现

`F.kl_div(input, target)` = `Σ target·(log target − input)`。PCT 写的是
`F.kl_div(p.log(), q)` = **KL(q‖p)**（第四轮终审统一写法）。标准 JS =
`½[KL(p‖m)+KL(q‖m)]`，而 reverse-JS = `½[KL(m‖p)+KL(m‖q)]`。**方向不同**——
准确名称 reverse-JS-type divergence。

（5 站 loss 放大 6–11.5× 是此方向差异在 loss 层级的体现，**数值存档、不作结论**，
下游影响需独立验证。）

### 3.5c 全训练验证（作废，第二/三轮终审后）

> ⚠️ 下图 3.5c–3.5f 的训练数据全部基于**已作废的 runner**（含 target leakage、
> epochs 失效、伪 horizon、loss 尺度混淆），且第三轮终审后"official vs corrected"
> 的占比差异只来自 reverse-JS 方向（非 outer-product）。**此节 ratio 一律作废，
> 不再作为任何 claim，仅存档。**

**结论（第三轮终审定稿）**：唯一站得住的静态差异是 reverse-JS-type divergence
（`F.kl_div(log p, q)=KL(q‖p)`）。其下游 RMSE 影响**需独立验证**，本报告不再引用
任何 2.83–8.50× / 2–4× / 6–11.5× 数字作为结论。

v3 多站多 seed 复核（5 站 × 2 seed = 10 任务，10×4090 并行，各 30 epochs）：

### 3.5c–3.5d 全训练 ratio 表 —— 已作废并删除（第六轮治理）

> 原 3.5c/3.5d 含 "2.83–8.50× / 3.3–6.4× / 10/10 稳健" 等 ratio 数字，
> 均基于已作废 runner（target leakage / epochs 失效 / loss 尺度混淆 / 伪 horizon）。
> 按照第六轮"禁止用失效 runner 的 RMSE 比例支撑代码缺陷危害"，此处数字一律删除，
> 仅保留以下唯一结论：**官方 `F.kl_div(p.log(), q)=KL(q‖p)` 与论文标准 JS 方向不同
> （reverse-JS-type），下游数值幅度未经验证，不作任何性能 claim。**

### 3.5e 三版本对照（2026-09-03，pct_three_version.py，RNG 对齐后）

三版本定义（冻结，第三轮修正）：
- **paper-spec**：严格按论文公式（Eq.2 KDE + Eq.4-5 标准 JS）on 真 joint (wind,power)。
- **official-code**：忠实移植 daxin007/PCT-WPF：配对样本 2D product-kernel KDE
  （与 paper-spec joint **数学等价**）+ `F.kl_div(p.log(), q)`（reverse-JS）。
- **corrected-parity**：official-code 把 reverse-JS 修正为标准 JS → 对齐 paper-spec。

kelmarsh, seed=0, epochs=60（RNG per-version 对齐）：

```
official_code      = 0.85890
paper_spec         = 0.10727
corrected_parity   = 0.10727
```

**结论（第三轮作废 + 重新表述）**：
- 三个数值**作废**（基于已作废 runner：target leakage 等）。不再引用 8.0×。
- 可保留的静态事实：official-code 与 paper-spec 的**唯一差异是 reverse-JS 方向**；
  joint 密度两者数学等价（配对 product-kernel）。
- corrected-parity ≡ paper-spec 只证明"把 reverse-JS 改回标准 JS 后两臂一致"，
  是同一实现链的内部一致性，不是对外部作者结果的对齐。

### 3.5f Gradient-matched 对照（2026-09-04，响应终审 r4）

终审指出：reversed-KL vs corrected-JS 的固定 1:1 比较混入了 loss 尺度差异
（reversed KL 数值大 6-11.5×）。用 per-step 梯度范数匹配消除混淆：

kelmarsh, 5 seed, 40 epochs：

| 条件 | 相对差 vs 无约束 | t_p |
|---|---|---|
| reversed-KL（梯度匹配）| +201.2% | 0.004 |
| standard-JS（梯度匹配）| +47.2% | 0.008 |
| reversed-KL（raw 1:1，混淆对照）| +506.3% | 0.0001 |
| unconstrained | 基线 0.0780 | — |

**结论（作废存档，第三/四轮后）**：
1. 全部具体数字（+201%/+506%/2-4×）基于已作废 runner，**不再引用**。
2. 可保留方向：reverse-JS 与 standard-JS **方向不同**（静态数学事实），
   其下游数值影响需独立验证。

### 3.6 教训（固化，防止再犯）

**静态读代码发现"瑕疵" ≠ 有数值意义的发现。** 必须数值验证后才决定是否写进论文，
否则又是"实现 bug 当发现"的翻版。判定顺序固定为：
1. 静态事实（数学不等价）→ 只写"不等价"，不写"后果"；
2. 数值影响（loss/梯度/最终预测）→ 验证后才写"后果"与严重性等级。

## 4. 对审计论文的意义（修正版：不夸大）

原 story：审计「物理约束在部署时是否可观测」。
核查后 story（第三轮终审定稿，单一事实源）分两层：

1. (a) 约束先验的来源/生命周期/实现忠实度/伤害审计 —— 主审计维度（C1/C2）。
2. (b) PCT 唯一站得住的静态代码差异 = **reverse-JS-type divergence**
   （`F.kl_div(log p, q)=KL(q‖p)` 与标准 JS 方向不同）。joint 密度两者数学等价，
   **已撤回 independence-joint 指控**。

「忠实读官方代码」的价值仍在：reverse-JS 方向只有读代码才能发现；其论文价值取决于
下游数值影响，而该影响**需独立验证**。

## 5. 后续行动（已更新，第三轮后）

1. ~~逐式 parity~~ —— 作废（对比数学等价的两种 joint，无意义，见 §3.4）。
2. ~~kl_div 数值对照~~ —— 存档（6-11.5× 不再作为结论引用；需独立验证）。
3. ~~全训练确证~~ —— 作废（runner 含 target leakage/epochs/伪 horizon）。
4. **无 GPU submission freeze**（第四轮终审指示）：
   - 21 篇单一编码表；
   - PRISMA 89 条逐篇 ledger；
   - 21 篇逐篇 code/Data Availability 核验；
   - 清除所有失效数字与 independence 表述；
   - 由最终 CSV 自动生成统计。
5. **TgDPF**：无官方代码；论文公式正确；PCT reverse-JS 是 PCT 自身的实现差异，
   不能推断 TgDPF 有同缺陷（无代码可核验）。
