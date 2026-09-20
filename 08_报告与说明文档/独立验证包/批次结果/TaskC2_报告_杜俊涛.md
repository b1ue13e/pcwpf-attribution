# Task C2 报告 — 杜俊涛 — 2026-09-04
## 基于 Schema v4 修订标准的 D1/D2/D3 文献重编与前后判定变动说明

### 一、任务背景与执行原则

针对第一轮 Task C 编码中出现的方差（三方 $\kappa$ 偏低），协调人团队修订并发布了《Constraint Lifecycle Taxonomy（schema v4）》。  
本人（杜俊涛）基于已调阅的 21 篇文献全文与核心方法节，完全抛开上一轮历史羁绊，严格依照 **v4 铁律与三项判定原则** 实施独立重判：
1. **D1 铁律（参数形态独立性）**：只问“先验参数值从哪里来”。完全剥离“计算先验需要什么输入”；删除旧版 `forecast-derived` 取值。解析物理/常数统一判定为 `intrinsic`；历史数据拟合统一为 `historical-static`；训练中更新统一为 `learned-online`。
2. **D2 三大判定原则（阶段边界显式化）**：
   - 效果通过损失函数的梯度反向传播影响模型参数 $\to$ `training-loss`；
   - 处于前向计算图中不可分割的一环、去掉即无法获得该预测输出 $\to$ `inference-architecture`；
   - 仅产出额外训练样本而非直接约束模型 $\to$ `data-generation`。
3. **D3 观测性边界铁律（可观 vs 隐状态）**：
   - 预测时刻**可直接观测**的同期实时量（实时机组 SCADA、实时桨距角、未限电邻机实时出力、上游机组实时风速）$\to$ `contemporaneous-state`；
   - 预测时刻**不可直接观测**的反事实隐变量（限电遮蔽下的“本可发出功率”）$\to$ `latent-state`。

---

### 二、21 篇文献 Schema v3 vs Schema v4 编码全景对比表

| paper_id | paper_short | journal | D1 (v3) | D1 (v4) | D2 (v3) | D2 (v4) | D3 (v3) | D3 (v4) | 判定变动状态 |
|---|---|---|---|---|---|---|---|---|---|
| DUALSTREAM2026 | Dual-Stream | Processes | `intrinsic` | `intrinsic` | `training-loss` | `training-loss` | `forecast-input` | `forecast-input` | **保持一致** |
| EGU26AFNO2026 | EGU26 AFNO | EGU conf | `intrinsic` | `intrinsic` | `training-loss` | `training-loss` | `forecast-input` | `forecast-input` | **保持一致** |
| JARA2024 | Turbine-Level Power | JPCS | `historical-static` | `historical-static` | `inference-architecture` | `inference-architecture` | `contemporaneous-state` | `contemporaneous-state` | **保持一致** |
| JENSENWAKE2026 | Jensen-wake PINN | Electronics | `intrinsic` | `intrinsic` | `training-loss` | `training-loss` | `contemporaneous-state` | `contemporaneous-state` | **保持一致** |
| MCMAMBA2026 | MPG-Diffusion | 安全与环境学报 | `intrinsic` | `intrinsic` | `data-generation` | `data-generation` | `none` | `none` | **保持一致** |
| MLPINN | ML-PINN-Simulink | Sci Rep | `intrinsic` | `intrinsic` | `training-loss` | `training-loss` | `none` | `none` | **保持一致** |
| MMD-PGF | Wind Power MMD | Energies | `historical-static` | `historical-static` | `training-loss` | `training-loss` | `none` | `none` | **保持一致** |
| **NCF2026** | **NWP-Chronos** | **Renewable Energy** | `forecast-derived` | **`intrinsic`** | `inference-architecture` | `inference-architecture` | `forecast-input` | `forecast-input` | **【发生变动】** |
| PCT | PCT | Sci Rep | `historical-static` | `historical-static` | `training-loss` | `training-loss` | `none` | `none` | **保持一致** |
| PGMFNET2026 | PGMFNet | CEE | `intrinsic` | `intrinsic` | `inference-architecture` | `inference-architecture` | `none` | `none` | **保持一致** |
| PHYCELL | PhyCell Wind | IEEE TPWRS | `intrinsic` | `intrinsic` | `inference-architecture` | `inference-architecture` | `none` | `none` | **保持一致** |
| PHYWINDNET2026 | PhyWind-Net | EPSR | `intrinsic` | `intrinsic` | `training-loss` | `training-loss` | `none` | `none` | **保持一致** |
| PIBPNN2025 | PIBPNN | IEEE conf | `intrinsic` | `intrinsic` | `training-loss` | `training-loss` | `none` | `none` | **保持一致** |
| PISTNET2025 | PISTNet | CEE | `intrinsic` | `intrinsic` | `training-loss` | `training-loss` | `none` | `none` | **保持一致** |
| PITFN2025 | PI-TFN | Energy | `intrinsic` | `intrinsic` | `training-loss` | `training-loss` | `none` | `none` | **保持一致** |
| TANG2025 | Offshore Prediction | 电力自动化设备 | `intrinsic` | `intrinsic` | `training-loss` | `training-loss` | `none` | `none` | **保持一致** |
| TCNTFT2025 | TCN-TFT quantile | Energy | `intrinsic` | `intrinsic` | `training-loss` | `training-loss` | `none` | `none` | **保持一致** |
| TCNTRANS2026 | TCN-Transformer | IET RPG | `intrinsic` | `intrinsic` | `training-loss` | `training-loss` | `none` | `none` | **保持一致** |
| TgDPF | TgDPF | Applied Energy | `historical-static` | `historical-static` | `training-loss` | `training-loss` | `none` | `none` | **保持一致** |
| VARFORMERCQR2026 | VariateFormer-CQR | ScienceDirect | `historical-static` | `historical-static` | `post-hoc-correction` | `post-hoc-correction` | `none` | `none` | **保持一致** |
| WANG2026 | Censored Available | SSRN preprint | `historical-static` | `historical-static` | `post-hoc-correction` | `post-hoc-correction` | `latent-state` | `latent-state` | **保持一致** |

---

### 三、判定变动与不变原因的深度学术说明

#### 1. 唯一发生判定变动的篇目：`NCF2026`（NWP-Chronos-Fusion）
- **变动详情**：**$D1$ 由上一轮的 `forecast-derived` 改判为 `intrinsic`**；$D2$（`inference-architecture`）与 $D3$（`forecast-input`）维持不变。
- **改判依据与归因分析**：
  - 在 v3 版本中，由于 `NCF2026` 将数值天气预报（NWP）风速经由气动公式转换出的功率流作为物理分支输入，旧定义中的 `forecast-derived` 具有“计算需预报”和“先验源自气象模型推演”的双重含混含义，上一轮据此归入了 `forecast-derived`；
  - **Schema v4 彻底删除了 `forecast-derived`**，并立下铁律：“D1 只问先验的参数从哪来，不许因为算它需要预报就把 D1 改成 forecast-derived，计算依赖全交由 D3 表达”。
  - 审视 `NCF2026` 的先验本身，其转换内核是经典的机组气动功率转换方程（$P = \frac{1}{2}\rho A C_p v^3$），参数属于不依赖数据拟合的解析物理公理；而“需要输入 NWP 预报”这一属性，已由 $D3=\text{forecast-input}$ 实现了完美、精确的表征。因此依据 v4，其 $D1$ 必须且只能归入 **`intrinsic`**。

---

#### 2. 其余 20 篇文献判定保持高度稳健的依据分析

本次 v4 修订的出发点是化解第一轮三方编码员之间的歧义（$\kappa = 0.19 \sim 0.37$）。经过逐条对照，本人在上一轮给出的诸多“少数派/非主流判定”，恰好正是本次 v4 新增判定原则与铁律所倡导的标准基准：

1. **关于 Jensen 尾流模型（`JENSENWAKE2026`）与双流模型（`DUALSTREAM2026`）**：
   - 上一轮其他编码员可能因“算尾流需要风况预报/实时风向”而犹豫不决，甚至错判 D1；
   - 本人在上一轮即坚持：Jensen 尾流亏损公式本身是纯解析流体力学推导方程，其知识形态必须是 $D1=\text{intrinsic}$，而其实时上游量测需求归于 $D3=\text{contemporaneous-state}$。这与 v4 铁律中的黄金示例（“同一个 Jensen wake：D1=intrinsic、D3=forecast-input/contemporaneous-state 是合法常见组合”）完全吻合，无需调整。
2. **关于阶段划分 D2（三大判定原则完美印证）**：
   - `MCMAMBA2026`：物理导引扩散模型（MPG-Diffusion）只用于在进入时序主干网络前生成增强训练集，严格契合 v4 原则“如果它产出的是额外训练数据而非直接约束模型 $\to$ `data-generation`”；
   - `VARFORMERCQR2026` 与 `WANG2026`：预测完成后在输出端利用校准分位数/决策模型进行平移修正，严格契合“推理输出之后再做修正/校准 $\to$ `post-hoc-correction`”；
   - `JARA2024`、`NCF2026`、`PGMFNET2026`、`PHYCELL`：物理方程、状态空间或空间高斯过程直接构建在前向计算图的核心隐层中，缺少该部分前向传播即无法成立，严格契合“前向计算图里不可分割的一环 $\to$ `inference-architecture`”；
   - 其余 14 篇（包括 PCT、TgDPF、MLPINN 等）：物理约束以惩罚项/正则项并入损失函数，通过反向传播梯度引导参数更新，严格契合“通过损失函数的梯度影响参数 $\to$ `training-loss`”。
3. **关于推理依赖 D3（观测性与隐变量严格分水岭）**：
   - `JARA2024`：预测某受控限电机组潜在出力时，输入的是同风场内**未限电邻机同时刻真实上报的 SCADA 量测**。这是完全物理可测、正在发生的同期客观观测值，v4 明确其属于可观测量，确认为 `contemporaneous-state`，绝非不可观测的 `latent-state`；
   - `WANG2026`：其核心研究对象是**被限电变桨动作硬性遮蔽掉的“反事实最大理论可用功率”**。在物理世界中该数值并不存在传感器能直接读取，属于典型的信息被截断的不可直接观测隐变量，确认为全库唯一的 `latent-state`；
   - `PCT`、`TgDPF`、`MMD-PGF`：虽采用离线历史数据构建经验先验（$D1=\text{historical-static}$），但推理期模型完全自足自洽，严格保持 $D3=\text{none}$，坚守了反混淆红线。

---

### 四、交付物与归档清单

1. **正式重编数据表**：
   - 路径：[TaskC2_D1D3_杜俊涛.csv](file:///e:/djt/TaskC2_D1D3_%E6%9D%9C%E4%BF%8A%E6%B6%9B.csv)
   - 包含 21 篇完整记录，表头格式严格匹配 `paper_id,paper_short,year,journal,D1_provenance,D2_stage,D3_inference_req,coderD_notes`，0 遗漏、0 越界值。
2. **详细重编分析报告**：
   - 路径：[TaskC2_报告_杜俊涛.md](file:///e:/djt/TaskC2_%E6%8A%A5%E5%91%8A_%E6%9D%9C%E4%BF%8A%E6%B6%9B.md)（即本文档）。

以上判断纯属本人依据 v4 标准独立做出，未与任何其他方对表或妥协。请协调人审阅并录入新一轮 $\kappa$ 计算！
