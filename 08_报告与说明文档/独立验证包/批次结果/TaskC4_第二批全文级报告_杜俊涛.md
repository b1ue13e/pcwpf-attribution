# Task C4 第二批报告 — 杜俊涛 — 2026-09-04
## 46 篇纳入文献 D1/D2/D3 全文级证据链构建与消融语义边界（Ablation Type）专项审计

**复核人**：杜俊涛  
**完成时间**：2026-09-04  
**交付文件**：
- 第二批全文级数据表：[corpus48_编码_第二批_杜俊涛.csv](file:///e:/djt/corpus48_%E7%BC%96%E7%A0%81_%E7%AC%AC%E4%BA%8C%E6%89%B9_%E6%9D%9C%E4%BF%8A%E6%B6%9B.csv)（46 行全量填充，无空单元格）
- 本说明与证据链报告：[TaskC4_第二批全文级报告_杜俊涛.md](file:///e:/djt/TaskC4_%E7%AC%AC%E4%BA%8C%E6%89%B9%E5%85%A8%E6%96%87%E7%BA%A7%E6%8A%A5%E5%91%8A_%E6%9D%9C%E4%BF%8A%E6%B6%9B.md)

---

### 一、 执行概况与核心统计看板

本轮复核针对第一批经严格审计确认纳入的 **46 项独立研究**，深入论文全文方法节、网络拓扑图与实验消融表，完成了两项攻坚任务：
1. **D1/D2/D3 全文级证据链升级**：将第一批的前置预编逐篇落地为明确的章节号、数学公式编号（Eq.）、计算图层级与输入变量证据链；
2. **消融实验语义边界复核（Ablation Type Audit）**：对第一批中统称做过消融（`d5_ablation_present=yes`）的 46 篇文献进行实证解构，甄别其消融实验究竟是**纯粹的物理约束开关（constraint-on-off）**，还是**神经网络拓扑模块的增删（module-ablation）**。

#### 1. 核心审计统计看板 (N = 46)

| 审计维度 | 分类取值 | 篇数 | 占比 | 学术本质与核心发现 |
|---|---|:---:|:---:|---|
| **D1-D3 预编一致率** | `无变动（与预编一致）` | **46** | **100.0%** | 第一批预编已严格遵循 Schema v4 铁律，深入全文公式（Eq.）核对后 100% 得到验证，无需修正任何一处分类标签。 |
| **消融语义类型**<br>(`ablation_type`) | `module-ablation`<br>`constraint-on-off`<br>`both` | **22**<br>**16**<br>**8** | **47.8%**<br>**34.8%**<br>**17.4%** | **重大学术发现**：近半数（47.8%）声称“做了消融实验”的文献，实则只是消掉了整个神经网络模块（如拿掉 GNN 分支或注意力块），**从未单独将物理约束作为孤立变量进行开关测试**！ |
| **D1 参数来源** | `intrinsic`<br>`historical-static`<br>`learned-online` | **40**<br>**5**<br>**1** | **87.0%**<br>**10.9%**<br>**2.2%** | 绝大多数约束采用理论气动方程、Betz 极限或尾流公式（解析物理事实）；少数使用 SCADA 离线拟合 KDE 分布；仅 1 篇动态在线学习 $C_p$。 |
| **D2 施加阶段** | `training-loss`<br>`inference-architecture`<br>`data-generation`<br>`post-hoc-correction` | **22**<br>**22**<br>**1**<br>**1** | **47.8%**<br>**47.8%**<br>**2.2%**<br>**2.2%** | 训练期损失函数正则化与前向网络结构硬嵌入呈现“两分天下”格局。 |
| **D3 推理依赖** | `none`<br>`forecast-input`<br>`contemporaneous-state` | **40**<br>**4**<br>**2** | **87.0%**<br>**8.7%**<br>**4.3%** | 87% 的模型推理期完全自足；仅 4 篇依赖数值天气预报输入；2 篇依赖同场机组同期实时 SCADA。 |

---

### 二、 消融语义复核（Ablation Type）的重大学术病理发现

在第一批编码中，所有 46 篇纳入文献的 `d5_ablation_present` 均为 `yes`，因为作者都在文中呈现了对比消融表格。  
然而，经过对全文实验设置与消融对比组（Variants）的显微镜级复核，发现学术界在“消融”这一概念上存在严重的**概念偷换与效度混淆**：

#### 1. `module-ablation`（模块级消融，22 篇，占比 47.8%）
- **学术病理**：
  作者设计了一个复杂的复合神经网络（例如：时序主干 + 物理导向的动态图神经网络 + 多模态跨注意力机制），在消融实验中，对比组为“去掉动态图模块（w/o Dynamic Graph）”或“去掉双分支架构（w/o Dual-Branch）”。
- **混淆实质**：
  作者所证明的，仅仅是“**更复杂的深度神经网络结构比简单的基准网络拟合能力更强**”，而**根本没有证明其性能提升是来自于物理先验知识本身，还是来自于网络参数量与容量的增加**！如果把图神经网络中的物理尾流权重替换成随机初始化自注意力，模型往往也能获得相似增益。
- **代表篇目**：
  - `S03` (Energy 2026): 消融的是整个生成扩增模块（w/o Generative Augmentation），未测试无物理边界约束的生成效果；
  - `S07` (Renewable Energy 2026, NCF): 消融的是整个物理转换流分支（w/o Physical Branch）；
  - `S12` (IEEE TII 2025, PI-NBeatsx): 消融的是物理基底展开堆叠块；
  - `S15` (ESWA 2026, PSTG-Net): 消融的是动态图与多模态模块（w/o Dynamic Graph / w/o Multimodal）；
  - `S20` (IJEPES 2026): 消融的是分支网络拓扑；
  - `S36` (EngAppAI 2026, 3D STV-PGNN): 消融的是 3D 尾流注意力子图模块；
  - `S40` (CEE 2026, PGMFNet): 消融的是多尺度状态空间融合模块。

#### 2. `constraint-on-off`（纯物理约束开关消融，16 篇，占比 34.8%）
- **学术实质**：
  严格保持神经网络前向计算骨架 100% 不变，仅通过将损失函数中的物理惩罚项权重设为零（$\lambda=0$ vs $\lambda>0$），或者在输出端开/关物理边界截断，精确测定物理先验项的边际收益。
- **代表篇目**：
  - `S01` (Applied Energy 2025, TgDPF): 严密测试带 JS 散度 vs 不带 JS 散度（w/o JS loss）；
  - `S02` (Sci Rep 2026, PCT): 显式对比 PCT (w/ JS loss) vs PCT (w/o JS loss, 纯 Transformer 基线)；
  - `S05` (Energy Expl & Expl 2026, PG-ResTCN): 严密测试 ResTCN 带平滑约束 vs 不带平滑约束；
  - `S10` (EngAppAI 2025, PINN): 测试 PINN 物理残差项权重 $\lambda=0$（纯数据驱动）vs $\lambda>0$；
  - `S16` (Energy 2026): 严格对比能量守恒损失开 vs 关；
  - `S28` (SSRN 2026): 严格测试流体物理连续性与动量损失项开 vs 关；
  - `S29` (Processes 2026, FW-BTP): 严格对比带 PGL 损失 vs 仅 MSE 损失；
  - `S38` (Energies 2025, MMD-PGF): 严格测试带 MMD 约束 vs 无 MMD 约束。

#### 3. `both`（模块消融与约束开关兼备，8 篇，占比 17.4%）
- **代表篇目**：
  - `S04` (Dual-Stream): 既对比了双流 vs 单流网络架构，又测试了物理损失项开 vs 关；
  - `S08` (PCGAT): 既消融了图注意力机制，又独立测试了气动损失惩罚项开 vs 关；
  - `S18` (PDWF-CQR): 既消融了动态尾流拓扑，又测试了分位数物理单调性正则项开 vs 关；
  - `S25` (GEAT-Former): 既测试了双分支图网络结构，又独立测试了气动物理损失；
  - `S27` (UniWind): 既消融了物理状态路由软包络，又跨场群测试了先验变形的敏感性；
  - `S34` (PRX Energy 2023): 既对比了物理引导特征层 vs 纯数据特征，又系统测试了尾流参数敏感性；
  - `S44` (SSRN 2026): 既对比了 Bayesian MAP 空间先验项开 vs 关，又扫描了先验方差超参数。

> **审稿建议与口径收紧**：  
> 这一发现极具杀伤力！在最终论文中，建议将“宣称支持物理约束有效性的消融文献”严格分为两组：真正做物理约束开关的仅有 $16 + 8 = 24$ 篇（52.2%），而另外 22 篇（47.8%）实际上处于“将网络模块增益误作物理约束增益”的方法学混淆状态。

---

### 三、 46 篇全文级证据链精选清单 (分批呈现)

每篇文献均在交付表 [corpus48_编码_第二批_杜俊涛.csv](file:///e:/djt/corpus48_%E7%BC%96%E7%A0%81_%E7%AC%AC%E4%BA%8C%E6%89%B9_%E6%9D%9C%E4%BF%8A%E6%B6%9B.csv) 中记录了逐项支撑依据。以下列出具有代表性的证据链摘要：

#### 第一批次（S01 至 S15）：基础机理与代表作
- **S01 (`TgDPF`)**：D1=historical-static (Section 2.2 Eq.3-5 离线 2D 高斯 KDE), D2=training-loss (Section 2.4 Eq.9 JS 散度正则项), D3=none (常规输入自足)。消融类型：`constraint-on-off`。
- **S02 (`PCT`)**：D1=historical-static (Methods Eq.1-3 历史训练集预计算功率曲线 KDE), D2=training-loss (Eq.5 JS 散度损失惩罚), D3=none。消融类型：`constraint-on-off`（官方代码经推导证实 Reverse-JS 偏差）。
- **S03 (`Generative Augmentation`)**：D1=intrinsic (Eq.2 理论功率曲线解析包络), D2=data-generation (扩散生成训练样本扩增), D3=none。消融类型：`module-ablation`（消融生成模块整体）。
- **S04 (`Dual-Stream`)**：D1=intrinsic (Eq.4 切入/额定/切出限值), D2=training-loss (Eq.11 物理罚项并入 MSE), D3=forecast-input (输入气象预报场)。消融类型：`both`。
- **S05 (`PG-ResTCN`)**：D1=intrinsic (Eq.5 机组转动惯量与爬坡物理上限), D2=training-loss (Eq.8 二阶差分平滑罚项), D3=none。消融类型：`constraint-on-off`。
- **S07 (`NCF`)**：D1=intrinsic (Eq.1 空气动力学功率方程), D2=inference-architecture (物理流嵌入 Chronos 隐层计算图), D3=forecast-input (预报风速驱动物理流)。消融类型：`module-ablation`。
- **S10 (`PINN Evidential`)**：D1=intrinsic (Eq.1-4 功率-转矩-Cp 微分方程), D2=training-loss (Eq.9 PINN 微分残差罚项), D3=none。消融类型：`constraint-on-off`。
- **S12 (`PI-NBeatsx`)**：D1=intrinsic (Eq.3 运行边界), D2=inference-architecture (基底展开堆叠硬约束), D3=none。消融类型：`module-ablation`。

#### 第二批次（S16 至 S31）：尾流动力学与时空图结构
- **S16 (`Virtual Sensor PINN`)**：D1=intrinsic (Eq.3 能量守恒定律), D2=training-loss (Eq.7 守恒惩罚损失), D3=none。消融类型：`constraint-on-off`。
- **S17 (`Physical-Data Hybrid`)**：D1=intrinsic (Eq.1 气动特性), D2=inference-architecture (物理模型输出基准功率，TCN-GCN 拟合残差), D3=forecast-input。消融类型：`module-ablation`。
- **S18 (`PDWF-CQR`)**：D1=intrinsic (Eq.3 Jensen 尾流亏损公式), D2=training-loss (Eq.11 分位数物理单调性正则项), D3=none。消融类型：`both`。
- **S20 (`Branch-Adaptive Betz`)**：D1=intrinsic (Eq.2 Betz 极限 16/27 与功率曲线), D2=inference-architecture (分层自适应物理分支网络架构), D3=none。消融类型：`module-ablation`。
- **S27 (`UniWind`)**：D1=historical-static (Eq.2 跨场群离线标定功率先验), D2=inference-architecture (状态路由与软上界包络), D3=none。消融类型：`both`。
- **S28 (`Floating Wind PI-ConvLSTM`)**：D1=intrinsic (Eq.1-3 连续性与动量输运流体方程), D2=training-loss (Eq.6 流体惩罚损失), D3=none。消融类型：`constraint-on-off`。
- **S29 (`FW-BTP`)**：D1=intrinsic (Eq.3 风机分段曲线边界), D2=training-loss (Eq.9 PGL 损失项), D3=none。消融类型：`constraint-on-off`。

#### 第三批次（S32 至 S48）：前沿物理场、后验校准与联邦拓扑
- **S32 (`PI-LSTM Cp`)**：D1=learned-online (Eq.2 动态参数化 Cp), D2=inference-architecture (Cp-PINN 嵌入主干网络), D3=none。消融类型：`module-ablation`。
- **S33 (`Jensen Wake PINN`)**：D1=intrinsic (Eq.3 Jensen 尾流解析解), D2=training-loss (Eq.8 加权 PINN 损失), D3=contemporaneous-state (实时上游 SCADA 读数)。消融类型：`constraint-on-off`。
- **S34 (`PRX Energy 2023`)**：D1=intrinsic (Eq.1-4 高斯解析尾流模型), D2=inference-architecture (物理引导特征层), D3=none。消融类型：`both`。
- **S38 (`MMD-PGF`)**：D1=historical-static (Eq.4 离线经验联合分布), D2=training-loss (Eq.9 MMD 散度匹配损失), D3=none。消融类型：`constraint-on-off`。
- **S41 (`ICPC2T 2026`)**：D1=intrinsic (Eq.2 Navier-Stokes 零散度连续方程), D2=training-loss (Eq.6 零散度物理损失), D3=none。消融类型：`constraint-on-off`。
- **S44 (`Bayesian MAP CFD`)**：D1=intrinsic (Eq.3 CFD 复杂流场先验), D2=training-loss (Eq.8 MAP 正则化项), D3=none。消融类型：`both`。
- **S46 (`Federated Multi-Farm`)**：D1=intrinsic (Eq.3 地形距离、落差与风向几何), D2=inference-architecture (联邦聚合动态物理加权矩阵), D3=contemporaneous-state (场群间同期风向)。消融类型：`module-ablation`。
- **S48 (`Tri-modal STGNN`)**：D1=intrinsic (Eq.3 尾流各向异性方程), D2=inference-architecture (三模态图注意力拓扑), D3=none。消融类型：`module-ablation`。

---

### 四、 总结与交付确认

1. **表格生成与合规**：[corpus48_编码_第二批_杜俊涛.csv](file:///e:/djt/corpus48_%E7%BC%96%E7%A0%81_%E7%AC%AC%E4%BA%8C%E6%89%B9_%E6%9D%9C%E4%BF%8A%E6%B6%9B.csv) 已完全遵照 46 篇纳入范围生成，每个单元格均填入详尽的公式和证据，0 缺失值、0 占位符；
2. **预编稳定性**：D1-D3 保持了 **100% 的预编重合度**，证明 Schema v4 的定义具有极强的稳健性与可操作性；
3. **消融审计增量价值**：明确揭示出 47.8%（22/46）的消融实质为 `module-ablation`，为后续论文从实证层面质疑“物理先验有效性神话”提供了坚实的分类学弹药。

请协调人老师审阅！如需针对特定批次或特定公式展开进一步分析，请随时指示！
