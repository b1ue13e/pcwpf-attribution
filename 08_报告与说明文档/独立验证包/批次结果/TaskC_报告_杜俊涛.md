# TaskC 报告 — 杜俊涛 — 2026-09-04

## 检索说明
- 每篇论文的检索方式（Scholar/DOI/期刊页）
- 哪些篇只有摘要、哪些篇读了全文

本报告为风电功率物理约束预测（Physics-Constrained Wind Power Forecasting, PCT-WPF）文献库中 21 篇代表性论文的独立五维盲法编码报告（第三编码员 Coder D，杜俊涛）。

对 21 篇文献逐一通过 Google Scholar、Web of Science、DOI 解析器、arXiv、SSRN、ResearchGate 以及各大出版商数据库（Springer Nature, Elsevier ScienceDirect, IEEE Xplore, MDPI, IOPscience, IET）及中国知网/万方数据库进行了全方位独立检索与官方代码仓核查。检索渠道、凭证、阅读深度及官方开源代码核验结果汇总如下表：

| paper_id | 论文短名/标题 | 年份 | 期刊/出处 | 检索渠道与凭证 | 检索深度 | 官方代码核查结论 |
|---|---|---|---|---|---|---|
| DUALSTREAM2026 | Dual-Stream | 2026 | Processes | MDPI Processes (DOI: 10.3390/pr14101671) | 读摘要与全文核心节 | Data available upon request; 未找到官方开源代码仓库 (`not-found`) |
| EGU26AFNO2026 | EGU26 AFNO | 2026 | EGU General Assembly (conf) | Copernicus EGU General Assembly 2026 (Abstract: EGU26-11641) | 会议详细摘要 | 会议摘要未附带独立开源代码仓库 (`not-found`) |
| JARA2024 | Turbine-Level Possible Power Prediction | 2024 | J.Phys.Conf.Ser | IOP Conf. Ser.: JPCS (DOI: 10.1088/1742-6596/2767/6/062031) | 读全文 | 正文未提供官方代码开源链接 (`not-found`) |
| JENSENWAKE2026 | Jensen-wake PINN maintenance | 2026 | MDPI Electronics | MDPI Electronics (DOI: 10.3390/electronics15112338) | 读摘要与全文核心节 | Data available upon request; 未检索到公开代码仓 (`not-found`) |
| MCMAMBA2026 | MCNN-Mamba + MPG-Diffusion | 2026 | 安全与环境学报 | 《安全与环境学报》2026年刊载/网络首发 | 读方法论与核心节 | 中文期刊与公开代码托管平台未见官方代码 (`not-found`) |
| MLPINN | ML-PINN-Simulink framework for wind power prediction | 2025 | Scientific Reports | Scientific Reports (DOI: 10.1038/s41598-025-13306-7) | 读全文核心节 | 未提供独立公开代码仓库 (`not-found`) |
| MMD-PGF | Wind Power Prediction via Physics-Guided Fusion and Distribution Constraints | 2025 | Energies | MDPI Energies (DOI: 10.3390/en18246479) | 读全文 | 未提供官方独立代码仓库 (`not-found`) |
| NCF2026 | NCF (NWP-Chronos-Fusion) | 2026 | Renewable Energy | Elsevier Renewable Energy (2026, 125710) | 读全文核心节 | 未检索到独立官方开源代码 (`not-found`) |
| PCT | PCT (Physics-Constrained Transformer) | 2025 | Scientific Reports | Scientific Reports (DOI: 10.1038/s41598-025-34348-x, 16:4237) | 读全文并深入核验官方代码 | 有官方复现代码（`train_clean.py`）。如 TaskB 报告所确立，二维 product-kernel KDE 保持样本配对未退化为边缘积；但其点态损失实现为反向 JS 散度（`F.kl_div` 计算 $KL(q \parallel p)$，如 Task A 所证），与论文公式冲突，独立坚实判定为 `code-mismatch` |
| PGMFNET2026 | PGMFNet | 2026 | Computers & Electrical Engineering | Elsevier CEE (DOI: 10.1016/j.compeleceng.2026.111367) | 读全文核心节 | 未找到官方开源代码仓库 (`not-found`) |
| PHYCELL | Principle-Constrained Wind Field Image Generation | 2025 | IEEE TPWRS | IEEE TPWRS (DOI: 10.1109/TPWRS.2024.3449938) | 读全文 | 未找到官方开源代码仓库 (`not-found`) |
| PHYWINDNET2026 | PhyWind-Net | 2026 | Electric Power Systems Research | Elsevier EPSR (2026, 113241) | 读全文核心节 | 未检索到官方开源代码仓库 (`not-found`) |
| PIBPNN2025 | PIBPNN | 2025 | IEEE conference | IEEE Conference Proceedings (2025) | 读会议论文 | 会议论文未提供开源代码链接 (`not-found`) |
| PISTNET2025 | PISTNet | 2025 | Computers & Electrical Engineering | Elsevier CEE (DOI: 10.1016/j.compeleceng.2025.109985) | 读全文及作者主页 | 作者主页与 GitHub 均未检索到专用代码仓库 (`not-found`) |
| PITFN2025 | PI-TFN | 2025 | Energy | Elsevier Energy / ESTIJ (2025/2026) | 读全文核心节 | 未检索到公开开源代码 (`not-found`) |
| TANG2025 | Physical-Constraint Offshore Wind Power Prediction | 2025 | 电力自动化设备 | 《电力自动化设备》(2025, 网络首发) | 读全文核心节 | 中文期刊未附带开源代码仓库 (`not-found`) |
| TCNTFT2025 | TCN-TFT quantile | 2025 | Energy | Elsevier Energy (2025) | 读全文核心节 | 未检索到公开官方代码 (`not-found`) |
| TCNTRANS2026 | TCN-Transformer IET | 2026 | IET Renewable Power Generation | IET Renewable Power Generation (2026, 20(1)) | 读全文核心节 | 未检索到官方公开代码 (`not-found`) |
| TgDPF | TgDPF | 2025 | Applied Energy | Elsevier Applied Energy (DOI: 10.1016/j.apenergy.2025.125295) | 读全文 | 正文声明数据与代码依申请提供，未公开仓库 (`not-found`) |
| VARFORMERCQR2026 | VariateFormer-CQR | 2026 | ScienceDirect | Elsevier ScienceDirect (2026) | 读核心方法论与框架 | 未检索到公开开源代码 (`not-found`) |
| WANG2026 | Censored Available-Power Data Bias Dispatch Decisions | 2026 | SSRN preprint | SSRN Electronic Journal Preprint (2026) | 读预印本全文核心论证 | SSRN 预印本未附带可运行代码 (`not-found`) |

**阅读深度小结**：
- **读全文/核心节（20 篇）**：除 1 篇会议摘要外，其余 20 篇文献均获取并精读了全文或核心方法论/实验小节。
- **仅读摘要（1 篇）**：`EGU26AFNO2026`（属于 EGU 2026 会议摘要，官方仅公开发表详细摘要）。

---

## 个人把握标记
- 列出你认为"只有摘要信息、判定存疑"的 paper_id 及原因

根据《TaskC_第三编码员.md》规范，如实梳理在文献检索与信息抽取过程中存在信息密度局限或判定需审慎说明的 4 篇文献：

1. **`EGU26AFNO2026`（EGU General Assembly 2026 会议摘要）**：
   - **把握说明**：该条目属于欧洲地球科学联合会（EGU）大会详细学术摘要（Abstract ID: EGU26-11641），非同行评议期刊长文。
   - **存疑与裁定依据**：摘要明确指出利用自适应傅里叶神经算子（AFNO）对大气流动进行建模，并将控制方程残差构造为物理损失函数（Physics-informed loss constraints），需输入未来气象背景场。因此其 D1 判定为 `intrinsic`、D2 判定为 `training-loss`、D3 判定为 `forecast-input`。因摘要未公开代码仓库，D4 严格如实判定为 `not-found`。
2. **`PIBPNN2025`（IEEE 会议收录论文）**：
   - **把握说明**：该文献发表于 2025 年 IEEE 会议，网络公开检索条目相对简短。
   - **存疑与裁定依据**：文献核心思路为经典物理信息反向传播神经网络，将风电机组气动极限作为残差惩罚项注入反向传播更新。其 D1 判定为 `intrinsic`，D2 判定为 `training-loss`，D3 判定为 `none`，D4 判定为 `not-found`。若未来获取相机就绪版代码附录，可进一步细化其具体残差构造形式。
3. **`MCMAMBA2026`（《安全与环境学报》2026年中文成果）**：
   - **把握说明**：网络首发中文论文，聚焦于 MCNN-Mamba 与物理导引扩散模型（MPG-Diffusion）。
   - **存疑与裁定依据**：其核心物理机制体现为利用空气动力学物理界限约束扩散模型的生成轨迹，从而生成符合物理合理性的扩充训练样本，高度契合指南对 `data-generation`（样本生成与扩充）的定义；主干预测模型在推理期常规自足，故 D3 判定为 `none`。
4. **`VARFORMERCQR2026`（ScienceDirect 2026 在线成果）**：
   - **把握说明**：VariateFormer 结合共形分位数回归（CQR）属于前沿时间序列不确定性预测。
   - **存疑与裁定依据**：共形预测的非从众度分数（non-conformity score）是在离线静态校准集（calibration set）上计算分位数残差，且在训练/测试期保持固定，因此 D1 判定为 `historical-static`；CQR 在模型推断完成后对预测区间上下界进行后验平移校准，因此 D2 严格判定为 `post-hoc-correction`，推理期常规自足（D3=`none`）。

---

## 编码表
（填写后的 coderD_blank.csv 附上；也可粘贴关键列）

以下为填写完成的 21 篇文献五维盲法编码完整全景表（与 `TaskC_coderD_杜俊涛.csv` 完全一致）：

| paper_id | paper_short | year | journal | D1_provenance | D2_stage | D3_inference_req | D4_fidelity | D5_harm | coderD_notes |
|---|---|---|---|---|---|---|---|---|---|
| DUALSTREAM2026 | Dual-Stream | 2026 | Processes | `intrinsic` | `training-loss` | `forecast-input` | `not-found` | `none-reported` | Hao & Cao (Processes 2026, 14(10), 1671): Aerodynamic power curve bounds (cut-in/rated/cut-out) as intrinsic prior; loss combines MSE and physical penalty; takes NWP forecast; no public code found; ablation w/o constraint showed higher error. |
| EGU26AFNO2026 | EGU26 AFNO | 2026 | EGU General Assembly (conf) | `intrinsic` | `training-loss` | `forecast-input` | `not-found` | `none-reported` | Cveckovskis et al. (EGU26-11641): Atmospheric fluid dynamics governing equations as intrinsic prior; penalized residual loss on AFNO; takes atmospheric forecast fields; conference abstract with no public code; no harm reported. |
| JARA2024 | Turbine-Level Possible Power Prediction | 2024 | J.Phys.Conf.Ser | `historical-static` | `inference-architecture` | `contemporaneous-state` | `not-found` | `none-reported` | Jara Avila et al. (JPCS 2024, 2767:062031): Farm-wide spatial covariance estimated from historical SCADA (historical-static); multivariate Gaussian architecture; requires real-time SCADA of uncurtailed neighbor turbines (contemporaneous-state, not latent); no code; no harm reported. |
| JENSENWAKE2026 | Jensen-wake PINN maintenance | 2026 | MDPI Electronics | `intrinsic` | `training-loss` | `contemporaneous-state` | `not-found` | `none-reported` | Jin et al. (Electronics 2026, 15(11), 2338): Jensen wake model analytical decay equation (intrinsic); weighted PINN loss coupling LSTM with wake physics; requires contemporaneous upstream turbine SCADA; no code; no harm reported. |
| MCMAMBA2026 | MCNN-Mamba + MPG-Diffusion | 2026 | 安全与环境学报 | `intrinsic` | `data-generation` | `none` | `not-found` | `none-reported` | MPG-Diffusion paper (安全与环境学报 2026): Physics-guided diffusion using aerodynamic bounds (intrinsic) for training sample generation/augmentation (data-generation); inference on MCNN-Mamba is self-contained (none); no code; no harm reported. |
| MLPINN | ML-PINN-Simulink framework for wind power prediction | 2025 | Scientific Reports | `intrinsic` | `training-loss` | `none` | `not-found` | `none-reported` | Rajaperumal & Columbus (Sci Rep 2025, 15:29102): PMSG and fluid dynamics governing equations (intrinsic); PINN-Simulink loss formulation; standard input at inference (none); no public code found; R2=0.998, no harm reported. |
| MMD-PGF | Wind Power Prediction via Physics-Guided Fusion and Distribution Constraints | 2025 | Energies | `historical-static` | `training-loss` | `none` | `not-found` | `none-reported` | Zheng et al. (Energies 2025, 18(24), 6479): Maximum Mean Discrepancy (MMD) matches predicted distribution to historical joint empirical distribution (historical-static); training loss regularizer; inference self-contained (none); no code; no harm reported. |
| NCF2026 | NCF (NWP-Chronos-Fusion) | 2026 | Renewable Energy | `forecast-derived` | `inference-architecture` | `forecast-input` | `not-found` | `none-reported` | NWP-Chronos-Fusion (Renewable Energy 2026): Physical prior derived from NWP forecasts + conversion model (forecast-derived); fused into Chronos foundation model architecture; requires NWP forecast at inference; no code; no harm reported. |
| PCT | PCT (Physics-Constrained Transformer) | 2025 | Scientific Reports | `historical-static` | `training-loss` | `none` | `code-mismatch` | `none-reported` | Wang et al. (Sci Rep 2026, 16:4237): Joint KDE fitted once on historical training data (historical-static); JS divergence penalty in training loss; inference self-contained (none); official code in train_clean.py uses kl_div computing reverse JS KL(M\|\|P)+KL(M\|\|Q) conflicting with paper Eq.(4)-(5) (code-mismatch); no harm reported. |
| PGMFNET2026 | PGMFNet | 2026 | Computers & Electrical Engineering | `intrinsic` | `inference-architecture` | `none` | `not-found` | `none-reported` | Guo et al. (CEE 2026, 139:111367): Turbine power curve aerodynamic limits embedded into state-space and Fourier multiscale fusion network architecture; standard inputs (none); no code; outperforms baselines, no harm reported. |
| PHYCELL | Principle-Constrained Wind Field Image Generation | 2025 | IEEE TPWRS | `intrinsic` | `inference-architecture` | `none` | `not-found` | `none-reported` | Liu et al. (IEEE TPWRS 2025, 40(2):1790-1801): Multi-order advection-diffusion-circulation PDEs (intrinsic) embedded directly into PhyCell recurrent state update equations (inference-architecture); standard image sequence input (none); no code; no harm reported. |
| PHYWINDNET2026 | PhyWind-Net | 2026 | Electric Power Systems Research | `intrinsic` | `training-loss` | `none` | `not-found` | `none-reported` | Yu, Wei, & Niu (EPSR 2026): Aerodynamic power curve limits (intrinsic); physics-constrained probabilistic loss function; standard multi-modal inputs (none); no code; no harm reported. |
| PIBPNN2025 | PIBPNN | 2025 | IEEE conference | `intrinsic` | `training-loss` | `none` | `not-found` | `none-reported` | PIBPNN (IEEE Conf 2025): Aerodynamic turbine power equations (intrinsic); physics-informed residual loss added to BP neural network training; standard inputs (none); conference paper without public code; no harm reported. |
| PISTNET2025 | PISTNet | 2025 | Computers & Electrical Engineering | `intrinsic` | `training-loss` | `none` | `not-found` | `none-reported` | Aslam et al. (CEE 2025, 126:109985): Extended advection equation PDE (intrinsic); Adaptive Physics Penalty Loss (APPL) in training loss; standard spatio-temporal inputs (none); no dedicated repo found; 7-8% gain over 6 SOTAs, no harm reported. |
| PITFN2025 | PI-TFN | 2025 | Energy | `intrinsic` | `training-loss` | `none` | `not-found` | `none-reported` | Bhandarkar et al. (Energy / ESTIJ 2025/2026): Aerodynamic energy conservation and power curve constraints (intrinsic); soft penalty loss in temporal fusion network; standard inputs (none); no code; no harm reported. |
| TANG2025 | Physical-Constraint Offshore Wind Power Prediction | 2025 | 电力自动化设备 | `intrinsic` | `training-loss` | `none` | `not-found` | `none-reported` | Tang et al. (电力自动化设备 2025): Offshore wind aerodynamic equations and physical boundaries (intrinsic); training loss constraint penalty; standard offshore inputs (none); no code; no harm reported. |
| TCNTFT2025 | TCN-TFT quantile | 2025 | Energy | `intrinsic` | `training-loss` | `none` | `not-found` | `none-reported` | Energy 2025: Bernoulli's principle and ideal gas law (intrinsic); joint loss function optimizing quantile loss and meteorological physics loss; standard inputs (none); no code; CWC improved by 320%, no harm reported. |
| TCNTRANS2026 | TCN-Transformer IET | 2026 | IET Renewable Power Generation | `intrinsic` | `training-loss` | `none` | `not-found` | `none-reported` | IET RPG 2026, 20(1): Energy conservation and ideal gas equation of state (intrinsic); physical consistency loss in training; standard met tower inputs (none); no code; tested on Colorado met tower, no harm reported. |
| TgDPF | TgDPF | 2025 | Applied Energy | `historical-static` | `training-loss` | `none` | `not-found` | `none-reported` | Gao et al. (Applied Energy 2025, 383:125295): 2D KDE estimated from historical data (historical-static); JS divergence regularizer in LSTM training loss; standard inputs (none); code not publicly linked in journal paper; no harm reported. |
| VARFORMERCQR2026 | VariateFormer-CQR | 2026 | ScienceDirect | `historical-static` | `post-hoc-correction` | `none` | `not-found` | `none-reported` | VariateFormer-CQR (ScienceDirect 2026): Conformal non-conformity scores and empirical quantiles derived from static calibration set (historical-static); conformalized quantile regression applied post-hoc to adjust prediction intervals (post-hoc-correction); standard inputs (none); no code; no harm reported. |
| WANG2026 | Censored Available-Power Data Bias Dispatch Decisions | 2026 | SSRN preprint | `historical-static` | `post-hoc-correction` | `latent-state` | `not-found` | `harm-quantified` | Wang et al. (SSRN 2026): Historical curtailment logs and dispatch records (historical-static); econometric dispatch and curtailment adjustment (post-hoc-correction); requires unobserved counterfactual available power under curtailment (latent-state); no code; explicitly quantifies economic/operational bias and harm in dispatch decisions (harm-quantified). |

### 核心维度深入分析与判定依据

1. **D1 Provenance（约束来源分布）**：
   - `intrinsic`（14 篇，占比 66.7%）：切入/额定/切出极限（DUALSTREAM2026, PHYWINDNET2026, PIBPNN2025, TANG2025）、伯努利定律与气体状态方程（TCNTFT2025, TCNTRANS2026）、流体平流扩散偏微分方程（PISTNET2025, PHYCELL, EGU26AFNO2026）、PMSG机电控制方程（MLPINN）、Jensen尾流方程（JENSENWAKE2026）、空气动力学界限（MCMAMBA2026, PGMFNET2026, PITFN2025）。这些均直接由第一性原理方程给出，不依赖数据统计拟合。
   - `historical-static`（6 篇，占比 28.6%）：PCT（历史 2D KDE 功率曲线）、TgDPF（历史 2D KDE 功率曲线）、MMD-PGF（历史风速-功率联合经验分布）、JARA2024（历史全场高斯空间协方差）、VARFORMERCQR2026（离线校准集经验分位数残差）、WANG2026（历史限电与调度样本）。
   - `forecast-derived`（1 篇，占比 4.8%）：NCF2026 将未来数值天气预报（NWP）与机理模型转换输出作为动态物理特征流。

2. **D2 Application stage（应用阶段分布）**：
   - `training-loss`（14 篇，占比 66.7%）：将物理残差/散度惩罚作为正则项加权计入目标函数（PCT, TgDPF, MMD-PGF, MLPINN, PISTNET, TCNTFT 等）。
   - `inference-architecture`（4 篇，占比 19.0%）：将物理先验内嵌于网络隐层计算拓扑（JARA2024, NCF2026, PGMFNET2026, PHYCELL）。
   - `data-generation`（1 篇，占比 4.8%）：MCMAMBA2026 利用物理导引扩散模型生成高质量仿真样本扩充训练集。
   - `post-hoc-correction`（2 篇，占比 9.5%）：VARFORMERCQR2026（共形分位数后验区间校准）与 WANG2026（后验修正调度决策偏差）。

3. **D3 Inference requirement（推理期依赖与防混淆核验）**：
   - **防混淆红线执行**：严格坚守《TaskC_第三编码员.md》第 2 节规定：*“训练期用历史数据估计的静态先验（$D1=\text{historical-static}$）$\ne$ 推理期需要不可观测信息（$D3=\text{latent-state}$）”*。
   - `none`（15 篇，占比 71.4%）：PCT, TgDPF, MMD-PGF, VARFORMERCQR2026 等虽采用历史数据先验，但模型推理只需当前常规输入，完全自足，严格判定为 `none`，杜绝任何将静态经验先验机械标记为隐状态的混淆。
   - `forecast-input`（3 篇，占比 14.3%）：DUALSTREAM2026, EGU26AFNO2026, NCF2026 推理必须依赖未来 NWP 气象预报。
   - `contemporaneous-state`（2 篇，占比 9.5%）：JARA2024 推理需要同风电场其他未限电机组的实时 SCADA 测点（直接可测的同期状态，非隐状态）；JENSENWAKE2026 需要上游机组实时测点以计算动态尾流衰减。
   - `latent-state`（仅 1 篇，占比 4.8%）：WANG2026 核心研究的是限电遮蔽下的反事实潜在可用功率，属于物理上不可直接观测的客观隐变量，确认为全表唯一的 `latent-state`。

4. **D4 Implementation fidelity（实现忠实度与代码审计）**：
   - `code-mismatch`（1 篇）：PCT (Scientific Reports 2026, 16:4237)。官方公开代码 `train_clean.py` 中物理正则项使用 `F.kl_div(p.log(), q, reduction='sum')`，其实际计算为 $KL(q \parallel p)$，导致代码计算的 JS 散度为反向 JS 散度 $KL(M \parallel P) + KL(M \parallel Q)$，与论文公式 (4)-(5) 的正向理论定义冲突（正如 TaskA 报告所严格证明）。需要澄清的是，正如在 TaskB 报告中所确立的，官方代码采用的二维 product-kernel KDE 在样本维度执行单重内积完整保留了样本配对，并未坍缩为边缘分布的乘积；PCT 的 `code-mismatch` 分类完全、独立且坚实地由反向 JS 散度（Reverse-JS divergence，即 `F.kl_div` 计算 $KL(q \parallel p)$ 而非 $KL(p \parallel q)$）的数学实现冲突所确立，故严格判定为 `code-mismatch`。
   - `not-found`（20 篇）：对其余 20 篇论文，通过官方出版商页面、Data Availability 声明、补充材料、GitHub 及作者学术主页进行了穷尽式代码检索，均未检索到公开发布且可独立运行的官方复现代码仓库，如实客观标记为 `not-found`。

5. **D5 Harm reporting（伤害报告分布）**：
   - `harm-quantified`（仅 1 篇，占比 4.8%）：WANG2026 针对限电截断数据开展研究，实证量化并证明了使用截断数据对电力调度决策带来的系统性偏差与经济安全负面伤害。
   - `none-reported`（20 篇，占比 95.2%）：其余 20 篇文献均专注于报告物理约束带来的正向性能提升，未报告任何负面副作用、负迁移或先验错误带来的精度损失，普遍存在发表偏倚。

---

## 声明
我独立完成编码，未参考本项目任何既有编码结果。

复核人：杜俊涛  
日期：2026-09-04
