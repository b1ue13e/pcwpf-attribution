# TaskD 报告 — 杜俊涛 — 2026-09-04

## 逐篇判定

| 序号 | DOI | 标题 | 年份 | 判定 (decision) | 判定依据 (decision_basis) | 排除理由 (exclusion_reason) | 详细判定理由与具体技术依据 (notes) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 01 | `10.1016/j.apenergy.2025.125295` | Physics-constrained wind power forecasting aligned with probability distributions for nois | 2025 | **include** | `abstract-read` | `-` | Applied Energy 2025. Wind power forecasting incorporating domain physical knowledge (wind power curve) and probability distribution alignment to improve prediction accuracy and physical consistency under wind speed noise. |
| 02 | `10.2139/ssrn.4989270` | Physics-Constrained Wind Power Forecasting Aligned with Probability Distributions for Nois | 2024 | **include** | `abstract-read` | `-` | SSRN preprint of Applied Energy 2025 paper. Physics-constrained wind power forecasting using power curve constraints and probability distribution alignment. |
| 03 | `10.1038/s41598-025-34348-x` | Physics-constrained transformer for wind power forecasting | 2026 | **include** | `abstract-read` | `-` | Scientific Reports 2026. Proposes PCT (Physics-Constrained Transformer) embedding wind turbine power curve domain knowledge to constrain Transformer predictions to conform to physical power generation behavior. |
| 04 | `10.1016/j.energy.2026.141322` | Physics-constrained generative augmentation and hierarchical decomposition driven wind pow | 2026 | **include** | `abstract-read` | `-` | Energy 2026 (Nie et al.). Proposes physics-constrained generative augmentation adhering to wind turbine power laws combined with hierarchical signal decomposition for wind power forecasting under extreme weather. |
| 05 | `10.3390/pr14101671` | A Physics-Constrained Dual-Stream Dynamic Framework for Wind Power Forecasting Under Extre | 2026 | **include** | `abstract-read` | `-` | Processes 2026. Dual-stream architecture with Physics-Informed GRU (PI-GRU) and ERF bounded by physical power constraints and wind speed gradients for wind power forecasting under extreme weather. |
| 06 | `10.1177/01445987261480388` | A physics-constrained temporal modeling framework for robust short-term wind power forecas | 2026 | **include** | `abstract-read` | `-` | Energy Exploration & Exploitation 2026. Proposes PG-ResTCN incorporating a physics-based smoothness constraint into training loss enforcing turbine inertia and power ramping physics for short-term wind power forecasting. |
| 07 | `10.2139/ssrn.6837712` | Physics-Constrained and Generalization-Enhanced Industrial Large Model for Offshore Wind P | 2026 | **include** | `abstract-read` | `-` | SSRN 2026. Uses Chronos-2 foundation model with a physics-constrained penalty unit embedded into a hybrid loss function to enforce real-world physical power boundaries in offshore wind power forecasting. |
| 08 | `10.1016/j.renene.2026.125710` | NCF: A physics-constrained pre-trained method for short-term wind power forecasting | 2026 | **include** | `abstract-read` | `-` | Renewable Energy 2026 (Guo et al.). Proposes NCF pre-trained method embedding domain-specific physical constraints (theoretical power curve) for short-term wind power forecasting. |
| 09 | `10.1088/1361-6501/ae93ed` | Physics-constrained graph attention learning for wind turbine power forecasting from SCADA | 2026 | **include** | `abstract-read` | `-` | Measurement Science and Technology 2026. Proposes PCGAT using hybrid loss function incorporating aerodynamic constraints to ensure physical consistency in multi-horizon wind turbine power forecasting from SCADA data. |
| 10 | `10.1016/j.apenergy.2024.124068` | Physics-informed reinforcement learning for probabilistic wind power forecasting under ext | 2024 | **include** | `abstract-read` | `-` | Applied Energy 2024 (Liu et al.). Physics-informed reinforcement learning (PIRL) embedding physical operational constraints for probabilistic wind power forecasting under extreme meteorological events. |
| 11 | `10.1016/j.engappai.2025.113331` | Prediction of wind turbines power with physics-informed neural networks and evidential unc | 2025 | **include** | `abstract-read` | `-` | EngAppAI 2025. Physics-informed neural network for wind turbine power regression imposing physical governing equations (power, torque, power coefficient Cp) with evidential uncertainty estimation. |
| 12 | `10.32604/cmes.2023.027124` | Physics-Informed AI Surrogates for Day-Ahead Wind Power Probabilistic Forecasting with Inc | 2023 | **exclude** | `abstract-read` | `no-physics-constraint` | CMES 2023. Proposes GAN data augmentation and Extreme Learning Machine (ELM) surrogates for day-ahead wind power forecasting. Claims 'physics-informed' only by 'mining implicit physics information' from data, but lacks any explicit physical equations, power curves, loss regularizers, or boundary constraints. |
| 13 | `10.1109/tpwrs.2024.3405543` | Physics-Informed Learning Based Wind Farm Two-Machine Aggregation Model for Large-Scale Po | 2024 | **exclude** | `abstract-read` | `not-forecasting` | IEEE TPWRS 2024. Wind farm two-machine aggregation equivalent dynamic modeling via PDE-FIND for power system transient stability studies, not wind power forecasting. |
| 14 | `10.1016/j.energy.2026.141301` | A novel non-dimensional multiple operating conditions physics-informed neural networks for | 2026 | **include** | `abstract-read` | `-` | Energy 2026. PINN framework embedding non-dimensional Navier-Stokes/wake governing equations for wind turbine wake and power prediction in complex terrains. |
| 15 | `10.48550/arxiv.2307.14675` | Prediction of wind turbines power with physics-informed neural networks and evidential unc | 2023 | **include** | `abstract-read` | `-` | arXiv preprint of Record 11 (EngAppAI 2025). PINN for wind turbine power regression imposing physical governing equations (power, torque, Cp) and evidential uncertainty quantification. |
| 16 | `10.1109/tii.2025.3623559` | Physics-Informed Adaptive-Weight NBeatsx for Short-Term Wind Power Forecasting | 2025 | **include** | `abstract-read` | `-` | IEEE TII 2025. Operational physical constraints of wind turbines (cut-in, rated, cut-out wind speeds, rated power) explicitly embedded into NBeatsx architecture for short-term wind power forecasting. |
| 17 | `10.1016/j.ijepes.2025.111150` | High-precision short-term predicting method of wind farm power under extreme low temperatu | 2025 | **include** | `abstract-read` | `-` | IJEPES 2025. PINN based on SSVMD and Informer incorporating physical constraints of wind speed-power characteristics and low-temperature protection mechanisms for short-term wind power forecasting. |
| 18 | `10.1109/saupec65723.2025.10944420` | Physics-Informed Neural Networks for Enhanced Wind Power Forecasting: A Comparative Study | 2025 | **include** | `abstract-read` | `-` | IEEE SAUPEC 2025. Evaluates PINN integrating turbine physics with data-driven learning for wind power forecasting, especially during power ramp events. |
| 19 | `10.1016/j.eswa.2026.132224` | PSTG-Net: A physics-informed spatiotemporal multimodal dynamic graph network for wind powe | 2026 | **include** | `abstract-read` | `-` | ESWA 2026. Physics-informed spatiotemporal dynamic graph network integrating physical domain principles with multimodal graph modeling for wind power forecasting. |
| 20 | `10.1016/j.energy.2026.140691` | Physics informed virtual sensor modeling for wind power prediction | 2026 | **include** | `abstract-read` | `-` | Energy 2026. Integrates energy conservation law via energy-constrained loss functions and aerodynamic power conversion principles as virtual sensors for wind power prediction. |
| 21 | `10.1049/icp.2024.2343` | Physics informed data-driven approach for ultra-short term wind power forecasting based on | 2025 | **include** | `abstract-read` | `-` | IET 2025. Hybrid physics and data-driven approach: explicit physical model maps NWP wind speed to preliminary wind power, and TCN-GCN models the residual. |
| 22 | `10.3390/electronics14152985` | Dynamic Modeling and Online Updating of Full-Power Converter Wind Turbines Based on Physic | 2025 | **exclude** | `abstract-read` | `not-forecasting` | Electronics 2025. Dynamic equivalent modeling of full-power converter permanent magnet synchronous wind turbines via PINN for CloudPSS grid simulation, not wind power forecasting. |
| 23 | `10.1016/j.apenergy.2026.127849` | Online inertia estimation for wind-integrated power systems with ambient PMU measurements | 2026 | **exclude** | `abstract-read` | `maintenance-control` | Applied Energy 2026. Online inertia estimation for wind-integrated power systems using sliced parallel PINN with PMU measurements, not wind power forecasting (falls under inertia estimation / maintenance-control). |
| 24 | `10.1016/j.epsr.2026.113241` | Physics-informed dynamic wake graphs for probabilistic wind-farm power forecasting | 2026 | **include** | `abstract-read` | `-` | EPSR 2026. Proposes PDWF-CQR using physics-driven dynamic wake graph (mapping wake velocity deficits to edge weights) and physical-consistency regularization in quantile regression for probabilistic wind-farm power forecasting. |
| 25 | `10.1109/tia.2026.3653910` | Physics-Informed Adaptive Conformal Prediction for Wind Power Under Dynamic Weather Regime | 2026 | **include** | `abstract-read` | `-` | IEEE TIA 2026. Physics-informed adaptive conformal prediction incorporating turbine aerodynamic nonlinearities and physical constraints for wind power interval forecasting. |
| 26 | `10.1109/tpwrs.2025.3648770` | Graph Deviation Network With Physics-Informed Detection and Robust $H_{\infty }$ Control f | 2025 | **exclude** | `abstract-read` | `maintenance-control` | IEEE TPWRS 2025. Real-time cyberattack detection via PINN-GDN and robust H-infinity grid control for wind-integrated grids, not wind power forecasting. |
| 27 | `10.1016/j.ijepes.2026.111689` | A branch-adaptive framework for multi-timescale wind power forecasting: Optimization via h | 2026 | **include** | `abstract-read` | `-` | IJEPES 2026. Multi-timescale wind power forecasting explicitly embedding physical constraints including the Betz limit and turbine power curves into feature engineering. |
| 28 | `10.1016/j.enconman.2026.121974` | Physics-informed LLM-guided dynamic spatio-temporal graph deep network for short-term wind | 2026 | **include** | `abstract-read` | `-` | ECM 2026. Short-term wind farm power forecasting integrating physics-informed power-curve cleaning and LLM-guided dynamic spatio-temporal graph modeling wake interactions. |
| 29 | `(无DOI)` | Physics-informed neural network modelling of wind power production | 2026 | **include** | `abstract-read` | `-` | Huaqin Hu, LUT University 2026. PINN incorporating physical relationships (power vs wind speed, blade pitch angle, boundary constraints) as loss penalty terms for wind power prediction. |
| 30 | `10.2139/ssrn.6509860` | A Physics-informed Operating Condition Attention Model for Wind Power Forecasting | 2026 | **include** | `abstract-read` | `-` | SSRN 2026. Phy-OCA framework decoupling inputs into energy and efficiency branches based on aerodynamic power conversion mechanisms and wind speed-direction aerodynamic coupling for wind power forecasting. |
| 31 | `10.1109/repe66317.2025.11341663` | Physics-Informed Hybrid KNN-MLP for Enhanced Wind Power Forecasting | 2025 | **include** | `abstract-read` | `-` | IEEE REPE 2025. Hybrid KNN-MLP architecture incorporating physical constraints of turbine power curves and operating characteristics for wind power forecasting. |
| 32 | `10.1109/nnice68970.2026.11466169` | Physics-Informed Smart Maintenance for Offshore Wind Power: Prospects and Research Directi | 2026 | **exclude** | `abstract-read` | `maintenance-control` | IEEE NNICE 2026. Review of physics-informed smart maintenance and O&M strategies for offshore wind turbines, not a forecasting task. |
| 33 | `10.2139/ssrn.6598002` | Hybrid Physics-Informed Deep Learning Models for Wind Turbine Power Estimation | 2026 | **exclude** | `abstract-read` | `not-forecasting` | SSRN 2026. Proposes hybrid software sensors for real-time wind turbine power estimation and measurement error compensation for monitoring/performance assessment, not time-series forecasting. |
| 34 | `10.2139/ssrn.6221209` | Physics-informed dynamic wake graphs for probabilistic wind-farm power forecasting | 2026 | **include** | `abstract-read` | `-` | SSRN preprint of Record 24 (EPSR 2026). Probabilistic wind-farm power forecasting using physics-driven dynamic wake graphs and physical-consistency regularization. |
| 35 | `10.2139/ssrn.6553822` | Physics-Informed dual-branch graph neural network for wind farm cluster power forecasting | 2026 | **include** | `abstract-read` | `-` | SSRN 2026. GEAT-Former spatiotemporal graph network with dynamic wake modeling and aerodynamic physics-informed loss function coupling wind speed and power for wind farm cluster power forecasting. |
| 36 | `10.1016/j.epsr.2025.112707` | Coordinated power smoothing control for wind storage integrated system with physics-inform | 2026 | **exclude** | `abstract-read` | `maintenance-control` | EPSR 2026. Coordinated power smoothing control for wind storage systems using physics-informed reinforcement learning (PAMA-DDPG), not wind power forecasting. |
| 37 | `10.1016/j.energy.2026.142114` | Cross-scale regime-aware physics-informed forecasting for reliable wind-speed prediction i | 2026 | **include** | `abstract-read` | `-` | Energy 2026 (Shi et al.). Cross-scale regime-aware physics-informed forecasting for reliable wind speed prediction in risk-aware wind power operation. Meets criteria under wind speed forecasting with physics constraint (regime physics). Note: Candidate for marginal discussion as it forecasts wind speed rather than direct power. |
| 38 | `10.48550/arxiv.2607.01670` | UniWind: Toward Unified Day-Ahead Wind Power Forecasting via Physics-Informed State Routin | 2026 | **include** | `abstract-read` | `-` | arXiv:2607.01670. UniWind proposes physics-informed state routing combining a physical power curve prior with monotonic warping and a physical upper-bound soft envelope for day-ahead wind power forecasting. |
| 39 | `(无DOI)` | UniWind: Toward Unified Day-Ahead Wind Power Forecasting via Physics-Informed State Routin | 2026 | **include** | `abstract-read` | `-` | Duplicate record of Record 38 without DOI. UniWind day-ahead wind power forecasting via physics-informed state routing and physical upper-bound envelope. |
| 40 | `10.2139/ssrn.6170508` | Frequency Prediction Analysis of Wind Power System Based on Improved Physics-Informed Neur | 2026 | **exclude** | `abstract-read` | `not-forecasting` | SSRN 2026. Power system frequency (Hz) prediction using PINN incorporating power system dynamic swing equations, not wind power or wind speed forecasting. |
| 41 | `10.2139/ssrn.7239785` | Platform-Yaw-Aware Physics-Informed,Meta-Learning for Floating Wind Farm Wake and Power Pr | 2026 | **include** | `abstract-read` | `-` | SSRN 2026. Multistep floating wind-farm wake and power prediction using PI-ConvLSTM with fluid physics penalties on horizontal divergence and momentum transport. |
| 42 | `10.48550/arxiv.2412.17838` | Coordinated Power Smoothing Control for Wind Storage Integrated System with Physics-inform | 2024 | **exclude** | `abstract-read` | `maintenance-control` | arXiv preprint of Record 36 (EPSR 2026). Coordinated power smoothing control for wind-storage systems using PINN-assisted multi-agent DDPG, not wind power forecasting. |
| 43 | `10.1007/978-981-96-4067-6_15` | Active Power Optimization Considering Fatigue Damage in Wind Farms Using Physics-Informed | 2025 | **exclude** | `abstract-read` | `maintenance-control` | Springer LNEE 2025. Active power dispatch optimization and fatigue damage mitigation in wind farms using PINN, not wind power forecasting. |
| 44 | `10.1109/osmses69376.2026.11457195` | Physics-Informed Digital Twin for Pre-Deployment Validation of LSTM-Based Wind Power Forec | 2026 | **exclude** | `abstract-read` | `not-forecasting` | IEEE OSMSES 2026. Introduces a simulation engine/digital twin testing benchmark for pre-deployment validation of forecasting models, rather than developing a forecasting model. |
| 45 | `10.3390/pr14162560` | A Dual-Track Feature-Enhanced Physics-Informed Model for Accurate Wind Power Forecasting w | 2026 | **include** | `abstract-read` | `-` | Processes 2026. FW-BTP framework embedding wind turbine power curves into the loss function via Physics-Guided Loss Regularization (PGL) for wind power forecasting. |
| 46 | `10.2139/ssrn.6209521` | Simulation and Calculation of Wind Turbine Power under Wake Effects in Mountainous Wind Fa | 2026 | **exclude** | `abstract-read` | `not-forecasting` | SSRN 2026. Focuses on SCADA missing data completion/imputation and static power calculation using MP-PINN with Jensen wake constraints under active yaw control, not future power forecasting. |
| 47 | `10.32604/cmes.2026.081247` | Comparison of Physical, Gaussian Process, and Physics-Informed Gaussian Process Models for | 2026 | **exclude** | `abstract-read` | `not-forecasting` | CMES 2026. Compares models for static wind turbine power curve estimation (fitting Cp(lambda, beta)), which is a curve fitting task (explicitly listed under not-forecasting). |
| 48 | `10.2139/ssrn.6462606` | DSTC-Net: A Physics-Informed Dynamic Spatio-Temporal Coupling Network for Ultra-Short-Term | 2026 | **include** | `abstract-read` | `-` | SSRN 2026. DSTC-Net integrates physical factors into dynamic spatial correlation structures and adaptive feature fusion for ultra-short-term wind power forecasting. |
| 49 | `10.22541/au.177077715.51265060/v1` | Short-Term Solar-Wind Power Forecasting via Physics-Informed MISO MFAC with Dual-Source Ad | 2026 | **include** | `abstract-read` | `-` | Authorea 2026. Short-term solar and wind power forecasting via PI-MISO-MFAC with physics-constrained projection embedding renewable generation physical laws. |
| 50 | `10.1109/tce.2026.3690137` | An Input-Guided Physics-Informed Neural Network Method for Wind Power Estimation and Coord | 2026 | **exclude** | `abstract-read` | `not-forecasting` | IEEE TCE 2026. Focuses on false data injection attack (FDIA) detection and cyber-physical security monitoring using power estimation as a feature, not a forecasting task. |
| 51 | `10.1109/tce.2026.3670006` | Adaptive Physics-Informed Neural Network for Energy-Efficient Consumer Systems via Reliabl | 2026 | **include** | `abstract-read` | `-` | IEEE TCE 2026. PI-LSTM integrating turbine aerodynamic energy conversion principles and a Cp-PINN subnetwork for wind power forecasting under cold-wave conditions. |
| 52 | `10.1016/j.renene.2026.126199` | Optimization of wind-PV-storage aggregated inertia coefficient using physics-informed neur | 2026 | **exclude** | `abstract-read` | `maintenance-control` | Renewable Energy 2026. Optimization of aggregated inertia coefficients and frequency oscillation suppression using PINN, not wind power forecasting. |
| 53 | `10.3390/electronics15112338` | High-Penetration New Energy Power System Outage Loss Uncertainty Analysis-Oriented Ultra-S | 2026 | **include** | `abstract-read` | `-` | Electronics 2026. Ultra-short-term wind speed forecasting for wind farms integrating an LSTM with the Jensen wake model through a weighted PINN loss function. Included as it forecasts wind with explicit wake physics. |
| 54 | `10.1103/prxenergy.2.013009` | Physics-Guided Machine Learning for Wind-Farm Power Prediction: Toward Interpretability an | 2023 | **include** | `abstract-read` | `-` | PRX Energy 2023 (Zehtabiyan-Rezaie et al.). Physics-guided machine learning incorporating analytical Gaussian wake models and aerodynamic efficiency metrics for wind-farm power prediction across layouts. |
| 55 | `10.1016/j.engappai.2026.113798` | Development of a physics-guided bidirectional long short-term memory for wind power foreca | 2026 | **include** | `abstract-read` | `-` | EngAppAI 2026 (Sun et al.). Physics-guided BiLSTM embedding wind turbine power curve physical constraints and loss functions for wind power forecasting. |
| 56 | `10.1063/5.0223631` | Dynamic wake steering control for maximizing wind farm power based on a physics-guided neu | 2024 | **exclude** | `abstract-read` | `maintenance-control` | Physics of Fluids 2024. Dynamic wake steering yaw control optimization using model predictive control (MPC) to maximize farm power, not a wind power forecasting task. |
| 57 | `10.1016/j.engappai.2026.114301` | Spatiotemporal physics-guided graph neural network for wind farm power prediction | 2026 | **include** | `abstract-read` | `-` | EngAppAI 2026. 3D STV-PGNN incorporates a 3D wake analytical model to guide GNN architecture and attention edge weights for spatiotemporal wind farm power prediction. |
| 58 | `10.1016/j.epsr.2025.112527` | Regional wind power forecasting guided by physics-based wind field modeling and image fusi | 2025 | **include** | `abstract-read` | `-` | EPSR 2025. Regional wind power forecasting simulating wind evolution using fluid dynamics physical diffusion equations (PDEs) combined with Seq2Seq image prediction. |
| 59 | `10.3390/en18246479` | Wind Power Prediction Method Based on Physics-Guided Fusion and Distribution Constraints | 2025 | **include** | `abstract-read` | `-` | Energies 2025. Two-stage framework using power coefficient Cp fitting and Maximum Mean Discrepancy (MMD) distribution matching constraint on wind speed-power joint probability for wind power prediction. |
| 60 | `10.1016/j.renene.2025.124257` | A physics-guided deep learning methodology for wind farm power modeling under active yaw c | 2025 | **exclude** | `abstract-read` | `maintenance-control` | Renewable Energy 2025 (He et al.). Physics-guided deep learning power modeling under active yaw control for real-time wind farm flow control and yaw optimization, rather than time-series power forecasting. |
| 61 | `10.2139/ssrn.5622654` | Regional Wind Power Forecasting Guided by Physics-based Wind Field Modeling and Image Fusi | 2025 | **include** | `abstract-read` | `-` | SSRN preprint of Record 58 (EPSR 2025). Regional wind power forecasting using physical diffusion equations (PDEs) from fluid dynamics and image fusion. |
| 62 | `10.3390/jmse14141290` | Physics-Guided Deep Learning for Short-Term Offshore Wind Power Forecasting | 2026 | **include** | `abstract-read` | `-` | JMSE 2026. Hybrid LSTM-LightGBM bias correction integrated with physics-guided power conversion relationships for 24-h short-term offshore wind power forecasting. |
| 63 | `10.1016/j.compeleceng.2026.111367` | Physics-guided multiscale fusion network for ultra-short-term wind power forecasting | 2026 | **include** | `abstract-read` | `-` | Computers & Electrical Engineering 2026. PGMFNet incorporates physics-constrained loss derived from Jensen's wake theory for ultra-short-term wind power forecasting. |
| 64 | `10.1109/icpc2t68221.2026.11646324` | Physics-Guided Spatio-Temporal Transformers to Predict Urban Wind Fields with Accuracy and | 2026 | **include** | `abstract-read` | `-` | IEEE ICPC2T 2026. Predicts micro-scale urban wind fields using spatio-temporal Transformer constrained by incompressible Navier-Stokes continuity equations (divergence minimization) coupled to a turbine power model. |
| 65 | `10.1016/j.epsr.2026.112925` | A Physics-Guided Feature-Subspace ensemble framework for forecasting photovoltaic and wind | 2026 | **include** | `abstract-read` | `-` | EPSR 2026. Physics-Guided Feature-Subspace ensemble framework incorporating physical domain knowledge into feature subspaces for PV and wind power forecasting. |
| 66 | `10.1109/tia.2025.3625881` | Physics-Guided Short-Term Offshore Wind Power Forecasting Considering Wake and Blockage Ef | 2025 | **include** | `abstract-read` | `-` | IEEE TIA 2025. Short-term offshore wind power forecasting using coupled wake-blockage analytical model to derive structured physical inputs for conditional GAN. |
| 67 | `10.2139/ssrn.6906363` | Short-term wind power forecasting for data-limited wind farms in complex terrain: A physic | 2026 | **include** | `abstract-read` | `-` | SSRN 2026. Combines CFD-derived spatial priors with NWP and a physically constrained power curve model calibrated via MAP for short-term wind power forecasting in complex terrain. |
| 68 | `10.2139/ssrn.7346450` | Cross-Domain Few-Shot Wind Power Forecasting via Physics-Guided Retrieval-Enhanced Dual-Br | 2026 | **include** | `abstract-read` | `-` | SSRN 2026. Cross-domain few-shot wind power forecasting via hierarchical physics-guided retrieval combining embedding recall with empirical power-curve re-ranking. |
| 69 | `10.3390/su18147156` | Enhancing Wind Power Forecasting via Multi-Farm Coupling Under Data Isolation: A Physics-G | 2026 | **include** | `abstract-read` | `-` | Sustainability 2026. Physics-guided personalized federated approach using dynamic weight matrix from terrain distance, elevation, and real-time wind direction for multi-farm wind power forecasting. |
| 70 | `10.2139/ssrn.6462602` | Advancing short-term wind power forecasting via Physics-Guided Graph Neural Networks for s | 2026 | **include** | `abstract-read` | `-` | SSRN 2026. Spatiotemporal physics-guided graph neural network (PGNN) embedding wake-consistent interactions to super-resolve coarse NWP into turbine-scale inputs for multi-step wind power forecasting. |
| 71 | `10.1016/j.epsr.2026.113712` | Physics-guided tri-modal spatio-temporal graph neural networks for multi-step wind power f | 2026 | **include** | `abstract-read` | `-` | EPSR 2026. Tri-modal spatio-temporal graph neural network capturing wake-induced anisotropy and wake physics priors for multi-step wind power forecasting. |
| 72 | `10.1016/j.energy.2023.126908` | A novel data-driven deep learning approach for wind turbine power curve modeling | 2023 | **exclude** | `abstract-read` | `not-forecasting` | Energy 2023. Pure data-driven power curve modeling and outlier detection for wind turbines; static curve fitting rather than forecasting, and lacks physics constraints. |

---

## 汇总

- **检索命中总数**：72 篇
- **include: 53 篇**
- **exclude: 19 篇（按理由类别分布）**：
  - `not-forecasting`: 9 篇
  - `maintenance-control`: 9 篇
  - `no-physics-constraint`: 1 篇
  - `wind-speed-only`: 0 篇（注：具有显式物理约束的风速预测依指南纳入，见下节说明）
  - `not-wind`: 0 篇
  - `no-abstract`: 0 篇
- **need-abstract: 0 篇**
- **判定依据覆盖度 (decision_basis)**：72 篇均为 `abstract-read`（100% 达成摘要及核心方法研读）

### 排除篇目详细分类统计与技术剖析

1. **`not-forecasting`（非时间序列预测任务，共 9 篇）**：
   - 包含曲线拟合（curve fitting）、异常/网络攻击检测（anomaly/FDIA detection）、状态估计/软测量（state estimation/software sensors）、缺失数据插补（data imputation）、模型预部署测试基准等非预测任务：
     - Record 13 (`10.1109/tpwrs.2024.3405543`): 风电场双机等值动态暂态聚合建模，面向电网暂态稳定性分析，非时间序列预测。
     - Record 22 (`10.3390/electronics14152985`): 基于 PINN 的全功率变流器永磁风电机组仿真动态等值建模。
     - Record 33 (`10.2139/ssrn.6598002`): 混合物理引导深度学习风电机组功率软测量与状态估计，用于传感器误差补偿，非未来时步预测。
     - Record 40 (`10.2139/ssrn.6170508`): 电网频率（Hz）动态预测，非风电功率或风速预测。
     - Record 44 (`10.1109/osmses69376.2026.11457195`): LSTM 风电预测模型的数字孪生预部署测试引擎。
     - Record 46 (`10.2139/ssrn.6209521`): SCADA 缺失数据插补与稳态功率计算。
     - Record 47 (`10.32604/cmes.2026.081247`): 风电机组静态功率曲线拟合与估计（$C_p(\lambda, \beta)$ 拟合，指南明确定义为排除场景）。
     - Record 50 (`10.1109/tce.2026.3690137`): 基于物理信息神经网络的虚假数据注入攻击（FDIA）安全检测。
     - Record 72 (`10.1016/j.energy.2023.126908`): 纯数据驱动的风机功率曲线建模与异常值清洗，无物理先验且属静态拟合。

2. **`maintenance-control`（维护、控制、优化与惯量估计，共 9 篇）**：
   - 包含有功功率平滑控制（power smoothing control）、偏航控制优化（yaw steering MPC）、储能协调优化、惯量估计（inertia estimation）、智能运维综述等：
     - Record 23 (`10.1016/j.apenergy.2026.127849`): 基于 PMU 量测的风电并网系统在线惯量估计（指南明确点名“惯性估计等”归入此类）。
     - Record 26 (`10.1109/tpwrs.2025.3648770`): 抗网络攻击鲁棒 $H_\infty$ 电网控制。
     - Record 32 (`10.1109/nnice68970.2026.11466169`): 海上风电物理融合智能运维与状态检修前瞻综述。
     - Record 36 & 42 (`10.1016/j.epsr.2025.112707` / `arXiv:2412.17838`): 风储联合系统协调功率平滑控制（PAMA-DDPG 强化学习）。
     - Record 43 (`10.1007/978-981-96-4067-6_15`): 考虑疲劳损伤的风电场有功功率分配优化调度。
     - Record 52 (`10.1016/j.renene.2026.126199`): 风光储聚合惯量系数优化以抑制频率振荡。
     - Record 56 (`10.1063/5.0223631`): 基于物理引导神经网络动态尾流模型的动态偏航转向控制优化（MPC）。
     - Record 60 (`10.1016/j.renene.2025.124257`): 主动偏航控制下的风电场功率建模（旨在服务于实时流场偏航控制）。

3. **`no-physics-constraint`（纯数据驱动无显式物理约束，共 1 篇）**：
   - Record 12 (`10.32604/cmes.2023.027124`): 论文题为 *Physics-Informed AI Surrogates for Day-Ahead Wind Power Probabilistic Forecasting with Incomplete Data*。经研读摘要与全文方案，其核心算法为 GAN 数据生成结合极限学习机（ELM）进行日前概率预测，文中所谓“物理信息”仅宣称为“从数据中挖掘隐式物理信息（mining implicit physics information）”，并未建立任何显式物理微分方程、功率曲线解析式、Betz 极限边界或正则化损失项，属于伪装为物理约束的纯数据驱动方法，按规则严格排除。

---

## 你判为 include 但需要额外注意的篇目（可留空）

在判为 `include` 的 53 篇文献中，以下边缘篇目及复本特征需在后续系统综述与 meta-analysis 中重点关注：

1. **风速预测与功率预测边界交叉篇目**：
   - **Record 37** (`10.1016/j.energy.2026.142114`, Shi et al., *Energy* 2026): 任务输出为风速，但引入跨尺度物理天气模态先验（regime-aware physics-informed）服务于风电运行。依照指南“风电/风电场功率（或风速）预测 forecasting，并且使用了显式物理约束/先验”的标准纳入。若后续下游 meta-analysis 口径收窄为“仅限直接输出电功率”，此篇应优先复核。
   - **Record 53** (`10.3390/electronics15112338`, *Electronics* 2026): 预测超短期风速，但显式将 Jensen 尾流模型作为 PINN 加权损失项以约束上下游机组气动耦合，符合显式物理先验标准。
   - **Record 64** (`10.1109/icpc2t68221.2026.11646324`, *ICPC2T* 2026): 预测城市微尺度风场风速分布，网络中显式施加不可压缩 Navier-Stokes 连续性方程约束（散度极小化正则项），并最终耦合风机能量方程映射至发电功率。

2. **多能互补联合预测篇目**：
   - **Record 49** (`10.22541/au.177077715.51265060/v1`): 太阳能与风电联合短期功率预测，采用 PI-MISO-MFAC 嵌入非负性与物理输出投影约束。
   - **Record 65** (`10.1016/j.epsr.2026.112925`): 物理引导特征子空间集成框架同时预测光伏与风电功率。
   - 两者均包含实质性风电功率预测与物理先验，予以纳入。

3. **文献库中 6 组重复条目（Preprint / Publication Duplication）**：
   检索库中存在 6 组互为期刊正刊与 SSRN/arXiv 预印本的完全相同研究：
   - Record 01 (`Applied Energy 2025`) 与 Record 02 (`SSRN 2024`)
   - Record 11 (`EngAppAI 2025`) 与 Record 15 (`arXiv 2023`)
   - Record 24 (`EPSR 2026`) 与 Record 34 (`SSRN 2026`)
   - Record 36 (`EPSR 2026`) 与 Record 42 (`arXiv 2024`)（均被排除为 `maintenance-control`）
   - Record 38 (`arXiv UniWind`) 与 Record 39 (`无DOI UniWind`)
   - Record 58 (`EPSR 2025`) 与 Record 61 (`SSRN 2025`)
   建议在后续 PRISMA 流程图去重阶段（Deduplication）予以合并，独立研究净总数为 72 - 6 = 66 篇（净纳入 48 篇，净排除 18 篇）。

---

## 声明
我独立完成判定，未参考本项目任何既有筛选结果。

复核人：杜俊涛  
日期：2026-09-04
