# PRISMA 检索日志第二轮（2026 前沿，2026-09-03）

## 检索记录

| # | query | 关键命中（新增） |
|---|---|---|
| 3 | physics-informed wind power forecasting 2026 neural network constraint loss | PhyWind-Net, PGMFNet, NCF, EGU26 AFNO, Dual-Stream, Jensen-wake PINN |
| 4 | physics-constrained probabilistic wind power forecasting uncertainty quantification constraint 2026 | PhyWind-Net（确认）, TCN-Transformer IET, PhysDiffWind, MCNN-Mamba+MPG-Diffusion, VariateFormer-CQR |

## 新增编码对象（9 篇，2026 前沿）

1. **PhyWind-Net** (Yu/Wei/Niu, Electric Power Systems Research, 2026) —— multi-component
   physics loss（difficulty-adaptive / temporal coherence / operational boundary /
   uncertainty regularization / NLL）+ EKF + PatchTST dual-head。
   **uncertainty-aware constraint 的官方锚点候选**（含 probabilistic head）。
2. **PGMFNet** (Guo et al., Computers & Electrical Engineering, 2026) —— Jensen wake
   physics constraint + SSM + MSFNet + CBAM，dynamic weighting。
3. **NCF** (Guo/Xiong/Zeng/Wang, Renewable Energy, 2026) —— NWP-Chronos-Fusion，
   theoretical power curve physics fusion + piecewise-weighted loss。
4. **EGU26 AFNO** (Cvečkovskis et al., EGU General Assembly, 2026) —— AFNO + physics-informed
   loss（penalised residuals of balance equations），conference。
5. **Dual-Stream** (Hao & Cao, Processes, 2026) —— physics-constrained dual-stream，
   extreme weather。
6. **Jensen-wake PINN** (MDPI Electronics, 2026) —— Jensen wake loss λ3=0.4（含明确权重实验）
7. **MCNN-Mamba + MPG-Diffusion** (刘青/张玉栋, 安全与环境学报, 2026) —— physics-guided
   diffusion + extreme conformal prediction，寒潮场景。
8. **TCN-Transformer IET** (IET Renewable Power Generation, 2026) —— TCN+Transformer+QR
   interval，physics-informed（PICP/PINAW 报告）。
9. **VariateFormer-CQR** (ScienceDirect, 2026) —— conformalized quantile，joint solar-wind
   （物理 motivated 跨变量 attention；可能排除：非 explicit constraint）。

## 关键锚点更新

- **PhyWind-Net 补上了不确定性感知（uncertainty-aware）约束的锚点**：
  operational boundary penalties（rated-capacity + mechanical inertia）+ NLL +
  uncertainty regularization，且有 probabilistic dual-head——这正是方法③
  （probabilistic/uncertainty-aware constraint）的官方代码候选（需核实 code）。
- **Jensen-wake PINN（λ3=0.4）** 有明确的物理权重实验（λ3 调优），是
  "physics weight 对结果影响"的可编码证据。
