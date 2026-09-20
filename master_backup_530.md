# 预测-决策解耦定理与残差失配形式化证明白皮书
## Formal Theory Whitepaper: Prediction-Decision Decoupling Theorem, Residual Mismatch, and Excess Operational Risk

> **执行状态**：法定理论基石（Authoritative Theoretical Foundation）· 冻结生效  
> **版本**：v1.0 (2026-09-08)  
> **关联法定资产**：
> - 论文正文：`08_报告与说明文档/MANUSCRIPT_AUDIT_DRAFT.md` §3.6
> - 主线索引：`08_报告与说明文档/主线状态索引_MASTER.md`
> - 实证数据源：`06_实验结果与日志/constraint_round2/analysis/` (`summary.csv`, `paired_cost_intervals.csv`, `dynamic_price_settlement_table2.csv`)
> - 外部验证集：Hill of Towie（48.3 MW, 21台机组, 7,599 样本, 45,594 步长）与 Kelmarsh（12.3 MW, 6台机组）

---

## 摘要与导言（Executive Summary）

在风电功率预测与下游电力系统运行调度的交叉研究中，将物理/先验约束直接作为正则项融入损失函数（Coupled Physics-Constrained Loss）已成为顶级文献的主流范式。此类研究普遍持有一个未经理论检视的底层信念：*“在深度神经网络的经验风险中施加物理或先验分布惩罚，能够自动提升预测的物理一致性，进而无损甚至单调改善下游电力市场出清与调度决策的经济收益。”*

本白皮书从严谨的数学运筹学、泛函分析与测度论凸占优（SOSD）视角出发，彻底推翻这一信念。我们形式化建立了**“预测-决策解耦定理”（Prediction-Decision Decoupling Theorem）**与**“残差失配推导”（Residual Mismatch Derivation）**，在数学上证明：
1. **运筹损失的两阶段三项分解（Theorem 1）**：任意点预测驱动的非对称出清损失可严格解构为**“预测均值表征误差（Term I, MSE）”**、**“决策分位数失配误差（Term II, Calibration Mismatch）”**与**“高阶残差形变项（Term III）”**。
2. **“数值反转”现象的普适机制**：在未校准时，物理/分布约束所展现的表面微弱优势，纯粹是因为不相容物理梯度在参数空间施加了粗糙的“附带收缩”，偶然部分抵消了高达数十倍的分位数失配（Term II）；一旦执行解耦的最优后验决策校准，Term II 被纯粹常数平移消除，耦合物理正则化对条件均值 $\mu(X)$ 造成的内在方差膨胀与残差形变（Term I 恶化）被彻底暴露，必然导致**性能排序的确定性反转（The Calibration Inversion）**。
3. **正则化失配超额风险下界定理（Theorem 2）**：在局部强凸与先验不相容条件下，耦合物理约束的超额运行风险存在严格正下界 $\Delta \mathcal{R}_{disp} \ge \mathcal{O}(\lambda^2 \|\mathbf{g}_{phys}\|^2) > 0$。
4. **时变电价推论（Corollary 1）**：在具有极端厚尾价格尖峰的真实电力市场中，重尾价格乘子将局部曲率权重放大数十倍，闭环解释了 Table 2 中 Kelmarsh 风场单机每年多支出 £1,324 英镑罚款的实质经济亏损。
5. **测度论无分布假定完备证明（Appendix）**：利用二阶随机占优（Second-Order Stochastic Dominance, SOSD），消除了有限阶 Taylor 展开的任何高阶截断漏洞，给出非参数分布无关（Distribution-free）的绝对数学证明。

---

# 第一部分：正文推导篇（运筹直觉与摄动分析流）

---

## 1. 问题形式化与统一数学符号系统

为保证理论体系与后续实证评估代码（`constraint_round2`）实现 100% 逐行符号对齐，统一数学符号规范定义如下：

### 1.1 时序与调度维度
- **历史回溯窗口长度**：$L = 36$（对应 10 分钟分辨率下的过去 6 小时 SCADA 观测序列）；
- **预测前向视野长度**：$H = 36$（对应未来 6 小时逐 10 分钟功率预测序列）；
- **下游现货调度出清窗口**：$H_{disp} = \{1, 2, 3, 4, 5, 6\}$（对应日前/日内调度中紧邻预测起报点的下一个整点交付小时，共 6 个 10 分钟时步）；
- **预测起报步长（Origin Stride）**：$S = 6$（每隔 1 小时滚动发布一次下一小时出清预报，确保交付时段非重叠且无重复计费）。

### 1.2 变量与预测器
- **输入特征张量**：$X \in \mathcal{X} \subset \mathbb{R}^{L \times d_{in}}$，其中 $d_{in} = 2$，包含标幺化历史实测功率与归一化实测风速；
- **真实功率标量/向量**：$Y = [Y_1, \dots, Y_H]^\top \in [0, 1]^H$，针对任一时步 $h \in H_{disp}$，真实功率标幺值为 $Y_h \in [0, 1]$；
- **点预测模型**：$f_\theta: \mathcal{X} \to [0, 1]^H$，由参数 $\theta \in \Theta$ 决定；
- **调度决策出清申报量**：$p_h = \mathcal{P}_{[0,1]}(f_{\theta, h}(X) + \delta_h)$，其中 $\mathcal{P}_{[0,1]}(u) \triangleq \min(\max(u, 0), 1)$ 为向物理可行域 $[0, 1]$ 的投影算子，$\delta_h \in \mathbb{R}$ 为决策校准位移量。

### 1.3 非对称运筹调度损失（Asymmetric Dispatch Loss）
在双边平衡结算市场（Newsvendor / Balancing Mechanism）中，调度出清的单步运行考核损失为：
$$\ell_r(p, Y) = c_+ \max(p - Y, 0) + c_- \max(Y - p, 0)$$
其中：
- $c_+$ 为履约短缺惩罚单价（Shortfall Penalty, 实际发电 $Y$ 低于申报 $p$，导致系统缺额需高价平衡采购或考核）；
- $c_-$ 为出力富余惩罚单价（Surplus Penalty, 实际发电 $Y$ 高于申报 $p$，导致系统弃风或下调考核）；
- 惩罚损失比率（Penalty Ratio）：$r \triangleq \frac{c_+}{c_-} \in (0, \infty)$；
- 临界分位数（Critical Fractile）：$\tau \triangleq \frac{1}{r+1} \in (0, 1)$。

在标幺化基准下，取 $c_- = 1, c_+ = r$。根据分位数检验函数（Pinball Loss）的标准定义 $\rho_\tau(u) \triangleq u(\tau - \mathbb{I}(u < 0))$，非对称运行损失在变量 $u \triangleq Y - p$ 下可严格恒等变形为：
$$\ell_r(p, Y) = (r+1) \cdot \rho_\tau(Y - p)$$
*代数严格验证*：
- 若 $Y < p$（履约短缺，Shortfall）：$u = Y - p < 0$，$\rho_\tau(u) = (\tau - 1)(Y - p) = (1 - \tau)(p - Y) = \left(1 - \frac{1}{r+1}\right)(p - Y) = \frac{r}{r+1}(p - Y)$，乘以 $(r+1)$ 恰为 $r (p - Y) = c_+ (p - Y)$；
- 若 $Y \ge p$（出力富余，Surplus）：$u = Y - p \ge 0$，$\rho_\tau(u) = \tau(Y - p) = \frac{1}{r+1}(Y - p)$，乘以 $(r+1)$ 恰为 $1 \cdot (Y - p) = c_- (Y - p)$。恒等式在全定义域内严格精确成立。

### 1.4 条件统计量与误差映射
- **条件期望（真实条件均值）**：$\mu(X) \triangleq \mathbb{E}[Y | X]$；
- **条件概率密度与分布函数**：$f_{Y|X}(y)$ 与 $F_{Y|X}(y) \triangleq \mathbb{P}(Y \le y | X)$；
- **条件最优 Bayes 分位数**：$q_\tau(X) \triangleq F_{Y|X}^{-1}(\tau)$；
- **均值点预测误差**：$\varepsilon_\mu(X) \triangleq f_\theta(X) - \mu(X)$；
- **分位数-均值内在结构偏置**：$\Delta_\tau(X) \triangleq q_\tau(X) - \mu(X)$；
- **局部曲率权重函数**：$w_\tau(X) \triangleq \frac{r+1}{2} f_{Y|X}(q_\tau(X))$。

---