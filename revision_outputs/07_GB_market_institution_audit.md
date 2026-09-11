# 07. 英国电力市场制度实态核查与建模定性报告 (GB Market Institution Audit)

**核查基准**：英国国家能源监管机构（Ofgem）、Elexon 平衡与结算规约（Balancing and Settlement Code, BSC）、P305 修正案（Elexon P305 Modification）、英国商业能源与工业战略部（BEIS）官方文献及电力市场权威学术研究。

---

## 一、 2017 年大不列颠（GB）电力市场真实制度结构

### 1. 核心制度误区澄清：GB 不是强制性“日前+实时二阶段集中市场”
- **常见误读**：将 GB 市场等同于美国 PJM、ERCOT 或 CAISO 模式，误以为存在一个由独立系统运营商（ISO）强制组织、统一集中出清的日前电能量现货市场（Mandatory Day-Ahead Centralized Pool）。
- **真实法律与制度事实**：
  - 英国电力市场自 2001 年实施《新电力交易制度》（NETA）并于 2005 年扩展为 BETTA 以来，其基石是**自愿性分散交易（Decentralized, Voluntary Trading）**。
  - 超过 90% 的电能量通过**双边场外交易（Bilateral Over-the-Counter, OTC Contracts）**、远期电力合约（Forward/Futures Contracts）以及自愿性电力交易所（Voluntary Power Exchanges，如 Nord Pool Spot GB 与 EPEX SPOT UK）完成实物交割约定。
  - 系统运营商（National Grid ESO）不运营集中的日前电能量拍卖池，而是在关闸（Gate Closure，交付前 1 小时）之后，通过**平衡机制（Balancing Mechanism, BM）**接受买卖申报（Bids & Offers）以维持电网秒级实时频率与传输断面平衡。

### 2. P305 修正案与单一不平衡价格机制（Post-P305 Single Cash-Out Price）
- **制度背景**：在 2015 年 11 月之前，英国 BSC 采用双重不平衡价格（Dual Cash-Out Pricing），即系统买价（System Buy Price, SBP）与系统卖价（System Sell Price, SSP）存在显著人为利差，对间歇性可再生能源造成严苛惩罚。
- **P305 重大改革生效（2015-11-05 起）**：
  - Ofgem 启动《电力平衡重大制度审查》（Electricity Balancing SCR），由 Elexon 正式执行 **BSC P305 修正案**。
  - **核心变革**：全面废止双重价格，确立**单一不平衡现金流价格（Single Cash-Out Price）**机制。在任一 30 分钟结算周期（Settlement Period, SP）内，无论发电机处于超发（Spill）还是欠发（Shortfall），均适用统一的系统价格（System Price）。
  - **价格形成机制**：价格平均参考量（Price Average Reference, PAR）由 500 MWh 阶梯式压缩至 PAR50（2015年），并在 2018 年进一步压缩至 PAR1，直接将系统实时不平衡价格绑定至系统边际平衡动作成本。
  - **2017 年价格特征**：2017 全年 30 分钟 SBP 均值为 **£44.27/MWh**，最高尖峰达到 **£1,509.80/MWh**，且存在 68 个负电价周期（最低至 **-£73.15/MWh**）。

### 3. BSC 30 分钟结算时段不平衡暴露建模（Half-Hour Imbalance Exposure Modeling）
- 依据 BSC 结算制度（Trading Arrangements）：
  Elexon 结算系统将市场主体的合同电量（Account Contracted Volume，通过合同电量通知 ECVN 申报）与其实际计量电量（Metered Volume, $y_{\text{SP}}$）进行代数比较，计算能量不平衡量（Energy Imbalance Volume）。
  **必须严格区分 ECVN / Contracted Position 与 FPN**：最终物理通知（Final Physical Notification, FPN）是发电机向系统运行方（National Grid ESO/NESO）提交的预期物理出力曲线，用于系统调度与平衡机制（BM）动作；而合同持仓量（Contracted Volume）是通过双边交易通知（ECVN）确立的财务/结算头寸。在论文的单机简化核算模型中，$q_{\text{SP}}$ 严格定义为前瞻合同/申报头寸（forward-contracted / nominated energy position），而非物理通知 FPN。
  $$\Delta E_{\text{SP}} = y_{\text{SP}} - q_{\text{SP}} \quad [\text{MWh}]$$
- **半小时时段暴露聚合（Half-Hour Aggregation）**：在建模实现中，将 30 分钟窗口内的 3 次 10 分钟 SCADA 功率累加转换为半小时能量暴露，以对应英国 30 分钟结算周期（Settlement Period）的单一出清价格。此为 SCADA 时间序列向市场结算周期的离散化建模映射，而非 BSC 官方直接规定累加 SCADA 脉冲。

---

## 二、 论文中两阶段收益公式的科学定性

### 1. 避免概念混淆的符号修改
原稿中使用的公式：
$$R_{\text{total}} = P_{\text{DA}} \cdot q + P_{\text{imb}} \cdot (y - q)$$
若不加说明，极易被能源与电力市场审稿人（Reviewer #2）一票否决，指责作者“根本不懂英国电力制度，竟把自愿性双边市场写成强制集中出清的 Day-Ahead ISO 市场”。

### 2. 规范定性与修改方案
- 将 $P_{\text{DA}}$ 统一修正为 **$P_{\text{F}}$** 或 **$P_{\text{forward}}$**；
- 在正文 Section 3.2 与 Section 5.5 明确写入：
  > “Equation \eqref{eq:two_settlement} represents a **stylized two-stage economic accounting decomposition** (or generic forward-plus-imbalance representation) widely adopted in operational literature \cite{pinson2007trading}, rather than a statutory mandatory centralized day-ahead auction clearing rule under the 2017 UK BETTA framework.”

---

## 三、 零申报（Zero-Bid）实验的严肃科学定位

### 1. 彻底清除“零申报套利获得 £243k 利润”的荒谬表述
- **原稿危险倾向**：部分措辞容易使读者误以为风电场只要“预测为0、申报为0”，就能在真实英国电力市场中躺赚 £223k--£243k/(turb$\cdot$yr) 的净利润。
- **制度事实**：
  1. 零申报（$q \equiv 0$）意味着发电机完全放弃了远期双边合约市场或日前电力交易所的保底电价收益（$P_{\text{forward}} \cdot q = 0$）；
  2. 发电机将 100% 的物理电量以完全未对冲（Unhedged）的裸暴露形态抛入实时平衡结算系统；
  3. 在实际商业运作中，发电机面临接入协议违约、电网不平衡调度责任、平衡责任主体（BRP）通道抽成、以及负电价时段倒贴赔钱的高额风险。

### 2. 严正确立其“归谬诊断（Reductio Ad Absurdum）”定位
- **诊断功能**：
  Zero-Bid 实验的唯一合法学术使命是——**充当一个极端的负对照探针（Adversarial Diagnostic Probe）**。
- **科学命题**：
  它以极端反例证明：**脱离远期合约对冲的孤立不平衡结算现金流（Standalone Imbalance Cashflow），根本不是衡量预测质量（Forecast Quality）的有效代理指标！**
- **严格约束的学术界线**：
  全文严格贯彻执行三大等式否定：
  $$\text{Imbalance Cashflow Component} \neq \text{Trading Profit}$$
  $$\text{Imbalance Cashflow Component} \neq \text{Total Commercial Revenue}$$
  $$\text{Imbalance Cashflow Component} \neq \text{Forecast Quality Metric}$$
