# 论文命题-证据对齐矩阵与反事实威胁审查表
# (Claim-Evidence Alignment Matrix & Counterfactual Threat Audit)

> **版本**：v2.0 第一性原理科研叙事重构版（First-Principles Narrative Reconstruction Edition）  
> **定位**：严格规范论文中每一个科学主张（Claim）的事实依据、反事实竞争解释、已有受控实验、潜在审稿人反击威胁与法定许可学术措辞。严禁编造任何数据与超范围推断。

---

## 一、主张-证据穿透总表（The Master Claim-Evidence Matrix）

| 编号 | 科学主张（Claim） | 实证数据源与确切数值证据 | 反事实竞争解释 (Counterfactual) | 本文已有的受控实验 (Control) | 残余审稿人威胁 (Remaining Threat) | 法定许可学术措辞 (Permitted Wording) |
|:---|:---|:---|:---|:---|:---|:---|
| **CLM-01**<br>*(RQ1: 先验特异性)* | 在低样本与强约束下，真实气动功率曲线先验显著降低非对称调度损失，而打乱先验导致损失严重膨胀。 | **Penmanshiel** ($N=1,583, \lambda=1.0$):<br>• 无先验: $0.09881 \pm 0.00027$<br>• 真实先验: $0.09760 \pm 0.00044$ ($-1.22\%$)<br>• 伪打乱先验: $0.16611 \pm 0.00335$ ($+68.1\%$)<br>*(见 Table 1, Fig. 1a, `ablation_3d_key_comparison_table.csv`)* | “提升并非来自物理气动规律，而是额外的损失惩罚项对模型施加了通用的数值平滑或正则化。” | **Form- and weight-matched pseudo-scrambled prior**: 网络结构、划分、种子、超参 $\lambda=1.0$ 100% 相同，仅沿风速轴置换功率映射。伪先验损失暴增 68.1%，证伪平滑假说。 | 审稿人可能质疑 1.22% 幅度较小。 | *"Under sample-restricted training ($N=1{,}583$) and strong regularization ($\lambda=1.0$), embedding an authentic empirical power curve reduces normalized dispatch loss by 1.22\%, whereas permuting the aerodynamic mapping inflates loss by 68.1\%, demonstrating prior domain specificity."* |
| **CLM-02**<br>*(RQ1: 弱惩罚数值拟态)* | 在弱约束（$\lambda=0.01$）下，真实先验与伪打乱先验表现完全重合，属于非特异性数值平滑。 | **Kelmarsh** ($\lambda=0.01$):<br>• 真实先验: $0.08810 \pm 0.00009$<br>• 伪打乱先验: $0.08801 \pm 0.00008$<br>**Hill of Towie** ($\lambda=0.01$):<br>• 真实: $0.08922 \pm 0.00005$<br>• 打乱: $0.08919 \pm 0.00006$<br>*(均值差异 $<0.0001$，误差棒重合)* | “文献中大量宣称微弱物理损失（$\lambda \le 0.01$）带来的微小提升证明了深度网络理解了物理定律。” | **$\lambda$ 敏感性网格扫描**: 在 $\lambda=0.01$ 时，随机打乱的先验完全复现了真实先验的表现，直接推翻了“微弱物理损失代表物理推理”的推断。 | 审稿人可能辩解某些特定网络在弱 $\lambda$ 下有特异性。 | *"Under weak penalty scaling ($\lambda=0.01$), genuine and pseudo-scrambled priors yield statistically indistinguishable dispatch costs ($0.08810$ vs.\ $0.08801$), behaving empirically as non-specific numerical smoothing rather than domain reasoning."* |
| **CLM-03**<br>*(RQ2: 样本量有效边界)* | 物理先验的决策价值随训练样本量扩大而衰退甚至反转；小样本下方差削减占优，大样本下结构偏置主导。 | **跨 2 风场 $\times$ 5 样本量学习曲线** ($N \in [500, 5000]$):<br>• $N=500$: 真实先验均占优 (Kelmarsh $-2.00\%$, Hill $-3.62\%$);<br>• $N=5{,}000$: 无约束纯数据模型反超 (Kelmarsh $+0.33\%$, Hill $+1.73\%$ worse for prior);<br>• 伪先验在所有 $N$ 下暴增 $+43\%$ 到 $+98\%$.<br>*(见 `within_site_learning_curve_summary.csv`)* | “大样本下物理先验依然必然提升精度，现有差异只是跨风场地形造成的偶发现象。” | **Within-site sample-size learning curves**: 在相同风场内人为控制训练样本量 $N$，彻底解耦了样本规模与地形因果，观察到单调优势衰减与交叉反转。 | 审稿人可能质疑大样本下未对 $\lambda$ 进行退火（Annealing）。 | *"Across within-site learning curves, authentic priors yield cost reductions under restricted windows ($N \le 1{,}000$), but this advantage diminishes and slightly reverses as training volume expands to $N=5{,}000$, consistent with a transition from variance reduction to structural bias dominance."* |
| **CLM-04**<br>*(RQ2: 信息预算解耦)* | 小样本下真实先验的收益主要来自先验本身的物理代数几何，而非偷看全历史数据的额外季节性。 | **严格 500 样本预算对齐实验** ($N_{\text{prior}}=N_{\text{train}}=500$):<br>• Kelmarsh 预算对齐真实先验保留了全历史先验 **$77\%$** 的收益 ($-1.54\%$ vs $-2.00\%$);<br>• Hill of Towie 保留了 **$92\%$** 的收益 ($-3.34\%$ vs $-3.62\%$);<br>• 配对 $t$ 检验 $95\%$ CI 显著异于 0 ($p < 0.01$);<br>• 预算对齐打乱先验依然暴增 $+43.3\%$ 与 $+46.0\%$. | “先验之所以在小样本下有用，纯粹是因为拟合先验时偷看了 7--10 个月的额外历史季节性 SCADA 数据。” | **Strict Information-Budget Alignment**: 将先验的拟合样本严格限制在与模型训练完全相同的 500 个样本内（约 21 天），证明纯样本内拟合的几何先验仍能保留绝大部分收益。 | 历史全量先验依然提供了少量边际增量（$-0.46\%$ 和 $-0.28\%$）。 | *"Evaluating budget-matched priors fitted strictly on the identical 500 training samples confirms that domain-structured empirical prior effects retain 77\% to 92\% of the full-history benefit, refuting the conjecture of pure seasonal leakage."* |
| **CLM-05**<br>*(RQ2: 先验连续失配与工程漂移)* | 物理先验对失配极为敏感；连续打乱在 $\alpha \approx 0.07$ 即穿透基线，风机老化超过 10% 转化为经济惩罚。 | **Penmanshiel 连续漂移实验**:<br>• $\alpha=0.0 \to 0.1$: 成本由 $0.09712$ 升至 $0.09958$ (在 $\alpha \approx 0.07$ 穿透无约束 $0.09881$);<br>• $\alpha=0.2$: 损失恶化 $+5.9\%$;<br>• 风机容量降额 $>10\%$ 即导致先验劣于无约束基线;<br>• 负向测风仪漂移迅速激增短缺罚款。<br>*(见 `corruption_severity_summary.csv`)* | “物理约束即使不精确，也能提供定性约束，不会对模型造成灾难性后果。” | **Continuous perturbation & drift scans**: 采用连续插值参数 $\alpha \in [0, 1]$、额定容量降额与测风漂移，量化勾勒出物理先验的“有效容差窗口”。 | 真实风电场老化过程是动态多维的，单参数降额为格式化近似。 | *"Evaluating continuous prior corruption reveals an empirical breakdown crossover at $\alpha \approx 0.07$, while aerodynamic derating exceeding 10\% flips physical constraints into net economic penalties."* |
| **CLM-06**<br>*(RQ2: 气象代理时间错位)* | 离线训练损失中依赖滞后历史测风代理（$k=6 \to 36$ 步）会导致下游调度决策损失单调恶化。 | **三风场滞后扫描 ($k \in \{6, 12, 18, 36\}$ 步)**:<br>• Hill of Towie: 成本从 $0.09022$ ($k=6$) 单调升至 $0.09257$ ($k=36$), 在 $k \approx 12$ 穿透基线 $0.08924$;<br>• Kelmarsh: $0.08876 \to 0.09043$;<br>• Penmanshiel: $0.10036 \to 0.10473$.<br>*(见 Table 1, Fig. 1c)* | “审稿人若误以为这是推理阶段的数据延迟（Inference Latency），会质疑模型输入特征工程。” | **明确界定为训练损失代理滞后（$\mathcal{D}_3=\texttt{none}$）**: 模型推理时只用历史功率，实验隔离的是训练目标中代理状态与预测时刻的时间错位。 | 真实工业场景若提供高精度超短期 NWP，代理滞后问题可能减轻。 | *"In the absence of forward meteorological forecasts, regularizing training objectives against lagged historical weather proxies ($k > 12$ steps) monotonically degrades downstream dispatch performance across all tested sites."* |
| **CLM-07**<br>*(RQ3: 决策风险与对称点误差解耦)* | 对称回归指标（RMSE/MAE）无法反映非对称电网运行决策风险；RMSE 改善的模型在调度中可能表现更差。 | **Kelmarsh 与 Hill of Towie 对照**:<br>• Kelmarsh Info-Symmetric (Power + Wind): RMSE 改善 $1.16\%$ ($0.1697$ vs $0.1717$), 但调度成本恶化 $+1.11\%$ ($0.08910$ vs $0.08812$);<br>• Hill of Towie Info-Symmetric: RMSE 改善 ($0.1733$ vs $0.1740$), 但调度成本恶化 ($0.09042$ vs $0.08924$);<br>• Kelmarsh $\lambda=1.0$: RMSE 最优 ($0.1685$), 调度成本反弹 ($0.08828$). | “点预测误差越小，下游所有商业和电网调度指标必然单调改善。” | **Dual-Metric Evaluation**: 同步输出 RMSE 与非对称调度损失（$r=2, q=1/3$），直接捕获到 RMSE 降低而决策成本上升的实例。 | 理论反例 2（空间异方差）进一步提供了数学存在性证明。 | *"Symmetric regression metrics decouple from asymmetric operational risk: introducing input features or constraints that reduce RMSE can simultaneously elevate tail shortfall and dispatch penalties."* |
| **CLM-08**<br>*(RQ3: 单边市场现金流病态归谬)* | 脱离日前合约的单边平衡结算现金流存在结构性脱钩：Zero-Bid 机械获得六位数虚假现金流。 | **三风场 Zero-Bid ($q \equiv 0$) 诊断**:<br>• 短缺罚款: 全部为 **\pounds 0.00**;<br>• 单边现金流: Kelmarsh **+\pounds 223,882**, Hill **+\pounds 239,299**, Penmanshiel **+\pounds 243,450**;<br>• 非对称调度损失: 受到严厉惩罚 ($0.304 \to 0.337$, 与平均出力 $\bar{y}$ 对齐).<br>*(见 Table 3, Fig. 4a)* | “市场结算现金流是衡量预测经济价值的最直接商业标准。” | **Reductio ad Absurdum (Zero-Bid 归谬算子)**: 证明任何放弃日前申报、完全被动上网的策略都能在单边核算下由于非负上网电量而产生巨额现金流，证明该指标存在严重漏洞。 | 审稿人若不懂电力市场双结算机制会产生误解。 | *"Evaluating models on standalone unhedged balancing cashflows introduces structural distortion: a degenerate zero-bid baseline ($q \equiv 0$) mechanically generates six-figure apparent surpluses by converting generation into uncontracted spill."* |
| **CLM-09**<br>*(Supporting: 文献报告盲区)* | 现有 46 篇 Scoping 纳入文献中，0.0% 报告了运行失败边界或负面权衡，95.7% 存在开源代码赤字，50.0% 存在消融混杂。 | **46 篇文献逐篇全文字段提取**:<br>• 0/46 篇报告运行失败边界 (95% Wilson CI: [0.0%, 7.7%]);<br>• 44/46 篇无公开代码 (95% Wilson CI: [85.5%, 98.8%]);<br>• 23/46 篇通过切除多层神经网络子模块进行消融 (50.0%).<br>*(见 Table 5, `study_level_ledger_48.csv`)* | “文献中没有负面报告是因为物理约束在所有工业工况下都确实稳定有益。” | **三风场受控应力测试**: 本文实证展示了在大样本、弱 $\lambda$、打乱映射和代理滞后下的多处性能恶化，反衬出文献中 0% 报告率属于选择性报告偏误。 | 严禁推广至“全人类全部学术论文”，必须严格限定在 46 篇 Scoping cohort 内。 | *"Within the audited scoping cohort (46 studies, 2023--2026), 0.0\% of studies report operational failure regimes, 95.7\% lack public source code, and 50.0\% evaluate constraints via confounded module excisions."* |
| **CLM-10**<br>*(Supporting: 软件计算图错位)* | PCT 官方代码由于参数位置颠倒，实际执行了逆向 JS 形式，诱发 $\mathcal{O}(1/\epsilon)$ 逆密度梯度奇异点与 $2.2\times$--$2.7\times$ 梯度放大。 | **AST 静态解析 + 梯度求导 + 训练监控**:<br>• 官方代码 `F.kl_div(pred.log(), m)` 实际执行 $\text{KL}(M \parallel P_{\text{pred}})$;<br>• 导数包含 $-\frac{P_{\text{ref}, i}}{4 P_{\text{pred}, i}}$ 逆密度奇异点;<br>• 实测 $\lambda=1.0$ 时梯度范数放大 $2.20\times$ (Kelmarsh) 和 $2.72\times$ (Penmanshiel);<br>• 张量外积 $K_p^\top K_w$ 在代数上严格等价于 2D product KDE (负对照验证). | “代码实现与论文声明完全一致，或者颠倒参数只是无伤大雅的数值等价。” | **Mathematical Derivation & Empirical Contrast**: 给出封闭形式导数推导，并在相同数据集上对比官方反向 JS 与修正后标准 JS 的训练梯度与调度表现。 | 严禁使用“造假/篡改”等道德攻击词汇，必须严格限定为“实现分歧 (Implementation Divergence)”。 | *"Static AST inspection of the PCT benchmark uncovers an inverted argument order in its divergence loss, inducing an analytical $\mathcal{O}(1/\epsilon)$ inverse-density gradient singularity that amplifies gradient norms by $2.20\times$ to $2.72\times$."* |

---

## 二、红线审查清单：严禁出现的学术 Overclaim 与措辞自查

在后续正文修改中，必须对以下 6 种表述执行“一票否决制”过滤：

1. **绝对禁止无对照的因果断言**：
   - ❌ 禁用：*"Our experiments prove that complex terrain causally mandates the use of physical constraints."*
   - ✅ 修正：*"Across the evaluated sites, empirical priors yielded greater downside risk reduction in upland terrain with restricted training data, though terrain complexity remains confounded with turbine hardware and sample volume in operational telemetry."*

2. **绝对禁止宣称物理先验具有普遍通用有效性**：
   - ❌ 禁用：*"Physics-constrained models universally enhance forecasting accuracy and operational profit."*
   - ✅ 修正：*"Physical priors act as conditional inductive guardrails: their operational value is context-dependent, providing measurable benefits under data scarcity but diminishing or reversing under data abundance, prior misspecification, or temporal lag."*

3. **绝对禁止将弱惩罚下的性能差异归因于物理知识**：
   - ❌ 禁用：*"The minor improvement at $\lambda=0.01$ demonstrates that the network effectively internalizes aerodynamic physics."*
   - ✅ 修正：*"Because structurally scrambled controls reproduce the performance of authentic priors at $\lambda=0.01$, marginal gains under weak regularization remain consistent with non-specific numerical smoothing rather than domain reasoning."*

4. **绝对禁止将单边结算现金流表述为风电场商业总利润**：
   - ❌ 禁用：*"The model generated £12,174/turb·yr of commercial revenue in balancing market trading."*
   - ✅ 修正：*"In a stylized imbalance cashflow component diagnostic abstracting from forward contracts, distribution constraints yielded an apparent cashflow component of £12,174/(turb·yr), which our zero-bid diagnostic reveals to be decoupled from true predictive accuracy."*

5. **绝对禁止对审计对象进行主观道德指控**：
   - ❌ 禁用：*"The PCT paper fabricated results or contains fraudulent code."*
   - ✅ 修正：*"The official PCT training script exhibits an implementation divergence in its divergence loss argument ordering, altering analytical gradient scaling."*

6. **绝对禁止将 Scoping 统计跨域无限外推**：
   - ❌ 禁用：*"All wind power forecasting papers suffer from open-code deficits and reporting bias."*
   - ✅ 修正：*"Within our prespecified 46-study scoping cohort (2023--2026) under PRISMA-ScR protocols (95% Wilson CI: [0.0%, 7.7%]), zero studies reported operational failure regimes."*

---

## 三、PHASE 2 矩阵结论与实施通行证

- **证据真实性**：矩阵中的所有数值均已完成对 `06_实验结果与日志/` 原始数据的二次核对，无任何幽灵数字或计算错误；
- **自洽性闭环**：以 RQ1（特异性）、RQ2（边界地图）、RQ3（评价对齐）为核心的实证链条在现有数据下具有**100% 封闭支撑力**；
- **通行证签发**：正式批准进入后续的架构设计与重构阶段！
