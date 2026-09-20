# 论文唯一主故事线拓扑图（Unified Storyline Map）

```mermaid
graph TD
    A[Central Question: 物理约束收益是真实气动指导还是数值平滑？如何转化为市场真实决策价值？] --> B[Literature Gap: 46篇Scoping审计揭示 0%负面报告, 95.7%开源赤字, 50%消融混杂]
    B --> C[Software Audit: 唯一可运行训练基准 PCT 代码参数颠倒, 产生 O(1/eps) 逆密度奇异性与 2.7倍梯度放大]
    C --> D[Negative Control: 2D乘积核张量展开证明, 确认核外积保持双变量联合分布 (客观中立)]
    D --> E[Prospective Controlled Benchmark: 骨干与参数完全对齐的受控消融]
    
    subgraph Regime [实证四大分水岭边界 (Regime Transitions)]
        E1[1. 少样本稀缺区间: 真实先验 -1.22% vs 打乱先验 +68.1% (确立先验特异性)]
        E2[2. 信息预算对齐: 500样本预算匹配, 保留 77%-92% 收益 (证伪纯历史泄漏)]
        E3[3. 样本扩张与弱惩罚: N=5000 无约束反超; lambda=0.01 真实与打乱完全重合 (数值平滑)]
        E4[4. 训练代理陈旧度: 气象代理滞后 k>12 步时物理指导单调破坏调度]
    end
    
    E --> E1
    E --> E2
    E --> E3
    E --> E4
    
    E1 & E2 & E3 & E4 --> F[Decision Theory: 引理1与三项二阶展开 (解释未校准伪收益与校准后表征暴露)]
    F --> G[Market Settlement Decoupling: 英国 2017 BSC 动态结算]
    G --> G1[地形风场结算分化: Hill of Towie 节省 £578 vs Kelmarsh 亏损 -£932]
    G --> G2[Zero-Bid 诊断: q=0 攫取 £220k+ 现金流, 证明单边不平衡现金流与预测精度完全解耦]
    
    G1 & G2 --> H[Metascience Governance: 五门禁 REFORMS-PCWPF 顶刊报告清单]
    H --> I[Conclusion: 物理约束是情境依赖的防过拟合护栏, 绝非通用精度万灵药]
```

---

## 章节与证据映射矩阵（Section-to-Evidence Spine Mapping）

| 章节编号与法定标题 | 核心回答的科学问题 | 绑定核心证据与图表 | 剔除/移入附录内容 |
|:---|:---|:---|:---|
| **§1 Introduction** | 物理约束为何不能仅凭 RMSE 判断？归因与评价盲区何在？ | 46篇审计三项统计核心数字、三风场受控反差数据预告 | 严禁堆砌实验细节，突出问题下沉逻辑与三大贡献 |
| **§2 Mathematical Formulation & Market Decoupling** | 非对称调度损失如何引发误差解耦？标量校准的理论边界何在？ | 引理1证明、三项展开（Term I/II/III）、反例1与反例2、双结算方程 | 强调理论为诊断工具，加入 Diagnostic Caveat |
| **§3 Empirical Audit Framework & Methodology** | 如何建立全生命周期审计与严格受控的消融实验方案？ | 5D生命周期图（Fig. 1）、PRISMA流线图（Fig. 2）、三风场数据表 | 移除任何历史遗留模型描述 |
| **§4 Systematic Literature & Software Fidelity Audit** | 现有文献的盲区有多严重？开源代码是否忠实执行了数学公式？ | 生命周期分布图（Fig. 3）、Table 1（46篇统计）、PCT 梯度奇异性实测（Fig. 4）、2D核展开证明 | 纯粹客观技术中立，不进行主观学术道德指责 |
| **§5 Multi-Site Controlled Empirical Benchmark** | 真实气动先验何时有益、何时中性、何时有害？市场结算如何解耦？ | 3D受控消融图（Fig. 5）、学习曲线图（Fig. 6）、扰动严苛度图（Fig. 7）、动态结算表（Table 2, 3）、Zero-Bid诊断（Table 4, Fig. 8）、PatchTST探针（Table 5） | 将探索性超参搜索与噪声鲁棒性移入 Supplementary Material |
| **§6 Discussion, Identification Boundaries & Governance** | 这一实证图景如何改变领域认知？作者的因果识别边界在哪里？ | 四大边界条件、五门禁 REFORMS 清单（Table 6）、Confounding Disclaimers | 不重复 Results 中的数字，专注于机理归因与学风治理 |
| **§7 Conclusion** | 核心结论与对未来新能源系统的工程警示 | 紧凑提炼三大发现，给出三条可操作工程部署建议 | 250词以内，避免泛泛同义改写 |
