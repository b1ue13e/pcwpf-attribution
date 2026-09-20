# 论文图表结构优化报告

## 修改概述

根据 Applied Energy 期刊配图标准和论文可读性要求，已将正文图表从 **1 张扩展到 4 张**，并将核心附录图"转正"到正文。

---

## 修改详情

### 新增 Figure 2: 稳健性衰减折线图
- **位置**: §5.1 "Main-site guardrail ordering under corrupted SCADA" 之后
- **文件**: `paper_figures_canonical/main_text/fig2_robustness_decay.pdf`
- **功能**: 
  - X轴: 噪声强度 σ_eff (0.00 → 0.60)
  - Y轴: R² 性能指标
  - 展示 DLinear 在严重污染下的"断崖式下跌"(-0.9903)
  - 凸显 Ultra-LSNT 相对平缓的衰减趋势(0.7863)
- **数据来源**: `results/supplementary_evidence/deep_sota_gaussian_windcn.csv`

### 新增 Figure 3: 调度经济效益分组柱状图
- **位置**: §5.3 "Fixed-interface operational readout" (Table 5 之后)
- **文件**: `paper_figures_canonical/main_text/fig3_dispatch_economics.pdf`
- **功能**:
  - Panel (a): 总实时成本对比 (M CNY)
  - Panel (b): 弃风量对比 (M kWh)
  - 直观展示 iTransformer (32.16) 到 DLinear (54.47) 的 44% 成本差异
- **数据来源**: Table 5 (tab:master_dispatch)

### 转正 Figure 4: 路由诊断解释图 (原附录 I.5)
- **位置**: 新增 §5.4 "Routing diagnostics for operational interpretation"
- **文件**: `paper_figures_canonical/appendix/fig_app_interpretability.pdf` (Panel a)
- **文件**: `paper_figures_canonical/appendix/fig_app_routing_heatmap.pdf` (Panel b)
- **功能**:
  - 展示稀疏路由的条件计算行为
  - Panel (a): 昼夜专家激活份额
  - Panel (b): 事件窗口路由热力图与预测轨迹
- **原位置**: 附录 §A (已移除重复部分)

---

## 图表引用更新

| 图表 | 标签 | 引用位置 | 状态 |
|-----|------|---------|------|
| Figure 1 | `fig:workflow` | §3.3 | 原有 |
| **Figure 2** | `fig:robustness_decay` | §5.1 (新增) | ✅ 新增 |
| **Figure 3** | `fig:dispatch_economics` | §5.3 (新增) | ✅ 新增 |
| **Figure 4** | `fig:routing_diagnostics` | §5.4 (新增) | ✅ 转正 |

---

## LaTeX 修改文件

主文件: `main_PI-MoE_final_AE_ready.tex`

修改位置:
1. Line ~665: 插入 Figure 2 引用和说明
2. Line ~803: 插入 Figure 3 引用和说明  
3. Line ~830-833: 正文新增 §5.4 路由诊断
4. Line ~1337: 移除附录重复的路由诊断部分

---

## 文件清单

### 新生成的图表文件
```
paper_figures_canonical/main_text/
├── fig2_robustness_decay.pdf      (24,958 bytes)
├── fig2_robustness_decay.png      (PNG preview)
├── fig3_dispatch_economics.pdf    (40,244 bytes)
└── fig3_dispatch_economics.png    (PNG preview)
```

### 重用的附录图 (已转正)
```
paper_figures_canonical/appendix/
├── fig_app_interpretability.pdf   (Figure 4 Panel a)
└── fig_app_routing_heatmap.pdf    (Figure 4 Panel b)
```

### 生成脚本
```
generate_figure2_robustness_decay.py
generate_figure3_dispatch_economics.py
```

---

## 改进效果

### 修改前
- 正文图表: 1 张 (Figure 1)
- 正文表格: 5 个
- 附录图表: 5+ 张

### 修改后
- 正文图表: 4 张 (Figure 1-4)
- 正文表格: 5 个 (保留)
- 视觉呈现更丰富，避免"重表格、轻图片"问题

### 提升点
1. **可读性**: 折线图比表格更直观展示性能衰减趋势
2. **视觉冲击力**: 柱状图一眼看出成本/弃风差异
3. **解释性**: 路由诊断图帮助理解模型内部行为
4. **期刊匹配度**: 符合 Applied Energy 6-12 张图的常规范围

---

## 编译建议

```bash
# 清理旧编译文件
rm -f main_PI-MoE_final_AE_ready.aux main_PI-MoE_final_AE_ready.bbl

# 重新编译 (需运行 2-3 次以解析交叉引用)
pdflatex main_PI-MoE_final_AE_ready.tex
bibtex main_PI-MoE_final_AE_ready
pdflatex main_PI-MoE_final_AE_ready.tex
pdflatex main_PI-MoE_final_AE_ready.tex
```

---

## 注意事项

1. **图片路径**: 新图使用 `paper_figures_canonical/main_text/` 路径
2. **附录引用**: 原附录图路径不变，仅文字描述移至正文
3. **图表编号**: 确保编译后图表编号连续 (Figure 1→2→3→4)
4. **页码调整**: 新增图表可能导致后续章节页码变化

---

生成时间: 2026-04-01
