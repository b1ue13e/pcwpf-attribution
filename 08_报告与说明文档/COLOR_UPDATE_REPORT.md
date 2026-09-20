# 图表配色更新报告 - 莫兰迪色调

**更新时间**: 2026-04-01  
**参考图片**: 微信图片_20260330234645_422_17.jpg

---

## 🎨 配色方案提取

从参考图片中提取的莫兰迪色调（低饱和度柔和配色）：

| 颜色名称 | HEX值 | RGB值 | 用途 |
|---------|-------|-------|------|
| Pink Light | `#D58FB1` | R213 G143 B177 | LightGBM, 背景区 |
| **Pink Dark** | `#C54E88` | R197 G78 B136 | **Ultra-LSNT (主体)** |
| **Blue Gray** | `#7E98BD` | R126 G152 B189 | **iTransformer (最佳)** |
| **Purple Gray** | `#9693C2` | R150 G147 B194 | **TimeMixer** |
| **Gray** | `#7C7A7D` | R124 G122 B125 | **DLinear (最差)** |

---

## 📊 应用效果

### Figure 2: 稳健性衰减折线图

**文件**: `fig2_robustness_decay_v3.pdf` (39.9 KB)

配色分配：
- **Ultra-LSNT**: Pink Dark (#C54E88) - 深粉色，突出主体
- **iTransformer**: Blue Gray (#7E98BD) - 蓝灰色
- **TimeMixer**: Purple Gray (#9693C2) - 紫灰色
- **DLinear**: Gray (#7C7A7D) - 灰色，强调崩溃
- **LightGBM**: Pink Light (#D58FB1) - 浅粉色
- **Persistence**: 浅灰色 (#9CA3AF)

特点：
- 低饱和度配色符合学术期刊审美
- 区域背景使用莫兰迪色调的淡色
- 注释框使用对应颜色的浅色调

---

### Figure 3: 调度经济效益柱状图

**文件**: `fig3_dispatch_economics_v3.pdf` (42.7 KB)

配色分配（按性能排序）：
- **iTransformer** (最佳): Blue Gray (#7E98BD)
- **TimeMixer**: Purple Gray (#9693C2)
- **Ultra-LSNT** (主体): Pink Dark (#C54E88)
- **DLinear** (最差): Gray (#7C7A7D)

特点：
- 性能越好越偏冷色调(蓝灰)
- 主体方法使用暖色调(粉色)突出
- 最差性能使用中性灰警示

---

## ✅ 文件清单

### 新生成的莫兰迪配色图表
```
paper_figures_canonical/main_text/
├── fig2_robustness_decay_v3.pdf      (39,873 bytes) ✅
├── fig2_robustness_decay_v3.png      (预览图)
├── fig3_dispatch_economics_v3.pdf    (42,673 bytes) ✅
└── fig3_dispatch_economics_v3.png    (预览图)
```

### 生成脚本
```
generate_figure2_robustness_v3_morandi.py
generate_figure3_dispatch_v3_morandi.py
```

---

## 📝 LaTeX更新

已更新文件引用：
- Line ~661: `fig2_robustness_decay_v3.pdf`
- Line ~803: `fig3_dispatch_economics_v3.pdf`

编译状态：✅ 成功

---

## 🎯 配色优势

1. **学术性**: 莫兰迪色调低饱和度，专业不刺眼
2. **区分度**: 5种颜色足够区分7个模型
3. **一致性**: 与参考图片风格统一
4. **无障碍**: 色盲友好的柔和配色
5. **印刷友好**: 低饱和度适合黑白打印

---

## 🔍 视觉对比

| 特性 | 原版(v2) | 莫兰迪版(v3) |
|-----|---------|-------------|
| Ultra-LSNT | 深紫 #4B1475 | 深粉 #C54E88 |
| iTransformer | 森林绿 #2E7D32 | 蓝灰 #7E98BD |
| 饱和度 | 中高 | 低(莫兰迪) |
| 风格 | 科技感 | 学术高级感 |
| 印刷适应 | 良好 | 优秀 |

---

**完成状态**: ✅ 全部更新完成
