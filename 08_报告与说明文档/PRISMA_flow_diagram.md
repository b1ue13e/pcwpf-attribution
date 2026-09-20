# PRISMA Flow Diagram 记录（C2 系统审计，2026-09-04）

## 定位声明（诚实边界）

本审计是 **PRISMA-style pilot corpus 系统编码**，不是 full systematic review：
- 检索通道：**OpenAlex（聚合 Crossref/MAG 等元数据库）+ 搜索引擎 web_search**（两个
  独立通道）；Semantic Scholar API 尝试过但被 429 限流（无 key），未获得结果——
  如实记录，不以"三数据库"名义包装。
- 计数为**可复现的精确流程**（`prisma_exact_pull.py` + `prisma_exact_screen.py`；
  title 筛选为规则判定、纳入/排除最终以摘要为准）；早期"去重≈69"的估算已替换。
- 纳入标准明确（forecasting + 显式物理约束），排除 curve-fitting/非物理/控制/维护等。

## Flow diagram 数字（精确，2026-09-04 可复现）

```
IDENTIFICATION
  OpenAlex title.search (2023–2026)  n=74 (raw)
    Q1 physics-constrained wind power forecasting   n=9
    Q2 physics-informed wind power                  n=44
    Q3 physics-guided wind power                    n=20
    Q4 wind power curve deep learning               n=1
  OpenAlex after de-duplication (by DOI)            n=72
  web_search novel candidates (2 rounds)            n=17
  TOTAL identified (raw, after db dedup)            n=89

SCREENING (title-level)
  screened                                          n=89
  OpenAlex excluded at title (control/maintenance/  n=17
    aggregation/inertia/curve-modeling/solar-only/
    wind-speed-only/non-forecasting)
  OpenAlex assessed                                 n=55
  web_search novel -> assessed                      n=17

ASSESSED (abstract-level, inclusion criteria)
  assessed                                          n=72

INCLUDED (coded 26-article corpus)                  n=26
  (5 篇在编码后被排除：curve-fitting / 非物理约束 → 最终纳入 21 篇)
```

**诚实标注**：
- title 筛选（17 excluded）是**可复现规则判定**，边界篇（如 "power estimation 是否
  算 forecasting"）以摘要为准，逐篇 final 判定仍需人工复核——故 ASSESSED=72 是
  "规则+人工两层的当前值"，不是不可变的精确真值。
- 早期"去重≈69、title 排除≈38"两个估算值**已作废**，替换为上面的可复现数。

## 排除清单（5 篇 + 理由）

| paper_id | 排除理由 |
|---|---|
| XAIPC2023 | XAI power-curve 验证，非 forecasting |
| RISKREG2025 | chance-constrained 曲线拟合 + 异常检测，非 forecasting |
| SETVAL2025 | interval NN 曲线拟合，非 forecasting |
| CPDIGITAL2025 | Cp-λ-β 曲线估计（digital twin），非 forecasting |
| HUANG2026 | CEEMDAN 分解聚类，无物理约束 |

## 纳入标准（冻结）

1. 风电/风电场功率（或风速）**forecasting** 任务；
2. 显式物理约束/先验（power curve / Betz / KDE-JS / MMD / PDE / wake / 边界正则）；
3. 2023–2026；
4. 有可评估的英文/中文摘要。

## 待补（诚实列出）

1. Semantic Scholar 第二数据库（429 限流，已尝试未成功；可在取得 API key 后补）。
2. title-level 排除的精确计数（n≈38 目前是 total−assessed 的差值估计）。
3. 完整 PRISMA 2020 checklist 的逐项标注。
4. flow diagram 图（可用 Python graphviz / TikZ 生成）。
