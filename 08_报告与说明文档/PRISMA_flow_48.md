# PRISMA Flow（48 项口径，第六轮重生成）

> 唯一 study-level ledger = `study_level_ledger_48.csv`（48 行）。
> 本文件与旧 89/72/26/21 分母**不再并存**，后者作废。

```
IDENTIFICATION
  OpenAlex title.search (4 queries, 2023–2026)   72 unique records
  web_search novel candidates                    17
  TOTAL identified (raw)                         89

SCREENING (title, 独立 Task D 复核)
  screened                                       72 (OpenAlex 去重后)
  → include                                      53
  → exclude                                      19
      (not-forecasting 9 / maintenance-control 9 / no-physics 1)

DEDUPLICATION (预印本 vs 正刊，保留正刊)
  53 include → 合并 5 组重复 → 48 unique studies

INCLUSION CHECK (全文级, 杜俊涛)
  48 → include 46, exclude 2
      (S26 wind-speed-only; S42 no-explicit-physics-constraint)

FINAL CODED CORPUS                               46 studies
```

## 关键去重键（5 组）

| 组 | 正刊（保留）| 预印本（合并）|
|---|---|---|
| 1 | Applied Energy 2025 (TgDPF) | SSRN 2024 |
| 2 | EngApplAI 2025 (evidential PINN) | arXiv 2023 |
| 3 | EPSR 2026 (dynamic wake graphs) | SSRN 2026 |
| 4 | arXiv 2026 (UniWind) | （无 DOI 重复）|
| 5 | EPSR 2025 (regional wind field) | SSRN 2025 |

## 状态声明

- 46 篇纳入口径的全部百分比只从这个 48-study ledger 计算。
- 旧 21 篇（FINAL21）不再作为任何分母；MASTER 已改指 `study_level_ledger_48.csv`。
