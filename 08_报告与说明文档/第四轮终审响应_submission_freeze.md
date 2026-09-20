# 第四轮终审响应（submission freeze 治理，2026-09-04）

## 裁决（接受）

Reject / No-Go，4.3/10。核心：不是新事实错误，而是**治理不彻底**——
旧表述未刷干净、口径未统一、缺逐篇 ledger 与 code/DA 核验。

## 逐项执行结果（无 GPU submission freeze）

### 1. PCT 核查报告"单一事实源"清理 ✅
- 后半多次残留 "independence-joint / 两个实现缺陷 / outer-product 导致 RMSE 差异"
  → 全部改为撤回声明或作废存档。
- 统一 KL 写法为第四轮终审指定形式：
  `F.kl_div(log p, q) = KL(q‖p)` → 候选差异 = `½[KL(m‖p)+KL(m‖q)]`
  vs 标准 JS `½[KL(p‖m)+KL(q‖m)]` → **reverse-JS-type divergence**。
- 所有 3-8× / 2-4× / 6-11.5× / 8.0× 数字标注"基于已作废 runner，不再引用"。

### 2. 无损骨架摘要修正 ✅
- "independence (marginal-product) density" → 删除；改为 `F.kl_div(log p, q)
  = KL(q‖p)` 的 reverse-JS 表述。
- "19/21 report no negative effects" → "18/21"（采纳第四轮口径，最保守）。

### 3. 21 篇口径统一（单一编码表 + 自动统计）✅
- 新建 `literature_coding_table_FINAL21.csv`（`build_final21.py` 从 coder_C 仲裁生成）。
- 自动统计 `final_stats.json`：
  - D4 = 20 no-code（→ 修订 not-found）+ 1 code-mismatch
  - D5 = 19 none-reported + 1 ablation-absent + 1 harm-quantified
  - **口径消解**：正文统一写 "19/21 未报告任何负面效应"，并在脚注说明
    "其中 1 篇 simultaneously ablation-absent；若按最保守纯 none 口径为 18/21"。

### 4. PRISMA 89 条逐篇 ledger ✅
- `prisma_ledger.py` → `tmp/prisma_ledger.csv`（115 行可复现底稿）。
- 流程：89 identified → 63 assessed → 26 excluded-title → 21 included + 5
  excluded-coding。
- **诚实定位**：excluded-title 分类是**可复现 title 规则，非摘要级人工**；
  规则微调使 assessed/excluded 各浮动约 ±9。逐篇最终核定需摘要阅读。

### 5. 21 篇逐篇 code/DA 核验 ✅
- `code_da_check_21.py` → `tmp/code_da_check_21.csv`。
- D4 修订：**20 not-found（Crossref 检索 2026-09-04 未找到公共仓库）+ 1
  code-mismatch（PCT）**。
- 不再断言 "no-code"，改为 "not-found (as of 2026-09-04)"。

## 仍未完成（诚实，非本轮可单人解决）

1. **21 篇逐篇摘要级人工 exclusion 复核**（PRISMA ledger 的 ±9 浮动要靠真人摘要阅读）。
2. **reverse-JS 公式的独立验证**（第四轮终审明确"需独立验证"，我无法充当第二人）。
3. **完整稿件**（第四轮终审定稿标准：写出一份完整稿件后再终审）。

## 当前诚实定位（第四轮后冻结）

4.3/10。最现实定位仍是**二区 / scoping empirical audit**，不是一区。可保留
（无损）：
- Constraint Lifecycle 五维框架
- training prior ≠ inference requirement 区分
- 21 篇 harm 少被报告的初步发现（19/21 未报告负面效应）
- reverse-JS-type 候选差异（PCT 唯一站得住静态差异）
- 完整失败与撤回治理记录（5 轮撤回）

## 下一步决策点（需用户定）

- 若投二区：可开始写 full manuscript（Methods=Protocol v2 + PRISMA ledger；
  Results=FINAL21 表 + reverse-JS 静态案例；治理记录作 Limitations/Supplementary）。
- 若冲一区：唯一路径是第二研究者独立验证 reverse-JS + 独立重跑训练，我做不了。
