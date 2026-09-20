# Task D — PRISMA 排除逐篇复核

> 材料：本文件 + `prisma_screening_blank.csv`（72 行空白表）+ 各论文公开摘要。
> 你的任务：对 72 篇检索命中逐篇判定"是否应纳入本研究"，用你的判断，不是关键词规则。

## 1. 纳入 / 排除标准

**纳入（include）**：论文任务是**风电/风电场功率（或风速）预测 forecasting**，并且
**使用了显式物理约束/先验**（功率曲线、Betz 极限、KDE-JS/MMD 分布匹配、PDE、尾流
模型、边界正则等）。

**排除（exclude）**，理由类别：
- `not-forecasting`：曲线拟合、异常检测、控制优化、状态估计等非预测任务
- `no-physics-constraint`：纯数据驱动、无显式物理先验
- `wind-speed-only`：只预测风速且没有功率预测或物理约束
- `not-wind`：光伏/太阳能/其它非风电
- `maintenance-control`：维护、控制、优化、惯性估计等
- `no-abstract`：找不到摘要，无法判定

**存疑**：若标题明显相关但无法仅凭摘要判定，标 `need-abstract`，并在 notes 写清
缺什么信息。

## 2. 操作步骤

1. 打开 `prisma_screening_blank.csv`（72 行，含 odoi / title / year）。
2. 对每篇，用 odoi（若提供）或 title 找到论文摘要。
3. 填三列：
   - `decision`：`include` / `exclude` / `need-abstract`
   - `decision_basis`：`title-only` / `abstract-read`（你实际读到了什么程度）
   - `exclusion_reason`：若 exclude，写上面的理由类别；若 include 留空
4. 若某篇 odoi 为空，用 title 检索；实在找不到标 `no-abstract`。

## 3. 报告模板

```markdown
# TaskD 报告 — <你的名字> — <日期>

## 逐篇判定
（可附填写后的 csv，正文给汇总）

## 汇总
- include: N 篇
- exclude: M 篇（按理由类别分布）
- need-abstract: K 篇

## 你判为 include 但需要额外注意的篇目（可留空）
...

## 声明
我独立完成判定，未参考本项目任何既有筛选结果。
```

## 4. 提交

- `prisma_screening_blank.csv`（填写后，改名 `TaskD_screening_你的名字.csv`）
- `TaskD_报告_你的名字.md`
