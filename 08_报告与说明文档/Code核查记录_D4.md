# Code availability 核查记录（D4 fidelity 落地，2026-09-04）

## 方法与边界（诚实声明）

- 检索手段：GitHub 定向检索 + web_search（标题/方法名 + "code"/"github"）。
- 检索日期：2026-09-04。
- **只做了代表性核查**（本语料中可执行审计锚点 + 方法族代表），未逐篇穷举；
  未命中的保持 `unverified`，不伪造为 `no-code`。

## 结论一句话

> 除 PCT（有官方仓库 `daxin007/PCT-WPF`，已核查为 code-mismatch）和 TgDPF 的
> 谱系论文外，本语料 26 篇中**未检索到任何其它可公开获取的官方代码仓库**。
> 实现忠实度（D4）在领域内系统性不可核验——这本身就是审计发现。

## 逐篇核查状态

| paper_id | D4 判定 | 依据 |
|---|---|---|
| PCT | code-mismatch | 官方仓库 daxin007/PCT-WPF 存在，已逐行核查 KL 反参数 + outer joint |
| TgDPF | no-code | "in line with PCT" 谱系，无独立官方仓库；论文正文公式是标准 KDE/JS |
| TANG2025 | no-code | 摘要级，无全文无代码 |
| PITFN2025 | parity-only | 公式 (20) 明确，但从论文公式重建，无官方代码对照 |
| JENSENWAKE2026 | parity-only | λ3=0.4 公式明确，无官方代码 |
| TCNTRANS2026 | no-code | IET, abstract only |
| 其余 20 篇 | unverified | 检索未命中官方仓库；需后续逐篇 Data Availability 声明核对 |

## 对 C2 的贡献

D4 `unverified` 21/26 + `no-code` 2/26 + `parity-only` 2/26 = **25/26 不可核验或
无可核验对象**，仅 PCT 1 篇有官方代码且实为 mismatch。

这量化了领域审计的核心 gap 之一：**论文声称的物理约束几乎都没有可核验的官方
实现，唯一有代码的代表性案例（PCT）经核验与论文公式不一致。**

## Crossref 元数据核查（2026-09-04，补 P2 第 4 项）

对 12 篇有 DOI 的论文查询 Crossref `assertion`（data-availability 字段）：

- **有 data-availability 元数据**：仅 2 篇（TgDPF、PISTNET2025）
- **无 data-availability 元数据**：10 篇（PCT、MMD-PGF、MLPINN、JARA2024、
  CPDIGITAL2025、DUALSTREAM2026、PGMFNET2026、RISKREG2025、SETVAL2025 等）
- 全部 12 篇 Open Access（license 字段存在）。

**结论**：即使开放获取，26 篇语料的 data-availability 声明也系统性缺失
（12 篇 DOI 样本中仅 2 篇有）。这与 D4 unverified 互为因果：**可获取全文 ≠
可核验实现**，因为绝大多数论文未声明数据/代码可得性。

## 待办（转向 P2 收尾）

1. ~~Crossref data-availability 核查~~ 已完成（见上）。
2. 未提供 DOI 的 14 篇（中文/会议/预印本）用 CNKI 元数据补查（如需）。
