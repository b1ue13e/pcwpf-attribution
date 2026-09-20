# 条件性归纳护栏：物理先验何时增强、模仿或损害风电预测决策

> **Conditional Inductive Guardrails: Identifying When Physical Priors Enhance, Mimic, or Impair Decisions in Wind Power Forecasting**
>
> Applied Energy 投稿项目。当前状态：**minor revision 修订中**。本仓库为**私有仓库**，审稿结束后可转公开。

## 项目定位

论文的核心问题只有一个：**归因**——文献中报告的"物理先验带来的性能提升"，有多少真正来自先验里的物理内容本身？通过形式与权重完全匹配的伪置乱（pseudo-scrambled）安慰剂对照、预算匹配的信息消融、边界压力测试与决策对齐度量（非对称调度损失，r=2），把物理先验刻画为**条件性归纳护栏**（conditional inductive guardrails），并给出 C1–C5 报告清单。

## 目录结构

| 目录 | 内容 |
|---|---|
| `01_论文投稿材料/active_manuscript/` | **canonical 投稿正文**：LaTeX 源、figures、tables、main.pdf |
| `01_论文投稿材料/chinese_manuscript/` | 中文对照稿 |
| `04_数据集/constraint_rebuild_v2/*.csv` | 三风场规范 10 分钟 SCADA CSV + Elexon 系统价格（原始 zip 与全量数据从 Zenodo / BMRS 获取） |
| `05_实验代码与脚本/` | 全部实验、审计与图生成脚本 |
| `06_实验结果与日志/` | 分析级证据（审计报告 JSON、bootstrap CSV、消融对比表）；大体积运行缓存不入库 |
| `03_图表与绘图素材/` | 矢量图源（TIFF 生产母版不入库） |
| `.agents/scripts/verify_gate.py` | 验证门：任务状态 + 跨工件数值一致性检查 |
| `.agent_state/` | 任务台账 / 证据台账 / 决策日志 / 评审报告（可靠性协议状态） |

## 本地验证

```bash
python .agents/scripts/verify_gate.py   # exit 0 即 PASS
```

## 本地编译投稿 PDF

```bash
# 方式一（推荐，单二进制）
tectonic 01_论文投稿材料/active_manuscript/main.tex
# 方式二（传统工具链）
cd 01_论文投稿材料/active_manuscript && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

## CI（GitHub Actions）

[![verify](https://github.com/b1ue13e/pcwpf-attribution/actions/workflows/verify.yml/badge.svg)](https://github.com/b1ue13e/pcwpf-attribution/actions/workflows/verify.yml)

每次 push / PR：① 运行验证门（状态文件 + 规范数值一致性）→ ② tectonic 云端编译 `main.tex` → ③ 上传 `main.pdf` 为构建产物。

## 数据来源（公开许可）

- Kelmarsh：`10.5281/zenodo.5841834`（Cubico，CC-BY 4.0）
- Penmanshiel：`10.5281/zenodo.5946808`（Cubico，CC-BY 4.0）
- Hill of Towie：`10.5281/zenodo.20204946`（RES/TRIG，CC-BY 4.0）
- Elexon System Buy Price：BMRS portal（UK Open Government Licence v3.0）

## 分支与发布

- 本地 `revision` 分支同步推送到远端默认分支 `main`
- Release `v2026.09-r1`：修订快照，附 `main.pdf`

## 隐私

未发表稿件内容请勿外传；`09_项目申请_简历_个人材料/`、`90_环境与第三方依赖/`、`99_待确认归档/` 等目录已被 .gitignore 排除，不会进入 GitHub。
