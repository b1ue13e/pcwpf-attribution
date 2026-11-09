# 条件性归纳护栏：物理先验何时增强、模仿或损害风电预测决策（中文说明）

[English](README.md) · [GitHub 仓库](https://github.com/b1ue13e/pcwpf-attribution)

> **Conditional Inductive Guardrails: Identifying When Physical Priors Enhance, Mimic, or Impair Decisions in Wind Power Forecasting**

本项目研究风电预测中的实证归因问题：文献中报告的“物理先验带来的预测增益”，有多少真正来自物理领域知识本身，又有多少源自正则化容量、特征尺度或评估假象？

## 研究概述与范围说明

本公开仓库的远端默认分支（`main`）作为学术证据与稿件归档库。研究结合形式与权重匹配的伪置乱安慰剂对照、预算匹配的信息消融、边界压力测试以及非对称调度损失度量，将物理先验系统性地刻画为条件性归纳护栏。

稿件正文围绕四层证据金字塔展开（核心结论 -> 边界图谱 -> 决策对齐 -> 辅助核验）。

## 环境配置与运行检查

本工程基于标准 Python 3.12 环境以及用于 LaTeX 编译的 Tectonic 工具链。

### 运行规范一致性验证门

执行跨工件数值与状态一致性核验：

```bash
python .agents/scripts/verify_gate.py
```

该脚本自动核验任务台账、断言名册以及稿件与证据文件之间的核心数值一致性。

### 编译论文稿件

使用 Tectonic 从源码编译投稿 PDF：

```bash
cd "01_论文投稿材料/active_manuscript"
tectonic -X compile --keep-logs main.tex
```

GitHub Actions 持续集成环境（[`.github/workflows/verify.yml`](.github/workflows/verify.yml)）在每次 push 与 PR 时自动执行验证门并编译上传 PDF。

### 复现边界与数据说明

- **仓库追踪内容**：LaTeX 稿件源码、断言证据矩阵、审查报告、验证脚本及矢量图源。
- **排除内容说明**：外部风场原始 10 分钟 SCADA 数据与多随机种子重训练大文件依据 `.gitignore` 未纳入 Git 跟踪。
- **证据核查**：关于报道数值的来源出处与详细计算，参见 [`CLAIM_EVIDENCE_MATRIX.md`](CLAIM_EVIDENCE_MATRIX.md) 与 [`PHASE5_SUBMISSION_READINESS_REPORT.md`](PHASE5_SUBMISSION_READINESS_REPORT.md)。

## 关键文档导航

- [断言证据矩阵](CLAIM_EVIDENCE_MATRIX.md) 与 [断言映射图](claim_evidence_map.md)：稿件主张与实验证据的逐条映射。
- [论文架构设计](NEW_MANUSCRIPT_ARCHITECTURE.md)：四层证据金字塔与章节组织说明。
- [阶段五审阅报告](PHASE5_SUBMISSION_READINESS_REPORT.md)：独立审阅人复核核心数值与文字边界的记录。
- [科研契约与工作规范](START_PROMPT.md)：防漂移协议与执行纪律。
- [CI 工作流配置](.github/workflows/verify.yml)：自动化检查入口。

本地开发环境与独立分支可能包含未推送的本地工程记录与临时数据；本页仅对应本远端分支已跟踪的文件集合。
