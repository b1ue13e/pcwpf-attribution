# PCT 原始数据索取记录（P2，2026-09-04）

## 目的

判断我们的 PCT 案例研究能否升级为 exact reproduction，还是必须保留
"transfer stress test" 定位。依据：PCT 仓库 README 明确"数据需向 corresponding
author 索取"，论文正文未附带数据集。

## 论文与作者

- 标题：Physics-constrained transformer for wind power forecasting
- DOI：10.1038/s41598-025-34348-x（Scientific Reports, 2025）
- 作者：Ding Wang, Qiang Luo, Yuntian Chen
- 官方代码：https://github.com/daxin007/PCT-WPF

## 索取信模板（待发送）

---
**Subject: Data request for reproducibility audit of PCT (s41598-025-34348-x)**

Dear Prof. Chen / Dr. Wang,

We are conducting an independent reproducibility and specification-fidelity audit
of the physics-constrained wind power forecasting method described in your
Scientific Reports paper (DOI 10.1038/s41598-025-34348-x) and its official
implementation (github.com/daxin007/PCT-WPF).

To determine whether our results constitute an exact reproduction rather than a
transfer stress test, we respectfully request access to:

1. The `江苏十分钟数据/*.csv` dataset referenced by the repository (25 turbines,
   10-minute SCADA), or an equivalent anonymized subset with the same schema;
2. Confirmation of the training script configuration used in the paper
   (network variant, epochs, learning rate, train/test split, noise injection
   procedure).

We will cite the dataset per your requested terms, or keep it confidential if
required. Please let us know the preferred data-use agreement.

Sincerely,
[Name / Affiliation / Contact]
---

## 当前状态（如实记录）

- 截至 2026-09-04：**尚未发送**（索取信模板已备，待用户确认发送人与机构邮箱）。
- 已核实：论文正文与 README 均无数据链接，仓库不含数据文件。
- 因此当前定位不变：**transfer stress test，非 exact reproduction**。

## 索取结果三支路（预注册）

| 结果 | 后续定位 |
|---|---|
| 获得数据 + 配置 | 重跑 exact reproduction，published vs reproduced 对照 |
| 请求被拒/无回应 | 书面记录索取失败 → 论文明确写"数据不可得，故为 transfer stress test" |
| 获得部分（仅配置）| 可做架构+超参 parity，数据仍不可得 → 半 exact |
