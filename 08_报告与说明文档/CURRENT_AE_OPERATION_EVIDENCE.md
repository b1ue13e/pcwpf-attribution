# Current AE Operation Evidence

当前用于主稿与 AE discussion 的 operation 证据包，统一以：

- [results_export_cfg_lower_corr_20260329_012420](/F:/论文1/results_export_cfg_lower_corr_20260329_012420)

为唯一 canonical source。

## 可直接对应主稿的文件

- [operation_high_risk_summary_seed42.csv](/F:/论文1/results_export_cfg_lower_corr_20260329_012420/stage1_summary_cfg_lower_corr/operation_high_risk_summary_seed42.csv)
- [operation_decoupling_summary_seed42.csv](/F:/论文1/results_export_cfg_lower_corr_20260329_012420/stage1_summary_cfg_lower_corr/operation_decoupling_summary_seed42.csv)
- [operation_seed_stability_mean_std.csv](/F:/论文1/results_export_cfg_lower_corr_20260329_012420/stage1_summary_cfg_lower_corr/operation_seed_stability_mean_std.csv)
- [seed_level_scoreboard.csv](/F:/论文1/results_export_cfg_lower_corr_20260329_012420/seed_level_scoreboard.csv)
- [config_level_scoreboard.csv](/F:/论文1/results_export_cfg_lower_corr_20260329_012420/config_level_scoreboard.csv)

## 当前结论

- 这套 `cfg_lower_corr` 结果已经修复了先前 `Ultra-LSNT + FG-MoE` 导出预测在多 seed 下失稳的问题。
- 这套结果与主稿 [main_PI-MoE_final_AE_ready.tex](/F:/论文1/main_PI-MoE_final_AE_ready.tex) 和 AE 文本 [AE_results_discussion_final_20260328_en.md](/F:/论文1/AE_results_discussion_final_20260328_en.md) 中的 operation 数字一致。
- 当前可支持的口径仍然是：
  - `controller-side admissibility`
  - `fixed-interface burden reduction`
- 当前仍不支持的更强口径包括：
  - `universal economic superiority`
  - `interface-independent ranking`
  - `uncertainty-aware operational valuation`

## 不再混用的旧结果包

以下旧包不要再作为主稿 operation 数字来源：

- [upload_results_ae_operation_20260328_104442](/F:/论文1/upload_results_ae_operation_20260328_104442)

原因是它与主稿现行数字并不完全一致，容易再次造成“正文数字和 summary csv 不可追溯”的问题。
