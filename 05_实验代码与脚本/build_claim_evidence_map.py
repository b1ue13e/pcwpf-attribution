"""
Generate 06_claim_evidence_map.csv
Audit of every core claim across Abstract, Intro, Results, Discussion, and Conclusion.
"""
import csv
from pathlib import Path

OUT_DIR = Path('e:/论文1/revision_outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

claims = [
    {
        'Claim_ID': 'CLM-01',
        'Manuscript_Location': 'Abstract / Sec 5.1 / Highlights',
        'Original_Claim_Text': 'Physics-constrained wind power forecasting literature contains zero documented negative trade-offs (0.0%).',
        'Evidence_Level': 'Level 1 (Direct Observation)',
        'Scope_Correction_Needed': 'YES',
        'Calibrated_Claim_Text': 'Within the audited scoping cohort of 46 studies (2023-2026), zero studies documented operational negative trade-offs (0/46, 0.0%; 95% Wilson CI: [0.0%, 7.7%]).',
        'Raw_Evidence_Source': '06_实验结果与日志/constraint_round2/analysis/corpus46_scoping_ledger.csv',
        'Audit_Verdict': 'APPROVED_CALIBRATED'
    },
    {
        'Claim_ID': 'CLM-02',
        'Manuscript_Location': 'Abstract / Sec 5.2 / Highlights',
        'Original_Claim_Text': 'Literature suffers from an alarming 95.7% code deficit and software untrustworthiness.',
        'Evidence_Level': 'Level 1 (Direct Observation)',
        'Scope_Correction_Needed': 'YES',
        'Calibrated_Claim_Text': 'In the audited 46-study cohort, 44 studies (95.7%) provided no accessible public code repository, impeding independent computational verification.',
        'Raw_Evidence_Source': '06_实验结果与日志/constraint_round2/analysis/corpus46_scoping_ledger.csv (D4_fidelity)',
        'Audit_Verdict': 'APPROVED_CALIBRATED'
    },
    {
        'Claim_ID': 'CLM-03',
        'Manuscript_Location': 'Sec 5.3 / Table 1',
        'Original_Claim_Text': 'Half of all studies confound capacity with physics, proving reported gains are false.',
        'Evidence_Level': 'Level 2 (Supported Interpretation)',
        'Scope_Correction_Needed': 'YES',
        'Calibrated_Claim_Text': 'Half of the audited studies (23/46, 50.0%) evaluate constraints by excising multi-layer neural submodules rather than toggling penalty multipliers, confounding parametric capacity with physical inductive guidance.',
        'Raw_Evidence_Source': '06_实验结果与日志/constraint_round2/analysis/corpus46_scoping_ledger.csv (ablation_type)',
        'Audit_Verdict': 'APPROVED_CALIBRATED'
    },
    {
        'Claim_ID': 'CLM-04',
        'Manuscript_Location': 'Abstract / Sec 5.4 / Fig 2a',
        'Original_Claim_Text': 'Strong constraints reduce cost by 1.22% while scrambled priors trigger a catastrophic 68.1% surge on Penmanshiel.',
        'Evidence_Level': 'Level 3 (Causal Identification under Matched Backbone)',
        'Scope_Correction_Needed': 'MINOR (Metric precision)',
        'Calibrated_Claim_Text': 'On Penmanshiel under data-scarce training (N_train=1,583), embedding the authentic power curve prior (lambda=1.0) reduces mean normalized asymmetric decision loss by 1.22% (0.09760 vs 0.09881), whereas disrupting the wind-to-power mapping via a pseudo-scrambled control inflates decision loss by 68.1% (0.16611), isolating prior specificity under strong regularization.',
        'Raw_Evidence_Source': '06_实验结果与日志/ablation_3d_results/ablation_3d_key_comparison_table.csv (Rows 48, 63, 67)',
        'Audit_Verdict': 'APPROVED_CALIBRATED'
    },
    {
        'Claim_ID': 'CLM-05',
        'Manuscript_Location': 'Sec 5.4 / Fig 2b',
        'Original_Claim_Text': 'Weak penalties prove that physical gains are purely numerical smoothing.',
        'Evidence_Level': 'Level 2 (Supported Interpretation)',
        'Scope_Correction_Needed': 'YES',
        'Calibrated_Claim_Text': 'At weak regularization (lambda=0.01) on Kelmarsh, genuine physical curve priors and pseudo-scrambled controls yield closely matched decision costs (0.08810 +- 0.00009 vs 0.08801 +- 0.00008). This parity indicates that marginal gains observed at weak penalty scales cannot be specifically attributed to aerodynamic principles and remain consistent with non-specific numerical smoothing.',
        'Raw_Evidence_Source': '06_实验结果与日志/ablation_3d_results/ablation_3d_key_comparison_table.csv (Rows 24, 39, 45)',
        'Audit_Verdict': 'APPROVED_CALIBRATED'
    },
    {
        'Claim_ID': 'CLM-06',
        'Manuscript_Location': 'Sec 5.4 / Fig 3',
        'Original_Claim_Text': 'Physical priors are universally superior under small sample sizes and fail under big data due to hypothesis-space saturation.',
        'Evidence_Level': 'Level 2 (Supported Interpretation)',
        'Scope_Correction_Needed': 'YES',
        'Calibrated_Claim_Text': 'Within-site learning curves across N_train in [500, 5000] show that fixed historical physical priors lower mean dispatch costs when model training windows are restricted (N_train <= 1000, -2.00% on Kelmarsh, -3.62% on Hill of Towie), whereas expanding model training windows to N_train = 5000 closes this margin (+0.33% on Kelmarsh, +1.73% on Hill of Towie). We emphasize that the empirical prior was estimated over the entire pre-cutoff history, providing an extra seasonal data budget that assists restricted model training windows.',
        'Raw_Evidence_Source': '06_实验结果与日志/learning_curve/within_site_learning_curve_summary.csv',
        'Audit_Verdict': 'APPROVED_CALIBRATED'
    },
    {
        'Claim_ID': 'CLM-07',
        'Manuscript_Location': 'Sec 5.4 / Fig 2c',
        'Original_Claim_Text': 'Test-time telemetry delay causes monotonic performance failure.',
        'Evidence_Level': 'Level 2 (Supported Interpretation)',
        'Scope_Correction_Needed': 'YES (P0 Fatal Error Corrected)',
        'Calibrated_Claim_Text': 'Evaluating proxy staleness in the training objective (scanning historical wind proxy lag k in [6, 36]) produces monotonic cost inflation across all three farms. This experiment isolates the sensitivity of the training loss to temporal proxy misalignment rather than runtime inference delay, as the forecaster takes only past power at test time.',
        'Raw_Evidence_Source': '05_实验代码与脚本/constraint_round2/run_cluster_ablation.py:53',
        'Audit_Verdict': 'APPROVED_CALIBRATED'
    },
    {
        'Claim_ID': 'CLM-08',
        'Manuscript_Location': 'Sec 5.5 / Table 4 / Fig 4a',
        'Original_Claim_Text': 'Zero-Bid strategy makes £243k commercial profit proving GB market is broken.',
        'Evidence_Level': 'Level 1 (Direct Diagnostic Calculation)',
        'Scope_Correction_Needed': 'YES (Institutional & Terminology Overhaul)',
        'Calibrated_Claim_Text': 'A degenerate zero-bid baseline (y_hat = 0) incurs zero shortfall penalty and generates an apparent unhedged balancing cashflow of +£223k to +£243k/(turb*yr) across all three farms by leaving all physical generation uncontracted. This result serves as a diagnostic reductio ad absurdum demonstrating that standalone unhedged balancing cashflow is structurally decoupled from forecasting skill, and cannot serve as an index of forecast quality or commercial value.',
        'Raw_Evidence_Source': '08_报告与说明文档/mainline_audit_20260909/verified_evidence.json & Table 4 recomputation',
        'Audit_Verdict': 'APPROVED_CALIBRATED'
    },
    {
        'Claim_ID': 'CLM-09',
        'Manuscript_Location': 'Sec 5.5 / Fig 4b',
        'Original_Claim_Text': 'Distribution constraints prove cashflow inversion on Hill of Towie.',
        'Evidence_Level': 'Level 1 & 2 (Direct Observation & Interpretation)',
        'Scope_Correction_Needed': 'MINOR',
        'Calibrated_Claim_Text': 'On Hill of Towie, distribution regularization increases nominal balancing cashflow by +42.5% (£8,543 to £12,174/(turb*yr)) while concurrently degrading normalized decision risk by +6.5% and point RMSE by +3.5%, exposing empirical metric inversion where apparent monetary gains arise from negative forecast bias during positive price periods rather than physical accuracy.',
        'Raw_Evidence_Source': '06_实验结果与日志/ablation_3d_results/ablation_3d_key_comparison_table.csv (Rows 6, 23)',
        'Audit_Verdict': 'APPROVED_CALIBRATED'
    },
    {
        'Claim_ID': 'CLM-10',
        'Manuscript_Location': 'Sec 5.2 / Sec 6',
        'Original_Claim_Text': 'PCT code bug creates singular gradient explosions in all Transformer training.',
        'Evidence_Level': 'Level 2 (Supported Interpretation of Code Audit)',
        'Scope_Correction_Needed': 'YES',
        'Calibrated_Claim_Text': 'Software inspection of the official Physics-Constrained Transformer repository (PCT, S02) reveals an inverted parameter argument in F.kl_div(input, target) that computes a reverse-argument JS divergence (1.614 nats vs standard JS bound of 0.693 nats), altering backpropagated gradient scales by approximately 2.5x. While local density gradient divergence is analytically demonstrable under sharp support boundaries, its impact in end-to-end training remains modulated by empirical gradient clipping.',
        'Raw_Evidence_Source': '08_报告与说明文档/mainline_audit_20260909/verified_evidence.json (js_diagnostic)',
        'Audit_Verdict': 'APPROVED_CALIBRATED'
    },
    {
        'Claim_ID': 'CLM-11',
        'Manuscript_Location': 'Sec 7',
        'Original_Claim_Text': 'Proposition 1 is a new decision theory proving prediction-decision inversion.',
        'Evidence_Level': 'Level 3 (Decision-Theoretic Lemma)',
        'Scope_Correction_Needed': 'YES (De-escalated to Lemma)',
        'Calibrated_Claim_Text': 'Lemma 1 (Decision-Theoretic Operational Baseline): Under asymmetric linear decision loss with penalty ratio r, the risk-minimizing operational bid corresponds to the conditional quantile q = 1/(1+r). This classical result (Koenker 2001, Pinson 2007) is operationalized here to define the post-hoc quantile calibration baseline against which physical regularizers are audited.',
        'Raw_Evidence_Source': 'Koenker (2001), Pinson et al. (2007)',
        'Audit_Verdict': 'APPROVED_CALIBRATED'
    },
    {
        'Claim_ID': 'CLM-12',
        'Manuscript_Location': 'Sec 7',
        'Original_Claim_Text': 'Decomposition identifies Term III as the exact causal mechanism for cross-site differences.',
        'Evidence_Level': 'Level 2 (Diagnostic Analytical Decomposition)',
        'Scope_Correction_Needed': 'YES',
        'Calibrated_Claim_Text': 'The three-term risk decomposition separates Point MSE (Term I), Quantile Mismatch (Term II), and Residual Interaction Cov(f - mu, delta - Delta_tau) (Term III). Term III provides an analytical framework consistent with observed trade-offs, but is not separately identified as an independent empirical mechanism in the current test bench. Counterexamples 1 and 2 demonstrate the analytical possibility of metric inversion rather than causal proof for individual wind farms.',
        'Raw_Evidence_Source': 'Analytical Derivation in Sec 7',
        'Audit_Verdict': 'APPROVED_CALIBRATED'
    }
]

# Write CSV
csv_path = OUT_DIR / '06_claim_evidence_map.csv'
fieldnames = list(claims[0].keys())
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(claims)

print(f"Generated {csv_path} with {len(claims)} calibrated claims.")
