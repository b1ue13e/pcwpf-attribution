# Supplementary Information Architecture and Outline

**Manuscript Title:** Conditional Inductive Guardrails: Identifying When Physical Priors Enhance, Mimic, or Impair Decisions in Wind Power Forecasting  
**Target Journal:** *Applied Energy* (Elsevier)  
**Document Type:** Comprehensive Supplementary Material (Online Appendix)  

---

## 1. Structural Overview of Supplementary Information

The Supplementary Material provides exhaustive empirical ledgers, complete mathematical proofs, software decompilation scripts, and extended sensitivity analyses supporting the findings presented in the main manuscript.

```
Supplementary Material
├── Section S1: PRISMA-ScR Systematic Scoping Search Protocol
├── Section S2: Full 46-Study Extraction Ledger & Inter-Rater Reliability
├── Section S3: Computational Graph, Tensor Couplings & Software AST Audit
├── Section S4: Extended Mathematical Proofs & Counterexample Formulations
├── Section S5: Wind Farm SCADA Telemetry & Market Settlement Mechanics
├── Section S6: Extended Tabular Results, Confidence Intervals & Ablations
└── Section S7: The REFORMS-PCWPF Reporting Checklist & Audit Protocol
```

---

## 2. Section-by-Section Content Specifications

### Section S1: PRISMA-ScR Systematic Scoping Protocol & Search Queries
- **Database Search Strings:** Complete, unabridged search strings for IEEE Xplore, ScienceDirect, Web of Science, and arXiv spanning publication years 2023 to 2026.
  - *Query Syntax:* `("physics-informed" OR "physics-constrained" OR "physics-guided" OR "domain-guided") AND ("wind power" OR "wind speed" OR "turbine power") AND ("deep learning" OR "neural network" OR "transformer")`.
- **Screening & Eligibility Criteria:** Formal two-stage PRISMA screening funnel (title/abstract screening followed by full-text evaluation).
- **14-Study Sensitivity Query:** Specific search formulation and comparative distribution demonstrating that reporting patterns (absence of failure reporting, low code openness) hold robustly under broader keyword variations.

### Section S2: Full 46-Study Extraction Ledger & Multi-Coder Consensus
- **The 21-Field Extraction Ledger:** Detailed definitions for all 21 extracted methodological variables (publication venue, physical prior type, constraint enforcement mechanism, capacity confounding status, baseline strength, negative control implementation, code availability, failure regime reporting).
- **Inter-Rater Reliability Protocol:**
  - Double-blind independent coding by Coder A (Primary Author) and Coder C (Second Author).
  - Unweighted and linear-weighted Cohen's kappa statistics across all 21 fields (overall $\kappa = 0.88$, $95\%$ CI: $[0.82, 0.94]$).
  - Tripartite arbitration logs with verbatim quotes justifying consensus resolutions for all initially discordant coding fields.
- **Full 46-Study Master Table:** Complete tabulated extraction ledger spanning all 46 included studies with exact DOI links.

### Section S3: Computational Graph, Tensor Coupling Derivations & Software AST Audit
- **AST Decompilation Methodology:** Python `ast` parsing script utilized to inspect the public training graph of the Physics-Constrained Transformer (PCT, *Scientific Reports* 2026).
- **Directional Loss Divergence:**
  - Side-by-side mathematical and code comparison between the published formula $D_{\text{JS}}(P_{\text{pred}} \parallel P_{\text{ref}})$ and the executed PyTorch loss `js_div(y_ref, y_pred)`.
  - Detailed mathematical derivation proving that the executed form induces an inverse-density singularity $-\frac{P_{\text{ref}, i}}{4 P_{\text{pred}, i}} \sim \mathcal{O}(1/\epsilon)$, triggering $2.20\times$ to $2.72\times$ gradient norm amplification, whereas Standard-JS exhibits finite upper bounds ($\frac{1}{2}\ln 2$) and logarithmic mildness ($\mathcal{O}(|\log \epsilon|)$).
- **Mathematical Refutation of 2D Product-Kernel Skepticism:** Complete algebraic proof demonstrating that the outer-product kernel $K(x, y) = K_1(x) \otimes K_2(y)$ rigorously maintains bivariate coordinate coupling in reproducing kernel Hilbert spaces, disproving community skepticism and validating its mathematical soundness.

### Section S4: Extended Mathematical Proofs & Counterexample Formulations
- **Proof of Proposition 1 (Prediction-Decision Decoupling):**
  - Analytical derivation demonstrating that for any asymmetric operational loss function $L(y, \hat{y}; r)$, there exist forecasters $f_1, f_2$ such that $\text{RMSE}(f_1) < \text{RMSE}(f_2)$ while $\mathbb{E}[L(f_1)] > \mathbb{E}[L(f_2)]$.
- **Proof of Counterexample 1 (Info-Symmetric Distortion):**
  - Formal proof that symmetric point training objectives incentivize deep networks to bias predictions toward the conditional mean rather than the quantile-optimal asymmetric decision boundary.
- **Proof of Counterexample 2 (Degenerate Zero-Bid Baseline & Cashflow Decoupling):**
  - Mathematical demonstration showing that under the standalone balancing cashflow diagnostic $\Pi_{\text{bal}} = (y - q) \cdot \text{SBP}$, setting $q \equiv 0$ mechanically maximizes apparent cashflow $\Pi_{\text{bal}} = y \cdot \text{SBP} > 0$ under non-negative prices, creating a severe metric inversion artifact.

### Section S5: Telemetry Preprocessing & Electricity Market Mechanics
- **SCADA Preprocessing & Quality Control:**
  - Filtering criteria for curtailment, sensor freezes, negative generation spikes, and pitch-angle maintenance flags across Kelmarsh, Hill of Towie, and Penmanshiel.
  - Normalization parameters, turbine count divisions, and annualization procedures.
- **UK Elexon BSC Settlement Mechanics:**
  - Mathematical definition of System Buy Price (SBP) and System Sell Price (SSP).
  - Step-by-step formulation of the stylized two-settlement forward-plus-imbalance cashflow model versus the unhedged standalone diagnostic.

### Section S6: Extended Tabular Results, Confidence Intervals & Ablation Diagnostics
- **Comprehensive Ablation Tables:** Complete multi-seed metrics (dispatch cost $r=2, q=1/3$, RMSE, MAE, paired bootstrap differences, and Wilcoxon signed-rank test $p$-values) across:
  - Penalty grid: $\lambda \in \{0.00, 0.01, 0.05, 0.10, 0.50, 1.00\}$;
  - Sample sizes: $N_{\text{train}} \in \{200, 500, 1{,}000, 1{,}583, 3{,}000, 5{,}000, 7{,}035\}$;
  - Continuous corruption levels: $\alpha \in [0.0, 1.0]$ in steps of $0.05$;
  - Engineering drift: Turbine aging derating ($0\%$ to $25\%$) and anemometer calibration error ($-20\%$ to $+20\%$);
  - Backbones: DLinear vs. PatchTST vs. Persistence.
- **Statistical Significance Tables:** 10,000-resample paired Student-$t$ and bootstrap confidence intervals for all primary claims.

### Section S7: The REFORMS-PCWPF Reporting Checklist
- **The 5-Gate Actionable Reporting Standard:**
  1. *Gate 1: Parameter-Isolated Ablation Standard* (Mandatory neural capacity parity).
  2. *Gate 2: Form- and Weight-Matched Negative Control Protocol* (Mandatory pseudo-scrambled comparison).
  3. *Gate 3: Information-Budget Alignment* (Mandatory disclosure of prior fitting sample volume vs. model training volume).
  4. *Gate 4: Proxy Staleness & Latency Stress Testing* (Mandatory evaluation under operational telemetry delays).
  5. *Gate 5: Decision-Aligned Dual Settlement Evaluation* (Mandatory coupled market evaluation and prohibition of unhedged cashflow claims).
- **Author Submission Compliance Template & Reviewer Evaluation Rubric.**

---

## 3. Publication and Archival Assets Plan

1. **Compiled PDF Document:** `supplementary_material.pdf` (Clean LaTeX build with booktabs tables and vector figures).
2. **Tabular Data Ledgers (CSV):**
   - `table_s1_prisma_ledger_46studies.csv`
   - `table_s2_coder_agreement_kappa.csv`
   - `table_s3_multi_farm_extended_metrics.csv`
   - `table_s4_information_budget_5seed.csv`
3. **Executable Reproduction Notebooks:**
   - `notebook_s1_ast_code_audit.ipynb`
   - `notebook_s2_market_settlement_simulation.ipynb`
   - `notebook_s3_statistical_bootstrap.ipynb`
