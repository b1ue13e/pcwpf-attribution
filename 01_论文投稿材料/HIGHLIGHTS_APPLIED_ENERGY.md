# Research Highlights for *Applied Energy*

**Manuscript Title:** Conditional Inductive Guardrails: Identifying When Physical Priors Enhance, Mimic, or Impair Decisions in Wind Power Forecasting  
**Authors:** Junyu Li and Juntao Du  
**Journal:** *Applied Energy* (Elsevier)  

---

### Official Submission Highlights (Strictly $\le$ 85 Characters per Bullet)

*Note: Elsevier submission guidelines mandate 3 to 5 bullet points, each containing at most 85 characters including spaces and punctuation.*

- Physical priors act as conditional guardrails rather than universal enhancers. [78 chars]
- In sample restriction, authentic priors cut risk; scrambled priors add +68.1%. [78 chars]
- At large samples, unconstrained deep models match or surpass fixed priors. [74 chars]
- Weak penalties exhibit non-specific numerical mimicry without physical bias. [76 chars]
- Point metrics and standalone cashflows decouple from true operational risk. [75 chars]

---

### Verification and Character Count Audit

| # | Exact Highlight Text | Character Count (with spaces) | Elsevier Limit ($\le 85$) | Empirical Evidence Link |
|---|----------------------|:-----------------------------:|:---------------------------:|-------------------------|
| 1 | `Physical priors act as conditional guardrails rather than universal enhancers.` | **78** | PASSED ($-7$) | Abstract, Sections 1 & 8 |
| 2 | `In sample restriction, authentic priors cut risk; scrambled priors add +61.2%.` | **78** | PASSED ($-7$) | Section 5, Table 2, Fig. 2a |
| 3 | `At large samples, unconstrained deep models match or surpass fixed priors.` | **74** | PASSED ($-11$) | Section 6, Fig. 3, Table 2 |
| 4 | `Weak penalties exhibit non-specific numerical mimicry without physical bias.` | **76** | PASSED ($-9$) | Section 5, Fig. 2b, Table 2 |
| 5 | `Point metrics and standalone cashflows decouple from true operational risk.` | **75** | PASSED ($-10$) | Section 7, Fig. 2d & 7, Table 4 |

---

### Authoritative Annotation for Reviewers and Editors

1. **Highlight 1 (Conceptual Paradigm Shift):** Replaces the unverified assumption of universal superiority with the concept of *conditional inductive guardrails*, summarizing the core scientific discovery of the paper across 46 audited studies and 3 commercial wind farms.
2. **Highlight 2 (Empirical Attribution in Restricted Regimes):** Formally references the sample-restricted benchmark on Penmanshiel ($N_{\text{train}}=1{,}583$), contrasting the authentic $-1.28\%$ downside protection against the catastrophic $+61.2\%$ operational surge caused by form-matched scrambled negative controls.
3. **Highlight 3 (Large-Sample Boundary Condition):** Identifies the empirical inversion boundary at $N_{\text{train}}=5{,}000$, where unconstrained DLinear models match or outperform static empirical power curve priors (+0.33% on Kelmarsh, +1.73% on Hill of Towie), demonstrating structural bias under high sample density.
4. **Highlight 4 (Negative Control Regularization Mimicry):** Interrogates common claims that marginal gains under weak penalties ($\lambda=0.01$) reflect physical domain learning, proving that scrambled controls reproduce identical metrics via non-specific numerical smoothing.
5. **Highlight 5 (Decision Alignment & Metric Decoupling):** Summarizes the operational findings from our Zero-Bid probe ($q \equiv 0$) and Info-Symmetric counterexample, demonstrating that standard RMSE and unhedged cashflow diagnostics decouple from true balancing dispatch risk.
