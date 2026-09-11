# Cover Letter for Submission to *Applied Energy*

**To:**  
Professor J. Yan  
Editor-in-Chief, *Applied Energy*  
Elsevier Science  

**Date:** September 11, 2026  

**Subject:** Submission of Original Research Manuscript: *"Auditing the Decision Value of Physics-Constrained Wind Power Forecasting"*

Dear Professor Yan and Members of the Editorial Board,

We are pleased to submit our original research manuscript entitled **"Auditing the Decision Value of Physics-Constrained Wind Power Forecasting"** for consideration for publication as a Full Research Paper in *Applied Energy*.

### Motivation and Core Relevance to *Applied Energy*
Over the past five years, the integration of physical governing laws—such as aerodynamic power envelopes and wake interaction models—into deep neural network architectures has emerged as one of the most prominent frontiers in renewable energy forecasting. The foundational premise of this paradigm is that domain physical principles supply dependable inductive biases that safeguard generalization and translate into genuine operational decision value. 

However, despite dozens of published architectures claiming superior accuracy, empirical scrutiny has revealed a critical governance gap: published papers exclusively report uniform performance gains, leaving operators and grid planners without knowledge of whether gains truly stem from physical priors, when constraints are neutral, and under what conditions they fail.

### What This Paper Contributes
Rather than proposing another incremental neural architecture, this work delivers the first end-to-end empirical audit of physics-constrained wind power forecasting spanning literature reporting, software execution, and imbalance-settlement and decision-risk diagnostics:

1. **Systematic Scoping Audit ($N=46$ Studies, 2023--2026):** Guided by the PRISMA-ScR protocol across 46 contemporary studies (38 peer-reviewed journal articles and 8 preprints, 2023--2026), supplemented by a 14-study Scopus/IEEE sensitivity query, we document three pervasive reporting patterns: a complete absence of negative trade-off or failure regime documentation ($0.0\%$, 46/46), an acute open-code deficit ($95.7\%$ lacking public code, 95% Wilson CI: [85.5%, 98.8%]), and widespread structural confounding in ablation designs ($50.0\%$ conflating neural submodule capacity with physical domain knowledge, 23/46, 95% Wilson CI: [36.1%, 63.9%]).
2. **Software AST & Computational Graph Audit:** Of the 46 audited studies, exactly two provided public repositories (PINN-UQ with verified code alignment and PCT with divergent implementation); we audited the sole publicly released end-to-end training codebase in this domain (PCT, *Scientific Reports* 2026) via static AST decompilation. We identified a directional implementation divergence in which the executed PyTorch loss reverses the argument order of the published Jensen--Shannon divergence, inducing a $2.20\times$ to $2.72\times$ gradient norm amplification and premature operational degradation under weak regularization. Separately, algebraic tensor derivation confirms that the paired kernel product maintains bivariate coupling via a standard 2D product-kernel formulation, refuting prior community skepticism and providing an essential negative control.
3. **Multi-Farm Controlled Prospective Ablation across Dedicated 10-GPU Cluster:** Leveraging open, commercial-grade 10-minute SCADA records across three UK wind farms (Kelmarsh, Hill of Towie, and Penmanshiel) evaluated across 3--4 independent seeds per farm and coupled with 2017 UK Elexon Balancing and Settlement Code (BSC) price telemetry, we establish key empirical boundary conditions:
   - *Sample Restriction & Prior Misspecification:* Under restricted training samples ($N_{\text{train}}=1{,}583$ on Penmanshiel), an authentic physical prior reduces normalized asymmetric decision loss by $1.22\%$, whereas a form- and weight-matched pseudo-scrambled prior inflates it by $+68.1\%$, exposing acute vulnerability to prior misspecification under strong regularization.
   - *Information Budgets and Learning Curves:* Fixed empirical priors provide modest benefits under restricted model training windows ($N \le 1,000$). Evaluating a controlled ablation under matched sample windows at $N=500$ indicates that domain-structured empirical prior effects under a matched temporal data window retain most of the mean downside risk reduction observed with the full-history prior (approximately $77\%$ and $92\%$ in the two sites), while the seasonal historical pool supplies an incremental gain. Expanding model training windows to $N=5{,}000$ closes the gap and slightly reverses the ranking (+0.33% on Kelmarsh, +1.73% on Hill of Towie).
   - *Numerical Regularization Mimicry:* At weak penalty scales ($\lambda \le 0.01$), genuine and scrambled priors yield near-identical dispatch costs, demonstrating that marginal gains reflect non-specific numerical smoothing rather than aerodynamic reasoning.
   - *Training Proxy Misalignment:* Lagged historical weather proxies in the offline training objective monotonically inflate dispatch costs as proxy delay increases from 1 h to 6 h.
   - *Metric Decoupling & The Zero-Bid Diagnostic:* We uncover an evaluation artifact in unhedged balancing market cashflows, demonstrating through a degenerate Zero-Bid probe ($q \equiv 0$) that surface cashflows decouple from forecasting capability.
4. **Actionable Peer-Review Governance:** We operationalize our findings into a five-gate REFORMS-PCWPF reporting checklist (parameter-isolated ablations, form- and weight-matched negative controls, proxy staleness stress-tests, coupled forward-plus-imbalance evaluations, and computational graph parity) to advance reproducible, trustworthy renewable forecasting.

### Author Confirmations
- This manuscript represents original work that has not been published previously and is not currently under consideration for publication elsewhere.
- All authors have approved the manuscript and agree with its submission to *Applied Energy*.
- All operational SCADA datasets (Cubico / TRIG / RES via Zenodo) and market price records (Elexon) utilized in this work are fully open-access under CC-BY 4.0 and UK OGL v3.0 licenses.
- The complete Python verification suite, AST parsers, and data extraction ledgers are openly archived with full reproducibility guarantees.

Given *Applied Energy*'s dedicated readership in energy systems informatics, wind power analytics, and power market operations, we believe this manuscript will generate immediate interest and serve as a cornerstone reference for researchers and industrial practitioners seeking to deploy trustworthy physics-informed AI in safety-critical power systems.

Thank you very much for your time, consideration, and editorial leadership.

Sincerely,

**Junyu Li** (Corresponding Author)  
School of Electrical and Information Engineering  
Email: `ljylikezmn999@gmail.com`  

**Juntao Du**  
School of Electrical and Information Engineering  
