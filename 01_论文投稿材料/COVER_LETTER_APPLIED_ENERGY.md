# Cover Letter for Submission to *Applied Energy*

**To:**  
Professor J. Yan  
Editor-in-Chief, *Applied Energy*  
Elsevier Science  

**Date:** September 14, 2026  

**Subject:** Submission of Original Research Manuscript: *"Conditional Inductive Guardrails: Identifying When Physical Priors Enhance, Mimic, or Impair Decisions in Wind Power Forecasting"*  

**Corresponding Author:**  
Junyu Li  
School of Statistics and Applied Mathematics  
Anhui University of Finance and Economics, Bengbu 233030, China  
Email: `ljylikezmn999@gmail.com`  

**Co-Author:**  
Juntao Du  
School of Statistics and Applied Mathematics  
Anhui University of Finance and Economics, Bengbu 233030, China  

---

Dear Professor Yan and Members of the Editorial Board,

We are pleased to submit our original research article entitled **"Conditional Inductive Guardrails: Identifying When Physical Priors Enhance, Mimic, or Impair Decisions in Wind Power Forecasting"** for consideration for publication as a Full Research Paper in *Applied Energy*.

### Executive Overview & Core Selling Proposition

Over the past five years, integrating physical governing principles—such as aerodynamic turbine power curves and wake interaction physics—into deep neural forecasting backbones has emerged as a prominent frontier in energy systems engineering and machine learning for power systems. The existing physics-constrained forecasting literature almost uniformly reports positive performance gains. However, published works systematically lack controlled experimental designs capable of separating genuine physical prior effects from generic numerical regularization, auxiliary historical data budgets, proxy misalignments, and evaluation metric artifacts.

Rather than introducing another incremental neural forecasting architecture, this manuscript presents **a controlled attribution and empirical identification study identifying the decision value and failure boundaries of physical priors in operational wind power forecasting**. In this work, by establishing form-matched negative controls (pseudo-scrambled priors), information-budget matching, proxy latency ladders, and decision-aligned market stress tests, we provide the identifiable operating boundaries of physical priors in power systems. Holding forecasting backbones, parameter capacities, data splits, and post-hoc quantile calibrations strictly constant across three commercial wind farms coupled with 2017 UK Elexon Balancing and Settlement Code (BSC) market clearing prices, we delineate the explicit operating envelope within which physical inductive biases serve as indispensable guardrails, when they merely mimic numerical smoothers, and where they induce structural bias or operational penalties.

### Scope Alignment with *Applied Energy*

*Applied Energy* explicitly covers renewable energy forecasting models, artificial intelligence applications in energy systems, energy economics and electricity markets, power system operations, and optimization and decision-making under uncertainty. Our manuscript directly addresses the intersection of these core themes:
1. **Forecasting Models & AI in Energy Systems:** We audit contemporary physics-constrained transformer and deep recurrent models, establishing rigorous negative-control benchmarks and identifying implementation pitfalls in executed loss formulations.
2. **Decision-Aligned Optimization & Energy Markets:** We evaluate forecasts through dynamic two-settlement electricity market cashflows, demonstrating why symmetric point metrics (such as RMSE) systematically decouple from asymmetric operational imbalance risks.
3. **Power System Operation & Safety Boundaries:** We delineate data scarcity, sensor drift, and proxy latency thresholds beyond which nominally "physics-informed" models fail in real-world grid operations.

### Key Scientific Contributions

1. **Systematic Scoping Audit ($N=46$ Studies, 2023–2026):** Guided by the PRISMA-ScR protocol across 46 contemporary studies (38 peer-reviewed journal articles and 8 preprints, supplemented by a 14-study sensitivity query), we identify three pervasive governance vulnerabilities:
   - *Complete Absence of Negative Failure Reporting:* $0.0\%$ (0/46) documented negative trade-offs, operational hazards, or failure regimes.
   - *Severe Open-Code Deficit:* $95.7\%$ (44/46) withheld end-to-end training and inference scripts ($95\%$ Wilson CI: $[85.5\%, 98.8\%]$).
   - *Structural Ablation Confounding:* $50.0\%$ (23/46) confounded neural submodule capacity shifts with genuine domain physical knowledge ($95\%$ Wilson CI: $[36.1\%, 63.9\%]$).

2. **Computational Graph & AST Software Audit:** Auditing the sole publicly released end-to-end training codebase in this domain via static abstract syntax tree (AST) decompilation, we identified a directional implementation divergence in which executed PyTorch code reverses the arguments of the published Jensen–Shannon divergence, inducing a $2.20\times$ to $2.72\times$ gradient norm surge and premature degradation under weak regularization. Conversely, rigorous tensor derivations refute community skepticism by confirming that published paired product kernels mathematically preserve bivariate coordinate coupling.

3. **Multi-Farm Controlled Empirical Identification Protocol:** Utilizing open-access commercial SCADA telemetry across three UK wind farms (Kelmarsh, Hill of Towie, and Penmanshiel) across multi-seed training runs and real BSC half-hourly imbalance prices:
   - *Prior Specificity under Sample Restriction:* On Penmanshiel under restricted sample volume ($N_{\text{train}} = 1{,}583$), authentic aerodynamic power curve priors reduce normalized asymmetric dispatch risk by $-1.28\%$, whereas form- and weight-matched pseudo-scrambled controls trigger a $+61.2\%$ surge in operational dispatch costs, establishing genuine prior specificity.
   - *Information-Budget Decoupling:* Implementing a strictly budget-matched information control ($N_{\text{prior}} = N_{\text{train}} = 500$) reveals that domain geometric priors fitted strictly on the contemporaneous sample retain $77\%$ to $92\%$ of the downside risk reduction achieved with the full multi-month archive, proving that benefits are predominantly driven by inductive domain geometry rather than hidden historical data volume.
   - *Large-Sample Regime Inversion:* As training data expand to $N = 5{,}000$, unconstrained data-driven backbones match or slightly outperform fixed priors ($+0.33\%$ on Kelmarsh, $+1.73\%$ on Hill of Towie), demonstrating that static constraints act as inductive guardrails in data-restricted regimes but induce structural bias when sample density suffices.
   - *Weak-Penalty Regularization Mimicry:* Under weak penalty multipliers ($\lambda = 0.01$), authentic and structurally scrambled aerodynamic curves produce indistinguishable dispatch costs, proving that marginal gains cited in prior literature reflect non-specific numerical smoothing rather than domain physical reasoning.
   - *Proxy Staleness Stress-Testing:* Replacing contemporaneous wind observations in offline training losses with lagged weather proxies monotonically degrades dispatch performance, crossing the unconstrained baseline by $k = 1$\,h on Hill of Towie.
   - *Prediction-Decision Decoupling & The Zero-Bid Probe:* We formally show that symmetric point accuracy (RMSE) decouples from asymmetric operational loss. Introducing a degenerate Zero-Bid probe ($q \equiv 0$), we reveal that unhedged standalone balancing cashflows mechanically reward severe strategic under-nomination, generating six-figure apparent trading revenues without forecasting skill.

4. **Actionable Governance via REFORMS-PCWPF Checklist:** We operationalize these findings into a five-gate reporting checklist for authors, reviewers, and editors, establishing standards for negative-control reporting, information-budget matching, proxy staleness disclosure, and computational graph parity.

---

### Suggested Independent Reviewers

To ensure rigorous, authoritative, and balanced peer review, we suggest the following recognized experts who possess deep expertise in wind power forecasting, machine learning in energy systems, and electricity market operations. None of the nominated reviewers have co-authored publications, collaborated on research projects, or have any institutional conflicts of interest with the authors.

1. **Prof. Pierre Pinson**  
   *Chair in Data-centric Engineering, Dyson School of Design Engineering, Imperial College London, UK*  
   *Email:* `p.pinson@imperial.ac.uk`  
   *Field of Expertise:* Wind power forecasting, probabilistic renewable modeling, electricity market integration, decision-focused energy analytics.  
   *Rationale:* Prof. Pinson is an internationally renowned pioneer in market-aligned renewable forecasting and decision-risk modeling (e.g., landmark works on asymmetric penalties and market clearing). He is exceptionally qualified to evaluate our two-settlement market formulations, decision-aligned loss functions, and prediction-decision decoupling analyses.

2. **Prof. Ning Zhang**  
   *Professor, Department of Electrical Engineering, Tsinghua University, Beijing, China*  
   *Email:* `ningzhang@tsinghua.edu.cn`  
   *Field of Expertise:* Physics-informed machine learning in power systems, renewable energy forecasting, power system operations.  
   *Rationale:* Prof. Zhang has published extensively on integrating domain physical constraints into deep learning models for energy systems. He possesses the domain depth necessary to evaluate our controlled attribution protocol, negative-control permutations, and lifecycle constraint taxonomy.

3. **Prof. Henrik Madsen**  
   *Professor, Department of Applied Mathematics and Computer Science (DTU Compute), Technical University of Denmark, Lyngby, Denmark*  
   *Email:* `hmad@dtu.dk`  
   *Field of Expertise:* Grey-box physical modeling, wind power prediction, stochastic time series, renewable integration.  
   *Rationale:* Prof. Madsen is a recognized world authority on combining physical governing equations with statistical time-series models. His expertise in grey-box model identification makes him ideal to evaluate our information-budget controls and weak-penalty numerical regularization mimicry findings.

4. **Prof. Bri-Mathias Hodge**  
   *Professor, Department of Electrical, Computer and Energy Engineering, University of Colorado Boulder; Chief Scientist, National Renewable Energy Laboratory (NREL), USA*  
   *Email:* `bri-mathias.hodge@colorado.edu`  
   *Field of Expertise:* Wind and solar integration, operational grid analytics, renewable forecasting benchmarks.  
   *Rationale:* Dr. Hodge leads major international benchmark initiatives on power system operations and forecasting evaluation. He can authoritatively assess our engineering drift stress tests, sensor calibration tolerances, and practical grid reliability implications.

5. **Prof. Yi Wang**  
   *Assistant Professor, Department of Electrical and Electronic Engineering, The University of Hong Kong (HKU), Hong Kong SAR*  
   *Email:* `yiwang@eee.hku.hk`  
   *Field of Expertise:* Data-driven power systems, machine learning reproducibility, energy informatics.  
   *Rationale:* Prof. Wang has extensive publications in *Applied Energy* on deep learning models and data-driven methods for renewable integration. He is ideally suited to evaluate our open-code audit, software AST verification, and multi-seed prospective empirical benchmarks.

---

### Reviewer Exclusion Request (Authors of Examined Open-Source Implementation)

To ensure strict scientific impartiality and standard peer-review objectivity, we respectfully request the exclusion of authors whose publicly released implementation is directly examined in the manuscript:
- Specifically, because our manuscript conducts an empirical, architectural, and mathematical code audit of published open-source software in this domain (the Physics-Constrained Transformer, PCT), we respectfully request that direct co-authors or collaborating members from the research group that developed the examined open-source repository (Shenyang Institute of Automation, Chinese Academy of Sciences) be excluded as peer reviewers. This request is made solely because authors whose publicly released implementation is directly examined in the manuscript have an inherent conflict of interest under standard Elsevier editorial guidelines, and is intended solely to preserve independent evaluation.

---

### Mandatory Author Declarations & Integrity Assurances

- **Originality:** This manuscript represents original research that has not been published previously, is not under consideration for publication elsewhere, and will not be submitted elsewhere while under review at *Applied Energy*.
- **Author Approvals:** All authors have approved the final manuscript and mutually agreed to its submission to *Applied Energy*.
- **Open Data & Reproducibility:** All operational SCADA telemetry (Cubico / TRIG / RES via Zenodo) and market price records (Elexon) utilized in this work are fully open-access under CC-BY 4.0 and UK Open Government Licence v3.0. Complete Python verification scripts, AST parsers, seed-wise logs, and multi-coder extraction ledgers are openly archived with complete reproducibility documentation.
- **AI Disclosure:** Generative AI tools (OpenAI ChatGPT and Google Gemini) were utilized strictly for structural formatting, grammar checking, and code linting assistance. All scientific analyses, derivations, data syntheses, and interpretations were performed, verified, and authorized solely by the authors, who assume full responsibility for the integrity of the work.

Thank you very much for your time, consideration, and editorial leadership. We look forward to receiving the reviewers' constructive feedback.

Sincerely,

**Junyu Li** (Corresponding Author)  
School of Statistics and Applied Mathematics  
Anhui University of Finance and Economics, Bengbu 233030, China  
Email: `ljylikezmn999@gmail.com`  

**Juntao Du**  
School of Statistics and Applied Mathematics  
Anhui University of Finance and Economics, Bengbu 233030, China  
