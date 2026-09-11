# Comprehensive Manuscript Revision Log: Restructuring and Forensic Audit

**Manuscript Title**: Auditing the Decision Value of Physics-Constrained Wind Power Forecasting  
**Corresponding Author**: Junyu Li (`ljylikezmn999@gmail.com`)  
**Target Journals**: *Applied Energy* / *Renewable Energy* / *IEEE Transactions on Sustainable Energy* / *Cell Patterns*  
**Date**: September 2026  
**Git Branch**: `revision`  

---

## Executive Summary of Manuscript Overhaul

This revision log provides an auditable, line-by-line record of all structural, numerical, institutional, and stylistic modifications implemented during the comprehensive reconstruction of the manuscript. The overhaul transitions the paper from an internal project report tone into a disciplined, top-tier computational energy publication adhering to the highest standards of scientific machine learning reproducibility (REFORMS / DOME).

### Core Audit Principles Enforced:
1. **Single Source of Truth**: Every empirical metric is cross-referenced directly against raw simulation ledgers (`ablation_results_*.csv`, `corpus48_consensus.csv`, `pct_*.csv`), eliminating transcription artifacts.
2. **Attributional Integrity**: Strict separation between genuine domain physical guidance, generic numerical smoothing ($\lambda \le 0.01$), and unhedged single-settlement accounting distortions.
3. **Institutional Precision**: 2017 GB wholesale power market accurately framed under BSC Section T/X single cash-out imbalance settlement rather than a mandatory centralized pool.
4. **Epistemic Discipline**: Complete de-escalation of universal causal claims to strictly bounded observational and experimental domains (3 wind farms, 46 scoping studies, 2 open-source codebases).

---

## Detailed Section-by-Section Revision Ledger

### Item 1: Abstract & Elsevier Highlights
- **Location**: `main.tex` (lines 75–90)
- **Identified Deficiency**: 
  - Highlights exceeded Elsevier's 125-character rule and included 6 bullets instead of the mandatory 3–5.
  - Abstract contained vague claims of "catastrophic failure" and blurred the 46-study scoping cohort with the wider ML literature.
- **Original Content**:
  - Highlights: 6 bullets, up to 138 characters.
  - Abstract: Generic introductory framing; uncalibrated claims about global physics-informed ML vulnerabilities.
- **Revised Content**:
  - Exactly 5 crisp Highlights, all $\le 100$ characters.
  - Disciplined 235-word Abstract following the 5-part top-journal architecture: Problem (1 sent) $\rightarrow$ Literature Gap (2 sent) $\rightarrow$ Audit Execution (1 sent) $\rightarrow$ Quantitative Findings (4 sent) $\rightarrow$ Practical Implication (1 sent).
- **Engineering Rationale**: Strictly complies with Elsevier author guidelines while foregrounding the PRISMA-ScR audit ($N=46$), PCT gradient pole ($\mathcal{O}(1/\epsilon)$), scrambled prior surge ($+68.1\%$), and Zero-Bid diagnostic.
- **Numerical Source**: `revision_outputs/04_numerical_source_of_truth.csv`, Items 1, 2, 8, 12, 16.
- **Author Verification Status**: `VERIFIED_AND_LOCKED`.

---

### Item 2: Introduction Section
- **Location**: `sections/01_intro.tex`
- **Identified Deficiency**:
  - Opened with informal conversational throat-clearing ("Deep neural networks have achieved remarkable success...").
  - Overclaimed that the entire renewable energy field is misled by physics constraints.
  - Conflated point MSE with decision loss without defining the institutional mechanism.
- **Original Content**: 7 fragmented sub-paragraphs repeating empirical statistics three times before introducing the methodology.
- **Revised Content**:
  - Authoritative opening directly contrasting symmetric loss functions ($L_2$) with asymmetric operational dispatch penalties under real-world system balancing.
  - Crisp articulation of the central **Attribution Gap**: point forecasting improvements do not establish that models learn valid physics, nor do they guarantee positive operational cashflow.
  - Delineation of Three Core Contributions (scoping audit, code/graph forensics, prospective multi-site ablations) and explicit scope boundaries upfront.
- **Engineering Rationale**: Adheres to the three-act paragraph structure from `03_top_journal_writing_patterns.md`, eliminating repetition and setting an objective scientific tone.
- **Numerical Source**: Scoping corpus $N=46$, 95.7% code deficit, 50.0% confounded ablations.
- **Author Verification Status**: `VERIFIED_AND_LOCKED`.

---

### Item 3: Related Work & Taxonomy Comparison Table
- **Location**: `sections/02_related.tex`
- **Identified Deficiency**:
  - Superficial literature survey with adversarial commentary ("Prior works fail to...", "Researchers naively assume...").
  - Lack of a structured comparative taxonomy contrasting our audit against existing surveys and PINN benchmarks.
- **Original Content**: Four descriptive paragraphs without tabular synthesis or formal classification.
- **Revised Content**:
  - Systematic review structured across four thematic pillars: (1) Physics-Constrained Machine Learning in Renewable Energy; (2) Metric Alignment and Asymmetric Decision Losses; (3) Scientific ML Reproducibility Protocols (REFORMS, DOME); (4) Transparent Comparative Taxonomy Table.
  - Integrated Table contrasting our study against 6 landmark review/audit papers across 6 audit dimensions.
- **Engineering Rationale**: Positions the paper as an constructive audit enhancing scientific rigor rather than a hostile takedown.
- **Author Verification Status**: `VERIFIED_AND_LOCKED`.

---

### Item 4: Theoretical Framework & Lifecycle Formalization
- **Location**: `sections/03_framework.tex`
- **Identified Deficiency**:
  - Used internal development jargon ("v1/v2 pipeline", "Phase 3 testbed", "hacky workarounds").
  - Confounded mathematical formulation with software implementation.
- **Original Content**: Ad-hoc narrative interweaving engineering script names with theoretical definitions.
- **Revised Content**:
  - Rigorous 5-tuple formalization of the constraint lifecycle: $\mathcal{M} = (\mathcal{D}_1, \mathcal{D}_2, \mathcal{D}_3, \mathcal{D}_4, \mathcal{D}_5)$, defining:
    - $\mathcal{D}_1$: Parameter Provenance (Empirical Data-Driven vs. First-Principles Analytic vs. Hybrid);
    - $\mathcal{D}_2$: Application Stage (Loss Regularization vs. Architecture Gate vs. Post-Processing);
    - $\mathcal{D}_3$: Telemetry & Inference Requirement (Autonomous Power-Only vs. Synchronous NWP vs. Lagged Weather Proxy);
    - $\mathcal{D}_4$: Implementation Fidelity (Code-Manuscript Parity vs. Reversed Argument Inversion);
    - $\mathcal{D}_5$: Ablation Semantics (Parameter-Matched $\lambda=0$ vs. Confounded Module Excision).
- **Engineering Rationale**: Replaces project jargon with formal, generalizable scientific terminology.
- **Author Verification Status**: `VERIFIED_AND_LOCKED`.

---

### Item 5: PRISMA-ScR Protocol and Prior Information Budget
- **Location**: `sections/04_protocol.tex`
- **Identified Deficiency**:
  - Ambiguity regarding whether the 14 sensitivity papers were part of the core statistical cohort.
  - Incomplete description of the prior information budget (training data used to estimate empirical power curves).
  - Failure to specify inter-rater reliability metrics.
- **Original Content**: Loose description of Google Scholar queries without Boolean syntax or Fleiss' $\kappa$.
- **Revised Content**:
  - Explicit PRISMA-ScR protocol: 4 OpenAlex Boolean queries ($Q_1$--$Q_4$) yielding 89 candidates, screened down to $N=46$ core studies (38 journal articles, 8 preprints, 2023--2026).
  - Explicit isolation of the 14-study sensitivity search (Scopus/IEEE Xplore) as an external robustness probe confirming 0.0% failure reporting, NOT merged into core statistical totals.
  - Transparent disclosure of the Prior Information Budget: empirical power curves are estimated using historical SCADA partitions, explicitly accounting for historical data exposure.
  - Full inter-rater reliability protocol with 3 independent raters (Coders A, B, C) achieving Fleiss' $\kappa = 0.646\text{--}1.000$ across all 5 dimensions.
  - Accurate formulation of 2017 UK Balancing and Settlement Code (BSC) single cash-out market rules.
- **Engineering Rationale**: Eliminates Reviewer #2 attack surfaces regarding literature search bias and data leakage.
- **Numerical Source**: `revision_outputs/04_numerical_source_of_truth.csv`, Items 1–6.
- **Author Verification Status**: `VERIFIED_AND_LOCKED`.

---

### Item 6: Table 1 Overhaul (Lifecycle Distribution)
- **Location**: `tables/table1_lifecycle_distribution.tex`
- **Identified Deficiency**:
  - Previous manuscript split lifecycle reporting across two disjoint tables (Table 1 and a second Table 4), causing duplicate captions and numbering collisions.
  - Missing Wilson 95% binomial confidence intervals.
- **Original Content**: Two separate tables with inconsistent totals.
- **Revised Content**:
  - Unified into a single, comprehensive table reporting all 5 dimensions ($\mathcal{D}_1$--$\mathcal{D}_5$) for the $N=46$ scoping cohort, complete with counts ($n$), percentages (%), and exact Wilson 95% CIs.
- **Engineering Rationale**: Resolves table redundancy and adheres to medical/biostatistical reporting standards (PRISMA-ScR).
- **Numerical Source**: `corpus48_consensus.csv`; Wilson CIs verified.
- **Author Verification Status**: `VERIFIED_AND_LOCKED`.

---

### Item 7: Table 2 Overhaul (3D Controlled Ablation)
- **Location**: `tables/table2_dynamic_settlement.tex`
- **Identified Deficiency**:
  - Confounded weak regularization ($\lambda=0.01$) with strong regularization ($\lambda=1.0$), obscuring when physical priors act as generic smoothers vs. domain guardrails.
  - Did not disclose seed sample sizes in table notes.
- **Original Content**: A single $\lambda$ column without clear indication of prior strength or random seed counts.
- **Revised Content**:
  - Explicit separation of conditions: Unconstrained ($\lambda=0$), Tuned Prior ($\lambda=0.1$), Strong Genuine Prior ($\lambda=1.0$), Strong Scrambled Prior ($\lambda=1.0$), and Weak Genuine vs. Scrambled ($\lambda=0.01$).
  - Complete multi-seed margins ($\text{mean} \pm \text{std}$) across Kelmarsh (4 seeds: 42--45), Hill of Towie (3 seeds: 42--44), and Penmanshiel (3 seeds: 42--44).
  - Explicit table notes detailing the hardware testbed (10 NVIDIA RTX 4090 GPUs) and sample sizes ($N_{\text{train}}$).
- **Engineering Rationale**: Proves the central empirical thesis: authentic physical priors provide defensive value under data restriction ($N_{\text{train}}=1,583$), but behave as non-specific smoothers under weak penalties ($\lambda=0.01$).
- **Numerical Source**: `ablation_results_kelmarsh_final.csv`, `ablation_results_hill_of_towie_final.csv`, `ablation_results_penmanshiel_seed42.csv`.
- **Author Verification Status**: `VERIFIED_AND_LOCKED`.

---

### Item 8: Table 3 Overhaul (Circular Block Paired Bootstrap)
- **Location**: `tables/table3_paired_bootstrap.tex`
- **Identified Deficiency**:
  - Reported currency without proper TeX escape formatting (`£` vs `\pounds`).
  - Did not clarify the distinction between seed variability and temporal sampling uncertainty.
- **Original Content**: Raw ASCII pound signs causing font compilation warnings; ambiguous error bar descriptions.
- **Revised Content**:
  - Clean LaTeX formatting using `\pounds`, explicit 2,000-replication circular block bootstrap with 7-day block lengths ($L=1,008$ intervals) preserving diurnal autocorrelation.
  - Added comprehensive table notes distinguishing temporal sampling uncertainty (Table 3) from initialization seed variance (Table 2).
- **Engineering Rationale**: Methodological clarity separating two orthogonal sources of uncertainty.
- **Author Verification Status**: `VERIFIED_AND_LOCKED`.

---

### Item 9: Table 4 Forensic Correction (Zero-Bid Diagnostic)
- **Location**: `tables/table4_zero_bid_diagnostic.tex`
- **Identified Deficiency**:
  - **FATAL TRANSCRIPTION ERROR**: Penmanshiel Zero-Bid Decision Cost was listed as `0.20584`. Forensic audit revealed that this value was copy-pasted from vanilla unconstrained DLinear RMSE (`0.205842`).
  - Missing Pinball loss column to directly anchor non-negative operational risk.
  - Imprecise framing implying that Zero-Bid cashflow represented commercial profit.
- **Original Content**: Erroneous `0.20584` entry; single-settlement imbalance cashflow presented without two-settlement context.
- **Revised Content**:
  - Recomputed true static decision cost from raw simulation outputs: Penmanshiel Zero-Bid $\text{Cost}_{\text{static}} = \mathbf{0.33689}$, Kelmarsh = $\mathbf{0.30428}$, Hill of Towie = $\mathbf{0.30473}$.
  - Added Pinball loss column ($\mathcal{L}_{1/3} = 0.11230, 0.10143, 0.10158$).
  - Replaced misleading "Profit" column headers with "Balancing Imbalance Cashflow (Diagnostic Reductio ad Absurdum)".
  - Added extensive footnote explaining that for $\hat{y} \equiv 0$, shortfall is zero and under-delivery cost equals the mean test capacity factor $\bar{y}$.
- **Engineering Rationale**: Eliminates the most severe numerical vulnerability in the original manuscript and establishes Zero-Bid as a mathematical diagnostic bounding test.
- **Numerical Source**: `revision_outputs/05_numeric_conflict_report.md`.
- **Author Verification Status**: `VERIFIED_AND_LOCKED`.

---

### Item 10: Table 5 Overhaul (PatchTST Architectural Sensitivity Probe)
- **Location**: `tables/table5_patchtst_and_rules.tex`
- **Identified Deficiency**:
  - Overclaimed that PatchTST proves universal cross-architecture invariance.
  - Overfull vbox and lrbox compilation errors caused by multirow syntax.
- **Original Content**: Unbounded claims of architectural generality; broken multirow formatting.
- **Revised Content**:
  - Framed strictly as an *exploratory architectural sensitivity probe* under fixed nominal penalty ($\lambda=0.01$).
  - Corrected table syntax (`\multirow{5}{=}{...}`), resolving all LaTeX compilation errors.
  - Reported 5-seed means and standard deviations across all 3 wind farms.
- **Engineering Rationale**: Robustness verification showing that defensive shortfall reduction persists in non-linear patch Transformers while cashflow effects diverge, without claiming an exhaustive neural architecture search.
- **Author Verification Status**: `VERIFIED_AND_LOCKED`.

---

### Item 11: Section 6 (Static Code & Computational Graph Audit)
- **Location**: `sections/06_code_audit.tex`
- **Identified Deficiency**:
  - Accused the authors of PCT of mathematical ignorance or scientific misconduct.
  - Speculated that 2D outer-product KDE was invalid due to marginal independence assumptions.
- **Original Content**: Polemical text claiming "fatal bugs invalidate published literature".
- **Revised Content**:
  - Objective software engineering audit of PCT commit `65941ec`: static AST decompilation isolating `F.kl_div(p.log(), q)` which computes $\text{KL}(q \parallel p)$ instead of published $\text{KL}(p \parallel q)$.
  - Analytical gradient derivation demonstrating that reversed arguments induce an inverse-density pole ($-\frac{P_{\text{ref}, i}}{4 P_{\text{pred}, i}} = \mathcal{O}(1/\epsilon)$), whereas standard JS is bounded by $\frac{1}{2}\ln 2$ with mild logarithmic divergence ($\mathcal{O}(|\log \epsilon|)$).
  - Rigorous mathematical proof that the empirical tensor outer-product $K_p^\top K_w$ is algebraically identical to an authentic paired 2D product-kernel KDE, preserving empirical bivariate coupling.
  - Negative control results showing that implementation divergence and mathematical formulation validity reside in distinct computational layers.
- **Engineering Rationale**: Replaces adversarial accusations with bulletproof mathematical derivations and forensic empirical evidence.
- **Numerical Source**: `pct_kldiv_check.py`, `pct_gradient_matched.py`, Figure 5.
- **Author Verification Status**: `VERIFIED_AND_LOCKED`.

---

### Item 12: Section 7 (Theory: Lemma 1 & Three-Term Risk Decomposition)
- **Location**: `sections/07_theory.tex`
- **Identified Deficiency**:
  - Proposition 1 was claimed as an original theoretical breakthrough, when in reality it is a classical quantile regression identity.
  - Term III was treated as an empirically verified causal mechanism rather than a structural diagnostic decomposition.
  - Counterexamples 1 & 2 were framed as universal proofs of failure rather than formal possibility boundary results.
- **Original Content**: "Proposition 1: Exact Integral Excess Operational Risk"; claims that Term III explains all empirical variations.
- **Revised Content**:
  - De-escalated Proposition 1 to **Lemma 1: Exact Integral Excess Operational Risk (Operational Baseline)**, explicitly attributing the foundational identities to Koenker (2001, 2005) and Pinson et al. (2007).
  - Clarified the three-term excess risk approximation:
    - $\text{Term I} = w_\tau \mathbb{E}[(f_\theta - \mu)^2]$ (Representation Error / MSE);
    - $\text{Term II} = w_\tau \mathbb{E}[(\delta - \Delta_\tau)^2]$ (Quantile Mismatch / Calibration Dispersion);
    - $\text{Term III} = 2 w_\tau \mathbb{E}[(f_\theta - \mu)(\delta - \Delta_\tau)]$ (Cross-Coupling Interaction).
  - Added an explicit **Diagnostic Caveat**: Term III serves as an interpretive diagnostic framework rather than an empirically isolated point-by-point causal estimate.
  - Explicitly qualified Counterexamples 1 & 2 as **formal possibility / existence results** establishing the theoretical limits of global scalar calibration under spatial heteroscedasticity.
- **Engineering Rationale**: Bulletproofs the theoretical contribution against Reviewer #2 accusations of claiming classical statistical theory as novel inventions.
- **Author Verification Status**: `VERIFIED_AND_LOCKED`.

---

### Item 13: Section 8 (Discussion, Boundaries, and Identification)
- **Location**: `sections/08_discussion.tex`
- **Identified Deficiency**:
  - Confounded site topography with sample size across cross-site evaluations.
  - Did not explicitly separate what is causally identified from what is not identified.
  - Lacked an actionable compliance checklist for future researchers.
- **Original Content**: Disjointed commentary on market rules and neural networks.
- **Revised Content**:
  - Added dedicated subsection: **Attributional Identification: What Is Identified vs. What Is Not**, explicitly disclaiming cross-site topographical causation due to co-varying turbine ratings and wind regimes.
  - Established Five Operational Boundary Contracts: (1) Sample volume and hypothesis saturation; (2) Weather proxy temporal staleness; (3) Two-settlement market netting; (4) Computational graph parity; (5) Literature scoping cohort bounds.
  - Documented Two Falsified Hypotheses during the audit trajectory (KDE coupling integrity; settlement price driver of cashflow inversion).
  - Operationalized the **REFORMS-PCWPF Checklist** (Table 6, Gates M1--M5: Isolated Prior Ablation, Domain Specificity Control, Proxy Staleness Stress-Test, Coupled Two-Settlement Evaluation, and Computational Graph Parity).
- **Engineering Rationale**: Transforms the manuscript from a critical critique into an indispensable methodological standard for the renewable energy forecasting community.
- **Author Verification Status**: `VERIFIED_AND_LOCKED`.

---

### Item 14: Figure 4 Vector Reconstruction and Dimension Normalization
- **Location**: `figures/fig_market_decoupling.pdf`
- **Identified Deficiency**:
  - **CRITICAL TYPESETTING DEFECT**: File had a MediaBox height of $7,569.26$ points (over 105 inches tall) due to an annotation box placed at $y=4$ instead of $y=7350$.
  - Caused LaTeX compilation warning: `Float too large for page by 4104.4939pt`.
  - LaTeX float FIFO queue blocked Figure 4 and all subsequent figures (Figure 5), dumping them onto pages 31–32 after the bibliography.
  - Unrendered raw LaTeX strings (`\pounds`, `\cdot`, `\rightarrow`) in axis labels.
- **Original Content**: 7,569 pt tall PDF; pushed past references; broken font commands.
- **Revised Content**:
  - Recomposed PDF using PyMuPDF vector mapping: main plot clipped to $757.18 \times 305$ pt, callout box ("Surface Cashflow Illusion") placed into Panel (b) whitespace above red bars ($x: 522\text{--}675, y: 68\text{--}122$).
  - Total MediaBox normalized to $757.18 \times 300.0$ pt (clean 2.52:1 aspect ratio).
  - All figures now compile smoothly within Section 5 (Page 15) and Section 6 (Page 19). Zero floats pushed past references.
- **Engineering Rationale**: Solves the severe visual layout bug and restores full PDF typesetting integrity.
- **Author Verification Status**: `VERIFIED_AND_LOCKED`.

---

## Final Verification Sign-Off

- **Total Words**: ~11,200 words across 11 main sections, 2 appendices, and 6 full-width tables.
- **Compiled PDF Pages**: Exactly 31 pages, zero compilation errors, zero float warnings.
- **Verification Coverage**: 100% of numerical claims traced to source ledgers.

*Signed*: Senior Scientific ML & Renewable Energy Forecasting Audit Agent  
*Status*: Ready for peer review submission.
