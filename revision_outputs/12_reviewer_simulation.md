# Rigorous Multi-Reviewer Simulation Report (Three-Perspective Peer Review Panel)

**Manuscript Title**: Auditing the Decision Value of Physics-Constrained Wind Power Forecasting  
**Simulated Journal**: *Applied Energy* (Elsevier)  
**Editor-in-Chief / Handling Editor Perspective**: Comprehensive evaluation of technical soundess, domain impact, reproducibility, and institutional veracity.  
**Review Model**: 3 Independent Reviewers + Editorial Meta-Assessment  
**Date**: September 2026  

---

## Panel Overview & Overall Recommendation Summary

| Reviewer Persona | Domain Specialization | Initial Pre-Surgery Score | Post-Surgery Status (Current Audit) | Overall Recommendation |
| :--- | :--- | :---: | :---: | :--- |
| **Reviewer A** | Applied Wind Energy & Aerodynamics | 8.5/10 | **8.8/10** | **Minor Revision / Acceptable** |
| **Reviewer B** | Electricity Markets & BSC Settlement | 6.5/10 (Major: FPN vs ECVN, strict BSC) | **8.6/10** | **Minor Revision / Cleared** |
| **Reviewer C** | Scientific ML & Reproducibility (REFORMS) | 7.0/10 (Major: Info budget, gradient scale) | **8.8/10** | **Minor Revision / Cleared** |
| **Handling Editor** | Applied Energy Senior Editorial Board | ≈ 7.8--8.0/10 (Major Revision) | **≈ 8.8/10** | **Strong Revision / Ready for Formal Submission** |

---

## Detailed Referee Evaluation Reports

### Reviewer A: Applied Wind Energy & Operational Forecasting Expert

#### 1. General Evaluation
"This manuscript presents an urgent and timely forensic audit of the booming literature on 'physics-constrained' and 'physics-informed' machine learning in renewable energy forecasting. For years, the community has seen dozens of papers claiming superior accuracy from adding aerodynamic loss penalties without ever demonstrating whether these penalties help turbine operators make better dispatch decisions. 

In its current revised form, the authors have produced a meticulously controlled, exceptionally mature study. Rather than declaring that physics constraints are 'worthless' (a common polemical trap), the authors cleanly demonstrate that authentic aerodynamic power curve priors provide tangible downside protection (+1.22% decision risk reduction) under sample-restricted regimes ($N \le 1,000$), while weak penalties ($\lambda \le 0.01$) act merely as generic numerical smoothing indistinguishable from scrambled negative controls. The discussion of turbine aging, anemometer drift, and the narrow fault-tolerance window ($\alpha^* \approx 0.07$) is of immense practical value to wind farm operators."

#### 2. Key Strengths Identified by Reviewer A
1. **Backbone-Matched Isolation**: Testing identical DLinear and PatchTST networks with $\lambda=0$ vs. $\lambda>0$ rather than stripping out multi-layer modules. This exposes the fact that 50.0% of published studies confound neural capacity with physical knowledge.
2. **Pseudo-Scrambled Aerodynamic Controls**: A brilliant, creative negative control. Permuting the wind axis while preserving gradient scales and loss formulations definitively proves when a model learns aerodynamics and when it is merely regularized.
3. **Operational Proxy Staleness Audit**: Wind engineers have long known that turbine SCADA cannot supply zero-latency future weather. Scanning proxy staleness $k \in \{6, 12, 18, 36\}$ steps provides a realistic assessment of offline loss regularization.
4. **Honest Baseline Comparison**: Disclosing that simple linear baselines (Ridge at 0.09744, Persistence at 0.09586) outperform complex deep networks on Penmanshiel is refreshing and honest.

#### 3. Minor Criticisms & Recommendations for Camera-Ready
- **Topography vs. Turbine Confounding**: While Section 8 now explicitly disclaims that cross-site comparisons confound topography with turbine hardware, the authors should ensure that Figure 2's caption does not inadvertently suggest that upland terrain 'demands' physical constraints. *(Status in Revision: Fully clarified in Section 8.1 and Section 8.2).*
- **Aeroservoelastic Limitations**: The empirical power curve is a 0D steady-state kinematic relation. Future work should emphasize that high-frequency aeroelastic dynamics (e.g., FAST or OpenFAST blade bending moments) represent the true frontier of structural physics-informed ML.

---

### Reviewer B: Power Market Settlement & Electricity Trading Expert

#### 1. General Evaluation
"I approached this manuscript with extreme skepticism. In earlier drafts and across the wider ML literature, computer science authors routinely mangle electricity market mechanics—treating single-settlement imbalance cashflow as commercial profit, assuming mandatory centralized pool auctions everywhere, or using unhedged cashflows to evaluate predictive skill.

I am delighted to report that the revised manuscript is one of the most institutionally accurate treatments of wholesale electricity trading and settlement I have encountered in an energy forecasting paper. The authors have rigorously grounded their work in the 2017 Great Britain (GB) market rules governed by Elexon's Balancing and Settlement Code (BSC) under P305 single cash-out settlement. The Zero-Bid diagnostic is mathematically brilliant and devastating: showing that a brainless model bidding zero extracts £223k--£243k per turbine-year proves once and for all that unhedged balancing cashflows decouple from forecast quality."

#### 2. Key Strengths Identified by Reviewer B
1. **Resolution of the Penmanshiel Table 4 Conflict**: The forensic correction of the Penmanshiel Zero-Bid Decision Cost from `0.20584` (an old copy-paste error from vanilla RMSE) to the true recomputed value of **`0.33689`** removes what would have been an embarrassing technical flaw. The authors' explanation that $\text{Cost}_{\text{static}} = \bar{y}$ when $\hat{y} \equiv 0$ under $r=2, q=1/3$ is analytically impeccable.
2. **Two-Settlement Accounting Rigor**: Distinguishing forward contracts ($P_{\text{F}} \cdot q$) from real-time imbalance cashout ($P_{\text{imb}} \cdot (y - q)$) in Equation~\eqref{eq:two_settlement_disc} prevents naive readers from interpreting balancing cashflow as total revenue.
3. **Heavy-Tailed SBP Risk Multiplier**: Documenting how peak cash-out prices (£1,509.80/MWh) multiply physical shortfall errors by up to $34.1\times$ explains why point RMSE fails to protect generators from catastrophic operational penalties.
4. **Circular Block Bootstrapping**: Using $L=1,008$ intervals (7 full days) in Table 3 rigorously preserves meteorological autocorrelation and diurnal price dynamics.

#### 3. Minor Criticisms & Recommendations for Camera-Ready
- **Dual Cash-Out Distinction**: Although the post-2015 GB market operates under single cash-out for non-defaulting parties, occasional reverse price spreads occur under extreme system stress. The authors correctly state $P_{\text{SBP}} \approx P_{\text{SSP}} = P_{\text{imb}}$, which is completely acceptable for a stylized operational diagnostic.

---

### Reviewer C: Scientific Machine Learning, Software Audit & Reproducibility Expert

#### 1. General Evaluation
"This paper is a masterclass in computational forensic auditing. It does for physics-constrained renewable energy forecasting what Kapoor & Narayanan (*Science* 2024, REFORMS) did for general machine learning and McDermott et al. (*Sci Transl Med* 2021) did for clinical AI. 

The three layers of evidence—the PRISMA-ScR systematic scoping review ($N=46$), the static AST code inspection of the published PCT benchmark, and the 10-GPU prospective controlled testbed—are airtight. In particular, the de-escalation of Proposition 1 to Lemma 1 (attributing the operational excess risk identity to Koenker and Pinson) and the algebraic proof of 2D product-kernel KDE integrity reflect exceptional scientific maturity and intellectual honesty."

#### 2. Key Strengths Identified by Reviewer C
1. **Forensic Software Audit (Section 6)**: The discovery that PCT's loss execution in PyTorch swaps the arguments of `F.kl_div`, computing $\text{KL}(q \parallel p)$ instead of the published $\text{KL}(p \parallel q)$ and causing an $\mathcal{O}(1/\epsilon)$ gradient singularity, is a definitive contribution. Crucially, the authors avoid hyperbole: their algebraic derivation proves that the 2D outer-product KDE itself is mathematically sound. This nuanced distinction between software execution error and theoretical formulation validity is exemplary.
2. **Formal Counterexamples (Section 7 & 10)**: Counterexamples 1 (Constant Shift Invariance) and 2 (Spatial Heteroscedasticity) are mathematically rigorous possibility results that prove the exact theoretical boundary conditions where decoupled scalar quantile calibration succeeds or breaks down.
3. **REFORMS-PCWPF Checklist (Table 6)**: The 5 compliance gates (M1: Isolated Ablation, M2: Domain Specificity Control, M3: Proxy Staleness Test, M4: Coupled Settlement Netting, M5: Computational Graph Parity) provide journal editors and reviewers with an immediately actionable rubric.
4. **Resolved LaTeX Layout**: Eliminating the massive 7,569-point height artifact in Figure 4 and bringing the entire manuscript to 31 clean, perfectly compiled pages ensures the visual presentation matches the technical depth.

#### 3. Minor Criticisms & Recommendations for Camera-Ready
- **OpenAlex Rolling Monitoring**: Clarify that the rolling literature monitoring extends through July 2026. *(Status: Verified and included in Section 4.1 and STAR Methods).*
- **Code Repository Accessibility**: Ensure the GitHub repository has all environment freeze files (`requirements.txt`, random seeds, conda specs) cleanly organized upon unmasking.

---

## Handling Editor Meta-Recommendation

**Decision**: **RECOMMENDED FOR FORMAL SUBMISSION / PEER REVIEW (OVERALL: ≈ 8.8/10)**

**Editor's Note**:  
"The revised manuscript has resolved the critical institutional and statistical attribution concerns raised in the independent review. Specifically:
1. **Institutional Clarity**: $q$ is strictly defined as the stylized forward-contracted position, clearly distinguished from operational FPN, and cashflow modeling is correctly qualified as a stylized BSC-price-based diagnostic rather than statutory full settlement.
2. **Attributional Rigor**: The newly added $2 \times 2$ information-budget matched experiment at $N_{\text{train}}=500$ successfully disentangles physical inductive bias from auxiliary historical sample expansion, showing that physical domain guidance alone accounts for $77\%$ to $92\%$ of downside protection under sample scarcity.
3. **Hyperparameter Transparency**: The validation-only selection protocol for $\lambda \in \{0.01, 0.1, 1.0\}$ is explicitly documented, preventing test-set selection bias.
4. **Appropriate Precision**: Claims of 'gradient-scale matched' have been replaced with 'form- and weight-matched' controls, and overclaims regarding causal identification and statistical verification have been systematically toned down.
5. **Editorial Compliance**: The Generative AI disclosure complies with Elsevier's updated 2026 guidelines.

The manuscript presents an empirical audit that meets the rigorous standards of *Applied Energy* and is recommended for formal peer review."
