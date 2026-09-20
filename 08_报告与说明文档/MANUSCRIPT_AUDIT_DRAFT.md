# From Specification to Deployment: A Systematic Audit of Constraint Provenance, Implementation Fidelity, and Harm in Physics-Constrained Wind Power Forecasting

**Authors**: Junyu Li$^{1,*}$, Juntao Du$^{2}$, et al.  
*Affiliations*: $^{1}$School of Electrical and Information Engineering; $^{2}$Independent Verification Specialist  
*Correspondence*: ljylikezmn999@gmail.com  

---

## Abstract

Physics-constrained deep learning has emerged as a prominent paradigm in wind power forecasting, operating under the pervasive default premise that embedding physical domain knowledge (such as turbine power curves, aerodynamic Betz limits, and wake deficit models) inherently improves generalization, physical plausibility, and operational reliability. In this paper, we conduct a systematic empirical audit of this premise across the entire lifecycle of constraint definition, implementation, and deployment, without relying on any single trained neural architecture. 

First, we introduce the **Constraint Lifecycle Taxonomy (Schema v4)**, a five-dimensional framework that formally decouples parameter provenance (D1), application stage (D2), test-time inference requirements (D3), implementation fidelity (D4), and harm reporting (D5). 

Second, we execute an empirical full-text audit across 46 recent physics-constrained wind forecasting studies (2023–2026) meeting rigorous scoping criteria. The audit uncovers three systemic patterns:
1. **Unilateral Benefit Reporting**: Zero out of 46 studies (0.0%) report any negative performance trade-offs, degradation, or operational penalties incurred by their physical constraints under evaluated protocols.
2. **Reproducibility Gap**: Under documented public repository search protocols, 44/46 studies provided no public code (95.7% not-found). Among the two publicly available codebases identified, one exhibited a directional mathematical discrepancy with its published specification (1/46 code-mismatch, PCT), while one faithfully implemented its specified algebraic constraints (1/46 code-aligned, PINN-UQ).
3. **Causal Confounding in Ablation Studies**: While 100% of analyzed papers claim ablation validation in their narratives, strict isolated ablation of the physical constraint (comparing $\lambda=0$ vs. $\lambda>0$) occurs in only 52.2% (24/46); 47.8% (22/46) merely ablate surrounding neural network modules (e.g., GNN layers or attention streams), conflating architectural capacity gains with physical inductive biases.

Third, we present a concrete static software audit of the sole publicly released codebase, the Physics-Constrained Transformer (PCT, *Scientific Reports* 2025). We prove that its loss function executes PyTorch `F.kl_div(p.log(), q)` = $\text{KL}(q \parallel p)$, implementing an asymmetric reverse-JS-type divergence rather than the paper's specified symmetric Jensen–Shannon divergence. Simultaneously, formal mathematical derivation demonstrates that its paired kernel product is equivalent to a standard 2D product-kernel density estimator, correcting earlier misconceptions regarding marginal independence assumptions. 

Fourth, we theoretically formalize the **Prediction-Decision Decoupling Framework** and derive the exact integral representation of excess operational dispatch risk, establishing three open mechanistic hypotheses and formal boundary counterexamples (constant shift invariance and spatial heteroscedasticity) that delineate when decoupled calibration dominates. We demonstrate that asymmetric operational dispatch risk decomposes into prediction representation MSE (Term I), decision quantile mismatch (Term II), and residual distortion (Term III). In uncalibrated regimes, apparent benefits of physical penalties often stem from accidental shrinkage masking large quantile mismatches. Decoupled quantile calibration eliminates Term II, exposing the underlying representation distortion under high signal-to-noise ratios. Under real-world UK Elexon BSC 30-minute physical net settlement, empirical audits across operational wind farms reveal the **double-edged nature of physical constraints**: at the flat-terrain Kelmarsh site, physical priors incur an annual shortfall surge of $+£932.56/\text{(turb}\cdot\text{yr)}$ and balancing net cashflow loss of $-£3,264.86/\text{(turb}\cdot\text{yr)}$, whereas in complex upland terrain (Hill of Towie and Penmanshiel), physical regularization acts as a protective inductive guard, with Penmanshiel power curve ($0.09753$) and distribution ($0.09804$) constraints outperforming unconstrained DLinear ($0.09864$) while trailing simple linear baselines (Ridge $0.09744$, Persistence $0.09586$), and Hill of Towie generating $+£1,481.27/\text{(turb}\cdot\text{yr)}$ in balancing mechanism net cashflow component. These terrain-dependent findings are framed as open mechanistic hypotheses rather than universal causal laws. We conclude with an actionable audit protocol and governance checklist to guide rigorous renewable forecasting.

**Keywords**: Physics-Constrained Deep Learning; Wind Power Forecasting; Prediction-Decision Decoupling; Double-Edged Constraints; Systematic Audit; Implementation Fidelity; Reproducibility.

---

## 1. Introduction

Accurate wind power forecasting is vital for modern power system operation, grid dispatch scheduling, unit commitment, and renewable energy market trading [1-3]. Over the past half-decade, data-driven approaches have evolved rapidly from classical autoregressive models and gradient boosted decision trees [4] to recurrent neural networks, temporal convolutional networks, and deep Transformers [5-7]. 

Concurrently, the energy engineering community has embraced **physics-informed** and **physics-constrained machine learning** [8,9]. By encoding fundamental physical equations—such as turbine aerodynamic power curves, Betz aerodynamic limits, Jensen wake deficit formulations, and generator electromechanical relationships—researchers seek to constrain deep models to physically admissible operating regions, improve data efficiency, and eliminate non-physical artifacts.

A fundamental premise underpins this extensive body of work: *incorporating physical constraints is inherently and universally beneficial*. Throughout the published literature, physical priors are routinely introduced as inductive biases that guarantee improved out-of-sample accuracy, prevent physical violations, and enhance operational safety. 

However, in industrial deployments and real-world energy operations, this assumption faces severe engineering realities:
- **Parameter Misspecification**: Physical models require specific parameters (e.g., blade roughness, air density, turbine rotor radius, empirical wake decay coefficients) that drift over time or vary across sites.
- **Test-Time Dependency**: Certain constraints assume contemporaneous or future meteorological forecasts (e.g., numerical weather predictions) that may be absent, delayed, or corrupt during real-time inference.
- **Specification–Implementation Mismatches**: Subtle software translation errors between mathematical equations and machine learning codebases can silently alter optimization dynamics.
- **Confounded Evaluation**: Empirical improvements attributed to "physics" frequently stem from the added capacity of auxiliary neural sub-networks rather than the physical regularizer itself.
- **Coupling Trap Between Representation and Decision**: Forcing models to optimize coupled physical penalties distorts the conditional mean, while uncalibrated evaluations mistake accidental prediction shrinkage for operational efficacy.

Despite their critical practical implications, these factors remain systematically unexamined. To bridge this gap, this paper does not present yet another specialized neural architecture claiming marginal percentage gains over standard benchmarks. Instead, we present a comprehensive, evidence-grounded **scoping empirical audit** and **theoretical formalization** of physics constraints across the wind power forecasting lifecycle.

### 1.1 Core Contributions

This work delivers four primary contributions grounded in verifiable evidence and formal theory:

1. **The Constraint Lifecycle Taxonomy (Schema v4)**: We formalize a five-dimensional taxonomic framework that decouples constraint parameter provenance (D1), application stage (D2), test-time inference requirement (D3), software implementation fidelity (D4), and operational harm reporting (D5). In particular, it introduces an explicit decoupling rule distinguishing training-time empirical priors from deployment-time computational dependencies.
2. **Systematic Field Audit of 46 Domain Studies**: We execute a full-text scoping audit across 46 peer-reviewed studies published between 2023 and 2026. Supported by a pilot-stage multi-rater reliability record on 20 representative studies prior to schema cleansing (Fleiss' $\kappa = 0.646\text{--}1.000$ across D1--D3 and D5 ablation on included studies: D1: 1.000, D2: 0.938, D3: 0.646, D5: 0.818; with D4 implementation fidelity at $\kappa = 0.335$ on included studies and $\kappa = 0.379$ overall, reflecting open-source retrieval hurdles; and 18 included studies exhibiting a universal 100% literature reporting gap on harm with zero marginal variance, reported as `NaN (Po = 100%)`), we uncover that:
   - 0/46 studies report any negative performance trade-offs or harm metrics ($0.0\%$);
   - 44/46 studies provide no public code ($95.7\%$), with one codebase verified as `code-aligned` after retraction of an erroneous ODE mismatch charge (S10, $2.2\%$) and one exhibiting a directional loss mismatch (S02, $2.2\%$);
3. **Static Software Audit & Implementation Parity**: We perform an in-depth code audit of the Physics-Constrained Transformer (PCT) repository, identifying a directional discrepancy in its Jensen–Shannon loss implementation (`F.kl_div(p.log(), q)` = $\text{KL}(q \parallel p)$) while proving the mathematical validity of its joint product-kernel density estimator.
4. **Prediction-Decision Decoupling Theory & The Double-Edged Nature of Constraints**: We establish an exact integral excess risk formulation and three testable open mechanistic hypotheses, rigorously bounded by two formal counterexamples (constant shift invariance and spatial heteroscedasticity). We show why uncalibrated 'advantages' in literature are frequently artifacts of accidental negative bias masking quantile mismatch. Under standard decoupled quantile calibration, we show that physical regularization is double-edged: in high-SNR flat terrain (Kelmarsh), it degrades conditional mean representations and inflicts a $+£932.56/\text{(turb}\cdot\text{yr)}$ shortfall penalty surge; in complex-terrain/low-SNR environments (Hill of Towie and Penmanshiel), physical constraints serve as defensive regularizers that prevent over-bidding, delivering $+£1,481.27/\text{(turb}\cdot\text{yr)}$ in two-sided balancing mechanism net cashflow component on Hill of Towie and outperforming unconstrained DLinear on Penmanshiel ($0.09753$ vs. $0.09864$, though trailing simple Ridge $0.09744$ and Persistence $0.09586$). Appendix A provides the formal mathematical analysis of these boundary counterexamples.

---

## 2. Related Work & Empirical Motivation

### 2.1 Physics-Informed Machine Learning in Energy Systems
Physics-informed machine learning encompasses a wide spectrum of integration techniques [8,10]. In fluid dynamics and heat transfer, Physics-Informed Neural Networks (PINNs) [9] embed partial differential equations directly into the training loss via automatic differentiation. In renewable power forecasting, physical constraints generally fall into three functional groups:
1. **Aerodynamic Power Boundaries**: Upper and lower bounding curves dictated by cut-in, rated, and cut-out wind speeds, as well as theoretical aerodynamic limits ($P \le \frac{16}{27} \cdot \frac{1}{2} \rho A v^3$) [11,12].
2. **Wake Deficit and Spatial Interactions**: Analytical wake models (e.g., Jensen, Bastankhah–Porté-Agel) integrated into graph neural networks to model spatial dependencies across turbine arrays [13,14].
3. **Distributional and Statistical Regularization**: Empirical joint power-wind distributions matched via statistical divergence penalties (e.g., Wasserstein distance, Jensen–Shannon divergence) [15,16].

### 2.2 The Evaluation and Reproducibility Blind Spot
While hundreds of papers report substantial metric improvements from physics constraints, empirical reproducibility in machine learning has faced mounting scrutiny [17-19]. In time-series forecasting, several recent landmark studies demonstrated that sophisticated architectures often fail to outperform simple linear models when evaluated under identical, non-leaked protocols [20,21]. 

In physics-constrained forecasting, an analogous audit has been noticeably absent. Prior surveys primarily categorize mathematical equations or application domains [22,23] without auditing whether the published code matches the equations, whether constraints remain computable under communication delays, or whether physical misspecification incurs operational penalties.

---

## 3. Methodology: The Five-Dimensional Constraint Lifecycle

To establish an auditable foundation, we define the **Constraint Lifecycle Taxonomy (Schema v4)**. A physical constraint $C$ is modeled as a multi-stage engineering object passing through five orthogonal dimensions:

```mermaid
flowchart LR
    D1["D1: Parameter Provenance\n(Intrinsic vs. Empirical)"] --> D2["D2: Application Stage\n(Loss vs. Architecture)"]
    D2 --> D3["D3: Inference Requirement\n(Runtime Input Needs)"]
    D3 --> D4["D4: Software Fidelity\n(Specification vs. Code)"]
    D4 --> D5["D5: Harm & Ablation\n(Isolated vs. Confounded)"]
```

### 3.1 Dimension 1: Parameter Provenance (D1)
Dimension 1 asks strictly: *Where do the parameters of the physical constraint originate?*
- `intrinsic`: The constraint relies entirely on analytic equations and universal physical constants requiring no empirical estimation from local data (e.g., Betz limit $C_p \le 16/27$, nameplate rated capacity, Jensen analytical formulas).
- `historical-static`: Parameters are statistically estimated from historical operational data offline and frozen during training/inference (e.g., empirical power curves fitted via 2D kernel density estimation on historical SCADA).
- `learned-online`: Parameters are dynamically predicted or updated during runtime via an auxiliary sub-network (e.g., a learned dynamic $C_p(t)$ network).
- `unreported`: The study omits parameter calibration details.

### 3.2 Dimension 2: Application Stage (D2)
Dimension 2 asks strictly: *At what engineering phase is the physical constraint imposed?*
- `training-loss`: The constraint is formulated as a penalty term in the loss function $\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \mathcal{L}_{\text{phys}}$, acting via gradient backpropagation.
- `inference-architecture`: The constraint is hard-wired into the forward computational graph (e.g., projection layers, bounded activation functions, physical conversion streams), persisting through test-time inference.
- `data-generation`: The constraint guides data augmentation or generative models (e.g., physics-guided diffusion) to synthesize synthetic training samples.
- `post-hoc-correction`: The constraint is applied as an external filter or calibration step after the base model completes forward inference.

### 3.3 Dimension 3: Inference Requirement (D3)
Dimension 3 asks strictly: *What input variables are required to compute the constraint during test-time inference?*
- `none`: Inference requires only standard historical lookback sequences; no future information or external concurrent telemetry is needed.
- `forecast-input`: Evaluation requires concurrent future meteorological forecasts (e.g., future NWP wind speed $\hat{v}_{t+h}$) to compute the physical boundary.
- `contemporaneous-state`: Evaluation requires synchronized real-time telemetry from external turbines (e.g., upstream turbine yaw or inflow) at time $t$.
- `latent-state`: Requires unobservable internal states.

> **Orthogonality Rule (Schema v4)**: D1 and D3 are strictly orthogonal. For example, an analytical Jensen wake model is `intrinsic` in D1 (formula is derived from physics), but requires `forecast-input` or `contemporaneous-state` in D3 (evaluating the deficit requires inflow wind speed). Decoupling these dimensions eliminates historical ambiguities where models were mislabeled as "non-computable" simply because they used empirical parameters.

### 3.4 Dimension 4: Implementation Fidelity (D4)
Dimension 4 evaluates the verifiability and software alignment of released artifacts:
- `code-aligned`: Public code is available, executes successfully, and matches the published mathematical equations.
- `code-mismatch`: Public code is available but contains demonstrable mathematical, algorithmic, or structural divergences from the published text.
- `not-found`: No public software repository is linked, or the provided repository URL is dead/empty.
- `unverified`: Code exists but cannot be audited due to proprietary licensing or missing access rights.

### 3.5 Dimension 5: Harm Evaluation and Ablation Design (D5)
Dimension 5 examines whether the study rigorously evaluates the downside risk and causal attribution of the constraint:
- `negative_effect_reported`: Boolean (`yes`/`no`) recording whether the authors observe any regime where the physical constraint degrades forecasting accuracy or operational metrics.
- `ablation_type`:
  - `constraint-on-off`: The physical constraint is strictly toggled ($\lambda=0$ vs. $\lambda>0$) while keeping the model architecture identical.
  - `module-ablation`: The entire neural module (e.g., an auxiliary GNN branch or multi-modal encoder) is removed, confounding architectural capacity with physical knowledge.
  - `both`: Both isolated constraint toggling and module-level ablations are conducted.
- `prior_substitution_tested`: Evaluates whether the model is stress-tested under misspecified or corrupted physical parameters.

### 3.6 Theoretical Foundations: Prediction-Decision Decoupling, Exact Excess Risk Expansion, and Mechanistic Hypotheses

While empirical studies frequently report performance trade-offs when constraints are added, a foundational theoretical question remains: *Why do soft physical and distributional constraints frequently exhibit pseudo-gains under uncalibrated evaluation, yet decouple into divergent behaviors—from severe penalty surges to protective shelters—under standard operational decision calibration?* In this subsection, we formalize the **Prediction-Decision Decoupling Framework**. Rather than asserting ungrounded universal dominance theorems, we establish an exact integral excess operational risk representation, present two rigorous boundary counterexamples, and formalize three testable mechanistic hypotheses.

#### 3.6.1 Mathematical Formulation and Operational Setting
We consider a unified time-series dispatch formulation strictly parameterized to operational power systems:
- **Temporal Dimensions**: Historical lookback length $L = 36$ (corresponding to 6 hours at 10-minute SCADA resolution); multi-step forecast horizon $H = 36$ (forward 6 hours); operational dispatch settlement horizon $H_{disp} = \{1, 2, 3, 4, 5, 6\}$ (the next-hour delivery window comprising six 10-minute intervals); and origin stride $S = 6$ (hourly rolling forecast origination, guaranteeing non-overlapping settlement intervals without duplicate billing).
- **Variables**: Lookback feature sequence $X \in \mathcal{X} \subset \mathbb{R}^{L \times d_{in}}$ ($d_{in} = 2$, normalized historical power and wind speed); actual future power generation vector $Y = [Y_1, \dots, Y_H]^\top \in [0, 1]^H$, with per-step realization $Y_h \in [0, 1]$ for $h \in H_{disp}$; and point forecasting model $f_\theta: \mathcal{X} \to [0, 1]^H$ parameterized by $\theta \in \Theta$.
- **Operational Decision Rule**: In dispatch settlement, the commitment/bidding decision $p_h$ is generated via:
  $$p_h = \mathcal{P}_{[0,1]}\left( f_{\theta, h}(X) + \delta_h \right)$$
  where $\mathcal{P}_{[0,1]}(u) \triangleq \min(\max(u, 0), 1)$ denotes projection onto the physically admissible generation domain $[0, 1]$, and $\delta_h \in \mathbb{R}$ is the decision calibration offset.
- **Asymmetric Operational Balancing Loss**: In two-settlement balancing markets (e.g., standard Newsvendor dispatch), dispatch deviations incur asymmetric penalties:
  $$\ell_r(p_h, Y_h) = c_+ \max(p_h - Y_h, 0) + c_- \max(Y_h - p_h, 0)$$
  where $c_+$ is the unit shortfall penalty (under-delivery cashout buy cost at System Buy Price SBP when generation $Y_h < p_h$), $c_-$ is the unit surplus penalty (over-generation / curtailment spill cost when generation $Y_h \ge p_h$), and $r \triangleq c_+ / c_- \in (0, \infty)$ is the loss penalty ratio. Setting unit surplus penalty $c_- = 1$ and unit shortfall penalty $c_+ = r$, and defining the critical fractile $\tau \triangleq \frac{1}{r+1} \in (0, 1)$, this operational loss is algebraically equivalent to the scaled canonical pinball check loss with error argument $u \triangleq Y - p$:
  $$\ell_r(p, Y) = (r+1) \cdot \rho_\tau(Y - p), \quad \text{where } \rho_\tau(u) \triangleq u\left(\tau - \mathbb{I}(u < 0)\right)$$
  *(Verification: when $Y < p$ (shortfall), $u < 0$, $\rho_\tau(u) = (1 - \tau)(p - Y) = \frac{r}{r+1}(p - Y)$, so $(r+1)\rho_\tau(u) = r(p - Y) = c_+(p - Y)$; when $Y \ge p$ (surplus), $u \ge 0$, $\rho_\tau(u) = \tau(Y - p) = \frac{1}{r+1}(Y - p)$, so $(r+1)\rho_\tau(u) = 1(Y - p) = c_-(Y - p)$).*
- **Conditional Statistics**: Let $\mu(X) \triangleq \mathbb{E}[Y \mid X]$ denote the true conditional expectation; $f_{Y|X}(y)$ and $F_{Y|X}(y)$ denote the conditional PDF and CDF; $q_\tau(X) \triangleq F_{Y|X}^{-1}(\tau)$ denote the Bayes-optimal conditional quantile; $\varepsilon_\mu(X) \triangleq f_\theta(X) - \mu(X)$ denote the conditional point prediction error; $\Delta_\tau(X) \triangleq q_\tau(X) - \mu(X)$ denote the intrinsic quantile-mean structural bias; and $w_\tau(X) \triangleq \frac{r+1}{2} f_{Y|X}(q_\tau(X))$ denote the local curvature weight.

#### 3.6.2 Assumptions, Defenses, and Scope Boundaries
To ground the analysis mathematically while acknowledging real-world limits:

> **Assumption 1 (Local Convexity of Point Prediction Risk)**. The unconstrained point-prediction risk $\mathcal{R}_{pred}(\theta) \triangleq \mathbb{E}_{X,Y}[(Y - f_\theta(X))^2] = \text{MSE}(\theta)$ is locally convex around the empirical optimizer $\theta^*$. For linear temporal architectures such as DLinear ($f_\theta(X) = W X + b$), $\nabla_W^2 \mathcal{R}_{pred} = 2 \mathbb{E}[X X^\top] = 2 \Sigma_X$. While non-stationary atmospheric telemetry ensures $\Sigma_X \succ 0$ for feature weights, joint bias terms may introduce a low-dimensional null space (e.g., compensating seasonal and trend biases), which can be resolved via standard Ridge regularization.

> **Assumption 2 (Prior Incompatibility / Non-Vanishing Prior Gradient)**. Let $\mathcal{R}_{phys}(\theta)$ denote an auxiliary physical or distributional penalty term. In general operational telemetry, the unconstrained conditional mean optimizer $\theta^*$ differs from the empirical prior minimizer $\theta_{phys} \triangleq \arg\min \mathcal{R}_{phys}$, inducing a non-vanishing prior gradient $\mathbf{g}_{phys} \triangleq \left. \nabla_\theta \mathcal{R}_{phys}(\theta) \right|_{\theta = \theta^*} \ne \mathbf{0}$.

> **Assumption 3 (Quasi-Homoscedastic Approximation for Short Horizons)**. Over ultra-short dispatch delivery horizons ($h \in \{1..6\}$, 10–60 minutes), conditional variance exhibits localized stability $\sigma(X) \approx \bar{\sigma}$, allowing the structural bias $\Delta_\tau(X) \approx \bar{\sigma} z_\tau$ to be approximated by a scalar calibration offset $\delta^*$. However, when spatial heteroscedasticity is non-negligible, scalar calibration does not span the full conditional quantile surface, as formally proven in Counterexample 2.

#### 3.6.3 Proposition 1: Exact Integral Excess Operational Risk & Three-Term Decomposition

> **Proposition 1 (Exact Integral Excess Operational Risk)**. Let the true conditional distribution $F_{Y|X}$ be continuously differentiable with strictly positive conditional density $f_{Y|X} > 0$. For any unclipped internal decision rule $p(X) \in (0, 1)$, the expected operational dispatch risk $\mathcal{R}_{disp}(p) \triangleq \mathbb{E}_{X,Y}[\ell_r(p(X), Y)]$ relative to the Bayes-optimal quantile decision $q_\tau(X) = F_{Y|X}^{-1}(\tau)$ admits an exact, non-truncated integral representation:
> $$\mathcal{R}_{disp}(p) - \mathcal{R}_{disp}(q_\tau) = (r+1) \cdot \mathbb{E}_X \left[ \int_{q_\tau(X)}^{p(X)} \left[ F_{Y|X}(u) - \tau \right] du \right] \ge 0$$
> The excess operational risk is strictly positive for all $p(X) \ne q_\tau(X)$ and vanishes if and only if $p(X) = q_\tau(X)$ almost surely.

*Proof*. For any fixed commitment $p$, the conditional expected loss is:
$$\mathcal{L}(p; X) \triangleq \mathbb{E}_Y [\ell_r(p, Y) \mid X] = (r+1) \left[ (\tau - 1) \int_0^p (y - p) f_{Y|X}(y) dy + \tau \int_p^1 (y - p) f_{Y|X}(y) dy \right]$$
Differentiating with respect to $p$ via Leibniz's rule yields:
$$\frac{\partial \mathcal{L}(p; X)}{\partial p} = (r+1) \left[ (1 - \tau) \int_0^p f_{Y|X}(y) dy - \tau \int_p^1 f_{Y|X}(y) dy \right] = (r+1) \left[ F_{Y|X}(p) - \tau \right]$$
At the Bayes-optimal quantile $p = q_\tau(X)$, $F_{Y|X}(q_\tau(X)) = \tau$, so $\left. \frac{\partial \mathcal{L}}{\partial p} \right|_{q_\tau(X)} = 0$. By the Fundamental Theorem of Calculus:
$$\mathcal{L}(p; X) - \mathcal{L}(q_\tau(X); X) = \int_{q_\tau(X)}^p \frac{\partial \mathcal{L}(u; X)}{\partial u} du = (r+1) \int_{q_\tau(X)}^p [F_{Y|X}(u) - \tau] du$$
Because $F_{Y|X}(u)$ is strictly monotonic, $[F_{Y|X}(u) - \tau]$ has the identical sign as $[u - q_\tau(X)]$. Thus the integral is strictly non-negative, and taking the expectation over $X$ yields the exact result. $\blacksquare$

*Local Second-Order Taylor Expansion*: Linearizing $F_{Y|X}(u) - \tau \approx f_{Y|X}(q_\tau(X))(u - q_\tau(X))$ around $q_\tau(X)$ yields the familiar local quadratic decomposition:
$$\mathcal{R}_{disp}(p) = \mathcal{R}^*_{Bayes} + \underbrace{\mathbb{E}_X \left[ w_\tau(X) \cdot \varepsilon_\mu^2(X) \right]}_{\text{Term I: Prediction Representation Loss (MSE)}} + \underbrace{\mathbb{E}_X \left[ w_\tau(X) \cdot \left( \delta - \Delta_\tau(X) \right)^2 \right]}_{\text{Term II: Decision Quantile Mismatch (Bias Risk)}} + \underbrace{2 \mathbb{E}_X [w_\tau(X) \varepsilon_\mu(X) (\delta - \Delta_\tau(X))]}_{\text{Cross-Variation Term}} + \mathcal{O}\left( \mathbb{E}_X \left[ |p - q_\tau|^3 \right] \right)$$
where $w_\tau(X) \triangleq \frac{r+1}{2} f_{Y|X}(q_\tau(X)) > 0$. When scalar calibration $\delta^*$ aligns the global marginal quantile, Term II is largely suppressed, shifting the performance balance toward Term I.

#### 3.6.4 Theoretical Deconstruction of the Calibration Inversion Phenomenon
Proposition 1 explains why empirical studies frequently observe an inversion of physical constraint performance upon decision calibration:

```mermaid
flowchart LR
    subgraph UC ["Uncalibrated Regime (delta = 0)"]
        direction TB
        U1["Term II Dominates (delta = 0)<br>Structural Quantile Mismatch Delta_tau"] --> U2["Physical Penalty Gradient g_phys<br>Induces Downward Shrinkage"]
        U2 --> U3["Accidental Surplus Cost Drop (-1.75‰)<br>Conceals Shortfall Deterioration (+0.96‰)<br>Yields Apparent Pseudo-Gain (-1.50‰)"]
    end
    subgraph CC ["Quantile-Calibrated Regime (delta = delta*)"]
        direction TB
        C1["Term II Neutralized via Shift delta*<br>Shortfall Penalties Plummet by ~35%"] --> C2["Term I (MSE Representation) Exposed<br>Unconstrained MSE = 0.010713<br>Constrained MSE = 0.010817 (+0.000104)"]
        C2 --> C3["Site-Dependent Divergence Revealed:<br>High-SNR Flat Site (Kelmarsh): Inversion (+0.93‰)<br>Complex Upland Site (Penmanshiel): Protective Gain vs DLinear (-1.11‰)"]
    end
    UC ==>|Apply Standard Decoupled Quantile Calibration| CC
```

1. **The Uncalibrated Regime ($\delta = 0$)**: When uncalibrated point forecasts are fed directly into dispatch, Term II becomes $\mathbb{E}_X [w_\tau(X) \Delta_\tau(X)^2]$. For $r = 2$ ($\tau = 1/3$), the negative structural bias $\Delta_\tau(X) \approx -0.025$ causes uncalibrated point forecasts to systematically under-bid the required quantile, driving shortfall rates up to $56\%$. If a coupled physical penalty $\lambda \mathcal{R}_{phys}$ induces a downward shrinkage ($\varepsilon_\mu(X) < 0$), the net deviation $p - q_\tau = \varepsilon_\mu(X) - \Delta_\tau(X) \approx -|\varepsilon_\mu| - (-0.025) = 0.025 - |\varepsilon_\mu|$ is accidentally contracted. This contracts surplus penalties, manufacturing an illusion of operational benefit while actual shortfall accuracy often deteriorates.
2. **The Calibrated Regime ($\delta = \delta^* \approx -0.025$)**: Decoupled quantile calibration ($\delta^* = \arg\min_\delta \sum_i \ell_r(f(X_i) + \delta, Y_i)$) shifts the forecast, suppressing Term II and reducing shortfall penalties by $\sim 35\%$. This exposes Term I (conditional representation MSE). In nominal, high-SNR flat terrain (such as Kelmarsh), physical regularizers that distort conditional mean estimation are exposed, inverting the apparent pseudo-benefit into an excess cost. Conversely, in complex upland terrain with high turbulence (such as Hill of Towie and Penmanshiel), physical constraints act as protective inductive guards that prevent extreme over-bidding.

#### 3.6.5 Theory Boundaries: Counterexamples and Mechanistic Hypotheses

Crucially, **excess operational risk cannot be claimed as an inevitable universal theorem**. Mathematical rigor requires identifying boundary conditions where coupled constraints succeed or fail. We formalize two counterexamples that define the boundaries of decoupled calibration:

> **Counterexample 1 (Constant Shift Invariance: Non-Zero Gradient with Zero Excess Risk)**.  
> Let $Y \sim \text{Uniform}[0.25, 0.75]$ with unconstrained mean $\theta^* = 0.5$. Suppose an auxiliary prior penalty $\mathcal{R}_{phys}(\theta) = (\theta - 0.6)^2$ pulls the regularized predictor to $\theta_\lambda = 0.55$, strictly degrading MSE by $\Delta \text{MSE} = (0.55 - 0.5)^2 = 0.0025 > 0$.  
> For operational penalty ratio $r=2$ ($\tau = 1/3$), the optimal quantile is $q_{1/3} = 5/12$. The optimal scalar calibration shifts are $\delta^* = 5/12 - 0.5 = -1/12$ and $\delta_\lambda^* = 5/12 - 0.55 = -2/15$. Consequently, both models yield identical dispatch commitments:
> $$p^* = \theta^* + \delta^* = \frac{5}{12} = q_{1/3}, \quad p_\lambda = \theta_\lambda + \delta_\lambda^* = \frac{5}{12} = q_{1/3}$$
> The excess operational risk is identically zero: $\Delta \mathcal{R}_{disp} = \mathcal{R}(p_\lambda) - \mathcal{R}(p^*) \equiv 0$.  
> *Conclusion*: Constant bias induced by physical regularization is fully absorbed by 1D decision calibration. MSE inflation alone does not guarantee positive excess operational risk.

> **Counterexample 2 (Spatial Heteroscedasticity: Regularized Bias Dominating Conditional Mean)**.  
> Let $X \in \{0, 1\}$ with $\mathbb{P}(X=0) = \mathbb{P}(X=1) = 0.5$. Suppose $Y \mid X=0 \sim \text{Uniform}[0.1, 0.3]$ (mean $\mu_0 = 0.2$, $q_{0} = 1/6$) and $Y \mid X=1 \sim \text{Uniform}[0.1, 0.9]$ (mean $\mu_1 = 0.5$, $q_{1} = 11/30$). The structural quantile gaps are heterogeneous: $\Delta_0 = -1/30 \approx -0.0333$ while $\Delta_1 = -4/30 \approx -0.1333$.  
> Consider a biased predictor $f_\lambda = (11/60, 13/30)^\top$ regularized toward the quantile vector. Although $f_\lambda$ has worse point-prediction MSE ($\Delta \text{MSE} = +0.002361$), when restricted to scalar calibration $\delta \in \mathbb{R}$, density-weighted optimization yields:
> $$\mathcal{R}(\boldsymbol{\mu} + \delta^*) = 0.174167, \quad \mathcal{R}(f_\lambda + \delta_\lambda^*) = 0.168542$$
> The biased predictor achieves a lower operational risk by $-0.005625$ than the true conditional mean.  
> *Conclusion*: Under spatial heteroscedasticity, scalar calibration cannot span non-uniform quantile biases, allowing physical priors that reshape predictions to achieve lower operational risk.

These counterexamples establish that physical constraints must be understood through **three open conditional mechanistic hypotheses** (rather than universal laws):
1. **Masked Pseudo-Benefit Hypothesis**: In uncalibrated settings, coupled physical penalties frequently provide apparent benefits by accidentally shrinking predictions toward asymmetric market quantiles.
2. **Exposed Representation Deficit Hypothesis**: In high-SNR, homogeneous regimes, decoupled calibration eliminates quantile mismatch, exposing representation distortions induced by physical priors as excess operational risk.
3. **Double-Edged Nature Hypothesis**: In complex terrain, low SNR, or perturbed data regimes, physical constraints function as defensive inductive guards, preventing catastrophic over-bidding and stabilizing balancing cashflows.

#### 3.6.6 Dynamic Electricity Settlement Multiplier Effect & BSC 30-Minute Physical Netting

Under real-world balancing settlement rules (UK Elexon BSC P305), operational imbalances are net-settled over 30-minute Settlement Periods (SP) at dynamic System Buy Prices $\lambda_{SP}^{SBP}$:
$$E_{SP_k} = \sum_{h \in SP_k} (p_{t,h} - y_{t,h}) \cdot \left( \frac{P_{rated}}{6000} \right) \quad (\text{MWh})$$
$$\text{Cashout}_{SP_k} = \lambda_{SP_k}^{SBP} \cdot \max(E_{SP_k}, 0) \quad (\pounds), \quad \text{SurplusRev}_{SP_k} = \lambda_{SP_k}^{SBP} \cdot \max(-E_{SP_k}, 0) \quad (\pounds)$$
$$\text{NetCashflow}_{SP_k} = \text{SurplusRev}_{SP_k} - \text{Cashout}_{SP_k} = -\lambda_{SP_k}^{SBP} \cdot E_{SP_k} \quad (\pounds)$$

Defining the normalized price multiplier $W(X) \triangleq \lambda^{SBP}(X) / \bar{\lambda}^{SBP}$ with $\mathbb{E}[W(X)] = 1.0$, the effective operational loss scales with $W(X)$. When price distributions exhibit extreme heavy tails ($\lambda_{\max}^{SBP} = \pounds 1,509.80\text{/MWh} = 34.1\times$ mean $\bar{\lambda} = \pounds 44.27\text{/MWh}$ in 2017 UK data, alongside 68 negative price periods down to $-\pounds 73.15\text{/MWh}$), price multipliers dramatically amplify operational divergence across wind farm topologies:
- **High-SNR Flat Terrain (Kelmarsh)**: Distribution constraints corrupt conditional mean accuracy, and price spikes amplify these errors into an annual shortfall cashout increase of **$+£932.56\text{ / (turb}\cdot\text{yr)}$** ($95\%$ CI: $[+£648.20, +£1,236.74]$) and a balancing net cashflow drop of **$-£3,264.86\text{ / (turb}\cdot\text{yr)}$** ($95\%$ CI: $[-£4,365.40, -£2,223.50]$).
- **Complex Terrain & Lower SNR (Hill of Towie)**: Distribution constraints act as a defensive guard against over-prediction. Under true timeline BSC 30-minute netting, calibrated distribution constraints reduce annual shortfall cashout by **$-£578.57\text{ / (turb}\cdot\text{yr)}$** ($95\%$ CI: $[-£834.86, -£367.51]$), while delivering an annual balancing net cashflow gain of **$+£1,481.27\text{ / (turb}\cdot\text{yr)}$** ($95\%$ CI: $[+£802.29, +£2,203.32]$).

---

## 4. Audit Protocol & Experimental Test Bench

### 4.1 Corpus Screening and Retrieval Protocol
We conducted a comprehensive literature retrieval targeting peer-reviewed journal papers and top conference proceedings published between January 2023 and September 2026. The retrieval adhered to a structured PRISMA-style protocol across two independent search channels (OpenAlex and Web Search API) using standardized queries:
`("physics-constrained" OR "physics-informed" OR "physics-guided") AND ("wind power forecasting" OR "wind power prediction")`.

```mermaid
graph TD
    A["Initial Records Retrieved (N = 53)"] --> B["Deduplication (N = 48 Studies)\n- 5 Duplicate Pairs Merged (Preprint vs. Journal)"]
    B --> C["Full-Text Eligibility Screening (N = 48)"]
    C --> D["Excluded Studies (N = 2)\n- S26: Target is Wind Speed, not Power\n- S42: Feature Subspace, no explicit constraint"]
    C --> E["Final Included Corpus (N = 46 Studies)\n- Full-Text Audited Across D1-D5"]
```

The screening procedure executed as follows:
1. **Initial Identification**: 53 records were retrieved across the search channels.
2. **Deduplication**: 5 duplicate pairs (e.g., early SSRN/arXiv preprints later published in journals such as *Applied Energy*, *Eng. App. AI*, and *EPSR*) were resolved, retaining the definitive journal versions and yielding 48 unique studies (`S01` to `S48`).
3. **Eligibility Screening**: Two studies were excluded with explicit, auditable criteria recorded in `study_level_ledger_48.csv`:
   - `S26` (Energy 2026): Excluded because the predicted target is meteorological wind speed rather than generator wind power, lacking turbine conversion dynamics.
   - `S42` (EPSR 2026): Excluded because the "physics guidance" was merely heuristic feature grouping into random forests without any explicit physical equation or penalty.
4. **Final Included Corpus**: 46 studies met all eligibility criteria and were subjected to exhaustive full-text coding recorded in `corpus48_编码_全维度集成版_杜俊涛.csv`.

### 4.2 Pilot-Stage Multi-Rater Reliability Benchmark (N=20)
To establish the baseline reproducibility of the Schema v4 taxonomy prior to schema cleansing and harmonized training, an independent pilot-stage tripartite blind audit was executed across 20 representative studies (including 2 negative controls) by three raters (Coder A, Coder B, and Coder C) using an unseeded coding schema. Inter-rater reliability was formally quantified using Cohen's and Fleiss' kappa ($\kappa$):
- **Eligibility Screening**: Fleiss $\kappa = 1.000$ (100% agreement on identifying and excluding negative controls S26 and S42);
- **D1 (Parameter Provenance)**: Fleiss $\kappa = 1.000$ overall and on included studies (100% agreement);
- **D2 (Application Stage)**: Fleiss $\kappa = 0.950$ overall ($\kappa = 0.938$ on included studies, Almost Perfect agreement);
- **D3 (Inference Requirement)**: Fleiss $\kappa = 0.743$ overall ($\kappa = 0.646$ on included studies, Substantial agreement);
- **D4 (Implementation Fidelity)**: Fleiss $\kappa = 0.379$ overall ($\kappa = 0.335$ on included studies, Fair/Low agreement). This reflects the inherent difficulty and divergent audit depths in open-source code verification;
- **D5 (Harm Reporting & Ablation)**: Fleiss $\kappa = 0.851$ overall ($\kappa = 0.818$ on included studies) for ablation type. For harm reporting across the 18 included wind forecasting studies, all three raters unanimously coded zero instances of harm reporting. Because there was zero marginal variance ($P_e = 1.0, 1 - P_e = 0$), Kappa is mathematically undefined ($0/0$) and is strictly reported as `NaN (Po = 100%)`.

This pilot benchmark reflects raw inter-rater concordance before iterative schema clarification. Following consensus arbitration, the full 46-study corpus was audited line-by-line with explicit equation-level citations.

### 4.3 Software Audit Protocol (PCT & PINN-UQ)
Of the two publicly accessible codebases identified in the corpus, the primary focus of static divergence auditing is the Physics-Constrained Transformer (PCT) repository (`https://github.com/daxin007/PCT-WPF`, *Scientific Reports* 2025). The second accessible repository, PINN-UQ for wind turbines (S10, `https://github.com/alfonsogijon/WindTurbines_PINNs`, *Eng. App. AI* 2025), was audited and verified to faithfully implement its specified algebraic power equations ($P = g T \omega$ and $P = \frac{1}{2} C_p \rho A v^3$ via `sn.Tie`), formally retracting an earlier preliminary allegation of an ODE substitution. For PCT, the repository was cloned and subjected to static mathematical verification. We audited line-by-line parity between the published equations (Eq. 1–5 in the paper) and the PyTorch implementation in `train_clean.py` and `tst/loss.py`.

### 4.4 Downstream Dispatch and Dynamic Price Settlement Test Bench
To ground the theoretical framework in real-world power systems across heterogeneous terrains, we established an empirical dispatch test bench across three operational UK wind farms with 10-minute SCADA telemetry:
1. **Hill of Towie Wind Farm**: 48.3 MW rated capacity, 21 Siemens SWT-2.3-82 turbines, serving as a complex upland blind test site (7,599 test origins, 45,594 delivery steps across full-year 2017).
2. **Kelmarsh Wind Farm**: 12.3 MW rated capacity, 6 Senvion MM82 turbines, serving as a core flat-terrain development site (8,212 test origins, 49,272 delivery steps across full-year 2017).
3. **Penmanshiel Wind Farm**: 28.7 MW rated capacity, 14 Senvion MM82 turbines (WT01–WT15, with WT03 absent) located in the Scottish Borders [Zenodo 5946808], serving as an upland verification benchmark (8,440 test origins).

The operational evaluation adheres strictly to the frozen protocol:
- **Temporal Configuration**: $L=36$ (6-hour lookback), $H=36$ (6-hour forward forecast), $H_{disp}=\{1..6\}$ (next-hour delivery window, six 10-minute intervals), and $S=6$ (hourly rolling origination without overlapping settlement periods).
- **Benchmarked Models**: Unconstrained DLinear, Bounds-Constrained DLinear ($\lambda=0.01$), Empirical Power Curve Constrained DLinear ($\lambda=0.01$), Conditional Distribution Prior DLinear ($\lambda=0.01$), Ridge regression, and Persistence baseline across 5 random initialization seeds.
- **Offline Decision Calibration**: Post-hoc quantile offset $\delta_h$ is frozen strictly from the independent 2016 calibration set ($q = 1/(r+1) = 1/3$ for penalty ratio $r=2$) with zero access to 2017 SCADA or market prices.
- **Real-World BSC 30-Minute Physical Net Settlement**: Evaluated against official 2017 UK Elexon BSC System Buy Prices (mean £44.27/MWh, peak £1,509.80/MWh, with 68 negative price periods down to -£73.15/MWh). Cashflows account for 30-minute netting ($E_{SP} = \sum_{t \in SP} (p_t - y_t) \cdot P_{rated}/6000$ MWh), single-sided shortfall cashouts, and two-sided full market cashflows with 2,000-replication 7-day circular block paired bootstrap.

---

## 5. Empirical Audit Results

### 5.1 Finding 1: The Literature Harm Reporting Gap (0/46 = 0.0%)
Across all 46 full-text audited studies, **not a single paper reported any operational condition, data regime, or evaluation metric under which the physical constraint resulted in negative performance** (`d5_negative_effect_reported == 'no'` across 46/46 studies, Evidence ID: `EVID-C2-01`).

```
Total Included Studies:      46
Negative Effects Reported:   0  (0.0%)
Harm Metrics Evaluated:      0  (0.0%)
Prior Stress-Testing Done:   3  (6.5%)
```

In every reviewed manuscript, the proposed physics constraint is reported to uniformly improve accuracy over unconstrained baselines. Only 3 studies (6.5%, S01, S02, S10) evaluated any form of prior parameter perturbation, while 43/46 (93.5%) tested physical constraints only under ideal, nominal conditions. This demonstrates a pervasive literature reporting gap across the domain: physical constraints are treated exclusively as enhancements rather than engineering trade-offs.

### 5.2 Finding 2: The Public Code Reproducibility Barrier
Out of 46 reviewed studies, **44 papers (95.7%) provided no publicly accessible software codebase** (`D4_fidelity == 'not-found'`, Evidence ID: `EVID-C2-02`). Across the entire corpus:
- 44 studies provided neither a public repository link nor reproducible scripts (95.7%);
- 1 study (PCT, S02) provided a public GitHub link, but code auditing revealed a directional implementation mismatch with its published specification (2.2%, `code-mismatch`);
- 1 study (PINN-UQ, S10) provided a public GitHub repository verified as mathematically consistent with its published algebraic equations (2.2%, `code-aligned`).

This finding underscores that reproducibility in physics-constrained wind forecasting lags significantly behind general deep learning, preventing independent verification of reported physical advantages.

### 5.3 Finding 3: Distribution of Constraint Provenance, Inference Requirements, and Code Fidelity
Table 1 presents the empirical distributions across Dimensions D1, D2, D3, and D4 across the 46 studies (Evidence ID: `EVID-C2-04`).

**Table 1: Distribution of Constraint Lifecycle Dimensions Across 46 Audited Studies**

| Dimension | Category | Count (N=46) | Proportion (%) | Typical Exemplars |
|---|---|---:|---:|---|
| **D1: Parameter Provenance** | `intrinsic` | 40 | 87.0% | Turbine Betz envelope, Jensen wake equation, rated limits |
| | `historical-static` | 5 | 10.9% | Empirical KDE power curve fitted on historical training SCADA |
| | `learned-online` | 1 | 2.2% | Auxiliary PINN predicting dynamic $C_p(t)$ online |
| **D2: Application Stage** | `training-loss` | 22 | 47.8% | Soft penalty term added to MSE loss ($\lambda \mathcal{L}_{\text{phys}}$) |
| | `inference-architecture` | 22 | 47.8% | Dual-stream physical fusion branch, projection layers |
| | `data-generation` | 1 | 2.2% | Physics-guided generative diffusion augmentation |
| | `post-hoc-correction` | 1 | 2.2% | Conformal prediction calibration on post-inference residuals |
| **D3: Inference Requirement** | `none` | 40 | 87.0% | Standard historical inputs; no external runtime dependencies |
| | `forecast-input` | 4 | 8.7% | Requires future NWP meteorological wind speed at test time |
| | `contemporaneous-state` | 2 | 4.3% | Requires synchronized telemetry from upstream turbines |
| **D4: Implementation Fidelity** | `not-found` | 44 | 95.7% | No public code repository or dead link |
| | `code-mismatch` | 1 | 2.2% | PCT (reverse-KL implemented instead of forward JS loss) |
| | `code-aligned` | 1 | 2.2% | PINN-UQ (algebraic tie constraints matching published equations) |

The audit shows that the vast majority of physical constraints rely on intrinsic physical laws ($87.0\%$) and operate without requiring external runtime dependencies ($87.0\%$). Constraints are evenly split between soft loss penalties ($47.8\%$) and hard architectural integration ($47.8\%$).

### 5.4 Finding 4: Operational Dispatch Settlement and Site Divergence (Table 2)
To empirically evaluate whether physical constraints provide true operational benefits, Table 2 compiles the comprehensive 2017 dispatch settlement metrics under real-world dynamic balancing prices with BSC 30-minute physical netting across Hill of Towie and Kelmarsh (Evidence ID: `EVID-SETTLE-01`, derived from `dynamic_price_settlement_table2.csv`).

**Table 2: Operational Balancing Cashflow & Settlement Audit under Real-World Dynamic Prices (2017 UK Elexon SBP, BSC 30-Min Netting)**

| Site | Model & Constraint Variant | Policy | RMSE (pu) | Normalized Dynamic Cost $\text{Cost}_{dyn}$ (mean±std) | Single-Sided Shortfall £/(turb·yr) | Balancing Imbalance Cashflow £/(turb·yr) | Full Site Shortfall Cashout (£) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Hill of Towie**<br>(48.3 MW, 21 turb)<br>*External Blind Test* | **Unconstrained DLinear** | Uncalibrated<br>**Quantile Calibrated** | 0.1034<br>0.1074 | 0.09707 ± 0.00130<br>**0.08758 ± 0.00037** | £23,340.66<br>**£14,457.73** | -£7.70<br>**£19,616.06** | £463,296.14<br>**£286,976.13** |
| | **Bounds Constrained** ($\lambda=0.01$) | Uncalibrated<br>Quantile Calibrated | 0.1034<br>0.1074 | 0.09707 ± 0.00130<br>0.08758 ± 0.00037 | £23,340.75<br>£14,457.89 | -£7.85<br>£19,615.74 | £463,297.87<br>£286,979.17 |
| | **Power Curve Constrained** ($\lambda=0.01$) | Uncalibrated<br>Quantile Calibrated | 0.1034<br>0.1073 | 0.09701 ± 0.00129<br>0.08757 ± 0.00037 | £23,317.71<br>£14,480.73 | £6.49<br>£19,526.66 | £462,840.47<br>£287,432.56 |
| | **Distribution Prior** ($\lambda=0.01$) | Uncalibrated<br>Quantile Calibrated | 0.1040<br>0.1084 | 0.09540 ± 0.00117<br>0.08835 ± 0.00029 | £22,529.56<br>**£13,879.17** | -£139.71<br>**£21,097.33** | £447,196.26<br>**£275,492.00** |
| | **Ridge Regression** | Uncalibrated<br>**Quantile Calibrated** | 0.1027<br>0.1064 | 0.09512 ± 0.00000<br>**0.08685 ± 0.00000** | £22,742.22<br>**£14,650.23** | £421.31<br>**£18,464.10** | £451,417.52<br>**£290,797.13** |
| | **Persistence Baseline** | Uncalibrated<br>Quantile Calibrated | 0.1072<br>0.1120 | 0.09640 ± 0.00000<br>0.08992 ± 0.00000 | £23,479.38<br>£13,950.54 | -£1,460.29<br>£22,897.90 | £466,049.70<br>£276,908.73 |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Kelmarsh**<br>(12.3 MW, 6 turb)<br>*Core Development* | **Unconstrained DLinear** | Uncalibrated<br>**Quantile Calibrated** | 0.0985<br>0.1014 | 0.09578 ± 0.00223<br>**0.08829 ± 0.00029** | £19,013.89<br>**£12,977.84** | £4,209.70<br>**£17,289.68** | £112,520.57<br>**£76,800.38** |
| | **Bounds Constrained** ($\lambda=0.01$) | Uncalibrated<br>Quantile Calibrated | 0.0985<br>0.1014 | 0.09579 ± 0.00222<br>0.08829 ± 0.00029 | £19,018.38<br>£12,979.56 | £4,200.27<br>£17,285.06 | £112,547.15<br>£76,810.53 |
| | **Power Curve Constrained** ($\lambda=0.01$) | Uncalibrated<br>Quantile Calibrated | 0.0984<br>0.1014 | 0.09575 ± 0.00223<br>0.08825 ± 0.00028 | £19,013.01<br>£12,961.24 | £4,186.15<br>£17,309.19 | £112,515.37<br>£76,702.11 |
| | **Distribution Prior** ($\lambda=0.01$) | Uncalibrated<br>Quantile Calibrated | 0.0987<br>0.1005 | 0.09536 ± 0.00334<br>0.08922 ± 0.00060 | £19,085.36<br>**£13,910.40** | £2,439.15<br>**£14,024.82** | £112,943.51<br>**£82,319.11** |
| | **Ridge Regression** | Uncalibrated<br>**Quantile Calibrated** | 0.0988<br>0.1017 | 0.09724 ± 0.00000<br>0.08962 ± 0.00000 | £20,066.57<br>**£13,728.46** | £2,759.37<br>**£16,598.74** | £118,750.11<br>**£81,242.36** |
| | **Persistence Baseline** | Uncalibrated<br>Quantile Calibrated | 0.1014<br>0.1029 | 0.09729 ± 0.00000<br>0.09132 ± 0.00000 | £20,666.60<br>£15,166.78 | -£144.43<br>£12,410.39 | £122,300.95<br>£89,754.12 |

*Notes on Table 2*:
1. **Economic Scope & Market Clarification**: Balancing Imbalance Cashflow is defined strictly under UK Elexon BSC P305 rules as $-\lambda_{SP}^{SBP} \cdot E_{SP} = (y - p) \cdot \lambda_{SP}^{SBP}$ (Surplus Cashout Revenue minus Shortfall Cashout Expenditure). It measures imbalance settlement cashflow directly exchanged with the System Operator and **does not include wholesale forward or day-ahead contract sales ($p \cdot \lambda^{DA}$)**. Under conservative quantile calibration ($r=2, \tau=1/3$), models bid lower, resulting in actual generation exceeding declared bids ($y > p$) in approximately 67% of intervals. In positive system price environments, this surplus energy is credited at SBP, which explains why conservative predictors record positive balancing cashflow credits by selling uncontracted energy into the balancing pool rather than forward markets. Total plant operating profit must account for day-ahead contract commitments; Metric C represents solely the imbalance cashflow component.
2. **Timeline Aggregation & Coverage**: Evaluated across continuous timelines aggregated into intact 3-interval Settlement Periods (15,152 SPs on Hill of Towie, 86.5% coverage; 16,396 SPs on Kelmarsh, 93.6% coverage). Annualized figures scale the mean daily cashflow across scored calendar days to 365 days. Raw observed period farm totals are reported in the rightmost column.

Table 2 highlights two crucial empirical insights:
1. **The Inversion and Terrain Divergence**: In uncalibrated settings, the distribution prior appears marginally superior on both sites. Once decoupled quantile calibration is applied, however, the two wind farms diverge markedly:
   - At **Kelmarsh** (flat terrain, high SNR), unconstrained DLinear achieves the highest conditional mean fidelity. Calibrated distribution constraints incur an excess shortfall penalty of **$+£932.56\text{ / (turb}\cdot\text{yr)}$** and a balancing net cashflow drop of **$-£3,264.86\text{ / (turb}\cdot\text{yr)}$**.
   - At **Hill of Towie** (complex upland terrain), distribution regularization prevents over-bidding. Its single-sided shortfall penalty drops by **$-£578.57\text{ / (turb}\cdot\text{yr)}$**, and its two-sided balancing cashflow component increases by **$+£1,481.27\text{ / (turb}\cdot\text{yr)}$**, generating **+£31,107/yr** in balancing cashout credits across the 21-turbine farm.
2. **Third Benchmark Verification (Penmanshiel Wind Farm)**: On the third operational site, Penmanshiel (Scotland upland, 28.7 MW, 14 Senvion MM82 turbines), calibrated physical constraints also outperform unconstrained DLinear without inversion under nominal clean conditions ($r=2$): unconstrained DLinear achieves dispatch cost **$0.09864$**, while default power curve constrained ($\lambda=0.01, \text{factor}=1.0$) achieves **$0.09753$** and default distribution prior ($\lambda=0.01, \text{factor}=1.0$) achieves **$0.09804$** (with misspecified factor=0.9 reaching $0.09780$). Both physical constraints protect DLinear against representation degradation. However, neither physical constraint outperforms the simple empirical baselines on this site (Ridge: **$0.09744$**, Persistence: **$0.09586$**). This empirical divergence supports our Double-Edged Nature Hypothesis: physical constraints are neither universally harmful nor universally beneficial, but function as context-dependent inductive guards whose utility is contingent on the neural architecture and the operating regime.

### 5.5 Finding 5: Statistical Significance and Paired Bootstrap Contrasts
Table 3 presents 2,000-replication 7-day circular block bootstrap paired contrasts against unconstrained DLinear (Evidence ID: `EVID-BOOT-01`, derived from `dynamic_price_paired_intervals.csv`).

**Table 3: 7-Day Circular Block Paired Bootstrap Contrasts vs. Unconstrained DLinear (Calibrated Policy)**

| Site | Method Contrast | Dynamic Cost Diff $\Delta \text{Cost}_{dyn}$ (95% CI) | Cashout Diff £/(turb·yr) | Cashout Diff 95% CI | Balancing Cashflow Diff £/(turb·yr) | Balancing Cashflow Diff 95% CI |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Hill of Towie** | **Bounds** vs DLinear | $+0.000000$ ($[+0.000000, +0.000000]$) | $+£0.15$ | $[-£0.17, +£0.51]$ | $-£0.32$ | $[-£1.37, +£0.62]$ |
| | **Power Curve** vs DLinear | $-0.000008$ ($[-0.000021, +0.000007]$) | $+£22.99$ | $[+£10.93, +£38.43]$ | $-£89.40$ | $[-£133.51, -£54.70]$ |
| | **Distribution** vs DLinear | **$+0.000766$** ($[+0.000277, +0.001201]$) | **$-£578.57$** | $[-£834.86, -£367.51]$ | **$+£1,481.27$** | $[+£802.29, +£2,203.32]$ |
| | **Ridge** vs DLinear | $-0.000733$ ($[-0.001315, -0.000125]$) | $+£192.50$ | $[-£26.66, +£445.74]$ | $-£1,151.96$ | $[-£1,567.98, -£795.28]$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Kelmarsh** | **Bounds** vs DLinear | $+0.000001$ ($[+0.000000, +0.000002]$) | $+£1.72$ | $[-£0.01, +£4.63]$ | $-£4.62$ | $[-£12.67, +£0.05]$ |
| | **Power Curve** vs DLinear | $-0.000039$ ($[-0.000054, -0.000023]$) | $-£16.61$ | $[-£33.12, +£0.87]$ | $+£19.51$ | $[-£27.76, +£62.86]$ |
| | **Distribution** vs DLinear | **$+0.000932$** ($[+0.000423, +0.001385]$) | **$+£932.56$** | $[+£648.20, +£1,236.74]$ | **$-£3,264.86$** | $[-£4,365.40, -£2,223.50]$ |
| | **Ridge** vs DLinear | $+0.001338$ ($[+0.000917, +0.001794]$) | $+£750.61$ | $[+£330.70, +£1,151.06]$ | $-£690.94$ | $[-£1,669.76, +£409.51]$ |

As evidenced by Table 3, the paired contrasts confirm site-dependent divergence with definitive statistical rigor under circular block bootstrapping: at Kelmarsh, distribution constraints significantly increase shortfall cashout and decrease balancing cashflow; at Hill of Towie, they significantly reduce shortfall cashout and increase balancing cashflow by $+£1,481.27$/(turb·yr). Standard decoupled quantile calibration itself unlocks farm-wide shortfall savings of **£186,541/yr** on Hill of Towie and **£36,216/yr** on Kelmarsh compared to uncalibrated dispatch, proving that decision calibration is the dominant driver of operational economics.

---

## 6. Ablations, Fidelity Mismatches, and Diagnostic Case Study

### 6.1 Causal Confounding in Published Ablation Studies
A core claim in physics-constrained literature is that ablation experiments prove the utility of the physical inductive bias. Our audit examined the exact semantic design of every reported ablation study across the 46 papers (Evidence ID: `EVID-C2-03`).

**Table 4: Semantic Breakdown of Reported Ablation Studies (N=46)**

| Ablation Semantic | Count | Proportion (%) | Causal Interpretability |
|---|---:|---:|---|
| **Strict Constraint Isolation** (`constraint-on-off`) | 16 | 34.8% | **Valid**: Directly evaluates $\lambda=0$ vs. $\lambda>0$ on identical architecture. |
| **Both Strict & Module** (`both`) | 8 | 17.4% | **Valid**: Isolates both physical penalty and structural modules. |
| **Module-Only Ablation** (`module-ablation`) | 22 | 47.8% | **Confounded**: Removes entire deep network block (e.g., GNN layer); cannot attribute gain to physics. |
| **Strict Physical Isolation Total** | **24** | **52.2%** | Only roughly half the literature isolates the physical constraint. |

As revealed in Table 4, **nearly half of the literature (47.8%, 22 studies) suffers from causal confounding**. In these studies, authors claim to "validate physical constraints" by removing an entire structural sub-network (such as a spatial graph attention layer, a cross-attention fusion block, or a dual-stream encoder). Removing these blocks simultaneously eliminates millions of trainable parameters and representational capacity. Consequently, the observed degradation cannot be attributed to the physical equations, but reflects standard neural capacity reduction.

### 6.2 Case Study: The Physics-Constrained Transformer (PCT)
To investigate implementation fidelity, we performed a deep static code audit of the Physics-Constrained Transformer (PCT, *Scientific Reports* 2025). 

#### 6.2.1 Reverse-JS Divergence Mismatch
The published paper specifies that the model enforces physical turbine power curve alignment by penalizing the symmetric Jensen–Shannon (JS) divergence between the predicted joint distribution $p$ and the empirical power curve distribution $q$:
$$\mathcal{L}_{\text{JS}}(p \parallel q) = \frac{1}{2} \text{KL}\left(p \parallel \frac{p+q}{2}\right) + \frac{1}{2} \text{KL}\left(q \parallel \frac{p+q}{2}\right)$$

However, inspecting the released implementation in `train_clean.py:341` reveals the following loss calculation:
```python
# Official PCT Implementation (train_clean.py:341)
loss_physics = F.kl_div(pred_dist.log(), target_dist)
```
In PyTorch, the functional call `F.kl_div(input, target)` is defined mathematically as:
$$\text{kl\_div}(u, v) = \sum_i v_i \left(\log v_i - u_i\right)$$
Setting `input = pred_dist.log()` ($u = \log p$) and `target = target_dist` ($v = q$) computes:
$$\sum_i q_i \left(\log q_i - \log p_i\right) = \text{KL}(q \parallel p)$$
Consequently, the implementation optimizes the **reverse Kullback–Leibler divergence** $\text{KL}(q \parallel p)$ (and by extension a reverse-JS variant $\frac{1}{2}[\text{KL}(m \parallel p) + \text{KL}(m \parallel q)]$) rather than the standard forward divergence (Evidence ID: `EVID-C3-01`). Because the KL divergence is famously asymmetric, $\text{KL}(p \parallel q)$ is mode-covering, whereas $\text{KL}(q \parallel p)$ is mode-seeking and heavily penalizes zero-density regions of $p$, fundamentally altering the regularization behavior.

#### 6.2.2 Mathematical Verification of the 2D Product-Kernel Density Estimator
An earlier audit hypothesis had alleged that PCT's joint density computation in `train_clean.py`:
```python
# Joint KDE kernel computation
joint_kernel = torch.matmul(kernel_p.transpose(1, 2), kernel_w)
```
computed an outer product of marginal distributions, thereby erroneously enforcing an independence assumption ($p(w, P) = p(w)p(P)$) that destroys the physical power-speed correlation. 

To resolve this, an independent mathematical derivation was conducted. Let $s \in \{1, \dots, N\}$ index samples, and let $b_i, b_j$ index bin centers. `kernel_p` has shape $(1, N, B_p)$ with entries $\phi(b_i; P_s)$, and `kernel_w` has shape $(1, N, B_w)$ with entries $\phi(b_j; w_s)$. Transposing `kernel_p` and computing the batch matrix multiplication yields:
$$\left[K_p^\top K_w\right]_{i, j} = \sum_{s=1}^N \phi(b_i; P_s) \cdot \phi(b_j; w_s)$$
This expression is the exact definition of a **paired two-dimensional product-kernel density estimator** evaluated over paired observation samples $(w_s, P_s)$ (Evidence ID: `EVID-C3-02`). It does **not** assume independence. The allegation of marginal independence was proven mathematically invalid and formally retracted. This negative finding illustrates the necessity of rigorous mathematical derivation before leveling implementation criticisms against published artifacts.

---

## 7. Discussion, Threats to Validity & Governance Reflections

### 7.1 Methodological Limitations
We explicitly articulate three primary threats to the validity of this audit:
1. **Multi-Rater Scope and Full-Corpus Verification**: The inter-rater reliability protocol was benchmarked on a representative 20-study sample by three independent raters (Coder A, B, and C) through a rigorous, fully blind pilot inter-rater reliability test, achieving unanimous consensus on eligibility screening (Fleiss $\kappa = 1.000$) and high agreement across parameter provenance (D1: 1.000) and application stage (D2: 0.938). Code fidelity (D4: 0.335 on included studies, 0.379 overall) exhibited lower agreement prior to arbitration, reflecting the prevalence paradox and confirming that code fidelity verification cannot rely on surface text extraction alone. To extend independent multi-rater verification across all 46 included studies, a standalone double-blind review package for the remaining 26 studies is archived in the supplementary materials, enabling continuous external blind replication.
2. **Limited Public Software Baselines**: Out of 46 included studies, only two public code repositories were identified across the entire domain (PCT and PINN-UQ). While PCT exhibited a directional reverse-KL divergence mismatch and PINN-UQ was verified as code-aligned, we caution against generalizing PCT's specific software discrepancy to unreleased proprietary methods.
3. **Exploratory Status of Downstream Training Reruns**: Prior internal attempts to quantify exact downstream RMSE degradations (e.g., claiming 2.83–8.50$\times$ error magnification) were compromised by target leakage and loss-scale discrepancies in earlier benchmark scripts. Consequently, downstream training reruns remain strictly exploratory notes in the supplementary materials and are not asserted as primary statistical claims.

### 7.2 Governance Lessons from Internal Adversarial Audits
This manuscript itself underwent five iterative rounds of internal adversarial review, resulting in the formal retraction of multiple unsupported preliminary hypotheses (including the outer-product independence charge and unverified RMSE ratio assertions). We consider this transparent governance process a vital asset: rigorous empirical science requires actively searching for refutations of one's own findings before publication.

### 7.3 Actionable Checklist for Future Physics-Constrained Research
Based on our findings, we propose a four-point checklist for researchers developing physics-constrained forecasting models:
1. **Mandate Isolated Constraint Ablation**: Always report a baseline where the physical penalty is set to zero ($\lambda=0$) on the exact proposed network architecture to isolate physical gains from model capacity.
2. **Conduct Parameter Misspecification Stress-Tests**: Evaluate the model under perturbed, shifted, or noisy physical parameters (e.g., air density fluctuations, blade degradation) to quantify sensitivity.
3. **Verify Software Gradient Alignment**: Verify that software divergence functions match published equations in PyTorch/TensorFlow (paying particular attention to argument order in asymmetric divergence functions like `kl_div`).
4. **Disclose Runtime Input Requirements**: Explicitly state whether the constraint requires concurrent test-time forecasts (D3), ensuring operational feasibility during field deployment.

---

## 8. Conclusion

This paper has presented a systematic empirical audit of physics-constrained wind power forecasting. By establishing the five-dimensional Constraint Lifecycle Taxonomy, auditing 46 full-text domain studies, and verifying the sole public software codebase, we demonstrated that the prevailing assumption of universal constraint benefit remains largely unverified. The complete absence of reported negative trade-offs, combined with widespread code unavailability (97.8%) and prevalent ablation confounding (47.8%), highlights a pressing need for empirical rigor. Furthermore, both our formal theoretical derivations (Proposition 1, mechanistic hypotheses, and boundary counterexamples) and empirical dispatch audits under real-world BSC 30-minute dynamic balancing electricity prices demonstrate that physical constraints function as double-edged inductive priors: while distorting conditional mean representations and increasing shortfall penalties in simple flat terrain (Kelmarsh), they provide defensive regularization against extreme over-bidding in complex upland terrain (Hill of Towie and Penmanshiel). Decoupling prediction representation from risk decision calibration provides an essential operational baseline to isolate these effects. We encourage the energy and machine learning communities to adopt transparent lifecycle reporting, isolated ablation protocols, and open-source verification to ensure that physics-constrained models achieve genuine reliability in the decarbonized power grid.

---

## Appendix A: Mathematical Analysis of Decoupled Calibration Boundaries and Formal Counterexamples

This appendix provides a rigorous mathematical analysis of the boundary conditions governing decoupled decision calibration, formalizing the two counterexamples introduced in Section 3.6.5 and proving when decoupled calibration strictly dominates.

### A.1 Functional Setting and Hilbert Space Projections
Let $(\Omega, \mathcal{F}, \mathbb{P})$ be the foundational probability space. Let $X: \Omega \to \mathcal{X}$ be the measurable conditioning vector, and let $Y \in L^2(\Omega, \mathcal{F}, \mathbb{P})$ denote the true power generation target with range $[0, 1]$. We denote by $\sigma(X) \subset \mathcal{F}$ the sub-$\sigma$-algebra generated by $X$, and define the Hilbert space $\mathcal{H} \triangleq L^2(\Omega, \sigma(X), \mathbb{P})$ equipped with inner product $\langle f, g \rangle \triangleq \mathbb{E}_X[f(X)g(X)]$ and norm $\|f\|_{L^2} = \sqrt{\mathbb{E}[f^2]}$.

By the Hilbert Projection Theorem, the conditional expectation $\mu(X) \triangleq \mathbb{E}[Y \mid \sigma(X)]$ is the unique orthogonal projection of $Y$ onto $\mathcal{H}$, satisfying:
$$\mathbb{E}\left[ (Y - \mu(X)) \cdot h(X) \right] = 0, \quad \forall h \in \mathcal{H}$$
Consequently, for any candidate predictor $f \in \mathcal{H}$, the mean-squared error satisfies the Pythagorean decomposition:
$$\mathbb{E}\left[ (Y - f(X))^2 \right] = \mathbb{E}\left[ (Y - \mu(X))^2 \right] + \|f - \mu\|_{L^2}^2$$

### A.2 The Breakdown of Second-Order Stochastic Dominance (SOSD) for General Deviations
In dispatch decision analysis, the operational loss $\ell_r(p, Y) = (r+1)\rho_\tau(Y - p)$ is a convex function of forecast error $E = Y - p$. Under standard decision theory, error distribution $E^*$ is said to second-order stochastically dominate $E_\lambda$ ($E^* \succeq_{SSD} E_\lambda$) if and only if $\mathbb{E}[\phi(E^*)] \le \mathbb{E}[\phi(E_\lambda)]$ for all convex $\phi$.

By the Rothschild–Stiglitz Theorem, $E^* \succeq_{SSD} E_\lambda$ holds if and only if $E_\lambda$ is a mean-preserving spread of $E^*$, requiring:
$$E_\lambda \overset{d}{=} E^* + Z, \quad \text{with } \mathbb{E}[Z \mid E^*] = 0$$
Let $E^* = (Y - \mu(X)) - \delta^*$ and $E_\lambda = (Y - \mu(X)) - \delta_\lambda^* - (f_\lambda(X) - \mu(X))$. The difference is $Z = E_\lambda - E^* = (\delta^* - \delta_\lambda^*) - (f_\lambda(X) - \mu(X)) \triangleq -\eta(X)$. 
For a general non-uniform function $\eta(X)$, $\mathbb{E}[-\eta(X) \mid E^*] \ne 0$ in general because $E^*$ depends on $X$ through the conditional distribution $Y \mid X$. Thus, the martingale difference condition fails, and Second-Order Stochastic Dominance cannot be asserted distribution-free.

### A.3 Formal Analysis of Counterexample 1 (Constant Shift Invariance)
Consider $Y \sim \text{Uniform}[0.25, 0.75]$ with unconstrained mean $\theta^* = 0.5$.
1. **Prediction Risk**: $\mathcal{R}_{pred}(\theta) = \mathbb{E}[(Y - \theta)^2] = (\theta - 0.5)^2 + 1/48$, which is 2-strongly convex.
2. **Coupled Prior**: Let $\mathcal{R}_{phys}(\theta) = (\theta - 0.6)^2$. The prior gradient at $\theta^*$ is $\nabla \mathcal{R}_{phys}(0.5) = -0.2 \ne 0$.
3. **Regularized Solution**: For $\lambda = 1$, the regularized optimizer is $\theta_\lambda = 0.55$. The point MSE degrades strictly: $\Delta \text{MSE} = (0.55 - 0.5)^2 = 0.0025 > 0$.
4. **Decision Risk**: For $r = 2$ ($\tau = 1/3$), the true optimal quantile is $q_{1/3} = 0.25 + 0.5 \times (1/3) = 5/12 \approx 0.416667$.
   - Unconstrained optimal scalar shift: $\delta^* = q_{1/3} - \theta^* = 5/12 - 1/2 = -1/12$.
   - Regularized optimal scalar shift: $\delta_\lambda^* = q_{1/3} - \theta_\lambda = 5/12 - 0.55 = -2/15$.
   - Resulting decisions:
     $$p^* = \theta^* + \delta^* = \frac{5}{12} = q_{1/3}, \quad p_\lambda = \theta_\lambda + \delta_\lambda^* = \frac{5}{12} = q_{1/3}$$
   - Both decisions are identical to the Bayes-optimal quantile $q_{1/3}$. Therefore:
     $$\Delta \mathcal{R}_{disp} = \mathcal{R}(p_\lambda) - \mathcal{R}(p^*) \equiv 0$$
This formally disproves the existence of an inevitable strictly positive lower bound $\mathcal{O}(\lambda^2 \|\mathbf{g}_{phys}\|^2) > 0$. When physical regularization exerts a uniform translation, 1D decision calibration completely neutralizes the prior shift.

### A.4 Formal Analysis of Counterexample 2 (Spatial Heteroscedasticity)
Let $X \in \{0, 1\}$ with $\mathbb{P}(X=0) = \mathbb{P}(X=1) = 0.5$.
1. **Conditional Distributions**:
   - $Y \mid X=0 \sim \text{Uniform}[0.1, 0.3]$, with mean $\mu_0 = 0.2$ and quantile $q_{0} = 0.1 + 0.2 \times (1/3) = 1/6 \approx 0.166667$.
   - $Y \mid X=1 \sim \text{Uniform}[0.1, 0.9]$, with mean $\mu_1 = 0.5$ and quantile $q_{1} = 0.1 + 0.8 \times (1/3) = 11/30 \approx 0.366667$.
2. **Conditional Bias Vector**: $\boldsymbol{\Delta} = (\Delta_0, \Delta_1)^\top = (1/6 - 0.2, 11/30 - 0.5)^\top = (-1/30, -4/30)^\top$.
3. **Biased Prior Predictor**: Suppose a physical regularizer pulls predictions toward the quantile vector: $f_\lambda = \frac{\boldsymbol{\mu} + \mathbf{q}}{2} = (11/60, 13/30)^\top$.
   - The point-prediction MSE is strictly worse than the conditional mean:
     $$\text{MSE}(f_\lambda) - \text{MSE}(\boldsymbol{\mu}) = 0.5 \left( (11/60 - 0.2)^2 + (13/30 - 0.5)^2 \right) = 0.002361 > 0$$
4. **Optimal Scalar Calibrations**:
   - For conditional mean $\boldsymbol{\mu}$: optimizing scalar offset $\delta$ under check loss yields $\delta^* = -0.05$. Evaluating expected dispatch risk:
     $$\mathcal{R}(\boldsymbol{\mu} + \delta^*) = 0.174167$$
   - For regularized predictor $f_\lambda$: optimizing scalar offset $\delta_\lambda$ yields $\delta_\lambda^* = -0.025$. Evaluating expected dispatch risk:
     $$\mathcal{R}(f_\lambda + \delta_\lambda^*) = 0.168542$$
   - The biased model achieves a strictly lower dispatch loss:
     $$\mathcal{R}(f_\lambda + \delta_\lambda^*) - \mathcal{R}(\boldsymbol{\mu} + \delta^*) = -0.005625 < 0$$
Under spatial heteroscedasticity, scalar calibration cannot span non-constant quantile gaps $\Delta_\tau(X)$. Consequently, a coupled physical prior that partially reshapes the predictor toward the heterogeneous quantile surface can outperform the unconstrained conditional mean.

### A.5 Sufficient Conditions for Decoupled Dominance
Combining Proposition 1 with the counterexamples above yields the exact sufficient condition under which decoupled calibration strictly outperforms coupled regularizers:

> **Proposition A.1 (Sufficient Conditions for Decoupled Calibration Dominance)**.  
> Let the conditional distribution satisfy strict homoscedasticity $\sigma(X) \equiv \sigma$ such that the Bayes-optimal quantile is an exact translation of the conditional mean: $q_\tau(X) = \mu(X) + \sigma z_\tau$. Suppose the unconstrained model achieves the conditional mean $f^*(X) = \mu(X)$ a.s., while coupled regularization introduces a non-constant distortion $\eta(X) = f_\lambda(X) - \mu(X)$ with $\text{Var}_X(\eta(X)) > 0$.  
> Then for any scalar calibration $\delta \in \mathbb{R}$, $p_\lambda(X) \ne q_\tau(X)$ on a set of positive measure, and by Proposition 1:
> $$\mathbb{E}[\ell_r(p^*(X), Y)] < \mathbb{E}[\ell_r(p_\lambda(X), Y)]$$
> Decoupled calibration strictly dominates coupled loss constraints if and only if the target quantile surface is spanned by the translation group of the conditional mean. When non-stationary heteroscedasticity or severe noise corrupts data-driven representation learning, physical priors act as protective regularization guards.

---

## Data Availability Statement
All audit matrices, coding protocols, and evaluation scripts are publicly available in the project repository:
- Integrated 48-study full-text coding table: `study_level_ledger_48.csv` and `corpus48_编码_全维度集成版_杜俊涛.csv`.
- Traceable evidence ledger: `.experiment-to-paper/evidence.jsonl` and `.experiment-to-paper/claims_evidence.md`.
- Benchmark test logs: `06_实验结果与日志/table2_robustness_data.csv` and `06_实验结果与日志/latency_fixed.csv`.

## References
[1] H. Holttinen, et al., "Design and operation of power systems with large amounts of wind power," *IEA Wind TCP Task 25*, 2021.  
[2] J. Yan, et al., "A review on wind speed and wind power forecasting," *Renewable and Sustainable Energy Reviews*, vol. 58, pp. 429-441, 2016.  
[3] Y. Liu, et al., "Key technologies and challenges of wind power forecasting," *Proceedings of the CSEE*, vol. 34, no. 29, pp. 4991-5004, 2014.  
[4] G. Ke, et al., "LightGBM: A highly efficient gradient boosting decision tree," *NeurIPS*, 2017.  
[5] H. Zhou, et al., "Informer: Beyond efficient transformer for long sequence time-series forecasting," *AAAI*, 2021.  
[6] S. Wang, et al., "TimeMixer: Decomposable multiscale mixing for time series forecasting," *ICLR*, 2024.  
[7] Y. Liu, et al., "iTransformer: Inverted transformers are effective for time series forecasting," *ICLR*, 2024.  
[8] G. E. Karniadakis, et al., "Physics-informed machine learning," *Nature Reviews Physics*, vol. 3, no. 6, pp. 422-440, 2021.  
[9] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks," *Journal of Computational Physics*, vol. 378, pp. 686-707, 2019.  
[10] J. Willard, et al., "Integrating physics-based modeling with machine learning: A survey," *ACM Computing Surveys*, vol. 55, no. 1, 2022.  
[11] M. Wang, et al., "Physics-constrained generative augmentation driven wind power forecasting," *Energy*, vol. 312, p. 141322, 2026.  
[12] H. Zhang, et al., "NCF: A physics-constrained pre-trained method for short-term wind power forecasting," *Renewable Energy*, vol. 240, p. 125710, 2026.  
[13] L. Chen, et al., "Physics-informed dynamic wake graphs for probabilistic wind-farm power forecasting," *Electric Power Systems Research*, vol. 241, p. 113241, 2026.  
[14] K. Liu, et al., "Physics-constrained graph attention learning for wind turbine power forecasting," *Measurement Science and Technology*, vol. 37, p. 035102, 2026.  
[15] X. Luo, et al., "Physics-constrained wind power forecasting aligned with probability distributions," *Applied Energy*, vol. 381, p. 125295, 2025.  
[16] Y. Wang, X. Luo, et al., "Physics-constrained transformer for wind power forecasting," *Scientific Reports*, vol. 15, p. 34348, 2025.  
[17] M. Hutson, "Artificial intelligence faces reproducibility crisis," *Science*, vol. 359, no. 6377, pp. 725-726, 2018.  
[18] J. Pineau, et al., "Improving reproducibility in machine learning research," *JMLR*, vol. 22, no. 164, pp. 1-20, 2021.  
[19] B. Haibe-Kains, et al., "Transparency and reproducibility in artificial intelligence," *Nature*, vol. 586, pp. E14-E16, 2020.  
[20] A. Zeng, et al., "Are transformers effective for time series?" *AAAI*, 2023.  
[21] S. Garg, et al., "Can simple linear models outperform deep transformers in time series?" *ICML Workshop*, 2024.  
[22] D. Niu, et al., "A survey of wind power forecasting based on deep learning and physics-informed models," *Applied Energy*, vol. 350, p. 121703, 2023.  
[23] Z. Lin, et al., "Physics-guided deep learning for renewable energy forecasting: A comprehensive review," *Renewable and Sustainable Energy Reviews*, vol. 189, p. 113942, 2024.
