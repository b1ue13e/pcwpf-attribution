# -*- coding: utf-8 -*-
import os

base_dir = r"e:\论文1\01_论文投稿材料\active_manuscript"
sections_dir = os.path.join(base_dir, "sections")
os.makedirs(sections_dir, exist_ok=True)

# -------------------------------------------------------------
# 1. sections/01_intro.tex
# -------------------------------------------------------------
sec01 = r"""\section{Introduction}
\label{sec:intro}

In electric power systems and wholesale electricity markets, the operational value of wind power forecasting is governed by asymmetric decision losses in unit commitment, economic dispatch, and real-time balancing settlement \cite{holttinen2021,weron2014}. Under standard market clearing and system balancing rules, reserve shortfall penalties for under-delivery are typically substantially more severe than downward redispatch or spill penalties for over-delivery \cite{zhang2014wind}. In these asymmetric operational regimes, conventional symmetric regression metrics, such as Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), fail to reflect tail operational risk and downside financial exposure \cite{lipton2019,pinson2007trading}. To improve generalization and enforce physically realistic outputs, an expanding body of literature embeds aerodynamic limits, empirical power curves, kinematic wake equations, and joint probability distribution constraints directly into deep neural network architectures and loss functions \cite{karniadakis2021physics,raissi2019physics,sheng2025annreview,pct2026}. A foundational premise underpins this research paradigm: that reported reductions in forecasting error can be directly attributed to the valid physical domain knowledge encoded within the constraint.

However, prevailing reporting conventions, software availability deficits, and ablation designs obscure this attributional premise. First, contemporary studies predominantly document uniform performance gains across nominal test conditions; operational failure regimes, adverse trade-offs, and constraint breakdowns remain largely undocumented. Second, public source code availability remains rare in this domain, leaving the mathematical fidelity of published loss formulations and computational graph executions largely uninspected \cite{gundersen2018,kapoor2023,pineau2021}. Third, when ablation studies are reported, they frequently excise multi-layer neural submodules (such as graph convolutional branches or attention blocks) rather than toggling the constraint penalty multiplier while holding network capacity constant. Such architectural ablations inevitably confound domain inductive bias with parametric representation capacity. Consequently, the literature leaves central empirical questions unresolved: Does a physical penalty succeed because the network internalizes genuine aerodynamic relationships, or because the penalty merely acts as a non-specific numerical regularizer? Under what training sample budgets, temporal proxy delays, and evaluation metrics do these empirical advantages persist, diminish, or invert?

To examine the attributional basis of physics-constrained gains, we conduct an end-to-end empirical audit across three commercial UK wind farms (Kelmarsh, Hill of Towie, and Penmanshiel) coupled with 2017 UK Elexon Balancing and Settlement Code (BSC) market clearing prices. Holding the forecasting backbone and post-hoc quantile calibration constant, we introduce form- and weight-matched pseudo-scrambled priors as negative controls, systematically perturb prior parameters, and evaluate model behavior across sample sizes and loss formulations. Our controlled experiments reveal that physical constraints exhibit a fundamentally \textbf{double-edged nature} governed by sample volume, prior alignment, and decision calibration:
\begin{enumerate}[leftmargin=*]
\item \textbf{Prior Specificity under Sample Scarcity}: On Penmanshiel under restricted training data ($N_{\text{train}}=1{,}583$), embedding an authentic empirical power curve prior reduces normalized asymmetric decision loss by 1.22\%. Conversely, permuting the wind-to-power mapping via a pseudo-scrambled control inflates decision loss by 68.1\%, demonstrating prior specificity under strong regularization in low-sample regimes.
\item \textbf{Diminishing Advantages under Sample Expansion}: When evaluated across within-site training sample budgets ($N_{\text{train}} \in \{500, \dots, 5000\}$), fixed historical empirical priors provide modest dispatch cost reductions under constrained training windows ($N_{\text{train}} \le 1{,}000$). Evaluating an information-budget ablation under strictly matched sample windows ($N_{\text{prior}} = N_{\text{train}} = 500$) confirms that domain-structured empirical prior effects retain 77\% to 92\% of this benefit, refuting the hypothesis that prior gains stem solely from seasonal data leakage. However, as model training volume expands to $N_{\text{train}} = 5{,}000$, unconstrained data-driven models close the gap and slightly outperform fixed priors (+0.33\% on Kelmarsh, +1.73\% on Hill of Towie).
\item \textbf{Computational Graph Divergence and Settlement Decoupling}: Static AST decompilation of the open-source Physics-Constrained Transformer (PCT) benchmark uncovers an inverted argument order in its divergence loss, inducing an analytical inverse-density gradient singularity that amplifies gradient norms by $2.20\times$ to $2.72\times$. Downstream, market settlement analysis demonstrates that standalone unhedged balancing cashflows decouple from forecast quality: a degenerate zero-bid baseline ($q \equiv 0$) mechanically generates six-figure apparent cashflows purely through uncontracted physical spill, confirming that standalone unhedged imbalance accounting cannot serve as an index of forecasting skill.
\end{enumerate}

\subsection{Summary of Contributions}
This investigation addresses the attribution and evaluation of physics-constrained forecasting through three interconnected contributions:
\begin{itemize}[leftmargin=*]
\item \textbf{Systematic Scoping Audit of Literature and Software Fidelity}: Guided by the PRISMA Extension for Scoping Reviews (PRISMA-ScR), we evaluate 46 contemporary studies (2023--2026), quantifying the complete reporting absence of operational failure regimes (0/46), the open-code deficit (44/46), and the prevalence of structural module-level ablation confounding (23/46). Static AST inspection uncovers an inverted divergence formulation in the sole publicly accessible end-to-end training codebase (PCT), while formal algebraic tensor expansion verifies that its paired kernel product maintains authentic bivariate density estimation.
\item \textbf{Backbone-Matched Controlled Multi-Site Empirical Benchmark}: By pairing authentic priors with form- and weight-matched scrambled negative controls across multi-seed evaluations on three commercial wind farms, we distinguish domain-specific aerodynamic guidance from non-specific numerical smoothing. We delineate the empirical boundary conditions under which physical constraints assist as defensive guardrails, become indistinguishable from random regularization, or degrade downstream decision performance.
\item \textbf{Prediction-Decision Decoupling Theory and Governance Protocol}: We derive the exact integral representation of excess operational risk under asymmetric dispatch loss, decompose it into representation error, calibration dispersion, and cross-coupling interaction components, and establish formal counterexamples delineating the limits of scalar calibration. We demonstrate why unhedged single-settlement cashflows decouple from forecast quality and operationalize our findings into a five-gate REFORMS-PCWPF reporting checklist for reproducible renewable energy forecasting.
\end{itemize}

\paragraph{Empirical Scope and Methodological Boundaries}
This investigation operates within four defined empirical boundaries: first, the literature audit evaluates reporting and ablation practices across the 46 studies identified under our prespecified PRISMA-ScR protocol (38 journal articles and 8 preprints, 2023--2026), characterizing reporting patterns within this scoping cohort rather than asserting universal claims across unbounded keyword taxonomies; second, the software audit inspects the computational graph of the open-source PCT implementation; third, empirical ablations evaluate boundary behaviors across three commercial UK wind farms under matched DLinear backbones, with PatchTST serving as an architectural sensitivity probe; and fourth, downstream evaluations analyze operational decision risk and stylized imbalance cashflow diagnostics under UK BSC settlement prices, characterizing metric failure modes rather than full commercial portfolio clearing.
"""

# -------------------------------------------------------------
# 2. sections/02_related.tex
# -------------------------------------------------------------
sec02 = r"""\section{Related Work and Positioning}
\label{sec:related_work}

To position this scoping audit within computational energy systems and empirical machine learning methodology, we review three related literatures: (i) physics-constrained formulations in renewable energy forecasting, (ii) operational decision evaluation in wind power prediction, and (iii) reproducibility, reporting standards, and software auditing in scientific machine learning.

\subsection{Physical Constraints in Wind Power Forecasting}
Embedding physical domain knowledge into neural networks aims to combine governing physical principles with data-driven function approximation \cite{karniadakis2021physics,raissi2019physics}. In wind power forecasting, these constraints typically follow three distinct technical formulations:

\paragraph{Operational Envelope Bounds} Enforcing aerodynamic operating thresholds (such as cut-in, rated, and cut-out wind speeds, or theoretical Betz envelopes) via bounded activation functions or post-hoc projections guarantees that outputs respect physical capacity boundaries \cite{tgdpf2025}. However, hard clipping can alter gradient flows near boundary cutoffs. When turbulent boundary-layer conditions push turbine operations near rated capacity, zero-gradient saturation regions can impede gradient descent, restricting local adaptation to rapid meteorological transitions.

\paragraph{Kinematic Wake Formulations} Spatio-temporal models commonly embed steady-state kinematic wake equations---such as the 1D Jensen or Bastankhah--Port\'{e}-Agel Gaussian deficit models \cite{sheng2025annreview,s34_prx2023}---into Graph Neural Network (GNN) adjacency matrices. While this approach incorporates structural layout geometry, substituting static engineering approximations for dynamic Navier--Stokes fluid dynamics can constrain network representations during complex convective conditions. Furthermore, published studies frequently attribute performance gains to the embedded physical formulation without isolating whether the expressive capacity of the GNN backbone alone explains the improvement.

\paragraph{Distributional Regularization} Penalizing deviations from historical joint wind speed--power distributions via statistical divergence objectives (such as Wasserstein distance or Jensen--Shannon divergence \cite{pct2026,tgdpf2025}) introduces additional optimization dynamics \cite{krishnapriyan2021}. Static distribution alignment regularizes predictions against pre-computed empirical manifolds. When meteorological regimes shift or input observations suffer from temporal lag, divergence penalties can penalize genuine operational transients, creating optimization trade-offs between distribution fidelity and immediate dispatch accuracy.

\subsection{Decision-Theoretic Evaluation and Settlement Mechanics}
Wind power forecasting models have traditionally been evaluated using symmetric statistical error metrics, such as MAE, RMSE, and $R^2$ \cite{madsen2005,optis2022}. Madsen et al. \cite{madsen2005} established standardized verification protocols highlighting the importance of evaluating models across horizons, operational regimes, and normalized capacity metrics. Optis et al. introduced the \textit{WE-Validate} framework \cite{optis2022}, emphasizing distributionally resistant metrics tailored to complex atmospheric conditions.

However, power system operations demonstrate that statistical error reductions do not guarantee economic value under asymmetric market penalties \cite{weron2014,zhang2014wind}. Pinson et al. \cite{pinson2007trading} demonstrated that optimal bidding under asymmetric imbalance penalties is governed by conditional quantiles matching the cost ratio, highlighting the operational decoupling between point error and decision risk. Downstream grid dispatch penalizes reserve shortfalls far more severely than excess generation \cite{dao2018cointegration,yan2025economicvalue}. Furthermore, real-world balancing arrangements---such as the UK Elexon Balancing and Settlement Code (BSC)---aggregate energy deviations over discrete 30-minute settlement periods. In isolated single-component accounting, unhedged cashflows can create apparent monetary surpluses through systematic negative bias, decoupling cashflow from forecasting skill. Evaluating forecasters strictly on uncalibrated point errors or unhedged cashflows obscures their true operational efficacy.

\subsection{Reproducibility Standards and Software Auditing in Scientific AI}
Empirical machine learning has faced increasing scrutiny over data leakage, underspecification, and implementation discrepancies \cite{damour2022,gundersen2018,kapoor2023,lipton2019}. Kapoor and Narayanan \cite{kapoor2023} documented how subtle data leakage vectors undermine reproducibility across seventeen scientific domains. Gundersen and Kjensmo \cite{gundersen2018} evaluated 400 papers from major AI conferences, finding that fewer than 20\% shared executable source code or reproducible experimental protocols.

To address these vulnerabilities, procedural frameworks have gained traction. Pineau et al. \cite{pineau2021} developed the NeurIPS Machine Learning Reproducibility Checklist, demonstrating that explicit reporting mandates improve disclosure of code, data, hyperparameter choices, and error bounds. D'Amour et al. \cite{damour2022} showed that models with equivalent in-distribution test accuracy often diverge under distribution shift due to underspecified training criteria. Concurrently, Lipton and Steinhardt \cite{lipton2019} analyzed recurring pitfalls in empirical scholarship: confusing empirical association with theoretical explanation, failing to isolate independent sources of performance gains, and employing decorative mathematics that does not govern empirical outcomes.

While general machine learning venues have increasingly adopted these auditing norms, physics-constrained energy forecasting has developed largely without standardized reporting controls. As summarized in Table~\ref{tab:lit_comparison}, existing reviews focus primarily on model typologies, rarely evaluating constraint provenance, software execution fidelity, or ablation confounding. This investigation operationalizes these empirical standards through systematic auditing of lifecycle provenance, software parity, and backbone-matched controlled ablations.

\begin{table*}[!t]
\centering
\small
\caption{Positioning this Scoping Empirical Audit Relative to Existing Literature Reviews in Wind Power Forecasting and Scientific ML.}
\label{tab:lit_comparison}
\begin{tabularx}{\textwidth}{p{4.2cm}>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X}
\toprule
\textbf{Survey / Meta-Review} & \textbf{Decouples} & \textbf{Audits Stage} & \textbf{Checks Inference} & \textbf{Verifies Code} & \textbf{Scrutinizes} \\
\textbf{Literature Reference} & \textbf{Provenance ($\mathcal{D}_1$)} & \textbf{Separation ($\mathcal{D}_2$)} & \textbf{Telemetry ($\mathcal{D}_3$)} & \textbf{Parity ($\mathcal{D}_4$)} & \textbf{Ablation ($\mathcal{D}_5$)} \\
\midrule
Yan et al. (2016) \cite{yan2016review} & Partial & No & Partial & No & No \\
Maldonado-Correa et al. (2020) \cite{maldonadocorrea2020slr} & No & No & No & No & No \\
Karniadakis et al. (2021) \cite{karniadakis2021physics} & Conceptual & Conceptual & No & No & No \\
Optis et al. (WE-Validate, 2022) \cite{optis2022} & No & No & Partial & Open Benchmark & No \\
Sheng et al. (2025) \cite{sheng2025annreview} & No & Partial & No & No & No \\
\midrule
\textbf{This Scoping Empirical Audit} & \textbf{Yes (Schema v4)} & \textbf{Yes (Schema v4)} & \textbf{Yes (Schema v4)} & \textbf{Yes (Static Audit)} & \textbf{Yes (Strict Isolation)} \\
\bottomrule
\end{tabularx}
\end{table*}
"""

# -------------------------------------------------------------
# 3. sections/03_theory.tex
# -------------------------------------------------------------
sec03 = r"""\section{Mathematical Formulation and Decision-Theoretic Foundation}
\label{sec:theory}

In this section, we establish the mathematical foundation governing operational wind power dispatch and market settlement. We formulate the operational problem under asymmetric balancing costs, derive the exact integral representation of excess operational risk, decompose it into representation error and calibration components, and establish formal boundary counterexamples delineating the limits of scalar calibration.

\subsection{Operational Dispatch Problem Formulation}
Let $X \in \mathcal{X} \subset \mathbb{R}^{L \times D}$ denote historical multivariate telemetry, and let $Y \in [0, 1]$ denote normalized turbine active power output over the target delivery horizon. In operational power system balancing, transmission system operators evaluate declared bids under asymmetric pinball dispatch penalties \cite{weron2014,zhang2014wind}:
\begin{equation}
\ell(p, Y; r) = c_+ \max(p - Y, 0) + c_- \max(Y - p, 0),
\label{eq:pinball_loss}
\end{equation}
where $p \in [0, 1]$ is the declared dispatch commitment, $c_+$ is the shortfall unit penalty (under-generation where declared power exceeds physical delivery, $p > Y$, requiring expensive tertiary reserve activation), $c_-$ is the surplus unit penalty (over-generation, $Y > p$, inducing downward redispatch or curtailment), and $r = c_+ / c_-$ represents the cost asymmetry ratio. Normalizing by $c_-$, the asymmetric operational loss is:
\begin{equation}
\mathcal{L}_{\text{disp}}(p, Y) = r \max(p - Y, 0) + \max(Y - p, 0).
\label{eq:norm_disp_loss}
\end{equation}
The expected operational risk under policy $p(X)$ is $\mathcal{R}(p) = \mathbb{E}_{X, Y} [\mathcal{L}_{\text{disp}}(p(X), Y)]$.

\subsection{Lemma 1: Exact Integral Excess Operational Risk}
Lemma~\ref{lem:excess_risk} formalizes a classical excess risk analytical identity in statistical decision theory and quantile regression, grounded in the foundational formulations of Koenker \cite{koenker2005} and electricity market bidding principles established by Pinson et al. \cite{pinson2007trading}. The unconstrained Bayes-optimal dispatch decision minimizing $\mathcal{R}(p)$ is the conditional quantile:
\begin{equation}
q_\tau(X) = F_{Y|X}^{-1}(\tau), \quad \text{where } \tau = \frac{1}{r + 1}.
\label{eq:bayes_quantile}
\end{equation}
For an operational cost asymmetry ratio $r=2$, the critical quantile is $\tau = 1/3 \approx 0.333$.

\begin{lemma}[Exact Integral Excess Operational Risk \cite{koenker2005,pinson2007trading}]
\label{lem:excess_risk}
For any dispatch decision rule $p(X) \in [0, 1]$, the excess operational risk relative to the Bayes-optimal quantile $q_\tau(X)$ satisfies the exact integral identity:
\begin{equation}
\begin{aligned}
\mathcal{R}(p) - \mathcal{R}(q_\tau) &= (r + 1) \, \mathbb{E}_X \left[ \int_{q_\tau(X)}^{p(X)} \Big( F_{Y|X}(u) - \tau \Big) \, du \right] \\
&\ge 0,
\end{aligned}
\label{eq:exact_integral}
\end{equation}
which is non-negative everywhere. Under standard regularity conditions where $F_{Y|X}$ is strictly increasing on its support such that $q_\tau(X)$ is uniquely defined, equality holds if and only if $p(X) = q_\tau(X)$ almost surely.
\end{lemma}

\begin{proof}
For a fixed feature vector $X$, the expected conditional loss at commitment $p$ is:
\begin{equation}
\mathbb{E}[ \mathcal{L}_{\text{disp}}(p, Y) \mid X ] = r \int_0^p (p - y) dF(y) + \int_p^1 (y - p) dF(y),
\end{equation}
where $F(y) = F_{Y|X}(y)$. Taking the partial derivative with respect to $p$:
\begin{equation}
\frac{\partial}{\partial p} \mathbb{E}[ \mathcal{L}_{\text{disp}}(p, Y) \mid X ] = r F(p) - (1 - F(p)) = (r + 1) F(p) - 1.
\end{equation}
Recalling that $\tau = 1 / (r+1)$, this gradient simplifies to:
\begin{equation}
\frac{\partial}{\partial p} \mathbb{E}[ \mathcal{L}_{\text{disp}}(p, Y) \mid X ] = (r + 1) \Big( F(p) - \tau \Big).
\end{equation}
By the Fundamental Theorem of Calculus, integrating the gradient from $q_\tau(X)$ to $p(X)$ yields:
\begin{equation}
\mathbb{E}[ \mathcal{L}_{\text{disp}}(p, Y) \mid X ] - \mathbb{E}[ \mathcal{L}_{\text{disp}}(q_\tau, Y) \mid X ] = (r+1) \int_{q_\tau(X)}^{p(X)} \Big( F(u) - \tau \Big) du.
\end{equation}
Because $F(u)$ is monotonically non-decreasing, for any $u > q_\tau(X)$ we have $F(u) \ge \tau$, so the integrand is non-negative; for any $u < q_\tau(X)$ we have $F(u) \le \tau$, and integrating backwards from $q_\tau(X)$ to $p(X)$ also yields a positive integral. Taking the outer expectation over $\mathcal{X}$ completes the proof.
\end{proof}

\subsection{Two-Stage Decomposition and Decoupled Decision Calibration}
When $p(X)$ is in the neighborhood of $q_\tau(X)$, expanding $F_{Y|X}(u)$ via a first-order Taylor series around $q_\tau(X)$ yields:
\begin{equation}
F_{Y|X}(u) - \tau \approx f_{Y|X}(q_\tau(X)) \cdot (u - q_\tau(X)),
\end{equation}
which produces the local quadratic excess risk approximation:
\begin{equation}
\mathcal{R}(p) - \mathcal{R}(q_\tau) \approx \frac{r+1}{2} \, \mathbb{E}_X \big[ f_{Y|X}(q_\tau(X)) \cdot ( p(X) - q_\tau(X) )^2 \big].
\label{eq:quadratic_approx}
\end{equation}

Now consider the standard two-stage paradigm where a forecaster predicts $f_\theta(X)$ aiming for conditional expectation $\mu(X) = \mathbb{E}[Y \mid X]$, and the dispatch commitment is formed via a scalar quantile offset $\delta$: $p(X) = f_\theta(X) + \delta$. Expressing the prediction error as:
\begin{equation}
p(X) - q_\tau(X) = \big( f_\theta(X) - \mu(X) \big) + \big( \delta - \Delta_\tau(X) \big),
\end{equation}
where $\Delta_\tau(X) = q_\tau(X) - \mu(X)$ denotes the conditional quantile-mean shift. Under homoscedastic curvature $w_\tau = \frac{r+1}{2} f_{Y|X}(q_\tau)$, substituting into Eq.~\eqref{eq:quadratic_approx} decomposes excess operational risk into three distinct terms:
\begin{equation}
\begin{aligned}
\mathcal{R}(p) - \mathcal{R}(q_\tau) &\approx w_\tau \, \mathbb{E}_X \big[ (f_\theta(X) - \mu(X))^2 \big] \\
&\quad + w_\tau \, \mathbb{E}_X \big[ (\delta - \Delta_\tau(X))^2 \big] \\
&\quad + 2 w_\tau \, \mathbb{E}_X \big[ (f_\theta(X) - \mu(X))(\delta - \Delta_\tau(X)) \big],
\end{aligned}
\label{eq:three_terms}
\end{equation}
where the components represent \textbf{Term I} (Representation Error / MSE), \textbf{Term II} (Quantile Mismatch / Calibration Dispersion), and \textbf{Term III} (Cross-Coupling Interaction Component), respectively.

\subsection{Mechanistic Deductions from Decision-Theoretic Decomposition}
The three-term decomposition in Eq.~\eqref{eq:three_terms} provides a mathematical interpretation for our empirical findings, yielding three deductive mechanistic insights:

\begin{remark}[Mechanistic Deduction 1: Pseudo-Advantage Masking in Uncalibrated Regimes]
\label{rem:masking}
In uncalibrated forecasting regimes ($\delta=0$), Term~II ($w_\tau \mathbb{E}[\Delta_\tau(X)^2]$) is an irreducible data-dependent constant representing the dispersion between conditional mean and target quantile. A physics-regularized model with inferior conditional mean accuracy (larger Term~I) can paradoxically achieve lower uncalibrated operational risk if its structural bias $(f_\theta(X) - \mu(X))$ aligns with the negative quantile gap $\Delta_\tau(X) = q_\tau(X) - \mu(X) < 0$ (for $r=2, \tau=1/3$). This alignment drives the interaction Term~III ($2 w_\tau \mathbb{E}[(f_\theta(X) - \mu(X))(-\Delta_\tau(X))] < 0$) negative, accidentally cancelling out a portion of representation error. Upon post-hoc quantile calibration, $\delta$ absorbs this systematic offset, neutralizing Term~III and revealing that the apparent physical advantage was an artifact of uncalibrated evaluation.

\textit{Diagnostic Caveat}: We emphasize that Term~III serves as a decision-theoretic diagnostic framework for explaining how structural bias can mask forecasting error under uncalibrated evaluation, rather than an empirically isolated causal factor. In operational telemetry where the true conditional distribution $F_{Y|X}$ is unknown, isolating the exact empirical magnitude of Term~III point-by-point is intractable without strong parametric assumptions.
\end{remark}

\begin{remark}[Mechanistic Deduction 2: Regularization Distortion under Well-Sampled Regimes]
\label{rem:exposure}
Optimal quantile calibration ($\delta^* = \arg\min_\delta \hat{\mathcal{R}}(f_\theta + \delta)$) removes the unconditional component of Term~II. In well-sampled regimes where unconstrained models already achieve low conditional mean estimation variance ($\text{Term~I}$), injecting an uncalibrated physical loss penalty acts as a distortive structural bias that inflates Term~I without offering compensating variance reduction, generating the net operational degradation observed under flat open terrain.
\end{remark}

\begin{remark}[Mechanistic Deduction 3: Inductive Guardrails and Prior Sensitivity under Sample Restriction]
\label{rem:inductive_guardrails}
In operating regimes with restricted effective sample size, empirical parameter estimation is ill-posed, predisposing unconstrained models to heightened representational variance (inflating Term~I). Under sufficiently strong penalty weight ($\lambda=1.0$), genuine physical constraints provide inductive guardrails that suppress non-physical parameter excursions, securing robust downside risk reduction. Conversely, when this mapping is misspecified or permuted (as in our scrambled-prior negative control), the rigid penalty enforces counter-physical gradient updates, precipitating severe performance degradation and demonstrating acute vulnerability to prior misspecification under strong regularization.
\end{remark}

\subsection{Formal Boundary Counterexamples (Possibility Results)}
To ensure theoretical precision and delineate the limits of scalar calibration, we establish two formal possibility / existence counterexamples demonstrating that decoupled calibration dominance is conditional rather than absolute.

Under \textit{Constant Shift Invariance} (Counterexample~\ref{ex:constant_shift}), if a physical constraint modifies a forecaster purely by a uniform additive shift $f_{\text{phys}}(X) = f_{\text{uncons}}(X) + c$, the subsequent optimal scalar calibration absorbs $c$ completely ($\delta_{\text{phys}}^* = \delta_{\text{uncons}}^* - c$), resulting in identical dispatch decisions and zero excess risk difference ($\Delta \mathcal{R}_{\text{disp}} \equiv 0$) despite an apparent increase in prediction MSE. Under \textit{Spatial Heteroscedasticity} (Counterexample~\ref{ex:heteroscedasticity}) with input-dependent variance $\sigma(X)$, a global scalar calibration fails to compensate for localized variance shifts; here, an estimator that introduces structural bias and exhibits higher MSE can achieve strictly lower operational risk than the true conditional mean following scalar quantile offset. These existence results prove that while scalar calibration successfully decouples mean prediction from global quantile alignment, it does not guarantee Pareto dominance over locally adaptive or asymmetric loss training under severe conditional heteroscedasticity. Complete mathematical proofs for both counterexamples are provided in \ref{sec:appendix_proofs}.

\subsection{Two-Settlement Wholesale Accounting and Real-Time BSC Imbalance Netting}
In wholesale electricity markets, wind power producers operate within a two-settlement framework coupling forward contracts and physical real-time imbalance settlement \cite{morales2014,pinson2007trading}. Total commercial revenue is expressed as:
\begin{equation}
R_{\text{total}} = P_{\text{F}} \cdot q + P_{\text{imb}} \cdot (y - q),
\label{eq:two_settlement}
\end{equation}
where $P_{\text{F}}$ denotes the forward contract price (e.g., bilateral PPA or day-ahead exchange clearing price), $q$ is the forward-contracted or nominated energy position in the stylized model, $y$ is actual physical generation volume, and $P_{\text{imb}}$ is the system imbalance cash-out price. Under the UK Balancing and Settlement Code (BSC) single-price regime (post-P305), energy imbalance cashflows aggregate over discrete half-hour Settlement Periods (SP):
\begin{equation}
E_{\text{SP}} = \sum_{t \in \text{SP}} (y_t - q_t) P_{\text{rated}} \Delta t \quad [\text{MWh}], \quad \Delta t = \frac{1}{6}\text{ h},
\label{eq:bsc_netting}
\end{equation}
where $P_{\text{rated}}$ denotes total wind farm nameplate capacity in MW. Physical shortfall occurs when $E_{\text{SP}} < 0$, incurring settlement charges evaluated at the System Buy Price $\lambda_t^{\text{SBP}}$. The heavy-tailed price distribution (mean \pounds 44.27/MWh, peak \pounds 1,509.80/MWh) acts as a high-magnitude risk multiplier. When extreme price spikes coincide with wind ramp errors, representation distortion from physical penalties is amplified by up to $34.1\times$. However, as demonstrated in Section~\ref{sec:experiments}, evaluating forecasters on isolated unhedged balancing cashflows $P_{\text{imb}} \cdot (y - q)$ without forward revenue $P_{\text{F}} \cdot q$ creates metric decoupling, establishing that non-negative asymmetric decision loss serves as the primary ground-truth metric.
"""

# -------------------------------------------------------------
# 4. sections/04_methodology.tex
# -------------------------------------------------------------
sec04 = r"""\section{Empirical Audit Framework and Experimental Methodology}
\label{sec:methodology}

This section details the empirical audit framework, systematic literature scoping protocol, operational telemetry testbed, and controlled ablation designs.

\subsection{The Five-Dimensional Constraint Lifecycle Taxonomy}
Designating a neural network as ``physics-constrained'' often groups disparate mathematical, structural, and empirical operations under a single umbrella term \cite{lipton2019}. To establish an operational foundation for systematic auditing, we define a structured \textbf{Constraint Lifecycle Taxonomy}.

The primary unit of analysis is the individual \textbf{forecasting study} ($N=46$ unique scoping studies), characterized by its primary operational constraint mechanism. Formally, we characterize the primary constraint architecture $C$ of each audited study as a 5-tuple operating across orthogonal lifecycle stages:
\begin{equation}
C = \big\langle \mathcal{D}_1, \mathcal{D}_2, \mathcal{D}_3, \mathcal{D}_4, \mathcal{D}_5 \big\rangle,
\label{eq:constraint_tuple}
\end{equation}
where $\mathcal{D}_1$ represents parameter provenance, $\mathcal{D}_2$ denotes application stage, $\mathcal{D}_3$ specifies inference telemetry requirements, $\mathcal{D}_4$ captures implementation software fidelity, and $\mathcal{D}_5$ evaluates ablation semantics and harm reporting (Figure~\ref{fig:lifecycle}).

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{fig_constraint_lifecycle.pdf}
\caption{The Five-Dimensional Constraint Lifecycle Framework. Tracking domain physical constraints from parameter origin ($\mathcal{D}_1$) through computational graph formulation ($\mathcal{D}_2$), operational telemetry dependency ($\mathcal{D}_3$), software implementation parity ($\mathcal{D}_4$), and empirical ablation validity ($\mathcal{D}_5$).}
\label{fig:lifecycle}
\end{figure*}

\paragraph{Dimension 1: Parameter Provenance ($\mathcal{D}_1$)}
Parameter provenance identifies the epistemic origin and calibration basis of numerical constraint parameters: (1)~\textit{intrinsic constraints}, where parameters derive analytically from governing physical invariants or manufacturer limits without empirical fitting (e.g., the Betz limit $C_p \le 16/27$ or kinematic wake equations); (2)~\textit{historical-static priors}, where reference curves or joint distributions are estimated offline from historical operational SCADA archives and held fixed during inference; (3)~\textit{learned-online constraints}, where parameters adapt dynamically during inference via auxiliary neural subnetworks; and (4)~\textit{unreported}, where calibration details are omitted.

\paragraph{Dimension 2: Application Stage ($\mathcal{D}_2$)}
This dimension identifies where domain physics enters the computational pipeline: (1)~\textit{training-loss regularization} ($\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{phys}}$); (2)~\textit{inference-architecture constraints}, modifying layers to project outputs onto admissible manifolds; (3)~\textit{data-generation priors}, synthesizing augmented training scenarios; and (4)~\textit{post-hoc correction filters}, applied externally after inference.

\paragraph{Dimension 3: Inference Telemetry Requirements ($\mathcal{D}_3$)}
Inference telemetry identifies external operational data streams required during real-time deployment: (1)~\textit{none}, relying solely on target turbine historical power; (2)~\textit{forecast-input}, requiring future numerical weather prediction inputs; and (3)~\textit{contemporaneous-state}, requiring synchronized real-time measurements from adjacent turbines or met masts.

\paragraph{Dimension 4: Implementation Fidelity ($\mathcal{D}_4$)}
Implementation fidelity evaluates whether public software faithfully executes published mathematical formulations: (1)~\textit{code-aligned}, where public source code strictly matches published equations; (2)~\textit{code-mismatch}, where the codebase exhibits an algebraic or directional deviation; and (3)~\textit{not-found}, where no public repository or script accompanies the study.

\paragraph{Dimension 5: Harm Reporting and Ablation Semantics ($\mathcal{D}_5$)}
This dimension examines whether evaluations causally isolate physical constraints and report operational failure boundaries: (1)~\textit{constraint-on-off}, where treatment and baseline share identical parameter capacity, differing only in the activation of the physical prior ($\lambda=0$ vs.\ $\lambda>0$); (2)~\textit{module-ablation}, where the ablation excises an entire multi-parameter neural block; (3)~\textit{both}; and (4)~\textit{negative-effect-reported}, explicitly documenting operating regimes where physical constraints degrade downstream performance.

\subsection{Systematic Literature Audit Protocol (PRISMA-ScR)}
We conducted an empirical scoping audit in accordance with the PRISMA Extension for Scoping Reviews (PRISMA-ScR) guidelines \cite{tricco2018}. Literature retrieval was executed across two complementary channels covering January 1, 2023, through March 1, 2026, with rolling monitoring through July 2026. First, programmatic queries via the OpenAlex API using four targeted title queries ($Q_1$: \texttt{"physics-constrained" AND "wind"}, $Q_2$: \texttt{"physics-informed" AND "wind power"}, $Q_3$: \texttt{"physics-guided" AND "wind"}, $Q_4$: \texttt{"physics-embedded" AND "wind power"}) identified 72 candidate records. Second, targeted searches on Google Scholar, arXiv, and preprint servers yielded 17 additional records ($N=89$ total).

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{prisma_flow_diagram.pdf}
\caption{PRISMA-ScR Flow Diagram Illustrating Scoping Cohort Derivation ($89 \rightarrow 72 \rightarrow 53 \rightarrow 48 \rightarrow 46$). Cross-source deduplication eliminated 17 duplicates; title/abstract screening excluded 19 records; study-level deduplication consolidated 5 preprint/journal pairs; and full-text eligibility screening excluded 2 studies, leaving 46 eligible full-text studies.}
\label{fig:prisma_flow}
\end{figure}

Screening followed structured multi-phase filtering (Figure~\ref{fig:prisma_flow}): cross-source deduplication eliminated 17 duplicate records, leaving 72 unique entries. Title and abstract screening excluded 19 records (non-forecasting structural health monitoring, active pitch control, or purely data-driven studies). Study-level deduplication consolidated five preprint/journal pairs sharing identical authorship, datasets, and equations, reducing candidate entries from 53 to 48. Full-text evaluation against explicit eligibility criteria excluded two studies (S26 forecasted wind speed exclusively without a power constraint; S42 implemented heuristic feature clustering without physical equations), establishing our final representative scoping corpus of \textbf{$N=46$ unique scoping studies} (38 peer-reviewed journal articles and eight preprints). A supplementary sensitivity query expanding retrieval across Scopus and IEEE Xplore identified 14 additional papers, with full-text inspection confirming zero instances of documented operational failure boundaries.

\subsection{Static Software Audit Protocol}
Software availability was audited across full texts, footnotes, data availability statements, publisher repositories, and author archives. For identified public repositories, software inspection was conducted via Abstract Syntax Tree (AST) parsing and symbolic execution tracing against published manuscripts. PINN-UQ (S10, \textit{Eng. Appl. Artif. Intell.} 2025) was verified to faithfully execute its specified algebraic power equations ($P = g T \omega$ and $P = \frac{1}{2} C_p \rho A v^3$ via \texttt{sn.Tie}), earning classification as \textit{code-aligned}. The Physics-Constrained Transformer (PCT, S02, \textit{Sci. Rep.} 2026; commit \texttt{65941ec}) provided open PyTorch scripts (\texttt{train\_clean.py} and \texttt{tst/loss.py}) audited line-by-line for parity with published loss equations.

\subsection{Multi-Site Operational Wind Farm Testbed and SCADA Telemetry}
To ground the empirical audit in real-world power system operations across diverse terrains, we established an operational testbed across three commercial UK wind farms with 10-minute SCADA telemetry:
\begin{itemize}[leftmargin=*]
\item \textbf{Kelmarsh Wind Farm} (Northamptonshire, England): Located in flat open agricultural terrain, comprising six Senvion MM92 turbines (rated 2.05~MW each, hub height 78~m, farm capacity 12.3~MW), evaluated over $N_{\text{train}} = 5,098$ clean intervals as our primary development benchmark (8,212 test origins, 49,272 forward delivery steps across full-year 2017) \cite{zenodo_kelmarsh}.
\item \textbf{Hill of Towie Wind Farm} (Drummuir, Scotland): Located in complex upland terrain featuring micro-topographical wake interactions, comprising 21 Siemens SWT-2.3-82 turbines (rated 2.3~MW each, hub height 60~m, farm capacity 48.3~MW), evaluated as an independent spatiotemporal extrapolation benchmark across full-year 2017 with 7,599 forecast origins (45,594 delivery steps) \cite{zenodo_hilloftowie}.
\item \textbf{Penmanshiel Wind Farm} (Scottish Borders): Located in undulating upland terrain, comprising 14 Senvion MM82 turbines (rated 2.05~MW each, farm capacity 28.7~MW), configured with a restricted training partition ($N_{\text{train}} = 1,583$) across 8,440 test origins to evaluate model behavior under sample-restricted regimes \cite{zenodo_penmanshiel}.
\end{itemize}
Standard data sanitation protocols removed periods of turbine curtailment, anemometer icing, and communication dropouts ($P \le 0$ when $v > v_{\text{cut-in}}$ or active power frozen for $>6$ consecutive steps). Real-world balancing price dynamics were evaluated using official 2017 UK Elexon BSC System Buy Prices (mean \pounds 44.27/MWh, peak \pounds 1,509.80/MWh, with 68 negative price periods).

\subsection{Forecasting Pipeline, Controlled Ablation Designs, and Baselines}
The forecasting pipeline adopts an input lookback of $L=36$ (6 hours) and a forecast horizon of $H=36$ (6 hours), with downstream economic dispatch evaluated over the next-hour delivery window $H_{\text{disp}} = \{1, \dots, 6\}$ and an origination stride of $S=6$ steps to prevent overlapping settlement intervals. The core forecasting suite spans:
\begin{enumerate}[leftmargin=*]
\item \textit{Unconstrained DLinear}: Baseline linear mapping across trend and seasonal decomposition \cite{zeng2023dlinear}.
\item \textit{Bounds-Constrained DLinear} ($\lambda=0.01$): Penalizing outputs violating nameplate limits $[0, 1]$.
\item \textit{Empirical Power Curve Constrained DLinear} ($\lambda \in \{0.01, 0.1, 1.0\}$): Penalizing squared deviations from historical empirical power curves.
\item \textit{Conditional Distribution Prior DLinear} ($\lambda \in \{0.01, 0.1, 1.0\}$): Penalizing statistical divergence between predicted and historical joint speed--power distributions.
\item \textit{Pseudo-Scrambled Negative Controls}: To causally isolate domain aerodynamic guidance from non-specific numerical regularization, we construct form- and weight-matched pseudo-scrambled controls by randomly permuting the wind-to-power mapping along the wind speed axis while holding network architectures, loss formulations, and penalty weights $\lambda$ 100\% identical.
\item \textit{Strictly Budget-Matched Information Ablation} ($N_{\text{prior}}=500$): To decouple domain physical guidance from historical sample expansion, models trained on restricted windows ($N_{\text{train}}=500$) are evaluated against both budget-matched priors (fitted strictly on the identical 500 samples) and full-history priors (fitted on 7--10 months).
\item \textit{Architectural Sensitivity Probe (PatchTST)}: Multi-patch channel-independent Transformer (\cite{nie2023patchtst}, patch length 6, stride 4, 2-layer encoder with RevIN) evaluated across 60 independently trained models (3 farms $\times$ 4 constraints $\times$ 5 seeds) under nominal regularization ($\lambda=0.01$).
\item \textit{Classical Benchmarks}: Ridge regression and Persistence baselines.
\end{enumerate}

All models were evaluated across 3 to 5 independent random initialization seeds. Hyperparameters, architecture configurations, and post-hoc quantile calibration offsets $\delta^*$ were governed exclusively on the 2016 validation partition ($q = 1/(r+1) = 1/3$ for penalty ratio $r=2$). The 2017 test year was accessed only once model configurations were frozen, ensuring zero future information leakage. Statistical significance was evaluated via 2,000-replication circular block paired bootstrapping with 7-day block lengths ($L=1,008$ intervals) to preserve meteorological autocorrelation.
"""

# -------------------------------------------------------------
# 5. sections/05_audit.tex
# -------------------------------------------------------------
sec05 = r"""\section{Systematic Literature and Software Implementation Audit}
\label{sec:audit}

This section presents the results of the systematic literature scoping audit across the 46-study cohort, benchmarks multi-rater reliability, and details the static and empirical computational graph audit of the Physics-Constrained Transformer.

\subsection{Reporting Patterns and Code Availability in the Audited Scoping Cohort}
Across the core 46-study scoping cohort (2023--2026), zero studies reported operational failure regimes or negative trade-offs (0/46, 0.0\%; 95\% Wilson CI: [0.0\%, 7.7\%]; Table~\ref{tab:lifecycle_distribution}, Figure~\ref{fig:audit_distribution}). The supplementary 14-study Scopus/IEEE sensitivity query likewise identified zero negative trade-off reports, showing an identical reporting pattern in this external check. Furthermore, only three studies in the core cohort (6.5\%; S01, S02, S10) evaluated sensitivity to prior parameter perturbations; the remaining 43 studies (93.5\%) tested constraints exclusively under nominal design configurations. Restricting empirical evaluations to nominal conditions obscures operational failure boundaries and substitutes uniform improvement claims for engineering trade-offs.

Software availability inspection reveals an open-code deficit of 95.7\% (44/46, 95\% Wilson CI: [85.5\%, 98.8\%], $\mathcal{D}_4 = \texttt{not-found}$), with 44 studies providing neither executable code repositories nor reproducible scripts. Among the two studies with accessible code, PINN-UQ (S10) faithfully executes its published electromechanical tie equations ($\mathcal{D}_4 = \texttt{code-aligned}$), whereas the Physics-Constrained Transformer (PCT, S02) exhibits a directional implementation divergence in its divergence loss formulation ($\mathcal{D}_4 = \texttt{code-mismatch}$).

Ablation semantic audit ($\mathcal{D}_5$) reveals that half of the audited studies (23/46, 50.0\%, 95\% Wilson CI: [36.1\%, 63.9\%]) evaluate constraints by excising entire multi-layer neural submodules---such as graph convolutions, cross-attention blocks, or decomposition layers---rather than toggling the constraint penalty multiplier ($\lambda=0$ vs.\ $\lambda>0$; Table~\ref{tab:lifecycle_distribution}, Figure~\ref{fig:audit_distribution}d). Because excising architectural blocks removes thousands of trainable parameters simultaneously, reported performance gains reflect representational capacity rather than aerodynamic guidance.

\input{tables/table1_lifecycle_distribution.tex}

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{fig_field_audit_distribution.pdf}
\caption{Empirical Lifecycle Distributions across 46 Audited Studies ($N=46$). (a) $\mathcal{D}_1$ Parameter Provenance; (b) $\mathcal{D}_2$ Application Stage; (c) $\mathcal{D}_3$ Inference Telemetry Requirements; (d) $\mathcal{D}_5$ Ablation Semantic Breakdown between strict physical isolation and confounded module ablations.}
\label{fig:audit_distribution}
\end{figure*}

\subsection{Multi-Rater Reliability Benchmarking}
To assess the empirical stability of the taxonomy before full extraction, three independent raters (Coder A, Coder B, and Coder C) double-blindly coded a stratified representative subset of 20 candidate studies across all five dimensions. Study eligibility screening achieved unanimous consensus (Fleiss $\kappa = 1.000$). Parameter provenance ($\mathcal{D}_1$) demonstrated perfect agreement ($\kappa = 1.000$), while application stage ($\mathcal{D}_2$) yielded near-perfect agreement ($\kappa = 0.950$ overall, $\kappa = 0.938$ on included studies). Inference telemetry requirements ($\mathcal{D}_3$) achieved substantial agreement ($\kappa = 0.743$ overall, $\kappa = 0.646$ on included studies).

Implementation fidelity ($\mathcal{D}_4$) exhibited moderate concordance prior to consensus arbitration ($\kappa = 0.716$ overall in the stratified pilot, but $\kappa = 0.341$ when evaluated strictly on included studies). This reduction reflects the classical prevalence paradox described by Feinstein and Cicchetti: because 95.7\% of candidate studies omitted public code, expected chance agreement was high, which depresses Fleiss' $\kappa$ despite a raw observed agreement of 90.7\%. For ablation semantics ($\mathcal{D}_5$), raters achieved strong concordance ($\kappa = 0.851$ overall, $\kappa = 0.818$ on included studies). On harm reporting, all three raters unanimously coded zero instances of harm documentation across the included pilot studies ($0/18$); because marginal variance was zero, Cohen's $\kappa$ is mathematically indeterminate and is formally reported as 100\% observed agreement ($P_o = 1.000$).

\subsection{Software Implementation Fidelity: The Physics-Constrained Transformer}
Verifying physical constraint claims requires examining not only published equations, but also the computational fidelity of executed software. The Physics-Constrained Transformer (PCT \cite{pct2026}, \textit{Scientific Reports} 2026; commit \texttt{65941ec}) represents the sole publicly accessible training codebase for end-to-end physics-constrained Transformers and the sole audited repository exhibiting an implementation divergence between published text and code execution ($\mathcal{D}_4 = \texttt{code-mismatch}$).

In Section 2 of the PCT paper, the authors propose regularizing a Transformer forecaster by penalizing statistical divergence between the empirical joint distribution of historical wind speed and turbine power ($P_{\text{ref}}$) and the joint distribution predicted by the model ($P_{\text{pred}}$). The paper explicitly specifies the symmetric \textbf{Jensen--Shannon (JS) divergence}:
\begin{equation}
\mathcal{L}_{\text{JS}}(P_{\text{pred}} \parallel P_{\text{ref}}) = \frac{1}{2} \text{KL}\left(P_{\text{pred}} \,\Big\|\, \frac{P_{\text{pred}} + P_{\text{ref}}}{2}\right) + \frac{1}{2} \text{KL}\left(P_{\text{ref}} \,\Big\|\, \frac{P_{\text{pred}} + P_{\text{ref}}}{2}\right),
\label{eq:js_divergence}
\end{equation}
where $\text{KL}(p \parallel q) = \sum_i p_i \log \frac{p_i}{q_i}$. The authors state that symmetry and boundedness ($0 \le \mathcal{L}_{\text{JS}} \le \ln 2$) provide numerical stability.

However, static AST inspection of the official training script (\texttt{train\_clean.py}, lines 349--352; commit \texttt{65941ec}) traces the loss execution to:
\begin{center}
\small
\begin{minipage}{0.95\columnwidth}
\texttt{loss\_phy = 0.5 * (} \\
\hspace*{1em}\texttt{F.kl\_div(pred\_kde.log(), m\_kde, ...)} \\
\hspace*{1em}\texttt{+ F.kl\_div(true\_kde.log(), m\_kde, ...))}
\end{minipage}
\end{center}
In the PyTorch C++ ATen kernel backend (\texttt{native/Loss.cpp}, implementing \texttt{kl\_div\_backward\_cpu}), the functional specification for \texttt{torch.nn.functional.kl\_div(input, target)} evaluates:
\begin{equation}
\text{\texttt{kl\_div}}(\text{input}, \text{target}) = \sum_{i} \text{target}_i \cdot \big( \log \text{target}_i - \text{input}_i \big).
\label{eq:pytorch_kl}
\end{equation}
Passing \texttt{input} as $\log p$ and \texttt{target} as $q$ binds the tensor arguments such that the kernel computes:
\begin{equation}
\text{\texttt{kl\_div}}(\log p, q) = \sum_{i} q_i \cdot \big( \log q_i - \log p_i \big) = \text{KL}(q \parallel p).
\label{eq:reverse_kl}
\end{equation}
Consequently, the computational graph generated in \texttt{train\_clean.py} executes:
\begin{equation}
\mathcal{L}_{\text{code}} = \frac{1}{2} \text{KL}\left(M \parallel P_{\text{pred}}\right) + \frac{1}{2} \text{KL}\left(M \parallel P_{\text{ref}}\right), \quad \text{where } M = \frac{P_{\text{pred}} + P_{\text{ref}}}{2}.
\label{eq:reverse_js}
\end{equation}
Equation~\eqref{eq:reverse_js} establishes an altered objective, which we designate the \textbf{reverse-argument JS form}. Differentiating $\mathcal{L}_{\text{code}}$ directly with respect to predicted probability mass $P_{\text{pred}, i}$ yields:
\begin{equation}
\frac{\partial \mathcal{L}_{\text{code}}}{\partial P_{\text{pred}, i}} = \frac{1}{4} \left[ \log \left(\frac{M_i^2}{P_{\text{pred}, i} P_{\text{ref}, i}}\right) + 1 - \frac{P_{\text{ref}, i}}{P_{\text{pred}, i}} \right].
\label{eq:reverse_js_gradient}
\end{equation}
Equation~\eqref{eq:reverse_js_gradient} exposes an analytical \textbf{inverse-density singularity}: whenever the model assigns near-zero probability to a region where empirical training observations exist ($P_{\text{pred}, i} \to 0$ while $P_{\text{ref}, i} > 0$), the ratio $\frac{P_{\text{ref}, i}}{P_{\text{pred}, i}}$ diverges. Even under machine-precision clamping ($P_{\text{pred}, i} \ge \epsilon$), this term scales as $\mathcal{O}(1/\epsilon)$, generating steep negative gradient responses.

In contrast, for the standard mathematically specified Jensen--Shannon divergence $\mathcal{L}_{\text{JS}}$ in Equation~\eqref{eq:js_divergence}, differentiating with respect to $P_{\text{pred}, i}$ gives:
\begin{equation}
\frac{\partial \mathcal{L}_{\text{JS}}}{\partial P_{\text{pred}, i}} = \frac{1}{2} \log \left(\frac{P_{\text{pred}, i}}{M_i}\right).
\label{eq:standard_js_gradient}
\end{equation}
Because $M_i = \frac{1}{2}(P_{\text{pred}, i} + P_{\text{ref}, i}) \ge \frac{1}{2}P_{\text{pred}, i}$, the ratio $\frac{P_{\text{pred}, i}}{M_i} \in [0, 2]$, so the gradient of Standard-JS is strictly upper-bounded:
\begin{equation}
\frac{\partial \mathcal{L}_{\text{JS}}}{\partial P_{\text{pred}, i}} \le \frac{1}{2} \ln 2 \approx 0.3466.
\end{equation}
As predicted probabilities approach zero ($P_{\text{pred}, i} \to \epsilon$ with $P_{\text{ref}, i} > 0$), the gradient of Standard-JS exhibits only mild logarithmic divergence ($\mathcal{O}(|\log \epsilon|)$), substantially milder than the $\mathcal{O}(1/\epsilon)$ pole of the executed reverse-argument form.

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{fig_pct_loss_empirical_contrast.pdf}
\caption{Empirical Contrast between Official Reverse-JS and Corrected Standard-JS across Regularization Scales $\lambda \in [0.01, 1.0]$ on Kelmarsh and Penmanshiel (Multi-seed means and standard deviations). (a) Mean gradient $L_2$ norm: the reverse-argument JS form induces substantial gradient amplification at $\lambda = 1.0$ ($2.20\times$ on Kelmarsh, $2.72\times$ on Penmanshiel) due to the $-\frac{P_{\text{ref}, i}}{4 P_{\text{pred}, i}}$ inverse-density singularity ($\mathcal{O}(1/\epsilon)$ in Equation~\eqref{eq:reverse_js_gradient}), whereas for Corrected Standard-JS the signed partial derivative possesses a finite upper bound ($\frac{1}{2}\ln 2$) while exhibiting only mild logarithmic divergence ($\mathcal{O}(|\log \epsilon|)$). (b) Operational dispatch cost ($r=2$): Corrected Standard-JS prevents premature performance degradation at weak $\lambda=0.01$, while excessive regularization degrades dispatch across both variants.}
\label{fig:pct_contrast}
\end{figure*}

\subsection{Empirical Optimization Contrast: Official Reverse-JS vs.\ Corrected Standard-JS}
To determine whether this implementation divergence materially alters neural training, we trained models across Kelmarsh and Penmanshiel under five random seeds across $\lambda \in \{0.01, 0.1, 1.0\}$, benchmarked against unconstrained baselines (Figure~\ref{fig:pct_contrast}).

The reverse-argument JS form induces marked gradient amplification: at $\lambda = 1.0$, it produces an average gradient $L_2$ norm of $0.1665 \pm 0.0160$ on Kelmarsh and $0.2465 \pm 0.0217$ on Penmanshiel, representing a $2.20\times$ and $2.72\times$ amplification over Corrected Standard-JS ($0.0756 \pm 0.0048$ on Kelmarsh and $0.0907 \pm 0.0067$ on Penmanshiel; Figure~\ref{fig:pct_contrast}a). At weak regularization ($\lambda = 0.01$ on Penmanshiel), Corrected Standard-JS achieves a dispatch cost of $0.09804 \pm 0.00024$ (outperforming unconstrained $0.09864$), whereas the official reverse-argument implementation inflates dispatch cost to $0.09997 \pm 0.00086$ (a $+1.35\%$ degradation; Figure~\ref{fig:pct_contrast}b). At strong regularization ($\lambda=1.0$), both divergence variants degrade performance relative to unconstrained baselines ($0.10549$ vs $0.10292$). Correcting the implementation divergence mitigates gradient amplification and eliminates premature degradation under weak penalties, but does not prevent performance loss under excessive regularization.

\subsection{Negative Control: Algebraic Verification of 2D Product-Kernel KDE}
We establish an algebraic negative control to verify whether tensor outer-product operations preserve empirical feature coupling in the audited codebase. A potential concern regarding the tensor outer-product $G = K_p^\top K_w \in \mathbb{R}^{B_p \times B_w}$ is that it might assume statistical independence between wind speed and turbine power ($p(w, P) = p(w)p(P)$).

Formal algebraic expansion refutes this concern. Let $\{ (P_s, w_s) \}_{s=1}^N$ denote paired empirical observations in a batch. Let $\{b_i^p\}_{i=1}^{B_p}$ and $\{b_j^w\}_{j=1}^{B_w}$ denote 1D evaluation bin centers for power and wind speed, respectively. The 1D Gaussian kernel vectors evaluated in the codebase are:
\begin{equation}
\left[K_p\right]_{s,i} = \phi\left( \frac{b_i^p - P_s}{h_p} \right), \quad \left[K_w\right]_{s,j} = \phi\left( \frac{b_j^w - w_s}{h_w} \right),
\end{equation}
where $\phi(u) = \frac{1}{\sqrt{2\pi}} e^{-u^2/2}$ is the standard Gaussian kernel. Expanding the matrix product element at grid coordinate $(i,j)$ yields:
\begin{equation}
\left[ K_p^\top K_w \right]_{i,j} = \sum_{s=1}^N \left[K_p\right]_{s,i} \cdot \left[K_w\right]_{s,j} = \sum_{s=1}^N \phi\left( \frac{b_i^p - P_s}{h_p} \right) \cdot \phi\left( \frac{b_j^w - w_s}{h_w} \right).
\label{eq:kde_proof}
\end{equation}
Equation~\eqref{eq:kde_proof} is the exact analytical definition of a standard \textbf{paired 2D product-kernel Kernel Density Estimator} on correlated bivariate data. Because the summation contracts over concurrent sample index $s$, empirical joint coupling between $P_s$ and $w_s$ is preserved. This confirmation establishes that software discrepancies are localized to divergence argument order rather than upstream kernel construction, exemplifying an objective negative control.
"""

# -------------------------------------------------------------
# 6. sections/06_experiments.tex
# -------------------------------------------------------------
sec06 = r"""\section{Multi-Site Controlled Empirical Benchmark}
\label{sec:experiments}

Prospective controlled ablations across the three operational wind farms isolate prior specificity, numerical smoothing, information budgets, proxy staleness, and downstream settlement dynamics under matched model capacities.

\subsection{Defensive Inductive Bias and Sensitivity to Prior Misspecification}
On Penmanshiel under low-sample training ($N_{\text{train}} = 1,583$), unconstrained DLinear exhibits empirical over-fitting ($0.09881 \pm 0.00027$; Table~\ref{tab:controlled_ablation}, Figure~\ref{fig:causal_ablation}a). Embedding the authentic empirical power curve at $\lambda=1.0$ reduces normalized dispatch cost to $0.09760 \pm 0.00044$ ($-1.22\%$). In contrast, the form- and weight-matched Pseudo-Scrambled negative control inflates dispatch cost to $0.16611 \pm 0.00335$ ($+68.1\%$). This contrast demonstrates that in low-sample regimes, neural forecasters can benefit from valid aerodynamic guidance, but are acutely vulnerable to prior misspecification under strong regularization.

\input{tables/table2_dynamic_settlement.tex}

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{fig_causal_ablation.pdf}
\caption{Prospective 3D Controlled Ablation Benchmark across Three Operational Wind Farms. (a) High sensitivity to prior misspecification under strong constraints on Penmanshiel ($N_{\text{train}}=1,583$): authentic physical guidance provides downside risk protection ($-1.22\%$), whereas disrupting the physical mapping (Pseudo-Scrambled prior) triggers a $+68.1\%$ surge in operational dispatch cost. (b) Low-$\lambda$ numerical regularization mimicry on Kelmarsh ($N=5,098$), where genuine and scrambled priors yield near-identical decision costs at $\lambda = 0.01$, indicating non-specific numerical regularization rather than physical domain reasoning. (c) Monotonic cost inflation under training proxy staleness: dispatch cost escalation across all three farms as historical wind proxy lag $k$ in the training loss increases from 1h to 6h. (d) Decoupling of symmetric point error (RMSE) from asymmetric decision risk ($r=2$), demonstrating that point metrics mask tail operational hazards.}
\label{fig:causal_ablation}
\end{figure*}

\subsection{Within-Site Learning Curves and Information-Budget De-Aliasing}
To decouple sample volume from topography and disentangle physical inductive bias from auxiliary historical data budgets, we evaluated within-site learning curves across $N_{\text{train}} \in \{500, 1000, 1500, 3000, 5000\}$ on Kelmarsh and Hill of Towie across five random seeds (Figure~\ref{fig:within_site_learning_curve}), complemented by a controlled robustness ablation under strictly matched sample windows at $N_{\text{train}}=500$ ($\approx 21$ days of operational telemetry; Table~\ref{tab:information_budget}, all prior arms $\lambda=1.0$).

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{fig_within_site_learning_curve.pdf}
\caption{Within-Site Learning Curves across Sample Sizes $N_{\text{train}} \in \{500, 1000, 1500, 3000, 5000\}$ on Kelmarsh and Hill of Towie (Multi-seed means and standard deviations). Decoupling sample volume from topography confirms that fixed historical physical priors provide lower dispatch costs across both sites under restricted model training windows ($N_{\text{train}} \le 1,000$, $-2.00\%$ on Kelmarsh and $-3.62\%$ on Hill of Towie), whereas expanding model training windows to $N_{\text{train}} = 5,000$ closes this gap and slightly reverses the ranking (+0.33\% on Kelmarsh, +1.73\% on Hill of Towie). In contrast, misspecified scrambled priors under strong regularization ($\lambda=1.0$) induce severe dispatch cost penalties ($+43\%$ to $+98\%$) across all sample regimes.}
\label{fig:within_site_learning_curve}
\end{figure*}

In the baseline learning curve where the empirical prior is fitted on the full pre-cutoff historical record (7 months for Kelmarsh, 10 months for Hill of Towie), authentic priors reduce normalized dispatch cost from $0.09648 \pm 0.00314$ to $0.09455 \pm 0.00285$ ($-2.00\%$) on Kelmarsh, and from $0.10444 \pm 0.00312$ to $0.10066 \pm 0.00307$ ($-3.62\%$) on Hill of Towie. To address whether this empirical gain stems purely from physical inductive bias or from the extra historical data budget encoded in the prior, our controlled information-budget experiment fits the genuine prior strictly on the exact same 500 training samples ($N_{\text{prior}}=500$; Table~\ref{tab:information_budget}).

\input{tables/table_information_budget.tex}

Under this strictly budget-matched setting, the authentic power curve prior reduces dispatch cost to $0.09499 \pm 0.00307$ ($-1.54\%$) on Kelmarsh and $0.10095 \pm 0.00305$ ($-3.34\%$) on Hill of Towie. The budget-matched prior retains most of the mean improvement observed with the full-history prior (approximately $77\%$ on Kelmarsh and $92\%$ on Hill of Towie). Evaluated under paired differences across the five identical seeds, the budget-matched prior achieves a mean paired risk reduction of $-0.00149$ ($95\%$ CI: $[-0.00174, -0.00124]$) on Kelmarsh and $-0.00349$ ($95\%$ CI: $[-0.00368, -0.00331]$) on Hill of Towie relative to unconstrained baselines ($p < 0.01$). Expanding the prior to the full multi-month history yields a modest incremental gain of $-0.00044$ ($95\%$ CI: $[-0.00075, -0.00013]$) on Kelmarsh and $-0.00029$ ($95\%$ CI: $[-0.00041, -0.00016]$) on Hill of Towie. Conversely, scrambling the prior under the exact same 500-sample budget inflates dispatch cost to $0.13827 \pm 0.02333$ ($+43.3\%$) on Kelmarsh and $0.15248 \pm 0.00253$ ($+46.0\%$) on Hill of Towie, confirming that domain-structured empirical prior effects persist under restricted temporal data budgets.

As training volume expands to $N_{\text{train}} = 5{,}000$, unconstrained DLinear achieves $0.08812$ on Kelmarsh and $0.08936$ on Hill of Towie, closing the gap and slightly outperforming the fixed genuine prior ($0.08841$, $+0.33\%$ on Kelmarsh; $0.09091$, $+1.73\%$ on Hill of Towie). In contrast, scrambled priors remain severely detrimental ($+43\%$ to $+98\%$) across all sample sizes. These findings indicate that while authentic empirical priors provide downside protection in data-scarce windows, their relative advantage diminishes as training volume expands.

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{fig_prior_corruption_gradient.pdf}
\caption{Continuous Prior Corruption and Engineering Drift Severity Response Curves. (a) Continuous interpolation between authentic and scrambled priors $\alpha \in [0.0, 1.0]$, identifying an empirical crossover region around $\alpha \approx 0.07$ on Penmanshiel before physical regularizers worsen unconstrained dispatch costs. (b) Turbine aerodynamic aging: capacity derating exceeding 10\% flips nominal physical priors into economic penalties. (c) Anemometer sensor calibration drift: Penmanshiel retains an advantage under positive wind speed drift up to $+15\%$, whereas negative sensor under-reporting systematically inflates shortfall.}
\label{fig:corruption_gradient}
\end{figure*}

Evaluating continuous corruption levels $\alpha \in [0.0, 1.0]$ demonstrates that prior failure is not an artifact of total permutation (Figure~\ref{fig:corruption_gradient}a). On Penmanshiel, dispatch cost degrades monotonically with corruption, crossing the unconstrained baseline around $\alpha \approx 0.07$ (at $\alpha=0.2$, dispatch cost reaches $0.10469 \pm 0.00338$, a $+5.9\%$ degradation). Evaluating engineering drift reveals asymmetrical operational tolerances: aerodynamic aging exceeding $10\%$ rated capacity derating (Figure~\ref{fig:corruption_gradient}b) flips nominal physical priors into economic penalties. For anemometer calibration, Penmanshiel maintains a modest advantage over unconstrained models under positive sensor biases up to $+15\%$ ($0.09833$ vs.\ $0.09881$, Figure~\ref{fig:corruption_gradient}c), whereas negative wind speed under-reporting systematically inflates shortfall.

\subsection{Low-$\lambda$ Numerical Regularization Mimicry}
At weak regularization ($\lambda = 0.01$), genuine physical priors and pseudo-scrambled controls yield closely matched dispatch costs on Kelmarsh ($0.08810 \pm 0.00009$ vs.\ $0.08801 \pm 0.00008$; Table~\ref{tab:controlled_ablation}, Figure~\ref{fig:causal_ablation}b). Because random power curve permutations produce outcomes comparable to genuine aerodynamics at this weak penalty scale, marginal performance improvements reported under weak penalties cannot be specifically attributed to aerodynamic reasoning and remain consistent with non-specific numerical smoothing.

\subsection{Sensitivity to Training-Loss Meteorological Proxy Staleness}
Parametric scanning of historical wind speed proxy lag $k \in \{6, 12, 18, 36\}$ (evaluating 1\,h to 6\,h operational delay in the training-loss proxy query) produces strictly monotonic cost inflation across all three farms (Hill of Towie: from $0.09022$ at $k=6$ to $0.09257$ at $k=36$, crossing the unconstrained baseline $0.08924$ at $k \approx 12$; Figure~\ref{fig:causal_ablation}c). We emphasize that this experiment isolates the sensitivity of the training objective to temporal misalignment between the forecast horizon and the historical wind speed proxy used to index the power curve. It does not represent an inference-time telemetry delay, as the forecaster takes only past power measurements during test-time inference. However, it demonstrates that if an offline training loss regularizes predictions against lagged weather observations without forward meteorological forecasts, proxy staleness degrades the resulting inductive guidance.

\subsection{Dynamic Market Settlement and the Zero-Bid Settlement Anomaly}
Under real-world dynamic pricing across full-year 2017 Elexon BSC records (Table~\ref{tab:paired_bootstrap}), calibrated distribution constraints exhibit a pronounced double-edged divergence: on Kelmarsh, distribution constraints increase annual shortfall expenditure by $+£932.56$/(turb$\cdot$yr) (95\% CI: $[+£648.20, +£1,236.74]$) and incur net balancing cashflow loss of $-£3,264.86$/(turb$\cdot$yr). Conversely, on Hill of Towie, distribution constraints reduce shortfall by $-£578.57$/(turb$\cdot$yr) and increase net cashflow by $+£1,481.27$/(turb$\cdot$yr) (95\% CI: $[+£802.29, +£2,203.32]$).

\input{tables/table3_paired_bootstrap.tex}

Evaluating forecasters on isolated unhedged balancing market cashflows introduces a structural evaluation pitfall: under half-hour aggregation of modelled imbalance exposures with 2017 UK BSC cash-out prices, a degenerate Zero-Bid baseline ($q \equiv 0$) extracts \pounds 223k to \pounds 243k per turbine-year in apparent cashflow without generating forecasts (Table~\ref{tab:zero_bid_diagnostic}, Figure~\ref{fig:market_decoupling}a).

\input{tables/table4_zero_bid_diagnostic.tex}

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{fig_market_decoupling.pdf}
\caption{Market Settlement Decoupling and Metric Inversion Diagnostic under Half-Hour Imbalance Accounting. (a) Reductio ad absurdum: the degenerate zero-bid settlement anomaly ($q \equiv 0$) incurs zero shortfall penalty and extracts over \pounds 220k/turb$\cdot$yr in apparent cashflow without generating forecasts, exposing unhedged imbalance cashflow as an evaluation trap. (b) Empirical metric inversion on Hill of Towie, where distribution constraints increase nominal balancing cashflow by $+42.5\%$ while concurrently degrading genuine decision risk by $+6.5\%$ and RMSE by $+3.5\%$.}
\label{fig:market_decoupling}
\end{figure*}

In stylized wholesale accounting coupling forward commitments with real-time imbalance settlement ($R_{\text{total}} = P_{\text{F}} \cdot q + P_{\text{imb}} \cdot (y - q)$), isolating balancing cashflows $R_{\text{imb}} = P_{\text{imb}} \cdot (y - q)$ from forward market revenue converts total generation $y$ into uncontracted spill credited at $P_{\text{imb}}$, producing an apparent six-figure revenue while forfeiting forward market receipts. The Zero-Bid baseline acts as an empirical \textit{reductio ad absurdum}: standalone unhedged balancing cashflow is structurally degenerate when disconnected from forward commitments, confirming that unhedged imbalance accounting cannot serve as an index of forecast quality. Under asymmetric operational loss ($r=2, q=1/3$), Zero-Bid yields a high Decision Cost of $0.33689$ on Penmanshiel (Table~\ref{tab:zero_bid_diagnostic}), correctly penalizing non-delivery ($\text{Cost}_{\text{static}} = \bar{y} \approx 0.337$).

Furthermore, on Hill of Towie, distribution regularization increases nominal balancing cashflow by $+42.5\%$ (\pounds 8,543 to \pounds 12,174/(turb$\cdot$yr), Table~\ref{tab:zero_bid_diagnostic}) while concurrently degrading normalized decision risk ($\text{Cost}_{\text{static}}$) by $+6.5\%$ and RMSE by $+3.5\%$ (Figure~\ref{fig:market_decoupling}b). This divergence demonstrates that apparent monetary gains originate from negative bias coinciding with positive price spikes rather than physical inductive advantage, confirming that operational evaluation requires decoupling non-negative asymmetric decision risk from historical monetary settlement diagnostics.

\subsection{Cross-Backbone Architectural Sensitivity Probe (PatchTST)}
To assess whether empirical boundary behaviors extend beyond linear representations, we evaluated PatchTST \cite{nie2023patchtst} across 60 independently trained models (3 farms $\times$ 4 constraint conditions $\times$ 5 seeds; Table~\ref{tab:patchtst_and_rules}). This setup functions strictly as an \textit{architectural sensitivity probe} under fixed nominal regularization ($\lambda=0.01$).

\input{tables/table5_patchtst_and_rules.tex}

As detailed in Table~\ref{tab:patchtst_and_rules}, PatchTST confirms that distribution priors reduce dynamic shortfall across all three wind farms (saving \pounds 232.44, \pounds 127.49, and \pounds 790.51/(turb$\cdot$yr) on Hill of Towie, Kelmarsh, and Penmanshiel, respectively), indicating that defensive shortfall reduction persists in non-linear patch representations. However, net balancing cashflow diverges sharply: distribution regularization decreases net cashflow on Hill of Towie by $-\text{\pounds} 568.68$/(turb$\cdot$yr) under PatchTST, inverting the positive cashflow effect observed under DLinear. On Penmanshiel, simple empirical baselines (Ridge regression at $0.09744$ and Persistence at $0.09586$) outperform all constrained deep networks, demonstrating that external physical penalties cannot uniformly compensate for deep parameter overfitting in low-sample regimes.
"""

# -------------------------------------------------------------
# 7. sections/07_discussion.tex
# -------------------------------------------------------------
sec07 = r"""\section{Discussion, Identification Boundaries, and Reporting Governance}
\label{sec:discussion}

\subsection{Four Empirical Boundary Conditions for Physics-Constrained Models}
The scoping audit, algebraic code examination, and prospective multi-site ablations converge on four boundary conditions governing physical constraints:

\paragraph{Sample Restriction and Prior Misspecification Vulnerability}
On Penmanshiel ($N_{\text{train}}=1,583$), the calibrated power curve prior at $\lambda=1.0$ reduces asymmetric dispatch cost from $0.09881 \pm 0.00027$ to $0.09760 \pm 0.00044$ ($-1.22\%$). A form- and weight-matched scrambled pseudo-prior raises dispatch cost to $0.16611 \pm 0.00335$ ($+68.1\%$). The divergence between these outcomes---authentic priors securing modest downside protection whereas corrupted priors inflict severe damage---demonstrates a fundamental risk-return asymmetry: under strong regularization, whatever information the prior encodes is amplified, including its misspecifications. In the data-rich regime (Kelmarsh, $N_{\text{train}}=5,098$), unconstrained DLinear ($0.08812$), genuine physics ($0.08791$), and scrambled prior ($0.08801$) are numerically clustered, indicating hypothesis-space saturation.

\paragraph{Low-Penalty Scale and Non-Specific Numerical Regularization}
At $\lambda = 0.01$ on Kelmarsh, a genuine physical prior ($0.08810 \pm 0.00009$) and a structurally scrambled pseudo-prior ($0.08801 \pm 0.00008$) yield closely matched decision costs with overlapping seed margins. When random power curve permutations reproduce the same outcome as domain aerodynamics, the penalty term behaves empirically as a non-specific numerical regularizer rather than encoding domain physical laws. Published claims of domain physical benefit under $\lambda \in [10^{-4}, 10^{-2}]$ require negative-control ablations and sensitivity testing at $\lambda \ge 0.1$ to be credible.

\paragraph{Monotonic Cost Inflation under Training Proxy Observation Staleness}
When physical constraint formulations rely on meteorological proxies to query empirical power curves during loss computation, substituting lagged historical observations for contemporaneous forecasts monotonically inflates operational dispatch risk across all three wind farms ($k \in \{6, 12, 18, 36\}$ steps, 1\,h to 6\,h). On Hill of Towie, dispatch risk climbs from $0.09022$ at $k=6$ to $0.09257$ at $k=36$, exceeding the unconstrained baseline ($0.08924$) at $k \approx 12$. Regularization against lagged weather proxies rapidly erodes inductive guidance, presenting an operational limitation for proxy-dependent constraint designs.

\paragraph{Metric Decoupling and the Zero-Bid Diagnostic}
In wholesale power accounting coupling forward commitments with balancing cash-out settlement ($R_{\text{total}} = P_{\text{F}} \cdot q + P_{\text{imb}} \cdot (y - q)$), evaluations that drop forward contract commitments and report only balancing cashflow are measuring imbalance rent, not forecast skill. As an analytical imbalance-settlement diagnostic, the degenerate Zero-Bid baseline ($q \equiv 0$) provides a mathematical \textit{reductio ad absurdum}: extracting £223k--£243k per turbine-year across all three farms without generating forecasts. Standalone unhedged balancing cashflows are decoupled from forecasting skill, and operational evaluations must report asymmetric Pinball loss at the market penalty ratio or coupled forward-plus-imbalance cashflows.

\subsection{Attributional Identification: What Is Identified vs.\ What Is Not}
\label{sec:identification}
To maintain rigorous scientific standards, we explicitly articulate the attributional identification boundaries of this audit:

\paragraph{What Is Formally Identified}
\begin{enumerate}[leftmargin=*, label=(\alph*)]
\item \textbf{Data-Starvation and Information-Budget Dynamics}: In subsampled training sequences holding site topography and turbine hardware constant (Figure~\ref{fig:within_site_learning_curve}, Table~\ref{tab:information_budget}), an advantage is observed under restricted model-training windows ($N_{\text{train}} \le 1{,}000$). By evaluating a controlled ablation under matched sample windows at $N_{\text{train}}=500$, we observe that domain-structured empirical prior effects under a matched temporal data window retain 77\% to 92\% of the mean downside risk reduction observed with the full-history prior, while the seasonal historical pool supplies an incremental gain ($-0.46\%$ and $-0.28\%$).
\item \textbf{Physical Specificity vs.\ Non-Specific Smoothing}: By comparing genuine power curves against form- and weight-matched pseudo-scrambled controls under identical network architectures, optimization schedules, and loss parameters, the comparison supports domain-specific mapping effects beyond non-specific regularization.
\item \textbf{Loss Sensitivity to Historical Proxy Latency}: By scanning proxy staleness $k \in \{6, \dots, 36\}$ steps within the training loss, we isolate the mathematical degradation induced by lagged meteorological inputs when forward weather forecasts are absent.
\item \textbf{Software Discrepancy in Divergence Optimization}: Static AST analysis and gradient derivation isolate an exact algebraic inversion in the PCT code repository, establishing a concrete implementation divergence between published formulas and software execution.
\item \textbf{Metric Decoupling of Imbalance Cashflows}: The degenerate Zero-Bid baseline mathematically separates standalone unhedged balancing cashflows from true predictive accuracy.
\end{enumerate}

\paragraph{What Is Not Identified (Confounding Disclaimers)}
\begin{enumerate}[leftmargin=*, label=(\roman*)]
\item \textbf{Cross-Site Topographical Causation}: Differences across Kelmarsh, Hill of Towie, and Penmanshiel confound topographic complexity with turbine ratings (2.05\,MW vs.\ 2.30\,MW), hub heights, wake geometry, and local micro-climates. We do not claim that complex terrain causally mandates physical constraints; this remains a testable mechanistic hypothesis.
\item \textbf{Global Generalization Beyond Scoped Cohort}: The 95.7\% code deficit and 50.0\% confounded ablation rates characterize the audited 46-study scoping cohort (2023--2026) under PRISMA-ScR protocols, not the entire machine learning or power engineering literature.
\item \textbf{Software Bugs as Universal Practice}: Auditing PCT and PINN-UQ demonstrates that code-manuscript discrepancies can occur; it does not imply that all published physics-informed energy forecasting codebases contain errors.
\item \textbf{Decoupled Calibration Universality}: Counterexamples 1 and 2 are formal existence proofs establishing the theoretical limits of scalar quantile offsets under spatial heteroscedasticity, not an empirical assertion that scalar calibration always underperforms joint asymmetric training.
\end{enumerate}

\subsection{Two Conjectures Falsified During the Audit}
Two initial hypotheses were refuted during the empirical audit. First, forensic tensor derivation showed that PCT's product-kernel KDE preserves bivariate coupling algebraically, refuting the conjecture of marginal independence. Second, market diagnostic probing showed that post-calibration cashflow inversions are governed by settlement price asymmetries rather than physical regularizers. Documenting these falsification outcomes provides a transparent account of our investigative trajectory and prevents post-hoc rationalization.

\subsection{REFORMS-PCWPF Compliance Protocol}
\label{sec:reforms_pcwpf}
Table~\ref{tab:reforms_checklist} proposes five verification gates as a methodological reporting checklist for physics-constrained wind forecasting studies:

\input{tables/table_reforms_checklist.tex}

\begin{itemize}[leftmargin=*]
\item \textbf{Gate M1 (Parameter-Isolated Ablations):} Parameter-matched comparisons ($\lambda=0$ vs.\ $\lambda>0$) should be required to isolate physical prior contributions, avoiding submodule excisions (e.g., stripping GNN layers) that conflate representational capacity with domain knowledge.
\item \textbf{Gate M2 (Pseudo-Scrambled Negative Controls):} A minimum credible evaluation of physical reasoning claims should include form- and weight-matched permutation controls to separate aerodynamic guidance from non-specific numerical smoothing.
\item \textbf{Gate M3 (Operational Proxy Staleness Stress-Tests):} Multi-step constraint evaluations should avoid assuming zero-latency future weather inputs, reporting performance across $k \in \{6, \dots, 36\}$ steps when forward NWP feeds are absent.
\item \textbf{Gate M4 (Coupled Forward and Imbalance Evaluation):} Evaluations should avoid standalone unhedged balancing cashflows, reporting instead asymmetric Pinball loss at the market penalty ratio or coupled forward-plus-imbalance P\&L ($P_{\text{F}} \cdot q + P_{\text{imb}} \cdot (y - q)$).
\item \textbf{Gate M5 (Static Computational Graph Audit):} Reviewers should expect executable training scripts or verified computational graphs, verifying that software loss execution matches published definitions in argument ordering and reduction semantics.
\end{itemize}
"""

# -------------------------------------------------------------
# 8. sections/08_conclusion.tex
# -------------------------------------------------------------
sec08 = r"""\section{Conclusion}
\label{sec:conclusion}

Three findings from this empirical audit challenge the prevailing consensus on physics-constrained wind power forecasting.

Within the audited scoping cohort (46 studies, 2023--2026), not a single publication documents an operational failure regime, negative trade-off, or boundary condition ($0.0\%$). With 95.7\% of the cohort lacking public code and 50.0\% conflating neural module removal with physical ablation, this uniform positive consensus reflects widespread reporting conventions that fail to stress-test models outside nominal design envelopes.

At the software implementation level, of the two publicly accessible software repositories identified across the 46 studies, PINN-UQ was verified as code-aligned, while PCT executes its Jensen--Shannon divergence loss with reversed tensor arguments that induce an analytical inverse-density gradient singularity ($\mathcal{O}(1/\epsilon)$), an implementation divergence undetectable from published text alone. Algebraic derivation separately confirms that the paired kernel product remains mathematically sound, establishing that implementation fidelity and formulation validity reside in distinct computational layers.

At the empirical evaluation level, physical constraints exhibit a fundamentally double-edged nature. At $N_{\text{train}} = 1{,}583$ and $\lambda = 1.0$, an authentic power curve prior reduces normalized dispatch cost by 1.22\%; a form- and weight-matched scrambled prior raises it by 68.1\%, demonstrating acute sensitivity to prior misspecification under strong regularization. Weak constraints ($\lambda = 0.01$) yield outcomes comparable to structurally scrambled negative controls, behaving empirically like non-specific numerical smoothing. Under dynamic UK market settlement, physical constraints reduce dispatch costs in complex upland terrain while increasing annual shortfall penalties by £932.56/(turb$\cdot$yr) in flat terrain. Furthermore, the degenerate Zero-Bid anomaly demonstrates that standalone unhedged balancing cashflows decouple from predictive capability under wholesale settlement.

In our tested configurations, empirical power-curve priors provide measurable inductive value under restricted training windows, but their operational benefits require rigorous empirical boundaries. Establishing those boundaries demands parameter-isolated ablations, scrambled negative controls, static computational graph verification, and market evaluation protocols that separate forward commitment revenue from imbalance settlement cashflow.
"""

# -------------------------------------------------------------
# 9. sections/09_appendix.tex
# -------------------------------------------------------------
sec09 = r"""\section{Mathematical Proofs of Boundary Counterexamples}
\label{sec:appendix_proofs}

This appendix provides formal mathematical proofs for the two boundary counterexamples presented in Section~\ref{sec:theory}, delineating the theoretical limits of decoupled decision calibration.

\subsection{Proof of Counterexample 1: Constant Shift Invariance and Zero Excess Risk}
\label{subsec:counterexample1}

\begin{example}[Constant Shift Invariance]
\label{ex:constant_shift}
Let the target variable follow a uniform distribution $Y \sim U[0.25, 0.75]$, and consider an asymmetric cost ratio $r = 2$ ($\tau = 1/3$). Suppose an unconstrained model outputs the exact conditional expectation $f_0 = \mathbb{E}[Y] = 0.50$, while a physics-constrained model incurs a constant positive shrinkage shift $c = +0.05$, outputting $f_\lambda = 0.55$.
\end{example}

\paragraph{Prediction Representation Error (MSE)}
The mean squared prediction errors are:
\begin{align}
\text{MSE}(f_0) &= \text{Var}(Y) = \frac{0.50^2}{12} \approx 0.020833, \\
\text{MSE}(f_\lambda) &= \text{MSE}(f_0) + c^2 \approx 0.023333.
\end{align}
The physical constraint strictly inflates prediction MSE by $\Delta \text{MSE} = +0.0025$ ($+12.0\%$).

\paragraph{Uncalibrated Operational Dispatch Loss}
Under uncalibrated dispatch ($p = f$), the expected asymmetric loss is:
\begin{equation}
\mathcal{R}_{\text{uncal}}(p) = 2 \int_{0.25}^p (p - y) \cdot 2 \, dy + \int_p^{0.75} (y - p) \cdot 2 \, dy = 2(p - 0.25)^2 + (0.75 - p)^2.
\end{equation}
Evaluating at each prediction:
\begin{align}
\mathcal{R}_{\text{uncal}}(f_0 = 0.50) &= 2(0.25)^2 + (0.25)^2 = 0.187500, \\
\mathcal{R}_{\text{uncal}}(f_\lambda = 0.55) &= 2(0.30)^2 + (0.20)^2 = 0.220000.
\end{align}

\paragraph{Post-Hoc Decoupled Quantile Calibration}
The theoretical Bayes-optimal quantile at $\tau = 1/3$ is:
\begin{equation}
q_{1/3} = 0.25 + \tfrac{1}{3}(0.75 - 0.25) \approx 0.416667.
\end{equation}
Under decoupled quantile calibration, each model searches for its optimal scalar displacement:
\begin{align}
\delta_0^* &= q_{1/3} - f_0 \approx 0.416667 - 0.500000 = -0.083333, \\
\delta_\lambda^* &= q_{1/3} - f_\lambda \approx 0.416667 - 0.550000 = -0.133333.
\end{align}
Notice that the resulting dispatch commitments are identical:
\begin{equation}
p_0^* = f_0 + \delta_0^* = 0.416667, \quad p_\lambda^* = f_\lambda + \delta_\lambda^* = 0.416667.
\end{equation}
Consequently, the post-calibration operational risks are identical:
\begin{equation}
\mathcal{R}_{\text{cal}}(p_0^*) = \mathcal{R}_{\text{cal}}(p_\lambda^*) = 0.166667 \implies \Delta \mathcal{R}_{\text{disp}} \equiv 0.
\end{equation}

\paragraph{Theoretical Takeaway}
Counterexample 1 proves that \textbf{an increase in prediction MSE is compatible with invariant post-calibration operational risk}. When constraint-induced representation error consists of a uniform shift, scalar calibration absorbs the distortion entirely, establishing that strict positive excess risk lower bounds cannot hold unconditionally.

\subsection{Proof of Counterexample 2: Spatial Heteroscedasticity and Risk Reversal}
\label{subsec:counterexample2}

\begin{example}[Spatial Heteroscedasticity]
\label{ex:heteroscedasticity}
Let $X \in \{0, 1\}$ be a binary indicator with $P(X=0) = P(X=1) = 0.5$. Let the conditional target distributions exhibit heteroscedastic spread:
\begin{align}
Y \mid (X=0) &\sim U[0.20, 0.60] \quad (\mu_0 = 0.40, s_0 = 0.40), \\
Y \mid (X=1) &\sim U[0.30, 0.90] \quad (\mu_1 = 0.60, s_1 = 0.60).
\end{align}
Let the asymmetric cost ratio be $r = 2$ ($\tau = 1/3$). Consider two candidate forecasters: Model A (the unbiased conditional mean) with predictions $f_A(0) = 0.40$ and $f_A(1) = 0.60$; and Model B (a biased estimator with higher MSE) with predictions $f_B(0) = 0.40$ and $f_B(1) = 0.55$.
\end{example}

\paragraph{Prediction MSE Comparison}
The mean squared prediction errors are:
\begin{align}
\text{MSE}(f_A) &= 0.5 \left(\frac{0.40^2}{12}\right) + 0.5 \left(\frac{0.60^2}{12}\right) = \frac{13}{600} \approx 0.021667, \\
\text{MSE}(f_B) &= \text{MSE}(f_A) + 0.5 (0.55 - 0.60)^2 = 0.021667 + 0.001250 \approx 0.022917.
\end{align}
Model B has strictly higher prediction MSE by $\Delta \text{MSE} = +0.001250$ ($+5.8\%$).

\paragraph{Post-Hoc Global Scalar Calibration}
The local Bayes-optimal quantiles at $\tau = 1/3$ are:
\begin{align}
q_{1/3}(0) &= 0.20 + \tfrac{0.40}{3} = \tfrac{1}{3} \approx 0.333333, \\
q_{1/3}(1) &= 0.30 + \tfrac{0.60}{3} = 0.500000.
\end{align}
The Bayes-optimal operational dispatch loss is $\mathcal{R}(q_\tau) = 0.5(\frac{0.40}{3}) + 0.5(\frac{0.60}{3}) = \frac{1}{6} \approx 0.166667$. The local conditional densities are $f_0(y) = 2.5$ and $f_1(y) = 1.6667$.

Because local quantile-to-mean shifts differ ($\Delta_0 = q_{1/3}(0) - \mu_0 = -\frac{1}{15} \approx -0.066667$ versus $\Delta_1 = q_{1/3}(1) - \mu_1 = -0.100000$), a single global scalar offset $\delta^*$ cannot simultaneously achieve both local quantiles. The expected operational risk under global shift $\delta$ is:
\begin{equation}
\mathcal{R}(f + \delta) = 0.5 \, \mathcal{R}_0(f(0) + \delta) + 0.5 \, \mathcal{R}_1(f(1) + \delta).
\end{equation}

For \textbf{Model A} ($f_A = \mu$), the first-order optimality condition for $\delta_A^*$ is:
\begin{equation}
(r+1) \left[ 0.5 \cdot 2.5 (\delta_A^* - \Delta_0) + 0.5 \cdot 1.6667 (\delta_A^* - \Delta_1) \right] = 0,
\end{equation}
yielding the density-weighted optimal global offset:
\begin{equation}
\delta_A^* = \frac{2.5 (-\frac{1}{15}) + 1.6667 (-0.10)}{2.5 + 1.6667} = -0.080000.
\end{equation}
The excess operational risk for Model A is:
\begin{equation}
\mathcal{R}(f_A + \delta_A^*) - \mathcal{R}(q_\tau) = \frac{3}{4} \cdot 2.5 (-0.0800 + 0.06667)^2 + \frac{3}{4} \cdot 1.6667 (-0.0800 + 0.1000)^2 = 0.000833.
\end{equation}

For \textbf{Model B} ($f_B(0)=0.40, f_B(1)=0.55$), the quantile shifts relative to Model B predictions are:
\begin{equation}
\Delta_{B, 0} = q_{1/3}(0) - f_B(0) = -0.066667, \quad \Delta_{B, 1} = q_{1/3}(1) - f_B(1) = -0.050000.
\end{equation}
Because Model B's negative bias at $X=1$ brings its prediction closer to $q_{1/3}(1)$, the spread between local shifts shrinks from $0.0333$ to $0.0167$. The optimal global offset for Model B is:
\begin{equation}
\delta_B^* = \frac{2.5 (-0.066667) + 1.6667 (-0.050000)}{2.5 + 1.6667} = -0.060000.
\end{equation}
Evaluating the excess operational risk for Model B:
\begin{equation}
\mathcal{R}(f_B + \delta_B^*) - \mathcal{R}(q_\tau) = \frac{3}{4} \cdot 2.5 (-0.0600 + 0.06667)^2 + \frac{3}{4} \cdot 1.6667 (-0.0600 + 0.0500)^2 = 0.000208.
\end{equation}

\paragraph{Theoretical Takeaway}
Comparing excess operational risks reveals:
\begin{equation}
\mathcal{R}(f_B + \delta_B^*) - \mathcal{R}(q_\tau) = 0.000208 < 0.000833 = \mathcal{R}(f_A + \delta_A^*) - \mathcal{R}(q_\tau).
\end{equation}
Model B achieves a \textbf{75\% reduction in excess operational risk} despite having $+5.8\%$ higher prediction MSE. Counterexample 2 proves that under spatial heteroscedasticity, an estimator with higher MSE can achieve strictly lower post-calibration dispatch risk than the true conditional expectation.
"""

# -------------------------------------------------------------
# 10. main.tex
# -------------------------------------------------------------
main_content = r"""\documentclass[3p,twocolumn,times]{elsarticle}

\usepackage[T1]{fontenc}
\usepackage{textcomp}
\usepackage{lineno,hyperref}
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{siunitx}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{threeparttable}
\usepackage{array}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{tabularx}
\usepackage{ragged2e}
\usepackage{url}

\journal{Applied Energy}
\bibliographystyle{elsarticle-num}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue,
    pdfencoding=auto,
    psdextra
}
\pdfstringdefDisableCommands{\def\corref#1{}}

% Theorem environments
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}{Corollary}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{hypothesis}{Hypothesis}
\theoremstyle{remark}
\newtheorem{remark}{Remark}
\newtheorem{example}{Counterexample}

\usepackage{dblfloatfix}

% Float placement optimization
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.70}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\dblfloatpagefraction}{0.70}
\setcounter{topnumber}{5}
\setcounter{bottomnumber}{5}
\setcounter{totalnumber}{10}
\setcounter{dbltopnumber}{5}

\graphicspath{{figures/}}

\begin{document}

\begin{frontmatter}

\title{Auditing the Decision Value of Physics-Constrained Wind Power Forecasting}

\author[1]{Junyu Li\corref{cor1}}
\ead{ljylikezmn999@gmail.com}
\author[1]{Juntao Du}

\address[1]{School of Statistics and Applied Mathematics, Anhui University of Finance and Economics, Bengbu 233030, China}

\cortext[cor1]{Corresponding author}

\begin{highlights}
\item Scoping audit of 46 physics-constrained wind forecasting studies reveals 95.7\% open-code deficit.
\item No operational failure regimes or adverse trade-offs were reported across the audited cohort.
\item Static code inspection of PCT identifies inverted divergence arguments altering gradient scaling.
\item Form-matched negative controls distinguish domain aerodynamic guidance from numerical smoothing.
\item Degenerate zero-bid diagnostic demonstrates standalone balancing cashflows decouple from forecast skill.
\end{highlights}

\begin{abstract}
Reductions in point forecasting error from physical constraints do not establish that empirical gains stem from valid physical domain knowledge, nor do they guarantee operational decision value in electricity balancing markets. In renewable energy forecasting, physical constraints are widely credited with improving accuracy and physical realism, but prevailing evaluation protocols rarely isolate domain physical guidance from parametric capacity, audit software execution fidelity, or evaluate asymmetric operational costs. Here, we conduct an end-to-end empirical audit of physics-constrained wind power forecasting spanning literature scoping ($N=46$), computational graph inspection, and prospective controlled ablations across three commercial UK wind farms. Within the scoped cohort (2023--2026), 0.0\% of studies document operational failure boundaries, 95.7\% lack public source code, and 50.0\% evaluate constraints via confounded module excisions. Software inspection of the open-source PCT benchmark uncovers an inverted argument order in its divergence loss that induces an analytical $\mathcal{O}(1/\epsilon)$ gradient singularity, though its bivariate kernel coupling remains mathematically sound. In prospective multi-site ablations holding the neural backbone and post-hoc quantile calibration constant, strong empirical power curve constraints ($\lambda=1.0$) reduce normalized asymmetric decision loss by 1.22\% under sample restriction (Penmanshiel), whereas form- and weight-matched scrambled priors inflate loss by 68.1\%, demonstrating prior specificity under data scarcity. Conversely, under weak penalties ($\lambda=0.01$), genuine and scrambled priors yield indistinguishable performance, behaving empirically like non-specific numerical smoothing. Across within-site learning curves, prior advantages diminish as training volume expands to $N=5{,}000$, while offline historical proxy staleness ($k > 12$) monotonically degrades downstream dispatch. Under real-world UK balancing settlement, physical constraints exhibit a double-edged nature: reducing dispatch costs in complex upland terrain while increasing annual shortfall penalties by £932.56/(turb$\cdot$yr) in flat terrain. Finally, degenerate zero-bid probing demonstrates that standalone unhedged balancing cashflows decouple from forecast skill, confirming that verifiable physical forecasting claims require backbone-matched negative controls, disclosed data budgets, and decision-aligned evaluation.
\end{abstract}

\begin{keyword}
Physics-Constrained Deep Learning \sep Wind Power Forecasting \sep Prediction-Decision Decoupling \sep Conditional Inductive Priors \sep Systematic Audit \sep Implementation Fidelity \sep Machine Learning Reproducibility
\end{keyword}

\end{frontmatter}

\input{sections/01_intro.tex}
\input{sections/02_related.tex}
\input{sections/03_theory.tex}
\input{sections/04_methodology.tex}
\input{sections/05_audit.tex}
\input{sections/06_experiments.tex}
\input{sections/07_discussion.tex}
\input{sections/08_conclusion.tex}

\appendix
\input{sections/09_appendix.tex}

\section*{Declaration of Interests}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Author Contributions (CRediT)}
\textbf{Junyu Li}: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Visualization, Project administration, Writing -- original draft, Writing -- review \& editing. \textbf{Juntao Du}: Validation, Data curation, Software (Inter-rater reliability arbitration), Investigation, Writing -- review \& editing.

\section*{Declaration of Generative AI in the Writing Process}
During the preparation of this work, the authors used ChatGPT and Google Gemini to assist with literature synthesis, manuscript organization, language refinement, code review, and consistency checking. All scientific claims, calculations, references, experimental results, and final wording were independently reviewed and verified by the authors, who take full responsibility for the content.

\section*{Acknowledgments}
We thank the open science teams at Zenodo, Cubico Sustainable Investments, RES, TRIG, and Elexon for maintaining transparent public repositories of operational renewable energy telemetry and market clearing records. Computations were supported by dedicated high-performance computing infrastructure.

\section*{Data and Code Availability}
Raw operational SCADA telemetry from the Kelmarsh (DOIs: \href{https://doi.org/10.5281/zenodo.5841834}{10.5281/zenodo.5841834}) and Penmanshiel (\href{https://doi.org/10.5281/zenodo.5946808}{10.5281/zenodo.5946808}) wind farms is openly available on Zenodo under CC-BY 4.0 licenses from Cubico Sustainable Investments. Operational SCADA telemetry from the Hill of Towie wind farm is openly available on Zenodo under CC-BY 4.0 license released by RES on behalf of The Renewables Infrastructure Group (TRIG) (DOI: \href{https://doi.org/10.5281/zenodo.20204946}{10.5281/zenodo.20204946}). Elexon System Buy Price records are freely accessible through the Elexon BMRS portal under the UK Open Government Licence v3.0. The complete Python verification suite---including computational graph static parsers, circular block bootstrap scripts, Elexon BSC physical settlement calculators, and figure reproduction scripts---is openly archived at Zenodo and GitHub (\href{https://github.com/anonymous-audit/pcwpf-audit}{\texttt{anonymous-audit/pcwpf-audit}}). The full 46-study scoping extraction database with verbatim quotes and multi-coder consensus scores is provided as open supplementary data.

\setlength{\bibsep}{2.5pt plus 0.5pt minus 0.5pt}
\small
\bibliography{refs}

\end{document}
"""

# Write all files
files_to_write = {
    os.path.join(sections_dir, "01_intro.tex"): sec01,
    os.path.join(sections_dir, "02_related.tex"): sec02,
    os.path.join(sections_dir, "03_theory.tex"): sec03,
    os.path.join(sections_dir, "04_methodology.tex"): sec04,
    os.path.join(sections_dir, "05_audit.tex"): sec05,
    os.path.join(sections_dir, "06_experiments.tex"): sec06,
    os.path.join(sections_dir, "07_discussion.tex"): sec07,
    os.path.join(sections_dir, "08_conclusion.tex"): sec08,
    os.path.join(sections_dir, "09_appendix.tex"): sec09,
    os.path.join(base_dir, "main.tex"): main_content,
}

for path, content in files_to_write.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Written: {path}")

# Remove old files if they exist to avoid confusion
old_files_to_remove = [
    os.path.join(sections_dir, "03_framework.tex"),
    os.path.join(sections_dir, "04_protocol.tex"),
    os.path.join(sections_dir, "05_results.tex"),
    os.path.join(sections_dir, "06_code_audit.tex"),
    os.path.join(sections_dir, "07_theory.tex"),
    os.path.join(sections_dir, "08_discussion.tex"),
    os.path.join(sections_dir, "09_conclusion.tex"),
    os.path.join(sections_dir, "10_appendix.tex"),
    os.path.join(sections_dir, "11_starmethods.tex"),
]

for old_path in old_files_to_remove:
    if os.path.exists(old_path):
        os.remove(old_path)
        print(f"Removed legacy file: {old_path}")

print("Reconstruction build script finished successfully.")
