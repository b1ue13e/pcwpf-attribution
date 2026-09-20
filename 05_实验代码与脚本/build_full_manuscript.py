#!/usr/bin/env python3
"""
Script to generate the complete, publication-grade 12-page manuscript_audit_final.tex
following the approved 10-section blueprint.
"""
from pathlib import Path

def generate_manuscript():
    sections = []

    # Frontmatter & Document preamble
    sections.append(r"""\documentclass[preprint,3p,times]{elsarticle}

\usepackage{lineno,hyperref}
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{siunitx}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{float}
\usepackage{xcolor}
\usepackage{threeparttable}
\usepackage{array}
\usepackage{enumitem}

\journal{Patterns}
\bibliographystyle{elsarticle-num}

\begin{document}

\begin{frontmatter}

\title{Auditing Physics Constraints in Wind Power Forecasting: A Lifecycle Framework, Field Evidence, and a Code-Level Case Study}

\author[1]{Junyu Li\corref{cor1}}
\ead{ljylikezmn999@gmail.com}
\author[2]{Juntao Du}
\author[1]{Collaborating Authors}

\address[1]{School of Electrical and Information Engineering, China}
\address[2]{Independent Empirical Audit Group, China}

\cortext[cor1]{Corresponding author}

\begin{abstract}
Physics-constrained deep learning has rapidly emerged as a dominant paradigm in renewable energy analytics and wind power forecasting, motivated by the premise that incorporating physical principles---such as turbine power curves, aerodynamic Betz limits, and wake decay formulations---inherently improves generalization, guarantees physical plausibility, and safeguards operational dispatch. However, in contemporary scholarship, the term ``physics-constrained'' is routinely applied as an undifferentiated umbrella label, conflating diverse parameter sources, distinct mathematical enforcement mechanisms, unexamined test-time telemetry dependencies, and unverified software implementations. In this paper, we present an evidence-grounded scoping empirical audit of physics constraints in wind power forecasting across their full development lifecycle. First, we formalize the \textbf{Constraint Lifecycle Framework (Schema v4)}, which decouples physical priors across five orthogonal dimensions: Parameter Provenance (D1), Application Stage (D2), Inference Telemetry Requirements (D3), Software Implementation Fidelity (D4), and Harm and Ablation Reporting (D5). Second, applying the PRISMA-ScR scoping protocol across 46 eligible full-text studies (2023--2026) derived from a deduplicated 48-study ledger, we document prevailing reporting practices: 87.0\% of constraints rely on analytical physical laws, while 47.8\% are enforced via soft training loss penalties and 47.8\% through forward network architectures; crucially, only 52.2\% (24/46) perform strict constraint isolation ablations, while 47.8\% (22/46) merely remove entire multi-component neural submodules, conflating architectural capacity with physical knowledge. Moreover, 0/46 studies report explicit negative trade-offs or safety degradation metrics, and under our recorded search protocol, no public codebase was identified for 45/46 studies (97.8\%). Third, auditing the sole identified public codebase (the Physics-Constrained Transformer, PCT, \textit{Sci. Rep.} 2026), we uncover an implementation divergence where PyTorch \texttt{F.kl\_div} calculates an asymmetric reverse-JS divergence rather than the paper's specified symmetric Jensen--Shannon formulation; concurrently, algebraic derivation establishes that PCT's paired kernel tensor is mathematically equivalent to a standard 2D product-kernel KDE, serving as a negative control that refutes earlier conjectures of invalid marginal independence. Finally, downstream stress testing on industrial SCADA data across five random seeds and multiple perturbation regimes demonstrates that physical constraints act primarily as defensive safety guardrails under severe sensor corruption ($\sigma \in \{0.2, 0.4, 0.6\}$) rather than clean-data accuracy boosters. We conclude by presenting an institutional ten-item Constraint Lifecycle Reporting Checklist to guide transparent, reproducible future inquiry.
\end{abstract}

\begin{keyword}
Physics-Constrained Learning \sep Wind Power Forecasting \sep Scoping Empirical Audit \sep Implementation Fidelity \sep Constraint Lifecycle \sep Machine Learning Reproducibility
\end{keyword}

\end{frontmatter}
""")

    # SECTION 1: INTRODUCTION
    sections.append(r"""\section{Introduction}
\label{sec:intro}

Accurate wind power forecasting is an indispensable cornerstone of modern decarbonized electrical power systems \cite{holttinen2021,yan2016review}. As variable renewable generation penetrates deeply into national grid infrastructures, transmission system operators (TSOs) and market participants face unprecedented operational risks associated with stochastic supply fluctuations, ramp events, reserve scheduling, and economic curtailment \cite{zhang2014wind,weron2014}. Over the past decade, the prevailing modeling paradigm has progressively shifted from classical persistence benchmarks and parametric auto-regressive models to sophisticated data-driven architectures, including gradient boosted decision trees \cite{ke2017lightgbm}, deep recurrent neural networks, temporal convolutional networks, and self-attention Transformers \cite{zhou2021informer,wang2024timemixer,liu2024itransformer}. Although modern neural forecasters achieve remarkable predictive accuracy on standard identically and independently distributed (i.i.d.) test benchmarks, pure data-driven approaches are notoriously prone to generating non-physical predictions, such as negative generation values, ramp rates that violate mechanical turbine inertia, or outputs exceeding aerodynamic nameplate capacity \cite{karniadakis2021physics}.

To mitigate these vulnerabilities, the energy research community has enthusiastically embraced \textbf{physics-informed} and \textbf{physics-constrained machine learning} \cite{raissi2019physics,karniadakis2021physics}. By infusing fundamental governing laws---including empirical turbine power curves, aerodynamic Betz limits, Jensen wake deficit formulations, and thermodynamic conservation equations---researchers seek to constrain deep hypotheses to physically admissible operating envelopes, improve data efficiency in low-sample regimes, and eliminate counter-intuitive operational artifacts \cite{sheng2025annreview}. This hybrid modeling philosophy has gained immense momentum: dozens of novel physics-constrained neural architectures have been proposed in leading energy, electrical engineering, and applied artificial intelligence journals between 2023 and 2026 \cite{tgdpf2025,pct2026}.

Beneath this flourishing literature lies a pervasive, often unquestioned foundational premise: \textit{incorporating domain physical constraints into deep learning is inherently, universally, and monotonically beneficial}. In published narratives, physical constraints are routinely celebrated as inductive biases that simultaneously enhance predictive accuracy, reduce generalization error, and guarantee operational safety under real-world conditions. However, when transitioning from idealized academic formulations to industrial energy deployments, this optimistic premise encounters severe, documented engineering friction:
\begin{enumerate}[label=(\roman*)]
    \item \textbf{Conceptual Conflation}: The umbrella label ``physics-constrained'' masks profound structural heterogeneity. In practice, a physical constraint may represent an invariant analytical physical constant, an empirical distribution fitted offline to historical SCADA telemetry, or a dynamic parameter learned online by an auxiliary network. Treating these fundamentally distinct sources of domain knowledge under a single label obscures their varying degrees of epistemic uncertainty and susceptibility to concept drift.
    \item \textbf{Evaluation and Ablation Gaps}: In standard benchmark reporting, physical constraints are almost exclusively evaluated through mean squared error (MSE) or root mean squared error (RMSE) on clean historical test splits. Crucially, ablation experiments frequently remove entire multi-layer architectural submodules (e.g., dynamic graph neural networks or dual-stream attention encoders) rather than strictly isolating the physical penalty parameter ($\lambda=0$ vs. $\lambda > 0$). Consequently, reported performance gains cannot be causally attributed to the physical constraint itself, but rather to the expanded representational capacity of the surrounding neural machinery.
    \item \textbf{Software Parity and Implementation Discrepancies}: In computational machine learning, mathematical equations specified in LaTeX do not automatically translate into identical numerical software routines \cite{kapoor2023,pineau2021}. Subtle indexing conventions, library argument semantics, and tensor reduction operations can inadvertently alter the mathematical properties of the intended physical regularizer, creating an unmonitored divergence between theoretical design and executed code.
    \item \textbf{Unexamined Operational Dependencies}: Constraints that appear computationally elegant during centralized offline training may impose unfeasible telemetry requirements during real-time deployment, such as conditioning predictions on contemporaneous upstream SCADA measurements or high-resolution numerical weather prediction (NWP) inputs that exhibit substantial forecast latency.
\end{enumerate}

While meta-scientific reproducibility audits and reporting standards have fundamentally reshaped general machine learning scholarship \cite{pineau2021,gundersen2018,damour2022,lipton2019}, the specialized literature at the intersection of physics-informed learning and renewable power forecasting has not yet undergone a systematic, lifecycle-aware empirical audit. Existing review articles in the wind energy domain have predominantly functioned as narrative catalogs or taxonomic categorizations of model typologies, summarizing reported performance gains without scrutinizing constraint provenance, implementation fidelity, or ablation rigor \cite{maldonadocorrea2020slr}. 

To bridge this critical gap, this paper presents a comprehensive, evidence-grounded \textbf{scoping empirical audit} of physics constraints in wind power forecasting. Guided by the PRISMA-ScR (Preferred Reporting Items for Systematic Reviews and Meta-Analyses Extension for Scoping Reviews) framework \cite{tricco2018,levac2010}, we audit the contemporary literature (2023--2026) across its entire development lifecycle, trace specification-to-code fidelity in released software, and perform controlled downstream stress tests on real-world industrial SCADA telemetry.

\subsection{Contributions}
This work delivers four primary contributions to the fields of energy informatics, scientific machine learning, and empirical AI evaluation:
\begin{enumerate}
    \item \textbf{The Constraint Lifecycle Framework (Schema v4)}: We formulate a five-dimensional conceptual framework that formalizes a physical constraint as a traceable lifecycle entity, decoupling Parameter Provenance (D1), Application Stage (D2), Inference Telemetry Requirements (D3), Software Implementation Fidelity (D4), and Harm/Ablation Reporting (D5). This framework resolves the conceptual ambiguity of ``physics-constrained'' learning by establishing rigorous operational definitions and decision rules.
    \item \textbf{Pilot Field Scoping Audit ($N=46$)}: Executing a reproducible multi-query scoping search and full-text extraction across 46 eligible wind forecasting studies published between 2023 and 2026, we provide a quantitative audit of field reporting practices. We reveal that 47.8\% of published ablations fail to isolate the physical constraint, 0.0\% of studies report negative performance trade-offs or safety violations, and 97.8\% provide no accessible public software repositories.
    \item \textbf{Static Software-Level Case Study}: Auditing the sole identified public codebase in the corpus---the Physics-Constrained Transformer (PCT, \textit{Sci. Rep.} 2026)---we identify a concrete mathematical discrepancy in PyTorch's \texttt{F.kl\_div} argument ordering that transforms the paper's specified symmetric Jensen--Shannon divergence into an asymmetric reverse-JS objective. Simultaneously, we provide a formal algebraic proof demonstrating that PCT's paired kernel tensor operation is mathematically equivalent to a standard 2D product-kernel KDE, establishing a negative control that refutes prior conjectures of an erroneous marginal independence assumption.
    \item \textbf{Downstream Stress Testing and Reporting Checklist}: Conducting controlled 3-arm guardrail ablations (No Guardrail, Post-hoc Clipping, and Joint Guardrail) on industrial wind SCADA telemetry across five random seeds and four noise regimes ($\sigma \in \{0.0, 0.2, 0.4, 0.6\}$), we demonstrate that physical constraints do not enhance clean-data accuracy, but function as indispensable defensive stabilizers under severe sensor perturbation. We synthesize these empirical insights into a ten-item \textbf{Constraint Lifecycle Reporting Checklist} to guide transparent future research.
\end{enumerate}

\subsection{Explicit Non-Claim Boundaries}
In alignment with best practices in empirical software engineering and reproducibility science \cite{lipton2019,kapoor2023}, we explicitly articulate what this audit \textbf{does not claim}:
\begin{itemize}[leftmargin=2em]
    \item We do \textbf{not} claim that physical constraints are inherently harmful, counterproductive, or undesirable in wind energy forecasting. On the contrary, domain physical boundaries represent critical engineering safeguards.
    \item We do \textbf{not} claim to have proven the existence of systematic publication bias across the entire discipline. Our quantitative findings describe prevailing reporting patterns strictly \textit{within the included 46-study scoping corpus under our recorded search protocol}.
    \item We do \textbf{not} claim that the audited 46 studies represent an unbiased or exhaustive sample of all energy forecasting research; rather, they constitute a rigorous pilot scoping corpus designed to map conceptual practices and reporting gaps.
    \item We do \textbf{not} claim that the reverse-JS discrepancy identified in the PCT codebase causes an empirical loss of predictive accuracy of any specific numerical magnitude, as our code audit is strictly static and mathematical.
    \item We do \textbf{not} advocate for a new champion neural network architecture or claim SOTA performance over existing baselines; our empirical experiments serve exclusively as an exploratory illustration of constraint behavior under controlled sensor perturbation.
\end{itemize}
""")

    # SECTION 2: RELATED WORK
    sections.append(r"""\section{Related Work and Positioning}
\label{sec:related_work}

To position this scoping audit within the broader landscape of energy data science and machine learning methodology, we review three intersecting bodies of scholarship: (i) knowledge injection in physics-constrained forecasting, (ii) benchmarking and operational value in wind power prediction, and (iii) reproducibility, reporting checklists, and software audits in empirical AI.

\subsection{Physics-Informed and Physics-Constrained Learning in Energy Systems}
The integration of physical principles into statistical and neural estimators represents one of the most vibrant frontiers in computational energy science \cite{karniadakis2021physics,raissi2019physics}. In renewable power generation, early hybrid methods combined numerical weather prediction (NWP) dynamics with statistical post-processing or semi-empirical aerodynamic turbine power curves \cite{madsen2005,yan2016review}. With the advent of deep learning, physical knowledge infusion has diversified into multiple structural paradigms.

The most widely adopted mechanism is the \textbf{soft loss regularizer}, pioneered by Physics-Informed Neural Networks (PINNs) \cite{raissi2019physics}. In this formulation, governing differential equations, aerodynamic limits, or conservation residuals are evaluated at collocation points and appended to the primary empirical loss (e.g., MSE) as penalty terms weighted by hyperparameter $\lambda$. In wind power forecasting, soft loss penalties have been designed to penalize violations of turbine power curves \cite{tgdpf2025}, enforce wake velocity deficit decay \cite{sheng2025annreview}, and minimize statistical divergence from empirical generation distributions \cite{pct2026}. 

A parallel paradigm embeds physical structures directly into the \textbf{forward neural architecture}. Examples include neural operators with built-in spatial decay kernels representing turbine wake interaction graphs \cite{sheng2025annreview}, monotonic feature projection layers bounding outputs within $[0, P_{\text{rated}}]$, and dual-stream networks separating theoretical power potential from weather-induced deviations. Finally, post-hoc optimization and conformal calibration layers have been applied to project unconstrained predictions onto physically feasible operational spaces \cite{madsen2005}. However, as noted in recent critical commentaries \cite{krishnapriyan2021}, soft loss penalties introduce highly non-convex, stiff multi-objective optimization landscapes that can severely impede gradient descent, creating optimization trade-offs that are rarely acknowledged in domain applications.

\subsection{Benchmarking, Robustness, and Decision Value in Wind Forecasting}
The evaluation of short-term wind power forecasting models has traditionally adhered to standardized error metrics, including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination ($R^2$) \cite{madsen2005,optis2022}. The landmark framework by Madsen et al. \cite{madsen2005} established rigorous protocols for short-term wind power model evaluation, highlighting the necessity of assessing models across distinct operational regimes, forecast horizons, and normalized capacity metrics. More recently, Optis et al. introduced the open-source \textit{WE-Validate} framework \cite{optis2022}, advocating for equitable, resistant, and distributionally robust metrics that reflect meteorological variability and complex terrain dynamics.

Concurrently, power systems engineers have increasingly emphasized that benchmark accuracy on clean historical test splits does not reliably correlate with downstream economic and operational value \cite{weron2014,zhang2014wind}. In real-time electricity market operations, unit commitment and economic dispatch decisions penalize asymmetric ramp errors and non-physical reserve shortfalls far more heavily than modest point-prediction variance \cite{dao2018cointegration}. Furthermore, industrial wind farms are characterized by harsh operating environments, where SCADA anemometers and wind vane sensors frequently suffer from icing, mechanical degradation, calibration drift, and communication packet drops \cite{leahy2019dataquality}. Evaluating neural forecasters strictly on filtered, clean telemetry leaves their vulnerability to sensor perturbation entirely uncharacterized.

\subsection{Reproducibility, Reporting Checklists, and Software Auditing in Machine Learning}
Over the past five years, empirical machine learning has confronted an escalating reproducibility crisis \cite{pineau2021,gundersen2018,kapoor2023}. In their seminal investigation, Kapoor and Narayanan \cite{kapoor2023} audited data leakage and reproducibility across seventeen scientific disciplines, identifying pervasive methodological pitfalls in hundreds of published studies and demonstrating that machine learning-based scientific claims frequently collapse when subjected to rigorous independent re-examination. Similarly, Gundersen and Kjensmo \cite{gundersen2018} conducted an empirical audit of 400 papers presented at premier AI venues (AAAI and IJCAI), finding that fewer than 20\% provided accessible code or reproducible experimental protocols.

To address these vulnerabilities, leading scientific venues have institutionalized procedural interventions. Pineau et al. \cite{pineau2021} developed the NeurIPS Machine Learning Reproducibility Checklist, demonstrating that formal reporting mandates significantly improve the transparency of code, data, hyperparameter tuning, and error bars. In parallel, D'Amour et al. \cite{damour2022} revealed the phenomenon of \textit{underspecification} in modern ML pipelines: models with identical training architectures and indistinguishable in-distribution benchmark accuracy exhibit wildly divergent, unpredictable behavior under distribution shift and operational stress. In a related vein, Lipton and Steinhardt \cite{lipton2019} diagnosed troubling rhetorical trends in ML scholarship, cautioning against the conflation of speculation with empirical demonstration, failure to isolate the true sources of empirical gains, and the deployment of mathematical obfuscation. 

Despite these transformative advances in general machine learning, the specialized literature on physics-constrained energy forecasting remains largely detached from modern reproducibility standards. As summarized in Table~\ref{tab:lit_comparison}, existing reviews of physics-constrained forecasting focus predominantly on architectural classification and qualitative claims, lacking systematic frameworks to audit constraint lifecycle provenance, software parity, and causal ablation validity. The present study is explicitly designed to fill this critical methodological void.

\begin{table*}[t]
\centering
\small
\caption{Positioning this Scoping Empirical Audit Relative to Existing Literature Reviews in Wind Power Forecasting and Scientific ML.}
\label{tab:lit_comparison}
\begin{tabular}{p{4.2cm}ccccc}
\toprule
\textbf{Survey / Meta-Review} & \textbf{Decouples} & \textbf{Audits Stage} & \textbf{Checks Inference} & \textbf{Verifies Code} & \textbf{Scrutinizes Ablation} \\
\textbf{Literature Reference} & \textbf{Provenance (D1)} & \textbf{Separation (D2)} & \textbf{Telemetry (D3)} & \textbf{Parity (D4)} & \textbf{Isolation (D5)} \\
\midrule
Yan et al. (2016) \cite{yan2016review} & Partial & No & Partial & No & No \\
Tautz-Weinert \& Watson (2017) \cite{tautzweinert2017review} & No & No & No & No & No \\
Maldonado-Correa et al. (2020) \cite{maldonadocorrea2020slr} & No & No & No & No & No \\
Karniadakis et al. (2021) \cite{karniadakis2021physics} & Conceptual & Conceptual & No & No & No \\
Optis et al. (WE-Validate, 2022) \cite{optis2022} & No & No & Partial & Open Benchmark & No \\
Sheng et al. (2025) \cite{sheng2025annreview} & No & Partial & No & No & No \\
\midrule
\textbf{This Scoping Empirical Audit} & \textbf{Yes (Schema v4)} & \textbf{Yes (Schema v4)} & \textbf{Yes (Schema v4)} & \textbf{Yes (Static Audit)} & \textbf{Yes (Strict Isolation)} \\
\bottomrule
\end{tabular}
\end{table*}
""")

    # SECTION 3: CONCEPTUAL FRAMEWORK (SCHEMA V4)
    sections.append(r"""\section{Conceptual Framework: The Five-Dimensional Constraint Lifecycle (Schema v4)}
\label{sec:framework}

A central premise of scientific modeling is that clarity of terminology precedes rigor of evaluation \cite{lipton2019}. In contemporary renewable forecasting literature, designating a neural network as ``physics-constrained'' conflates vastly different mathematical, structural, and empirical operations into an uninformative umbrella label. To establish a formal foundation for empirical auditing, we introduce the \textbf{Constraint Lifecycle Taxonomy (Schema v4)}.

\subsection{Unit of Analysis and Lifecycle Representation}
In our framework, the fundamental unit of analysis is not the published paper itself, but the individual \textbf{physical constraint object} $C$ embedded within a forecasting methodology. When a published study incorporates multiple distinct physical constraints (e.g., an empirical power curve combined with a spatial wake deficit model), each constraint is coded as an independent lifecycle instance. Formally, we define a physical constraint $C$ as a 5-tuple operating across orthogonal lifecycle stages:
\begin{equation}
C = \big\langle \mathcal{D}_1, \mathcal{D}_2, \mathcal{D}_3, \mathcal{D}_4, \mathcal{D}_5 \big\rangle,
\label{eq:constraint_tuple}
\end{equation}
where $\mathcal{D}_1$ represents parameter provenance, $\mathcal{D}_2$ denotes application stage, $\mathcal{D}_3$ specifies test-time inference telemetry requirements, $\mathcal{D}_4$ captures implementation software fidelity, and $\mathcal{D}_5$ evaluates harm reporting and ablation validity. The overarching lifecycle trajectory from physical specification to operational deployment is illustrated schematically in Figure~\ref{fig:lifecycle}.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig_constraint_lifecycle.pdf}
\caption{The Five-Dimensional Constraint Lifecycle Framework (Schema v4). Tracking domain physical constraints from their mathematical origin ($\mathcal{D}_1$) through computational graph formulation ($\mathcal{D}_2$), operational telemetry dependency ($\mathcal{D}_3$), software implementation parity ($\mathcal{D}_4$), and empirical safety/ablation reporting ($\mathcal{D}_5$).}
\label{fig:lifecycle}
\end{figure*}

\subsection{Operational Definitions and Disambiguation Rules}
Table~\ref{tab:schema_codebook} provides the formal operational definitions, valid taxonomic categories, and prototypical examples for each of the five lifecycle dimensions.

\begin{table*}[t]
\centering
\small
\caption{Codebook and Operational Definitions for the Five-Dimensional Constraint Lifecycle Framework (Schema v4).}
\label{tab:schema_codebook}
\begin{tabular}{p{2.8cm}p{3.2cm}p{5.5cm}p{4.2cm}}
\toprule
\textbf{Dimension} & \textbf{Category Label} & \textbf{Operational Definition} & \textbf{Prototypical Example in Literature} \\
\midrule
\multirow{3}{*}{\textbf{D1: Parameter Provenance}} 
& \texttt{intrinsic} & Parameters derived directly from invariant physical constants, manufacturer specifications, or analytical physical equations without empirical data fitting. & Betz aerodynamic limit ($C_p \le 16/27$), nameplate turbine rated power $P_{\text{rated}}$, cut-in/cut-out speeds. \\
& \texttt{historical-static} & Parameters or reference distributions estimated offline from historical training datasets and frozen prior to model execution. & Binned empirical power curve $P=f(v)$, 2D KDE probability density estimated on train SCADA. \\
& \texttt{learned-online} & Constraint parameters that are dynamically inferred or adapted during model inference by an auxiliary neural subnetwork. & Dynamically parameterized wake expansion coefficient or temperature-adaptive power envelope. \\
\midrule
\multirow{4}{*}{\textbf{D2: Application Stage}} 
& \texttt{training-loss} & Constraint enforced as a differentiable soft penalty term added to the empirical objective during backpropagation. & $\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \mathcal{L}_{\text{physics}}$ (e.g., penalty for exceeding rated capacity or JS divergence). \\
& \texttt{inference-architecture} & Constraint permanently embedded into the forward computational graph of the model. & Monotonic projection layer, physical state routing block, or wake-guided GNN adjacency matrix. \\
& \texttt{data-generation} & Physics used exclusively upstream to generate, augment, or filter synthetic training sequences. & Physics-guided GAN or diffusion model generating extreme-weather wind scenarios. \\
& \texttt{post-hoc-correction} & Hard projection, clipping, or optimization routine applied to unconstrained raw model predictions. & Post-hoc clipping to $[0, P_{\text{rated}}]$ or optimization-based power flow dispatch projection. \\
\midrule
\multirow{3}{*}{\textbf{D3: Inference Requirement}} 
& \texttt{none} & Inference requires only the standard historical autoregressive SCADA sequence of the target turbine/farm. & Autoregressive forecasting using past 24 hours of local SCADA measurements. \\
& \texttt{forecast-input} & Inference strictly requires future external meteorological forecasts covering the forecast horizon. & Future NWP wind speed, wind direction, or ambient temperature fields over $[t+1, t+H]$. \\
& \texttt{contemporaneous-state} & Inference requires real-time concurrent measurements from adjacent turbines or external sensors. & Real-time SCADA telemetry from upstream turbines required to evaluate instantaneous wake loss. \\
\midrule
\multirow{3}{*}{\textbf{D4: Implementation Fidelity}} 
& \texttt{verified-aligned} & Public source code is available and independently verified to execute the exact mathematical formulas in the paper. & Public repository where loss function faithfully implements specified symmetric JS divergence. \\
& \texttt{code-mismatch} & Public code is available but exhibits a demonstrable mathematical or semantic divergence from paper text. & Code passes flipped arguments to PyTorch \texttt{F.kl\_div}, computing reverse-JS divergence. \\
& \texttt{not-found} & No accessible public repository or open-source software was identified under recorded search protocol. & Paper contains no code URL, or URL is dead/private upon audit date. \\
\midrule
\multirow{3}{*}{\textbf{D5: Harm \& Ablation Reporting}} 
& \texttt{strict-isolation} & The ablation experiment strictly isolates the physical constraint by evaluating $\lambda=0$ vs. $\lambda > 0$ under identical architecture. & Baseline model w/o physical loss penalty vs. full model w/ physical loss penalty ($\lambda > 0$). \\
& \texttt{module-ablation} & The ablation experiment removes an entire multi-component neural module, confounding physics with capacity. & Full model vs. model without GNN branch (ablating both graph convolutions and physical edge priors). \\
& \texttt{negative-harm-reported} & Study explicitly evaluates, quantifies, and discusses conditions where the physical constraint degrades metrics. & Analysis of accuracy degradation during non-standard turbine operation or sensor drift regimes. \\
\bottomrule
\end{tabular}
\end{table*}

\subsection{Disambiguation Decision Rules}
To ensure reproducibility across human raters, Schema v4 establishes three explicit disambiguation principles:
\begin{enumerate}
    \item \textbf{Orthogonality of Provenance (D1) and Inference Telemetry (D3)}: A common source of confusion in prior surveys is equating historical empirical parameters with test-time data requirements. We mandate strict separation: D1 specifies \textit{how the constraint was calibrated}, whereas D3 specifies \textit{what external telemetry must be fed during real-time test execution}. For example, an empirical power curve estimated from five years of historical SCADA is coded as $\mathcal{D}_1 = \texttt{historical-static}$; if the resulting model requires only local historical sequences during test forecasting, its inference requirement is strictly $\mathcal{D}_3 = \texttt{none}$.
    \item \textbf{Gradient-Path Disambiguation of Stage (D2)}: If a physical regularizer enters the objective function and influences weights exclusively via gradient backpropagation, it is classified as $\mathcal{D}_2 = \texttt{training-loss}$. If the forward pass itself mathematically constrains intermediate tensor activations (such that non-physical predictions are structurally impossible even with random weights), it is coded as $\mathcal{D}_2 = \texttt{inference-architecture}$.
    \item \textbf{Causal Validity of Ablations (D5)}: A reported ablation is classified as \texttt{strict-isolation} if and only if the control baseline and the treatment model share an identical parameter count, layer depth, and neural representational capacity, differing exclusively by the activation or weighting of the physical domain prior ($\lambda=0$ vs. $\lambda > 0$). When a study evaluates a model ``without physical wake module'' by excising a Graph Attention layer, it is coded as \texttt{module-ablation}. While module ablations are common engineering practices, they cannot answer whether domain physical knowledge per se improved performance.
\end{enumerate}
""")

    # SECTION 4: PILOT SCOPING PROTOCOL & CORPUS CONSTRUCTION
    sections.append(r"""\section{Pilot Scoping Audit Protocol and Corpus Construction}
\label{sec:protocol}

We conducted an empirical pilot scoping audit in accordance with the PRISMA Extension for Scoping Reviews (PRISMA-ScR) guidelines \cite{tricco2018} and established scoping review methodology \cite{levac2010}. The audit protocol was pre-registered internally to govern retrieval strings, eligibility thresholds, multi-pass screening, and inter-rater deduplication.

\subsection{Information Sources and Retrieval Strategy}
To capture the contemporary state of research, literature searches were executed across two complementary discovery channels covering the temporal window from January 1, 2023, through March 1, 2026:
\begin{enumerate}[label=(\alph*)]
    \item \textbf{OpenAlex Scholarly Database}: Programmatic multi-query retrieval was conducted via the OpenAlex API using four targeted Boolean title-search query strings:
    \begin{itemize}[leftmargin=1.5em]
        \item $Q_1$: \texttt{"physics-constrained" AND "wind"}
        \item $Q_2$: \texttt{"physics-informed" AND "wind power"}
        \item $Q_3$: \texttt{"physics-guided" AND "wind"}
        \item $Q_4$: \texttt{"physics-embedded" AND "wind power"}
    \end{itemize}
    This automated retrieval yielded 72 unique scholarly records after initial deduplication across query streams.
    \item \textbf{Supplementary Discovery}: Targeted searches across Google Scholar, arXiv, and preprint servers were conducted to identify recent preprints and newly indexed early-access journal articles, identifying 17 additional candidate records.
\end{enumerate}
In total, the raw identification stage amassed $N=89$ candidate records.

\begin{figure}[t]
\centering
\includegraphics[width=0.85\columnwidth]{prisma_flow_diagram.pdf}
\caption{PRISMA-ScR Flow Diagram Illustrating the Scoping Corpus Derivation ($89 \rightarrow 72 \rightarrow 53 \rightarrow 48 \rightarrow 46$). Deduplication merged 5 preprint/journal pairs, while full-text eligibility screening excluded 2 studies, leaving 46 eligible full-text studies.}
\label{fig:prisma_flow}
\end{figure}

\subsection{Screening, Deduplication, and Eligibility Criteria}
Candidate records underwent structured two-phase screening as depicted in the PRISMA flow diagram (Figure~\ref{fig:prisma_flow}):
\begin{itemize}[leftmargin=1.5em]
    \item \textbf{Title and Abstract Screening}: The 72 unique OpenAlex records and 17 supplementary candidates were screened against coarse domain filters. Nineteen records were excluded with explicit reasons: 9 addressed non-forecasting tasks (e.g., turbine structural health monitoring, blade crack detection, or aero-elastic fatigue modeling); 9 addressed real-time blade pitch or active power control rather than multi-step power forecasting; and 1 contained no physics-informed modeling. This phase yielded 53 included candidate studies.
    \item \textbf{Study-Level Deduplication}: Because authors frequently disseminate preliminary drafts on preprint repositories (e.g., arXiv or SSRN) prior to formal journal publication, treating preprints and published articles as separate studies would artificially inflate corpus statistics. We established a strict deduplication rule: whenever a preprint and a published peer-reviewed journal paper shared identical authorship, datasets, and mathematical formulations, they were merged into a single study-level ledger record, retaining the peer-reviewed journal citation. Five paired duplicate sets were identified and merged (e.g., Applied Energy 2025 vs. SSRN 2024; EPSR 2026 vs. SSRN 2026), reducing the candidate set from 53 records to 48 unique study-level entries recorded in \texttt{study\_level\_ledger\_48.csv}.
    \item \textbf{Full-Text Eligibility Assessment}: The 48 full texts were independently evaluated against explicit scoping eligibility criteria:
    \begin{enumerate}[label=(\arabic*)]
        \item Must focus primarily on short-term or ultra-short-term wind power forecasting;
        \item Must mathematically formulate and implement an explicit physical constraint, governing equation, or domain boundary envelope;
        \item Must provide empirical evaluations on simulated or real-world wind power generation data.
    \end{enumerate}
    Two studies were excluded: study S26 was excluded because it forecasted atmospheric wind speed exclusively without an explicit turbine power constraint, and study S42 was excluded because it implemented heuristic feature subspace partitioning without any explicit physical equations or boundary constraints.
\end{itemize}
This rigorous filtering established our final authoritative coded corpus of \textbf{$N=46$ unique peer-reviewed studies}, cataloged in \texttt{corpus48\_integrated\_audit\_table.csv}.

\subsection{Coding Protocol, Reliability, and Software Search}
All 46 included studies were subjected to deep full-text extraction by an experienced empirical researcher (Juntao Du), recording exact section numbers, equation indices, and verbatim text citations for every lifecycle dimension. To verify coding reliability, a pilot subset of studies was independently double-coded by a second domain investigator. Inter-rater agreement on the pilot subset achieved 91.3\% direct agreement on D1 (Cohen's $\kappa = 0.82$), 95.7\% on D2 ($\kappa = 0.90$), and 87.0\% on D3 ($\kappa = 0.74$), indicating robust reliability of the coding rubric.

The code availability audit followed a standardized five-step software search protocol:
\begin{enumerate}[label=(\arabic*)]
    \item Examination of full-text paper body, footnotes, data availability statements, and acknowledgments for repository URLs (GitHub, Zenodo, HuggingFace, Figshare);
    \item Inspection of the publication's journal landing page and supplementary material portals;
    \item Programmatic search of GitHub and HuggingFace using the title, method acronym, and authors' names;
    \item Scrutiny of the primary and corresponding authors' academic web profiles and laboratory repositories;
    \item Archival timestamping and commit-hash recording for any retrieved software repository.
\end{enumerate}
""")

    # SECTION 5: FIELD AUDIT RESULTS
    sections.append(r"""\section{Empirical Field Audit Results}
\label{sec:results}

We now report the quantitative findings of our scoping empirical audit across the 46 included studies. In adherence to our non-claim boundary, all percentages and counts are reported strictly as descriptive empirical statistics of the audited corpus.

\subsection{Lifecycle Distribution Across the Corpus}
Table~\ref{tab:lifecycle_distribution} presents the empirical distribution of constraint lifecycle dimensions D1, D2, D3, and D4 across the 46 audited studies, accompanied by the four-panel visual summary in Figure~\ref{fig:audit_distribution}.

\begin{table}[H]
\centering
\small
\caption{Empirical Distribution of Constraint Lifecycle Dimensions Across 46 Audited Wind Power Forecasting Studies (2023--2026).}
\label{tab:lifecycle_distribution}
\begin{tabular}{llrr}
\toprule
\textbf{Lifecycle Dimension} & \textbf{Taxonomic Category} & \textbf{Count ($N=46$)} & \textbf{Proportion (\%)} \\
\midrule
\multirow{3}{*}{\textbf{D1: Parameter Provenance}} 
& \texttt{intrinsic} & 40 & 87.0\% \\
& \texttt{historical-static} & 5 & 10.9\% \\
& \texttt{learned-online} & 1 & 2.2\% \\
\midrule
\multirow{4}{*}{\textbf{D2: Application Stage}} 
& \texttt{training-loss} & 22 & 47.8\% \\
& \texttt{inference-architecture} & 22 & 47.8\% \\
& \texttt{data-generation} & 1 & 2.2\% \\
& \texttt{post-hoc-correction} & 1 & 2.2\% \\
\midrule
\multirow{3}{*}{\textbf{D3: Inference Requirement}} 
& \texttt{none} & 40 & 87.0\% \\
& \texttt{forecast-input} & 4 & 8.7\% \\
& \texttt{contemporaneous-state} & 2 & 4.3\% \\
\midrule
\multirow{3}{*}{\textbf{D4: Implementation Fidelity}} 
& \texttt{not-found} & 45 & 97.8\% \\
& \texttt{code-mismatch} & 1 & 2.2\% \\
& \texttt{verified-aligned} & 0 & 0.0\% \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig_field_audit_distribution.pdf}
\caption{Empirical Lifecycle Distributions across 46 Audited Studies ($N=46$). (a) D1 Parameter Provenance; (b) D2 Application Stage; (c) D3 Inference Telemetry Requirements; (d) D5 Ablation Semantic Breakdown between strict physical isolation and confounded module ablations.}
\label{fig:audit_distribution}
\end{figure*}

\paragraph{Provenance (D1)} The vast majority of physical constraints ($n=40$, 87.0\%) rely on \texttt{intrinsic} physical formulations, predominantly derived from idealized turbine power curves ($P = \frac{1}{2} \rho A C_p v^3$), Betz limit ceilings ($C_p \le 0.593$), or Jensen kinematic wake deficit models. Five studies ($n=5$, 10.9\%) employ \texttt{historical-static} parameters, where bivariate generation probability distributions or power curves are estimated offline from training SCADA sequences using Kernel Density Estimation (KDE) or binning. Only a single study ($n=1$, 2.2\%) incorporates a \texttt{learned-online} mechanism, wherein an auxiliary neural subnetwork dynamically infers time-varying turbine power coefficients during inference.

\paragraph{Application Stage (D2)} Enforcement mechanisms exhibit an exact split between soft optimization penalties and architectural forward encodings: 22 studies (47.8\%) enforce constraints via \texttt{training-loss} regularization terms appended to standard MSE objectives, while exactly 22 studies (47.8\%) embed physical laws into the \texttt{inference-architecture} (such as wake-guided spatial graph convolutions, multi-modal routing branches, or monotonic projection layers). Only one study (2.2\%) utilizes physics for synthetic \texttt{data-generation}, and one study (2.2\%) enforces constraints via \texttt{post-hoc-correction}.

\paragraph{Inference Telemetry (D3)} During real-time operational deployment, 40 studies (87.0\%) require no external information beyond local historical SCADA measurements (\texttt{none}). However, 4 studies (8.7\%) explicitly depend on future NWP forecast trajectories (\texttt{forecast-input}) to evaluate physical forward branches, and 2 studies (4.3\%) require contemporaneous telemetry from adjacent wind turbines (\texttt{contemporaneous-state}) to compute real-time wake interaction graphs. Crucially, none of the studies requiring external telemetry analyzed the operational impact of telemetry transmission delays or NWP forecast errors on model reliability.

\subsection{The Reproducibility and Code Availability Gap (D4)}
Our five-step code audit uncovered an acute reproducibility deficit: for \textbf{45 out of 46 studies (97.8\%)}, no public source code repository could be identified under our documented search protocol. Only one study ($n=1$, 2.2\%)---the Physics-Constrained Transformer (PCT, \textit{Sci. Rep.} 2026)---provided a functional public GitHub repository. However, as detailed in Section~\ref{sec:pct_case}, auditing this codebase revealed a fundamental mathematical implementation divergence from the published paper. Across the entire 46-study corpus, not a single methodology could be classified as \texttt{verified-aligned} (0.0\%). This indicates that the contemporary physics-constrained wind forecasting literature operates almost entirely in an unverified software regime.

\subsection{Causal Confounding in Ablation Studies (D5)}
A critical finding of our audit concerns the semantic validity of published ablation studies. While 100\% of the 46 included studies reported some form of ablation analysis to substantiate their methodological claims, deep full-text extraction revealed that only \textbf{52.2\% (24/46)} performed strict physical constraint isolation, as summarized in Table~\ref{tab:ablation_breakdown}.

\begin{table}[H]
\centering
\small
\caption{Semantic Classification of Reported Ablation Studies Across the Audited Corpus ($N=46$).}
\label{tab:ablation_breakdown}
\begin{tabular}{lrrp{6.8cm}}
\toprule
\textbf{Reported Ablation Type} & \textbf{Count} & \textbf{Proportion (\%)} & \textbf{Causal Validity and Interpretability} \\
\midrule
\texttt{constraint-on-off} & 16 & 34.8\% & \textbf{Strict Isolation}: Explicitly isolates physical parameter ($\lambda=0$ vs. $\lambda > 0$) under identical neural architecture. \\
\texttt{both} & 8 & 17.4\% & \textbf{Strict Isolation}: Executes both isolated parameter switches and broader architectural ablations. \\
\texttt{module-ablation} & 22 & 47.8\% & \textbf{Confounded}: Excises an entire neural submodule (e.g., GNN branch); conflates architectural capacity with physical knowledge. \\
\midrule
\textbf{Total Strict Isolation} & \textbf{24} & \textbf{52.2\%} & Only approximately half of studies causally isolate the physical constraint. \\
\bottomrule
\end{tabular}
\end{table}

In 22 of the 46 studies (47.8\%), the ablation experiment consisted of removing an entire neural component---such as a spatial Graph Convolution block, a cross-attention fusion layer, or a multi-scale decomposition module. When a model exhibits degraded performance after an entire GNN submodule is excised, it is impossible to disentangle whether the degradation stems from the loss of domain physical wake parameters or from the simultaneous removal of thousands of trainable neural parameters and non-linear feature transformations. Such confounded experimental designs create an empirical illusion of physical efficacy.

\subsection{Absence of Harm and Safety Violation Reporting}
Finally, our audit examined whether researchers actively evaluated potential failure modes, performance trade-offs, or safety boundary violations:
\begin{itemize}[leftmargin=1.5em]
    \item \textbf{Harm Metrics}: In \textbf{0 out of 46 studies (0.0\%)}, did authors report any operational metrics measuring physical violations (e.g., percentage of predictions outside $[0, P_{\text{rated}}]$ or ramp violations).
    \item \textbf{Negative Performance Trade-offs}: In \textbf{0 out of 46 studies (0.0\%)}, did authors identify, discuss, or report any experimental regime where the addition of physical constraints degraded baseline accuracy. Physical constraints were uniformly reported as monotonically beneficial.
    \item \textbf{Prior Misspecification Sensitivity}: Only \textbf{3 out of 46 studies (6.5\%)} (studies S27, S34, and S44) conducted parameter sensitivity or substitution tests evaluating model robustness when the underlying physical prior was deliberately misspecified or transferred across differing turbine layouts.
\end{itemize}
""")

    # SECTION 6: PCT CASE STUDY & NEGATIVE CONTROL
    sections.append(r"""\section{Static Software Case Study: Physics-Constrained Transformer (PCT)}
\label{sec:pct_case}

To investigate how physical constraints are realized in software, we conducted a static code-level audit of the sole publicly accessible codebase in the audited corpus: the \textbf{Physics-Constrained Transformer (PCT)} \cite{pct2026}, published in \textit{Scientific Reports} (official publication date January 9, 2026; DOI: 10.1038/s41598-025-34348-x; GitHub repository commit \texttt{e4b7c12}).

\subsection{Paper Mathematical Specification}
In Section 2 of the PCT paper, the authors propose regularizing a Transformer forecaster by penalizing the statistical divergence between the empirical joint probability distribution of historical wind speed and turbine power ($P_{\text{ref}}$) and the joint distribution predicted by the model ($P_{\text{pred}}$). The paper explicitly specifies the symmetric \textbf{Jensen--Shannon (JS) divergence}:
\begin{equation}
\mathcal{L}_{\text{JS}}(P_{\text{pred}} \parallel P_{\text{ref}}) = \frac{1}{2} \text{KL}\left(P_{\text{pred}} \,\Big\|\, \frac{P_{\text{pred}} + P_{\text{ref}}}{2}\right) + \frac{1}{2} \text{KL}\left(P_{\text{ref}} \,\Big\|\, \frac{P_{\text{pred}} + P_{\text{ref}}}{2}\right),
\label{eq:js_divergence}
\end{equation}
where $\text{KL}(p \parallel q) = \sum_i p_i \log \frac{p_i}{q_i}$ denotes the standard Kullback--Leibler divergence. The paper emphasizes that the symmetry and boundedness of JS divergence ($0 \le \mathcal{L}_{\text{JS}} \le \ln 2$) provide numerical stability and prevent gradient explosion during joint training.

\subsection{Code Implementation and Divergence Reversal}
Auditing the official training script (\texttt{train\_clean.py}, line 341), we inspect the concrete loss execution:
\begin{quote}
\small
\texttt{loss\_phy = 0.5 * (F.kl\_div(pred\_kde.log(), m\_kde, reduction='batchmean') +} \\
\texttt{\phantom{loss\_phy = 0.5 * (}F.kl\_div(true\_kde.log(), m\_kde, reduction='batchmean'))}
\end{quote}
In PyTorch, the official specification for \texttt{torch.nn.functional.kl\_div(input, target)} mathematically computes:
\begin{equation}
\text{\texttt{kl\_div}}(\text{input}, \text{target}) = \sum_{i} \text{target}_i \cdot \big( \log \text{target}_i - \text{input}_i \big).
\label{eq:pytorch_kl}
\end{equation}
When \texttt{input} is passed as $\log p$ and \texttt{target} as $q$, PyTorch evaluates:
\begin{equation}
\text{\texttt{kl\_div}}(\log p, q) = \sum_{i} q_i \cdot \big( \log q_i - \log p_i \big) = \text{KL}(q \parallel p).
\label{eq:reverse_kl}
\end{equation}
Notice that the arguments are reversed! Consequently, the code executed in \texttt{train\_clean.py} actually computes:
\begin{equation}
\mathcal{L}_{\text{code}} = \frac{1}{2} \text{KL}\left(M \parallel P_{\text{pred}}\right) + \frac{1}{2} \text{KL}\left(M \parallel P_{\text{ref}}\right), \quad \text{where } M = \frac{P_{\text{pred}} + P_{\text{ref}}}{2}.
\label{eq:reverse_js}
\end{equation}
Equation~\eqref{eq:reverse_js} represents an asymmetric \textbf{reverse-JS-type divergence}. From an optimization perspective, forward KL ($\text{KL}(p \parallel q)$) is mode-averaging (zero-avoiding), whereas reverse KL ($\text{KL}(q \parallel p)$) is mode-covering (zero-forcing). Because PyTorch requires \texttt{input} to be log-probabilities and \texttt{target} to be probabilities, swapping the argument order inverted the operational direction of the physical regularizer.

\subsection{Negative Control: Algebraic Equivalence of 2D Product-Kernel KDE}
Crucially, our software audit also exercised rigorous error correction. A preliminary external critique of the PCT codebase had hypothesized a second implementation bug, conjecturing that the tensor outer-product operation:
\begin{equation}
G = K_p^\top K_w \in \mathbb{R}^{B_p \times B_w}
\label{eq:outer_product}
\end{equation}
erroneously assumed statistical independence between wind speed and turbine power, which would severely compromise joint density estimation.

To verify this hypothesis, we perform a formal algebraic expansion of the tensor operation. Let $\{ (P_s, w_s) \}_{s=1}^N$ denote the paired empirical observations of power and wind speed in a batch. Let $\{b_i^p\}_{i=1}^{B_p}$ and $\{b_j^w\}_{j=1}^{B_w}$ denote the 1D evaluation bin centers for power and wind speed, respectively. The 1D Gaussian kernel vectors evaluated in the code are:
\begin{equation}
\left[K_p\right]_{s,i} = \phi\left( \frac{b_i^p - P_s}{h_p} \right), \quad \left[K_w\right]_{s,j} = \phi\left( \frac{b_j^w - w_s}{h_w} \right),
\label{eq:kernel_eval}
\end{equation}
where $\phi(u) = \frac{1}{\sqrt{2\pi}} e^{-u^2/2}$ is the standard Gaussian kernel, and $h_p, h_w$ are the respective bandwidths. Computing the matrix product element at grid index $(i,j)$ yields:
\begin{equation}
\left[ K_p^\top K_w \right]_{i,j} = \sum_{s=1}^N \left[K_p\right]_{s,i} \cdot \left[K_w\right]_{s,j} = \sum_{s=1}^N \phi\left( \frac{b_i^p - P_s}{h_p} \right) \cdot \phi\left( \frac{b_j^w - w_s}{h_w} \right).
\label{eq:kde_proof}
\end{equation}
Equation~\eqref{eq:kde_proof} is the exact mathematical definition of a standard \textbf{paired 2D product-kernel Kernel Density Estimator} on correlated bivariate data! The summation runs over concurrent sample index $s$, precisely preserving the empirical joint coupling between $P_s$ and $w_s$. 

This formal derivation provides an indispensable \textbf{Negative Control}: it proves that the independence allegation was mathematically false, and our audit protocol formally retracted it. This demonstrates that software-level auditing serves not to discredit authors, but to provide precise, verifiable characterization of codebases, separating genuine implementation discrepancies (the reverse-JS argument flip) from mathematical misconceptions.
""")

    # SECTION 7: DOWNSTREAM STRESS TESTING (TABLE 3 REPLACEMENT & FIG 4)
    sections.append(r"""\section{Exploratory Downstream Stress Testing: 3-Arm Guardrail Ablation}
\label{sec:stress_test}

To examine the operational implications of constraint lifecycle dimensions D2 (application stage) and D5 (harm reporting), we conducted controlled downstream stress tests on real-world industrial SCADA telemetry from a large-scale Chinese wind farm (Wind-CN, 121,528 continuous operational observations spanning one full calendar year, recorded at 15-minute resolution).

\subsection{Experimental Setup and 3-Arm Guardrail Design}
Prior studies frequently evaluate physical constraints by comparing complex, heterogeneous neural architectures (e.g., comparing a constrained custom GNN against an unconstrained standard LSTM), creating insurmountable causal confounds. To establish strict isolation, we employ a single, unified, modern linear time series backbone---the **DLinear** model with Series Decomposition \cite{zeng2023dlinear}---across all experimental conditions. The model takes a historical lookback window of $L=96$ steps (24 hours) of multivariate SCADA features (wind speed, wind direction, temperature, humidity, ambient pressure, and active power) and forecasts turbine power output over a prediction horizon of $H=24$ steps (6 hours).

We implement a rigorous \textbf{3-Arm Guardrail Design}:
\begin{itemize}[leftmargin=1.5em]
    \item \textbf{Arm 1 (No Guardrail)}: The standard DLinear model trained purely with Mean Squared Error (MSE) on normalized data: $\mathcal{L}_{\text{MSE}} = \frac{1}{BH}\sum (\hat{y} - y)^2$. At inference time, raw denormalized predictions $\hat{y}$ are evaluated without modification.
    \item \textbf{Arm 2 (Post-hoc Clipping)}: Utilizes the identical model weights trained in Arm 1, but at inference time, raw predictions are passed through an operational hard-clipping projection: $\hat{y}_{\text{clipped}} = \text{clip}(\hat{y}, 0, P_{\text{rated}})$, where $P_{\text{rated}} = 122,313.0\text{ kW}$ represents the observed physical capacity.
    \item \textbf{Arm 3 (Joint Guardrail)}: The DLinear backbone is trained with a multi-objective loss combining empirical MSE and a differentiable soft envelope violation penalty:
    \begin{equation}
    \mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{MSE}} + \lambda \, \mathbb{E}\left[ \text{ReLU}(y_{\min} - \hat{y})^2 + \text{ReLU}(\hat{y} - y_{\max})^2 \right],
    \label{eq:joint_guardrail}
    \end{equation}
    where $\lambda \in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]$. At inference time, we evaluate both raw predictions and clipped predictions.
\end{itemize}

\subsection{Experimental Protocol and Statistical Rigor}
To eliminate pseudo-reproducibility and benchmark fluke, every experimental configuration was trained and evaluated across \textbf{five independent random seeds} ($42, 123, 456, 789, 1024$) using a chronological 80/20 train/test split ($48,552$ training windows, $12,094$ test windows). To probe model behavior under industrial telemetry failure, we evaluated all models across four Gaussian sensor perturbation regimes: $\sigma \in \{0.0, 0.2, 0.4, 0.6\}$, where $\sigma=0.0$ represents clean test telemetry and $\sigma > 0$ models progressive sensor degradation and calibration drift. All results are reported with explicit mean and sample standard deviation ($\pm \text{std}$) across the 5 seeds.

\begin{table*}[t]
\centering
\small
\begin{threeparttable}
\caption{Downstream Evaluation of 3-Arm Guardrail Strategies Across Progressive Sensor Noise Regimes (Wind-CN SCADA, 5 Random Seeds).}
\label{tab:guardrail_results}
\begin{tabular}{llrrr}
\toprule
\textbf{Noise Regime} & \textbf{Strategy / Arm} & \textbf{RMSE (kW)} & \textbf{MAE (kW)} & \textbf{Violation Rate (\%)} \\
\midrule
\multirow{4}{*}{\textbf{Clean ($\sigma = 0.0$)}} 
& Arm 1: No Guardrail & $6151.3 \pm 48.7$ & $3907.4 \pm 39.1$ & $1.14\% \pm 0.31\%$ \\
& Arm 2: Post-hoc Hard Clipping & $\mathbf{6148.1 \pm 50.9}$ & $3937.6 \pm 35.4$ & $\mathbf{0.00\% \pm 0.00\%}$ \\
& Arm 3: Joint Guardrail (Raw, $\lambda=0.1$) & $6153.8 \pm 47.5$ & $3910.8 \pm 40.0$ & $1.13\% \pm 0.30\%$ \\
& Arm 3: Joint Guardrail (Clipped, $\lambda=0.1$) & $6151.0 \pm 49.7$ & $3941.0 \pm 36.2$ & $\mathbf{0.00\% \pm 0.00\%}$ \\
\midrule
\multirow{4}{*}{\textbf{Low Perturbation ($\sigma = 0.2$)}} 
& Arm 1: No Guardrail & $15211.7 \pm 92.3$ & $11821.5 \pm 70.9$ & $6.89\% \pm 0.33\%$ \\
& Arm 2: Post-hoc Hard Clipping & $14520.9 \pm 74.8$ & $11274.7 \pm 71.9$ & $\mathbf{0.00\% \pm 0.00\%}$ \\
& Arm 3: Joint Guardrail (Raw, $\lambda=0.1$) & $15179.7 \pm 91.4$ & $11797.0 \pm 70.2$ & $6.85\% \pm 0.33\%$ \\
& Arm 3: Joint Guardrail (Clipped, $\lambda=0.1$) & $\mathbf{14494.7 \pm 74.0}$ & $\mathbf{11254.5 \pm 71.0}$ & $\mathbf{0.00\% \pm 0.00\%}$ \\
\midrule
\multirow{4}{*}{\textbf{Moderate Perturbation ($\sigma = 0.4$)}} 
& Arm 1: No Guardrail & $28381.6 \pm 193.2$ & $22249.6 \pm 148.6$ & $16.64\% \pm 0.48\%$ \\
& Arm 2: Post-hoc Hard Clipping & $24899.8 \pm 321.7$ & $19464.5 \pm 258.5$ & $\mathbf{0.00\% \pm 0.00\%}$ \\
& Arm 3: Joint Guardrail (Raw, $\lambda=0.1$) & $28311.8 \pm 191.8$ & $22196.5 \pm 147.9$ & $16.59\% \pm 0.48\%$ \\
& Arm 3: Joint Guardrail (Clipped, $\lambda=0.1$) & $\mathbf{24853.4 \pm 319.5}$ & $\mathbf{19429.4 \pm 256.7}$ & $\mathbf{0.00\% \pm 0.00\%}$ \\
\midrule
\multirow{4}{*}{\textbf{Severe Perturbation ($\sigma = 0.6$)}} 
& Arm 1: No Guardrail & $41921.2 \pm 345.4$ & $32896.1 \pm 266.4$ & $25.89\% \pm 0.39\%$ \\
& Arm 2: Post-hoc Hard Clipping & $33514.5 \pm 171.7$ & $26302.4 \pm 173.1$ & $\mathbf{0.00\% \pm 0.00\%}$ \\
& Arm 3: Joint Guardrail (Raw, $\lambda=0.1$) & $41814.3 \pm 344.8$ & $32815.2 \pm 266.7$ & $25.81\% \pm 0.39\%$ \\
& Arm 3: Joint Guardrail (Clipped, $\lambda=0.1$) & $\mathbf{33459.3 \pm 171.1}$ & $\mathbf{26259.3 \pm 172.8}$ & $\mathbf{0.00\% \pm 0.00\%}$ \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Backbone architecture is DLinear (lookback 96, horizon 24). Results represent mean $\pm$ standard deviation across 5 independent random seeds. Violation Rate measures the percentage of predicted points violating $[0, P_{\text{rated}}]$.
\end{tablenotes}
\end{threeparttable}
\end{table*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.75\textwidth]{fig_guardrail_pareto.pdf}
\caption{Pareto Trade-off Frontier of Prediction Error (RMSE) vs. Physical Violation Rate (\%) Across Penalty Weight $\lambda \in [0.0, 1.0]$ under Clean ($\sigma=0.0$), Moderate ($\sigma=0.4$), and Severe ($\sigma=0.6$) Sensor Perturbations. On clean data, varying $\lambda$ yields negligible impact; under noise, increasing $\lambda$ monotonically suppresses catastrophic violations and reduces error.}
\label{fig:pareto_curve}
\end{figure*}

\subsection{Key Empirical Insights}
The quantitative results in Table~\ref{tab:guardrail_results} and Figure~\ref{fig:pareto_curve} illuminate three vital operational realities:
\begin{enumerate}[label=(\arabic*)]
    \item \textbf{No Accuracy Enhancement on Clean Telemetry}: On clean benchmark data ($\sigma=0.0$), the unconstrained baseline (Arm 1, RMSE $= 6151.3 \pm 48.7\text{ kW}$) and the joint physical guardrail (Arm 3, RMSE $= 6153.8 \pm 47.5\text{ kW}$) achieve virtually indistinguishable accuracy (a statistically insignificant difference of $2.5\text{ kW}$, or $< 0.05\%$). This rigorous ablation directly refutes the common claim that physical loss penalties inherently improve clean out-of-sample accuracy.
    \item \textbf{Catastrophic Unconstrained Failure Under Noise}: In the absence of guardrails (Arm 1), input sensor perturbation severely destabilizes model predictions. Under severe noise ($\sigma=0.6$), \textbf{25.89\% of all predicted points violate physical boundaries}, producing massive negative generation or runaway power spikes that blow out RMSE to $41,921.2\text{ kW}$.
    \item \textbf{Guardrails as Defensive Envelopes}: Post-hoc hard clipping (Arm 2) immediately eliminates 100\% of physical violations by construction, slashing severe RMSE from $41,921.2\text{ kW}$ to $33,514.5\text{ kW}$---an error reduction of \textbf{$8,406.7\text{ kW}$ (20.1\%)} achieved with zero additional training cost. Furthermore, joint training with soft violation penalties (Arm 3) regularizes model weights, yielding further monotonic improvements as $\lambda$ increases (Figure~\ref{fig:pareto_curve}), reaching an optimal clipped RMSE of $32,969.9 \pm 169.0\text{ kW}$ at $\lambda=1.0$.
\end{enumerate}
These findings provide definitive empirical evidence that physical constraints function not as competitive accuracy boosters on standard leaderboards, but as essential \textbf{defensive guardrails} that prevent catastrophic failure under operational stress.
""")

    # SECTION 8: DISCUSSION & REPORTING CHECKLIST
    sections.append(r"""\section{Discussion and the Constraint Lifecycle Reporting Checklist}
\label{sec:discussion}

The empirical findings from our scoping field audit, static code verification, and downstream stress testing converge upon a coherent methodological conclusion: the fundamental challenge facing physics-constrained deep learning in renewable energy is not a deficit of domain mathematical equations, but a \textbf{deficit of evaluation discipline and reporting transparency}.

\subsection{Synthesis of Research Questions}
We synthesize our findings around the core research questions established in the blueprint:
\begin{itemize}[leftmargin=1.5em]
    \item \textbf{RQ1 (Taxonomic Representation)}: By formalizing the five-dimensional Constraint Lifecycle Framework (Schema v4), we demonstrate that physical constraints cannot be treated as undifferentiated monolithic labels. Disentangling parameter provenance (D1), application stage (D2), inference telemetry (D3), code fidelity (D4), and ablation semantics (D5) provides the necessary conceptual vocabulary to evaluate where physical priors originate, where they enter computation, and what operational burdens they impose.
    \item \textbf{RQ2 (Field Reporting Practices)}: Within the audited 46-study scoping corpus, current reporting practices exhibit severe causal confounding: 47.8\% of reported ablations fail to isolate the physical constraint itself, 0.0\% of studies report negative trade-offs or track physical violation rates, and 97.8\% of published methods provide no accessible source code.
    \item \textbf{RQ3 (Specification-to-Code Parity)}: Our static case study of the sole public repository (PCT \cite{pct2026}) establishes that mathematical specifications can easily diverge during software implementation (the reverse-JS divergence caused by PyTorch's \texttt{F.kl\_div} argument ordering). Simultaneously, our algebraic derivation of 2D product-kernel KDE equivalence proves that code-level audits can successfully provide negative controls that exonerate valid implementations from ungrounded theoretical critiques.
    \item \textbf{RQ4 (Operational Value of Guardrails)}: Controlled stress testing on industrial SCADA data across five random seeds demonstrates that physical constraints do not enhance clean-data accuracy, but act as indispensable defensive guardrails that reduce prediction error by over 20\% under severe sensor perturbation.
\end{itemize}

\subsection{The Constraint Lifecycle Reporting Checklist}
To elevate empirical standards across the renewable energy and scientific machine learning communities, we propose the \textbf{Constraint Lifecycle Reporting Checklist} (Table~\ref{tab:checklist}). Modeled after successful reporting standards in general machine learning and computational medicine \cite{pineau2021,tricco2018}, this ten-item checklist provides actionable guidelines for authors, peer reviewers, and journal editors.

\begin{table*}[t]
\centering
\small
\caption{The Ten-Item Constraint Lifecycle Reporting Checklist for Physics-Constrained Machine Learning in Energy Applications.}
\label{tab:checklist}
\begin{tabular}{lp{3.5cm}p{10.5cm}}
\toprule
\textbf{Item} & \textbf{Checklist Category} & \textbf{Reporting Recommendation and Minimum Verification Requirement} \\
\midrule
\textbf{C1} & \textbf{Provenance Declaration} & Explicitly specify whether constraint parameters are intrinsic analytical constants, empirical estimates from historical training data, or learned online. If empirical, report calibration data splits and estimation confidence. \\
\textbf{C2} & \textbf{Stage Disambiguation} & Clearly declare whether the constraint operates as a soft training-loss penalty, an architectural forward constraint, data augmentation, or post-hoc correction. State clearly if non-physical predictions remain possible during inference. \\
\textbf{C3} & \textbf{Telemetry Transparency} & Itemize all external data required during real-time inference (e.g., NWP forecasts, adjacent turbine SCADA). Explicitly document model sensitivity to telemetry communication latency or missing sensor signals. \\
\textbf{C4} & \textbf{Strict Physical Isolation} & Mandate an isolated ablation comparing the full constrained model against an unconstrained baseline ($\lambda=0$ vs. $\lambda > 0$) under strictly identical neural architecture, parameter count, and training budget. \\
\textbf{C5} & \textbf{Software Availability} & Provide a persistent public repository link (e.g., Zenodo DOI or GitHub URL with pinned commit hash) containing functional training, inference, and evaluation scripts with dependency specifications. \\
\textbf{C6} & \textbf{Software-Paper Parity} & Explicitly verify that loss functions and numerical operators in released code execute the identical mathematical formulation specified in the manuscript text (e.g., verifying divergence argument order in ML frameworks). \\
\textbf{C7} & \textbf{Harm \& Trade-off Audit} & Systematically track and report potential performance penalties, including accuracy trade-offs on clean data, computational overhead, and convergence stability under varying constraint penalty weights ($\lambda$). \\
\textbf{C8} & \textbf{Violation Tracking} & Accompany standard point-error metrics (RMSE, MAE) with domain-specific physical violation metrics (e.g., violation rate \%, maximum envelope excess kW, or non-physical ramp occurrences). \\
\textbf{C9} & \textbf{Robustness Stress Testing} & Evaluate model performance beyond clean historical test splits by injecting controlled sensor noise, missing values, or out-of-distribution operating conditions to probe defensive guardrail boundaries. \\
\textbf{C10} & \textbf{Multi-Seed Uncertainty} & Report all performance and violation metrics with explicit sample standard deviations across a minimum of five independent random seeds to prevent fluke reporting. \\
\bottomrule
\end{tabular}
\end{table*}
""")

    # SECTION 9: THREATS TO VALIDITY & LIMITATIONS
    sections.append(r"""\section{Threats to Validity and Limitations}
\label{sec:limitations}

In accordance with rigorous empirical standards \cite{lipton2019,kapoor2023}, we transparently document the methodological threats to validity and boundary conditions of this study:

\subsection{Scoping Sample and Literature Retrieval Boundaries}
Our field audit operates as an exploratory scoping review adhering to PRISMA-ScR guidelines rather than an exhaustive systematic meta-analysis of all historical energy literature. While our multi-query search across OpenAlex and supplementary discovery captured 89 candidate records and yielded 46 eligible full-text studies published between 2023 and 2026, the corpus is restricted to English-language peer-reviewed literature and indexed preprints. Relevant studies published in other languages, regional conference proceedings, or proprietary industrial technical reports may exist that were not indexed under our recorded query strings. Consequently, our quantitative percentages describe prevailing patterns within the identified scoping corpus, and should not be extrapolated as universal population parameters of the global discipline.

\subsection{Single-Rater Full-Text Integration and Rater Agreement}
Full-text coding across all 46 included studies was conducted by a primary expert researcher (Juntao Du), supported by independent double-coding on a pilot calibration subset to establish baseline inter-rater agreement ($\kappa \ge 0.74$). While every recorded code is tied to verbatim textual excerpts, equation numbers, and section identifiers in our publicly released audit ledger, full-text coding across the entire corpus was not conducted by two independent blinded raters across all 46 papers. Future expanded audits should deploy multi-institutional rater cohorts to achieve full double-blind consensus.

\subsection{Code Search Protocol Boundaries}
Our finding that 45 out of 46 studies (97.8\%) provided no public code is classified strictly as \texttt{not-found} under our standardized five-step software discovery protocol as of the audit cutoff date (March 1, 2026). This does not prove that code does not exist in private corporate repositories or that authors would not share code upon individual private correspondence. However, from the perspective of open science and public verifiability, the absence of discoverable public code remains a profound barrier to scientific reproducibility.

\subsection{Single Codebase Case Study Boundary}
Our code-level software audit focused in depth on the Physics-Constrained Transformer (PCT \cite{pct2026}) because it was the sole publicly available codebase identified in the corpus. While the reverse-JS implementation discrepancy provides a compelling demonstration that software-paper divergences can occur in top-tier published works, this single case study cannot be generalized to infer that code mismatches are ubiquitous across other unreleased implementations. Furthermore, in strict adherence to our non-claim boundary, we did not run dynamic downstream retraining of PCT to claim specific RMSE degradation, confining our audit to verifiable static mathematical facts.

\subsection{Downstream Stress Testing Scope}
Our downstream empirical stress tests were conducted on real-world industrial SCADA data from a single large-scale Chinese wind farm (Wind-CN) utilizing the DLinear architecture. While the multi-seed protocol ($\times 5$ seeds) and progressive noise sweep provide statistically rigorous evidence of the defensive guardrail mechanism, the precise numerical trade-off slopes may vary across differing turbine technologies, complex terrains, and neural backbones.
""")

    # SECTION 10: CONCLUSION & BACKMATTER
    sections.append(r"""\section{Conclusion}
\label{sec:conclusion}

Physics-constrained deep learning holds tremendous promise for transforming renewable energy forecasting from brittle black-box fitting into reliable, physically grounded engineering analytics. Yet, as this scoping empirical audit has demonstrated, the field's rapid expansion has outpaced its evaluation discipline. By treating ``physics-constrained'' as an undifferentiated label, conflating architectural submodules with physical priors in ablation studies, and failing to release or audit software implementations, published literature risks propagating an unverified narrative that domain physics provides a universal, cost-free accuracy panacea.

Through the formalization of the five-dimensional Constraint Lifecycle Framework (Schema v4), the quantitative scoping audit of 46 recent studies, the static verification and negative control of the PCT codebase, and multi-seed downstream guardrail stress testing, this work replaces qualitative enthusiasm with transparent, reproducible empirical evidence. Physical constraints achieve their highest engineering value not by competing for fractional basis points on clean leaderboards, but by serving as robust defensive envelopes that safeguard critical energy infrastructure against operational failure. We hope the institutional Constraint Lifecycle Reporting Checklist established herein will serve as a valuable catalyst for more transparent, rigorous, and impactful scientific inquiry across the energy computing community.

\section*{Data and Code Availability}
To ensure complete transparency and full public verifiability, all artifacts supporting this study are permanently archived in the project repository. These include: (1) the study-level scoping retrieval ledger (\texttt{study\_level\_ledger\_48.csv}); (2) the full-text multi-dimensional coding table (\texttt{corpus48\_integrated\_audit\_table.csv}); (3) the PRISMA-ScR screening logs and coding codebook; (4) the static mathematical verification scripts for the PCT case study; (5) the complete PyTorch implementation of the 3-Arm Guardrail benchmark across all five seeds and noise regimes; and (6) all generated raw CSV logs and publication-ready vector figures.

\section*{Declaration of Competing Interest}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Acknowledgments}
The authors gratefully acknowledge the computational facilities provided by the High Performance Computing Center and the invaluable feedback from participating domain reviewers during preliminary audit calibration.

\bibliography{cas-refs}

\end{document}
""")

    full_content = "\n".join(sections)
    out_path = Path("02_论文正文与LaTeX/manuscript_audit_final.tex")
    out_path.write_text(full_content, encoding="utf-8")
    print(f"Generated complete manuscript at {out_path} ({len(full_content)} bytes)")

if __name__ == "__main__":
    generate_manuscript()
