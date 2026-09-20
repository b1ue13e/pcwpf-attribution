# Graphical Abstract Design & Visual Specification

**Manuscript Title:** Conditional Inductive Guardrails: Identifying When Physical Priors Enhance, Mimic, or Impair Decisions in Wind Power Forecasting  
**Target Journal:** *Applied Energy* (Elsevier)  
**Target Format:** High-resolution single-panel vector graphic (PDF / TIFF / PNG, 300+ DPI, RGB/CMYK, landscape orientation, aspect ratio 16:9 or 2:1)  

---

## 1. Overall Architectural Concept: A Three-Panel Evidence Flow

The Graphical Abstract follows a clear left-to-right cognitive progression:
```
[ Panel 1: Current Literature Bias ]  ==>  [ Panel 2: Controlled Identification ]  ==>  [ Panel 3: Operational Decision Envelope ]
  (100% Positive Reports vs. Void)           (Form-Matched Controls & Budgets)             (Guardrails, Mimicry & Decoupling)
```

---

## 2. Detailed Panel Specifications

### Panel 1: The Status Quo & The Governance Deficit (Left Panel, 30% Width)
- **Header:** *The Status Quo: Unchecked Confounding in Physics-Informed ML*
- **Visual Elements:**
  - A stylized pyramid / stack representing the **$N=46$ Contemporary Audited Studies (2023–2026)**.
  - **Upper Callout Badge (Green):** *100.0% Uniform Positive Claims* (46/46 papers reporting accuracy improvements).
  - **Lower Reality Ledger (High-Contrast Amber/Crimson Badges):**
    - *Failure Regime Disclosure:* **0.0%** (0/46 studies report where physical constraints degrade or fail).
    - *Open Software Reproducibility:* **4.3%** public code (95.7% code concealment rate, 44/46).
    - *Ablation Confounding:* **50.0%** (23/46 conflate neural capacity expansion with physical domain knowledge).
  - **Connecting Transition Arrow:** Red dashed arrow pointing to the middle panel with the label: *"How do we isolate genuine physical prior value from numerical artifacts?"*

### Panel 2: Controlled Attribution & Identification Protocol (Center Panel, 35% Width)
- **Header:** *Controlled Identification Protocol (Holding Capacity & Splits Constant)*
- **Visual Elements:**
  - **The 2×2 Negative Control Crucible:**
    - Branch A: **Authentic Physical Prior ($C_{\text{emp}}$):** Fitted empirical power curve with monotonic aerodynamic geometry.
    - Branch B: **Pseudo-Scrambled Control ($C_{\text{scramble}}$):** Form-, weight-, and capacity-matched mapping with permuted speed-power coordinates.
  - **Information-Budget Matching Box ($N_{\text{prior}} = N_{\text{train}} = 500$):**
    - Showing that domain geometry preserves 77% to 92% of downside protection without needing multi-month archival data.
  - **Predefined Dual Stress Axes:**
    - Penalty Scale Grid: $\lambda \in \{0.01, 0.1, 1.0\}$ (separating smoothing from aerodynamic reasoning).
    - Proxy Staleness Ladder: Weather proxy delay $k \in [1\text{h}, 6\text{h}]$.
  - **Connecting Transition Arrow:** Blue solid arrow pointing to the right panel with the label: *"Empirical Operating Envelope Across 3 Operational Wind Farms"*

### Panel 3: The Operational Decision Envelope & Decoupling (Right Panel, 35% Width)
- **Header:** *The Decision Envelope: When Do Priors Help, Mimic, or Impair?*
- **Visual Elements:**
  - **Regime 1: Sample-Restricted Guardrail ($N_{\text{train}} \le 1{,}500$):**
    - Authentic Prior: **$-1.28\%$** Downside Dispatch Risk Reduction.
    - Scrambled Prior: **$+61.2\%$** Cost Surge (Catastrophic Prior Misspecification Penalty).
  - **Regime 2: Weak-Penalty Numerical Mimicry ($\lambda = 0.01$):**
    - Authentic ($0.08803$) $\approx$ Scrambled ($0.08792$). Indistinguishable metrics prove non-specific numerical smoothing.
  - **Regime 3: Large-Sample Inversion ($N_{\text{train}} \ge 5{,}000$):**
    - Unconstrained data-driven backbone matches or slightly surpasses fixed priors ($+0.33\%$ to $+1.73\%$), exposing structural bias.
  - **Regime 4: Prediction-Decision Decoupling Box (Crimson Accent):**
    - *Zero-Bid Diagnostic ($q \equiv 0$):* Standalone unhedged cashflows generate £239k/(turb$\cdot$yr) without forecasting skill, proving that point RMSE and unhedged cashflows decouple from true system balancing risk.

---

## 3. Visual Styling, Typography & Color Palette

- **Dimensions:** 16:9 aspect ratio, 1920 × 1080 pixels (or vector PDF equivalent, 14.0 in × 7.8 in).
- **Color Palette (Nature / Elsevier Morandi Style):**
  - Primary Identity Blue: `#2b6cb0` (Unconstrained baselines & official branding)
  - Authentic Prior Teal / Blue: `#0072B2` (Empirical power curve prior)
  - Scrambled Prior Crimson: `#D55E00` / `#DC2626` (Negative control and prior misspecification)
  - Market Cashflow Amber: `#dd6b20` (Balancing market cashflows)
  - Guardrail Forest Green: `#22543d` (Inductive protection zone)
  - Card Background: `#F8FAFC` (Clean slate container fills)
  - Border Gray: `#CBD5E1` (Crisp container borders)
- **Typography:**
  - Primary Font: DejaVu Sans / Arial (Sans-serif, 100% vector-embedded).
  - Title & Panel Headers: 12–14 pt Bold.
  - Callout Metrics: 10–11 pt Bold.
  - Descriptive Subtext: 8.5–9.5 pt Regular.
- **Delivery Formats:**
  - Vector PDF: `graphical_abstract.pdf` (PDF-1.4 normalized, zero rasterization artifacts).
  - High-Resolution Raster: `graphical_abstract.png` (300 DPI, RGB, lossless compression).
