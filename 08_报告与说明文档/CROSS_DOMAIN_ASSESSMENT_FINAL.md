# Cross-Domain Experiment: Honest Assessment

## Executive Summary

After completing comprehensive cross-domain experiments (Wind CN → Wind US domain adaptation), the data confirms a fundamental structural problem: **Ultra-LSNT is not the strongest backbone under any tested scenario.**

---

## Key Finding: iTransformer Dominates All Data Regimes

| Data Ratio | iTransformer R² | Ultra-LSNT R² | Gap (U - I) | Winner |
|------------|----------------|---------------|-------------|--------|
| 1%         | 0.683          | 0.536         | **-0.147**  | iTransformer |
| 5%         | 0.858          | 0.846         | **-0.012**  | iTransformer |
| 10%        | 0.881          | 0.861         | **-0.020**  | iTransformer |
| 50%        | 0.896          | 0.892         | **-0.004**  | iTransformer |
| 100%       | 0.896          | 0.889         | **-0.007**  | iTransformer |

**Verdict:** Even at the optimal data ratio (5%), iTransformer outperforms Ultra-LSNT by 0.012 R². The gap widens significantly at low data (1%).

---

## Three Fatal Problems: No Data Solution

### 1. Structural Contradiction (FATAL - No Fix Possible)

**Problem:** Paper admits Ultra-LSNT is "not the strongest backbone" but occupies headline methodology space (MoE routing, dual propagation) while iTransformer (actual best performer) gets 1 paragraph.

**Attempted Solutions:**
- ✗ Domain adaptation experiments (completed): Ultra-LSNT still loses
- ✗ Guardrail ablation (completed): Confirms FG-MoE improves safety but NOT R² ranking
- ✗ Cross-site pretraining (completed): No path for Ultra-LSNT to surpass iTransformer

**Reality:** No experiment design can make Ultra-LSNT surpass iTransformer without either:
1. Unfair comparison (different conditions for each model)
2. Major architectural changes (new paper scope)
3. Cherry-picking specific failure cases for iTransformer

**Acceptable Narrowing:** Frame paper as "Feasibility-Guided Mixture-of-Experts for SCADA-Corrupted Wind Forecasting" with Ultra-LSNT as "audited sparse carrier" providing deployment timing evidence. **Accept that this is a bounded engineering contribution, not a fundamental breakthrough.**

---

### 2. Operational Evidence Weakness (FATAL - No Fix Possible)

**Problem:** 1.4% cost reduction (30.88 → 30.44 M CNY) lacks:
- Economic significance reference (is 0.44 M CNY meaningful to grid operators?)
- AC feasibility validation (DC OPF only, copper-plate assumption)
- Market pricing mechanism (no real-time market data)

**Attempted Solutions:**
- ✗ IEEE-RTS24 dispatch mapping (completed): Still DC OPF
- ✗ Rolling horizon analysis (completed): Same 1.4% magnitude
- ✗ Lambda sensitivity (completed): Confirms narrow operating range

**Reality:** Fixing this requires:
1. AC OPF solver integration (MATPOWER/PandaPower) - outside current code scope
2. Real-time market data (PJM/ERCOT LMPs) - no data access
3. Statistical significance testing (t-test on cost differences) - can't fix significance with same effect size

**Acceptable Narrowing:** Frame as "proof-of-concept for forecast-feedback dispatch optimization" with explicit limitation: "DC OPF with fixed copper-plate assumption; AC feasibility and market pricing reserved for future work."

---

### 3. External Validity Gap (FATAL - No Fix Possible)

**Problem:** Single-site (Wind CN) main evidence. Wind US framed as "recalibration boundary" (i.e., failure case), not positive validation.

**Data Evidence:**
- Wind US zero-shot: R² = -0.045 (complete failure)
- After full recalibration (100% data): R² = 0.889 (still loses to iTransformer's 0.896)

**Attempted Solutions:**
- ✗ Standardization comparison (completed): Different normalization doesn't fix architecture gap
- ✗ Domain adaptation (completed): Requires 50%+ data to approach iTransformer performance
- ✗ Guardrail transfer (completed): Physical constraints transfer but don't improve accuracy

**Reality:** Cannot claim "external validity" with Wind US as currently framed. Two options:
1. **Acquire additional sites:** Multiple CN farms + multiple US farms → actual generalization claim
2. **Reframe honestly:** "Single-site operational validation with cross-site boundary analysis"

**Acceptable Narrowing:** Wind US explicitly framed as "stress test of calibration boundaries" rather than "generalization validation." Paper scope: single-site (Inner Mongolia) with explicit limitations.

---

## What CAN Be Fixed (With Remaining Cloud Time)

### Medium Priority: Contribution Focus Clarity

**Problem:** FG-MoE vs Jump Gate vs Ultra-LSNT sparse carrier distinction still muddy.

**Fixable With:**
1. Add "Methodology Hierarchy" figure showing:
   - Level 1: FG-MoE guardrail mechanism (main contribution)
   - Level 2: Jump Gate routing (implementation detail)
   - Level 3: Ultra-LSNT backbone (audited carrier)

2. Revise Abstract/Introduction to foreground FG-MoE, background Ultra-LSNT.

**Time Required:** 1-2 hours (writing, not compute)

---

## Recommended Strategy: Honest Narrowing

### Option A: Minimal Revision (Recommended)

Accept all three limitations explicitly:

1. **Backbone:** "Ultra-LSNT is not the strongest forecaster; we select it for sparse activation auditability, not raw accuracy supremacy."

2. **Operation:** "DC OPF proof-of-concept with fixed interface mapping; 1.4% cost reduction demonstrates feasibility-guidance value under copper-plate assumptions."

3. **Generalization:** "Single-site operational validation (Inner Mongolia); Wind US recalibration establishes boundary conditions, not external validity."

**Risk:** Reviewers may reject as "too limited." 
**Benefit:** Cannot be accused of overselling.

### Option B: Major Scope Expansion (Not Recommended)

Requirements:
- 3+ wind farm sites from different regions
- AC OPF integration
- Real-time market pricing data
- Architectural changes to Ultra-LSNT to surpass iTransformer

**Timeline:** 3-6 months, new experiments, new data collection.

---

## Final Recommendation

**Do NOT run more cloud experiments.** The cross-domain bundle has generated sufficient evidence to confirm the structural limitations. Additional compute will not:
- Make Ultra-LSNT surpass iTransformer
- Create AC OPF feasibility data
- Generate multi-site evidence

**Spend remaining cloud time on:**
1. Response letter writing (honest limitation framing)
2. Manuscript revision (foregrounding FG-MoE, backgrounding Ultra-LSNT)
3. Table/Figure updates to reflect bounded claims

The paper can still be salvageable as a **bounded engineering contribution** demonstrating feasibility-guided forecasting for SCADA-corrupted wind power. But it cannot claim to be a fundamental breakthrough in time series forecasting or operations research.

---

*Assessment completed: 2026-03-30*
*Cross-domain experiments confirm: No data path to resolve three fatal structural issues.*
