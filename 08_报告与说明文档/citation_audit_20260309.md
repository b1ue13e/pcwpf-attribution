# Citation Audit

- Source manuscript: `main_PI-MoE_final_AE_ready.tex`
- Bibliography source used for regeneration: `cas-refs.bib`
- Regenerated cited-only database target: `refs_cited_only.bib`
- Audit date: 2026-03-09

## Counts

- Total cited unique entries: `51`
- Intro raw unique citations: `22`
- Related-work raw unique citations: `29`
- Discussion raw unique citations: `14`
- Intro + related formal-journal share from `2020-2025`: `27 / 29 = 93.1%`
- Total non-article cited entries: `5`

## Hard Rule

- Audit rule for the rewritten narrative:
  every problem-building paragraph in `Section 1` and `Section 2` must carry at least one core judgment with formal journal support.
- Current status:
  satisfied for all five introduction paragraphs, the Section 2 framing paragraph, and all six related-work body paragraphs.
- Narrative-polish rule for this pass:
  summary-style paragraph endings were removed or rewritten so that paragraph closure now lands on literature-based limitations or test conditions rather than on abstract labels.
- Current status:
  `Section 2` no longer ends paragraphs with lines such as `What remains less resolved...`, `These two conversations do not yet line up cleanly...`, `The open problem...`, `Read together...`, `That creates...`, `What is still tested less often...`, or `Less often reported is whether...`; the final two body paragraphs now close on concrete literature contrasts and the specific evaluation condition left thinly covered by prior work.
- Claim-discipline rule for this pass:
  the manuscript positions Ultra-LSNT as a balanced controller-side solution under corrupted SCADA rather than as the top model across all ranking tables.
- Current status:
  the abstract, introduction bridge, dispatch interpretation, and conclusion now frame Ultra-LSNT through robustness, sparse edge profile, feasibility discipline, and shared-interface comparative value, while preserving that Chronos/Moirai lead the severe context panel and iTransformer leads the shared-interface L1/fallback dispatch screen.
- Soft-residue rule for this pass:
  narrator-style transition phrasing such as `To address...`, `To judge whether...`, and `Read that way...` should be removed from the highest-claim sentences.
- Current status:
  the introduction bridge now keeps one research-question sentence plus two response sentences without narrator-guided transitions, the abstract now closes on robustness and feasibility evidence only, and the dispatch-results paragraph now turns on interface dependence rather than on author-summary guidance.
- Main-text protocol-prose rule for this pass:
  Section 4 should keep its technical content while dropping `protocol card` phrasing and instruction-style lead-ins where the main text can carry the same point as controlled-comparison rationale.
- Current status:
  the `Experimental setup and robustness protocol` section now frames the shared pipeline, baseline controls, and dispatch screening through common-footing comparison language rather than through protocol-card labels or reader-instruction phrasing; appendix protocol sections remain untouched.

## Section 1 Narrative Map

- `P1` forecast-to-decision consequences
  Core judgment: wind forecast error propagates into reserve, curtailment, balancing, and cost.
  Support: `khaloie2020coordinated`, `houben2023optimal`, `li2024opf_highren`, `yan2025economicvalue`, `masood2025economic`
- `P2` SCADA reality
  Core judgment: operational SCADA is not a clean benchmark stream; missing intervals, duplicates, calibration shifts, and change points distort forecasting assumptions.
  Support: `leahy2019dataquality`, `letzgus2020`, `renstrom2020autoencoder`, `martincalzada2025optimized`
- `P3` edge reality
  Core judgment: controller-side deployment is batch-1, latency-limited, and memory-limited, so active-path size becomes a model-design issue.
  Support: `gooi2023edgeintelligence`, `mwangi2024sdniiotedge`
- `P4` partial solutions, still misaligned
  Core judgment: decomposition, deep forecasting, and physics-guided work each solve part of the problem, but usually under different evaluation assumptions.
  Support: `verdone2025review`, `zhang2021windreview`, `duan2025robust`, `zhu2025realtime`, `ma2022gcnwind`, `gao2025mmgpt`, `wu2024stellm`, `liu2024physics`, `parsa2025physics`, `wang2025review`
- `P5` research question and contribution bridge
  Core judgment: the paper is motivated by the joint pressure of operational value, corrupted SCADA, and deployable feasibility rather than by a single benchmark gap.
  Support: `yan2025economicvalue`, `sheng2025annreview`, `liu2024physics`
  Manuscript response: one research-question sentence followed by two compact response sentences; no interface, appendix, or mapping detail retained in the introduction.

## Section 2 Problem Map

- `Framing paragraph`
  Core judgment: the key literature comes from two adjacent problem settings rather than from a single model taxonomy.
  Support: `verdone2025review`, `parsa2025physics`, `yan2025economicvalue`
  Section title status: parent heading is now `Related work`, matching the problem-driven structure.

- `2.1 Operational SCADA Forecasting Under Corruption and Resource Constraints`
  Paragraph 1:
  Core judgment: most SCADA literature establishes that telemetry must be filtered and segmented before it supports reliable analytics.
  Support: `maldonadocorrea2020slr`, `sheng2025annreview`, `letzgus2020`, `renstrom2020autoencoder`, `vervlimmeren2025scalable`, `martincalzada2025optimized`
  Paragraph 2:
  Core judgment: non-stationary wind forecasting benefits from decomposition and deep representation learning, but many evaluations assume cleaned or reconstructed inputs or omit controller-side compute limits.
  Support: `verdone2025review`, `zhang2021windreview`, `duan2025robust`, `niu2020ceemdanbi`, `zhu2025realtime`, `ma2022gcnwind`, `gao2025mmgpt`, `wu2024stellm`
  Paragraph 3:
  Core judgment: literature is much thinner on forecasters that ingest corrupted live SCADA under field-device resource limits than on diagnostics built after stabilization or forecasting on cleaned signals.
  Support: `gooi2023edgeintelligence`, `mwangi2024sdniiotedge`

- `2.2 Feasibility-Aware Forecasting and Decision-Coupled Value`
  Paragraph 1:
  Core judgment: feasibility control spans classical turbine envelopes, cleaning-oriented operating-region reconstruction, and newer training-time physical constraints.
  Support: `astolfi2015datamining`, `martincalzada2025optimized`, `liu2024physics`, `parsa2025physics`, `liao2024trust`, `wang2025review`
  Paragraph 2:
  Core judgment: forecast accuracy does not transfer directly into dispatch value because value is mediated by the scheduling model and forecast-to-grid interface.
  Support: `khaloie2020coordinated`, `houben2023optimal`, `li2024opf_highren`, `liu2024vscopf`, `yan2025economicvalue`, `khaloie2025opfml`
  Paragraph 3:
  Core judgment: studies that train feasibility constraints from deployment-available signals and then compare the resulting forecasts through one common downstream interface remain rare across the split between physical-discipline and dispatch-value work.
  Support: `liu2024physics`, `liao2024trust`, `li2024opf_highren`, `yan2025economicvalue`

## Main Literature Backbone

- Operational value / forecast-to-decision:
  `khaloie2020coordinated`, `houben2023optimal`, `li2024opf_highren`, `liu2024vscopf`, `yan2025economicvalue`, `masood2025economic`, `khaloie2025opfml`
- SCADA quality / anomaly filtering / preprocessing:
  `maldonadocorrea2020slr`, `sheng2025annreview`, `letzgus2020`, `renstrom2020autoencoder`, `vervlimmeren2025scalable`, `martincalzada2025optimized`, `leahy2019dataquality`
- Non-stationary and deep forecasting:
  `verdone2025review`, `zhang2021windreview`, `duan2025robust`, `niu2020ceemdanbi`, `zhu2025realtime`, `ma2022gcnwind`, `gao2025mmgpt`, `wu2024stellm`
- Physics-guided / trustworthy / uncertainty-aware forecasting:
  `liu2024physics`, `parsa2025physics`, `liao2024trust`, `wang2025review`, `ellinas2025trustworthy`
- Edge deployment / controller constraints:
  `gooi2023edgeintelligence`, `mwangi2024sdniiotedge`

## Background / Standards / Classical Theory

- Kept only for definitions or classical background:
  `burton2011wind`, `iec61400_12_1_2017`, `jacobs1991moe`, `dragomiretskiy2014vmd`, `wu2009eemd`

## Experimental / Protocol / Evaluation Only

- Kept for dispatch setup, datasets, significance testing, or implementation framing rather than intro/related backbone:
  `botterud2011operational`, `draxl2015windtoolkit`, `devito2008`, `Hong2014GEFCom2012`, `diebold1995`, `schwartz2020greenai`, `strubell2019energy`, `dlt5002_2021`, `li2023vmdcnn`

## Downgraded Or Removed From Main Chain

- Removed from the introduction / related-work backbone:
  `ansari2024chronos`, `woo2024moirai`, `liu2024moiraimoe`, `shi2025timemoe`
- Sector reports and similar non-journal items also kept out of the main narrative:
  `iea2023`, `windeurope2023`

## Notes

- `Section 1` is now narrative and problem-building; it no longer carries interface, appendix, or mapping detail.
- `Section 2` is compressed into two question blocks, but each block is split into short natural paragraphs to avoid long review walls.
- The current polish pass thickened `Introduction` paragraphs 1 and 2, then replaced both hard-template endings and softer narrator-summary endings in `Section 2` with literature-friction closures.
- The current claim-tightening pass aligns the manuscript claim with a balanced controller-side positioning rather than an absolute-best framing.
- The current local-sentence pass removes dispatch from the abstract, compresses the introduction bridge, and de-hardens Section 4 main-text protocol prose without changing appendices or technical content.
- Chronos / Moirai remain experimental comparators only and do not support controller-side deployability claims in the rewritten narrative.
