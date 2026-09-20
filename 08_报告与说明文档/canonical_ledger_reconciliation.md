# Canonical literature ledger reconciliation

- Canonical source: `corpus48_编码_全维度集成版_杜俊涛.csv`, filtered to `inclusion_check=include`.
- Final included studies: **46** (48 ledger rows minus 2 excluded records).
- Six field-level conflicts in the 20-study arbitration sample were resolved using the saved consensus table; this is a reconciliation of existing records, not a new independent full-corpus reread.

## Counts used by the manuscript

- D1 provenance: `{'intrinsic': 40, 'historical-static': 5, 'learned-online': 1}`
- D2 stage: `{'inference-architecture': 23, 'training-loss': 21, 'data-generation': 1, 'post-hoc-correction': 1}`
- D3 inference requirement: `{'none': 36, 'forecast-input': 6, 'contemporaneous-state': 4}`
- D4 fidelity: `{'not-found': 44, 'code-mismatch': 1, 'code-aligned': 1}`
- Ablation type: `{'module-ablation': 23, 'constraint-on-off': 17, 'both': 6}`
- Negative-effect report: `{'no': 46}`

## Publication-status note

The ledger contains 31 journal records, 4 conference records, 10 records with SSRN/arXiv/working-paper identifiers, and 1 record without a DOI. These identifiers are not sufficient to assert peer-review status for every record; the manuscript therefore reports the 46-study cohort without a peer-reviewed/preprint breakdown until record-level status is verified.
## Manuscript synchronization notes
- Table 2 now labels the penalty conditions explicitly; =0.01$, 0.1, and 1.0 are separate contrasts and are not interpreted as matched comparisons.
- Site and terrain language is descriptive; terrain/SNR is retained only as a testable hypothesis.
- Monetary results are called stylized imbalance cashflow components; they are not total market profit.
- Clean-material agreement was recomputed in clean_coder_agreement_recomputed.csv; D4 included-subset A-B kappa is -0.029, so the pilot is not presented as uniformly high reliability.

