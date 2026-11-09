# Conditional Inductive Guardrails

[中文说明](README.zh-CN.md) · [Repository](https://github.com/b1ue13e/pcwpf-attribution)

> **Conditional Inductive Guardrails: Identifying When Physical Priors Enhance, Mimic, or Impair Decisions in Wind Power Forecasting**

This repository investigates empirical attribution in wind power forecasting: how much of the performance gain attributed to physical domain priors actually arises from physical domain knowledge rather than regularized capacity, feature scaling, or evaluation artifacts.

## Overview and Current Scope

The default branch (`main`) serves as a reproducible evidence and manuscript archive. The investigation uses pseudo-scrambled placebos, budget-matched ablations, boundary stress tests, and asymmetric dispatch loss to characterize physical priors as conditional inductive guardrails.

The active manuscript draft is organized under the four-level evidence structure (Centerpiece -> Boundary Map -> Decision Alignment -> Supporting Forensics).

## Setup and Verification

The repository relies on standard Python 3.12 tooling and Tectonic for document compilation.

### Run Verification Gate

Run the canonical numerical and state consistency gate across registered claims:

```bash
python .agents/scripts/verify_gate.py
```

This script audits task state, claim ledger consistency, and core numerical assertions across manuscript and evidence files.

### Compile Manuscript

To compile the manuscript PDF from source using Tectonic:

```bash
cd "01_论文投稿材料/active_manuscript"
tectonic -X compile --keep-logs main.tex
```

The compiled PDF is also uploaded as a build artifact in CI via [`.github/workflows/verify.yml`](.github/workflows/verify.yml).

### Reproducibility and Data Boundary

- **Tracked material**: Manuscript sources, claim-evidence matrices, forensic audits, verification scripts, and publication figures.
- **Excluded material**: Raw 10-minute SCADA telemetry from external wind farms and heavy multi-seed training runs are omitted from this Git repository per `.gitignore` to preserve a clean repository size.
- **Evidence audit**: Refer to [`CLAIM_EVIDENCE_MATRIX.md`](CLAIM_EVIDENCE_MATRIX.md) and [`PHASE5_SUBMISSION_READINESS_REPORT.md`](PHASE5_SUBMISSION_READINESS_REPORT.md) for full provenance of reported empirical figures.

## Documentation Navigation

- [Claim-Evidence Matrix](CLAIM_EVIDENCE_MATRIX.md) and [Claim Map](claim_evidence_map.md): complete mapping from manuscript claims to empirical evidence.
- [Manuscript Architecture](NEW_MANUSCRIPT_ARCHITECTURE.md): the four-tier evidence hierarchy and section design.
- [Phase 5 Review Report](PHASE5_SUBMISSION_READINESS_REPORT.md): independent review record verifying headline figures and noting residual text boundaries.
- [Task Contract](START_PROMPT.md): execution standards, non-negotiable checks, and anti-drift rules.
- [CI Workflow](.github/workflows/verify.yml): automated validation configuration.

Local development checkouts and feature branches may contain private working notes or unpushed logs; this documentation describes only the files checked into the remote default branch.
