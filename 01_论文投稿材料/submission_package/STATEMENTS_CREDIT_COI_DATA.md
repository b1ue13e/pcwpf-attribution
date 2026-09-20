# Author Declarations and Compliance Statements

**Manuscript Title:** Conditional Inductive Guardrails: Identifying When Physical Priors Enhance, Mimic, or Impair Decisions in Wind Power Forecasting  
**Journal:** *Applied Energy* (Elsevier)  
**Authors:** Junyu Li and Juntao Du  

---

### 1. Author Contributions (CRediT Taxonomy)

In accordance with Elsevier's CRediT (Contributor Roles Taxonomy) authorship guidelines, the specific contributions of each author are defined as follows:

- **Junyu Li (Corresponding Author):** Conceptualization, Methodology, Software, Formal analysis, Data curation, Investigation, Visualization, Project administration, Writing – original draft, Writing – review & editing.
  - *Summary:* Formulated the controlled attribution protocol and negative control framework; developed the multi-farm modeling and evaluation codebase; performed analytical proofs; executed multi-seed experimental runs; processed operational SCADA and market telemetry; drafted and revised the manuscript.

- **Juntao Du:** Methodology (review protocol), Validation, Data curation (scoping extraction double-coding), Writing – review & editing.
  - *Summary:* Co-developed the scoping review protocol; conducted independent double-coding and data validation across the audited literature corpus; verified statistical tables and paired confidence intervals; reviewed and contributed to manuscript revisions.

---

### 2. Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

- None of the authors have received financial support or consulting honoraria from commercial renewable forecasting software vendors, wind farm owner-operators, or electricity market participants related to the evaluated models.
- Neither author holds stock or proprietary financial positions in the entities whose open-access telemetry is evaluated (Cubico Sustainable Investments, RES, or TRIG).

---

### 3. Declaration of Generative AI in the Writing Process

During the preparation of this work, the authors used large language model tools (specifically OpenAI ChatGPT and Google Gemini) to assist with:
1. Preliminary literature synthesis and structural organization;
2. Grammar checking, academic tone calibration, and conciseness auditing;
3. Software static code linting and Python script formatting;
4. Cross-document consistency verification between tables, figures, and textual citations.

After utilizing these tools, the authors thoroughly reviewed, edited, and verified all text, mathematical proofs, experimental data, and code implementations. The authors take full, sole responsibility for the scientific integrity, factual accuracy, and original phrasing of the final published work.

---

### 4. Data and Code Availability Statement

To guarantee absolute reproducibility and facilitate transparent community auditing, all datasets, price records, extraction sheets, and executable code pipelines utilized in this work are openly available:

#### A. Open-Access Operational SCADA Telemetry
1. **Kelmarsh Wind Farm:** 10-minute continuous SCADA telemetry from 6 Senvion MM92 turbines (12.3 MW rated capacity, Northamptonshire, UK), released by Cubico Sustainable Investments under CC-BY 4.0 license. Available on Zenodo:  
   - DOI: [10.5281/zenodo.5841834](https://doi.org/10.5281/zenodo.5841834)
2. **Hill of Towie Wind Farm:** 10-minute continuous SCADA telemetry from 21 Siemens SWT-2.3-93 turbines (48.3 MW rated capacity, Drummuir, Scotland), released by RES on behalf of The Renewables Infrastructure Group (TRIG) under CC-BY 4.0 license. Available on Zenodo:  
   - DOI: [10.5281/zenodo.20204946](https://doi.org/10.5281/zenodo.20204946)
3. **Penmanshiel Wind Farm:** 10-minute continuous SCADA telemetry from 14 Senvion MM82 turbines (28.7 MW rated capacity, Scottish Borders, UK), released by Cubico Sustainable Investments under CC-BY 4.0 license. Available on Zenodo:  
   - DOI: [10.5281/zenodo.5946808](https://doi.org/10.5281/zenodo.5946808)

#### B. Electricity Market Telemetry
- **UK Elexon Balancing and Settlement Code (BSC):** Half-hourly System Buy Price (SBP) and System Sell Price (SSP) records for 2017 are freely accessible through the Elexon Balancing Mechanism Reporting Service (BMRS) portal under the UK Open Government Licence v3.0:  
   - Portal: [https://www.bmreports.com/bmrs/](https://www.bmreports.com/bmrs/)

#### C. Code and Verification Suite
- **GitHub Audit Repository:** Complete Python verification scripts, PyTorch neural models (DLinear, PatchTST, Physics-Constrained Transformer), static AST decompilation parsers, market settlement simulation modules, and seed-wise logs are openly archived at:  
   - URL: [https://github.com/anonymous-audit/pcwpf-audit](https://github.com/anonymous-audit/pcwpf-audit)
- **Zenodo Persistent Archive:** The immutable repository snapshot containing exact training checkpoints, processed datasets, and figure scripts is deposited on Zenodo:  
   - DOI: [10.5281/zenodo.10892341](https://doi.org/10.5281/zenodo.10892341)

#### D. Scoping Review Database
- The comprehensive 46-study extraction spreadsheet, including verbatim claim citations, coder ratings, Cohen's kappa calculation scripts, and tripartite arbitration logs, is deposited in both CSV and JSON formats within the replication archive.
