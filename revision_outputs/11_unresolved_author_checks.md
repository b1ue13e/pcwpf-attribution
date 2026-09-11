# Action Items Requiring Author Verification Before Journal Submission

**Project**: Restructuring "Auditing the Decision Value of Physics-Constrained Wind Power Forecasting"  
**Date**: September 2026  
**Document Status**: Pre-Submission Quality Gate (`revision` branch)  

---

This document outlines the explicit set of institutional, administrative, and logistical items marked as `AUTHOR_VERIFICATION_REQUIRED`. While all empirical numbers, mathematical derivations, software audits, and LaTeX typesetting have been completed, verified against raw ledgers, and locked, the corresponding author must review and resolve the following five administrative items prior to formal submission to the journal portal.

---

### Check 1: Anonymous Double-Blind vs. Open Science Repository URL
- **Current State in Manuscript**: The paper references `https://github.com/anonymous-audit/pcwpf-audit` in Section 4, Section 11 (STAR Methods), and Table Key Resources.
- **Decision Required**:
  - If submitting to a **single-blind journal** (*Applied Energy*, *Renewable Energy*, *IEEE TSTE*): The authors should unmask the repository URL to their authentic lab GitHub organization (e.g., `https://github.com/AUFE-EnergyLab/pcwpf-audit`) and ensure all raw CSV simulation logs and reproduction scripts are published.
  - If submitting to a **double-blind venue**: Maintain the anonymous repository link or provide an anonymous Zenodo sandbox link with author names scrubbed from commit histories.
- **Action for Author**: Confirm repository URL choice and update line 35 of `sections/11_starmethods.tex` accordingly.

---

### Check 2: Author Affiliations, Email, and Funding Grants
- **Current State in Manuscript**:
  - Authors: Junyu Li (Corresponding, `ljylikezmn999@gmail.com`), Juntao Du.
  - Affiliation: School of Statistics and Applied Mathematics, Anhui University of Finance and Economics, Bengbu 233030, China.
  - Acknowledgments: Generic acknowledgment of open-source data teams and computing infrastructure.
- **Decision Required**:
  - Verify whether institutional grant numbers (e.g., National Natural Science Foundation of China - NSFC, Anhui Provincial Natural Science Foundation, or University Key Project funds) must be explicitly listed in the Acknowledgments section.
  - Confirm whether co-author email addresses or ORCID IDs should be appended.
- **Action for Author**: Provide grant numbers if applicable to be inserted into `main.tex` (lines 120–122).

---

### Check 3: Target Journal Template & Submission Mode
- **Current State in Manuscript**: Formatted using the official Elsevier `elsarticle` documentclass with `[3p,twocolumn,times]` and `\journal{Applied Energy}`.
- **Decision Required**:
  - If submitting to *Applied Energy* or *Renewable Energy*: The manuscript is 100% formatted, compiled, and compliant.
  - If the authors decide to pivot to *IEEE Transactions on Sustainable Energy* (IEEE TSTE): The text will need to be wrapped into `IEEEtran.cls` with standard IEEE biography sections, and figures re-sized to IEEE column widths (3.5 in single, 7.1 in double).
- **Action for Author**: Confirm final target venue (*Applied Energy* recommended given the empirical energy market focus).

---

### Check 4: Data Licensing and Third-Party Attribution
- **Current State in Manuscript**:
  - Cubico Sustainable Investments SCADA: Zenodo DOIs `10.5281/zenodo.5841834` (Kelmarsh) and `10.5281/zenodo.5946808` (Penmanshiel) under CC-BY 4.0.
  - RES / The Renewables Infrastructure Group (TRIG) SCADA: Zenodo DOI `10.5281/zenodo.20204946` (Hill of Towie) under CC-BY 4.0.
  - Elexon Balancing and Settlement Code (BSC) Portal: UK Open Government Licence v3.0.
- **Decision Required**:
  - Verify that the usage of these open datasets adheres to all institutional agreements and that no proprietary NDA restrictions exist with respect to university computing clusters.
- **Action for Author**: Confirm all open data citations are acceptable.

---

### Check 5: Hardware Resource and Compute Budget Disclosures
- **Current State in Manuscript**:
  - Table Key Resources and Section 4 describe: "Dedicated cluster of 10 NVIDIA RTX 4090 GPUs distributed across two nodes: Node 1 (4 GPUs, Kelmarsh 4-seed ablations), Node 2 (6 GPUs, Hill of Towie and Penmanshiel 3-seed ablations)."
  - Total compute time: approximately 180 GPU-hours across all 3D ablations, continuous corruption scans, PatchTST evaluations, and bootstrap replications.
- **Decision Required**:
  - Confirm whether internal High-Performance Computing (HPC) centers or supercomputing cloud providers need formal acknowledgment by facility name.
- **Action for Author**: Insert formal computing center acknowledgment if required.

---

## Author Verification Sign-Off

When all 5 items above have been reviewed, please sign below and update the version tag in git:

- [ ] Item 1 (Repository URL): Verified
- [ ] Item 2 (Affiliations & Grants): Verified
- [ ] Item 3 (Target Journal Choice): Verified (*Applied Energy*)
- [ ] Item 4 (Data Licensing): Verified
- [ ] Item 5 (Compute Disclosures): Verified

*Date of Author Sign-Off*: ____________________  
*Signature*: __________________________________
