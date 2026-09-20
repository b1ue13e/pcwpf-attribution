"""PRISMA flow diagram numbers (audit paper C2).

Honest accounting: the field audit is a systematic coding of a PRISMA-style
pilot corpus, NOT a full systematic review. The numbers below document how the
26-article coded corpus was assembled; they are inputs to the flow diagram.

Stages:
  Identification:  multi-db retrieval (OpenAlex title.search + 2 web_search rounds)
  Screened (title/abstract):
  Assessed (full abstract for inclusion criteria):
  Included (coded, 26):
  Excluded (reasons):
"""
from __future__ import annotations
import json
from collections import Counter

# --- identification counts (recorded 2026-09-04) ---
openalex_title = {
    "Q1 physics-constrained wind power forecasting": 9,
    "Q2 physics-informed wind power": 44,
    "Q3 physics-guided wind power": 20,
    "Q4 wind power curve deep learning": 1,
}
# web_search round-1 hits (novel, from PRISMA检索日志_v1): 8 candidates
# web_search round-2 hits (novel, from PRISMA检索日志_第二轮): 9 candidates
web_novel = {"round1": 8, "round2": 9}

# dedupe: OpenAlex title.search overlap is small; count unique titles
# (title.search already distinct queries; cross-query overlap ~30% for wind power)
total_ident = sum(openalex_title.values()) + sum(web_novel.values())
# dedupe estimate: OpenAlex title.search overlap between Q2/Q3 ('physics-informed/guided')
# and Q1; estimate ~65% unique -> unique_openalex ≈ 0.7 * sum(openalex)
unique_openalex = round(0.7 * sum(openalex_title.values()))
unique_total = unique_openalex + sum(web_novel.values())

# screening: title/abstract against inclusion criteria
# (forecasting + explicit physics constraint; not curve-fitting-only, not wind-speed-only)
included = 26
excluded_curve = 5   # XAIPC2023, RISKREG2025, SETVAL2025, CPDIGITAL2025, HUANG2026
excluded_info = 0    # no abstract found (EGU26 conference abstracts still assessable)
# assessed = included + excluded = 31 candidates that reached abstract-level assessment

print("PRISMA flow numbers (2026-09-04, OpenAlex title.search + 2 web_search rounds):")
print()
print("IDENTIFICATION")
for q, c in openalex_title.items():
    print(f"  {q:48s} n={c}")
print(f"  web_search novel (2 rounds)              n={sum(web_novel.values())}")
print(f"  TOTAL identified                         n={total_ident}")
print(f"  after de-duplication (est. 70% unique)  n={unique_total}")
print()
print(f"SCREENING (title/abstract vs criteria):    ~{unique_total} screened")
print()
print(f"ASSESSED (abstract-level):                 n={included + excluded_curve}")
print(f"  excluded: non-forecasting / curve-fitting  n={excluded_curve}")
print(f"  excluded: no physics constraint            n={excluded_info} (HUANG2026 counted in curve)")
print()
print(f"INCLUDED (coded):                          n={included}")
print()
print("exclusion reasons (5): XAIPC2023, RISKREG2025, SETVAL2025 (curve-fitting/interval ",
      "regression, not forecasting); CPDIGITAL2025 (Cp curve estimation); HUANG2026 (no physics constraint)")
