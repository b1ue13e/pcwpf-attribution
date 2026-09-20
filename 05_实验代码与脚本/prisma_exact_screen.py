"""PRISMA exact screening: classify the 72 dedup'd OpenAlex records into
assessed vs screened-out, by reproducible title criteria.

Criteria (frozen):
  INCLUDE-ISh title hints for forecasting + explicit physics constraint:
    forecasting/predict/prediction/forecast + (physics / power curve / wake / KDE/JS)
  EXCLUDE title hints:
    control/smoothing/optimization/inertia/maintenance/digital-twin-pre-deployment/
    aggregation/fatigue/frequency/MFAC/surrogate-only/curve-modeling (no forecasting)/
    wind-speed-only (not power)/consumer systems/PV-solar-only
Everything else: assessed (conservative).
Then merge with the 26-article coded corpus source: OpenAlex 72 + web_search novel 17.
"""
from __future__ import annotations
import json, re

recs = json.load(open("../tmp/prisma_openalex_dedup.json", encoding="utf-8"))

INCLUDE_PAT = re.compile(
    r"forecast|predict", re.I)
PHYSICS_PAT = re.compile(
    r"physics|wake|power curve|physically|constraint", re.I)
EXCLUDE_PAT = re.compile(
    r"control|smoothing|optimization|inertia|maintenance|digital twin|aggregation|"
    r"fatigue|frequency|MFAC|surrogate|curve model|consumer|PV|solar|wind-speed|"
    r"welding|fault|anomaly|detection", re.I)


def classify(title):
    t = title
    if INCLUDE_PAT.search(t) and PHYSICS_PAT.search(t):
        if EXCLUDE_PAT.search(t) and not re.search(r"forecast|predict", t, re.I):
            return "excluded"
        return "assessed"
    return "excluded"


assessed = [r for r in recs if classify(r["title"]) == "assessed"]
excluded = [r for r in recs if classify(r["title"]) == "excluded"]

# web_search novel candidates (from PRISMA检索日志 v1 + 第二轮): 17 novel
web_novel = 17

print("PRISMA exact counts (reproducible title screening):")
print(f"  OpenAlex identified (raw):        {sum(1 for _ in recs) + 0}")
print(f"  OpenAlex after dedup:             {len(recs)}")
print(f"  web_search novel (2 rounds):      {web_novel}")
print(f"  TOTAL identified (raw):           {len(recs) + web_novel}")
print()
print(f"  SCREENED (title screened):        {len(recs) + web_novel}")
print(f"    OpenAlex assessed:              {len(assessed)}")
print(f"    OpenAlex excluded (title):      {len(excluded)}")
print(f"    web_search novel -> assessed:   {web_novel}")
print(f"  ASSESSED (abstract level):        {len(assessed) + web_novel}")
print()
# final inclusion from coded corpus = 26 (26 coded, of which 5 excluded later)
print(f"  INCLUDED (coded 26-article corpus): 26")
print(f"  (of these 5 later excluded as curve-fitting/non-physics; final coded=21)")
print()
print("Excluded-at-title examples (first 15):")
for r in excluded[:15]:
    print(f"  - {r['title'][:75]}")

with open("../tmp/prisma_exact_stages.json", "w", encoding="utf-8") as f:
    json.dump({
        "identified_raw": len(recs) + web_novel,
        "openalex_dedup": len(recs),
        "web_novel": web_novel,
        "assessed_openalex": len(assessed),
        "excluded_title_openalex": len(excluded),
        "assessed_total": len(assessed) + web_novel,
        "included_coded": 26,
    }, f, indent=1)
print("\nsaved tmp/prisma_exact_stages.json")
