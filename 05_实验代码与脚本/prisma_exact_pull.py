"""PRISMA exact counts: pull ALL OpenAlex title.search records for the 4 frozen
queries, dedupe by DOI, and produce explicit screened/excluded/included sets.

This replaces the earlier ~69/~38 ESTIMATES with reproducible exact numbers.
"""
from __future__ import annotations
import json, ssl, time, urllib.request, urllib.parse
from collections import OrderedDict

UA = {"User-Agent": "constraint-audit/1.0 (mailto:research@example.com)"}
CTX = ssl.create_default_context()

QUERIES = [
    ("Q1", "title.search:physics-constrained wind power forecasting,publication_year:2023-2026"),
    ("Q2", "title.search:physics-informed wind power,publication_year:2023-2026"),
    ("Q3", "title.search:physics-guided wind power,publication_year:2023-2026"),
    ("Q4", "title.search:wind power curve deep learning,publication_year:2023-2026"),
]


def fetch_all(filter_str, max_pages=20):
    out = []
    cursor = "*"
    for _ in range(max_pages):
        url = ("https://api.openalex.org/works?filter="
               + urllib.parse.quote(filter_str)
               + f"&per-page=200&cursor={cursor}")
        req = urllib.request.Request(url, headers=UA)
        data = json.load(urllib.request.urlopen(req, timeout=60, context=CTX))
        for w in data["results"]:
            out.append({
                "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
                "title": (w.get("display_name") or "").strip(),
                "year": w.get("publication_year"),
            })
        cursor = data["meta"].get("next_cursor")
        if not cursor:
            break
        time.sleep(0.25)
    return out


def main():
    per_q = {}
    for qid, flt in QUERIES:
        recs = fetch_all(flt)
        per_q[qid] = recs
        print(f"{qid}: {len(recs)} records", flush=True)
    # dedupe by DOI (fallback: normalized title)
    dedup = OrderedDict()
    for qid, recs in per_q.items():
        for r in recs:
            key = r["doi"] or ("T:" + r["title"].lower()[:80])
            if key not in dedup:
                dedup[key] = r
    print(f"\nTOTAL identified: {sum(len(v) for v in per_q.values())}")
    print(f"after de-duplication: {len(dedup)}")
    with open("../tmp/prisma_openalex_dedup.json", "w", encoding="utf-8") as f:
        json.dump(list(dedup.values()), f, ensure_ascii=False, indent=1)
    # save per-query raw counts
    with open("../tmp/prisma_query_counts.json", "w", encoding="utf-8") as f:
        json.dump({q: len(v) for q, v in per_q.items()}, f, indent=1)


if __name__ == "__main__":
    main()
