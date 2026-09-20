"""PRISMA multi-database retrieval via OpenAlex + Semantic Scholar APIs.

Queries (frozen before running, matching PRISMA protocol):
  Q1: physics-constrained wind power forecasting (title/abstract)
  Q2: physics-informed wind power forecasting constraint
  Q3: wind power curve constraint deep learning
  Q4: theory-guided wind power forecasting distribution

Year filter: 2023-2026. Records: doi, title, year, venue, authors.
Output: JSONL database records for the PRISMA flow diagram.
"""
from __future__ import annotations
import json, time, ssl, urllib.request, urllib.parse

UA = {"User-Agent": "constraint-audit/1.0 (mailto:research@example.com)"}
CTX = ssl.create_default_context()

def openalex(query, years="2023-2026", per_page=50, max_pages=4):
    out = []
    cursor = "*"
    for _ in range(max_pages):
        url = ("https://api.openalex.org/works"
               f"?search={urllib.parse.quote(query)}"
               f"&filter=publication_year:{years},type:article"
               f"&per-page={per_page}&cursor={cursor}")
        req = urllib.request.Request(url, headers=UA)
        data = json.load(urllib.request.urlopen(req, timeout=40, context=CTX))
        for w in data["results"]:
            out.append({
                "db": "openalex",
                "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                "title": (w.get("display_name") or "")[:180],
                "year": w.get("publication_year"),
                "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                "authors": [a["author"]["display_name"] for a in (w.get("authorships") or [])][:3],
            })
        cursor = data["meta"].get("next_cursor")
        if not cursor:
            break
        time.sleep(0.3)
    return out

def semantic_scholar(query, limit=50):
    url = ("https://api.semanticscholar.org/graph/v1/paper/search"
           f"?query={urllib.parse.quote(query)}&limit={limit}"
           "&fields=title,year,venue,externalIds,authors")
    req = urllib.request.Request(url, headers=UA)
    try:
        data = json.load(urllib.request.urlopen(req, timeout=40, context=CTX))
    except Exception as e:
        print("S2 fail:", e); return []
    out = []
    for p in data.get("data", []):
        out.append({
            "db": "semantic_scholar",
            "doi": (p.get("externalIds") or {}).get("DOI") or "",
            "title": (p.get("title") or "")[:180],
            "year": p.get("year"),
            "venue": p.get("venue") or "",
            "authors": [a["name"] for a in (p.get("authors") or [])][:3],
        })
    return out

QUERIES = [
    "physics-constrained wind power forecasting",
    "physics-informed wind power forecasting constraint loss",
    "wind power curve constraint deep learning regularization",
    "theory-guided distribution constraint wind power forecasting",
]

def main():
    all_recs = []
    seen = set()
    for q in QUERIES:
        recs = openalex(q)
        recs += semantic_scholar(q)
        for r in recs:
            key = (r["doi"] or r["title"][:60] + r["db"])
            if key in seen:
                continue
            seen.add(key)
            all_recs.append(r)
        print(f"{q[:45]:45s} -> {len(recs)} records (dedup total {len(all_recs)})")
        time.sleep(1)
    with open("../tmp/prisma_retrieval_multidb.jsonl", "w", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\ntotal unique records: {len(all_recs)}")
    from collections import Counter
    print("per db:", dict(Counter(r['db'] for r in all_recs)))
    print("per year:", dict(Counter(r['year'] for r in all_recs)))

if __name__ == "__main__":
    main()
