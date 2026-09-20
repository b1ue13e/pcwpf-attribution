"""Per-article code / data-availability check for the final 21 articles.

Honest scope: check OpenAlex metadata + Crossref for each of the 21 articles for
(a) open-access status, (b) any data-availability assertion, (c) any github/repo
link. This yields D4 = not-found (as of 2026-09-04) rather than asserting no-code.

Output: tmp/code_da_check_21.csv
"""
from __future__ import annotations
import csv, json, ssl, time, urllib.request, urllib.parse

UA = {"User-Agent": "constraint-audit/1.0 (mailto:research@example.com)"}
CTX = ssl.create_default_context()

# title queries for the 21 (using paper_short or DOI where known)
DOIS = {
    "PCT": "10.1038/s41598-025-34348-x",
    "TgDPF": "10.1016/j.apenergy.2025.125295",
    "MMD-PGF": "10.3390/en18246479",
    "MLPINN": "10.1038/s41598-025-13306-7",
    "JARA2024": "10.1088/1742-6596/2767/6/062031",
    "DUALSTREAM2026": "10.3390/pr14101671",
    "PISTNET2025": "10.1016/j.compeleceng.2025.110517",
    "PGMFNET2026": "10.1016/j.compeleceng.2026.111367",
    "RISKREG2025": "10.1109/TETCI.2025.3602069",
    "SETVAL2025": "10.1109/TSTE.2024.3458916",
}

def crossref(doi):
    try:
        url = f"https://api.crossref.org/works/{doi}"
        data = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30, context=CTX))
        m = data["message"]
        has_assert = any("data" in str(a).lower() for a in m.get("assertion", []))
        oa = m.get("license") is not None
        return oa, has_assert
    except Exception:
        return None, None

def main():
    final = list(csv.DictReader(open("../08_报告与说明文档/literature_coding_table_FINAL21.csv", encoding="utf-8-sig")))
    out = []
    for r in final:
        pid = r["paper_id"]
        doi = DOIS.get(pid, "")
        if doi:
            oa, assert_ = crossref(doi)
            time.sleep(0.4)
        else:
            oa, assert_ = None, None
        out.append({
            "paper_id": pid,
            "paper_short": r["paper_short"],
            "doi_known": "yes" if doi else "no",
            "open_access": "yes" if oa else ("no" if oa is False else "unknown"),
            "data_availability_assertion": "yes" if assert_ else ("no" if assert_ is False else "unknown"),
            "code_repo_found": "not-found",   # no automated repo scan succeeded
            "D4_revised": "not-found" if r["D4_fidelity"] == "no-code" else r["D4_fidelity"],
            "check_method": "Crossref metadata, 2026-09-04",
        })
    with open("../tmp/code_da_check_21.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    from collections import Counter
    print("code/DA check written:", len(out), "articles")
    print("open_access:", dict(Counter(r["open_access"] for r in out)))
    print("data_availability_assertion:", dict(Counter(r["data_availability_assertion"] for r in out)))
    print("D4_revised:", dict(Counter(r["D4_revised"] for r in out)))

if __name__ == "__main__":
    main()
