import urllib.request
import json
import time

def restore_abstract(inverted_index):
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)

dois = [
    ("S01", "10.1016/j.apenergy.2025.125295"),
    ("S02", "10.1038/s41598-025-34348-x"),
    ("S03", "10.1016/j.energy.2026.141322"),
    ("S04", "10.3390/pr14101671"),
    ("S07", "10.1016/j.renene.2026.125710"),
    ("S09", "10.1016/j.apenergy.2024.124068"),
    ("S10", "10.1016/j.engappai.2025.113331"),
    ("S11", "10.1016/j.energy.2026.141301"),
    ("S12", "10.1109/tii.2025.3623559"),
    ("S15", "10.1016/j.eswa.2026.132224"),
    ("S17", "10.1049/icp.2024.2343"),
    ("S18", "10.1016/j.epsr.2026.113241"),
    ("S19", "10.1109/tia.2026.3653910"),
    ("S20", "10.1016/j.ijepes.2026.111689"),
    ("S26", "10.1016/j.energy.2026.142114"),
    ("S27", "10.48550/arxiv.2607.01670"),
    ("S34", "10.1103/prxenergy.2.013009"),
    ("S38", "10.3390/en18246479"),
    ("S39", "10.3390/jmse14141290"),
    ("S42", "10.1016/j.epsr.2026.112925")
]

for sid, d in dois:
    url = f"https://api.openalex.org/works/https://doi.org/{d}"
    req = urllib.request.Request(url, headers={"User-Agent": "mailto:audit@research.org"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            title = data.get("title")
            authors = [a["author"]["display_name"] for a in data.get("authorships", [])]
            venue = data.get("primary_location", {}).get("source", {}).get("display_name", "")
            pub_date = data.get("publication_date", "")
            abstract = restore_abstract(data.get("abstract_inverted_index"))
            print(f"[{sid}] {d}")
            print(f"  Title: {title}")
            print(f"  Venue: {venue} | Date: {pub_date}")
            print(f"  Authors: {', '.join(authors[:4])}")
            print(f"  Abstract: {abstract[:150]}...\n")
    except Exception as e:
        print(f"[{sid}] {d} FAIL: {e}\n")
    time.sleep(0.3)
