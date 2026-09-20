import argparse
import os
import re
from typing import Dict, List


def parse_bib_entries(text: str) -> Dict[str, str]:
    entries = {}
    matches = list(re.finditer(r"@\w+\{\s*([^,]+),", text))
    for i, m in enumerate(matches):
        key = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entries[key] = text[start:end]
    return entries


def parse_cited_keys(aux_text: str) -> List[str]:
    keys: List[str] = []
    for m in re.finditer(r"\\citation\{([^}]*)\}", aux_text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


def is_arxiv_only(entry: str) -> bool:
    low = entry.lower()
    has_arxiv = ("arxiv" in low) or ("archiveprefix" in low and "eprint" in low)
    has_formal = ("journal" in low) or ("booktitle" in low) or ("publisher" in low and "arxiv" not in low)
    return has_arxiv and (not has_formal)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate no arXiv-only entries in cited main-chain references")
    ap.add_argument("--aux", type=str, default="main_PI-MoE_final_AE_ready.aux")
    ap.add_argument("--bib", type=str, default="cas-refs.bib")
    args = ap.parse_args()

    if not os.path.exists(args.aux):
        raise SystemExit(f"Missing aux: {args.aux}")
    if not os.path.exists(args.bib):
        raise SystemExit(f"Missing bib: {args.bib}")

    aux = open(args.aux, "r", encoding="utf-8", errors="ignore").read()
    bib = open(args.bib, "r", encoding="utf-8", errors="ignore").read()
    cited = parse_cited_keys(aux)
    entries = parse_bib_entries(bib)

    violations = []
    missing = []
    for k in cited:
        e = entries.get(k)
        if e is None:
            missing.append(k)
            continue
        if is_arxiv_only(e):
            violations.append(k)

    if missing:
        print("WARNING: missing cited keys in bib:", ", ".join(missing))
    if violations:
        print("arXiv-only violations:")
        for v in violations:
            print(v)
        raise SystemExit(1)

    print("Citation policy passed: no arXiv-only cited entries.")


if __name__ == "__main__":
    main()
