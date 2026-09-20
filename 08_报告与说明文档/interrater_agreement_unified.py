"""Unified inter-rater agreement across the full 26-article dual-coded table.

Computes per-field %-agreement + Cohen's kappa where BOTH coders have a value,
across the combined dual-coded subset (round-1 8 rows have coderB_ columns;
round-2 9 rows have coderA_ columns; original pilot 9 rows were arbitrated and
kept in arbitrated form so their agreement is recorded in
双人编码与仲裁记录_pilot.md section 2).

Also computes Krippendorff's alpha for the nominal 5-level observability field
on the 17 dual-coded rows, as the ruling requested.
"""
from __future__ import annotations
import csv
import numpy as np


def cohens_kappa(a, b, labels):
    n = len(a)
    o = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return o, (o - pe) / (1 - pe) if pe < 1 else 1.0


def alpha_nominal(data):
    """Krippendorff's alpha for nominal data. data: list of lists (coders x units)."""
    units = len(data[0])
    coders = len(data)
    n = units * coders
    # observed disagreement
    do = 0.0
    pairs = 0
    for u in range(units):
        vals = [data[c][u] for c in range(coders)]
        for i in range(coders):
            for j in range(i + 1, coders):
                if vals[i] != vals[j]:
                    do += 1
                pairs += 1
    do /= pairs if pairs else 1
    # expected disagreement
    de = 0.0
    pairs2 = 0
    for i in range(units):
        for j in range(units):
            if i == j:
                continue
            vi = [data[c][i] for c in range(coders)]
            vj = [data[c][j] for c in range(coders)]
            for a in vi:
                for b in vj:
                    if a != b:
                        de += 1
            pairs2 += coders * coders
    de /= pairs2 if pairs2 else 1
    return 1 - do / de if de else 1.0


def main():
    rows = list(csv.DictReader(open('literature_coding_table_dual.csv', encoding='utf-8-sig')))
    # build coder-pairs per field
    pairs = {
        'observability_level': [],
        'enforcement_mode': [],
        'deployment_source_stated': [],
        'harm_reported': [],
        'substitution_test': [],
        'oracle_leakage': [],
        'code_available': [],
    }
    for r in rows:
        # round-1: coder=A main vs coderB_ cols
        if r.get('coderB_level'):
            pairs['observability_level'].append((r['observability_level'], r['coderB_level']))
            pairs['enforcement_mode'].append((r['enforcement_mode'], r.get('coderB_enforcement', '')))
            pairs['deployment_source_stated'].append((r['deployment_source_stated'], r.get('coderB_ds', '')))
            pairs['harm_reported'].append((r['harm_reported'], r.get('coderB_harm', '')))
            pairs['substitution_test'].append((r['substitution_test'], r.get('coderB_subst', '')))
            pairs['oracle_leakage'].append((r['oracle_leakage'], r.get('coderB_oracle_leakage', '')))
            pairs['code_available'].append((r['code_available'], r.get('coderB_code', '')))
        # round-2: coder=main is B, coderA_ cols hold A
        if r.get('coderA_level'):
            pairs['observability_level'].append((r['coderA_level'], r['observability_level']))
            pairs['enforcement_mode'].append((r['coderA_enforcement'], r['enforcement_mode']))
            pairs['deployment_source_stated'].append((r['coderA_ds'], r['deployment_source_stated']))
            pairs['harm_reported'].append((r['coderA_harm'], r['harm_reported']))
            pairs['substitution_test'].append((r['coderA_subst'], r['substitution_test']))
            pairs['oracle_leakage'].append((r['coderA_oracle_leakage'], r['oracle_leakage']))
            pairs['code_available'].append((r['coderA_code'], r['code_available']))

    print(f"{'field':28s} {'n':>3s} {'agree%':>7s} {'kappa':>7s}")
    # collect observability for alpha
    obs_matrix = None
    for f, ps in pairs.items():
        if not ps:
            print(f'{f:28s}   -    -    -')
            continue
        a = [x for x, y in ps]; b = [y for x, y in ps]
        n = len(ps)
        o, k = cohens_kappa(a, b, sorted(set(a + b)))
        print(f'{f:28s} {n:3d} {o*100:6.1f}% {k:7.3f}')
    # Krippendorff alpha on observability (nominal), dual-coded subset
    obs = pairs['observability_level']
    if obs:
        obs_matrix = [[x for x, _ in obs], [y for _, y in obs]]
        # filter out empty strings
        alpha = alpha_nominal(obs_matrix)
        print(f'Krippendorff alpha (observability, nominal, n={len(obs)}): {alpha:.3f}')


if __name__ == '__main__':
    main()
