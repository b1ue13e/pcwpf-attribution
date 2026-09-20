"""Paired significance test for prior-source substitution results.

Compares each prior source vs unconstrained across (site, seed) pairs.
Because each condition uses ALIGNED RNG (same shuffle order), the (site, seed)
pairs are matched: paired Wilcoxon signed-rank + paired t-test on RMSE deltas.

Reads result lines from prior experiments:
  v3 (5 conditions, 6 tasks): pct_oracle_sub_v3.txt
  seed234 additions:          pct_prior_seed234.txt (after download)
"""
from __future__ import annotations
import re
import numpy as np
from scipy import stats


def parse(line):
    # e.g. "kelmarsh seed=0 epochs=60 | oracle=0.09564 lagged=... nwp_linear=... nwp_measured=... unconstrained=0.08667"
    site = line.split()[0]
    seed = int(re.search(r'seed=(\d+)', line).group(1))
    nums = dict(re.findall(r'(hist|oracle|lagged|nwp_linear|nwp_measured|uncon\w*)=([\d.e+-]+)', line))
    # normalize keys: 'oracle' == historical-static; unconstrained key may be 'unconstrained' or 'uncon'
    out = {'site': site, 'seed': seed}
    m = re.findall(r'(\w+)=([\d.]+)', line)
    for k, v in m:
        out[k] = float(v)
    return out


def main():
    import glob, os
    files = []
    for pat in ['../tmp/pct_prior_seed234.txt', '../tmp/pct_oracle_sub_v3.txt', '../tmp/pct_oracle_multi_v2.txt']:
        if os.path.exists(pat):
            files.append(pat)
    rows = []
    for fp in files:
        for line in open(fp, encoding='utf-8'):
            line = line.strip()
            if '=' not in line or 'seed=' not in line:
                continue
            try:
                rows.append(parse(line))
            except Exception as e:
                print('skip:', line[:60], e)
    # dedupe by (site, seed): keep the 5-condition version (has nwp_measured)
    seen = {}
    for r in rows:
        key = (r['site'], r['seed'])
        if key not in seen or ('nwp_measured' in r and 'nwp_measured' not in seen[key]):
            seen[key] = r
    rows = list(seen.values())
    print(f'parsed {len(rows)} unique (site,seed) rows from {files}')
    # normalize: historical-static key may be 'oracle' or 'hist'
    def get(r, keys):
        for k in keys:
            if k in r:
                return r[k]
        return None
    hist = [get(r, ['oracle', 'hist_static', 'hist']) for r in rows]
    lagged = [get(r, ['lagged']) for r in rows]
    nwpl = [get(r, ['nwp_linear']) for r in rows]
    nwpm = [get(r, ['nwp_measured']) for r in rows]
    uncon = [get(r, ['unconstrained', 'uncon']) for r in rows]
    names = [('historical-static', hist), ('lagged', lagged), ('nwp_linear', nwpl), ('nwp_measured', nwpm)]
    print('\npairwise vs unconstrained (paired, matched RNG):')
    for nm, arr in names:
        pairs = [(a, b) for a, b in zip(arr, uncon) if a is not None and b is not None]
        if len(pairs) < 4:
            print(f'{nm:18s} n={len(pairs)} (insufficient)')
            continue
        deltas = [a - b for a, b in pairs]
        rel = [100 * (a - b) / b for a, b in pairs]
        t = stats.ttest_rel(np.array([a for a, _ in pairs]), np.array([b for _, b in pairs]))
        w = stats.wilcoxon(np.array(deltas))
        # count how many pairs are worse (positive delta = worse)
        worse = sum(1 for d in deltas if d > 0)
        print(f'{nm:18s} n={len(pairs):2d} mean_rel={np.mean(rel):+7.1f}% '
              f'worse={worse}/{len(pairs)} t_p={t.pvalue:.4f} wilcox_p={w.pvalue:.4f}')


if __name__ == '__main__':
    main()
