"""Cross-site x cross-proxy audit of the point-vs-distribution dichotomy.

For each (site, proxy) pair, score the three constraint families with the
unified protocol. Reproduces the dichotomy claim across sites and proxies.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from audit_protocol import (
    AuditResult, JSSpec, audit_constraint,
    constraint_betz_penalty, constraint_js_distribution, constraint_proxy_envelope,
    fit_curve, power_curve_pu,
)

SITES = {
    "wind_cn": "../04_数据集/prepared_priority_datasets/wind_cn_core_15min.csv",
    "kelmarsh": "../04_数据集/prepared_priority_datasets/kelmarsh_uk_2016.csv",
    "penmanshiel": "../04_数据集/prepared_priority_datasets/penmanshiel_scotland_2016.csv",
    "smarteole": "../04_数据集/prepared_priority_datasets/smarteole_france_2020.csv",
    "wind_us": "../04_数据集/prepared_priority_datasets/wind_us_core_15min.csv",
}


def make_proxies(ws: np.ndarray) -> dict:
    n = len(ws) - 1
    proxies = {"persistence": ws[:-1]}
    if len(ws) >= 3:
        proxies["linear"] = (2 * ws[1:-1] - ws[:-2])[:n]
    # EWMA a=0.6
    a = 0.6
    ew = np.empty_like(ws)
    ew[0] = ws[0]
    for t in range(1, len(ws)):
        ew[t] = a * ws[t - 1] + (1 - a) * ew[t - 1]
    proxies["ewma_0.6"] = ew[1:]
    return proxies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="../08_报告与说明文档/跨站跨proxy审计矩阵.csv")
    ap.add_argument("--sites", default=None, help="comma list; default all")
    args = ap.parse_args()

    site_names = args.sites.split(",") if args.sites else list(SITES.keys())
    rows = []
    for site in site_names:
        path = SITES[site]
        df = pd.read_csv(path)
        ws = df["wind_speed"].to_numpy(np.float64)
        p = df["power"].to_numpy(np.float64)
        valid = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
        ws, p = ws[valid], p[valid]
        p_rated, v_in, v_rated = fit_curve(ws, p)
        v_true = ws[1:]
        pred = p[1:]
        true = p[1:]
        for pname, v_proxy in make_proxies(ws).items():
            vp = np.asarray(v_proxy, dtype=np.float64)
            m = min(len(vp), len(v_true))
            vp = vp[:m]
            vt = v_true[:m]
            pd_ = pred[:m]
            tr_ = true[:m]
            for cname, cfn in [
                ("proxy_envelope", lambda: constraint_proxy_envelope(pd_, vp, p_rated, v_in, v_rated)),
                ("betz", lambda: constraint_betz_penalty(pd_, vp, p_rated, v_in, v_rated)),
                ("js", lambda: constraint_js_distribution(pd_, vp, JSSpec(p_rated=p_rated, v_in=v_in, v_rated=v_rated))),
            ]:
                kind = "distribution" if cname == "js" else "point"
                r = audit_constraint(cname, kind, cfn, pd_, tr_, vt, vp, p_rated, v_rated)
                rows.append({
                    "site": site, "proxy": pname, "constraint": cname, "kind": kind,
                    "over%": round(r.over_rated_rate, 4), "neg%": round(r.neg_rate, 4),
                    "ramp_cut": round(r.ramp_legit_cut, 4),
                    "asym_pt": round(r.ramp_asymmetry_pt, 2),
                })
                print(f"{site:12s} {pname:12s} {cname:16s} asym={r.ramp_asymmetry_pt:+6.1f}pt "
                      f"ramp_cut={r.ramp_legit_cut:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n=== saved {len(out)} rows -> {args.output} ===")
    # dichotomy summary
    piv = out.groupby(["constraint"]).asym_pt.agg(["mean", "std", "min", "max"])
    print("\n=== ramp-asymmetry by constraint (all sites x proxies) ===")
    print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
