"""Encode D1-D5 (constraint lifecycle) for 26 articles from abstract-level info.
coder_A single-pass; coder_B to be done independently per v3 protocol.

D1 provenance: intrinsic / historical-static / forecast-derived / learned-online / unreported
D2 stage      : training-loss / post-hoc / inference-arch / data-generation / unreported
D3 inference  : none / forecast-input / contemporaneous-state / latent-state / unreported
D4 fidelity   : code-aligned / code-mismatch / no-code / parity-only / unverified
D5 harm       : none-reported / ablation-absent / harm-quantified / substitution-tested
"""
from __future__ import annotations
import csv

# paper_id -> (D1, D2, D3, D4, D5)
ENC = {
    "HUANG2026":       ("unreported", "unreported", "unreported", "unverified", "none-reported"),
    "JARA2024":        ("historical-static", "inference-arch", "latent-state", "unverified", "none-reported"),
    "MLPINN":          ("intrinsic", "training-loss", "contemporaneous-state", "unverified", "none-reported"),
    "MMD-PGF":         ("historical-static", "training-loss", "none", "unverified", "none-reported"),
    "PCT":             ("historical-static", "training-loss", "none", "code-mismatch", "none-reported"),
    "PHYCELL":         ("forecast-derived", "training-loss", "forecast-input", "unverified", "none-reported"),
    "TANG2025":        ("forecast-derived", "training-loss", "forecast-input", "no-code", "none-reported"),
    "TgDPF":           ("historical-static", "training-loss", "none", "no-code", "none-reported"),
    "WANG2026":        ("historical-static", "inference-arch", "latent-state", "unverified", "none-reported"),
    "PISTNET2025":     ("forecast-derived", "training-loss", "forecast-input", "unverified", "none-reported"),
    "PIBPNN2025":      ("intrinsic", "training-loss", "contemporaneous-state", "unverified", "none-reported"),
    "PITFN2025":       ("forecast-derived", "training-loss", "forecast-input", "parity-only", "none-reported"),
    "TCNTFT2025":      ("forecast-derived", "training-loss", "forecast-input", "unverified", "none-reported"),
    "CPDIGITAL2025":   ("historical-static", "training-loss", "contemporaneous-state", "unverified", "none-reported"),
    "XAIPC2023":       ("unreported", "unreported", "unreported", "unverified", "none-reported"),
    "RISKREG2025":     ("unreported", "unreported", "unreported", "unverified", "none-reported"),
    "SETVAL2025":      ("unreported", "unreported", "unreported", "unverified", "none-reported"),
    "PHYWINDNET2026":  ("intrinsic", "training-loss", "contemporaneous-state", "unverified", "none-reported"),
    "PGMFNET2026":     ("forecast-derived", "training-loss", "forecast-input", "unverified", "none-reported"),
    "NCF2026":         ("forecast-derived", "training-loss", "forecast-input", "unverified", "none-reported"),
    "EGU26AFNO2026":   ("forecast-derived", "training-loss", "forecast-input", "unverified", "none-reported"),
    "DUALSTREAM2026":  ("unreported", "training-loss", "unreported", "unverified", "none-reported"),
    "JENSENWAKE2026":  ("forecast-derived", "training-loss", "contemporaneous-state", "parity-only", "harm-quantified"),
    "MCMAMBA2026":     ("forecast-derived", "data-generation", "forecast-input", "unverified", "none-reported"),
    "TCNTRANS2026":    ("unreported", "training-loss", "unreported", "unverified", "none-reported"),
    "VARFORMERCQR2026":("unreported", "unreported", "unreported", "unverified", "none-reported"),
}

D1MAP = {"intrinsic":"intrinsic","historical-static":"historical-static","forecast-derived":"forecast-derived",
         "learned-online":"learned-online","unreported":"unreported"}
D2MAP = {"training-loss":"training-loss","post-hoc":"post-hoc-correction","inference-arch":"inference-architecture",
         "data-generation":"data-generation","unreported":"unreported"}

def main():
    src = list(csv.DictReader(open('literature_coding_table_dual.csv', encoding='utf-8-sig')))
    fields = list(src[0].keys()) + ['D1_provenance','D2_stage','D3_inference_req','D4_fidelity','D5_harm']
    for r in src:
        pid = r['paper_id']
        d1,d2,d3,d4,d5 = ENC.get(pid, ("unreported","unreported","unreported","unverified","none-reported"))
        r['D1_provenance'] = D1MAP.get(d1, d1)
        r['D2_stage'] = D2MAP.get(d2, d2)
        r['D3_inference_req'] = d3
        r['D4_fidelity'] = d4
        r['D5_harm'] = d5
    with open('literature_coding_table_v3_D1D5.csv','w',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(src)
    print('v3 D1-D5 table written:', len(src), 'rows (coder_A)')
    from collections import Counter
    for dim in ['D1_provenance','D2_stage','D3_inference_req','D4_fidelity','D5_harm']:
        print(dim, dict(Counter(r[dim] for r in src)))

if __name__ == '__main__':
    main()
