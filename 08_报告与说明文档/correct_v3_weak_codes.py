"""Apply full-text/abstract corrections to 7 weak-coded articles (P2 item 1).

Corrections (decided 2026-09-04 from full abstracts):
  JARA2024:   D3 latent-state -> contemporaneous-state
              (possible power inferred from farm-wide contemporaneous SCADA + environment;
               the target is 'expected behaviour of a turbine', inferred not latent-observed)
  MLPINN:     D2 training-loss -> training-loss (keep; but add data-aug hybrid note)
              D3 contemporaneous-state -> forecast-input
              (input = MERRA-2 hourly wind, not contemporaneous turbine SCADA)
  MCMAMBA2026: D3 forecast-input -> none
              (physics constraint only in MPG-Diffusion data-augmentation at train-prep;
               MCNN-Mamba inference needs no physics prior)
  DUALSTREAM2026: D2 unreported -> training-loss
                  D3 unreported -> contemporaneous-state
              ('both streams strictly bounded by physical constraints';
               fusion needs real-time wind-speed gradients/volatility)
  CPDIGITAL2025: EXCLUDE (Cp curve estimation for digital twin, NOT forecasting)
  TCNTRANS2026: keep (abstract-only, weak anchor); mark D1/D2/D3 unreported with note
  HUANG2026:    already EXCLUDE (no physics constraint)
"""
from __future__ import annotations
import csv

CORRECT = {
    "JARA2024": {"D3_inference_req": "contemporaneous-state",
                 "note": "FULLTEXT: possible power inferred from farm-wide contemporaneous SCADA + env conditions (white-box inference), not latent-observed"},
    "MLPINN": {"D3_inference_req": "forecast-input",
               "note": "FULLTEXT: input = MERRA-2 hourly wind (reanalysis/forecast), not contemporaneous turbine SCADA; PINN physics = PMSG d-q + BEM"},
    "MCMAMBA2026": {"D3_inference_req": "none",
                    "note": "FULLTEXT: physics constraint only in MPG-Diffusion data-augmentation (pre-training); MCNN-Mamba inference needs no physics prior"},
    "DUALSTREAM2026": {"D2_stage": "training-loss",
                       "D3_inference_req": "contemporaneous-state",
                       "note": "FULLTEXT: both streams strictly bounded by physical constraints; fusion uses real-time wind-speed gradients + volatility"},
    "CPDIGITAL2025": {"D2_stage": "training-loss",
                      "note": "EXCLUDE-ADJUSTMENT: Cp curve estimation for digital twin (parameterized model + penalty), NOT forecasting task"},
    "TCNTRANS2026": {"note": "WEAK: IET abstract-only, physics detail unavailable; keep unreported"},
}

def main():
    rows = list(csv.DictReader(open('literature_coding_table_v3_arbitrated.csv', encoding='utf-8-sig')))
    for r in rows:
        pid = r['paper_id']
        if pid in CORRECT:
            c = CORRECT[pid]
            for k, v in c.items():
                if k == 'note':
                    r['notes'] = (r.get('notes', '') + ' | ' + v).strip(' |')
                else:
                    r[k] = v
    with open('literature_coding_table_v3_final.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('v3_final written:', len(rows), 'rows')
    from collections import Counter
    print('D3 (final):', dict(Counter(r['D3_inference_req'] for r in rows)))
    print('D2 (final):', dict(Counter(r['D2_stage'] for r in rows)))

if __name__ == '__main__':
    main()
