"""coder_B independent D1-D5 encoding. Re-derives from constraint_math_form +
input_variable + notes text ONLY (does not read coder_A's ENC dict), using fixed
decision rules, to approximate an independent second coder.

Rules (declared before coding):
D1: 'intrinsic' if form mentions Betz/rated/cut-in/mechanical/physical constant;
    'historical-static' if KDE/JS/MMD + 'historical/train';
    'forecast-derived' if NWP/advection/Jensen/wake/meteorological/curve-model;
    else 'unreported'.
D2: 'training-loss' if regularization/penalty/loss; 'data-generation' if
    diffusion/generative; 'inference-architecture' if reconstruction/latent
    inference; else 'unreported'.
D3: 'forecast-input' if NWP/forecast/meteorological/wake-field; 'latent-state' if
    latent/possible-power/curtailment; 'contemporaneous-state' if SCADA/turbine
    state/upstream measured; 'none' if purely historical-static prior frozen
    pre-training; else 'unreported'.
D4: 'code-mismatch' only PCT (verified defect); 'parity-only' if we rebuilt from
    formula; 'no-code' if abstract-only/no code; else 'unverified'.
D5: 'harm-quantified' only if explicit negative result reported (Jensen-wake
    lambda3 sweep documented); else 'none-reported'.
"""
from __future__ import annotations
import csv

def d1(form):
    f = form.lower()
    if any(k in f for k in ['betz', 'rated', 'cut-in', 'cut-in', 'mechanical operating',
                            'nonneg', 'power coefficient bounds', 'pmsg', 'bem']):
        return 'intrinsic'
    if any(k in f for k in ['kde', 'js_div', 'jensen-shannon', 'mmd', 'kl']):
        return 'historical-static'
    if any(k in f for k in ['nwp', 'advection', 'jensen', 'wake', 'meteorological',
                            'bernoulli', 'ideal gas', 'atmospheric', 'power curve',
                            'aerodynamic power curve']):
        return 'forecast-derived'
    if 'diffusion' in f or 'tweedie' in f:
        return 'forecast-derived'
    return 'unreported'

def d2(form, notes):
    f = (form + ' ' + notes).lower()
    if any(k in f for k in ['diffusion', 'generative', 'denoising']):
        return 'data-generation'
    if any(k in f for k in ['reconstruction', 'latent', 'possible power', 'similarity']):
        return 'inference-architecture'
    if any(k in f for k in ['loss', 'penalty', 'regulariz', 'constraint', 'residual']):
        return 'training-loss'
    return 'unreported'

def d3(form, notes, inputv):
    f = (form + ' ' + notes + ' ' + inputv).lower()
    if any(k in f for k in ['latent', 'possible power', 'curtailment', 'available power']):
        return 'latent-state'
    if any(k in f for k in ['nwp', 'forecast', 'meteorological', 'advection', 'wake',
                            'atmospheric', 'jensen']):
        return 'forecast-input'
    if any(k in f for k in ['scada', 'upstream measured', 'contemporaneous', 'turbine state',
                            'pitch', 'lambda', 'beta']):
        return 'contemporaneous-state'
    if any(k in f for k in ['kde', 'js', 'mmd', 'historical']):
        return 'none'
    return 'unreported'

def d4(pid, notes, form):
    if pid == 'PCT':
        return 'code-mismatch'
    if 'abstract only' in notes.lower() or 'no code' in notes.lower():
        return 'no-code'
    if 'explicit' in notes.lower() and 'formula' in form.lower():
        return 'parity-only'
    return 'unverified'

def d5(notes):
    if 'lambda3' in notes.lower() and 'weight' in notes.lower():
        return 'harm-quantified'
    return 'none-reported'

def main():
    src = list(csv.DictReader(open('literature_coding_table_dual.csv', encoding='utf-8-sig')))
    out = []
    for r in src:
        form = r['constraint_math_form'] or ''
        notes = r['notes'] or ''
        inputv = r['input_variable'] or ''
        pid = r['paper_id']
        out.append((pid,
                    d1(form), d2(form, notes), d3(form, notes, inputv),
                    d4(pid, notes, form), d5(notes)))
    for pid, a, b, c, d, e in out:
        print(f'{pid:16s} D1={a:18s} D2={b:20s} D3={c:22s} D4={d:14s} D5={e}')
    import json
    json.dump({p: [a,b,c,d,e] for p,a,b,c,d,e in out},
              open('coderB_v3_D1D5.json','w',encoding='utf-8'), indent=1)
    print('---coder_B saved---')

if __name__ == '__main__':
    main()
