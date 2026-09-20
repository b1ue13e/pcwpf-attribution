"""Read-only manuscript/evidence inspection; no training or source modification."""
from pathlib import Path
import csv
import json
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SCANNER = Path(r'C:\Users\lidong\.agents\skills\ai-detox\scripts\scan_ai_style.py')

def rows(path):
    with (ROOT / path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

report = {}
for language, folder in [('en', 'active_manuscript'), ('zh', 'chinese_manuscript')]:
    base = ROOT / '01_论文投稿材料' / folder
    paths = [base / 'main.tex'] + sorted((base / 'sections').glob('*.tex'))
    merged = '\n\n'.join('% SOURCE: ' + str(p.relative_to(ROOT)) + '\n' + p.read_text(encoding='utf-8-sig') for p in paths)
    source = OUT / f'manuscript_scan_input_{language}.tex'
    source.write_text(merged, encoding='utf-8')
    result = subprocess.run([sys.executable, str(SCANNER), str(source)], capture_output=True, encoding='utf-8')
    (OUT / f'style_scan_{language}.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
    report[language] = {'files': len(paths), 'characters': len(merged), 'scanner_exit_code': result.returncode}

ab = rows('06_实验结果与日志/ablation_3d_results/ablation_3d_master_summary.csv')
report['ablation_seeds'] = sorted({int(r['seed']) for r in ab})
groups = defaultdict(list)
for r in ab:
    if r['policy'] == 'calibrated' and float(r['ratio']) == 2:
        groups[(r['site'], r['arm'])].append(float(r['cost_static']))
report['ablation_selected'] = {}
for (site, arm), vals in groups.items():
    if arm in ['vanilla_unconstrained', 'genuine_curve_l1.0', 'pseudo_scrambled_curve_l1.0', 'genuine_curve_l0.01', 'pseudo_scrambled_curve_l0.01'] or arm.startswith('lag_scan_curve'):
        report['ablation_selected'][site + '/' + arm] = {'n': len(vals), 'mean': statistics.mean(vals), 'std': statistics.stdev(vals)}
report['penmanshiel_effects_pct'] = {arm: 100*(statistics.mean(groups[('penmanshiel', arm)])/statistics.mean(groups[('penmanshiel', 'vanilla_unconstrained')])-1) for arm in ['genuine_curve_l1.0','pseudo_scrambled_curve_l1.0']}

ledger = rows('06_实验结果与日志/constraint_round2/analysis/corpus46_scoping_ledger.csv')
report['corpus'] = {'rows':len(ledger), 'columns': list(ledger[0]), 'categorical_counts': {k: dict(Counter(r[k] for r in ledger)) for k in ledger[0] if len({r[k] for r in ledger}) <= 12}}
included = [r for r in ledger if r['inclusion_check'] == 'include']
report['corpus']['included_counts'] = {k:dict(Counter(r[k] for r in included)) for k in ['D4_fidelity','ablation_type','d5_negative_effect_reported']}

report['tables'] = {}
for p in (ROOT / '01_论文投稿材料/active_manuscript/tables').glob('*.tex'):
    s = p.read_text(encoding='utf-8-sig')
    report['tables'][p.name] = {'caption': re.search(r'\\caption\{([^\n]+)',s).group(1) if '\\caption{' in s else ''}

report['learning_effects_pct'] = []
learning = rows('06_实验结果与日志/learning_curve/within_site_learning_curve_summary.csv')
for site in ['kelmarsh', 'hill_of_towie']:
    for n in ['500', '5000']:
        g = {r['arm']: float(r['mean']) for r in learning if r['site']==site and r['n_train']==n}
        report['learning_effects_pct'].append({'site':site,'n':n,'genuine':100*(g['genuine_prior']/g['unconstrained']-1),'scrambled':100*(g['scrambled_prior']/g['unconstrained']-1)})

skip = {'.git','.venv','.venv_broken_20260525_223330','.wheelhouse','__pycache__','node_modules','90_环境与第三方依赖'}
inventory = []
def walk(path):
    for p in path.iterdir():
        if p.name in skip:
            continue
        if p.is_dir():
            walk(p)
        elif p.is_file():
            inventory.append(str(p.relative_to(ROOT)))
walk(ROOT)
(OUT / 'project_file_inventory.txt').write_text('\n'.join(inventory), encoding='utf-8')
report['inventory_by_top_directory'] = dict(Counter(p.split('\\')[0] for p in inventory))
report['inventory_files'] = len(inventory)
(OUT / 'evidence_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['corpus','tables']}, ensure_ascii=False, indent=2))
print('Corpus:',json.dumps(report['corpus'],ensure_ascii=False))
