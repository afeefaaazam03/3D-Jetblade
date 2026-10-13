"""Table: calibrated threshold vs textbook statistical thresholds (sealed test). New; reads results only."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import chi2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bladedelta import evaluate as ev  # noqa: E402
from bladedelta.metrics import conformal_threshold  # noqa: E402
EPS, K = 2 / 255, 8
H = {json.loads(l)['case_id']: json.loads(l) for l in (ROOT / 'results/phase2/h0stats_test.jsonl').read_text().splitlines()}
R = [r for r in ev.rows('test', 'ours') if r['case_id'] in H]
neg = [r for r in R if r['kind'] in ev.NEG]; obs = [r for r in R if r['kind'] in ev.DENT and (r['visible_fraction'] or 0) >= ev.OBS]
cl = lambda r, t: r['fit_gain'] > t and r['pred_changed_vertices'] > 0
loc = lambda r, t: cl(r, t) and r.get('centre_err') is not None and r['centre_err'] < ev.LOC
chi = lambda r, a: np.mean(np.square(H[r['case_id']]['sigma'])) * chi2.ppf(1 - a, K) / (2 * EPS)
cal360 = [ev.gain(r) for s in ('calib', 'calib2') for r in ev.rows(s, 'ours') if r['kind'] in ev.NEG]
rows = []
t = json.loads((ROOT / 'METHOD_FREEZE.json').read_text())['thresholds']['ours']['tau_alpha05']
rows.append(('Ours: conformal on 60 calibration negatives', '5', np.mean([cl(r, t) for r in neg]), np.mean([loc(r, t) for r in obs])))
t2 = conformal_threshold(cal360, .02)
rows.append(('Ours: conformal on 360 calibration negatives', '2', np.mean([cl(r, t2) for r in neg]), np.mean([loc(r, t2) for r in obs])))
for a in (.05, .01):
    rows.append((f'Textbook $\\chi^2_{{{K}}}$ likelihood-ratio threshold', f'{a * 100:.0f}', np.mean([cl(r, chi(r, a)) for r in neg]), np.mean([loc(r, chi(r, a)) for r in obs])))
rows.append(('A-contrario detector on the $H_0$ residual (NFA $<1$)', '$\\approx$0', np.mean([H[r['case_id']]['nfa'] < 1 for r in neg]), np.mean([H[r['case_id']]['nfa'] < 1 for r in obs])))
L = [r'\begin{table}[t]\centering\small', r'\begin{tabular}{lccc}', r'\toprule', r'Decision rule & Target FA (\%) & Actual FA (\%) & Visible dents found (\%)\\', r'\midrule']
L += [f"{n} & {tg} & {fa * 100:.0f} & {f * 100:.0f}\\\\" for n, tg, fa, f in rows]
L += [r'\bottomrule', r'\end{tabular}', r'\caption{Calibrated versus textbook decision rules on the same sealed-test fits. FA: claims on the 90 healthy or stain-only cases. Found: visible dents located (for the a-contrario detector, any detection). The textbook rules assume i.i.d.\ image noise; the real nuisance mismatch is far larger, so they cannot control false alarms.}', r'\label{tab:statbaselines}', r'\end{table}']
(ROOT / 'paper/tables/stat_baselines.tex').write_text('\n'.join(L) + '\n')
for n, tg, fa, f in rows: print(f'{n[:50]:50s} target {tg:>4} actual {fa*100:3.0f}% found {f*100:3.0f}%')
