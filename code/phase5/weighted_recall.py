"""Phase 5 (new code): safety-weighted recall alongside standard recall, from existing results (no new runs).

Weight of a visible defect = its true depth in mm (deeper = more dangerous); cracks count double (they can grow).
  standard recall = located / visible defects
  weighted recall = sum(weight of located) / sum(weight of visible)
  critical recall = located / visible defects that are >= 0.2 mm deep or are cracks
Located = reported (fit gain > frozen threshold) and centre within 0.1 blade span.

    python phase5/weighted_recall.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bladedelta.data import mm_per_unit  # noqa: E402

T1 = json.loads((ROOT / 'METHOD_FREEZE.json').read_text())['thresholds']['ours']['tau_alpha05']
T11 = json.loads((ROOT / 'METHOD_FREEZE_V1_1.json').read_text())['thresholds']['v11']['tau_alpha05']
SETS = [('Rotor37 sealed test (v1)', 'test', 'ours', T1, None), ('Rotor37 second test (v1.1)', 'test2', 'v11', T11, None),
        ('Industrial held-out (v1.1)', 'cad6', 'v11', T11, (1003, 1006)), ('Four defects (v1.1)', 'cad6dmg', 'v11', T11, None)]


def main():
    rows = []
    for name, split, meth, tau, blades in SETS:
        C = {c['case_id']: c for c in json.loads((ROOT / f'data/benchmark/{split}/cases.json').read_text())}
        items = []
        for f in sorted((ROOT / f'results/{split}/{meth}').glob(f'{split}_*.json')):
            r = json.loads(f.read_text()); c = C[r['case_id']]
            if not c.get('dent') or (blades and not blades[0] <= c['source'] <= blades[1]):
                continue
            if split != 'cad6dmg' and (r.get('visible_fraction') or 0) < .5:
                continue                                   # cad6dmg defects are all placed in view of both cameras
            depth = float(r['truth_max_disp']) * mm_per_unit(c['source'])
            crack = c.get('defect') == 'crack'
            located = r['fit_gain'] > tau and r['pred_changed_vertices'] > 0 and r.get('centre_err') is not None and r['centre_err'] < .1
            items.append((depth * (2 if crack else 1), located, depth >= .2 or crack))
        w = np.array([x[0] for x in items]); L = np.array([x[1] for x in items]); crit = np.array([x[2] for x in items])
        rows.append(dict(set=name, n=len(items), recall=L.mean(), weighted=(w * L).sum() / w.sum(),
                         n_critical=int(crit.sum()), critical=L[crit].mean() if crit.any() else float('nan')))
    out = ROOT / 'results/phase5/weighted_recall.json'; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print('%-30s %4s %9s %9s %14s' % ('set', 'n', 'recall', 'weighted', 'critical'))
    for r in rows:
        print('%-30s %4d %8.0f%% %8.0f%% %8.0f%% (n=%d)' % (r['set'], r['n'], 100 * r['recall'], 100 * r['weighted'], 100 * r['critical'], r['n_critical']))


if __name__ == '__main__':
    main()
