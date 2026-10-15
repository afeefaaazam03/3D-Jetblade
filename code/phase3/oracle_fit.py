"""Phase 3 diagnosis (new code; frozen estimator only imported). Why do visible dents get a weak fit gain?

For missed visible dents whose proposals already landed near the dent, rerun the frozen estimate() with the dent start
forced to the TRUE centre and radius ('loc'), and additionally the true depth / shape ('full').
If the oracle gain clears the threshold, the misses are an optimisation problem (fixable); if not, the signal is too weak.

    python phase3/oracle_fit.py <split>
Writes results/phase3/oracle_<split>.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from bladedelta import estimator as E  # noqa: E402
from bladedelta.data import mm_per_unit  # noqa: E402
from bladedelta.geometry import surface_context  # noqa: E402
from bladedelta.renderer import load_shader  # noqa: E402
from run_eval import rgb  # noqa: E402

B = ROOT / 'data/benchmark'
T = json.loads((ROOT / 'METHOD_FREEZE.json').read_text())['thresholds']['ours']['tau_alpha05']


def targets(split):
    C = {c['case_id']: c for c in json.loads((B / split / 'cases.json').read_text())}
    out = []
    for f in sorted((ROOT / f'results/{split}/ours').glob(f'{split}_*.json')):
        r = json.loads(f.read_text()); c = C[r['case_id']]
        if not c['dent'] or (r['visible_fraction'] or 0) < .5:
            continue
        v = surface_context(c['source'])['v']; tc = v[c['dent']['center']]
        pd = min(np.linalg.norm(v[p] - tc) for p in r['proposals'])
        found = r['fit_gain'] > T and r['pred_changed_vertices'] > 0 and r['centre_err'] is not None and r['centre_err'] < .1
        if not found and pd <= .1:
            out.append((c, r))
    return out


def run(c, shader, mode):
    d = c['dent']; orig_sp, orig_nb = E.Problem.score_proposals, E.Problem.new_block

    def sp(self, colour, k, kind='dent'):
        if kind != 'dent':
            return orig_sp(self, colour, k, kind)
        return [(int(d['center']), float(d['radius']))], {}

    def nb(self, vid, kind, radius=.08):
        blk = orig_nb(self, vid, kind, radius)
        if kind == 'dent' and mode == 'full':
            with torch.no_grad():
                blk['sdepth'].fill_(d['depth'] * 1e3)
                a = min(max(d.get('aspect', 1.), .21), .99)
                blk['la'].fill_(float(np.log((a - .2) / .8 / (1 - (a - .2) / .8))))
                blk['ang'].fill_(float(d.get('angle', 0.)))
        return blk
    E.Problem.score_proposals, E.Problem.new_block = sp, nb
    try:
        cfg = E.Config(); sid = c['source']; split = c['case_id'].rsplit('_', 2)[0]
        imgs = [rgb(B / split / 'renders' / c['case_id'] / f'{v}.png') for v in cfg.views]
        refs = [rgb(B / split / 'refs' / f'ref_{sid:04d}' / f'{v}.png') for v in cfg.views]
        return E.estimate(sid, imgs, shader, cfg, bias=refs)
    finally:
        E.Problem.score_proposals, E.Problem.new_block = orig_sp, orig_nb


def main():
    split = sys.argv[1]; out = ROOT / f'results/phase3/oracle_{split}.jsonl'; out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['case_id'] for l in out.read_text().splitlines()} if out.exists() else set()
    shader = load_shader()
    for c, r in targets(split):
        if c['case_id'] in done:
            continue
        mm = mm_per_unit(c['source']); row = dict(case_id=c['case_id'], depth_mm=r['truth_max_disp'] * mm, gain_frozen=r['fit_gain'], tau=T)
        for mode in ('loc', 'full'):
            res = run(c, shader, mode)
            row[f'gain_{mode}'] = float(res['fit_gain']); row[f'depth_{mode}_mm'] = float(res['depth']) * mm
        with out.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(row, flush=True)


if __name__ == '__main__':
    main()
