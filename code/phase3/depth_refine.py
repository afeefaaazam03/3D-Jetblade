"""Phase 3 (new code; frozen v1.1 only imported): multi-depth refinement of an already reported dent.

The detect/abstain decision is unchanged (frozen fit gain vs frozen threshold). For a reported dent only, the dent is
refit at the reported location and radius from several starting depths; the lowest-loss fit (including the frozen one)
gives the depth. False alarms therefore cannot change; only the depth estimate can.

    python phase3/depth_refine.py <split> [case_id ...]
Writes results/phase3/refine_<split>.jsonl (resumable).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from bladedelta import estimator as E, v1_1  # noqa: E402
from bladedelta.data import mm_per_unit  # noqa: E402
from bladedelta.geometry import surface_context  # noqa: E402
from bladedelta.renderer import load_shader  # noqa: E402
from run_eval import rgb  # noqa: E402

B = ROOT / 'data/benchmark'
T = json.loads((ROOT / 'METHOD_FREEZE_V1_1.json').read_text())['thresholds']['v11']['tau_alpha05']
INIT_MM = (.2, .5, 1.)


def refit(c, r, shader, init_mm):
    sid = c['source']; mm = mm_per_unit(sid); v = surface_context(sid)['v']
    vid = int(np.argmin(np.linalg.norm(v - np.asarray(r['centre']), axis=1)))
    orig_sp, orig_nb = E.Problem.score_proposals, E.Problem.new_block

    def sp(self, colour, k, kind='dent'):
        return orig_sp(self, colour, k, kind) if kind != 'dent' else ([(vid, float(r['radius']))], {})

    def nb(self, vid_, kind, radius=.08):
        blk = orig_nb(self, vid_, kind, radius)
        if kind == 'dent':
            with torch.no_grad():
                blk['sdepth'].fill_(init_mm / mm * 1e3)
        return blk
    E.Problem.score_proposals, E.Problem.new_block = sp, nb
    try:
        cfg = E.Config(); split = c['case_id'].rsplit('_', 2)[0]
        imgs = [rgb(B / split / 'renders' / c['case_id'] / f'{x}.png') for x in cfg.views]
        refs = [rgb(B / split / 'refs' / f'ref_{sid:04d}' / f'{x}.png') for x in cfg.views]
        return v1_1.estimate_v11(sid, imgs, shader, cfg, bias=refs)
    finally:
        E.Problem.score_proposals, E.Problem.new_block = orig_sp, orig_nb


def main():
    split = sys.argv[1]; only = set(sys.argv[2:])
    out = ROOT / f'results/phase3/refine_{split}.jsonl'; out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['case_id'] for l in out.read_text().splitlines()} if out.exists() else set()
    C = {c['case_id']: c for c in json.loads((B / split / 'cases.json').read_text())}
    shader = load_shader()
    for f in sorted((ROOT / f'results/{split}/v11').glob(f'{split}_*.json')):
        r = json.loads(f.read_text()); cid = r['case_id']
        if cid in done or (only and cid not in only) or not (r['fit_gain'] > T and r['pred_changed_vertices'] > 0):
            continue
        c = C[cid]; mm = mm_per_unit(c['source'])
        best = dict(init='frozen', loss_h1=r['loss_h1'], gain=r['fit_gain'], depth_mm=r['pred_max_disp'] * mm)
        tries = [dict(best)]
        for d0 in INIT_MM:
            res = refit(c, r, shader, d0)
            t = dict(init=d0, loss_h1=float(res['loss_h1']), gain=float(res['fit_gain']), depth_mm=float(np.abs(res['displacement']).max()) * mm)
            tries.append(t)
            if t['loss_h1'] < best['loss_h1']:
                best = t
        row = dict(case_id=cid, kind=c['kind'], source=c['source'],
                   true_mm=float(r['truth_max_disp'] or 0) * mm, frozen_mm=r['pred_max_disp'] * mm, refined_mm=best['depth_mm'],
                   chosen=best['init'], centre_ok=r.get('centre_err') is not None and r['centre_err'] < .1, tries=tries)
        with out.open('a') as fo:
            fo.write(json.dumps(row) + '\n')
        print(cid, 'true %.3f frozen %.3f refined %.3f (%s)' % (row['true_mm'], row['frozen_mm'], row['refined_mm'], best['init']), flush=True)


if __name__ == '__main__':
    main()
