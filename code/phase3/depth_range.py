"""Phase 3 (new code; frozen v1.1 only imported): depth ranges for reported dents (profile likelihood).

For a dent v1.1 reported, fit H0 once (as the frozen estimator does), then refit the dent at the reported location and
radius with its depth FIXED on a grid, everything else free. A depth is "consistent with the photos" if its loss is within
tau (the frozen detection threshold) of the best: the photos cannot tell it apart from the best depth any better than they
can tell a dent from no dent. The range is reported as [lo, hi]; hi = open if the deepest grid value is still consistent.

    python phase3/depth_range.py <split> [--blades 1003,1004,...]
Writes results/phase3/range_<split>.jsonl (resumable).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from bladedelta.estimator import Config  # noqa: E402
from bladedelta.v1_1 import ProblemV11  # noqa: E402
from bladedelta.data import mm_per_unit  # noqa: E402
from bladedelta.geometry import surface_context  # noqa: E402
from bladedelta.renderer import load_shader  # noqa: E402
from run_eval import rgb  # noqa: E402

B = ROOT / 'data/benchmark'
T = json.loads((ROOT / 'METHOD_FREEZE_V1_1.json').read_text())['thresholds']['v11']['tau_alpha05']
GRID_MM = (.02, .05, .1, .2, .35, .5, .75, 1., 1.5)


def h0(pb, cfg):
    pb.fit_nuisance(); pb.set_interior_mask(cfg.interior_margin_px)
    l0, _ = pb.optimise(None, None, cfg.iters_h0 // 2, cfg.h_nuisance)
    return l0, {k: v.detach().clone() for k, v in pb.nuis.items()}


def fit_at(pb, cfg, nuis0, vid, r, depth_units=None):
    for k, v in nuis0.items():
        pb.nuis[k].data.copy_(v)
    d = pb.new_block(vid, 'dent', r)
    if depth_units is not None:
        with torch.no_grad():
            d['sdepth'].fill_(depth_units * 1e3)
        d['sdepth'].requires_grad_(False)
    l, _ = pb.optimise(d, None, cfg.iters_h1, cfg.h_nuisance)
    return l, float(d['sdepth'].clamp_min(0)) * 1e-3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('split'); ap.add_argument('--blades', default='')
    a = ap.parse_args()
    blades = {int(x) for x in a.blades.split(',') if x}
    out = ROOT / f'results/phase3/range_{a.split}.jsonl'; out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['case_id'] for l in out.read_text().splitlines()} if out.exists() else set()
    C = {c['case_id']: c for c in json.loads((B / a.split / 'cases.json').read_text())}
    cfg = Config(); shader = load_shader()
    for f in sorted((ROOT / f'results/{a.split}/v11').glob(f'{a.split}_*.json')):
        r = json.loads(f.read_text()); cid = r['case_id']; c = C[cid]; sid = c['source']
        if cid in done or (blades and sid not in blades) or not (r['fit_gain'] > T and r['pred_changed_vertices'] > 0):
            continue
        if not c['dent'] or r.get('centre_err') is None or r['centre_err'] >= .1:
            continue                                           # depth ranges only for correctly located dents
        mm = mm_per_unit(sid); v = surface_context(sid)['v']
        vid = int(np.argmin(np.linalg.norm(v - np.asarray(r['centre']), axis=1)))
        imgs = [rgb(B / a.split / 'renders' / cid / f'{x}.png') for x in cfg.views]
        refs = [rgb(B / a.split / 'refs' / f'ref_{sid:04d}' / f'{x}.png') for x in cfg.views]
        torch.manual_seed(0)
        pb = ProblemV11(sid, imgs, shader, cfg, refs)
        l0, nuis0 = h0(pb, cfg)
        lfree, dfree = fit_at(pb, cfg, nuis0, vid, float(r['radius']))
        d0 = max(dfree * mm, .01)
        grid = sorted({round(x, 4) for x in [d0 * f for f in (.25, .5, .7, .85, 1.15, 1.4, 2, 3, 5)] + list(GRID_MM)})
        prof = [(dm, fit_at(pb, cfg, nuis0, vid, float(r['radius']), dm / mm)[0]) for dm in grid]
        best = min([lfree] + [l for _, l in prof])
        ok = [dm for dm, l in prof if l <= best + T] + [dfree * mm]
        lo = min(ok)
        hi = None if (ok and max(ok) == grid[-1]) else max(ok)
        true_mm = float(r['truth_max_disp']) * mm
        row = dict(case_id=cid, source=sid, true_mm=true_mm, point_mm=r['pred_max_disp'] * mm, lo_mm=lo, hi_mm=hi,
                   covered=bool(true_mm >= lo * .999 and (hi is None or true_mm <= hi * 1.001)),
                   l0=l0, l_free=lfree, profile=prof)
        with out.open('a') as fo:
            fo.write(json.dumps(row) + '\n')
        print(cid, 'true %.3f point %.3f range [%.2f, %s] covered=%s' % (true_mm, row['point_mm'], lo, 'open' if hi is None else '%.2f' % hi, row['covered']), flush=True)


if __name__ == '__main__':
    main()
