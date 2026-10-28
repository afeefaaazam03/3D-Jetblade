"""Phase 6, task 3 (new code): per-class evidence for every case, for class-specific thresholds.

For each case: fit the nuisance-only explanation H0 (with the stain search, as frozen), take the top score-test proposal,
and fit each physical specialist there (dent, carved-notch nick, dark-line crack; phase-5 round-2 models).
gain_k = L(H0) - L(H0 + specialist k). Also records the stain gain (H0 without vs with the stain patch) and crack length.

    python phase6/class_gains.py <split> [--blades 1001,1002] [--kinds healthy,colour]
Writes results/phase6/gains_<split>.jsonl (resumable).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'scripts', ROOT / 'phase4', ROOT / 'phase5'):
    sys.path.insert(0, str(p))
import specialists as S1  # noqa: E402
import specialists2 as S2  # noqa: E402
from bladedelta.estimator import Config  # noqa: E402
from bladedelta.data import mm_per_unit  # noqa: E402
from bladedelta.geometry import surface_context  # noqa: E402
from bladedelta.renderer import load_shader  # noqa: E402
from run_eval import rgb  # noqa: E402

B = ROOT / 'data/benchmark'


def run(c, split, shader, cfg, edges, edge_len):
    sid, cid = c['source'], c['case_id']
    imgs = [rgb(B / split / 'renders' / cid / f'{x}.png') for x in cfg.views]
    refs = [rgb(B / split / 'refs' / f'ref_{sid:04d}' / f'{x}.png') for x in cfg.views]
    torch.manual_seed(0); pb = S2.Specialists2(sid, imgs, shader, cfg, refs)
    pb.fit_nuisance(); pb.set_interior_mask(cfg.interior_margin_px)
    l_none, _ = pb.optimise(None, None, cfg.iters_h0 // 2, cfg.h_nuisance); nuis0 = {k: v.detach().clone() for k, v in pb.nuis.items()}
    l_h0, col0, nuis_col = l_none, None, nuis0
    for vid in [x for x, _ in pb.score_proposals(None, cfg.starts, kind='stain')[0]]:
        for k, v in nuis0.items(): pb.nuis[k].data.copy_(v)
        col = pb.new_block(vid, 'colour', .08); l, _ = pb.optimise(None, col, cfg.iters_h0, cfg.h_nuisance)
        if l < l_h0: l_h0, col0, nuis_col = l, col, {k: v.detach().clone() for k, v in pb.nuis.items()}
    for k, v in nuis0.items(): pb.nuis[k].data.copy_(v)
    (vid, r0), = pb.score_proposals(None, 1)[0][:1] or [((int(np.flatnonzero(pb.airfoil.cpu().numpy())[0]), .06))]
    v = pb.ctx['v']; ed = np.flatnonzero(edges); evid = int(ed[np.argmin(np.linalg.norm(v[ed] - v[vid], axis=1))])
    starts = [(nuis0, None)] + ([(nuis_col, col0)] if col0 is not None else [])
    makers = {'dent': [lambda: pb.new_block(vid, 'dent', r0)],
              'nick': [lambda a=a: S2.block2(pb, 'nick2', a, R=.04, ang=0.) for a in {vid, evid}],
              'crack': [lambda ang=ang: S2.block2(pb, 'crack2', vid, width=1.2 * edge_len, ang=ang) for ang in np.radians([0, 60, 120])]}
    mm = mm_per_unit(sid); out = dict(case_id=cid, source=sid, defect=c.get('defect', c['kind']), l_none=l_none, l_h0=l_h0, stain_gain=l_none - l_h0, proposal=int(vid))
    for kind, mk in makers.items():
        best = (np.inf, None)
        for m in mk:
            for st, col in starts:
                l, b = S1.fit(pb, cfg, st, m(), col)
                if l < best[0]: best = (l, b)
        b = best[1]; out[f'gain_{kind}'] = l_h0 - best[0]; out[f'depth_{kind}_mm'] = float(b['sdepth'].clamp_min(0)) * 1e-3 * mm
        out[f'centre_{kind}'] = S1.centre(pb, b).tolist()
        if kind == 'crack': out['crack_length_mm'] = float(torch.exp(b['lr'])) * 2 * mm
    if c.get('dent'):
        out['centre_err_proposal'] = float(np.linalg.norm(v[vid] - v[c['dent']['center']]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('split'); ap.add_argument('--blades', default=''); ap.add_argument('--kinds', default=''); ap.add_argument('--defects', default=''); ap.add_argument('--out', default='')
    a = ap.parse_args(); blades = {int(x) for x in a.blades.split(',') if x}; kinds = {x for x in a.kinds.split(',') if x}; defects = {x for x in a.defects.split(',') if x}
    out = ROOT / f'results/phase6/gains_{a.split}{a.out}.jsonl'; out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['case_id'] for l in out.read_text().splitlines()} if out.exists() else set()
    cfg = Config(); shader = load_shader(); cache = {}
    for c in json.loads((B / a.split / 'cases.json').read_text()):
        if c['case_id'] in done or (blades and c['source'] not in blades) or (kinds and c['kind'] not in kinds) or (defects and c.get('defect') not in defects): continue
        sid = c['source']
        if sid not in cache:
            ctx = surface_context(sid); e = ctx['v'][ctx['airfoil'][:, 0]] - ctx['v'][ctx['airfoil'][:, 1]]
            cache[sid] = (S1.edge_vertices(ctx) & (ctx['boundary_distance'] >= .1), float(np.median(np.linalg.norm(e, axis=1))))
        row = run(c, a.split, shader, cfg, *cache[sid])
        with out.open('a') as f: f.write(json.dumps(row, default=float) + '\n')
        print(c['case_id'], row['defect'], ' '.join(f"{k}={row['gain_' + k]:.2f}" for k in ('dent', 'nick', 'crack')), 'stain=%.2f' % row['stain_gain'], flush=True)


if __name__ == '__main__':
    main()
