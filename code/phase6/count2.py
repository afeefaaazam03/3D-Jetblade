"""Phase 6, task 2, version 2 (new code): greedy explain-away counting.

H0 = nuisance + best stain patch (as the frozen estimator). Then repeat: score-test proposals (NMS 0.15 span) on the
current explanation, fit a dent at each, accept the best if its extra evidence exceeds the dent class threshold
(task 3), bake it into the model and continue. Count = number of accepted dents.

    python phase6/count2.py <split> [--defects ...] [--max N]
Writes results/phase6/count2_<split>.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'scripts'):
    sys.path.insert(0, str(p))
from bladedelta.estimator import Config  # noqa: E402
from bladedelta.v1_1 import ProblemV11  # noqa: E402
from bladedelta.renderer import load_shader  # noqa: E402
from run_eval import rgb  # noqa: E402

B = ROOT / 'data/benchmark'
cal = [json.loads(l) for l in (ROOT / 'results/phase6/gains_cad6.jsonl').read_text().splitlines()]
g = sorted(r['gain_dent'] for r in cal); TAU = g[min(len(g) - 1, int(np.ceil((len(g) + 1) * .95)) - 1)]
K, NMS, MAXN = 4, .15, 4


def count_case(c, split, shader, cfg):
    sid, cid = c['source'], c['case_id']
    imgs = [rgb(B / split / 'renders' / cid / f'{x}.png') for x in cfg.views]; refs = [rgb(B / split / 'refs' / f'ref_{sid:04d}' / f'{x}.png') for x in cfg.views]
    torch.manual_seed(0); pb = ProblemV11(sid, imgs, shader, cfg, refs)
    pb.fit_nuisance(); pb.set_interior_mask(cfg.interior_margin_px)
    lcur, _ = pb.optimise(None, None, cfg.iters_h0 // 2, cfg.h_nuisance); best_col = None
    nuis = {k: v.detach().clone() for k, v in pb.nuis.items()}
    for vid in [x for x, _ in pb.score_proposals(None, cfg.starts, kind='stain')[0]]:
        for k, t in nuis.items(): pb.nuis[k].data.copy_(t)
        col = pb.new_block(vid, 'colour', .08); l, _ = pb.optimise(None, col, cfg.iters_h0, cfg.h_nuisance)
        if l < lcur: lcur, best_col, nbest = l, col, {k: v.detach().clone() for k, v in pb.nuis.items()}
    if best_col is not None: nuis = nbest
    acc = []
    for _ in range(MAXN):
        for k, t in nuis.items(): pb.nuis[k].data.copy_(t)
        old = cfg.proposal_nms; cfg.proposal_nms = NMS; props, _ = pb.score_proposals(best_col, K); cfg.proposal_nms = old
        v = pb.ctx['v']; props = [(vid, r) for vid, r in props if all(np.linalg.norm(v[vid] - np.asarray(a['centre'])) > NMS for a in acc)]
        cand = None
        for vid, r0 in props:
            for k, t in nuis.items(): pb.nuis[k].data.copy_(t)
            d = pb.new_block(vid, 'dent', r0); l, _ = pb.optimise(d, best_col, cfg.iters_h1, cfg.h_nuisance)
            if cand is None or l < cand[0]: cand = (l, d, {k: t.detach().clone() for k, t in pb.nuis.items()})
        if cand is None or lcur - cand[0] <= TAU: break
        l, d, n2 = cand
        with torch.no_grad():
            ctr = (pb.v[d['vid']] + d['uv'][0] * d['e1'] + d['uv'][1] * d['e2']).cpu().numpy(); pb.v = pb.v + pb.dent(d).detach()   # bake the accepted dent
        acc.append(dict(gain=float(lcur - l), centre=ctr.tolist())); lcur, nuis = l, n2
    v = pb.ctx['v']
    truth = [d['center'] for d in c.get('defects', []) if d['type'] != 'stain'] or ([c['dent']['center']] if c.get('dent') and c.get('defect', 'dent') != 'stain' else [])
    matched = sum(any(np.linalg.norm(np.asarray(a['centre']) - v[t]) < .1 for a in acc) for t in truth)
    return dict(case_id=cid, true_count=len(truth), count=len(acc), matched=matched, detections=acc, tau=TAU)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('split'); ap.add_argument('--defects', default=''); ap.add_argument('--max', type=int, default=0)
    a = ap.parse_args(); defects = {x for x in a.defects.split(',') if x}
    out = ROOT / f'results/phase6/count2_{a.split}.jsonl'; done = {json.loads(l)['case_id'] for l in out.read_text().splitlines()} if out.exists() else set()
    cfg = Config(); shader = load_shader(); per = {}
    for c in json.loads((B / a.split / 'cases.json').read_text()):
        d = c.get('defect', c['kind'])
        if c['case_id'] in done or (defects and d not in defects) or (a.max and per.get(d, 0) >= a.max): continue
        per[d] = per.get(d, 0) + 1; row = count_case(c, a.split, shader, cfg)
        with out.open('a') as f: f.write(json.dumps(row, default=float) + '\n')
        print(c['case_id'], 'true', row['true_count'], 'counted', row['count'], 'matched', row['matched'], flush=True)


if __name__ == '__main__':
    main()
