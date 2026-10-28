"""Phase 6, task 2 (new code): defect counting with non-maximum suppression.

After the nuisance-only fit (H0, as frozen), take the top K score-test proposals that are spatially separated (NMS at
0.15 blade span), fit a dent independently at each (from the H0 state), and count the proposals whose own evidence
gain exceeds the frozen threshold tau. Accepted detections closer than the NMS radius are merged.

    python phase6/count.py <split> [--defects dent,stain,healthy] [--max N]
Writes results/phase6/count_<split>.jsonl
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
T = json.loads((ROOT / 'METHOD_FREEZE_V1_1.json').read_text())['thresholds']['v11']['tau_alpha05']
K, NMS = 5, .15


def count_case(c, split, shader, cfg):
    sid, cid = c['source'], c['case_id']
    imgs = [rgb(B / split / 'renders' / cid / f'{x}.png') for x in cfg.views]; refs = [rgb(B / split / 'refs' / f'ref_{sid:04d}' / f'{x}.png') for x in cfg.views]
    torch.manual_seed(0); pb = ProblemV11(sid, imgs, shader, cfg, refs)
    pb.fit_nuisance(); pb.set_interior_mask(cfg.interior_margin_px)
    l0, _ = pb.optimise(None, None, cfg.iters_h0 // 2, cfg.h_nuisance); nuis0 = {k: v.detach().clone() for k, v in pb.nuis.items()}
    old = cfg.proposal_nms; cfg.proposal_nms = NMS
    props, _ = pb.score_proposals(None, K); cfg.proposal_nms = old
    v = pb.ctx['v']; acc = []
    for vid, r0 in props:
        for k, t in nuis0.items(): pb.nuis[k].data.copy_(t)
        d = pb.new_block(vid, 'dent', r0); l1, _ = pb.optimise(d, None, cfg.iters_h1, cfg.h_nuisance); g = l0 - l1
        with torch.no_grad(): ctr = (pb.v[d['vid']] + d['uv'][0] * d['e1'] + d['uv'][1] * d['e2']).cpu().numpy()
        if g > T and all(np.linalg.norm(ctr - a['centre']) > NMS for a in acc):
            acc.append(dict(vid=int(vid), gain=float(g), centre=ctr.tolist()))
    truth = [d['center'] for d in c.get('defects', [])] or ([c['dent']['center']] if c.get('dent') else [])
    ttype = [d['type'] for d in c.get('defects', [])] or [c.get('defect', c['kind'])]
    phys = [t for t, ty in zip(truth, ttype) if ty != 'stain']
    matched = sum(any(np.linalg.norm(np.asarray(a['centre']) - v[t]) < .1 for a in acc) for t in phys)
    return dict(case_id=cid, true_count=len(phys), count=len(acc), matched=matched, detections=acc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('split'); ap.add_argument('--defects', default=''); ap.add_argument('--max', type=int, default=0)
    a = ap.parse_args(); defects = {x for x in a.defects.split(',') if x}
    out = ROOT / f'results/phase6/count_{a.split}.jsonl'; done = {json.loads(l)['case_id'] for l in out.read_text().splitlines()} if out.exists() else set()
    cfg = Config(); shader = load_shader(); n = 0; per = {}
    for c in json.loads((B / a.split / 'cases.json').read_text()):
        d = c.get('defect', c['kind'])
        if c['case_id'] in done or (defects and d not in defects): continue
        if a.max and per.get(d, 0) >= a.max: continue
        per[d] = per.get(d, 0) + 1
        row = count_case(c, a.split, shader, cfg)
        with out.open('a') as f: f.write(json.dumps(row, default=float) + '\n')
        print(c['case_id'], 'true', row['true_count'], 'counted', row['count'], 'matched', row['matched'], flush=True)


if __name__ == '__main__':
    main()
