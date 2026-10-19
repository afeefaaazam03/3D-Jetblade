"""Phase 5 (new code; frozen v1.1 only imported): damage-type specialists + judge.

Detection is unchanged: the frozen v1.1 evidence test decides "physical damage reported" or not, so false alarms are the
frozen ones. For every case we then fit, through the same differentiable renderer and nuisance model:
  stain  - colour-only patch (the frozen H0 stain search),
and, for reported cases, at the reported location:
  dent   - the frozen smooth dent model,
  nick   - material removed into the surface around an edge point, Gaussian fall-off (generator uses a polynomial one),
  crack  - narrow groove along a straight segment with Gaussian cross-section (generator uses a parabolic one).
Judge: reported -> type = specialist with the lowest loss; "type uncertain" if the runner-up is within tau (the frozen
detection threshold, i.e. the photos prefer the winner by less than they must prefer a dent over no dent).
Not reported -> "stain" if the stain patch gains more than tau over nuisance only, else "nothing".

    python phase5/specialists.py cad6dmg
Writes results/phase5/specialists_<split>.jsonl (resumable).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'phase4'))
from bladedelta.damage import tangent_frame  # noqa: E402
from bladedelta.estimator import DEV, Config  # noqa: E402
from bladedelta.v1_1 import ProblemV11  # noqa: E402
from bladedelta.renderer import load_shader  # noqa: E402
from build_dmg4 import edge_vertices  # noqa: E402
from run_eval import rgb  # noqa: E402

B = ROOT / 'data/benchmark'
T = json.loads((ROOT / 'METHOD_FREEZE_V1_1.json').read_text())['thresholds']['v11']['tau_alpha05']


def frame(pb, vid):
    nc = pb.n[vid].detach()
    e1, e2 = tangent_frame(nc.cpu().numpy().astype(np.float64))
    return nc, torch.tensor(e1, dtype=torch.float32, device=DEV), torch.tensor(e2, dtype=torch.float32, device=DEV)


def param(x):
    return torch.tensor(float(x), device=DEV, requires_grad=True)


class Specialists(ProblemV11):
    def dent(self, d):
        if d is None or d.get('type', 'dent') == 'dent':
            return super().dent(d)
        p = self.v[d['vid']] + d['uv'][0] * d['e1'] + d['uv'][1] * d['e2']
        depth = d['sdepth'].clamp_min(0) * 1e-3
        if d['type'] == 'nick':
            r = torch.exp(d['lr'])
            q2 = ((self.v - p) ** 2).sum(1) / r ** 2
            w = torch.exp(-3 * q2) * self.airfoil
            return -(depth * w)[:, None] * d['nc'][None]
        # crack
        half = torch.exp(d['lr']); width = torch.exp(d['lw'])
        a = torch.cos(d['ang']) * d['e1'] + torch.sin(d['ang']) * d['e2']
        t = ((self.v - p) @ a).clamp(-half, half)
        dist2 = ((self.v - p - t[:, None] * a[None]) ** 2).sum(1)
        w = torch.exp(-dist2 / width ** 2) * ((self.n @ d['nc']) > .5) * self.airfoil
        return -(depth * w)[:, None] * self.n


def block(pb, kind, vid, **kw):
    nc, e1, e2 = frame(pb, vid)
    b = dict(type=kind, vid=int(vid), nc=nc, e1=e1, e2=e2, uv=torch.zeros(2, device=DEV, requires_grad=True),
             sdepth=param(.05))
    if kind == 'nick':
        b.update(lr=param(np.log(kw['r'])))
    else:
        b.update(lr=param(np.log(kw['half'])), lw=param(np.log(kw['width'])), ang=param(kw['ang']))
    return b


def fit(pb, cfg, start, blk, col):
    for k, v in start.items():
        pb.nuis[k].data.copy_(v)
    c = None if col is None else {k: (v.detach().clone().requires_grad_(v.requires_grad) if torch.is_tensor(v) else v) for k, v in col.items()}
    l, _ = pb.optimise(blk, c, cfg.iters_h1, cfg.h_nuisance)
    return l, blk


def centre(pb, b):
    with torch.no_grad():
        return (pb.v[b['vid']] + b['uv'][0] * b['e1'] + b['uv'][1] * b['e2']).cpu().numpy()


def run_case(c, r, shader, cfg, edges, edge_len):
    sid, cid, split = c['source'], c['case_id'], c['case_id'].rsplit('_', 2)[0]
    imgs = [rgb(B / split / 'renders' / cid / f'{x}.png') for x in cfg.views]
    refs = [rgb(B / split / 'refs' / f'ref_{sid:04d}' / f'{x}.png') for x in cfg.views]
    torch.manual_seed(0)
    pb = Specialists(sid, imgs, shader, cfg, refs)
    pb.fit_nuisance(); pb.set_interior_mask(cfg.interior_margin_px)
    l_none, _ = pb.optimise(None, None, cfg.iters_h0 // 2, cfg.h_nuisance)
    nuis0 = {k: v.detach().clone() for k, v in pb.nuis.items()}
    # stain specialist: the frozen H0 colour search
    l_stain, col0, nuis_col = np.inf, None, nuis0
    for vid in [x for x, _ in pb.score_proposals(None, cfg.starts, kind='stain')[0]]:
        for k, v in nuis0.items():
            pb.nuis[k].data.copy_(v)
        col = pb.new_block(vid, 'colour', .08)
        l, _ = pb.optimise(None, col, cfg.iters_h0, cfg.h_nuisance)
        if l < l_stain:
            l_stain, col0, nuis_col = l, col, {k: v.detach().clone() for k, v in pb.nuis.items()}
    out = dict(case_id=cid, defect=c['defect'], l_none=l_none, l_stain=l_stain, stain_gain=l_none - min(l_stain, l_none),
               reported=bool(r['fit_gain'] > T and r['pred_changed_vertices'] > 0), frozen_gain=r['fit_gain'])
    if not out['reported']:
        out['label'] = 'stain' if out['stain_gain'] > T else 'nothing'
        return out
    v = pb.ctx['v']; vid = int(np.argmin(np.linalg.norm(v - np.asarray(r['centre']), axis=1)))
    ed = np.flatnonzero(edges); evid = int(ed[np.argmin(np.linalg.norm(v[ed] - v[vid], axis=1))])
    starts = [(nuis0, None)] + ([(nuis_col, col0)] if col0 is not None else [])
    cands = {'dent': [lambda: pb.new_block(vid, 'dent', float(r['radius']) or .06)],
             'nick': [lambda rr=rr, a=a: block(pb, 'nick', a, r=rr) for rr in (.03, .06) for a in {vid, evid}],
             'crack': [lambda ang=ang: block(pb, 'crack', vid, half=.1, width=1.2 * edge_len, ang=ang)
                       for ang in np.radians([0, 45, 90, 135])]}
    res = {}
    for kind, makers in cands.items():
        best = (np.inf, None)
        for mk in makers:
            for st, col in starts:
                l, b = fit(pb, cfg, st, mk(), col)
                if l < best[0]:
                    best = (l, b)
        res[kind] = dict(loss=best[0], centre=centre(pb, best[1]).tolist(),
                         depth=float(best[1]['sdepth'].clamp_min(0)) * 1e-3)
    order = sorted(res, key=lambda k: res[k]['loss'])
    margin = res[order[1]]['loss'] - res[order[0]]['loss']
    tv = v[c['dent']['center']] if c['dent'] else None
    out.update(specialists=res, best=order[0], margin=margin, label=order[0] if margin >= T else 'uncertain',
               centre_err_best=None if tv is None else float(np.linalg.norm(np.asarray(res[order[0]]['centre']) - tv)),
               centre_err_frozen=r.get('centre_err'))
    return out


def main():
    split = sys.argv[1]
    out = ROOT / f'results/phase5/specialists_{split}.jsonl'; out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['case_id'] for l in out.read_text().splitlines()} if out.exists() else set()
    cfg = Config(); shader = load_shader(); cache = {}
    for c in json.loads((B / split / 'cases.json').read_text()):
        if c['case_id'] in done:
            continue
        sid = c['source']
        if sid not in cache:
            from bladedelta.geometry import surface_context
            ctx = surface_context(sid)
            e = ctx['v'][ctx['airfoil'][:, 0]] - ctx['v'][ctx['airfoil'][:, 1]]
            cache[sid] = (edge_vertices(ctx) & (ctx['boundary_distance'] >= .1), float(np.median(np.linalg.norm(e, axis=1))))
        r = json.loads((ROOT / f'results/{split}/v11/{c["case_id"]}.json').read_text())
        row = run_case(c, r, shader, cfg, *cache[sid])
        with out.open('a') as f:
            f.write(json.dumps(row, default=float) + '\n')
        print(c['case_id'], c['defect'], '->', row['label'], ('margin %.2f' % row['margin']) if 'margin' in row else '', flush=True)


if __name__ == '__main__':
    main()
