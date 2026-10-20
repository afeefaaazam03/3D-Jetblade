"""Phase 5, round 2 (new code): improved nick and crack specialists; same judge, detection unchanged (frozen v1.1).

  nick2  - carved notch: surface inside a soft-edged ellipsoid (radius R, aspect, free in-plane orientation) centred
           depth-(R) outside the surface is pushed radially onto the ellipsoid surface. The generator uses a hard-edged
           sphere along the normal, so the shapes differ (soft boundary, anisotropy, orientation).
  crack2 - thin dark line along a fitted curved path (centre, angle, half-length, curvature, width, darkness) plus a
           tiny groove along the same path. The generator uses a random-walk path with fixed width and darkness.
Dent and stain specialists are the round-1 ones.

    python phase5/specialists2.py cad6dmg2
Writes results/phase5/specialists2_<split>.jsonl (resumable).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'phase5')); sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'phase4'))
import specialists as S1  # noqa: E402
from bladedelta.estimator import DEV  # noqa: E402

T = S1.T
NPTS = 24


class Specialists2(S1.Specialists):
    def path(self, d):
        p = self.v[d['vid']] + d['uv'][0] * d['e1'] + d['uv'][1] * d['e2']
        a = torch.cos(d['ang']) * d['e1'] + torch.sin(d['ang']) * d['e2']
        b = -torch.sin(d['ang']) * d['e1'] + torch.cos(d['ang']) * d['e2']
        t = torch.linspace(-1, 1, NPTS, device=DEV) * torch.exp(d['lr'])
        return p + t[:, None] * a + (d['curv'] * t ** 2)[:, None] * b

    def line_weight(self, d):
        pts = self.path(d)
        dist = torch.cdist(self.v, pts).min(1).values
        return torch.exp(-(dist / torch.exp(d['lw'])) ** 2) * ((self.n @ d['nc']) > .5) * self.airfoil

    def dent(self, d):
        if d is None or d.get('type') not in ('nick2', 'crack2'):
            return super().dent(d)
        depth = d['sdepth'].clamp_min(0) * 1e-3
        if d['type'] == 'crack2':
            return -(depth * self.line_weight(d))[:, None] * self.n
        R = torch.exp(d['lr']); asp = torch.sigmoid(d['la']) * .8 + .2
        f1 = torch.cos(d['ang']) * d['e1'] + torch.sin(d['ang']) * d['e2']
        f2 = -torch.sin(d['ang']) * d['e1'] + torch.cos(d['ang']) * d['e2']
        P = self.v[d['vid']] + d['uv'][0] * d['e1'] + d['uv'][1] * d['e2'] + d['nc'] * (R - depth)
        q = self.v - P
        qs = torch.stack([q @ f1, q @ f2 / asp, q @ d['nc']], 1)           # ellipsoid -> unit-sphere coordinates
        rho = qs.norm(dim=1).clamp_min(1e-9)
        soft = torch.sigmoid((R - rho) / (.12 * R)) * self.airfoil         # soft inside indicator
        push = (R / rho - 1).clamp_min(0)                                  # scale onto the surface
        return (soft * push)[:, None] * q

    def predict(self, dent=None, colour=None):
        if dent is None or dent.get('type') != 'crack2':
            return super().predict(dent, colour)
        c0 = self.colour
        dark = torch.sigmoid(dent['ls']) * self.line_weight(dent)
        self.colour = lambda c: c0(c) * (1 - dark)[:, None]
        try:
            return super().predict(dent, colour)
        finally:
            self.colour = c0


def param(x):
    return torch.tensor(float(x), device=DEV, requires_grad=True)


def block2(pb, kind, vid, **kw):
    nc, e1, e2 = S1.frame(pb, vid)
    b = dict(type=kind, vid=int(vid), nc=nc, e1=e1, e2=e2, uv=torch.zeros(2, device=DEV, requires_grad=True), ang=param(kw['ang']))
    if kind == 'nick2':
        b.update(lr=param(np.log(kw['R'])), la=param(2.), sdepth=param(kw['R'] * .5 * 1e3))
    else:
        b.update(lr=param(np.log(.08)), lw=param(np.log(kw['width'])), curv=param(0.), ls=param(-.5), sdepth=param(.02))
    return b


def run_case(c, r, shader, cfg, edges, edge_len):
    S1.Specialists = Specialists2
    orig = S1.block
    S1.block = lambda pb, kind, vid, **kw: (block2(pb, 'nick2', vid, R=kw['r'], ang=0.) if kind == 'nick'
                                            else block2(pb, 'crack2', vid, width=kw['width'], ang=kw['ang']))
    try:
        return S1.run_case(c, r, shader, cfg, edges, edge_len)
    finally:
        S1.block = orig


def main():
    split = sys.argv[1]
    out = ROOT / f'results/phase5/specialists2_{split}.jsonl'
    done = {json.loads(l)['case_id'] for l in out.read_text().splitlines()} if out.exists() else set()
    from bladedelta.estimator import Config
    from bladedelta.geometry import surface_context
    from bladedelta.renderer import load_shader
    cfg = Config(); shader = load_shader(); cache = {}
    for c in json.loads((S1.B / split / 'cases.json').read_text()):
        if c['case_id'] in done:
            continue
        sid = c['source']
        if sid not in cache:
            ctx = surface_context(sid)
            e = ctx['v'][ctx['airfoil'][:, 0]] - ctx['v'][ctx['airfoil'][:, 1]]
            cache[sid] = (S1.edge_vertices(ctx) & (ctx['boundary_distance'] >= .1), float(np.median(np.linalg.norm(e, axis=1))))
        r = json.loads((ROOT / f'results/{split}/v11/{c["case_id"]}.json').read_text())
        row = run_case(c, r, shader, cfg, *cache[sid])
        with out.open('a') as f:
            f.write(json.dumps(row, default=float) + '\n')
        print(c['case_id'], c['defect'], '->', row['label'], ('margin %.2f' % row['margin']) if 'margin' in row else '', flush=True)


if __name__ == '__main__':
    main()
