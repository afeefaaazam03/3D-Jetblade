"""Phase 4b (new code): more realistic nicks and cracks on held-out industrial blades 1003-1006.

  nick  - a sharp notch: material inside a sphere (radius 1-3 mm) centred just outside a leading/trailing-edge point is
          removed; surface vertices inside the sphere are pushed radially onto the sphere, leaving a sharp rim and a
          notch of 0.5-2 mm depth that cuts the blade outline.
  crack - a hairline crack as it appears in photos: a thin dark line (one mesh edge wide) along a slightly wavy path of
          5-15 mm, plus an almost-zero groove (0.02-0.05 mm).
8 of each per blade = 64 cases, centred where both input cameras see them.

    python phase4/build_dmg5.py cases | render
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'phase4'))
import build_benchmark as bb  # noqa: E402
from bladedelta.cameras import FIT_VIEWS, camera  # noqa: E402
from bladedelta.cases import visible_vertices  # noqa: E402
from bladedelta.data import mm_per_unit  # noqa: E402
from bladedelta.geometry import load_healthy, surface_context  # noqa: E402
from build_dmg4 import edge_vertices  # noqa: E402

OUT = ROOT / 'data/benchmark/cad6dmg2'


def nick_disp(ctx, c, depth, radius):
    v, n = ctx['v'], ctx['normals']
    P = v[c] + n[c] * (radius - depth)                  # sphere reaches `depth` into the material
    d = v - P; dist = np.linalg.norm(d, axis=1)
    inside = (dist < radius) & np.isin(np.arange(len(v)), ctx['used'])
    out = np.zeros_like(v)
    out[inside] = P + d[inside] / np.maximum(dist[inside, None], 1e-9) * radius - v[inside]
    return out


def crack_path(ctx, c, length, rng):
    v, n = ctx['v'], ctx['normals']; nc = n[c]
    a = rng.normal(size=3); a -= a.dot(nc) * nc; a /= np.linalg.norm(a); b = np.cross(nc, a)
    t = np.linspace(-.5, .5, 60) * length
    wig = np.cumsum(rng.normal(scale=length * .012, size=60)); wig -= wig.mean()
    pts = v[c] + t[:, None] * a + wig[:, None] * b
    used = np.asarray(ctx['used']); front = used[n[used] @ nc > .5]
    snap = front[np.argmin(np.linalg.norm(v[front][None] - pts[:, None], axis=2), axis=1)]
    return v[snap]                                        # path snapped onto the blade surface


def crack_fields(ctx, pts, width, groove):
    v, n = ctx['v'], ctx['normals']
    dist = np.min(np.linalg.norm(v[:, None] - pts[None], axis=2), axis=1)
    on = np.isin(np.arange(len(v)), ctx['used']) & (dist < 3 * width)
    w = np.where(on, np.exp(-(dist / width) ** 2), 0.)
    disp = -(groove * w)[:, None] * n
    albedo = 1 - .7 * w[:, None] * np.ones(3)[None]           # dark line (light trapped in the gap)
    return disp, albedo.astype(np.float32)


def build_cases():
    if (OUT / 'cases.json').exists():
        raise SystemExit('cad6dmg2 case list already frozen')
    rng = np.random.default_rng(9113); cases = []
    for sid in range(1003, 1007):
        mm = mm_per_unit(sid); ctx = surface_context(sid)
        both = visible_vertices(sid, FIT_VIEWS[:1]) & visible_vertices(sid, FIT_VIEWS[1:])
        interior = both & (ctx['boundary_distance'] >= .15)
        edges = edge_vertices(ctx) & both & (ctx['boundary_distance'] >= .15)
        e = ctx['v'][ctx['airfoil'][:, 0]] - ctx['v'][ctx['airfoil'][:, 1]]; edge_len = float(np.median(np.linalg.norm(e, axis=1)))
        k = 0
        for defect in ['nick'] * 8 + ['crack'] * 8:
            cap = bb.capture(rng, bb.CAPTURES[k % 4])
            c = dict(case_id=f'cad6dmg2_{sid:04d}_{k:02d}', source=sid, defect=defect, kind='dent', capture=cap, colour=None)
            if defect == 'nick':
                r = float(rng.uniform(1, 3) / mm)
                c['dent'] = dict(center=int(rng.choice(np.flatnonzero(edges))), radius=r,
                                 depth=float(min(rng.uniform(.5, 2.) / mm, .9 * r)), family='nick_sharp')
            else:
                c['dent'] = dict(center=int(rng.choice(np.flatnonzero(interior))), length=float(rng.uniform(5, 15) / mm),
                                 width=float(1.5 * edge_len), groove=float(rng.uniform(.02, .05) / mm),
                                 seed=int(rng.integers(1 << 30)), family='crack_hairline')
                c['dent']['depth'] = c['dent']['groove']
            cases.append(c); k += 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cases.json').write_text(json.dumps(cases, indent=0)); print(len(cases), 'cases')


def fields(c):
    ctx = surface_context(c['source']); d = c['dent']
    if c['defect'] == 'nick':
        return nick_disp(ctx, d['center'], d['depth'], d['radius']), None
    pts = crack_path(ctx, d['center'], d['length'], np.random.default_rng(d['seed']))
    return crack_fields(ctx, pts, d['width'], d['groove'])


def render():
    cases = json.loads((OUT / 'cases.json').read_text()); items = []
    for c in cases:
        root = OUT / 'cases' / c['case_id']; root.mkdir(parents=True, exist_ok=True)
        h = load_healthy(c['source']); disp, alb = fields(c)
        np.savez(root / 'mesh.npz', vertices=(h['vertices'] + disp).astype(np.float32), triangles=h['triangles'],
                 triangle_component=h['triangle_component'])
        np.save(root / 'displacement.npy', disp.astype(np.float32))
        albedo = None
        if alb is not None:
            albedo = root / 'albedo.npy'; np.save(albedo, alb)
        (root / 'case.json').write_text(json.dumps(c, indent=1))
        views = [dict(view_id=v, c2w=camera(v, **p)['c2w_blender'].tolist()) for v, p in c['capture']['views'].items()]
        items.append(dict(id=c['case_id'], mesh=str(root / 'mesh.npz'), albedo=None if albedo is None else str(albedo),
                          light_gains=c['capture'].get('light_gains', [1, 1, 1]), views=views, out=str(OUT / 'renders' / c['case_id'])))
    if not (OUT / 'refs').exists():
        shutil.copytree(ROOT / 'data/benchmark/cad6/refs', OUT / 'refs')
    job = OUT / 'job.json'; job.write_text(json.dumps(dict(device='GPU', samples=32, items=items)))
    subprocess.run([str(bb.BLENDER), '--background', '--factory-startup', '--python-exit-code', '1',
                    '--python', str(ROOT / 'bladedelta/render_blender.py'), '--', str(job)], check=True)


if __name__ == '__main__':
    {'cases': build_cases, 'render': render}[sys.argv[1]]()
