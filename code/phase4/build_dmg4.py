"""Phase 4 (new code): basic detection test on four defect types on the held-out industrial blades 1003-1006.

Defects (truth in millimetres, converted per blade):
  dent   - existing smooth dent families (bladedelta.damage.dent_field), depth 0.2-1.0 mm, radius 1.5-5 mm
  nick   - chip on the leading/trailing edge: material removed into the edge, depth 0.5-2.0 mm, radius 1-3 mm
  crack  - narrow groove along a straight segment, length 5-15 mm, width ~1 mesh edge, depth 0.15-0.4 mm
  stain  - colour-only patch (not damage; must NOT be reported)
plus healthy blades. Every case is centred where both input cameras see it. Per blade: 6 dent, 8 nick, 8 crack,
6 stain, 6 healthy = 34; 136 cases. Rendered with the same Blender setup; the method sees only CAD + photos.

    python phase4/build_dmg4.py cases | render
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import build_benchmark as bb  # noqa: E402
from bladedelta.cameras import FIT_VIEWS, camera  # noqa: E402
from bladedelta.cases import all_views, visible_vertices  # noqa: E402
from bladedelta.damage import FAMILIES, colour_field, dent_field  # noqa: E402
from bladedelta.data import mm_per_unit  # noqa: E402
from bladedelta.geometry import load_healthy, surface_context  # noqa: E402

OUT = ROOT / 'data/benchmark/cad6dmg'
MIX = [('dent', 6), ('nick', 8), ('crack', 8), ('stain', 6), ('healthy', 6)]


def edge_vertices(ctx):
    """Leading/trailing-edge vertices: largest normal turning between neighbours."""
    g = ctx['graph'].tocoo(); n = ctx['normals']
    turn = np.zeros(len(n)); np.maximum.at(turn, g.row, 1 - np.einsum('ij,ij->i', n[g.row], n[g.col]))
    ok = np.zeros(len(n), bool); ok[ctx['used']] = True
    return ok & (turn >= np.quantile(turn[ctx['used']], .95))


def nick_field(ctx, c, depth, radius):
    v, n = ctx['v'], ctx['normals']
    q = np.linalg.norm(v - v[c], axis=1) / radius
    w = np.clip(1 - q ** 2, 0, None) ** 1.5
    w[np.setdiff1d(np.arange(len(v)), ctx['used'])] = 0
    return -(depth * w)[:, None] * n[c][None]


def crack_field(ctx, c, depth, length, width, rng):
    v, n = ctx['v'], ctx['normals']; nc = n[c]
    a = rng.normal(size=3); a -= a.dot(nc) * nc; a /= np.linalg.norm(a)
    p0, p1 = v[c] - a * length / 2, v[c] + a * length / 2
    t = np.clip((v - p0) @ a / length, 0, 1)
    dist = np.linalg.norm(v - (p0 + t[:, None] * length * a), axis=1)
    w = np.clip(1 - (dist / width) ** 2, 0, None) * (n @ nc > .5)
    w[np.setdiff1d(np.arange(len(v)), ctx['used'])] = 0
    return -(depth * w)[:, None] * n
 

def build_cases():
    if (OUT / 'cases.json').exists():
        raise SystemExit('cad6dmg case list already frozen')
    rng = np.random.default_rng(7041); cases = []
    for sid in range(1003, 1007):
        mm = mm_per_unit(sid); ctx = surface_context(sid)
        both = visible_vertices(sid, FIT_VIEWS[:1]) & visible_vertices(sid, FIT_VIEWS[1:])
        interior = both & (ctx['boundary_distance'] >= .15)
        edges = edge_vertices(ctx) & both & (ctx['boundary_distance'] >= .15)
        e = ctx['v'][ctx['airfoil'][:, 0]] - ctx['v'][ctx['airfoil'][:, 1]]; edge_len = float(np.median(np.linalg.norm(e, axis=1)))
        k = 0
        for defect, count in MIX:
            for i in range(count):
                cap = bb.capture(rng, bb.CAPTURES[k % 4])
                c = dict(case_id=f'cad6dmg_{sid:04d}_{k:02d}', source=sid, defect=defect, capture=cap,
                         kind={'dent': 'dent', 'nick': 'dent', 'crack': 'dent', 'stain': 'colour', 'healthy': 'healthy'}[defect],
                         dent=None, colour=None)
                if defect == 'dent':
                    ctr = int(rng.choice(np.flatnonzero(interior)))
                    c['dent'] = dict(center=ctr, depth=float(np.exp(rng.uniform(np.log(.2), np.log(1.))) / mm),
                                     radius=float(rng.uniform(1.5, 5) / mm), family=str(rng.choice(FAMILIES)),
                                     aspect=float(rng.uniform(.4, 1)), angle=float(rng.uniform(0, np.pi)))
                elif defect == 'nick':
                    ctr = int(rng.choice(np.flatnonzero(edges)))
                    c['dent'] = dict(center=ctr, depth=float(rng.uniform(.5, 2.) / mm), radius=float(rng.uniform(1, 3) / mm), family='nick')
                elif defect == 'crack':
                    ctr = int(rng.choice(np.flatnonzero(interior)))
                    c['dent'] = dict(center=ctr, depth=float(rng.uniform(.15, .4) / mm), length=float(rng.uniform(5, 15) / mm),
                                     width=float(max(1.2 * edge_len, .4 / mm)), seed=int(rng.integers(1 << 30)), family='crack')
                elif defect == 'stain':
                    ctr = int(rng.choice(np.flatnonzero(interior)))
                    c['colour'] = dict(center=ctr, radius=float(rng.uniform(.04, .12)), strength=float(rng.uniform(.15, .6)),
                                       tint=[1., float(rng.uniform(.6, 1.)), float(rng.uniform(.4, 1.))])
                cases.append(c); k += 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cases.json').write_text(json.dumps(cases, indent=0))
    print(len(cases), 'cases')


def displacement(c):
    ctx = surface_context(c['source']); d = c['dent']
    if c['defect'] == 'dent':
        return dent_field(c['source'], d['center'], d['depth'], d['radius'], d['family'], d['aspect'], d['angle'])
    if c['defect'] == 'nick':
        return nick_field(ctx, d['center'], d['depth'], d['radius'])
    if c['defect'] == 'crack':
        return crack_field(ctx, d['center'], d['depth'], d['length'], d['width'], np.random.default_rng(d['seed']))
    return np.zeros_like(ctx['v'])


def render():
    cases = json.loads((OUT / 'cases.json').read_text()); items = []
    for c in cases:
        root = OUT / 'cases' / c['case_id']; root.mkdir(parents=True, exist_ok=True)
        h = load_healthy(c['source']); disp = displacement(c)
        np.savez(root / 'mesh.npz', vertices=(h['vertices'] + disp).astype(np.float32), triangles=h['triangles'],
                 triangle_component=h['triangle_component'])
        np.save(root / 'displacement.npy', disp.astype(np.float32))
        albedo = None
        if c['colour']:
            k = c['colour']; albedo = root / 'albedo.npy'
            np.save(albedo, colour_field(c['source'], k['center'], k['radius'], k['strength'], k['tint']).astype(np.float32))
        (root / 'case.json').write_text(json.dumps(c, indent=1))
        views = [dict(view_id=v, c2w=camera(v, **p)['c2w_blender'].tolist()) for v, p in c['capture']['views'].items()]
        items.append(dict(id=c['case_id'], mesh=str(root / 'mesh.npz'), albedo=None if albedo is None else str(albedo),
                          light_gains=c['capture'].get('light_gains', [1, 1, 1]), views=views, out=str(OUT / 'renders' / c['case_id'])))
    refs = OUT / 'refs'
    if not refs.exists():
        shutil.copytree(ROOT / 'data/benchmark/cad6/refs', refs)
    job = OUT / 'job.json'; job.write_text(json.dumps(dict(device='GPU', samples=32, items=items)))
    subprocess.run([str(bb.BLENDER), '--background', '--factory-startup', '--python-exit-code', '1',
                    '--python', str(ROOT / 'bladedelta/render_blender.py'), '--', str(job)], check=True)


if __name__ == '__main__':
    {'cases': build_cases, 'render': render}[sys.argv[1]]()
