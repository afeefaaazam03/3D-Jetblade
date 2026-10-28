"""Phase 6 (new code): extra rendered datasets for the professor's tasks.

  multi     - held-out blades 1003-1006: blades with 2 or 3 separate defects (dents / sharp nicks), for counting (task 2).
              10 per blade = 40.
  folded    - held-out blades 1003-1006: irregular "folded" dents (a dent creased by a sharp fold line with a raised ridge),
              for the judge test (task 6). 6 per blade = 24.
  yolotrain - development blades 1001-1002: dent / sharp nick / hairline crack / stain / healthy, all 7 views,
              to train the YOLOv8 baseline (task 5). 70 per blade = 140.
Defect models are those of phase 4/4b (dent_field, sharp nick, hairline crack, colour stain).

    python phase6/build_extra.py cases <split> | render <split> | boxes <split>
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'scripts', ROOT / 'phase4'):
    sys.path.insert(0, str(p))
import build_benchmark as bb  # noqa: E402
from bladedelta.cameras import FIT_VIEWS, camera  # noqa: E402
from bladedelta.cases import all_views, visible_vertices  # noqa: E402
from bladedelta.damage import FAMILIES, colour_field, dent_field  # noqa: E402
from bladedelta.data import mm_per_unit  # noqa: E402
from bladedelta.geometry import load_healthy, surface_context  # noqa: E402
from build_dmg4 import edge_vertices  # noqa: E402
from build_dmg5 import crack_fields, crack_path, nick_disp  # noqa: E402

B = ROOT / 'data/benchmark'
BLADES = {'multi': range(1003, 1007), 'folded': range(1003, 1007), 'yolotrain': (1001, 1002)}


def pools(sid):
    ctx = surface_context(sid)
    both = visible_vertices(sid, FIT_VIEWS[:1]) & visible_vertices(sid, FIT_VIEWS[1:])
    e = ctx['v'][ctx['airfoil'][:, 0]] - ctx['v'][ctx['airfoil'][:, 1]]
    return ctx, np.flatnonzero(both & (ctx['boundary_distance'] >= .15)), np.flatnonzero(edge_vertices(ctx) & both & (ctx['boundary_distance'] >= .15)), float(np.median(np.linalg.norm(e, axis=1)))


def spec(rng, kind, sid, mm, interior, edges, edge_len, avoid=()):
    v = surface_context(sid)['v']
    for _ in range(500):
        c = int(rng.choice(edges if kind == 'nick' else interior))
        if all(np.linalg.norm(v[c] - v[a]) > .25 for a in avoid):
            break
    if kind in ('dent', 'folded'):
        return dict(type=kind, center=c, depth=float(np.exp(rng.uniform(np.log(.25), np.log(1.))) / mm), radius=float(rng.uniform(2, 5) / mm),
                    family=str(rng.choice(FAMILIES)), aspect=float(rng.uniform(.5, 1)), angle=float(rng.uniform(0, np.pi)), seed=int(rng.integers(1 << 30)))
    if kind == 'nick':
        r = float(rng.uniform(1, 3) / mm); return dict(type='nick', center=c, radius=r, depth=float(min(rng.uniform(.5, 2.) / mm, .9 * r)))
    if kind == 'crack':
        return dict(type='crack', center=c, length=float(rng.uniform(5, 15) / mm), width=float(1.5 * edge_len), groove=float(rng.uniform(.02, .05) / mm), seed=int(rng.integers(1 << 30)))
    return dict(type='stain', center=c, radius=float(rng.uniform(.04, .12)), strength=float(rng.uniform(.15, .6)), tint=[1., float(rng.uniform(.6, 1.)), float(rng.uniform(.4, 1.))])


def fields(sid, defects):
    ctx = surface_context(sid); n = len(ctx['v']); disp = np.zeros((n, 3)); alb = np.ones((n, 3)); masks = []
    for d in defects:
        before_d, before_a = disp.copy(), alb.copy()
        if d['type'] == 'dent':
            disp += dent_field(sid, d['center'], d['depth'], d['radius'], d['family'], d['aspect'], d['angle'])
        elif d['type'] == 'folded':
            disp += dent_field(sid, d['center'], d['depth'], d['radius'], 'elliptic', d['aspect'], d['angle'])
            rng = np.random.default_rng(d['seed']); v, nrm = ctx['v'], ctx['normals']; nc = nrm[d['center']]
            a = rng.normal(size=3); a -= a.dot(nc) * nc; a /= np.linalg.norm(a); b = np.cross(nc, a)
            q = v - v[d['center']]; inside = np.linalg.norm(q, axis=1) < d['radius'] * 1.1
            s = q @ b; w = d['radius'] * .12
            crease = -.7 * d['depth'] * np.exp(-(s / w) ** 2) + .35 * d['depth'] * np.exp(-((s - 2.2 * w) / w) ** 2)
            disp += (np.where(inside, crease, 0.) * (nrm @ nc > .3))[:, None] * nrm
        elif d['type'] == 'nick':
            disp += nick_disp(ctx, d['center'], d['depth'], d['radius'])
        elif d['type'] == 'crack':
            pts = crack_path(ctx, d['center'], d['length'], np.random.default_rng(d['seed'])); dd, aa = crack_fields(ctx, pts, d['width'], d['groove'])
            disp += dd; alb *= aa
        elif d['type'] == 'stain':
            alb *= colour_field(sid, d['center'], d['radius'], d['strength'], d['tint'])
        masks.append((np.linalg.norm(disp - before_d, axis=1) * mm_per_unit(sid) > .02) | (np.abs(alb - before_a).max(1) > .05))
    return disp, alb, masks


def build_cases(split):
    out = B / split
    if (out / 'cases.json').exists():
        raise SystemExit(split + ' already frozen')
    rng = np.random.default_rng({'multi': 6101, 'folded': 6202, 'yolotrain': 6303}[split]); cases = []
    for sid in BLADES[split]:
        mm = mm_per_unit(sid); ctx, interior, edges, el = pools(sid); k = 0
        if split == 'multi':
            plan = [['dent', 'dent']] * 3 + [['dent', 'nick']] * 2 + [['dent', 'dent', 'dent']] * 3 + [['dent', 'nick', 'dent']] * 2
        elif split == 'folded':
            plan = [['folded']] * 6
        else:
            plan = [['dent']] * 15 + [['nick']] * 15 + [['crack']] * 15 + [['stain']] * 15 + [[]] * 10
        for kinds in plan:
            cap = bb.capture(rng, bb.CAPTURES[k % 4]); ds = []
            for kd in kinds:
                ds.append(spec(rng, kd, sid, mm, interior, edges, el, avoid=[d['center'] for d in ds]))
            first = next((d for d in ds if d['type'] != 'stain'), None)
            c = dict(case_id=f'{split}_{sid:04d}_{k:02d}', source=sid, defects=ds, capture=cap,
                     kind='dent' if first else ('colour' if ds else 'healthy'), defect=ds[0]['type'] if ds else 'healthy',
                     dent=None if not first else dict(center=first['center'], depth=first.get('depth', first.get('groove', 0.)), family=first['type']), colour=None)
            cases.append(c); k += 1
    out.mkdir(parents=True, exist_ok=True); (out / 'cases.json').write_text(json.dumps(cases, indent=0)); print(split, len(cases), 'cases')


def render(split):
    out = B / split; cases = json.loads((out / 'cases.json').read_text()); items = []
    for c in cases:
        root = out / 'cases' / c['case_id']; root.mkdir(parents=True, exist_ok=True)
        h = load_healthy(c['source']); disp, alb, masks = fields(c['source'], c['defects'])
        np.savez(root / 'mesh.npz', vertices=(h['vertices'] + disp).astype(np.float32), triangles=h['triangles'], triangle_component=h['triangle_component'])
        np.save(root / 'displacement.npy', disp.astype(np.float32)); np.save(root / 'masks.npy', np.array(masks, bool) if masks else np.zeros((0, len(disp)), bool))
        albedo = None
        if (alb != 1).any():
            albedo = root / 'albedo.npy'; np.save(albedo, alb.astype(np.float32))
        views = c['capture']['views'] if split == 'yolotrain' else {v: c['capture']['views'][v] for v in FIT_VIEWS}
        items.append(dict(id=c['case_id'], mesh=str(root / 'mesh.npz'), albedo=None if albedo is None else str(albedo), light_gains=c['capture'].get('light_gains', [1, 1, 1]),
                          views=[dict(view_id=v, c2w=camera(v, **p)['c2w_blender'].tolist()) for v, p in views.items()], out=str(out / 'renders' / c['case_id'])))
    if not (out / 'refs').exists():
        shutil.copytree(B / 'cad6/refs', out / 'refs')
    job = out / 'job.json'; job.write_text(json.dumps(dict(device='GPU', samples=32, items=items)))
    subprocess.run([str(bb.BLENDER), '--background', '--factory-startup', '--python-exit-code', '1', '--python', str(ROOT / 'bladedelta/render_blender.py'), '--', str(job)], check=True)


def boxes(split):
    """2D boxes per view: project each defect's changed vertices that face the camera."""
    out = B / split; cases = json.loads((out / 'cases.json').read_text()); allb = {}
    for c in cases:
        ctx = surface_context(c['source']); h = load_healthy(c['source']); masks = np.load(out / 'cases' / c['case_id'] / 'masks.npy')
        disp = np.load(out / 'cases' / c['case_id'] / 'displacement.npy'); V = h['vertices'] + disp
        for vid, p in c['capture']['views'].items():
            cam = camera(vid, **p); X = np.c_[V, np.ones(len(V))] @ cam['w2c_cv'].T; uv = (X[:, :3] @ cam['K'].T); uv = uv[:, :2] / uv[:, 2:3]
            centre = cam['c2w_blender'][:3, 3]; facing = np.einsum('ij,ij->i', ctx['normals'], centre[None] - V) > 0
            bl = []
            for d, m in zip(c['defects'], masks):
                sel = m & facing
                if sel.sum() < 2:
                    continue
                x0, y0 = uv[sel].min(0); x1, y1 = uv[sel].max(0); pad = 3
                bl.append(dict(type=d['type'], box=[float(max(0, x0 - pad)), float(max(0, y0 - pad)), float(min(511, x1 + pad)), float(min(511, y1 + pad))]))
            allb[f"{c['case_id']}/{vid}"] = bl
    (out / 'boxes.json').write_text(json.dumps(allb)); print(split, len(allb), 'views with box lists')


if __name__ == '__main__':
    {'cases': build_cases, 'render': render, 'boxes': boxes}[sys.argv[1]](sys.argv[2])
