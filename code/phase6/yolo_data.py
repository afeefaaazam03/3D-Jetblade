"""Phase 6, task 5 (new code): YOLO dataset. Train/val = yolotrain (development blades 1001-1002, all 7 views);
test = held-out blades 1003-1006, the two input views our method uses: cad6dmg dents/stains/healthy + cad6dmg2 sharp
nicks/hairline cracks. Boxes come from projecting each defect's changed vertices (exact ground truth).
Classes: 0 dent, 1 nick, 2 crack, 3 stain.

    python phase6/yolo_data.py
Writes data/yolo/{images,labels}/{train,val,test} and data/yolo/blade.yaml
"""
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'phase6'))
from bladedelta.cameras import FIT_VIEWS, camera  # noqa: E402
from bladedelta.data import mm_per_unit  # noqa: E402
from bladedelta.geometry import load_healthy, surface_context  # noqa: E402

CLS = {'dent': 0, 'nick': 1, 'crack': 2, 'stain': 3}
B = ROOT / 'data/benchmark'; Y = ROOT / 'data/yolo'


def project_box(sid, V, mask, view, pose):
    ctx = surface_context(sid); cam = camera(view, **pose)
    X = np.c_[V, np.ones(len(V))] @ cam['w2c_cv'].T; uv = X[:, :3] @ cam['K'].T; uv = uv[:, :2] / uv[:, 2:3]
    facing = np.einsum('ij,ij->i', ctx['normals'], cam['c2w_blender'][:3, 3][None] - V) > 0; sel = mask & facing
    if sel.sum() < 2: return None
    x0, y0 = uv[sel].min(0) - 3; x1, y1 = uv[sel].max(0) + 3
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(511, x1), min(511, y1)
    return [(x0 + x1) / 2 / 512, (y0 + y1) / 2 / 512, (x1 - x0) / 512, (y1 - y0) / 512]


def write(split_dir, name, img, lines):
    (Y / 'images' / split_dir).mkdir(parents=True, exist_ok=True); (Y / 'labels' / split_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(img, Y / 'images' / split_dir / f'{name}.png'); (Y / 'labels' / split_dir / f'{name}.txt').write_text('\n'.join(lines))


def main():
    if Y.exists(): shutil.rmtree(Y)
    random.seed(0); n = {'train': 0, 'val': 0, 'test': 0}
    boxes = json.loads((B / 'yolotrain/boxes.json').read_text())
    for c in json.loads((B / 'yolotrain/cases.json').read_text()):
        sd = 'val' if random.random() < .15 else 'train'
        for vid in c['capture']['views']:
            lines = [f"{CLS[b['type']]} {(b['box'][0] + b['box'][2]) / 1024:.5f} {(b['box'][1] + b['box'][3]) / 1024:.5f} {(b['box'][2] - b['box'][0]) / 512:.5f} {(b['box'][3] - b['box'][1]) / 512:.5f}" for b in boxes[f"{c['case_id']}/{vid}"]]
            write(sd, f"{c['case_id']}_{vid}", B / 'yolotrain/renders' / c['case_id'] / f'{vid}.png', lines); n[sd] += 1
    test_meta = {}
    for split, keep in (('cad6dmg', ('dent', 'stain', 'healthy')), ('cad6dmg2', ('nick', 'crack'))):
        for c in json.loads((B / split / 'cases.json').read_text()):
            if c['defect'] not in keep: continue
            sid = c['source']; root = B / split / 'cases' / c['case_id']; h = load_healthy(sid)
            disp = np.load(root / 'displacement.npy'); V = h['vertices'] + disp
            mask = np.linalg.norm(disp, axis=1) * mm_per_unit(sid) > .02
            if (root / 'albedo.npy').exists(): mask |= np.abs(np.load(root / 'albedo.npy') - 1).max(1) > .05
            for vid in FIT_VIEWS:
                lines = []
                if c['defect'] != 'healthy':
                    bx = project_box(sid, V, mask, vid, c['capture']['views'][vid])
                    if bx: lines.append(f"{CLS[c['defect']]} " + ' '.join(f'{t:.5f}' for t in bx))
                name = f"{c['case_id']}_{vid}"; write('test', name, B / split / 'renders' / c['case_id'] / f'{vid}.png', lines); n['test'] += 1
                test_meta[name] = dict(case_id=c['case_id'], defect=c['defect'])
    (Y / 'test_meta.json').write_text(json.dumps(test_meta))
    (Y / 'blade.yaml').write_text(f"path: {Y}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: dent\n  1: nick\n  2: crack\n  3: stain\n")
    print(n)


if __name__ == '__main__':
    main()
