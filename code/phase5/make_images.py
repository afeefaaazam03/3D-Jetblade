"""Example pictures for the results page: for one case of each defect type, the photo and the change vs the healthy
reference render (amplified), cropped to the blade. Writes site/img/latest/*.png (new code)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'data/benchmark/cad6dmg'; OUT = ROOT / 'site/img/latest'; OUT.mkdir(parents=True, exist_ok=True)
C = json.loads((B / 'cases.json').read_text())
R = {c['case_id']: json.loads((ROOT / f"results/cad6dmg/v11/{c['case_id']}.json").read_text()) for c in C}
picked = {}
for c in C:
    d = c['defect']
    if d in picked or c['capture']['condition'] != 'nominal':
        continue
    picked[d] = c
for d, c in picked.items():
    best = None
    for v in ('v004', 'v005'):
        a = np.asarray(Image.open(B / 'renders' / c['case_id'] / f'{v}.png').convert('RGB')).astype(float)
        b = np.asarray(Image.open(B / 'refs' / f"ref_{c['source']:04d}" / f'{v}.png').convert('RGB')).astype(float)
        diff = np.abs(a - b).mean(-1)
        if best is None or diff.max() > best[2].max():
            best = (a, b, diff)
    a, b, diff = best
    mask = b.mean(-1) > 8
    ys, xs = np.where(mask); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    s = max(y1 - y0, x1 - x0) // 2 + 10; cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    box = (max(cx - s, 0), max(cy - s, 0), min(cx + s, a.shape[1]), min(cy + s, a.shape[0]))
    heat = colormaps['inferno'](np.clip(diff / max(np.percentile(diff[mask], 99.9), 1), 0, 1))[..., :3] * 255
    Image.fromarray(a.astype(np.uint8)).crop(box).resize((300, 300)).save(OUT / f'{d}_photo.png')
    Image.fromarray(heat.astype(np.uint8)).crop(box).resize((300, 300)).save(OUT / f'{d}_diff.png')
    print(d, c['case_id'])
(OUT / 'examples.json').write_text(json.dumps({d: c['case_id'] for d, c in picked.items()}))
