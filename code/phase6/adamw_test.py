"""Phase 6, task 4 (new code): Adam vs AdamW for the method's fits, dev split only (frozen code untouched; the optimiser
class is swapped at run time). Compares with the stored Adam results in results/dev/ours.

    python phase6/adamw_test.py [weight_decay]
Writes results/phase6/adamw_dev.jsonl
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from bladedelta.estimator import Config, estimate  # noqa: E402
from bladedelta.renderer import load_shader  # noqa: E402
from bladedelta.metrics import geometry_metrics  # noqa: E402
from run_eval import rgb  # noqa: E402

WD = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2
_Adam = torch.optim.Adam
torch.optim.Adam = lambda groups, **kw: torch.optim.AdamW(groups, weight_decay=WD, **kw)
B = ROOT / 'data/benchmark/dev'; out = ROOT / 'results/phase6/adamw_dev.jsonl'; out.parent.mkdir(parents=True, exist_ok=True)
done = {json.loads(l)['case_id'] for l in out.read_text().splitlines()} if out.exists() else set()
C = {c['case_id']: c for c in json.loads((B / 'cases.json').read_text())}
old = {f.stem: json.loads(f.read_text()) for f in (ROOT / 'results/dev/ours').glob('dev_*.json')}
dents = [k for k, r in old.items() if C[k]['dent'] and (r['visible_fraction'] or 0) >= .5]
negs = sorted([k for k, r in old.items() if not C[k]['dent']])[:12]
cfg = Config(); shader = load_shader()
for cid in dents + negs:
    if cid in done: continue
    c = C[cid]; sid = c['source']; t0 = time.time()
    imgs = [rgb(B / 'renders' / cid / f'{v}.png') for v in cfg.views]; refs = [rgb(B / 'refs' / f'ref_{sid:04d}' / f'{v}.png') for v in cfg.views]
    res = estimate(sid, imgs, shader, cfg, bias=refs)
    truth = np.load(B / 'cases' / cid / 'displacement.npy').astype(np.float64); m = geometry_metrics(sid, truth, res['displacement'])
    o = old[cid]
    row = dict(case_id=cid, dent=bool(c['dent']), wd=WD, seconds=time.time() - t0,
               adamw=dict(l0=res['loss_h0'], l1=res['loss_h1'], gain=res['fit_gain'], depth_rel_err=m.get('depth_rel_err'), centre_err=m.get('centre_err')),
               adam=dict(l0=o['loss_h0'], l1=o['loss_h1'], gain=o['fit_gain'], depth_rel_err=o.get('depth_rel_err'), centre_err=o.get('centre_err')))
    with out.open('a') as f: f.write(json.dumps(row, default=float) + '\n')
    print(cid, 'gain adam %.2f adamw %.2f | L1 adam %.2f adamw %.2f' % (o['fit_gain'], res['fit_gain'], o['loss_h1'], res['loss_h1']), flush=True)
