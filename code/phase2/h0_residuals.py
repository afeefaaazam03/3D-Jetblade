"""Phase 2: per-case statistics of the nuisance-only fit, for two reviewer baselines (frozen method only imported).

For each case: fit H0 exactly as the frozen estimator does, then on the interior evidence domain record
  - sigma: robust residual scale per view (MAD), pixel count P, and L0;
  - an a-contrario score: the residual (mean over RGB) is smoothed with Gaussians of several widths; each smoothed value is
    a z-score under an i.i.d. N(0, sigma^2) noise model; NFA = (#tests) * P(|Z| >= z_max). Detection if NFA < 1.
Used by phase2/baselines_analysis.py for (1) the textbook chi-square threshold on the fit gain and (2) the a-contrario detector.

    python phase2/h0_residuals.py <split>
Writes results/phase2/h0stats_<split>.jsonl (resumable).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'phase2'))
from bladedelta.estimator import Config  # noqa: E402
from bladedelta.renderer import load_shader  # noqa: E402
from detectability import h0_state, render  # noqa: E402
from run_eval import rgb  # noqa: E402

B = ROOT / 'data/benchmark'
SCALES = (2., 4., 8.)


def gauss(x, s):
    r = int(3 * s); t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-.5 * (t / s) ** 2); k = k / k.sum()
    y = F.conv2d(F.pad(x[:, None], (r, r, 0, 0), mode='replicate'), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(y, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1))[:, 0], float((k[:, None] * k[None]).pow(2).sum().sqrt())


@torch.no_grad()
def stats(pb):
    img = render(pb); m = pb.mask > 0
    r = (img - pb.obs).mean(-1)                              # (B,H,W)
    sig = []
    for b in range(len(r)):
        x = r[b][m[b]]
        sig.append(float(1.4826 * (x - x.median()).abs().median().clamp_min(.5 / 255)))
    sigma = torch.tensor(sig, device=r.device)[:, None, None]
    zmax, ntest = 0., 0
    for s in SCALES:
        rs, knorm = gauss(r * m, s)
        cover, _ = gauss(m.float(), s)
        z = (rs / cover.clamp_min(1e-6)) / (sigma * knorm / cover.clamp_min(1e-6)) * m   # mean residual / its noise std
        zmax = max(zmax, float(z.abs().max())); ntest += int(m.sum())
    nfa = 2 * ntest * float(norm.sf(zmax))
    return dict(sigma=sig, P=int(m.sum()), zmax=zmax, n_tests=2 * ntest, nfa=nfa, log10_nfa=float(np.log10(max(nfa, 1e-300))))


def main():
    split = sys.argv[1]
    out = ROOT / f'results/phase2/h0stats_{split}.jsonl'
    done = {json.loads(l)['case_id'] for l in out.read_text().splitlines()} if out.exists() else set()
    cfg = Config(); shader = load_shader()
    for c in json.loads((B / split / 'cases.json').read_text()):
        cid, sid = c['case_id'], c['source']
        if cid in done or not (B / split / 'renders' / cid / 'v004.png').exists():
            continue
        t0 = time.time()
        imgs = [rgb(B / split / 'renders' / cid / f'{v}.png') for v in cfg.views]
        refs = [rgb(B / split / 'refs' / f'ref_{sid:04d}' / f'{v}.png') for v in cfg.views]
        pb = h0_state(sid, imgs, refs, shader, cfg)
        row = dict(case_id=cid, source=sid, kind=c['kind'], **stats(pb), seconds=time.time() - t0)
        with out.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(cid, c['kind'], 'zmax %.2f log10NFA %.1f sigma %s' % (row['zmax'], row['log10_nfa'], [round(s * 255, 2) for s in row['sigma']]), flush=True)


if __name__ == '__main__':
    main()
