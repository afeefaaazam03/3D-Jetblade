"""Phase 2: minimum-detectable-depth map over the blade surface (new code; frozen method only imported).

For a capture, fit H0 as the frozen estimator does, then for every candidate centre c on the visible airfoil place a
hypothetical dent of fixed radius (benchmark Gaussian profile) and compute its profiled information s(c) after the
gains, brightness field and a stain patch at c are allowed to explain it (see detectability.py). The minimum detectable
depth is h*(c) = sqrt(2 tau / s(c)). Uses the same linearisation as detectability.py, batched over centres.

    python phase2/detect_map.py <split> <case_id> [--radius 0.08] [--step 0.03]
Writes results/phase2/maps/<case_id>_r<radius>.npz (centres, h_min_mm, s, nuisance_share).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'phase2'))
from bladedelta.damage import dent_field  # noqa: E402
from bladedelta.estimator import Config  # noqa: E402
from bladedelta.renderer import load_shader  # noqa: E402
from bladedelta.geometry import surface_context  # noqa: E402
from detectability import h0_state, render  # noqa: E402
from run_eval import rgb  # noqa: E402

B = ROOT / 'data/benchmark'


@torch.no_grad()
def nuisance_block(pb, cfg, sel):
    cols, lam = [], []
    for name, sig in (('lgain', cfg.gain_sigma), ('field', cfg.field_sigma)):
        flat = pb.nuis[name].view(-1)
        for k in range(flat.numel()):
            d = 1e-3
            flat[k] += d; ip = sel(render(pb)); flat[k] -= 2 * d; im = sel(render(pb)); flat[k] += d
            cols.append((ip - im) / (2 * d)); lam.append(2. / sig ** 2)
    return cols, lam


def map_case(split, cid, radius=.08, step=.03, cfg=Config(), shader=None):
    shader = shader or load_shader()
    c = next(x for x in json.loads((B / split / 'cases.json').read_text()) if x['case_id'] == cid)
    sid = c['source']
    imgs = [rgb(B / split / 'renders' / cid / f'{v}.png') for v in cfg.views]
    refs = [rgb(B / split / 'refs' / f'ref_{sid:04d}' / f'{v}.png') for v in cfg.views]
    pb = h0_state(sid, imgs, refs, shader, cfg)
    return _map(pb, sid, cid, radius, step, cfg)


@torch.no_grad()
def _map(pb, sid, cid, radius, step, cfg):
    tau = json.loads((ROOT / 'METHOD_FREEZE.json').read_text())['thresholds']['ours']['tau_alpha05']
    m = pb.mask > 0
    sel = lambda img: img[m].reshape(-1)
    r0 = sel(render(pb) - pb.obs); e = cfg.charbonnier
    w = e * e / (r0 * r0 + e * e) ** 1.5
    base_cols, base_lam = nuisance_block(pb, cfg, sel)
    ctx = surface_context(sid)
    air = np.zeros(len(ctx['v']), bool); air[ctx['used']] = True
    pool = np.flatnonzero(air & (ctx['boundary_distance'] >= max(.1, 1.2 * radius)))
    P = ctx['v'][pool]; chosen = [0]; dist = np.linalg.norm(P - P[0], axis=1)
    while dist.max() > step:                          # farthest-point grid with spacing ~step
        chosen.append(int(np.argmax(dist))); dist = np.minimum(dist, np.linalg.norm(P - P[chosen[-1]], axis=1))
    centres = pool[chosen]
    out_s, out_raw = [], []
    for vid in centres:
        u = dent_field(sid, int(vid), 1.0, radius, 'gaussian', 1., 0.)
        u = u / max(np.linalg.norm(u, axis=1).max(), 1e-12)
        ut = torch.tensor(u, dtype=torch.float32, device=pb.v.device); eps = 1e-4
        jd = (sel(render(pb, eps * ut)) - sel(render(pb, -eps * ut))) / (2 * eps)
        if float(jd.abs().max()) == 0.:
            out_s.append(0.); out_raw.append(0.); continue
        cols, lam = list(base_cols), list(base_lam)
        blk = pb.new_block(int(vid), 'colour', radius)
        for k in range(3):
            d = 1e-3
            blk['lmul'][k] = d; ip = sel(render(pb, col=blk)); blk['lmul'][k] = -d; im = sel(render(pb, col=blk)); blk['lmul'][k] = 0.
            cols.append((ip - im) / (2 * d)); lam.append(0.)
        Jn = torch.stack(cols, 1); WJn = Jn * w[:, None]
        A = Jn.T @ WJn + torch.diag(torch.tensor(lam, device=Jn.device)) + 1e-9 * torch.eye(Jn.shape[1], device=Jn.device)
        b = WJn.T @ jd
        s_raw = float(jd @ (w * jd)); s = max(s_raw - float(b @ torch.linalg.solve(A, b)), 0.)
        out_s.append(s); out_raw.append(s_raw)
    s = np.array(out_s); raw = np.array(out_raw)
    hmin = np.where(s > 0, np.sqrt(2 * tau / np.maximum(s, 1e-30)) * 74, np.inf)
    o = ROOT / 'results/phase2/maps'; o.mkdir(parents=True, exist_ok=True)
    np.savez(o / f'{cid}_r{radius:.2f}.npz', centres=centres, h_min_mm=hmin, s=s, s_raw=raw,
             nuisance_share=np.where(raw > 0, 1 - s / np.maximum(raw, 1e-30), np.nan), source=sid, radius=radius, tau=tau)
    return centres, hmin


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('split'); ap.add_argument('case_id')
    ap.add_argument('--radius', type=float, default=.08); ap.add_argument('--step', type=float, default=.03)
    a = ap.parse_args()
    cen, h = map_case(a.split, a.case_id, a.radius, a.step)
    fin = np.isfinite(h)
    print(a.case_id, 'centres', len(cen), 'detectable', int(fin.sum()), 'h_min mm median', round(float(np.median(h[fin])), 3) if fin.any() else None,
          'p10', round(float(np.percentile(h[fin], 10)), 3) if fin.any() else None)
