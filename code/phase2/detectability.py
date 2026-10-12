"""Phase 2 (new code; the frozen method is only imported, never modified).

Predicted detectability of a dent from the evidence it would add over the nuisance-only explanation.

At the fitted nuisance-only state (H0, exactly as the frozen estimator fits it), linearise the rendered image in
  - the dent depth h along the dent's own shape u (unit depth),
  - the nuisance parameters free during the dent test: 3 light gains, the per-view brightness field, and a stain
    (albedo) patch at the same place and size as the dent.
With the robust loss's local curvature W (Charbonnier, at the H0 residuals) and the nuisance prior Lambda, the evidence a
dent of depth h adds is approximately
    G(h) = 1/2 h^2 s,   s = jd' W jd - jd' W Jn (Jn' W Jn + Lambda)^-1 Jn' W jd
i.e. the dent's information left after the best nuisance explanation (a profiled / efficient Fisher information).
It is reported if G > tau, so the minimum detectable depth at that place is h* = sqrt(2 tau / s).

    python phase2/detectability.py test [--ids file.json]
Writes results/phase2/detectability_<split>.jsonl (one line per dent case; resumable).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from bladedelta.estimator import Config, Problem  # noqa: E402  (frozen, read-only)
from bladedelta.renderer import load_shader  # noqa: E402
from run_eval import rgb  # noqa: E402

OUT = ROOT / 'results/phase2'
B = ROOT / 'data/benchmark'


def h0_state(sid, imgs, refs, shader, cfg):
    """Fit H0 exactly as estimate() does before the dent search."""
    pb = Problem(sid, imgs, shader, cfg, refs)
    pb.fit_nuisance()
    pb.set_interior_mask(cfg.interior_margin_px)
    pb.optimise(None, None, cfg.iters_h0 // 2, cfg.h_nuisance)
    return pb


@torch.no_grad()
def render(pb, dv=None, col=None):
    v0 = pb.v
    if dv is not None:
        pb.v = v0 + dv
    try:
        img, _ = pb.predict(None, col)
    finally:
        pb.v = v0
    return img


@torch.no_grad()
def profiled_information(pb, u, centre_vid, radius, cfg, eps_h=1e-4):
    """s = efficient information of dent depth (unit-depth shape u) after profiling out gains, field and a stain patch."""
    m = pb.mask > 0                                   # (B,H,W) interior evidence domain
    sel = lambda img: img[m].reshape(-1)              # masked pixels x 3 channels
    img0 = render(pb)
    r0 = sel(img0 - pb.obs)
    e = cfg.charbonnier
    w = e * e / (r0 * r0 + e * e) ** 1.5              # Charbonnier curvature at the H0 residual
    ut = torch.tensor(u, dtype=torch.float32, device=pb.v.device)
    jd = (sel(render(pb, eps_h * ut)) - sel(render(pb, -eps_h * ut))) / (2 * eps_h)
    cols, lam = [], []
    for name, sig in (('lgain', cfg.gain_sigma), ('field', cfg.field_sigma)):
        t = pb.nuis[name]; flat = t.view(-1)
        for k in range(flat.numel()):
            d = 1e-3
            flat[k] += d; ip = sel(render(pb)); flat[k] -= 2 * d; im = sel(render(pb)); flat[k] += d
            cols.append((ip - im) / (2 * d)); lam.append(2. / sig ** 2)
    blk = pb.new_block(centre_vid, 'colour', radius)  # stain patch at the dent: the shape/albedo ambiguity
    for k in range(3):
        d = 1e-3
        blk['lmul'][k] = d; ip = sel(render(pb, col=blk)); blk['lmul'][k] = -d; im = sel(render(pb, col=blk)); blk['lmul'][k] = 0.
        cols.append((ip - im) / (2 * d)); lam.append(0.)
    Jn = torch.stack(cols, 1)                          # (P, k)
    WJn = Jn * w[:, None]
    A = Jn.T @ WJn + torch.diag(torch.tensor(lam, device=Jn.device)) + 1e-9 * torch.eye(Jn.shape[1], device=Jn.device)
    b = WJn.T @ jd
    s_raw = float(jd @ (w * jd))
    s = s_raw - float(b @ torch.linalg.solve(A, b))
    return max(s, 0.), s_raw


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('split'); ap.add_argument('--ids'); ap.add_argument('--method', default='ours')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f'detectability_{a.split}.jsonl'
    done = {json.loads(l)['case_id'] for l in out.read_text().splitlines()} if out.exists() else set()
    tau = json.loads((ROOT / 'METHOD_FREEZE.json').read_text())['thresholds']['ours']['tau_alpha05']
    cfg = Config(); shader = load_shader()
    cases = [c for c in json.loads((B / a.split / 'cases.json').read_text()) if c['dent']]
    if a.ids:
        keep = set(json.loads(Path(a.ids).read_text())); cases = [c for c in cases if c['case_id'] in keep]
    for c in cases:
        cid, sid = c['case_id'], c['source']
        if cid in done:
            continue
        t0 = time.time()
        imgs = [rgb(B / a.split / 'renders' / cid / f'{v}.png') for v in cfg.views]
        refs = [rgb(B / a.split / 'refs' / f'ref_{sid:04d}' / f'{v}.png') for v in cfg.views]
        truth = np.load(B / a.split / 'cases' / cid / 'displacement.npy').astype(np.float64)
        h_true = float(np.linalg.norm(truth, axis=1).max())
        u = truth / h_true
        pb = h0_state(sid, imgs, refs, shader, cfg)
        s, s_raw = profiled_information(pb, u, c['dent']['center'], c['dent']['radius'], cfg)
        res = json.loads((ROOT / 'results' / a.split / a.method / f'{cid}.json').read_text())
        row = dict(case_id=cid, source=sid, kind=c['kind'], condition=c['capture']['condition'], h_true=h_true,
                   depth_mm=h_true * 74, visible_fraction=res.get('visible_fraction'), s=s, s_raw=s_raw,
                   nuisance_share=1 - s / s_raw if s_raw > 0 else None,
                   pred_gain=.5 * h_true ** 2 * s, h_min_mm=(np.sqrt(2 * tau / s) * 74) if s > 0 else None,
                   actual_gain=res['fit_gain'], detected=bool(res['fit_gain'] > tau and res['pred_changed_vertices'] > 0),
                   located=bool(res['fit_gain'] > tau and res.get('centre_err') is not None and res['centre_err'] < .1),
                   tau=tau, seconds=time.time() - t0)
        with out.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(cid, f"depth {row['depth_mm']:.3f}mm pred_gain {row['pred_gain']:.2f} actual {row['actual_gain']:.2f} "
              f"h_min {row['h_min_mm'] if row['h_min_mm'] is None else round(row['h_min_mm'], 3)}mm nuis {row['nuisance_share']}", flush=True)


if __name__ == '__main__':
    main()
