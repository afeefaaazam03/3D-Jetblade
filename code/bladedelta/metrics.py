"""Geometry metrics on the airfoil surface (normalized blade-height units)."""
from __future__ import annotations

import numpy as np

from .geometry import surface_context

CHANGE_EPS = 2e-5   # a vertex "moved" if displaced more than this (~4% of the shallowest benchmark dent)


def geometry_metrics(sid: int, truth: np.ndarray, pred: np.ndarray) -> dict:
    ctx = surface_context(sid)
    air = ctx['used']
    t, p = truth[air], pred[air]
    tn, pn = np.linalg.norm(t, axis=1), np.linalg.norm(p, axis=1)
    support = tn > 1e-9
    err = np.linalg.norm(p - t, axis=1)
    out = dict(whole_rms=float(np.sqrt((err ** 2).mean())),
               whole_rms_cad=float(np.sqrt((tn ** 2).mean())),
               pred_changed_vertices=int((pn > CHANGE_EPS).sum()),
               pred_max_disp=float(pn.max()), truth_max_disp=float(tn.max()))
    if support.any():
        out['local_rms'] = float(np.sqrt((err[support] ** 2).mean()))
        out['local_rms_cad'] = float(np.sqrt((tn[support] ** 2).mean()))
        out['rel_local'] = out['local_rms'] / out['local_rms_cad']        # 1 = no better than CAD, 0 = exact
        out['rel_whole'] = out['whole_rms'] / out['whole_rms_cad']
        out['depth_rel_err'] = float(abs(pn.max() - tn.max()) / tn.max())
        ti, pi = air[np.argmax(tn)], air[np.argmax(pn)]
        out['centre_err'] = float(np.linalg.norm(ctx['v'][ti] - ctx['v'][pi])) if pn.max() > CHANGE_EPS else None
        # volume of displaced material along the surface normal (signed, vertex-area weighted ~ uniform)
        out['vol_rel_err'] = float(abs(p.sum(0) @ ctx['normals'][ti] - t.sum(0) @ ctx['normals'][ti]) /
                                   max(abs(t.sum(0) @ ctx['normals'][ti]), 1e-12))
    out['false_change'] = (not support.any()) and out['pred_changed_vertices'] > 0
    out['reported_change'] = out['pred_changed_vertices'] > 0
    return out


def conformal_threshold(neg_scores, alpha=.05):
    """Smallest tau with P(score > tau | negative) <= alpha (split-conformal quantile)."""
    s = np.sort(np.asarray(neg_scores, float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(s[min(k, n - 1)]) if k < n else float('inf')


def visible_vertices_raster(sid: int, views) -> np.ndarray:
    """Vertices that appear (unoccluded) in at least one of ``views`` at the nominal cameras."""
    import torch
    import nvdiffrast.torch as dr
    from .cameras import camera
    from .diffrender import cams_to_torch, clip_space, ctx
    c = surface_context(sid)
    v = torch.tensor(c['v'], dtype=torch.float32, device='cuda')
    t = torch.tensor(c['f'], dtype=torch.int32, device='cuda')
    K, w = cams_to_torch([camera(x) for x in views])
    r, _ = dr.rasterize(ctx(), clip_space(v, K, w), t, (512, 512))
    vis = np.zeros(len(c['v']), bool)
    for b in range(len(views)):
        tid = r[b, ..., 3].long() - 1
        tid = tid[tid >= 0].unique().cpu().numpy()
        vis[c['f'][tid].ravel()] = True
    return vis


def visible_fraction(sid: int, truth: np.ndarray, views) -> float | None:
    d = np.linalg.norm(truth, axis=1)
    sup = d > 1e-9
    if not sup.any():
        return None
    vis = visible_vertices_raster(sid, tuple(views))
    return float((d * vis)[sup].sum() / d[sup].sum())
