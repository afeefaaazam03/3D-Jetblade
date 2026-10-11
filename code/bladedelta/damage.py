"""Ground-truth damage generators for the benchmark.

Truth uses geodesic distance on the healthy airfoil and several profile
families. The estimator (estimator.py) uses a different, continuous family, so
exact recovery is impossible by construction and errors are meaningful.
All lengths are in normalized blade-height units (height = 1).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .geometry import surface_context

FAMILIES = ('cubic_rim', 'gaussian', 'elliptic')


def tangent_frame(n: np.ndarray):
    a = np.array([0., 0., 1.]) if abs(n[2]) < .9 else np.array([1., 0., 0.])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)


def _rim_compensated(sid, distance, radius, normal):
    """The earlier project's cubic core + volume-compensating rim profile."""
    q = distance / radius
    core = np.where(q < .5, (1 - 4 * q ** 2) ** 3, 0.)
    outer = (q > .5) & (q < 1)
    rim = np.zeros_like(q); rim[outer] = (16 * (q[outer] - .5) * (1 - q[outer])) ** 3
    return core, rim


def dent_field(sid: int, center: int, depth: float, radius: float, family: str,
               aspect: float = 1., angle: float = 0.) -> np.ndarray:
    """Per-vertex displacement (N, 3) of an inward dent on the airfoil."""
    ctx = surface_context(sid)
    v, n = ctx['v'], ctx['normals']
    nc = n[center]
    dist = dijkstra(ctx['graph'], directed=False, indices=center, limit=3 * radius)
    dist[~np.isfinite(dist)] = np.inf
    near = np.isfinite(dist) & (np.einsum('ij,j->i', n, nc) > .3)
    if family == 'cubic_rim':
        core, rim = _rim_compensated(sid, np.where(near, dist, np.inf), radius, nc)
        mag = -core + .35 * rim
    elif family == 'gaussian':
        sigma = radius / 2.5
        mag = -np.exp(-.5 * (dist / sigma) ** 2) * (dist < radius * 1.2)
    elif family == 'elliptic':
        t1, t2 = tangent_frame(nc)
        ca, sa = np.cos(angle), np.sin(angle)
        e1, e2 = ca * t1 + sa * t2, -sa * t1 + ca * t2
        d = v - v[center]
        s2 = (d @ e1 / radius) ** 2 + (d @ e2 / (radius * aspect)) ** 2
        mag = -np.clip(1 - s2, 0, None) ** 2
    else:
        raise ValueError(family)
    mag = np.where(near, mag, 0.)
    out = np.zeros_like(v)
    out[:] = (depth * mag)[:, None] * nc
    return out


def colour_field(sid: int, center: int, radius: float, strength: float, tint=(1., .82, .7)) -> np.ndarray:
    """Per-vertex RGB albedo multiplier: a soft darkened / rust-tinted patch."""
    ctx = surface_context(sid)
    dist = dijkstra(ctx['graph'], directed=False, indices=center, limit=2 * radius)
    w = np.where(np.isfinite(dist), np.clip(1 - (dist / radius) ** 2, 0, None) ** 2, 0.)
    tint = np.asarray(tint)
    return 1 - strength * w[:, None] * (1 - tint[None, :] * .55)


def eligible_centers(sid: int, radius: float, visible: np.ndarray | None = None) -> np.ndarray:
    """Airfoil vertices far enough from the boundary; optionally visible ones only."""
    ctx = surface_context(sid)
    ok = np.zeros(len(ctx['v']), bool); ok[ctx['used']] = True
    ok &= ctx['boundary_distance'] >= max(.1, 1.2 * radius)
    if visible is not None:
        ok &= visible
    return np.flatnonzero(ok)
