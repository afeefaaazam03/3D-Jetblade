"""Healthy Rotor37 blade meshes and surface utilities.

Reuses the audited closureguard decode / surface / collar / normalization code
so every healthy mesh matches the one earlier experiments used, byte for byte.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

import os

PACKAGE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get('BLADEDELTA_MESHES', PACKAGE.parent / 'examples/meshes'))

# Source IDs already opened by earlier project experiments. Never use for test.
EXPOSED_IDS = tuple(range(0, 13))


def _construct(sid: int) -> dict:
    raise FileNotFoundError(f'mesh for source {sid} not found in {CACHE}; set BLADEDELTA_MESHES or see README (building meshes from the Rotor37 dataset needs the original decoding pipeline)')


def _construct_original(sid: int) -> dict:
    import pyarrow.parquet as pq
    from closureguard.data.rotor37_numeric_pickle_v1_1 import decode_numeric_pickle, validate_plaid_sample
    from closureguard.data.rotor37_surface_v1_1 import extract_surface
    from closureguard.data.rotor37_hub_collar_v1 import construct_hub_collar
    from render_rotor37_nick_pilot_v1_1 import normalized_source

    parquet = pq.ParquetFile(SHARD)
    group, row = divmod(sid, 100)
    payload = parquet.read_row_group(group, columns=['sample']).column('sample')[row].as_py()
    sample = validate_plaid_sample(decode_numeric_pickle(payload))
    surface = extract_surface(sample, base_name='Base_2_3', zone_name='Zone', time=0.0)
    collar = construct_hub_collar(surface['vertices'], surface['triangles'])
    if not collar['audit']['signed_volume_positive']:
        raise ValueError(f'collar winding failed for source {sid}')
    normalized, _ = normalized_source(collar['vertices'], source_vertex_ids=np.arange(len(surface['vertices'])))
    return dict(vertices=normalized.astype(np.float64),
                triangles=collar['derivative_triangles'].astype(np.int32),
                triangle_component=(collar['face_roles'] != 0).astype(np.uint8))


def load_healthy(sid: int) -> dict:
    """Healthy mesh for Rotor37 shard-0 sample ``sid`` (cached as .npz)."""
    path = CACHE / f'source{sid:04d}.npz'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh = _construct(sid)
        np.savez(path, **mesh)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    p = v[f]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n = np.zeros_like(v)
    for j in range(3):
        np.add.at(n, f[:, j], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


def edge_graph(v: np.ndarray, f: np.ndarray):
    e = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    u, counts = np.unique(np.sort(e, axis=1), axis=0, return_counts=True)
    w = np.linalg.norm(v[u[:, 0]] - v[u[:, 1]], axis=1)
    g = coo_matrix((np.r_[w, w], (np.r_[u[:, 0], u[:, 1]], np.r_[u[:, 1], u[:, 0]])), shape=(len(v),) * 2).tocsr()
    return g, u, counts


@lru_cache(maxsize=32)
def surface_context(sid: int) -> dict:
    """Airfoil-only normals, edge graph and distance to the airfoil boundary."""
    m = load_healthy(sid)
    v, f, c = m['vertices'], m['triangles'], m['triangle_component']
    air = f[c == 0]
    used = np.unique(air)
    n = vertex_normals(v, air)
    g, edges, counts = edge_graph(v, air)
    boundary = np.unique(edges[counts == 1])
    bd = dijkstra(g, directed=False, indices=boundary, min_only=True)
    return dict(mesh=m, v=v, f=f, c=c, airfoil=air, used=used, normals=n, graph=g, boundary_distance=bd)
