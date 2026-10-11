"""Public API.

    from bladedelta import analyze
    res = analyze(observed=[img_v004, img_v005], healthy=[ref_v004, ref_v005], source_id=83)
    print(res.decision, res.depth_mm, res.fit_gain, res.threshold)

Inputs follow the BladeDelta capture rig (bladedelta/cameras.py): two 512x512 RGB photographs from views v004 and v005,
and healthy renders of the CAD from the same nominal cameras. The CAD mesh is looked up by source id in examples/meshes
(or $BLADEDELTA_MESHES). A CUDA GPU and nvdiffrast are required; one case takes about 1-2 minutes.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPAN_MM = 74.0          # Rotor37 blade span; meshes are normalised to unit span


@dataclass
class Result:
    decision: str                 # 'dent supported' or 'not supported'
    fit_gain: float               # evidence: loss(no dent) - loss(dent)
    threshold: float              # calibrated threshold for this version
    depth_mm: float
    radius_mm: float
    centre: list                  # dent centre, normalised blade coordinates
    displacement: np.ndarray = field(repr=False)   # (N, 3) per-vertex displacement (zeros if not supported)
    version: str = 'v1.1'


def thresholds():
    return json.loads((ROOT / 'models/thresholds.json').read_text())


def load_shader(device='cuda'):
    import torch
    from .diffrender import NeuralShader
    sh = NeuralShader().to(device)
    sh.load_state_dict(torch.load(ROOT / 'models/shader.pt', map_location=device)); sh.eval()
    for p in sh.parameters():
        p.requires_grad_(False)
    return sh


def analyze(observed, healthy, source_id, version='v1.1', shader=None, seed=0):
    """Run the evidence-bounded estimator on one capture. Returns a Result."""
    from .estimator import Config, estimate
    from .v1_1 import estimate_v11
    shader = shader or load_shader()
    cfg = Config()
    run = estimate_v11 if version == 'v1.1' else estimate
    r = run(source_id, list(observed), shader, cfg, bias=list(healthy)) if version == 'v1.1' else \
        estimate(source_id, list(observed), shader, cfg, bias=list(healthy), seed=seed)
    t = thresholds()[version]
    ok = r['fit_gain'] > t and np.abs(r['displacement']).max() > 2e-5
    return Result(decision='dent supported' if ok else 'not supported', fit_gain=float(r['fit_gain']), threshold=t,
                  depth_mm=float(r['depth']) * SPAN_MM if ok else 0., radius_mm=float(r['radius']) * SPAN_MM if ok else 0.,
                  centre=r['centre'] if ok else [], displacement=r['displacement'] if ok else 0 * r['displacement'], version=version)
