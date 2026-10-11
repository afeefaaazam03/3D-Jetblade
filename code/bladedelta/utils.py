"""Small helpers for examples and the demo."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples'


def read_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('RGB')).copy()


def list_examples():
    return sorted(p.name for p in EXAMPLES.iterdir() if p.is_dir() and (p / 'truth.json').exists())


def load_example(name):
    d = EXAMPLES / name
    t = json.loads((d / 'truth.json').read_text())
    obs = [read_rgb(d / f'observed_{v}.png') for v in t['views']]
    ref = [read_rgb(d / f'healthy_{v}.png') for v in t['views']]
    pre = {k: json.loads((d / f'precomputed_{k}.json').read_text()) for k in ('v1', 'v1.1') if (d / f'precomputed_{k}.json').exists()}
    return dict(truth=t, observed=obs, healthy=ref, precomputed=pre, panel={k: d / f'panel_{k}.jpg' for k in ('v1', 'v1.1')})
