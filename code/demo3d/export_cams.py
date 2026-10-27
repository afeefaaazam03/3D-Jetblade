"""Export the two input-camera centres per demo blade, in the mesh coordinates of site/demo3d/mesh_<id>.bin (new code)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bladedelta.cameras import FIT_VIEWS, camera  # noqa: E402
from bladedelta.data import mm_per_unit  # noqa: E402
from bladedelta.geometry import load_healthy  # noqa: E402

cases = json.loads((ROOT / 'site/demo3d/cases.json').read_text())
out = {}
for sid in sorted({c['mesh'] for c in cases}):
    v = load_healthy(sid)['vertices']; mu = v.mean(0); mm = mm_per_unit(sid)
    out[str(sid)] = {vid: ((camera(vid)['c2w_blender'][:3, 3] - mu) * mm).round(2).tolist() for vid in FIT_VIEWS}
(ROOT / 'site/demo3d/cams.json').write_text(json.dumps(out))
print(len(out), 'blades;', out[str(cases[0]['mesh'])])
