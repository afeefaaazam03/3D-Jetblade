"""Phase 3 diagnosis (new code): why are some found dents measured far too shallow?
For given cases, run the v1.1 estimator (a) as frozen, (b) dent start at truth location, (c) also at true depth/shape.
    python phase3/depth_probe.py <case_id> ...
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'phase3'))
import oracle_fit as O  # noqa: E402
from bladedelta import estimator as E, v1_1  # noqa: E402
from bladedelta.data import mm_per_unit  # noqa: E402
from bladedelta.renderer import load_shader  # noqa: E402

shader = load_shader()
_est = E.estimate
O.E.estimate = lambda sid, imgs, sh, cfg, bias=None: _est(sid, imgs, sh, cfg, bias=bias, problem=v1_1.ProblemV11(sid, imgs, sh, cfg, bias))
for cid in sys.argv[1:]:
    split = cid.rsplit('_', 2)[0]
    c = next(x for x in json.loads((O.B / split / 'cases.json').read_text()) if x['case_id'] == cid)
    mm = mm_per_unit(c['source'])
    out = dict(case=cid, true_mm=round(c['dent']['depth'] * mm, 3))
    for mode in ('loc', 'full'):
        r = O.run(c, shader, mode)
        out[mode] = dict(gain=round(r['fit_gain'], 2), depth_mm=round(r['depth'] * mm, 3), radius=round(r['radius'], 3))
    print(json.dumps(out), flush=True)
