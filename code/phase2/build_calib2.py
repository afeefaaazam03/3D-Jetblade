"""Phase 2: extra calibration negatives (healthy + stain only) on the calibration shapes 25-36, new seed.
Only the two input views are rendered; healthy references are shared with the calib split (read-only symlink).

    python phase2/build_calib2.py cases | render
"""
import json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import build_benchmark as bb  # noqa: E402  (read-only use)
from bladedelta.cameras import FIT_VIEWS  # noqa: E402
from bladedelta.cases import all_views, sample_case  # noqa: E402
OUT = ROOT / 'data/benchmark/calib2'
KINDS = ['healthy'] * 10 + ['colour'] * 15        # 25 negatives per shape, 300 in total


def build_cases():
    if (OUT / 'cases.json').exists():
        raise SystemExit('calib2 already built')
    rng = np.random.default_rng(2626)
    cases = []
    for sid in range(25, 37):
        for k, kind in enumerate(KINDS):
            cap = bb.capture(rng, bb.CAPTURES[rng.integers(4)])
            cap['views'] = {v: cap['views'][v] for v in FIT_VIEWS}
            c = sample_case(rng, f'calib2_{sid:04d}_{k:02d}', sid, kind, all_views(), capture=cap)
            c['capture'] = cap; cases.append(c)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cases.json').write_text(json.dumps(cases, indent=0))
    if not (OUT / 'refs').exists():
        os.symlink('../calib/refs', OUT / 'refs')
    print(len(cases), 'cases')


if __name__ == '__main__':
    build_cases() if sys.argv[1] == 'cases' else bb.render('calib2')
