"""Phase 6, task 1 (new code): the professor's output vector for one test case.

    [Defective (bool), Defect Type (str), Intensity (mm), Count (int)]

Defective  = frozen v1.1 evidence test (fit gain > threshold).
Type       = judge label from the specialists (phase 5) when available, else "dent"; "none" if not defective;
             "uncertain" when the judge abstains on type.
Intensity  = depth of the winning specialist in mm (0.0 if not defective).
Count      = number of separate defects reported (phase 6 counting if available, else 1 / 0).

    python phase6/output_vector.py <case_id> [--json]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bladedelta.data import mm_per_unit  # noqa: E402

T11 = json.loads((ROOT / 'METHOD_FREEZE_V1_1.json').read_text())['thresholds']['v11']['tau_alpha05']
SPEC = {'cad6dmg': 'specialists_cad6dmg.jsonl', 'cad6dmg2': 'specialists2_cad6dmg2.jsonl'}


def _jsonl(p):
    return {json.loads(l)['case_id']: json.loads(l) for l in p.read_text().splitlines()} if p.exists() else {}


def output_vector(cid):
    split = cid.rsplit('_', 2)[0]
    r = json.loads((ROOT / f'results/{split}/v11/{cid}.json').read_text())
    sid = r['source']; mm = mm_per_unit(sid)
    defective = bool(r['fit_gain'] > T11 and r['pred_changed_vertices'] > 0)
    if not defective:
        return [False, 'none', 0.0, 0]
    dtype, depth = 'dent', float(r['pred_max_disp']) * mm
    s = _jsonl(ROOT / 'results/phase5' / SPEC.get(split, '')).get(cid) if split in SPEC else None
    if s and s.get('label'):
        dtype = s['label']
        best = s['specialists'][s['best']]
        depth = float(best['depth']) * mm
    cnt = _jsonl(ROOT / f'results/phase6/count_{split}.jsonl').get(cid)
    count = int(cnt['count']) if cnt else 1
    return [True, dtype, round(depth, 3), count]


if __name__ == '__main__':
    cid = sys.argv[1]; v = output_vector(cid)
    print(json.dumps(v) if '--json' in sys.argv else f'{cid}: {v}')
