"""Phase 4 (new code): classical SIFT change-detection baseline (image level, no 3D).

For each input view: SIFT keypoints on the photo and on the healthy reference render (same nominal camera), ratio-test
matching, RANSAC homography to absorb small camera error, then count photo keypoints on the blade that have no match
within 3 px after alignment. Score = unmatched count summed over the 2 views. Threshold = split-conformal 95% quantile
on healthy + stain cases of blades 1001-1002 (same rule as ours). Reports detection only (no location, no depth).

    python phase4/sift_baseline.py
Writes results/phase4/sift.jsonl
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'data/benchmark'; VIEWS = ('v004', 'v005')
sift = cv2.SIFT_create(contrastThreshold=.02)


def score(split, cid, sid):
    tot = 0
    for v in VIEWS:
        a = cv2.imread(str(B / split / 'renders' / cid / f'{v}.png'), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(B / split / 'refs' / f'ref_{sid:04d}' / f'{v}.png'), cv2.IMREAD_GRAYSCALE)
        mask = (cv2.erode((b > 8).astype(np.uint8), np.ones((9, 9), np.uint8)) * 255)
        ka, da = sift.detectAndCompute(a, mask); kb, db = sift.detectAndCompute(b, None)
        if da is None or db is None or len(kb) < 4:
            tot += 0 if da is None else len(ka); continue
        m = cv2.BFMatcher().knnMatch(da, db, k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < .8 * x[1].distance]
        H = None
        if len(good) >= 8:
            H, _ = cv2.findHomography(np.float32([ka[g.queryIdx].pt for g in good]), np.float32([kb[g.trainIdx].pt for g in good]), cv2.RANSAC, 3.)
        pa = np.float32([k.pt for k in ka]); pb = np.float32([k.pt for k in kb])
        if H is not None:
            pa = cv2.perspectiveTransform(pa[None], H)[0]
        matched = {g.queryIdx for g in good if np.linalg.norm(pa[g.queryIdx] - pb[g.trainIdx]) < 3}
        tot += len(ka) - len(matched)
    return tot


def main():
    out = ROOT / 'results/phase4/sift.jsonl'; out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ('cad6', 'cad6dmg'):
        for c in json.loads((B / split / 'cases.json').read_text()):
            if split == 'cad6' and not (c['source'] <= 1002 and c['kind'] in ('healthy', 'colour')):
                continue
            if not (B / split / 'renders' / c['case_id'] / 'v005.png').exists():
                continue
            rows.append(dict(split=split, case_id=c['case_id'], source=c['source'], defect=c.get('defect', c['kind']),
                             score=score(split, c['case_id'], c['source'])))
    cal = sorted(r['score'] for r in rows if r['split'] == 'cad6'); n = len(cal)
    tau = cal[min(n - 1, int(np.ceil((n + 1) * .95)) - 1)]
    with out.open('w') as f:
        for r in rows:
            r['tau'] = tau; r['reported'] = r['score'] > tau; f.write(json.dumps(r) + '\n')
    print('calibration n', n, 'tau', tau)
    for d in ('dent', 'nick', 'crack', 'stain', 'healthy'):
        R = [r for r in rows if r['split'] == 'cad6dmg' and r['defect'] == d]
        if R:
            print(d, 'reported %d/%d' % (sum(r['reported'] for r in R), len(R)))


if __name__ == '__main__':
    main()
