"""Build site/phase2.html from results/phase2 (new; does not touch the existing results page). CPU only."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / 'results/phase2'
SITE = ROOT / 'site'
IMG = SITE / 'img/phase2'


def auroc(p, n):
    if not p or not n:
        return None
    p, n = np.array(p)[:, None], np.array(n)[None]
    return float(((p > n) + .5 * (p == n)).mean())


def load(split):
    f = RES / f'detectability_{split}.jsonl'
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def stats(R):
    det = [r for r in R if r['detected']]; mis = [r for r in R if not r['detected']]
    vis = [r for r in R if (r['visible_fraction'] or 0) >= .5]
    f = lambda key: auroc([key(r) for r in det], [key(r) for r in mis])
    tau = R[0]['tau']
    pred = [r['pred_gain'] > tau for r in R]; act = [r['detected'] for r in R]
    return dict(n=len(R), n_det=len(det), auc=f(lambda r: r['pred_gain']),
                auc_vis=auroc([r['pred_gain'] for r in vis if r['detected']], [r['pred_gain'] for r in vis if not r['detected']]),
                auc_depth=f(lambda r: r['depth_mm']), auc_visfrac=f(lambda r: r['visible_fraction'] or 0),
                auc_dv=f(lambda r: r['depth_mm'] * (r['visible_fraction'] or 0)),
                acc=float(np.mean([p == a for p, a in zip(pred, act)])),
                tp=sum(p and a for p, a in zip(pred, act)), fp=sum(p and not a for p, a in zip(pred, act)),
                fn=sum(a and not p for p, a in zip(pred, act)), tn=sum(not p and not a for p, a in zip(pred, act)))


def scatter(R, name):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    tau = R[0]['tau']
    for flag, col, lab in ((True, '#1f6feb', 'found by the method'), (False, '#cf222e', 'not found')):
        X = [r for r in R if r['detected'] == flag]
        ax.scatter([max(r['pred_gain'], 1e-3) for r in X], [max(r['actual_gain'], 1e-3) for r in X], s=14, c=col, alpha=.75, label=lab)
    ax.axvline(tau, ls='--', c='k', lw=.8); ax.axhline(tau, ls=':', c='gray', lw=.8)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-3, 1e4); ax.set_ylim(1e-3, 1e4)
    ax.set_xlabel('predicted evidence (before the dent search)'); ax.set_ylabel('actual evidence (fit gain)')
    ax.legend(frameon=False, fontsize=8, loc='upper left'); ax.set_title('Predicted vs actual, per dent', fontsize=10)
    fig.tight_layout(); fig.savefig(IMG / name, dpi=120); plt.close(fig)


def map_figure():
    """Render each computed detection-limit map on the blade (camera v005)."""
    from bladedelta.cameras import camera
    from bladedelta.geometry import surface_context
    out = []
    for f in sorted((RES / 'maps').glob('*.npz')):
        z = np.load(f); sid = int(z['source']); ctx = surface_context(sid)
        split = 'test2' if f.name.startswith('test2_') else ('cad6' if f.name.startswith('cad6_') else 'test')
        cid = f.name.split('_r')[0]
        img = plt.imread(ROOT / f'data/benchmark/{split}/renders/{cid}/v005.png')
        c = camera('v005'); w = c['w2c_cv']; v = ctx['v'][z['centres']]; n = ctx['normals'][z['centres']]
        X = v @ w[:3, :3].T + w[:3, 3]; uv = X @ c['K'].T; uv = uv[:, :2] / uv[:, 2:]
        cam = -w[:3, :3].T @ w[:3, 3]; fac = ((cam - v) * n).sum(1) > 0
        h = np.where(np.isfinite(z['h_min_mm']), z['h_min_mm'], 9.)
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        ax.imshow(img)
        sc = ax.scatter(uv[fac, 0], uv[fac, 1], c=np.clip(h[fac], 0, 1.), cmap='viridis_r', s=22, vmin=0, vmax=1.)
        cb = fig.colorbar(sc, ax=ax, fraction=.046); cb.set_label('smallest detectable dent depth (mm)')
        ax.axis('off'); ax.set_title(f'{cid}: radius {float(z["radius"]) * 74:.1f} mm dents', fontsize=9)
        name = f'map_{f.stem}.png'; fig.savefig(IMG / name, dpi=110, bbox_inches='tight'); plt.close(fig)
        fin = np.isfinite(z['h_min_mm'])
        out.append((name, cid, float(np.median(z['h_min_mm'][fin])) if fin.any() else None, int(fin.sum()), len(fin)))
    return out


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    rows, figs = [], []
    for split, label in (('test', 'Sealed test (blades 37–66)'), ('test2', 'Second sealed test (blades 67–96)')):
        R = load(split)
        if len(R) < 10:
            continue
        s = stats(R); scatter(R, f'scatter_{split}.png'); figs.append((f'scatter_{split}.png', label))
        rows.append((label, s))
    maps = map_figure() if (RES / 'maps').exists() else []
    status = (RES / 'STATUS.txt').read_text().splitlines()[-6:] if (RES / 'STATUS.txt').exists() else []
    pct = lambda x: '–' if x is None else f'{x:.2f}'
    body = [f'''<span class="eyebrow">Phase 2 · updated {time.strftime('%d %b %Y, %H:%M UTC', time.gmtime())}</span>
<h1>Knowing what the photos can show: detection limits</h1>
<p>The new idea for the paper: before searching for a dent, compute how much evidence a dent of a given depth <i>would</i> leave in
these photos after camera, lighting and a stain have explained everything they can. That gives the <b>smallest detectable dent at every
point on the blade</b>, so the method bounds its misses as well as its false alarms. Below we test whether this prediction agrees with what
the frozen method actually did. Nothing in the method was changed or re-tuned. <a href="results.html">← main results</a></p>''']
    if rows:
        body.append('<h2>Does the prediction match reality?</h2><div class="tablewrap"><table><thead><tr><th>Test set</th><th>Dents</th>'
                    '<th>AUROC: predicted evidence → found</th><th>…visible dents only</th><th>Depth alone</th><th>Visibility alone</th>'
                    '<th>Depth × visibility</th><th>Predicted correctly at the threshold</th></tr></thead><tbody>'
                    + ''.join(f"<tr><td>{l}</td><td>{s['n']} ({s['n_det']} found)</td><td><b>{pct(s['auc'])}</b></td><td><b>{pct(s['auc_vis'])}</b></td>"
                              f"<td>{pct(s['auc_depth'])}</td><td>{pct(s['auc_visfrac'])}</td><td>{pct(s['auc_dv'])}</td>"
                              f"<td>{s['acc'] * 100:.0f}% (TP {s['tp']}, TN {s['tn']}, FP {s['fp']}, FN {s['fn']})</td></tr>" for l, s in rows)
                    + '</tbody></table></div><p class="muted">AUROC 1.0 means the prediction perfectly separates the dents the method finds from those it '
                      'misses; 0.5 is chance. The prediction beats depth and visibility heuristics because it also accounts for viewing angle, lighting '
                      'and whether a stain could mimic the dent.</p>')
        body.append('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:14px">'
                    + ''.join(f'<figure><img src="img/phase2/{f}" alt="{l}"><figcaption>{l}: each dot is one dent. Left of the dashed line the '
                              f'prediction says "not detectable".</figcaption></figure>' for f, l in figs) + '</div>')
    if maps:
        body.append('<h2>Detection-limit maps</h2><p>The smallest dent depth these two photographs could reveal, at each point of the blade '
                    '(yellow: tiny dents are detectable; dark: only deep dents, or none). Computed for a healthy capture, before any search.</p>'
                    '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:14px">'
                    + ''.join(f'<figure><img src="img/phase2/{n}" alt="map {c}"><figcaption>{c}: median limit {m:.2f} mm; '
                              f'{k} of {t} points detectable at all.</figcaption></figure>' for n, c, m, k, t in maps if m is not None) + '</div>')
    body.append('<h2>Progress</h2><pre style="white-space:pre-wrap;font-size:.85rem">' + '\n'.join(status) + '</pre>')
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BladeDelta Detection Limits</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,600..800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<link rel="stylesheet" href="styles.css">
<style>main.p2{{max-width:1100px;margin:0 auto;padding:32px 16px 64px;display:grid;gap:18px}} .muted{{color:var(--muted);font-size:.9rem}}</style>
</head><body><main class="p2">
{chr(10).join(body)}
</main></body></html>'''
    (SITE / 'phase2.html').write_text(page)
    print('phase2.html written;', [(l, round(s['auc'], 3)) for l, s in rows], 'maps', len(maps))


if __name__ == '__main__':
    main()
