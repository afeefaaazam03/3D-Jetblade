"""Build site/results_latest.html from result files (new code). Re-run any time; partial results are shown as such."""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
T11 = json.loads((ROOT / 'METHOD_FREEZE_V1_1.json').read_text())['thresholds']['v11']['tau_alpha05']
TYPES = ['dent', 'nick', 'crack', 'stain', 'healthy']


def table(head, rows, cls=''):
    h = ''.join(f'<th>{x}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def pct(a, b):
    return f'{a}/{b} <span class="m">({100 * a / b:.0f}%)</span>' if b else '–'


def four_defects():
    C = {c['case_id']: c for c in json.loads((ROOT / 'data/benchmark/cad6dmg/cases.json').read_text())}
    S = {json.loads(l)['case_id']: json.loads(l) for l in (R / 'phase4/sift.jsonl').read_text().splitlines()}
    a = collections.defaultdict(lambda: [0, 0, 0, 0])
    for f in (R / 'cad6dmg/v11').glob('cad6dmg_*.json'):
        r = json.loads(f.read_text()); d = C[r['case_id']]['defect']
        rep = r['fit_gain'] > T11 and r['pred_changed_vertices'] > 0
        loc = rep and r.get('centre_err') is not None and r['centre_err'] < .1
        a[d][0] += 1; a[d][1] += rep; a[d][2] += loc; a[d][3] += S[r['case_id']]['reported']
    rows = []
    for d in TYPES:
        n, rep, loc, s = a[d]
        neg = d in ('stain', 'healthy')
        rows.append([d.capitalize() + (' <span class="m">(should be 0)</span>' if neg else ''),
                     n, pct(rep, n) if neg else pct(loc, n), pct(s, n)])
    return table(['Defect', 'Cases', 'Ours: located (damage) / reported (negatives)', 'SIFT: reported'], rows)


def specialists():
    p = R / 'phase5/specialists_cad6dmg.jsonl'
    if not p.exists():
        return '<p class="m">Not started.</p>', 0
    X = [json.loads(l) for l in p.read_text().splitlines()]
    labels = ['dent', 'nick', 'crack', 'uncertain', 'stain', 'nothing']
    M = {t: collections.Counter() for t in TYPES}
    for x in X:
        M[x['defect']][x['label']] += 1
    rows = [[t.capitalize(), sum(M[t].values())] + [f'<b>{M[t][l]}</b>' if (l == t or (t == 'healthy' and l == 'nothing')) and M[t][l] else (M[t][l] or '') for l in labels] for t in TYPES]
    dmg = [x for x in X if x['defect'] in ('dent', 'nick', 'crack') and x.get('reported') and x.get('centre_err_frozen') is not None and x['centre_err_frozen'] < .1]
    typed = [x for x in dmg if x['label'] != 'uncertain']
    corr = sum(x['label'] == x['defect'] for x in typed)
    neg = [x for x in X if x['defect'] in ('stain', 'healthy')]
    neg_ok = sum((x['label'] == 'stain') if x['defect'] == 'stain' else (x['label'] == 'nothing') for x in neg)
    loc_ok = sum(x['centre_err_best'] is not None and x['centre_err_best'] < .1 for x in dmg)
    summ = (f'<div class="kpis"><div><b>{pct(corr, len(typed))}</b><span>type correct (when a type is given)</span></div>'
            f'<div><b>{pct(len(dmg) - len(typed), len(dmg))}</b><span>"type uncertain"</span></div>'
            f'<div><b>{pct(loc_ok, len(dmg))}</b><span>winning specialist still in the right place</span></div>'
            f'<div><b>{pct(neg_ok, len(neg))}</b><span>stains / healthy labelled correctly</span></div></div>')
    note = f'<p class="m">{len(X)} of 136 cases processed{" (still running; refresh later)" if len(X) < 136 else ""}. Type accuracy is over damage the frozen detector located.</p>'
    return note + summ + table(['True type', 'n'] + [l.capitalize() for l in labels], rows, 'conf'), len(X)


def weighted():
    W = json.loads((R / 'phase5/weighted_recall.json').read_text())
    return table(['Test set', 'Visible defects', 'Standard recall', 'Safety-weighted recall', 'Critical recall'],
                 [[w['set'], w['n'], f"{100 * w['recall']:.0f}%", f"<b>{100 * w['weighted']:.0f}%</b>", f"{100 * w['critical']:.0f}% <span class='m'>(n={w['n_critical']})</span>"] for w in W])


def main():
    spec, n = specialists()
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BladeDelta Latest Results</title><style>
:root{{--bg:#f7f8fb;--card:#fff;--fg:#1f2937;--m:#5a6577;--ac:#0047bb;--line:#e3e7ef;--hl:#e8f0fe}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0f1420;--card:#171e2d;--fg:#e6e9ef;--m:#9aa3b2;--ac:#7aa7ff;--line:#2a3347;--hl:#1d2a44}}}}
:root[data-theme="dark"]{{--bg:#0f1420;--card:#171e2d;--fg:#e6e9ef;--m:#9aa3b2;--ac:#7aa7ff;--line:#2a3347;--hl:#1d2a44}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:28px 16px 60px}}h1{{font-size:28px;margin:0 0 4px;color:var(--ac)}}
h2{{margin:0 0 6px;font-size:20px}}.sub{{color:var(--m);margin:0 0 22px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:16px 0}}
.m{{color:var(--m);font-size:.92em}}.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:15px}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{color:var(--m);font-weight:600;font-size:13px;text-transform:uppercase;letter-spacing:.03em}}
table.conf td:not(:first-child),table.conf th:not(:first-child){{text-align:center}}table.conf b{{background:var(--hl);padding:2px 8px;border-radius:6px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:10px 0}}.kpis div{{background:var(--hl);border-radius:10px;padding:10px 12px}}
.kpis b{{display:block;font-size:20px}}.kpis span{{color:var(--m);font-size:13px}}ul{{margin:6px 0;padding-left:20px}}
</style></head><body><main>
<h1>BladeDelta: latest results</h1><p class="sub">Finding 3D damage on turbine blades from a CAD model and a few photos, and saying when the photos are not enough. All results on computer-rendered blades. Author: Afeefa Azam.</p>
<section><h2>1. Damage-type specialists + judge</h2>
<p>Four specialists (dent, nick, crack, stain) each try to explain the photos; a judge picks the clear winner, says <i>type uncertain</i> when two fit about equally well, and calls it a stain or nothing when no physical damage is detected. Detection is the unchanged frozen method, so false alarms do not change. 136 cases on four held-out industrial blade designs.</p>{spec}</section>
<section><h2>2. Four defect types: detection</h2><p>Frozen method, 2 photos, no retuning. Stains and healthy blades should not be reported.</p>{four_defects()}
<p class="m">False alarms 6 of 48 (12.5%) are above the 5% target set on Rotor37 blades; recalibrating on these designs is the next step.</p></section>
<section><h2>3. Safety-weighted recall</h2><p>Deeper defects count more (weight = true depth in mm; cracks count double). Critical = deeper than 0.2&nbsp;mm, or any crack.</p>{weighted()}
<p class="m">On Rotor37 most misses are tiny dents: plain recall 54%, but 92% of dangerous dents are found.</p></section>
<section><h2>4. What we learned about depth</h2><ul>
<li>Location works across blade designs; depth does not: found dents are measured at a median 60% of their true depth on new designs (89% on Rotor37).</li>
<li>Four photos instead of two find more dents (45 vs 36 of 60) but do not improve depth.</li>
<li>Depth ranges do not help: they contained the true depth in only 36 of 109 cases, so the error is a systematic bias, not ambiguity.</li>
<li>Next: low-angle (grazing) light and an appearance model trained on industrial blades.</li></ul></section>
<p class="m">Built from the result files; refresh after new runs finish.</p></main></body></html>"""
    out = ROOT / 'site/results_latest.html'; out.write_text(html); print('written', out, 'specialist cases', n)


if __name__ == '__main__':
    main()
