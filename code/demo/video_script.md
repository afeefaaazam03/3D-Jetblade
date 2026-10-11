# BladeDelta demo video — script, storyboard, recording guide (3:00)

Audience: researchers, reviewers and engineers who have not read the paper. Tone: plain, specific, no hype.
A captioned draft cut (`demo/bladedelta_demo_draft.mp4`) uses the same storyboard; record narration over it or re-record the screen.

| Time | Section | On screen | Narration (read at ~140 words/min) |
|---|---|---|---|
| 0:00–0:20 | Problem | Photo of a blade; zoom on a dark patch. Overlay four labels: dent · stain · lighting · camera error | "A dark patch on a blade could be a dent, a stain, a lighting change or a camera that moved by a degree. An inspector needs to know which — and if it is a dent, where it is and how deep." |
| 0:20–0:40 | Input | `try.html`: the two input photos and the healthy CAD render; drag the before/after slider | "The input is the healthy CAD model of the blade and two calibrated photographs. A 0.1 millimetre dent changes the picture by only a few grey levels — far less than a small camera error." |
| 0:40–1:20 | Method (4 stages) | Diagram (paper Fig. 1) or four title cards: 1 fit nuisance · 2 propose dents · 3 fit dent · 4 evidence test | "First, explain the photos with nuisance only: camera pose from the blade outline, light levels, and a possible stain. Second, a score test over the whole surface proposes where a dent would help most. Third, fit a continuous dent there with a differentiable renderer. Fourth, compare: how much better does 'nuisance plus dent' explain the photos than nuisance alone? Report the dent only if that evidence beats a threshold calibrated on healthy and stained blades." |
| 1:20–1:50 | Output | `try.html` → "Visible dent": verdict, depth 0.35 mm, evidence bar; then the colour-coded depth map | "Here the method reports a dent 0.35 millimetres deep, at the right place; the true depth is 0.36. The depth map shows the reported shape next to the truth." |
| 1:50–2:10 | Abstention | `try.html` → "Stain only", then "Hidden dent", "Camera moved" | "A stain changes the colour, not the shape: not supported. A dent on the far side leaves no evidence in these photos: not supported — the method declines rather than guess. A moved camera: not supported." |
| 2:10–2:30 | Transfer | `results.html` → "Your six blades": v1 vs v1.1 examples and the per-method chart | "On new industrial blade designs, the frozen model raised false alarms on most healthy blades. We traced it to the camera check and the reference correction; fixing both, without retraining, cut false alarms on held-out blades from 51 to 4 percent." |
| 2:30–2:45 | Results | Paper Fig. 2 (recall vs false alarms) and the depth table | "At a matched five percent false-alarm rate, it locates 51 percent of visible dents, against at most 19 for deformation-field baselines and a 2D detector; 12 of 13 dents deeper than 0.2 millimetres are found." |
| 2:45–3:00 | Links | Title card with URLs | "Paper, code, notebook and the interactive demo are linked below." Show: afeefaaazam03.github.io/3D-Jetblade/try.html · github.com/afeefaaazam03/3D-Jetblade (code/) |

## Recording instructions

1. **Setup:** 1920×1080 screen, browser zoom 110%, dark-mode off, notifications off. Open tabs in order: `try.html`, `results.html`, the paper PDF at Fig. 2.
2. **Capture:** OBS Studio → Settings → Output: MP4, 30 fps, CRF 18. Scene = "Display capture" cropped to the browser.
3. **Narration:** record voice separately (Audacity, 48 kHz mono, noise reduction), one take per section; leave 0.5 s gaps.
4. **Edit:** cut each section to its time slot; add the section title as a lower-third for 2 s; duck nothing (no music needed).
5. **Assemble with ffmpeg** (if editing without an NLE):
   `ffmpeg -i screen.mp4 -i voice.wav -c:v libx264 -crf 20 -c:a aac -b:a 160k -shortest bladedelta_demo.mp4`
6. **Captions:** paste the narration column into a `.srt` (one line per sentence) and upload it with the video.
7. **Upload:** YouTube (unlisted while the paper is under double-blind review), then add the link to the README and `try.html`.

## Accuracy notes (keep the narration consistent with the paper)
- Say "12 of 13" for dents deeper than 0.2 mm, not "all".
- The demo examples are rendered blades; real photographs are pending physical validation — say so if asked.
- The baselines are reimplementations of CADSplat's core idea, not the official model.
