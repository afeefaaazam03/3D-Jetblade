# BladeDelta: evidence-bounded 3D damage estimation

Given a healthy CAD model of a blade and a few calibrated photographs, BladeDelta decides whether the photographs support a
**physical dent** better than every non-physical explanation (camera error, lighting change, a stain). If they do, it reports
the dent's location, depth and shape on the CAD surface; if they do not, it says **"not supported"** instead of guessing.
The decision threshold is calibrated on healthy and stain-only blades, so the false-alarm rate is controlled.

**Paper:** [`paper/BladeDelta.pdf`](paper/BladeDelta.pdf) · **Demo:** see [Demo](#demo) · **Notebook:** [`notebooks/BladeDelta_quickstart.ipynb`](notebooks/BladeDelta_quickstart.ipynb)

## Results (sealed test, 30 unseen Rotor37 blades)

| | Ours | MLP field* | 2D detector | Free-form field* | Discrete search |
|---|---|---|---|---|---|
| Visible dents located at 5% false alarms | **51%** | 8% | 16% | 19% | 5% |
| Visible dents > 0.2 mm reported | **12 / 13** | | | | |
| Median depth error (reported dents) | **0.03 mm** | | | | |

On six unseen industrial blade designs, v1.1 (which makes the healthy-reference correction follow the silhouette and the
camera) cuts false alarms from 51% to 4% on held-out blades (AUROC 0.72 → 0.95), with no change on Rotor37.
\*Reimplementations of the core idea of CADSplat, not the official model.

## Installation

Requires Linux, Python 3.10+, an NVIDIA GPU and CUDA (for [nvdiffrast](https://github.com/NVlabs/nvdiffrast)).

```bash
git clone <this repository> && cd BladeDelta
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
pip install -e .
```

## Quick start

```python
from bladedelta import analyze
from bladedelta.utils import load_example

ex = load_example('01_dent_found')
res = analyze(ex['observed'], ex['healthy'], source_id=ex['truth']['source'], version='v1.1')
print(res.decision, f'{res.depth_mm:.2f} mm', 'evidence', round(res.fit_gain, 1), 'threshold', round(res.threshold, 2))
print('truth:', ex['truth']['dent'])
```

Each example folder holds the two input photographs (`observed_v004.png`, `observed_v005.png`), the healthy CAD renders from
the same cameras, the ground truth, and the precomputed outputs of v1 and v1.1 (with a picture: true dent as orange rings,
reported dent as blue dots).

| Example | Truth | v1.1 decision |
|---|---|---|
| `01_dent_found` | dent, visible | dent supported, located |
| `02_hidden_dent` | dent on the far side | not supported (nothing visible) |
| `03_stain_only` | colour change, no shape change | not supported |
| `04_healthy` | nothing changed | not supported |
| `05_camera_error` | healthy, cameras moved up to 1.5° | not supported |

## Demo

```bash
pip install gradio
python demo/app.py            # http://localhost:7860
```

The demo shows the five examples. With a GPU it can re-run the method live ("Run live"); without one it shows the
precomputed outputs of the same frozen model, clearly labelled.

## Scope and limitations

- Inputs must come from the BladeDelta camera rig (`bladedelta/cameras.py`); arbitrary photographs need camera calibration.
- The appearance model was trained on rendered Rotor37 blades only. Real photographs are future work (pending physical validation).
- One dent per blade; edge nicks and material loss are out of scope.

## Citation

```bibtex
@misc{bladedelta2026,
  title  = {Which Damage Do the Images Support? Evidence-Bounded 3D Damage Estimation on Turbine Blades},
  note   = {Under review},
  year   = {2026}
}
```

## Licence

Code: MIT (see `LICENSE`). Example data are rendered from public Rotor37 geometry (PLAID, CC BY-SA 4.0) and are shared under
CC BY-SA 4.0.
