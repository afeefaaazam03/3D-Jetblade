"""BladeDelta interactive demo (Gradio).

    pip install gradio && python demo/app.py

Five pre-loaded sealed-test cases show the frozen model's real outputs (precomputed, works on CPU, e.g. Hugging Face Spaces).
On a machine with a CUDA GPU and nvdiffrast the method can also be re-run live, including on your own data: two photographs
from the BladeDelta rig views v004/v005, plus the CAD mesh (.npz, BladeDelta format) and optionally healthy renders.
"""
import json
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / 'examples').exists() else HERE
EX = ROOT / 'examples'
sys.path.insert(0, str(ROOT))

TITLES = {'01_dent_found': 'Visible dent', '02_hidden_dent': 'Hidden dent (far side)', '03_stain_only': 'Stain only',
          '04_healthy': 'Healthy blade', '05_camera_error': 'Healthy, camera moved'}
WHY = {'01_dent_found': 'A real dent the two cameras can see.',
       '02_hidden_dent': 'A real dent on the side neither camera sees: the photographs contain no evidence of it.',
       '03_stain_only': 'A colour change with no shape change (appearance only).',
       '04_healthy': 'Nothing changed on the blade.',
       '05_camera_error': 'Nothing changed on the blade, but the cameras were moved by up to 1.5 degrees.'}


def live_available():
    try:
        import torch  # noqa: F401
        import nvdiffrast.torch  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


LIVE = live_available() and os.environ.get('BLADEDELTA_PRECOMPUTED_ONLY') != '1'


def thresholds():
    return json.loads((ROOT / 'models/thresholds.json').read_text())


def headline(decision, depth_mm):
    return f'Dent found, {depth_mm:.2f} mm deep' if decision == 'dent supported' else 'Not supported: no dent reported'


def confidence_text(gain, tau):
    ratio = gain / tau if tau else 0
    return (f'Evidence score {gain:.2f} vs threshold {tau:.2f} ({ratio:.1f}x). The threshold is set so that about 5% of '
            f'healthy or stained blades exceed it.')


def show_example(name, version):
    d = EX / name
    t = json.loads((d / 'truth.json').read_text())
    p = json.loads((d / f'precomputed_{version}.json').read_text())
    truth = WHY[name] + (f" True dent: {t['dent']['depth_mm']:.2f} mm deep, radius {t['dent']['radius_mm']:.1f} mm." if t['dent'] else '')
    photos = [Image.open(d / f'observed_{v}.png') for v in t['views']]
    return (headline(p['decision'], p['depth_mm']), round(p['depth_mm'], 3) if p['decision'] == 'dent supported' else 0.0,
            confidence_text(p['fit_gain'], p['threshold']), Image.open(d / f'panel_{version}.jpg'),
            f'Precomputed output of the frozen model ({version}). Ground truth: {truth}', photos[0], photos[1])


def run_live_example(name, version):
    if not LIVE:
        return show_example(name, version)[:5]
    from bladedelta import analyze
    from bladedelta.utils import load_example
    ex = load_example(name)
    r = analyze(ex['observed'], ex['healthy'], source_id=ex['truth']['source'], version=version)
    viz = overlay(ex['observed'][1], r, ex['truth']['source'])
    return (headline(r.decision, r.depth_mm), round(r.depth_mm, 3), confidence_text(r.fit_gain, r.threshold), viz,
            f'Computed live just now ({version}).')


def overlay(img, r, source_id, view='v005'):
    """Photo with the reported dent drawn in blue (vertices that moved), projected through the nominal camera."""
    out = np.asarray(img).copy()
    if r.decision != 'dent supported':
        return Image.fromarray(out)
    from bladedelta.cameras import camera
    from bladedelta.geometry import surface_context
    ctx = surface_context(source_id)
    sel = np.linalg.norm(r.displacement, axis=1) > 2e-5
    c = camera(view); w = c['w2c_cv']
    X = ctx['v'][sel] @ w[:3, :3].T + w[:3, 3]; uv = X @ c['K'].T; uv = (uv[:, :2] / uv[:, 2:]).astype(int)
    for u, v in uv:
        if 2 <= u < out.shape[1] - 2 and 2 <= v < out.shape[0] - 2:
            out[v - 2:v + 3, u - 2:u + 3] = (40, 110, 230)
    return Image.fromarray(out)


def run_upload(photo_a, photo_b, cad_file, ref_a, ref_b, version):
    if not LIVE:
        return ('Live analysis needs a CUDA GPU', 0.0, 'This server has no GPU. Run `python demo/app.py` on a GPU machine, '
                'or use the Colab notebook.', None, '')
    if photo_a is None or photo_b is None or cad_file is None:
        return ('Missing input', 0.0, 'Upload both photographs and the CAD mesh (.npz).', None, '')
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp()); sid = 9000
    shutil.copy(cad_file if isinstance(cad_file, str) else cad_file.name, tmp / f'source{sid:04d}.npz')
    os.environ['BLADEDELTA_MESHES'] = str(tmp)
    import importlib
    import bladedelta.geometry as g
    importlib.reload(g); g.CACHE = tmp
    from bladedelta import analyze
    obs = [np.asarray(Image.fromarray(photo_a).convert('RGB').resize((512, 512))), np.asarray(Image.fromarray(photo_b).convert('RGB').resize((512, 512)))]
    refs = None
    if ref_a is not None and ref_b is not None:
        refs = [np.asarray(Image.fromarray(x).convert('RGB').resize((512, 512))) for x in (ref_a, ref_b)]
    r = analyze(obs, refs if refs else obs, source_id=sid, version=version) if refs else analyze(obs, [None, None], source_id=sid, version=version)
    return (headline(r.decision, r.depth_mm), round(r.depth_mm, 3), confidence_text(r.fit_gain, r.threshold), overlay(obs[1], r, sid), 'Computed live.')


CSS = '.gr-prose h1 {font-size: 1.6rem}'
with gr.Blocks(title='BladeDelta demo', css=CSS) as demo:
    gr.Markdown('# BladeDelta: does the photograph support a dent?\n'
                'Given a healthy CAD model and two photographs, the method reports a dent (location and depth) only when the '
                'images support it better than camera error, lighting or a stain; otherwise it says **not supported**.\n\n'
                'In the result picture, **orange rings** are the true dent and **blue dots** are what the method reports; '
                'middle: healthy reference; right: difference.')
    with gr.Tab('Examples'):
        with gr.Row():
            name = gr.Dropdown([(TITLES[k], k) for k in TITLES], value='01_dent_found', label='Example')
            version = gr.Radio([('v1.1 (recommended)', 'v1.1'), ('v1 (frozen, paper main test)', 'v1')], value='v1.1', label='Model')
        with gr.Row():
            pa = gr.Image(label='Input photo, camera 1', height=260)
            pb = gr.Image(label='Input photo, camera 2', height=260)
        with gr.Row():
            go = gr.Button('Analyze (precomputed result)', variant='primary')
            live = gr.Button('Run live on GPU' if LIVE else 'Run live (needs a GPU; not available here)', interactive=LIVE)
        decision = gr.Textbox(label='Decision')
        with gr.Row():
            depth = gr.Number(label='Depth (mm)')
            conf = gr.Textbox(label='Confidence')
        viz = gr.Image(label='Result: photo with true (orange) and reported (blue) dent | healthy | difference')
        note = gr.Markdown()
        go.click(show_example, [name, version], [decision, depth, conf, viz, note, pa, pb])
        live.click(run_live_example, [name, version], [decision, depth, conf, viz, note])
        name.change(show_example, [name, version], [decision, depth, conf, viz, note, pa, pb])
        demo.load(show_example, [name, version], [decision, depth, conf, viz, note, pa, pb])
    with gr.Tab('Your own data (GPU)'):
        gr.Markdown('Upload two 512x512 photographs taken from the BladeDelta rig views v004 and v005, the healthy CAD mesh '
                    '(.npz with `vertices`, `triangles`, `triangle_component`), and ideally healthy renders from the same cameras. '
                    + ('' if LIVE else '**This server has no GPU, so this tab cannot run here.**'))
        with gr.Row():
            ua = gr.Image(label='Photo, view v004'); ub = gr.Image(label='Photo, view v005')
        with gr.Row():
            ra = gr.Image(label='Healthy render, v004 (optional)'); rb = gr.Image(label='Healthy render, v005 (optional)')
        cad = gr.File(label='CAD mesh (.npz)')
        uv = gr.Radio(['v1.1', 'v1'], value='v1.1', label='Model')
        ugo = gr.Button('Analyze', variant='primary', interactive=LIVE)
        u_dec = gr.Textbox(label='Decision'); u_depth = gr.Number(label='Depth (mm)'); u_conf = gr.Textbox(label='Confidence')
        u_viz = gr.Image(label='Reported dent'); u_note = gr.Markdown()
        ugo.click(run_upload, [ua, ub, cad, ra, rb, uv], [u_dec, u_depth, u_conf, u_viz, u_note])
    gr.Markdown('Paper and code: [github.com/afeefaaazam03/3D-Jetblade](https://github.com/afeefaaazam03/3D-Jetblade/tree/main/code) · '
                'Results: [afeefaaazam03.github.io/3D-Jetblade/results.html](https://afeefaaazam03.github.io/3D-Jetblade/results.html)')

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 7860)))
