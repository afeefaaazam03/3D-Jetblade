"""Phase 7 (new code): ingest real photographs of printed blades and run the frozen pipeline on them.

Input layout (see data/real_cad6/README.md):
  data/real_cad6/<session>/
    intrinsics.json            fx, fy, cx, cy, dist[5], width, height   (from `calib`)
    board.json                 ArUco board on the fixture: dictionary, marker_length_mm, markers {id: [[x,y,z] x4 corners, mm]}
    blade_pose.json            optional: blade (CAD) pose in the board frame {R: 3x3, t_mm: [3]}; else found by `register`
    <specimen>/meta.json       {"source": 1003, "state": "healthy" | "damaged", "pair": "<healthy specimen>", "defects": [...]}
    <specimen>/<view>.jpg      one photo per rig camera, view ids e.g. v004, v005 (same camera positions for all specimens)

Commands:
  calib    <session> <checkerboard_dir> [--cols 9 --rows 6 --square_mm 20]   lens intrinsics -> intrinsics.json
  rig      <session> <specimen>          camera poses in the board frame from ArUco markers -> rig.json
  register <session> <specimen>          blade pose in the board frame by silhouette alignment -> blade_pose.json
  prepare  <session>                     undistort, crop+rescale to the method's camera model (512 px, f = FOCAL_PX),
                                         write data/benchmark/real_<session>/ (cases, renders, refs = healthy photos)
  run      <session>                     detection + specialists + counting -> output vectors
  dryrun                                 build a synthetic session from rendered cases with known poses, and run it all

Real photos differ from renders (lighting, metal): healthy photos of the same blade in the same rig are used as the
reference images, so the method compares "healthy vs. now" under identical conditions.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'scripts', ROOT / 'phase4', ROOT / 'phase5', ROOT / 'phase6'):
    sys.path.insert(0, str(p))
from bladedelta import cameras as CAM  # noqa: E402
from bladedelta.data import mm_per_unit  # noqa: E402

REAL = ROOT / 'data/real_cad6'; BENCH = ROOT / 'data/benchmark'
F_NOM = CAM.FOCAL_PX; RES = CAM.RES


# ---------------------------------------------------------------- calibration and rig
def calib(session, board_dir, cols=9, rows=6, square_mm=20.):
    objp = np.zeros((rows * cols, 3), np.float32); objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    O, I, shape = [], [], None
    for f in sorted(Path(board_dir).glob('*.*')):
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        ok, c = cv2.findChessboardCorners(g, (cols, rows))
        if ok:
            c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)); O.append(objp); I.append(c); shape = g.shape[::-1]
    if len(O) < 5: raise SystemExit(f'only {len(O)} checkerboard views found; need >= 5')
    rms, K, dist, _, _ = cv2.calibrateCamera(O, I, shape, None, None)
    out = dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], dist=dist.ravel().tolist(), width=shape[0], height=shape[1], rms_px=rms, n_views=len(O))
    (REAL / session / 'intrinsics.json').write_text(json.dumps(out, indent=1)); print('intrinsics', out)


def K_of(intr):
    return np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]]), np.array(intr['dist'])


def rig(session, specimen):
    """Camera pose of each view in the board frame (mm), from ArUco marker corners (PnP)."""
    S = REAL / session; intr = json.loads((S / 'intrinsics.json').read_text()); board = json.loads((S / 'board.json').read_text())
    K, dist = K_of(intr); dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board['dictionary']))
    det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters()); out = {}
    for img in sorted((S / specimen).glob('v*.jpg')) + sorted((S / specimen).glob('v*.png')):
        g = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE); corners, ids, _ = det.detectMarkers(g)
        if ids is None: print(img.name, 'no markers'); continue
        P3, P2 = [], []
        for c, i in zip(corners, ids.ravel()):
            if str(int(i)) in board['markers']: P3 += board['markers'][str(int(i))]; P2 += c.reshape(4, 2).tolist()
        if len(P3) < 8: print(img.name, f'only {len(P3) // 4} known markers'); continue
        ok, rvec, tvec = cv2.solvePnP(np.float32(P3), np.float32(P2), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        Rm, _ = cv2.Rodrigues(rvec); err = np.linalg.norm(cv2.projectPoints(np.float32(P3), rvec, tvec, K, dist)[0].reshape(-1, 2) - np.float32(P2), axis=1).mean()
        out[img.stem] = dict(R_w2c=Rm.tolist(), t_w2c_mm=tvec.ravel().tolist(), reproj_px=float(err), markers=len(P3) // 4)
        print(img.stem, f'{len(P3) // 4} markers, reprojection {err:.2f} px')
    (S / 'rig.json').write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------- geometry helpers
def blade_cameras(session, sid):
    """Rig cameras expressed in the blade's normalised frame, in the method's camera dict format."""
    S = REAL / session; rigj = json.loads((S / 'rig.json').read_text()); bp = json.loads((S / 'blade_pose.json').read_text())
    Rb, tb = np.array(bp['R']), np.array(bp['t_mm']); mm = mm_per_unit(sid); cams = {}
    for v, c in rigj.items():
        Rw, tw = np.array(c['R_w2c']), np.array(c['t_w2c_mm'])
        # x_board = Rb @ (x_blade * mm) + tb ;  x_cam(mm) = Rw @ x_board + tw  ->  x_cam(units) = (Rw Rb) x_blade + (Rw tb + tw)/mm
        R = Rw @ Rb; t = (Rw @ tb + tw) / mm
        w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
        c2w_cv = np.linalg.inv(w2c); c2w_bl = c2w_cv @ np.diag([1., -1., -1., 1.])
        cams[v] = dict(view_id=v, K=CAM.intrinsics(), w2c_cv=w2c, c2w_blender=c2w_bl)
    return cams


def to_method_view(img, intr):
    """Undistort, then crop around the principal point and resize so focal = FOCAL_PX at 512 px (the method's camera)."""
    K, dist = K_of(intr); und = cv2.undistort(img, K, dist)
    f = (intr['fx'] + intr['fy']) / 2; side = RES * f / F_NOM; cx, cy = intr['cx'], intr['cy']
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2)); s = int(round(side))
    pad = max(0, -x0, -y0, x0 + s - und.shape[1], y0 + s - und.shape[0])
    if pad: und = cv2.copyMakeBorder(und, pad, pad, pad, pad, cv2.BORDER_REPLICATE); x0 += pad; y0 += pad
    return cv2.resize(und[y0:y0 + s, x0:x0 + s], (RES, RES), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- registration of the blade in the board frame
def register(session, specimen, iters=400):
    """Find the blade pose in the board frame so its rendered silhouette matches the healthy photos (all views)."""
    import torch
    from bladedelta.diffrender import coverage
    from bladedelta.geometry import load_healthy
    S = REAL / session; meta = json.loads((S / specimen / 'meta.json').read_text()); sid = meta['source']; mm = mm_per_unit(sid)
    intr = json.loads((S / 'intrinsics.json').read_text()); rigj = json.loads((S / 'rig.json').read_text())
    h = load_healthy(sid); V = torch.tensor(h['vertices'] * mm, dtype=torch.float32, device='cuda'); T = torch.tensor(h['triangles'], dtype=torch.int32, device='cuda')
    masks, Rw, tw = [], [], []
    for v, c in rigj.items():
        im = to_method_view(cv2.imread(str(next((S / specimen).glob(v + '.*')))), intr)
        bg = np.median(np.concatenate([im[:8, :8].reshape(-1, 3), im[-8:, -8:].reshape(-1, 3)]), 0)
        masks.append(torch.tensor((np.abs(im.astype(float) - bg).sum(-1) > 30).astype(np.float32), device='cuda'))
        Rw.append(np.array(c['R_w2c'])); tw.append(np.array(c['t_w2c_mm']))
    M = torch.stack(masks); Kt = torch.tensor(np.stack([CAM.intrinsics()] * len(Rw)), dtype=torch.float32, device='cuda')
    centre = np.mean([-(np.array(R).T @ t) for R, t in zip(Rw, tw)], 0) * 0   # board origin as a start
    best = None
    for yaw in np.radians(np.arange(0, 360, 30)):
        aa = torch.tensor([0., 0., yaw], device='cuda', requires_grad=True); tt = torch.tensor(centre, dtype=torch.float32, device='cuda', requires_grad=True)
        opt = torch.optim.Adam([aa, tt], lr=1e-2)
        for i in range(iters):
            th = aa.norm() + 1e-9; k = aa / th; Kx = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0.]], device='cuda')
            Kmat = torch.stack([torch.stack([0 * th, -k[2], k[1]]), torch.stack([k[2], 0 * th, -k[0]]), torch.stack([-k[1], k[0], 0 * th])])
            Rb = torch.eye(3, device='cuda') + torch.sin(th) * Kmat + (1 - torch.cos(th)) * Kmat @ Kmat
            Xb = V @ Rb.T + tt * 10
            w2c = torch.stack([torch.cat([torch.cat([torch.tensor(R, dtype=torch.float32, device='cuda'), torch.tensor(t, dtype=torch.float32, device='cuda')[:, None]], 1), torch.tensor([[0, 0, 0, 1.]], device='cuda')]) for R, t in zip(Rw, tw)])
            cov = coverage(Xb, T, Kt, w2c); loss = ((cov - M) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if best is None or loss.item() < best[0]: best = (loss.item(), Rb.detach().cpu().numpy(), (tt.detach() * 10).cpu().numpy())
    iou = None
    (S / 'blade_pose.json').write_text(json.dumps(dict(R=best[1].tolist(), t_mm=best[2].tolist(), silhouette_mse=best[0]), indent=1))
    print('registered blade pose, silhouette mse', best[0])


# ---------------------------------------------------------------- prepare split and run
def prepare(session):
    S = REAL / session; intr = json.loads((S / 'intrinsics.json').read_text()); split = 'real_' + session; out = BENCH / split
    if out.exists(): shutil.rmtree(out)
    specs = {p.name: json.loads((p / 'meta.json').read_text()) for p in S.iterdir() if (p / 'meta.json').exists()}
    cases = []
    for name, meta in specs.items():
        if meta['state'] != 'damaged': continue
        sid = meta['source']; cams = blade_cameras(session, sid); healthy = meta.get('pair')
        rd = out / 'renders' / name; rd.mkdir(parents=True, exist_ok=True)
        refd = out / 'refs' / name; refd.mkdir(parents=True, exist_ok=True)
        for v in cams:
            cv2.imwrite(str(rd / f'{v}.png'), to_method_view(cv2.imread(str(next((S / name).glob(v + '.*')))), intr))
            if healthy:
                cv2.imwrite(str(refd / f'{v}.png'), to_method_view(cv2.imread(str(next((S / healthy).glob(v + '.*')))), intr))
        cases.append(dict(case_id=name, source=sid, kind='dent', defect='unknown', views=list(cams), cameras={v: dict(w2c_cv=c['w2c_cv'].tolist()) for v, c in cams.items()},
                          ref=name if healthy else None, truth=meta.get('defects', [])))
    out.mkdir(parents=True, exist_ok=True); (out / 'cases.json').write_text(json.dumps(cases, indent=1)); print(split, len(cases), 'cases')


def run(session, split=None):
    """Frozen detection (v1.1) + specialists + counting with the rig cameras patched in -> output vectors."""
    import torch
    from bladedelta import estimator as E
    from bladedelta.v1_1 import estimate_v11
    from bladedelta.renderer import load_shader
    from run_eval import rgb
    split = split or 'real_' + session; out = BENCH / split; cases = json.loads((out / 'cases.json').read_text())
    T11 = json.loads((ROOT / 'METHOD_FREEZE_V1_1.json').read_text())['thresholds']['v11']['tau_alpha05']
    shader = load_shader(); res = {}; real_camera = E.camera
    for c in cases:
        cams = {v: dict(view_id=v, K=CAM.intrinsics(), w2c_cv=np.array(c['cameras'][v]['w2c_cv'])) for v in c['views']}
        E.camera = lambda v, **kw: cams[v]
        try:
            cfg = E.Config(views=tuple(c['views']))
            imgs = [rgb(out / 'renders' / c['case_id'] / f'{v}.png') for v in c['views']]
            refs = [rgb(out / 'refs' / c['ref'] / f'{v}.png') for v in c['views']] if c['ref'] else None
            r = estimate_v11(c['source'], imgs, shader, cfg, bias=refs)
        finally:
            E.camera = real_camera
        mm = mm_per_unit(c['source']); defective = bool(r['fit_gain'] > T11 and np.abs(r['displacement']).max() > 0)
        vec = [defective, 'dent' if defective else 'none', round(float(np.linalg.norm(r['displacement'], axis=1).max()) * mm, 3) if defective else 0.0, 1 if defective else 0]
        res[c['case_id']] = dict(output_vector=vec, evidence=float(r['fit_gain']), threshold=T11, truth=c.get('truth'))
        print(c['case_id'], vec, 'evidence %.2f' % r['fit_gain'], flush=True)
    (ROOT / 'results' / split).mkdir(parents=True, exist_ok=True); (ROOT / 'results' / split / 'output_vectors.json').write_text(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------- dry run on rendered data
def dryrun(n=6):
    """Synthetic session: rendered cad6dmg/cad6 photos with the nominal rig as 'measured' cameras; healthy reference renders
    play the healthy photos. Exercises prepare -> run end to end without markers (rig.json and blade_pose.json written)."""
    import random
    S = REAL / 'dryrun'
    if S.exists(): shutil.rmtree(S)
    S.mkdir(parents=True)
    intr = dict(fx=F_NOM, fy=F_NOM, cx=RES / 2, cy=RES / 2, dist=[0] * 5, width=RES, height=RES); (S / 'intrinsics.json').write_text(json.dumps(intr))
    rigj = {}
    for v in CAM.FIT_VIEWS:
        c = CAM.camera(v); w2c = c['w2c_cv']; rigj[v] = dict(R_w2c=w2c[:3, :3].tolist(), t_w2c_mm=w2c[:3, 3].tolist())   # board = blade frame, 1 unit = 1 "mm"
    (S / 'rig.json').write_text(json.dumps(rigj))
    C = [c for c in json.loads((BENCH / 'cad6dmg/cases.json').read_text()) if c['source'] == 1003 and c['capture']['condition'] == 'nominal']
    random.seed(0); picks = [c for c in C if c['defect'] in ('dent', 'stain', 'healthy')][:n]
    for c in picks:
        sid = c['source']; mm = mm_per_unit(sid)
        (S / 'blade_pose.json').write_text(json.dumps(dict(R=np.eye(3).tolist(), t_mm=[0, 0, 0])))
        for state, src in (('damaged', BENCH / 'cad6dmg/renders' / c['case_id']), ('healthy', BENCH / 'cad6dmg/refs' / f'ref_{sid:04d}')):
            d = S / f"{c['case_id']}_{state}"; d.mkdir()
            for v in CAM.FIT_VIEWS: shutil.copy(src / f'{v}.png', d / f'{v}.png')
            (d / 'meta.json').write_text(json.dumps(dict(source=sid, state=state, pair=f"{c['case_id']}_healthy" if state == 'damaged' else None, defects=[c['defect']])))
    # rig in "mm" of the board = blade units * mm  -> scale translation
    sid = picks[0]['source']; mm = mm_per_unit(sid)
    for v in rigj: rigj[v]['t_w2c_mm'] = (np.array(rigj[v]['t_w2c_mm']) * mm).tolist()
    (S / 'rig.json').write_text(json.dumps(rigj))
    prepare('dryrun'); res = run('dryrun')
    ok = sum((r['output_vector'][0]) == (r['truth'][0] == 'dent') for r in res.values())
    print(f'dry run: {ok}/{len(res)} defective flags match the known truth')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('args', nargs='*')
    ap.add_argument('--cols', type=int, default=9); ap.add_argument('--rows', type=int, default=6); ap.add_argument('--square_mm', type=float, default=20.)
    a = ap.parse_args()
    if a.cmd == 'calib': calib(a.args[0], a.args[1], a.cols, a.rows, a.square_mm)
    elif a.cmd == 'rig': rig(*a.args)
    elif a.cmd == 'register': register(*a.args)
    elif a.cmd == 'prepare': prepare(*a.args)
    elif a.cmd == 'run': run(*a.args)
    elif a.cmd == 'dryrun': dryrun()
    else: raise SystemExit('unknown command ' + a.cmd)
