"""Phase 6, task 5 (new code): train YOLOv8n on data/yolo, evaluate mAP on the held-out test images and time inference.
Also reports case-level numbers comparable to ours: a case counts as 'damage reported' if either view has a
dent/nick/crack box above the confidence threshold."""
import json
import time
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]; Y = ROOT / 'data/yolo'; OUT = ROOT / 'results/phase6'; OUT.mkdir(parents=True, exist_ok=True)
model = YOLO('yolov8n.pt')
model.train(data=str(Y / 'blade.yaml'), epochs=80, imgsz=512, batch=16, project=str(ROOT / 'runs_yolo'), name='blade', exist_ok=True, verbose=False, plots=False, seed=0)
best = YOLO(str(ROOT / 'runs_yolo/blade/weights/best.pt'))
m = best.val(data=str(Y / 'blade.yaml'), split='test', imgsz=512, verbose=False, plots=False)
names = ['dent', 'nick', 'crack', 'stain']
res = dict(mAP50=float(m.box.map50), mAP50_95=float(m.box.map), per_class_mAP50={names[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.ap50)},
           speed_ms_per_image=dict(m.speed))
imgs = sorted((Y / 'images/test').glob('*.png')); best.predict(str(imgs[0]), imgsz=512, verbose=False)
t0 = time.time(); preds = best.predict([str(p) for p in imgs], imgsz=512, verbose=False, conf=.25); dt = (time.time() - t0) / len(imgs)
meta = json.loads((Y / 'test_meta.json').read_text()); case = {}
for p, r in zip(imgs, preds):
    mt = meta[p.stem]; dmg = any(int(c) in (0, 1, 2) for c in r.boxes.cls.tolist())
    k = mt['case_id']; case.setdefault(k, dict(defect=mt['defect'], reported=False)); case[k]['reported'] |= dmg
agg = {}
for v in case.values(): a = agg.setdefault(v['defect'], [0, 0]); a[0] += 1; a[1] += v['reported']
res['end_to_end_ms_per_image'] = dt * 1000; res['case_level_reported_conf0.25'] = {k: f'{b}/{a}' for k, (a, b) in agg.items()}
(OUT / 'yolo_results.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
