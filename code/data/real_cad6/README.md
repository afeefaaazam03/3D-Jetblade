# Real blade photos: capture protocol and folder layout

Processed by `phase7/real_ingest.py`. Tested end to end on rendered data (`dryrun`) and on a synthetic ArUco image
(camera pose recovered to 0.07 deg / 0.3 mm). Not yet tested on real photos.

## Capture protocol (per session)
1. **Lens calibration:** 15–25 photos of a printed checkerboard (default 9x6 inner corners, 20 mm squares) at different
   angles, with the same camera, lens, focus and zoom as the inspection photos. Do not change focus or zoom afterwards.
2. **Fixture with ArUco markers:** at least 3 markers (DICT_4X4_50) rigidly fixed around the blade holder, visible in every
   view. Measure each marker's 4 corner positions (mm) in one fixture frame.
3. **Fixed cameras** at the inspection positions, named v004, v005, ... (keep the same camera positions for all blades).
4. **Fixed lights**, and also one low-angle (grazing) light if available. Keep lighting identical between healthy and damaged photos.
5. **For each blade: photograph it HEALTHY first, then damage it and photograph again**, without moving cameras or lights.
   The healthy photos are the reference images (they cancel real-world lighting and material differences).
6. **Ground truth:** for each defect, record type, approximate location, and depth (dial gauge) or a 3D scan.

## Folder layout
```
data/real_cad6/<session>/
  checkerboard/*.jpg                      lens calibration photos
  board.json                              {"dictionary": "DICT_4X4_50", "marker_length_mm": 40,
                                           "markers": {"0": [[x,y,z], [x,y,z], [x,y,z], [x,y,z]], ...}}
                                          corner order per marker: top-left, top-right, bottom-right, bottom-left
                                          as seen on the printed marker (OpenCV convention)
  <blade>_healthy/v004.jpg, v005.jpg ...  + meta.json {"source": 1003, "state": "healthy"}
  <blade>_damaged/v004.jpg, v005.jpg ...  + meta.json {"source": 1003, "state": "damaged", "pair": "<blade>_healthy",
                                                        "defects": [{"type": "dent", "depth_mm": 0.4, "location": "..."}]}
```
`source` is the blade design id (1001–1006 = the six industrial CAD models).

## Commands
```
python phase7/real_ingest.py calib    <session> data/real_cad6/<session>/checkerboard   # -> intrinsics.json
python phase7/real_ingest.py rig      <session> <blade>_healthy                         # -> rig.json (camera poses)
python phase7/real_ingest.py register <session> <blade>_healthy                         # -> blade_pose.json (silhouette fit)
python phase7/real_ingest.py prepare  <session>                                         # -> data/benchmark/real_<session>/
python phase7/real_ingest.py run      <session>                                         # -> results/real_<session>/output_vectors.json
python phase7/real_ingest.py dryrun                                                     # self-test on rendered data
```
`register` (aligning the CAD blade to the fixture from its silhouette) is implemented but untested until real photos exist.
