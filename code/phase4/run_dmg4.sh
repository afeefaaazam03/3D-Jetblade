#!/bin/bash
# Phase 4: render the four-defect cases, then run frozen v1.1 (2 views) on them.
cd "$(dirname "$0")/.."
P=../venv_cvpr/bin/python
echo "render start $(date)" > results/phase4_STATUS.txt
CUDA_VISIBLE_DEVICES="" $P phase4/build_dmg4.py render > results/phase4_render.log 2>&1 || { echo "render FAILED" >> results/phase4_STATUS.txt; exit 1; }
echo "render done $(date)" >> results/phase4_STATUS.txt
$P scripts/run_extras.py cad6dmg v11 > results/phase4_v11.log 2>&1
echo "v11 done $(date)" >> results/phase4_STATUS.txt
