#!/bin/bash
# Phase 4b: frozen v1.1 detection, then phase-5 specialists, on the realistic nick/crack cases.
cd "$(dirname "$0")/.."
P=../venv_cvpr/bin/python
echo "v11 start $(date)" > results/phase4b_STATUS.txt
$P scripts/run_extras.py cad6dmg2 v11 > results/phase4b_v11.log 2>&1
echo "v11 done $(date)" >> results/phase4b_STATUS.txt
$P phase5/specialists.py cad6dmg2 > results/phase5_specialists2.log 2>&1
echo "specialists done $(date)" >> results/phase4b_STATUS.txt
