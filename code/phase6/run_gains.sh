#!/bin/bash
cd "$(dirname "$0")/.."; P=../venv_cvpr/bin/python; L=results/phase6_STATUS.txt
$P phase6/class_gains.py cad6 --blades 1001,1002 --kinds healthy,colour > results/phase6_gains_calib.log 2>&1; echo "gains calib done $(date)" >> $L
$P phase6/class_gains.py cad6dmg > results/phase6_gains_dmg.log 2>&1; echo "gains cad6dmg done $(date)" >> $L
$P phase6/class_gains.py cad6dmg2 > results/phase6_gains_dmg2.log 2>&1; echo "gains cad6dmg2 done $(date)" >> $L
