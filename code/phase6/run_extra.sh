#!/bin/bash
cd "$(dirname "$0")/.."
P=../venv_cvpr/bin/python; L=results/phase6_STATUS.txt; mkdir -p results/phase6
for s in folded multi yolotrain; do echo "render $s start $(date)" >> $L; CUDA_VISIBLE_DEVICES="" $P phase6/build_extra.py render $s > results/phase6_render_$s.log 2>&1 && echo "render $s done $(date)" >> $L || echo "render $s FAILED" >> $L; done
CUDA_VISIBLE_DEVICES="" $P phase6/build_extra.py boxes yolotrain >> results/phase6_render_yolotrain.log 2>&1 && echo "boxes done $(date)" >> $L
for s in folded multi; do $P scripts/run_extras.py $s v11 > results/phase6_v11_$s.log 2>&1; echo "v11 $s done $(date)" >> $L; done
$P phase5/specialists2.py folded > results/phase6_spec_folded.log 2>&1; echo "specialists folded done $(date)" >> $L
