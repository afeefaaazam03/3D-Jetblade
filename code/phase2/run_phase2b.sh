#!/bin/bash
# Phase 2b (new): maps at benchmark dent sizes, then H0 statistics for the chi-square and a-contrario baselines.
cd "$(dirname "$0")/.." || exit 1
P=../venv_cvpr/bin/python; ST=results/phase2/state; STATUS=$PWD/results/phase2/STATUS.txt
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $STATUS; }
stage() { local name=$1; shift; [ -f $ST/$name.done ] && return 0
  for a in 1 2 3; do log "START $name (try $a)"; flock /tmp/bladedelta_gpu.lock "$@" >> results/phase2/log_$name.txt 2>&1 && { touch $ST/$name.done; log "DONE  $name"; return 0; }; log "FAIL  $name"; sleep 60; done; }
log "phase2b started"
for r in 0.04 0.06; do for c in test_0037_0 test_0042_0; do stage map_${c}_r$r $P phase2/detect_map.py test $c --radius $r --step 0.03; done; done
stage h0_calib $P phase2/h0_residuals.py calib
stage h0_test  $P phase2/h0_residuals.py test
log "PHASE2B_DONE"
