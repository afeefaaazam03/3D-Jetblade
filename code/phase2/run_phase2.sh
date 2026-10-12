#!/bin/bash
# Phase 2 chain (new; existing chains and code untouched). GPU stages share the global lock.
cd "$(dirname "$0")/.." || exit 1
P=../venv_cvpr/bin/python; ST=results/phase2/state; mkdir -p $ST; STATUS=$PWD/results/phase2/STATUS.txt
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $STATUS; }
stage() {
  local name=$1; shift
  [ -f $ST/$name.done ] && return 0
  for attempt in 1 2 3; do
    log "START $name (try $attempt)"
    flock /tmp/bladedelta_gpu.lock "$@" >> results/phase2/log_$name.txt 2>&1 && { touch $ST/$name.done; log "DONE  $name"; return 0; }
    log "FAIL  $name (try $attempt)"; sleep 60
  done
}
log "phase2 started"
stage detect_test   $P phase2/detectability.py test
stage detect_test2  $P phase2/detectability.py test2
[ -f data/benchmark/calib2/cases.json ] || $P phase2/build_calib2.py cases >> results/phase2/log_calib2.txt 2>&1
stage calib2_render $P phase2/build_calib2.py render
stage calib2_v1     $P scripts/run_eval.py calib2 ours
stage calib2_v11    $P scripts/run_extras.py calib2 v11
log "PHASE2_DONE"
