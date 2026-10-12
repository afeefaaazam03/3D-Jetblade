#!/bin/bash
# Push the Phase 2 page to GitHub Pages at every checkpoint (new phase2 step finished, or a new map). New script.
cd "$(dirname "$0")/.." || exit 1
P=../venv_cvpr/bin/python; G=/home/afeefaazam/gitwork/bladedelta-site; L=$PWD/results/phase2/PUBLISH.txt
snap() { { ls results/phase2/state results/phase2/maps 2>/dev/null; wc -l results/phase2/*.jsonl 2>/dev/null | tail -1; } | md5sum; }
push() {
  CUDA_VISIBLE_DEVICES="" $P phase2/build_page.py >> results/phase2/log_publish.txt 2>&1 || { echo "$(date -u +%FT%TZ) build FAILED" >> $L; return; }
  mkdir -p $G/img/phase2 $G/code/phase2; cp site/phase2.html $G/; cp site/img/phase2/* $G/img/phase2/ 2>/dev/null
  cp phase2/*.py phase2/*.sh $G/code/phase2/ 2>/dev/null
  (cd $G && git add -A && if ! git diff --cached --quiet; then
     git -c user.name=afeefaaazam03 -c user.email=afeefa.aazam@gmail.com commit -qm "Phase 2 checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin main && echo "$(date -u +%FT%TZ) pushed: $1" >> $L
   fi)
}
S=""
while true; do
  N=$(snap)
  if [ "$N" != "$S" ]; then S=$N; push "$(tail -n 1 results/phase2/STATUS.txt 2>/dev/null | cut -c22-)"; fi
  grep -q PHASE2_DONE results/phase2/STATUS.txt 2>/dev/null && [ -f results/phase2/FINAL_PUSHED ] && exit 0
  grep -q PHASE2_DONE results/phase2/STATUS.txt 2>/dev/null && touch results/phase2/FINAL_PUSHED
  sleep 120
done
