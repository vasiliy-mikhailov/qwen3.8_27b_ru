#!/bin/sh
# f1: r1's recipe applied to the FP8 projections. AutoRound tunes the FP8 rows
# (attention, Gated DeltaNet, MLP 56-63) as FP8 in the r1 context (the NVFP4 MLP
# and everything upstream at 4 bits); only those rows are kept and merged into
# r1's store, so f1 differs from r1 in its FP8 part alone.
# A four-layer pilot first checks that FP8 layers are tuned ([fp8] > 0) and that
# the merge accepts their rows (shapes, dtypes, BF16 row scales).
set -u
W=/home/vmihaylov/qwen3.8_27b_ru
M=/home/vmihaylov/models
U="--user $(id -u):$(id -g) -e HOME=/tmp"
docker stop inference-ninfer >/dev/null
trap 'docker start inference-ninfer >/dev/null' EXIT
trap 'exit 1' INT TERM HUP
rm -rf $M/ar-pilotf $M/ar-pilotf-merged
docker run --rm --gpus all --shm-size 16g $U -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $W:/w -v $M:/models qwen-quant:latest \
  python /w/quant/autoround_nvfp4.py --model /models/Qwen3.8-27B-bf16 --calib /w/data/calib/calib_ru.jsonl \
    --out /models/ar-pilotf --truncate 4 --nsamples 8 --iters 5 --context nvfp4 --fp8 2>&1 \
  | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'quantized [0-9]+/[0-9]+ layers|\[mem\] block|\[fp8\]|done in|Error|Traceback' > /tmp/pilotf.log
docker run --rm $U -v $W:/w -v $M:/models qwen-quant:latest \
  python /w/quant/merge_nvfp4.py --base /models/ar-r1-merged --ar /models/ar-pilotf --out /models/ar-pilotf-merged --take fp8 2>&1 \
  | grep -E "replaced|FP8|Error|error|expected|vs new|no place|nothing" >> /tmp/pilotf.log
cat /tmp/pilotf.log
if grep -q "done in" /tmp/pilotf.log && grep -q "\[fp8\] [1-9]" /tmp/pilotf.log && grep -q "and 13 FP8 matrices" /tmp/pilotf.log \
   && ! grep -qi "error\|no place\|vs new" /tmp/pilotf.log; then
  echo "PILOT_OK"
  if [ -n "${PILOT_ONLY:-}" ]; then echo "prod restarted"; exit 0; fi
  trap - EXIT
  # r1's recipe in full, so f1 differs from it only in what is tuned
  MERGE_BASE=/models/ar-r1-merged MERGE_TAKE=fp8 exec $W/quant/run_full.sh f1 --context nvfp4 --fp8 \
    --iters 100 --batch 2 --grad-acc 4 --nsamples 256 "$@"
fi
echo "PILOT_FAILED"
echo "prod restarted"
