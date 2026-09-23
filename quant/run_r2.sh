#!/bin/sh
# r2: the same pipeline as r1, with non-target Linears kept out of AutoRound.
# A four-layer pilot first checks that blocks without any NVFP4 target (0 and 3
# here, 56-63 in the real run) pass through, and that AutoRound now reports
# three quantized layers per targeted block instead of all of them.
set -u
W=/home/vmihaylov/qwen3.8_27b_ru
M=/home/vmihaylov/models
docker stop inference-ninfer >/dev/null
rm -rf $M/ar-pilot2
docker run --rm --gpus all --shm-size 16g --user $(id -u):$(id -g) -e HOME=/tmp \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True -v $W:/w -v $M:/models qwen-quant:latest \
  python /w/quant/autoround_nvfp4.py --model /models/Qwen3.8-27B-bf16 --calib /w/data/calib/calib_ru.jsonl \
    --out /models/ar-pilot2 --truncate 4 --layers 1-2 --nsamples 8 --iters 5 2>&1 \
  | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'quantized [0-9]+/[0-9]+ layers|\[mem\] block|done in|Error|Traceback' > /tmp/pilot2.log
cat /tmp/pilot2.log
if grep -q "done in" /tmp/pilot2.log && ! grep -q "quantized [4-9]/" /tmp/pilot2.log; then
  echo "PILOT_OK"
  exec $W/quant/run_full.sh r2 "$@"
fi
echo "PILOT_FAILED"
docker start inference-ninfer >/dev/null
echo "prod restarted"
