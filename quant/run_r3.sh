#!/bin/sh
# r3: MLP rounding tuned in a context coarser than r1: attention and the other
# non-target Linears simulated as MXFP4 (see --context in autoround_nvfp4.py).
# A four-layer pilot first checks that the MXFP4 context is applied: AutoRound
# must report every Linear of a targeted block as quantized (8/8), not only the
# MLP, and skip blocks without targets (0/8).
set -u
W=/home/vmihaylov/qwen3.8_27b_ru
M=/home/vmihaylov/models
docker stop inference-ninfer >/dev/null
rm -rf $M/ar-pilot3
docker run --rm --gpus all --shm-size 16g --user $(id -u):$(id -g) -e HOME=/tmp \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True -v $W:/w -v $M:/models qwen-quant:latest \
  python /w/quant/autoround_nvfp4.py --model /models/Qwen3.8-27B-bf16 --calib /w/data/calib/calib_ru.jsonl \
    --out /models/ar-pilot3 --truncate 4 --layers 1-2 --nsamples 8 --iters 5 --context mxfp4 2>&1 \
  | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'quantized [0-9]+/[0-9]+ layers|\[mem\] block|\[ctx\]|done in|Error|Traceback' > /tmp/pilot3.log
cat /tmp/pilot3.log
if grep -q "done in" /tmp/pilot3.log && grep -q "quantized 8/8" /tmp/pilot3.log && grep -q "quantized 0/8" /tmp/pilot3.log && grep -q "\[ctx\] [1-9]" /tmp/pilot3.log; then
  echo "PILOT_OK"
  exec $W/quant/run_full.sh r3 --context mxfp4 "$@"
fi
echo "PILOT_FAILED"
docker start inference-ninfer >/dev/null
echo "prod restarted"
