#!/bin/sh
# Russian-calibrated NVFP4, end to end, in one production outage:
# AutoRound on the MLP of layers 0-55 -> merge into the production store ->
# ninfer artifact -> perplexity, the 80 comment prompts, the tool-call probe,
# the second (replication) set of 80 prompts.
# Production is stopped for the GPU and restarted whatever happens.
#
# Usage: run_full.sh TAG [autoround_nvfp4.py options...]
# MERGE_BASE (default: the unsloth store) and MERGE_TAKE (nvfp4|fp8|both, default
# nvfp4) choose what the new tensors are merged into; f1 merges its FP8 rows into
# r1's store and keeps r1's MLP.
set -u
TAG=$1; shift
W=/home/vmihaylov/qwen3.8_27b_ru
M=/home/vmihaylov/models
R=$W/results/quant_ru/$TAG
AR=$M/ar-$TAG; MERGED=$M/ar-$TAG-merged; ART=$M/ar-$TAG-ninfer
U="--user $(id -u):$(id -g) -e HOME=/tmp"
QUIET='^==|CUDA Version|NVIDIA Corporation|Container image|license|^\s*$|Skipping import|NGC-DL|By pulling|A copy of|GPU functionality|docs.nvidia|Use the NVIDIA'
mkdir -p $R $AR $MERGED $ART
log() { echo "$(date +%H:%M:%S) $*"; }

docker stop inference-ninfer >/dev/null
trap 'docker rm -f ar-serve >/dev/null 2>&1; docker start inference-ninfer >/dev/null; log "prod restarted"' EXIT
trap 'exit 1' INT TERM HUP

log "autoround $*"
docker run --rm --gpus all --shm-size 16g $U -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $W:/w -v $M:/models qwen-quant:latest \
  python /w/quant/autoround_nvfp4.py --model /models/Qwen3.8-27B-bf16 \
    --calib /w/data/calib/calib_ru.jsonl --out /models/ar-$TAG "$@" 2>&1 \
  | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*m//g' \
  | grep -E --line-buffered '\[mem\] block|\[fp8\]|quantized [0-9]+/[0-9]+ layers|Applying AutoRound|done in|Error|Traceback|error:' > $R/autoround.log
grep -q "done in" $R/autoround.log || { log "autoround failed"; tail -5 $R/autoround.log; exit 1; }
log "$(tail -1 $R/autoround.log)"

log "merge"
docker run --rm $U -v $W:/w -v $M:/models qwen-quant:latest \
  python /w/quant/merge_nvfp4.py --base ${MERGE_BASE:-/models/unsloth-Qwen3.8-27B-NVFP4} --ar /models/ar-$TAG \
    --out /models/ar-$TAG-merged --take ${MERGE_TAKE:-nvfp4} 2>&1 \
  | grep -Ev "$QUIET" | tee $R/merge.log
[ -s $MERGED/model.safetensors ] || { log "merge failed"; exit 1; }
cp $MERGED/merge_report.json $AR/autoround_run.json $R/

log "convert"
docker run --rm --gpus all $U -v $M:/models -v /home/vmihaylov/ninfer:/ninfer -w /ninfer qwen-quant:latest \
  python -m tools.convert --model /models/Qwen3.8-27B-bf16 --recipe qwen3_8_27b_nvfp4 \
    --source quantized=/models/ar-$TAG-merged --source dflash2=/models/z-lab-Qwen3.8-27B-DFlash2 \
    --components text,vision,mtp,dflash2 --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
    --proposal --name qwen3.8-27b-ar-$TAG --out /models/ar-$TAG-ninfer/qwen3_8_27b_nvfp4.ninfer 2>&1 | grep wrote
[ -s $ART/qwen3_8_27b_nvfp4.ninfer ] || { log "convert failed"; exit 1; }

log "perplexity"
mkdir -p $R/ppl
docker run --rm --gpus all $U -v $ART:/models:ro -v $W/data/corpora/baseline-corpus:/corpus:ro -v $R/ppl:/out \
  --entrypoint ninfer-perplexity ninfer:ppl /models/qwen3_8_27b_nvfp4.ninfer \
  --corpus /corpus/manifest.json --kv-dtype k8v4 --output /out 2>&1 | tail -3

log "serve"
KEY=probe-$(date +%s)
docker run -d --name ar-serve --gpus all -p 127.0.0.1:8097:8080 -v $ART:/models:ro ninfer:ppl \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer --host 0.0.0.0 --port 8080 --api-key $KEY \
  --model-id ar --max-context 262144 --kv-capacity 262144 --kv-dtype k8v4 \
  --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 600000 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft --preserve-thinking >/dev/null
for i in $(seq 1 120); do sleep 5; docker logs ar-serve 2>&1 | grep -qi listening && break; done
log "generate"
python3 $W/train/gen_api.py --url http://127.0.0.1:8097 --model ar --key $KEY --arm ar_$TAG \
  --prompts $W/train/prompts_comments.jsonl --out $R/gen_ar.jsonl --workers 1 \
  --extra '{"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": false}}' > $R/gen.log 2>&1
tail -1 $R/gen.log
log "tool probe"
python3 $W/quant/toolcall_probe.py --url http://127.0.0.1:8097 --model ar --key $KEY --arm ar_$TAG \
  --out $R/toolprobe.jsonl | tee $R/toolprobe.txt
log "generate set 2"
mkdir -p $W/results/quant_ru/set2
python3 $W/train/gen_api.py --url http://127.0.0.1:8097 --model ar --key $KEY --arm ar_$TAG \
  --prompts $W/train/prompts_comments2.jsonl --out $W/results/quant_ru/set2/gen_ar_$TAG.jsonl --workers 1 \
  --extra '{"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": false}}' 2>&1 | tail -1
log ALLDONE
