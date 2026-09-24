#!/bin/sh
# Evaluate one ninfer artifact the way run_full.sh evaluates its own: perplexity
# on the baseline corpus, the 80 comment prompts, the tool-call probe. Serving
# flags match production except the context, which can be cut for artifacts
# whose weights leave no room for 262k tokens of KV (numerics do not depend on it).
# Production is stopped for the GPU and restarted whatever happens.
#
# Usage: eval_artifact.sh TAG ARTIFACT_DIR [MAX_CONTEXT]
set -u
TAG=$1; ART=$2; CTX=${3:-262144}
W=/home/vmihaylov/qwen3.8_27b_ru
R=$W/results/quant_ru/$TAG
U="--user $(id -u):$(id -g) -e HOME=/tmp"
mkdir -p $R/ppl
log() { echo "$(date +%H:%M:%S) $*"; }

docker stop inference-ninfer >/dev/null
trap 'docker rm -f ar-serve >/dev/null 2>&1; docker start inference-ninfer >/dev/null; log "prod restarted"' EXIT
trap 'exit 1' INT TERM HUP

log "perplexity $ART"
docker run --rm --gpus all $U -v $ART:/models:ro -v $W/data/corpora/baseline-corpus:/corpus:ro -v $R/ppl:/out \
  --entrypoint ninfer-perplexity ninfer:ppl /models/qwen3_8_27b_nvfp4.ninfer \
  --corpus /corpus/manifest.json --kv-dtype k8v4 --output /out 2>&1 | tail -3

log "serve (context $CTX)"
KEY=probe-$(date +%s)
docker run -d --name ar-serve --gpus all -p 127.0.0.1:8097:8080 -v $ART:/models:ro ninfer:ppl \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer --host 0.0.0.0 --port 8080 --api-key $KEY \
  --model-id ar --max-context $CTX --kv-capacity $CTX --kv-dtype k8v4 \
  --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 600000 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft --preserve-thinking >/dev/null
for i in $(seq 1 120); do sleep 5; docker logs ar-serve 2>&1 | grep -qi listening && break; done
docker logs ar-serve 2>&1 | grep -E "weights ready|capacity" | tail -2
log "generate"
python3 $W/train/gen_api.py --url http://127.0.0.1:8097 --model ar --key $KEY --arm $TAG \
  --prompts $W/train/prompts_comments.jsonl --out $R/gen.jsonl --workers 1 \
  --extra '{"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": false}}' > $R/gen.log 2>&1
tail -1 $R/gen.log
log "tool probe"
python3 $W/quant/toolcall_probe.py --url http://127.0.0.1:8097 --model ar --key $KEY --arm $TAG \
  --out $R/toolprobe.jsonl | tee $R/toolprobe.txt
log ALLDONE
