#!/bin/sh
# Does the r1 compressed-tensors store (unsloth layout, r1's NVFP4 MLP) run in vLLM?
# Serves it with vLLM, generates the first prompt set and runs the tool-call probe
# through vLLM's own tool parser. Production is stopped for the GPU and restarted
# whatever happens.
#
# Usage: vllm_smoke.sh TAG STORE_DIR   (e.g. vllm_r1 /home/vmihaylov/models/ar-r1-merged)
set -u
TAG=$1; STORE=$2
W=/home/vmihaylov/qwen3.8_27b_ru
R=$W/results/quant_ru/$TAG
IMAGE=vllm/vllm-openai:v0.27.1
mkdir -p $R
log() { echo "$(date +%H:%M:%S) $*"; }
docker stop inference-ninfer >/dev/null
trap 'docker logs vllm-smoke > $R/serve_full.log 2>&1; docker rm -f vllm-smoke >/dev/null 2>&1; docker start inference-ninfer >/dev/null; log "prod restarted"' EXIT
trap 'exit 1' INT TERM HUP
log "serve $STORE with $IMAGE"
# 32 GB leaves little room next to 20 GiB of weights: text only (no vision encoder
# budget), 16k context, headroom for the allocator, and few sequences -- each one
# carries Gated DeltaNet recurrent state for 48 layers (~150 MB), and vLLM's default
# of 256 sequences reserves far more than the card has.
docker run -d --name vllm-smoke --gpus all --ipc=host -p 127.0.0.1:8096:8000 -v $STORE:/model:ro \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $IMAGE \
  --model /model --served-model-name r1 --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 8 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 >/dev/null
for i in $(seq 1 180); do
  sleep 5
  curl -sf http://127.0.0.1:8096/health >/dev/null && break
  docker ps --format '{{.Names}}' | grep -q vllm-smoke || break
done
docker logs vllm-smoke 2>&1 | grep -iE "quantiz|compressed|nvfp4|fp4|Loading weights took|model weights took|KV cache|maximum concurrency|Error|error" | tail -12 > $R/serve.log
cat $R/serve.log
curl -sf http://127.0.0.1:8096/health >/dev/null || { log "server did not come up"; docker logs --tail 40 vllm-smoke > $R/serve_fail.log 2>&1; exit 1; }
log generate
python3 $W/train/gen_api.py --url http://127.0.0.1:8096 --model r1 --arm $TAG \
  --prompts $W/train/prompts_comments.jsonl --out $R/gen.jsonl --workers 8 \
  --extra '{"chat_template_kwargs": {"enable_thinking": false}}' > $R/gen.log 2>&1
tail -1 $R/gen.log
log "tool probe"
python3 $W/quant/toolcall_probe.py --url http://127.0.0.1:8096 --model r1 --arm $TAG \
  --extra '{"chat_template_kwargs": {"enable_thinking": false}}' --out $R/toolprobe.jsonl | tee $R/toolprobe.txt
log ALLDONE
