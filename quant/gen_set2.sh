#!/bin/sh
# Replication: the second prompt set through every artifact, same flags as the
# first set. Production is generated live first; the others need the GPU, so
# production is stopped once for all of them and restarted at the end.
set -u
W=/home/vmihaylov/qwen3.8_27b_ru
M=/home/vmihaylov/models
P=$W/train/prompts_comments2.jsonl
EXTRA='{"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": false}}'
log() { echo "$(date +%H:%M:%S) $*"; }
mkdir -p $W/results/quant_ru/set2

K=$(grep -o "\-\-api-key [^ ]*" /home/vmihaylov/recreate-ninfer.sh | cut -d" " -f2)
IP=$(docker inspect inference-ninfer -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}")
log "prod (live)"
python3 $W/train/gen_api.py --url http://$IP:8080 --model qwen-3.8-27b-nvfp4 --key "$K" --arm nvfp4 \
  --prompts $P --out $W/results/quant_ru/set2/gen_nvfp4.jsonl --workers 1 --extra "$EXTRA" 2>&1 | tail -1

docker stop inference-ninfer >/dev/null
trap 'docker rm -f ar-serve >/dev/null 2>&1; docker start inference-ninfer >/dev/null; log "prod restarted"' EXIT
for spec in ar_r1:ar-r1-ninfer:262144 ar_r2:ar-r2-ninfer:262144 actfix:actfix-ninfer:262144 fp8mlp:fp8mlp-ninfer:32768; do
  arm=${spec%%:*}; rest=${spec#*:}; dir=${rest%%:*}; ctx=${rest#*:}
  log "$arm serve"
  KEY=probe-$(date +%s)
  docker rm -f ar-serve >/dev/null 2>&1
  docker run -d --name ar-serve --gpus all -p 127.0.0.1:8097:8080 -v $M/$dir:/models:ro ninfer:ppl \
    ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer --host 0.0.0.0 --port 8080 --api-key $KEY \
    --model-id ar --max-context $ctx --kv-capacity $ctx --kv-dtype k8v4 \
    --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 600000 \
    --spec dflash2 --draft-tokens 7 --lm-head-draft --preserve-thinking >/dev/null
  for i in $(seq 1 120); do sleep 5; docker logs ar-serve 2>&1 | grep -qi listening && break; done
  python3 $W/train/gen_api.py --url http://127.0.0.1:8097 --model ar --key $KEY --arm $arm \
    --prompts $P --out $W/results/quant_ru/set2/gen_$arm.jsonl --workers 1 --extra "$EXTRA" 2>&1 | tail -1
done
docker rm -f ar-serve >/dev/null 2>&1
log ALLDONE
