#!/bin/sh
# Evaluate one Flash Next GGUF the way the baseline was measured: a strata pack
# built from it, production's serving config with only the pack, the GGUF and
# the log swapped, every prompt set in train/ (greedy, thinking off, 700
# tokens, eight streams) and the tool-call probe. Production is stopped for the
# GPU and restarted whatever happens. Sets already generated are skipped, so a
# rerun only adds the sets that appeared since.
#
# Usage: eval_arm.sh TAG GGUF_DIR [PACK]   (PACK defaults to TAG; built if missing)
set -u
TAG=$1; G=$2; PACK=${3:-$1}
W=/home/vmihaylov/qwen3.8_27b_ru; S=/home/vmihaylov/strata; R=$W/results/flashnext/$TAG
IMG=ghcr.io/niko1221/strata:0.2.9-cuda13
FIRST=$(cd $G && ls *-00001-of-*.gguf | head -1)
log() { echo "$(date +%H:%M:%S) $*"; }
mkdir -p $R

if [ ! -f $S/packs/$PACK/dense.bin ]; then
  log "pack $PACK from $FIRST"
  docker run --rm --network none --user $(id -u):$(id -g) -v $G:/models:ro -v $S:/work -v $S/data:/data -w /opt/strata/terminal \
    -e STRATA_GGUF_PY=/opt/strata/seed/third_party/llama.cpp/gguf-py --entrypoint /opt/strata/terminal/python/bin/python3 $IMG \
    tools/iq_pack.py --gguf /models/$FIRST --out /work/packs/$PACK --compat-bf16 > $R/pack.log 2>&1
  [ -f $S/packs/$PACK/dense.bin ] || { tail -5 $R/pack.log; log "pack failed"; exit 1; }
fi
python3 - $S/strata-flashnext.json $S/eval-$TAG.json $PACK $FIRST $TAG <<'EOF'
import json, sys
src, dst, pack, first, tag = sys.argv[1:]
c = json.load(open(src))
a = c["args"]
a[a.index("--pack") + 1] = f"/work/packs/{pack}"
a[a.index("--native") + 1] = f"/models/{first}"
c["tokenizer"] = f"/work/packs/{pack}/tokenizer"
c["log"] = f"/work/eval-{tag}.log"
json.dump(c, open(dst, "w"), indent=1)
EOF

todo() {
  for set in 1 2 3 4 5 6; do
    P=$W/train/prompts_comments.jsonl; [ $set = 1 ] || P=$W/train/prompts_comments$set.jsonl
    [ -f $P ] || continue
    [ -s $R/gen_set$set.jsonl ] && [ $(wc -l < $R/gen_set$set.jsonl) -ge $(wc -l < $P) ] && continue
    echo $set
  done
}
SETS=$(todo)
[ -z "$SETS" ] && [ -s $R/toolprobe.jsonl ] && { log "$TAG: nothing to do"; exit 0; }

KEY=probe-$(date +%s)
docker stop -t 120 strata-flashnext >/dev/null
trap 'docker rm -f fn-eval >/dev/null 2>&1; cat /home/vmihaylov/flash-next/UD-Q3_K_XL/*00002-of-00003.gguf >/dev/null; docker start strata-flashnext >/dev/null; log "prod restarted"' EXIT
trap 'exit 1' INT TERM HUP
log "serve $TAG"
cat $G/*.gguf >/dev/null
docker run -d --name fn-eval --gpus all --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:8099:8080 -e STRATA_API_KEY=$KEY \
  -v $G:/models:ro -v $S:/work -w /opt/strata/terminal --entrypoint /opt/strata/terminal/python/bin/python3 $IMG \
  -m serve.server --engine strata --config /work/eval-$TAG.json --host 0.0.0.0 --port 8080 >/dev/null
for i in $(seq 1 120); do
  st=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8099/health); [ "$st" = 200 ] && break
  docker ps -q -f name=^fn-eval$ | grep -q . || { log "strata exited"; docker logs --tail 30 fn-eval; exit 1; }
  sleep 10
done
[ "$st" = 200 ] || { log "not ready after 20 min"; docker logs --tail 30 fn-eval; exit 1; }
log "ready"
EXTRA='{"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": false}}'
for set in $SETS; do
  P=$W/train/prompts_comments.jsonl; [ $set = 1 ] || P=$W/train/prompts_comments$set.jsonl
  python3 $W/train/gen_api.py --url http://127.0.0.1:8099 --model qwen-3.8-flash-next --key $KEY --arm $TAG \
    --prompts $P --out $R/gen_set$set.jsonl --workers 8 --extra "$EXTRA" > $R/gen_set$set.log 2>&1
  log "set $set: $(tail -1 $R/gen_set$set.log)"
done
if [ ! -s $R/toolprobe.jsonl ]; then
  python3 $W/quant/toolcall_probe.py --url http://127.0.0.1:8099 --model qwen-3.8-flash-next --key $KEY --arm $TAG \
    --out $R/toolprobe.jsonl > $R/toolprobe.txt 2>&1
  log "tool probe: $(tail -1 $R/toolprobe.txt)"
fi
log ALLDONE
