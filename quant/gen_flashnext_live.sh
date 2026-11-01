#!/bin/sh
# Baseline for Qwen3.8-Flash-Next: both prompt sets through the live strata
# endpoint, same settings as every 27B arm (greedy, thinking off, 700 tokens),
# two streams so production agents keep most of the engine.
# Usage: gen_flashnext_live.sh ARM
set -u
ARM=$1
W=/home/vmihaylov/qwen3.8_27b_ru
R=$W/results/flashnext/$ARM
K=$(grep -o "=.*" /home/vmihaylov/strata/key.env | head -1 | cut -c2-)
IP=$(docker inspect strata-flashnext -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}")
EXTRA='{"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": false}}'
mkdir -p $R
for set in 1 2; do
  P=$W/train/prompts_comments.jsonl; [ $set = 2 ] && P=$W/train/prompts_comments2.jsonl
  python3 $W/train/gen_api.py --url http://$IP:8080 --model qwen-3.8-flash-next --key "$K" --arm $ARM \
    --prompts $P --out $R/gen_set$set.jsonl --workers 2 --extra "$EXTRA" > $R/gen_set$set.log 2>&1
  echo "set $set: $(tail -1 $R/gen_set$set.log)"
done
echo ALLDONE
