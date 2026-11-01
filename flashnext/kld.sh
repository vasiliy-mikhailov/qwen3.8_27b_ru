#!/bin/sh
# How close each Q3 arm stays to a finer quant on Russian text it was not
# calibrated on: KL divergence of the next-token distribution against
# UD-Q4_K_XL, and how often the top token agrees. The text is the 27B r1
# answers to prompt sets 1-2 (another model's Russian, so no arm is favoured by
# its own greedy output). Production is stopped for the GPU and restarted
# whatever happens.
#
# Usage: kld.sh CHUNKS ARM_DIR...   (each ARM_DIR holds one split GGUF)
set -u
CH=$1; shift
FN=/home/vmihaylov/flash-next; K=/mnt/steam/flash-next/kld
R=/home/vmihaylov/qwen3.8_27b_ru/results/flashnext/kld
IMG=ghcr.io/ggml-org/llama.cpp:full-cuda
log() { echo "$(date +%H:%M:%S) $*"; }
mkdir -p $K $R
docker stop -t 120 strata-flashnext >/dev/null
trap 'cat $FN/UD-Q3_K_XL/*00002-of-00003.gguf >/dev/null; docker start strata-flashnext >/dev/null; log "prod restarted"' EXIT
trap 'exit 1' INT TERM HUP
ppl() {  # MODEL_DIR NCPU OUT extra...
  d=$1; n=$2; o=$3; shift 3
  first=$(cd $d && ls *-00001-of-*.gguf | head -1)
  docker run --rm --gpus all --user $(id -u):$(id -g) -v $FN:/fn:ro -v $d:/m:ro -v $K:/k --entrypoint /app/llama-perplexity $IMG \
    -m /m/$first -f /fn/recipes/heldout_ru.txt -c 2048 -b 2048 -ub 1024 -ngl 999 --n-cpu-moe $n --chunks $CH "$@" \
    > $R/$o.log 2>&1
  grep -E "Final estimate|Mean +KLD|Same top p|Mean +ln|RMS Δp|Maximum KLD|99\.0% +KLD" $R/$o.log | sed "s/^/  /"
}
log "reference logits: UD-Q4_K_XL"
[ -s $K/base_q4.bin ] || ppl $FN/UD-Q4_K_XL 6 ref_q4 --kl-divergence-base /k/base_q4.bin
for d in "$@"; do
  a=$(basename $d)
  log "arm $a"
  ppl $d 0 $a --kl-divergence-base /k/base_q4.bin --kl-divergence
done
log KLD_DONE
