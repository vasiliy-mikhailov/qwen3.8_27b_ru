#!/bin/sh
# Russian importance matrix for Qwen3.8-Flash-Next, computed on UD-Q4_K_XL (fits
# GPU + RAM with a short context; Q8_0 does not). The calibration text is the
# 27B set decoded back to text (flashnext reuses its tokenizer and chat markup).
# Production strata uses the whole GPU, so it is stopped and restarted whatever happens.
#
# Usage: imatrix.sh OUT.gguf [N_CPU_MOE [llama-imatrix args...]]
#   e.g. to continue a run: imatrix.sh imatrix_ru.gguf 0 --in-file /fn/recipes/part.gguf --from-chunk 10
set -u
OUT=$1; NCPU=${2:-14}; shift; [ $# -gt 0 ] && shift
FN=/home/vmihaylov/flash-next
IMG=ghcr.io/ggml-org/llama.cpp:full-cuda
log() { echo "$(date +%H:%M:%S) $*"; }
ls $FN/UD-Q4_K_XL/*.part >/dev/null 2>&1 && { log "UD-Q4_K_XL still downloading"; exit 1; }
docker stop -t 120 strata-flashnext >/dev/null
# shard 2 holds the Engram table; warm it so production answers at full speed at once
trap 'cat $FN/UD-Q3_K_XL/*00002-of-00003.gguf >/dev/null; docker start strata-flashnext >/dev/null; log "prod restarted"' EXIT
trap 'exit 1' INT TERM HUP
log "imatrix on UD-Q4_K_XL, $NCPU expert layers on CPU"
docker run --rm --gpus all --user $(id -u):$(id -g) -v $FN:/fn --entrypoint /app/llama-imatrix $IMG \
  -m /fn/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -f /fn/recipes/calib_ru.txt --parse-special -c 2048 -b 2048 -ub 1024 -ngl 999 --n-cpu-moe $NCPU \
  --no-ppl --output-format gguf -o /fn/recipes/$OUT "$@" 2>&1 | tr '\r' '\n' > $FN/recipes/$OUT.log
grep -E "chunk|entries|partial|skipping|saved|error|failed|offload" $FN/recipes/$OUT.log | tail -12
[ -s $FN/recipes/$OUT ] || { log "imatrix failed"; exit 1; }
log "IMATRIX_DONE $(ls -la $FN/recipes/$OUT)"
