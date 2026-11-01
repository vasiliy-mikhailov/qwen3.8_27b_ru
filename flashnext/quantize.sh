#!/bin/sh
# Rebuild UD-Q3_K_XL's exact per-tensor layout from Q8_0 with a given imatrix.
# CPU only: production keeps serving (low CPU weight, so it wins any contention).
# Two runs that differ only in the imatrix (unsloth's vs Russian) isolate what
# the calibration is worth.
#
# Usage: quantize.sh TAG IMATRIX_FILE   (file under /home/vmihaylov/flash-next)
set -u
TAG=$1; IMAT=$2
FN=/home/vmihaylov/flash-next; Q8=/mnt/steam/flash-next/Q8_0
IMG=ghcr.io/ggml-org/llama.cpp:full-cuda
FIRST=$(ls $Q8/*-00001-of-*.gguf | head -1)
ls $Q8/*.part >/dev/null 2>&1 && { echo "Q8_0 still downloading"; exit 1; }
mkdir -p $FN/$TAG
echo "$(date +%H:%M:%S) quantize $TAG from $(basename $FIRST) with $IMAT"
docker run --rm --cpu-shares 128 --user $(id -u):$(id -g) -v $FN:/fn -v $Q8:/q8:ro --entrypoint /app/llama-quantize $IMG \
  --allow-requantize --keep-split --imatrix /fn/$IMAT \
  --tensor-type-file /fn/recipes/ud-q3_k_xl.types.txt \
  /q8/$(basename $FIRST) /fn/$TAG/Qwen3.8-Flash-Next-$TAG.gguf Q3_K_M 20 > $FN/$TAG/quantize.log 2>&1
tail -4 $FN/$TAG/quantize.log
ls -la $FN/$TAG
echo "$(date +%H:%M:%S) QUANT_DONE"
