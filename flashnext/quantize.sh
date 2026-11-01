#!/bin/sh
# Rebuild UD-Q3_K_XL's exact per-tensor layout from Q8_0 or BF16 with a given imatrix.
# CPU only: production keeps serving (low CPU weight, so it wins any contention).
# Two runs that differ only in the imatrix (unsloth's vs Russian) isolate what
# the calibration is worth. Afterwards every tensor the imatrix does not steer
# (exact F32 gates that Q8_0 rounded, the embedding lookup, the output head) is
# copied from UD-Q3_K_XL, so an arm differs from production only where the
# imatrix acts.
#
# Usage: quantize.sh TAG IMATRIX_FILE [SOURCE_DIR]   (imatrix under /home/vmihaylov/flash-next;
#        source defaults to the Q8_0 download)
set -u
TAG=$1; IMAT=$2
FN=/home/vmihaylov/flash-next; Q8=${3:-/mnt/steam/flash-next/Q8_0}
IMG=ghcr.io/ggml-org/llama.cpp:full-cuda
FIRST=$(ls $Q8/*-00001-of-*.gguf | head -1)
ls $Q8/*.part >/dev/null 2>&1 && { echo "source still downloading"; exit 1; }
mkdir -p $FN/$TAG
echo "$(date +%H:%M:%S) quantize $TAG from $(basename $FIRST) with $IMAT"
docker run --rm --cpu-shares 128 --user $(id -u):$(id -g) -v $FN:/fn -v $Q8:/q8:ro --entrypoint /app/llama-quantize $IMG \
  --allow-requantize --keep-split --imatrix /fn/$IMAT \
  --tensor-type-file /fn/recipes/ud-q3_k_xl.types.txt \
  /q8/$(basename $FIRST) /fn/$TAG/Qwen3.8-Flash-Next-$TAG.gguf Q3_K_M 20 > $FN/$TAG/quantize.log 2>&1
tail -4 $FN/$TAG/quantize.log
ls $FN/$TAG/*-00001-of-*.gguf >/dev/null 2>&1 || { echo "quantize failed"; exit 1; }
docker run --rm --cpu-shares 128 --user $(id -u):$(id -g) -v $FN:/fn -v /home/vmihaylov/qwen3.8_27b_ru/quant:/q:ro --entrypoint python3 $IMG \
  /q/graft_gguf.py /fn/$TAG/$(cd $FN/$TAG && ls *-00001-of-*.gguf) /fn/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf \
  /fn/recipes/imatrix_unsloth.gguf /fn/$IMAT 2>&1 | tee $FN/$TAG/graft.log
ls -la $FN/$TAG
echo "$(date +%H:%M:%S) QUANT_DONE"
