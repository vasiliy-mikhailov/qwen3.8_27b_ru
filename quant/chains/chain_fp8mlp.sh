#!/bin/sh
# After r2 has finished and the FP8-MLP artifact is written, evaluate it.
until grep -q "prod restarted" /tmp/run_full_r2.log 2>/dev/null; do sleep 30; done
until [ -s /home/vmihaylov/models/fp8mlp-ninfer/qwen3_8_27b_nvfp4.ninfer ] && grep -q "^wrote" /tmp/convert_fp8mlp.log; do
  grep -qi "error\|Traceback" /tmp/convert_fp8mlp.log && { echo "convert failed"; exit 1; }
  sleep 30
done
sleep 60
exec /home/vmihaylov/qwen3.8_27b_ru/quant/eval_artifact.sh fp8mlp /home/vmihaylov/models/fp8mlp-ninfer 32768
