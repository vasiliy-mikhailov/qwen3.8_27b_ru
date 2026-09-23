#!/bin/sh
# Production FP4 weights + r2's activation scales ("safe"): what the scales alone buy.
# Built on the CPU as soon as r2's AutoRound output exists, evaluated after fp8mlp.
M=/home/vmihaylov/models
W=/home/vmihaylov/qwen3.8_27b_ru
Q="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $W:/w -v $M:/models -v /home/vmihaylov/ninfer:/ninfer"
until [ -s $M/ar-r2/autoround_run.json ]; do sleep 30; done
sleep 30
mkdir -p $M/actfix-merged $M/actfix-ninfer $W/results/quant_ru/actfix
nice -n 10 $Q qwen-quant:latest python /w/quant/merge_nvfp4.py --base /models/unsloth-Qwen3.8-27B-NVFP4 \
  --ar /models/ar-r2 --out /models/actfix-merged --weights prod --act-scale safe 2>&1 | grep -E "replaced|codes|scale" | tee $W/results/quant_ru/actfix/merge.log
cp $M/actfix-merged/merge_report.json $W/results/quant_ru/actfix/
nice -n 10 $Q -w /ninfer qwen-quant:latest python -m tools.convert --model /models/Qwen3.8-27B-bf16 --recipe qwen3_8_27b_nvfp4 \
  --source quantized=/models/actfix-merged --source dflash2=/models/z-lab-Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal --name qwen3.8-27b-actfix --device cpu --out /models/actfix-ninfer/qwen3_8_27b_nvfp4.ninfer 2>&1 | grep wrote
until grep -q "prod restarted\|convert failed" /tmp/eval_fp8mlp.log 2>/dev/null; do sleep 30; done
sleep 60
exec $W/quant/eval_artifact.sh actfix $M/actfix-ninfer
