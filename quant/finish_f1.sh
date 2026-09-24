#!/bin/sh
# f1's AutoRound output was fine; the first merge refused it because a weight and
# its scale sat in different shards. Redo merge and conversion on the CPU while
# production serves, then evaluate (production stops only for that part).
set -u
W=/home/vmihaylov/qwen3.8_27b_ru
M=/home/vmihaylov/models
R=$W/results/quant_ru/f1
Q="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $W:/w -v $M:/models -v /home/vmihaylov/ninfer:/ninfer"
log() { echo "$(date +%H:%M:%S) $*"; }
log merge
rm -rf $M/ar-f1-merged $M/ar-f1-ninfer; mkdir -p $M/ar-f1-merged $M/ar-f1-ninfer
nice -n 10 $Q qwen-quant:latest python /w/quant/merge_nvfp4.py --base /models/ar-r1-merged --ar /models/ar-f1 \
  --out /models/ar-f1-merged --take fp8 2>&1 | grep -E "replaced|FP8|rows|Error|error" | tee $R/merge.log
[ -s $M/ar-f1-merged/model.safetensors ] || { log "merge failed"; exit 1; }
cp $M/ar-f1-merged/merge_report.json $M/ar-f1/autoround_run.json $R/
log convert
nice -n 10 $Q -w /ninfer qwen-quant:latest python -m tools.convert --model /models/Qwen3.8-27B-bf16 --recipe qwen3_8_27b_nvfp4 \
  --source quantized=/models/ar-f1-merged --source dflash2=/models/z-lab-Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal --name qwen3.8-27b-ar-f1 --device cpu --out /models/ar-f1-ninfer/qwen3_8_27b_nvfp4.ninfer 2>&1 | grep wrote
[ -s $M/ar-f1-ninfer/qwen3_8_27b_nvfp4.ninfer ] || { log "convert failed"; exit 1; }
log "compare with r1"
$Q qwen-quant:latest python /w/quant/cmp_artifacts.py /models/ar-r1-ninfer/qwen3_8_27b_nvfp4.ninfer \
  /models/ar-f1-ninfer/qwen3_8_27b_nvfp4.ninfer 2>&1 | grep -E "objects|prefix" | tee $R/cmp_r1.txt
exec $W/quant/eval_artifact.sh f1 $M/ar-f1-ninfer
