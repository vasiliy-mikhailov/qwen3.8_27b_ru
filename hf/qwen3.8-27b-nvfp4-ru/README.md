---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
language:
  - ru
  - en
base_model:
  - Qwen/Qwen3.8-27B
  - unsloth/Qwen3.8-27B-NVFP4
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - nvfp4
  - fp8
  - w4a4
  - autoround
  - russian
  - blackwell
  - rtx-5090
---

# Qwen3.8-27B NVFP4 for NInfer, re-rounded for Russian

A replacement for
[neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)
that makes fewer Russian language errors. Same file size, same format and layouts, so the same
kernels and memory: only the rounding of the 4-bit MLP weights and their activation scales
differ, plus the public model name stored in the file (see *Requirements and use*).

On 160 prompts of Russian technical writing (code comments, docstrings, code reviews,
specifications, READMEs, explanations of failing tests), blind annotation by LLM agents counted **0.70 hard
language errors per page against 1.03 for the official artifact (−32 %; 85 against 130 hard errors
on the same prompts, sign test p = 0.004)**.

The work, the measurements and every script are in
[github.com/vasiliy-mikhailov/qwen3.8_27b_ru](https://github.com/vasiliy-mikhailov/qwen3.8_27b_ru)
(documentation in Russian, see `QUANT-RU.md`).

## What was changed

The official artifact is [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
converted with NInfer's `qwen3_8_27b_nvfp4` recipe (we reproduced it byte for byte, all 1,246 objects).
In its text model, the MLP of layers 0–55 (168 matrices) is NVFP4, quantized by unsloth (the
checkpoint's config records static activation ordering, which suggests GPTQ); the rest of the text
model is FP8 or BF16 and loses little. Vision, MTP and DFlash2 weights use grouped integer formats
and are not touched here.

Only those 168 matrices were redone:

- **Weights.** [AutoRound](https://github.com/intel/auto-round) (through
  [llm-compressor](https://github.com/vllm-project/llm-compressor)) learned the rounding and clipping
  of each decoder block against the BF16 block on a Russian calibration set.
- **Activation scales.** Each NVFP4 input has one static global scale; values above the calibrated
  maximum are clipped by the kernel. The Russian calibration saw larger maxima than the original
  one for 104 of the 168 matrices (65 of 112 distinct inputs, since gate and up share one; up to
  4.9×), and there the larger maximum is used; elsewhere the original scale is kept.

Compared with the official file, 216 of 1,246 objects differ: the 112 NVFP4 weight objects and
104 activation-scale objects. All FP8, BF16, vision, MTP and DFlash2 objects are identical.

### Calibration

256 sequences of 2,048 tokens: Russian Stack Overflow questions and answers in the chat template
(96), whole source files with Russian comments (56), agent turns that read a file and write it back
with Russian comments through tool calls (48), Russian comment blocks (24), plain code without
Cyrillic (32). Almost no English prose, on purpose: only short fixed reasoning lines and tool
descriptions inside the agent turns. None of it overlaps the evaluation prompts.

### Tuning context

AutoRound tunes a whole decoder block at a time. In this run every Linear of the block —
attention and Gated DeltaNet projections as well as the MLP — was simulated as W4A4 NVFP4 during
tuning, and each block's quantized output fed the next one. Only the MLP tensors were kept.
The MLP rounding was therefore fitted in a noisier context than it meets at inference, where
attention runs in FP8. This was an accident. A second run tuned in a clean context (attention
in BF16) was not significantly better than the official artifact (below), while this one was on
both prompt sets; the direct difference between the two runs (−20 %, p = 0.33) is not significant
on its own. One possible reading is robustness: quantization error accumulates from layer to
layer at inference, and an MLP fitted to cancel more noise than it meets may cope better. A run
with an even coarser tuning context is under way to test it.

Settings: 100 iterations per block, effective batch 8 (2 × 4 accumulation), default learning
rate, sequential per block, 2 h 12 min on one RTX 5090.

## Evaluation

All numbers compare artifacts on the same engine, NInfer. Generation and the tool probe used
`ninfer-serve` with the production flags below (DFlash2 K=7, K8V4 KV cache; the all-FP8 row with a
32k context, since its weights leave no room for 262k), greedy decoding and thinking off, with up
to 700 new tokens for the language sets and 2,000 for the tool probe. Perplexity used
`ninfer-perplexity` (K8V4, 4,096-token windows, 2,048 stride).

### Russian language errors in generation

Two sets of 80 prompts (8 task types × 10), the second written afterwards to replicate the first.
Every answer was annotated blind — the annotators saw one answer at a time with no indication of
which model wrote it — by two independent annotators (grammar and lexicon lenses), and every
candidate error was then judged by two independent skeptics (a normative editor and a developer);
an error counts only when both confirm it. **Annotators, skeptics and the pairwise judges below
are LLM agents (Claude), not human linguists.** Hard errors are agreement, government, non-existent or wrong words,
spelling, foreign characters and nonsense; calques and punctuation are counted separately. A page
is 250 Russian words.

| artifact | set 1 | set 2 | both sets (160 prompts) | per page vs official | hard errors, sign test |
|---|---|---|---|---|---|
| official NVFP4 | 1.24 | 0.83 | 1.03 | — | 130 |
| **this artifact** | **0.73** | **0.68** | **0.70** | **−32 %** | **85, p = 0.004** |
| all MLP in FP8 (does not fit 262k context) | 1.02 | 0.65 | 0.83 | −20 % | 103, p = 0.040 |
| AutoRound, clean tuning context | 0.96 | 0.78 | 0.87 | −16 % | 106, p = 0.24 |
| official weights, new activation scales only | 1.01 | 0.78 | 0.90 | −13 % | 111, p = 0.22 |

The sign test is over prompts: this artifact had fewer hard errors than the official one on 62
prompts and more on 33 (95 % bootstrap interval of the change in hard-error count −49 % … −17 %).
In a separate blind pairwise comparison on set 1 (three position-balanced LLM judges per pair),
it was preferred in 50 of 80 pairs (62 %, p = 0.033).

Caveats. Greedy decoding turns any change of rounding into a different text within the first few
words, so each set of answers is one sample; that is why the second set exists. Re-annotating the
same official answers three times gave 70, 88 and 77 hard errors, so only comparisons within one
annotation batch are meaningful, and all rows of the table come from shared batches.

### Perplexity

2.2 M tokens, same streams for every artifact; difference in mean negative log-likelihood from the
official artifact, in nats (lower is better):

| | Russian code register | Russian reference | code | English reference | English long-form | Chinese | overall |
|---|---|---|---|---|---|---|---|
| this artifact | +0.0018 | −0.0013 | +0.0018 | +0.0224 | −0.0016 | −0.0050 | +0.0024 |
| all MLP in FP8 | −0.0060 | −0.0090 | −0.0047 | −0.0229 | −0.0092 | −0.0174 | −0.0102 |

Perplexity barely moves: even eight bits in every MLP buys only 0.01 nats, and at this scale it
does not track generation errors. English reference text (Wikipedia) is 0.022 nats worse, while
English long-form prose is unchanged. The calibration alone does not explain it: the
clean-context run used the same calibration and improved English reference by 0.019.

### Tool calls

A probe of 40 agent turns, all expecting a structured tool call: read a file to comment it;
write the file back with Russian comments as one long argument; run the tests; react to a failing
test. A turn is clean when the server returns structured `tool_calls` with valid JSON arguments,
every required parameter and an expected tool. 40 of 40 clean, as for the official artifact. The
probe checks the call format, not whether the fix is right.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_nvfp4.ninfer` |
| Size | 23,719,715,844 bytes (22.09 GiB) |
| SHA-256 | `bde2d964b7d2d59f809af42dfa5f0e2349adc522cafaf84e4ab67cbb11982b3b` |
| Container version | 3 |
| Architecture | `Qwen3_5ForCausalLM` |
| Public model name | `qwen3.8-27b-ar-r1` |
| Components | Text, Vision, MTP, DFlash2, proposal head |
| Stored objects | 1,246 (1,240 tensors, 6 resources) |

```bash
printf '%s  %s\n' \
  'bde2d964b7d2d59f809af42dfa5f0e2349adc522cafaf84e4ab67cbb11982b3b' \
  'qwen3_8_27b_nvfp4.ninfer' | sha256sum --check
```

## Requirements and use

As for the official artifact: [NInfer](https://github.com/Neroued/ninfer) built from
source (tested at revision
[`9e163ee`](https://github.com/Neroued/ninfer/commit/9e163eee4b8acec21ab0ac765107b6a3f287b217)),
64-bit Linux, NVIDIA GeForce RTX 5090 (`sm_120a`), CUDA 13.1 or newer.

The server configuration used for the generation and tool measurements above:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype k8v4 \
  --max-concurrency 1 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft \
  --preserve-thinking \
  --model-id qwen3.8-27b
```

The public model name stored in this file is `qwen3.8-27b-ar-r1`, and `ninfer-serve` uses it as
the model id unless `--model-id` is given. Clients configured for the official artifact expect
`qwen3.8-27b`; pass `--model-id qwen3.8-27b` (as above) to keep them working unchanged. See the
[official model card](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) and the NInfer
documentation for the CLI, multimodal input and other serving options.

## How it was built

```bash
# 0. calibration set: scripts/build_register_corpus.py downloads and filters the sources
#    (ru.stackoverflow, GitHub code), quant/build_calib.py packs them; third-party text is not
#    redistributed, so it has to be rebuilt
python quant/build_calib.py Qwen3.8-27B data calib_ru.jsonl
# 1. AutoRound on the MLP of layers 0-55, noisy context (r1 was run before the --context flag
#    existed; nvfp4 is the behaviour it had)
python quant/autoround_nvfp4.py --model Qwen3.8-27B --calib calib_ru.jsonl --out ar-r1 \
  --context nvfp4 --iters 100 --batch 2 --grad-acc 4 --nsamples 256
# 2. put its NVFP4 tensors into the unsloth store; keep the larger of the two activation maxima
python quant/merge_nvfp4.py --base unsloth-Qwen3.8-27B-NVFP4 --ar ar-r1 --out ar-r1-merged --act-scale safe
# 3. NInfer's official recipe, as for the official artifact
python -m tools.convert --model Qwen3.8-27B --recipe qwen3_8_27b_nvfp4 \
  --source quantized=ar-r1-merged --source dflash2=Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal --name qwen3.8-27b-ar-r1 --out qwen3_8_27b_nvfp4.ninfer
```

AutoRound samples batches at random, so a rerun gives a statistically equivalent artifact, not a
byte-identical one; the published file is the one every number above was measured on.

Sources: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at `1d4bf0f2`,
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) at `f0b7c9e7`,
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) at `50307d4c`.
`artifact-manifest.json` records them in full.

## License

Apache-2.0, like the weights and code it is built from: the Qwen3.8-27B weights, the unsloth NVFP4
checkpoint, the z-lab DFlash2 drafter and NInfer. This artifact modifies the NVFP4 MLP weights and
activation scales of the unsloth checkpoint as described above. The calibration set (Russian
Stack Overflow content under CC BY-SA 4.0 and public GitHub code) was used only to measure
activations during tuning; none of its text is contained in the artifact.

## Кратко по-русски

Замена официального NVFP4-артефакта ninfer для Qwen3.8-27B того же размера и скорости, которая
реже ошибается в русском. Переокруглены только 4-битные MLP слоёв 0–55: AutoRound на русской
калибровке, подгонка в «шумном» контексте. На 160 заданиях по русскому техническому тексту
слепая разметка насчитала 0,70 твёрдой ошибки на страницу против 1,03 у официального
(−32 %; 85 ошибок против 130 на тех же заданиях, p = 0,004). Размечали LLM-агенты (Claude),
не люди. Вызовы инструментов — 40 из 40. Английский справочный текст чуть хуже (+0,022 ната
перплексии). В файле записано имя модели `qwen3.8-27b-ar-r1`: чтобы клиенты не заметили
замены, запускайте сервер с `--model-id qwen3.8-27b`. Подробности и скрипты — в
[репозитории](https://github.com/vasiliy-mikhailov/qwen3.8_27b_ru).
