I gave a model extra Russian lessons. It came back with a Chinese character inside a Russian word.

And by the metric I would normally trust, it had improved.

In September I wrote how the same trick worked on Qwen3.8-27B: re-rounding the weights on Russian data cut hard errors by a third, 1.03 → 0.70 per page (p = 0.004). Perplexity didn't move.

Then production moved to Qwen3.8-Flash-Next: a 180B MoE in 3-bit GGUF (unsloth UD-Q3_K_XL, 90 GB) on one RTX PRO 6000. It already makes half the errors of the tuned 27B: 14 vs 28 on 40 shared prompts in one blind batch (p = 0.019).

So I repeated the idea the llama.cpp way: the same Russian calibration set → an importance matrix (imatrix) → a quant in exactly unsloth's per-tensor layout. Same size, format and engine. The control: the same pipeline with unsloth's own imatrix.

Place your bet: better, worse, or no change?

480 prompts per variant, all three variants shuffled into one blind batch. Two LLM annotators marked up every answer, two LLM skeptics checked every candidate error. Plus 480 blind pairwise comparisons, three LLM judges per pair.

Hard errors per page of Russian technical text (250 words):
• original unsloth quant: 0.44
• control, unsloth's imatrix: 0.40
• Russian imatrix: 0.54

On 27B the trick removed a third of the errors. Here it added a third: +36% vs the control (p = 0.003), +20% vs the original (p = 0.08). Against the original, the judges picked the Russian variant in 47% of pairs. More misspellings, more wrong case after verbs, more wrong words. Fewer calques, which is little consolation.

Real answers of the Russian variant:
• «Начинаем с правойmost цифры»: "rightmost", half Russian, half English.
• «в指定ованный бакет Amazon S3»: the Chinese character from the first line.
• In a Kotlin code review, tax brackets became «Брандмауэры». Firewalls.

Now the twist. KL divergence to a finer 4-bit quant, measured on Russian text the calibration never saw, ranked the variants the other way round. The Russian variant came out closer to the reference than the control: 0.101 vs 0.115 (original: 0.096). Its perplexity was the best of the three. The metric that looks most like fidelity pointed confidently in the wrong direction. On 27B, perplexity did the same.

Why? A hypothesis, not yet tested. An imatrix weights rounding error by how strongly activations fire on the calibration text. Russian calibration moves precision to where activations on that text are large, and seemingly takes it from the features that pick the form of a word during generation. Two more suspects: the matrix was computed on Q4 activations instead of BF16, and unsloth's calibration is 1.5× larger.

Production stays on the original quant. Scripts, six prompt sets, all 1,440 annotated answers and the negative result are public.

Calibrate on what matters. Measure with what the reader sees.

https://github.com/vasiliy-mikhailov/qwen3.8_27b_ru

#LLM #Quantization #GGUF #Qwen #OpenSource
