Draft (not yet published). Image: hf/qwen3.8-flash-next-ru-linkedin.png

I calibrated the model on Russian. It got worse at Russian.

Last time, re-rounding Qwen3.8-27B on Russian data cut its errors by a third. So when production moved to Qwen3.8-Flash-Next (180B MoE, 3-bit GGUF, one RTX PRO 6000), I repeated the idea the llama.cpp way: same Russian calibration set, turned into an importance matrix, quantized into exactly the per-tensor layout of unsloth's UD-Q3_K_XL. Same size, same format, same engine.

The good news first: Flash-Next in 3 bits already makes half the errors of the tuned 27B. In one blind batch, on 40 shared prompts: 14 vs 28 hard errors (p = 0.019).

Then the experiment. 480 prompts per variant, three variants shuffled into one blind batch (two LLM annotators + two LLM skeptics per answer), plus 480 blind pairwise judgements. Hard errors per page of Russian technical text:
• original unsloth quant: 0.44
• my pipeline with unsloth's importance matrix: 0.40
• my pipeline with the Russian one: 0.54 (+36% vs its control, p = 0.003)

More misspellings, more wrong case after verbs, more wrong words ("tax brackets" became "firewalls"). Fewer calques, which is little consolation.

The instructive part: KL divergence against a finer 4-bit quant, measured on Russian text the calibration never saw, ranked them the other way round. The Russian variant came out closer to the reference than its control, with the best perplexity of the three. The metric that looks most like fidelity pointed confidently in the wrong direction, just as perplexity stayed flat on 27B while errors dropped by a third.

Production stays on the original quant. Scripts, six prompt sets, all 1,440 annotated answers and the negative result are public.

Calibrate on what you care about. Measure on what the reader sees.

https://github.com/vasiliy-mikhailov/qwen3.8_27b_ru

#LLM #Quantization #GGUF #Qwen #OpenSource
