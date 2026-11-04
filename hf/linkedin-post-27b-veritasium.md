I cut a model's errors by a third. The standard metric didn't notice.

Qwen3.8-27B in 4-bit NVFP4 runs at 300 tokens per second on a single RTX 5090 with NInfer. Its Russian, though:
• «Слиять два отсортированных списка» — not a Russian word.
• «Запускаем первую горуутину» — goroutine, with an extra letter.
• «…монохронного таймера» — right above a call to System.monotonic_time.

I re-rounded only the 4-bit MLP weights (168 matrices) with AutoRound on Russian calibration data: ru.stackoverflow Q&A, code with Russian comments, agent turns with tool calls. Same file size, same format, same kernels.

Then I counted what a reader sees. 160 prompts of Russian technical writing. Two LLM annotators marked up every answer blind, two LLM skeptics checked every error; an error counts only if both confirm it.

Hard errors per page: 1.03 → 0.70 (−32%, sign test p = 0.004). Tool calls: 40/40, as before.

Perplexity: +0.002 nats. By the usual yardstick, nothing happened.

Now a bet. The best run came either from the correct config or from a config with a mistake in it. Which one?

The mistake. During tuning, attention was simulated in 4-bit too, so the MLP learned to cancel more noise than it meets in production. The "correct" clean run gave about half the gain. An even noisier run gave no more.

Why not simply fine-tune on Russian? I tried that first and hit a ceiling. Every Russian corpus I had — ru.stackoverflow, GitHub comments, even Wikipedia — has more errors per page than the model itself.

One more check, in the same blind batches: a variant with every MLP in FP8 made 0.83 errors per page against 0.70. And that one doesn't fit 262k context on 32 GB.

Not a full fix — but life got easier.

Model (Apache-2.0): https://huggingface.co/vasiliy-mikhailov/Qwen3.8-27B-nvfp4-ru-NInfer
Scripts, annotations and the mistakes: https://github.com/vasiliy-mikhailov/qwen3.8_27b_ru

#LLM #Quantization #NVFP4 #Qwen #OpenSource
