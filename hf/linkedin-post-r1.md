Published on LinkedIn 2026-09-26: https://www.linkedin.com/feed/update/urn:li:activity:7509680941913526272/
Image: hf/qwen3.8-27b-nvfp4-ru-linkedin.png

Familiar?

"Слиять два отсортированных списка" — not a Russian word.
"Запускаем первую горуутину" — goroutine, with an extra letter.
"…монохронного таймера" — right above a call to System.monotonic_time.

All of it at 300 tokens per second: Qwen3.8-27B in 4-bit NVFP4 on a single RTX 5090 with NInfer.

We kept the speed. And the errors — just a third fewer.

What changed: I re-rounded only the 4-bit MLP weights (168 matrices) with AutoRound on Russian calibration data — Stack Overflow Q&A, code with Russian comments, agent turns with tool calls. Same file size, same format, same kernels.

How it was measured: not perplexity, but what a reader sees. 160 prompts of Russian technical writing (code comments, docstrings, reviews, specs). Every answer was annotated blind by two LLM annotators, then checked by two LLM skeptics; an error counts only if both confirm it. Hard errors per page: 1.03 → 0.70 (−32%, sign test p = 0.004). Tool calls: 40/40, as before.

What surprised me:
• Perplexity didn't move (+0.002 nats) while errors dropped by a third. Wrong instrument for this question.
• The best run came from a config mistake: during tuning, attention was simulated in 4-bit too, so the MLP learned to cancel more noise than it meets in production. The "correct" clean run gave about half the gain; an even noisier one gave no more.
• In the same blind batches it made no more errors than a model with every MLP in FP8 (0.70 vs 0.83 per page) — and that one doesn't fit 262k context on 32 GB.
• Fine-tuning had hit a ceiling first: every Russian corpus I had (ru.stackoverflow, GitHub comments, even Wikipedia) has more errors per page than the model itself.

Not a full fix — but life got easier.

Model (Apache-2.0): https://huggingface.co/vasiliy-mikhailov/Qwen3.8-27B-nvfp4-ru-NInfer
Scripts, annotations and the mistakes: https://github.com/vasiliy-mikhailov/qwen3.8_27b_ru

#LLM #Quantization #NVFP4 #Qwen #OpenSource
