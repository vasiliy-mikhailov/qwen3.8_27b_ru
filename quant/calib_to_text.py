#!/usr/bin/env python3
"""Turn the token-id calibration set (calib_ru.jsonl, built for Qwen3.8-27B) back
into text for llama-imatrix.

Qwen3.8-Flash-Next shares the 27B tokenizer (qwen35, 248,320 tokens) and its chat
markup, so the same 256 sequences -- ru.stackoverflow as chat, code with Russian
comments, agent turns writing such code through tool calls, comment blocks, plain
code -- calibrate it too. Special tokens are kept as text; run llama-imatrix with
--parse-special so they become tokens again.

Usage: calib_to_text.py TOKENIZER_DIR calib_ru.jsonl OUT.txt
"""
import json, sys
from transformers import AutoTokenizer

tok_dir, src, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(tok_dir)
n = chars = 0
with open(out, "w", encoding="utf-8") as f:
    for line in open(src, encoding="utf-8"):
        text = tok.decode(json.loads(line)["input_ids"], skip_special_tokens=False)
        f.write(text + "\n")
        n += 1; chars += len(text)
print(f"{n} sequences, {chars/1e6:.2f} M characters -> {out}")
