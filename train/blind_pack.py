#!/usr/bin/env python3
"""Shuffle and anonymise two arms of generations for blind error annotation.

Annotators must not know which arm a text came from, or any preference for the
fine-tune leaks into the count. Each text gets an opaque tag; the key mapping
tag -> (arm, prompt id) is written separately and read only after annotation.
"""
import json, random, sys

import os
gen_files, out_dir, out_key = sys.argv[1:-2], sys.argv[-2], sys.argv[-1]
os.makedirs(out_dir, exist_ok=True)
rows = []
for f in gen_files:
    rows += [json.loads(l) for l in open(f, encoding="utf-8")]
random.Random(20260923).shuffle(rows)
items, key = [], {}
for i, r in enumerate(rows):
    tag = f"T{i:03d}"
    key[tag] = {"arm": r["arm"], "id": r["id"], "kind": r["kind"]}
    items.append({"tag": tag, "prompt": r["prompt"], "text": r["text"]})
    # One file per text: annotators read it themselves, so nothing about the
    # arm travels with it -- only an opaque tag, the task and the answer.
    with open(os.path.join(out_dir, tag + ".txt"), "w", encoding="utf-8") as fh:
        fh.write("ЗАДАНИЕ, на которое отвечала модель:\n" + r["prompt"] +
                 "\n\nОТВЕТ МОДЕЛИ (его и нужно проверить):\n<<<\n" + r["text"] + "\n>>>\n")
json.dump(items, open(os.path.join(out_dir, "items.json"), "w", encoding="utf-8"), ensure_ascii=False)
json.dump(key, open(out_key, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(f"{len(items)} текстов -> {out_dir}; ключ -> {out_key}")
