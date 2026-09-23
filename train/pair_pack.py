#!/usr/bin/env python3
"""Pack blind pairwise comparisons: same prompt, two arms, which reads better?

"Chosen in N% of cases" is a number a user understands; perplexity is not.
Judges see two answers to one task labelled only ОТВЕТ 1 and ОТВЕТ 2.

Language models tend to favour a position, so every pair is written in BOTH
orders. Judges are assigned orders so that position is balanced: two judges see
opposite orders of the same pair, the third sees one chosen by the pair index.
The key records which arm sat in which position.
"""
import json, os, sys

gen_x, gen_y, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
X = {json.loads(l)["id"]: json.loads(l) for l in open(gen_x, encoding="utf-8")}
Y = {json.loads(l)["id"]: json.loads(l) for l in open(gen_y, encoding="utf-8")}
ids = sorted(set(X) & set(Y))
key = {}
for i, pid in enumerate(ids):
    x, y = X[pid], Y[pid]
    for order, (first, second) in (("o1", (x, y)), ("o2", (y, x))):
        tag = f"P{i:03d}{order}"
        with open(os.path.join(out_dir, tag + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("ЗАДАНИЕ:\n" + x["prompt"] + "\n\nОТВЕТ 1:\n<<<\n" + first["text"] +
                     "\n>>>\n\nОТВЕТ 2:\n<<<\n" + second["text"] + "\n>>>\n")
        key[tag] = {"id": pid, "first": first["arm"], "second": second["arm"]}
json.dump(key, open(os.path.join(out_dir, "key.json"), "w"), indent=0)
jobs = []
for i in range(len(ids)):
    jobs += [f"P{i:03d}o1", f"P{i:03d}o2", f"P{i:03d}o{1 + i % 2}"]
json.dump(jobs, open(os.path.join(out_dir, "jobs.json"), "w"))
print(f"{len(ids)} пар, {len(jobs)} заданий судьям -> {out_dir}")
