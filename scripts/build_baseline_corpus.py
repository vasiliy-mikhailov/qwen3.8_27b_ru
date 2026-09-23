#!/usr/bin/env python3
"""Build one perplexity corpus covering the target language and the regression guards.

ninfer ships a fixed 1M-token corpus with four domains: English reference,
English long-form, Chinese reference and ninfer's own C++/CUDA code. Those are
exactly the guards a Russian fine-tune needs — if Russian improves while code
perplexity climbs, the trade is visible in the same run and the same units.

Russian is added as extra streams from a reference file, so one invocation
reports target and guards side by side, per domain.
"""
import json, os, shutil, sys

BUNDLED = sys.argv[1]           # .../eval/corpora/perplexity-1m
RU_FILE = sys.argv[2]           # human Russian reference text
DEST = sys.argv[3]
RU_STREAMS = int(sys.argv[4]) if len(sys.argv) > 4 else 4

os.makedirs(os.path.join(DEST, "data"), exist_ok=True)
src = json.load(open(os.path.join(BUNDLED, "manifest.json"), encoding="utf-8"))

streams = []
for s in src["streams"]:
    rel = s["path"]
    dst = os.path.join(DEST, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(os.path.join(BUNDLED, rel), dst)
    streams.append(s)

text = open(RU_FILE, encoding="utf-8").read()
# Split on paragraph boundaries so no stream starts mid-sentence.
paras = [p for p in text.split("\n\n") if p.strip()]
per = len(paras) // RU_STREAMS
for i in range(RU_STREAMS):
    chunk = paras[i * per:(i + 1) * per] if i < RU_STREAMS - 1 else paras[i * per:]
    rel = f"data/russian/{i:02d}.txt"
    os.makedirs(os.path.join(DEST, "data/russian"), exist_ok=True)
    with open(os.path.join(DEST, rel), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(chunk))
    streams.append({"id": f"ruwiki-{i:02d}", "domain": "russian_reference", "path": rel})

ids = [s["id"] for s in streams]
quick = []
seen = set()
for s in streams:
    if s["domain"] not in seen:
        seen.add(s["domain"])
        quick.append(s["id"])
manifest = {"corpus_id": "qwen38-ru-baseline-v1", "streams": streams,
            "modes": {"quick": quick, "full": ids}}
json.dump(manifest, open(os.path.join(DEST, "manifest.json"), "w"), ensure_ascii=False)
import collections
print("domains:", dict(collections.Counter(s["domain"] for s in streams)))
print("streams:", len(streams), "-> ", DEST)
