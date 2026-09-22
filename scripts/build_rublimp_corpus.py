#!/usr/bin/env python3
"""Turn RuBLiMP minimal pairs into a ninfer perplexity corpus.

ninfer-perplexity reports total_nll per stream, so putting each sentence of a pair in its
own stream yields per-pair accuracy from a single model load: the model is correct on a pair
when it assigns lower NLL to the grammatical sentence. Every window starts from empty state,
so a one-sentence stream is scored without cross-sentence contamination.
"""
import json, os, random, sys
import pyarrow.parquet as pq

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/rublimp"
DEST = sys.argv[2] if len(sys.argv) > 2 else "data/rublimp-corpus"
PER = int(sys.argv[3]) if len(sys.argv) > 3 else 100

rng = random.Random(20260922)
os.makedirs(os.path.join(DEST, "data"), exist_ok=True)
streams, pairs = [], []

for fn in sorted(os.listdir(SRC)):
    if not fn.endswith(".parquet"):
        continue
    paradigm = fn[: -len(".parquet")]
    rows = pq.read_table(os.path.join(SRC, fn)).to_pylist()
    # source_sentence is the grammatical member; target_sentence carries the perturbation.
    rows = [r for r in rows if r["source_sentence"] and r["target_sentence"]]
    for r in rng.sample(rows, min(PER, len(rows))):
        pid = "{}-{}".format(paradigm, r["id"])
        for side, text in (("ok", r["source_sentence"]), ("bad", r["target_sentence"])):
            sid = pid + "-" + side
            rel = os.path.join("data", sid + ".txt")
            with open(os.path.join(DEST, rel), "w", encoding="utf-8") as fh:
                fh.write(text.strip() + "\n")
            streams.append({"id": sid, "domain": paradigm, "path": rel})
        pairs.append({"pair": pid, "paradigm": paradigm, "level": r["level"],
                      "phenomenon": r["phenomenon"]})

ids = [s["id"] for s in streams]
manifest = {"corpus_id": "rublimp-minimal-pairs",
            "streams": streams,
            "modes": {"quick": ids[:200], "full": ids}}
json.dump(manifest, open(os.path.join(DEST, "manifest.json"), "w"), ensure_ascii=False)
json.dump(pairs, open(os.path.join(DEST, "pairs.json"), "w"), ensure_ascii=False)
print("paradigms={} pairs={} streams={}".format(
    len(set(p["paradigm"] for p in pairs)), len(pairs), len(streams)))
