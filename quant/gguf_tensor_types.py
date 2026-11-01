#!/usr/bin/env python3
"""Write a llama-quantize --tensor-type-file that reproduces a GGUF's per-tensor types.

Reads every shard of a (split) GGUF and emits one anchored, escaped regex per
tensor: ^name$=type. Quantising another source with this file and the same
imatrix rebuilds the layout of the original, so two quants that differ only in
their imatrix can be compared fairly. Also prints the type histogram and the
tokenizer/chat-template fingerprint, to check calibration-text compatibility.

Usage: gguf_tensor_types.py FIRST_SHARD.gguf OUT.txt
"""
import collections, glob, hashlib, os, re, sys

sys.path.insert(0, "/app/gguf-py")
import gguf  # noqa: E402

first, out = sys.argv[1:3]
shards = sorted(glob.glob(re.sub(r"-00001-of-(\d+)\.gguf$", r"-*-of-\1.gguf", first))) or [first]
types, hist = {}, collections.Counter()
meta = {}
for sh in shards:
    r = gguf.GGUFReader(sh)
    for t in r.tensors:
        types[t.name] = t.tensor_type.name.lower()
        hist[t.tensor_type.name] += 1
    if sh == shards[0]:
        for k in ("tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.chat_template"):
            if k in r.fields:
                f = r.fields[k]
                v = bytes(f.parts[f.data[0]]).decode(errors="replace")
                meta[k] = v if k != "tokenizer.chat_template" else hashlib.sha1(v.encode()).hexdigest()[:12] + " len=" + str(len(v))
        if "tokenizer.ggml.tokens" in r.fields:
            meta["n_tokens"] = len(r.fields["tokenizer.ggml.tokens"].data)
with open(out, "w") as f:
    for name, ty in sorted(types.items()):
        f.write("^" + re.escape(name) + "$=" + ty + "\n")
print(len(types), "tensors from", len(shards), "shards ->", out)
print("types:", dict(hist.most_common()))
print("tokenizer:", meta)
