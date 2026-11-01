#!/usr/bin/env python3
"""Compare two (split) GGUFs tensor by tensor: same names, same types, same bytes?

Checks that a requantised file rebuilds the layout of the original. Per type:
how many tensors match byte for byte and, for the rest, the median share of
bytes that differ. Q8_0 requantised from Q8_0 should come out identical; the
imatrix-driven types differ by how much the source and the imatrix moved them.

Usage: cmp_gguf.py A_FIRST_SHARD.gguf B_FIRST_SHARD.gguf
"""
import collections, glob, re, statistics, sys

import numpy as np

sys.path.insert(0, "/app/gguf-py")
import gguf  # noqa: E402


def tensors(first):
    shards = sorted(glob.glob(re.sub(r"-00001-of-(\d+)\.gguf$", r"-*-of-\1.gguf", first))) or [first]
    out = {}
    for sh in shards:
        for t in gguf.GGUFReader(sh).tensors:
            out[t.name] = t
    return out


a, b = tensors(sys.argv[1]), tensors(sys.argv[2])
only = sorted(set(a) ^ set(b))
if only:
    print(f"{len(only)} tensors in one file only, e.g. {only[:5]}")
by_type = collections.defaultdict(lambda: {"n": 0, "same": 0, "diff": []})
mismatch = []
for name in sorted(set(a) & set(b)):
    ta, tb = a[name], b[name]
    if ta.tensor_type != tb.tensor_type:
        mismatch.append(f"{name}: {ta.tensor_type.name} vs {tb.tensor_type.name}")
        continue
    s = by_type[ta.tensor_type.name]
    s["n"] += 1
    da, db = np.asarray(ta.data).view(np.uint8).reshape(-1), np.asarray(tb.data).view(np.uint8).reshape(-1)
    if da.size == db.size and np.array_equal(da, db):
        s["same"] += 1
    else:
        s["diff"].append(float((da != db).mean()) if da.size == db.size else 1.0)
print(f"type mismatches: {len(mismatch)}")
for m in mismatch[:10]:
    print("  ", m)
for ty, s in sorted(by_type.items(), key=lambda kv: -kv[1]["n"]):
    d = f", differing bytes median {statistics.median(s['diff']):.1%}" if s["diff"] else ""
    print(f"{ty:8s} {s['n']:5d} tensors, {s['same']:5d} identical{d}")
