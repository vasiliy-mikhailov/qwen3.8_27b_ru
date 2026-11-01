#!/usr/bin/env python3
"""Take every tensor that does not depend on the imatrix from the original GGUF.

A requantisation from Q8_0 reproduces the original's layout, but a tensor the
original stored exactly (F32 gates, say) arrives through Q8_0 rounded, and a
tensor quantised without imatrix data (embedding lookups, the output head)
arrives from a rounded source. Neither has anything to do with the calibration
being tested, so their bytes are copied from the original in place: same name,
type and size, overwritten at the tensor's offset. What is left differing is
exactly the set of tensors the imatrix steers.

A tensor depends on the imatrix when any of the given imatrix files has an
entry for it and its type is quantised (not F32/F16/BF16).

Usage: graft_gguf.py TARGET_FIRST_SHARD DONOR_FIRST_SHARD IMATRIX.gguf [IMATRIX.gguf ...] [--dry-run]
"""
import collections, glob, re, sys

import numpy as np

sys.path.insert(0, "/app/gguf-py")
import gguf  # noqa: E402

args = [x for x in sys.argv[1:] if x != "--dry-run"]
dry = "--dry-run" in sys.argv
target, donor, imats = args[0], args[1], args[2:]
FLOAT = {"F32", "F16", "BF16"}


def shards(first):
    return sorted(glob.glob(re.sub(r"-00001-of-(\d+)\.gguf$", r"-*-of-\1.gguf", first))) or [first]


steered = set()
for f in imats:
    for t in gguf.GGUFReader(f).tensors:
        steered.add(t.name.rsplit(".", 1)[0])
src = {}
for sh in shards(donor):
    for t in gguf.GGUFReader(sh).tensors:
        src[t.name] = t

stats = collections.Counter()
for sh in shards(target):
    r = gguf.GGUFReader(sh, "r" if dry else "r+")
    for t in r.tensors:
        d = src.get(t.name)
        if d is None or d.tensor_type != t.tensor_type or d.n_bytes != t.n_bytes:
            raise SystemExit(f"{t.name}: no matching tensor in the donor")
        ty = t.tensor_type.name
        if ty not in FLOAT and t.name in steered:
            stats["steered, kept"] += 1
            continue
        a = np.asarray(t.data).view(np.uint8).reshape(-1)
        b = np.asarray(d.data).view(np.uint8).reshape(-1)
        if np.array_equal(a, b):
            stats["already identical"] += 1
            continue
        stats[f"copied {ty}"] += 1
        if not dry:
            a[:] = b
    if not dry:
        r.data.flush()
print(("would be: " if dry else "") + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))
