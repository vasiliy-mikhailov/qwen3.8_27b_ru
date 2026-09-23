#!/usr/bin/env python3
"""Compare two .ninfer artifacts object by object.

Used to show that converting the untouched unsloth NVFP4 store reproduces the
production artifact byte for byte, and that a re-quantised store differs from
it only in the MLP objects it replaced.

Usage (ninfer checkout at /ninfer): cmp_artifacts.py A.ninfer B.ninfer
"""
import collections, hashlib, sys

sys.path.insert(0, "/ninfer")
from tools.artifact.reader import Artifact  # noqa: E402


def digests(path):
    out = {}
    with Artifact.open(path) as a:
        for o in a.objects:
            h = hashlib.sha1()
            for chunk in a.iter_object(o.id):
                h.update(chunk)
            out[o.id] = h.hexdigest()
    return out


x, y = digests(sys.argv[1]), digests(sys.argv[2])
same = sum(1 for k in x if y.get(k) == x[k])
diff = [k for k in x if k in y and y[k] != x[k]]
only = sorted(set(x) ^ set(y))
print(f"objects: {len(x)} vs {len(y)}, identical {same}, differ {len(diff)}, unmatched {len(only)}")
print("differ by prefix:", dict(collections.Counter(k.rsplit("/", 1)[0] for k in diff).most_common(12)))
print("sample:", diff[:12])
