#!/usr/bin/env python3
"""Minimal-pair accuracy across engines, restricted to a common subset of pairs.

ninfer reports carry total_nll per stream directly. llama.cpp is scored by the
grammar-forcing trick in llamacpp_score.py, whose totals are not comparable to
ninfer's in absolute terms — the grammar emits more, shorter tokens. Only the
within-pair comparison is meaningful, which is all accuracy needs, so the arms
are compared on accuracy and never on raw NLL.
"""
import collections, json, statistics, sys


def from_ninfer(path):
    d = json.load(open(path))
    return {s["id"]: s["total_nll"] for s in d["streams"]}


def from_llamacpp(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("error"):
            out[r["id"]] = r["total_nll"]
    return out


arms = []
for spec in sys.argv[1:]:
    kind, name, path = spec.split(":", 2)
    arms.append((name, from_ninfer(path) if kind == "ninfer" else from_llamacpp(path)))

common = set.intersection(*[{k[:-3] for k in a[1] if k.endswith("-ok")} for a in arms])
common = {b for b in common if all(b + "-bad" in a[1] for a in arms)}

per = {n: collections.defaultdict(lambda: [0, 0]) for n, _ in arms}
for base in common:
    paradigm = base.rsplit("-", 1)[0]
    for name, nll in arms:
        per[name][paradigm][0] += nll[base + "-bad"] > nll[base + "-ok"]
        per[name][paradigm][1] += 1

names = [n for n, _ in arms]
print(f"{'paradigm':44s}" + "".join(f"{n:>16s}" for n in names))
for p in sorted(per[names[0]]):
    row = f"{p:44s}"
    for n in names:
        hit, tot = per[n][p]
        row += f"{100*hit/tot:15.1f}%"
    print(row)
print("-" * (44 + 16 * len(names)))
row = f"{'OVERALL':44s}"
for n in names:
    hit = sum(v[0] for v in per[n].values())
    tot = sum(v[1] for v in per[n].values())
    row += f"{100*hit/tot:15.1f}%"
print(row)
print(f"\nпар в общем подмножестве: {len(common)}")
