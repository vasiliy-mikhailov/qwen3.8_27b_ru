#!/usr/bin/env python3
"""Turn a ninfer perplexity report over the RuBLiMP corpus into minimal-pair accuracy.

A pair counts as correct when the grammatical sentence gets the lower total NLL. Because the
two members of a pair differ in one word, total NLL is the right comparison: length is nearly
equal and normalising by tokens would reward the shorter member.
"""
import json, sys, collections, statistics

reports = sys.argv[1:]
if not reports:
    sys.exit("usage: score_rublimp.py <report.json> [<report.json> ...]")

table = {}
for path in reports:
    d = json.load(open(path))
    nll = {s["id"]: s["total_nll"] for s in d["streams"]}
    per = collections.defaultdict(lambda: [0, 0])
    margins = collections.defaultdict(list)
    for sid in nll:
        if not sid.endswith("-ok"):
            continue
        base = sid[: -len("-ok")]
        bad = base + "-bad"
        if bad not in nll:
            continue
        paradigm = base.rsplit("-", 1)[0]
        m = nll[bad] - nll[sid]          # positive when the model prefers the grammatical one
        per[paradigm][0] += m > 0
        per[paradigm][1] += 1
        margins[paradigm].append(m)
    table[path] = (per, margins)

names = list(table)
paradigms = sorted(table[names[0]][0])
print(f"{'paradigm':44s}" + "".join(f"{n.split('/')[-2]:>22s}" for n in names))
for p in paradigms:
    row = f"{p:44s}"
    for n in names:
        hit, tot = table[n][0][p]
        row += f"{100*hit/tot:12.1f}% {statistics.mean(table[n][1][p]):+8.2f}"
    print(row)
print("-" * (44 + 22 * len(names)))
row = f"{'OVERALL':44s}"
for n in names:
    hit = sum(v[0] for v in table[n][0].values())
    tot = sum(v[1] for v in table[n][0].values())
    allm = [m for v in table[n][1].values() for m in v]
    row += f"{100*hit/tot:12.1f}% {statistics.mean(allm):+8.2f}"
print(row)
print("\nколонки: точность на парах | средний зазор в натах (NLL_плохого − NLL_хорошего)")
