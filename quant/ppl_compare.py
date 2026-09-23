#!/usr/bin/env python3
"""Per-domain perplexity of several ninfer artifacts against production.

Usage: ppl_compare.py BASE_REPORT NAME=REPORT [NAME=REPORT ...]
Prints mean NLL per domain and the difference from the base in nats; all
reports must score the same streams (same corpus, same token counts).
"""
import json, sys

base = json.load(open(sys.argv[1]))
runs = [(s.split("=", 1)[0], json.load(open(s.split("=", 1)[1]))) for s in sys.argv[2:]]
bd = {d["domain"]: d for d in base["domains"]}
print(f"{'domain':24s} {'prod':>8s}" + "".join(f" {n:>16s}" for n, _ in runs))
for dom, b in bd.items():
    row = f"{dom:24s} {b['mean_nll']:8.4f}"
    for _, r in runs:
        d = {x["domain"]: x for x in r["domains"]}[dom]
        assert d["scored_tokens"] == b["scored_tokens"], dom
        row += f" {d['mean_nll']:8.4f} {d['mean_nll'] - b['mean_nll']:+7.4f}"
    print(row)
row = f"{'overall':24s} {base['overall']['mean_nll']:8.4f}"
for _, r in runs:
    row += f" {r['overall']['mean_nll']:8.4f} {r['overall']['mean_nll'] - base['overall']['mean_nll']:+7.4f}"
print(row)
