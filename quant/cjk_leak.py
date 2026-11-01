#!/usr/bin/env python3
"""Count CJK characters leaking into the answers, per arm.

The failure seen in long agent runs on Flash Next: a hieroglyph inside Russian
text («у单一 решения»). Counted mechanically in the visible answer (reasoning is
not counted): Hiragana/Katakana, CJK ideographs (incl. extension A and
compatibility ideographs) and Hangul. Reported the way the budget-2027
benchmarks report it, so the numbers compare: characters per 100k generated
tokens, and the share of answers with at least one.

Usage: cjk_leak.py GEN.jsonl [GEN.jsonl ...]
"""
import collections, json, re, sys

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
arms = collections.defaultdict(lambda: {"texts": 0, "hit": 0, "chars": 0, "tokens": 0, "examples": []})
for f in sys.argv[1:]:
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        a = arms[r["arm"]]
        found = list(CJK.finditer(r["text"]))
        a["texts"] += 1
        a["tokens"] += r.get("new_tokens", 0)
        a["chars"] += len(found)
        a["hit"] += bool(found)
        for m in found[:1]:
            a["examples"].append(f'{r["id"]}: «{r["text"][max(0, m.start() - 15):m.end() + 15]}»'.replace("\n", " "))
print(f"{'arm':16s} {'answers':>7s} {'with CJK':>8s} {'chars':>6s} {'tokens':>8s} {'per 100k':>8s}")
for name, a in arms.items():
    print(f"{name:16s} {a['texts']:7d} {a['hit']:8d} {a['chars']:6d} {a['tokens']:8d} {1e5 * a['chars'] / max(a['tokens'], 1):8.2f}")
    for e in a["examples"][:5]:
        print("   ", e)
