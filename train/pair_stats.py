#!/usr/bin/env python3
"""Tally blind pairwise judgements into a preference rate per pair of arms.

Each prompt has three judgements; the prompt's verdict is the majority, and a
tie or an all-different split counts as a tie. Position bias is reported too:
how often judges picked ОТВЕТ 1 regardless of which arm stood there.
"""
import json, math, re, sys, collections

journal, key_path, arm_x, arm_y = sys.argv[1:5]
key = json.load(open(key_path))
label, votes = {}, collections.defaultdict(list)
pos = collections.Counter()
for line in open(journal, encoding="utf-8"):
    e = json.loads(line)
    if e.get("type") == "started":
        label[e["agentId"]] = e.get("label", "")
    elif e.get("type") == "result" and isinstance(e.get("result"), dict):
        lab = re.sub(r"#\d+$", "", label.get(e["agentId"], ""))
        k = key.get(lab)
        if not k:
            continue
        w = e["result"]["winner"]
        pos[w] += 1
        arm = {"1": k["first"], "2": k["second"]}.get(w, "tie")
        votes[k["id"]].append(arm)
res = collections.Counter()
for pid, v in votes.items():
    c = collections.Counter(v)
    top, n = c.most_common(1)[0]
    res[top if n >= 2 and top != "tie" else "tie"] += 1
wx, wy, t = res[arm_x], res[arm_y], res["tie"]
n = wx + wy
p = min(1.0, 2 * sum(math.comb(n, j) for j in range(min(wx, wy) + 1)) / 2 ** n) if n else 1.0
print(f"пар: {len(votes)}   {arm_y} лучше: {wy}   {arm_x} лучше: {wx}   одинаково: {t}")
print(f"{arm_y} выбран в {100*wy/max(n,1):.0f}% решённых пар;  знаковый тест p = {p:.3f}")
print(f"позиционный перекос: ОТВЕТ 1 — {pos['1']}, ОТВЕТ 2 — {pos['2']}, одинаково — {pos['tie']}")
