#!/usr/bin/env python3
"""Combine two skeptics' verdicts into the confirmed error list.

A candidate counts as a confirmed error only when BOTH skeptics call it one --
the normativist and the working developer. Anything either of them marks as
technical_only (the claim is wrong but the Russian is fine) is set aside: it is
a defect of content, not of language, and is reported separately.

Category follows the normativist when the two disagree.
"""
import json, re, sys

journal, cands_path, out_path = sys.argv[1:4]
cands = json.load(open(cands_path, encoding="utf-8"))
label, verd = {}, {}
for line in open(journal, encoding="utf-8"):
    e = json.loads(line)
    if e.get("type") == "started":
        label[e["agentId"]] = e.get("label", "")
    elif e.get("type") == "result" and isinstance(e.get("result"), dict):
        m = re.match(r"^(T\d+)([ND])$", label.get(e["agentId"], ""))
        if m:
            verd[(m.group(1), m.group(2))] = {v["id"]: v for v in e["result"].get("verdicts", [])}

out, stats = {}, {"candidates": 0, "confirmed_language": 0, "technical": 0, "split": 0,
                  "rejected": 0, "unverified_texts": 0}
for tag, v in cands.items():
    conf, tech = [], []
    cl = v["candidates"]
    if cl and ((tag, "N") not in verd or (tag, "D") not in verd):
        stats["unverified_texts"] += 1
    for c in cl:
        stats["candidates"] += 1
        n = verd.get((tag, "N"), {}).get(c["id"])
        d = verd.get((tag, "D"), {}).get(c["id"])
        if not n or not d:
            continue
        if n["is_error"] and d["is_error"]:
            row = {**c, "category": n["category"]}
            if n["technical_only"] or d["technical_only"]:
                tech.append(row)
                stats["technical"] += 1
            else:
                conf.append(row)
                stats["confirmed_language"] += 1
        elif n["is_error"] != d["is_error"]:
            stats["split"] += 1
        else:
            stats["rejected"] += 1
    out[tag] = {"confirmed": conf, "technical": tech, "naturalness": v["naturalness"],
                "replied_in_russian": v["replied_in_russian"]}
json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
print(stats)
