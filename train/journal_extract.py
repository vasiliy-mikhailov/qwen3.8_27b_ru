#!/usr/bin/env python3
"""Pull structured agent results out of a workflow journal into a flat file.

A workflow's return value lands in the conversation, so it is kept to a short
summary; the full per-agent results are recovered from journal.jsonl, where each
agent's `started` line carries its label and its `result` line its output.
Labels are <tag><lens>, e.g. T017B.
"""
import json, re, sys

journal, out = sys.argv[1], sys.argv[2]
label, rows = {}, []
for line in open(journal, encoding="utf-8"):
    e = json.loads(line)
    if e.get("type") == "started":
        label[e["agentId"]] = e.get("label", "")
    elif e.get("type") == "result" and isinstance(e.get("result"), dict):
        lab = label.get(e["agentId"], "")
        m = re.match(r"^(T\d+)(.*)$", lab)
        if not m:
            continue
        rows.append({"tag": m.group(1), "lens": m.group(2), **e["result"]})
json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(f"{len(rows)} результатов -> {out}")
