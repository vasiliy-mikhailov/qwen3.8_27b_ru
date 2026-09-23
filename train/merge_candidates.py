#!/usr/bin/env python3
"""Between the annotation and verification passes: keep only real quotes, merge duplicates.

An annotator can misquote, and two annotators often flag the same spot. Both
are handled in plain code rather than by another agent:

  * a candidate whose quote is not a verbatim substring of the text (after
    whitespace normalisation) is dropped -- it cannot be checked, and a
    hallucinated quote must not become a counted error;
  * candidates whose quoted spans overlap are merged into one, so an error
    found by both annotators is counted once.
"""
import json, os, re, sys

blind_dir, annot_path, out_path = sys.argv[1:4]
norm = lambda s: re.sub(r"\s+", " ", s).strip()
ann = json.load(open(annot_path, encoding="utf-8"))
by_tag = {}
for a in ann:
    by_tag.setdefault(a["tag"], []).append(a)

stats = {"raw": 0, "bad_quote": 0, "kept": 0, "merged_away": 0}
out = {}
for tag, lst in by_tag.items():
    raw = open(os.path.join(blind_dir, tag + ".txt"), encoding="utf-8").read()
    body = norm(raw.split("<<<", 1)[1].rsplit(">>>", 1)[0])
    spans = []
    for a in lst:
        for e in (a.get("errors") or []):
            stats["raw"] += 1
            q = norm(e.get("quote", ""))
            i = body.find(q) if q else -1
            if i < 0:
                stats["bad_quote"] += 1
                continue
            spans.append((i, i + len(q), e))
    spans.sort(key=lambda x: (x[0], -x[1]))
    merged = []
    for s, t, e in spans:
        if merged and s < merged[-1][1]:
            stats["merged_away"] += 1
            ps, pt, pe = merged[-1]
            keep = e if len(e.get("explanation", "")) > len(pe.get("explanation", "")) else pe
            merged[-1] = (ps, max(pt, t), keep)
        else:
            merged.append((s, t, e))
    cands = [{"id": k, "quote": e["quote"], "correction": e.get("correction", ""),
              "category": e.get("category", ""), "explanation": e.get("explanation", "")}
             for k, (_, _, e) in enumerate(merged)]
    stats["kept"] += len(cands)
    out[tag] = {"candidates": cands,
                "naturalness": [a.get("naturalness") for a in lst],
                "replied_in_russian": [a.get("replied_in_russian") for a in lst]}
    with open(os.path.join(blind_dir, tag + ".cand.json"), "w", encoding="utf-8") as fh:
        json.dump(cands, fh, ensure_ascii=False, indent=1)
json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
print(stats)
print("текстов с кандидатами:", sum(1 for v in out.values() if v["candidates"]))
