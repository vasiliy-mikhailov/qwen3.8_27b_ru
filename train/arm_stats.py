#!/usr/bin/env python3
"""User-facing error metrics for any set of annotated arms, plus paired comparisons.

Takes annotation rounds as confirmed.json:key.json pairs and the generation
files the arms came from. Arms are named round:arm, e.g. r4:q8. For each arm:
hard errors per page (250 Russian words), share of answers with no hard error,
soft errors (calques, punctuation), contradictions with the code, naturalness.
For each requested pair: a paired comparison over shared prompts -- which arm
had fewer hard errors on how many prompts, sign test, bootstrap interval.

Usage:
  arm_stats.py --rounds r1=conf.json:key.json r4=... --gens g1.jsonl ... \
               --arms r1:base r4:q8 r4:nvfp4 --pairs r4:nvfp4,r4:q8 r1:base,r4:q8
"""
import argparse, json, math, random, re, collections

HARD = {"согласование", "управление", "несуществующее_слово", "неверное_слово",
        "орфография", "чужой_алфавит", "смысл"}
SOFT = {"калька", "пунктуация"}
CYR = re.compile(r"[А-Яа-яЁё]+")
PAGE = 250

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", nargs="+", required=True)
ap.add_argument("--gens", nargs="+", required=True)
ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--pairs", nargs="*", default=[])
ap.add_argument("--alias", nargs="*", default=[], help="arm=gen_arm when a re-annotated arm reuses texts")
a = ap.parse_args()

alias = dict(x.split("=") for x in a.alias)
words = {}
for f in a.gens:
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        words[(r["arm"], r["id"])] = len(CYR.findall(r["text"]))

cell = collections.defaultdict(dict)
for spec in a.rounds:
    rnd, files = spec.split("=")
    cp, kp = files.split(":")
    conf = json.load(open(cp, encoding="utf-8"))
    key = json.load(open(kp, encoding="utf-8"))
    for tag, v in conf.items():
        k = key[tag]
        name = f"{rnd}:{k['arm']}"
        nat = [x for x in v.get("naturalness", []) if isinstance(x, (int, float))]
        cell[name][k["id"]] = {
            "hard": sum(1 for e in v["confirmed"] if e["category"] in HARD),
            "soft": sum(1 for e in v["confirmed"] if e["category"] in SOFT),
            "tech": len(v.get("technical", [])),
            "nat": sum(nat) / len(nat) if nat else None,
            "words": words.get((alias.get(k["arm"], k["arm"]), k["id"]), 0),
        }

print(f"{'рука':18s} {'ответов':>7s} {'ошибок/стр':>10s} {'без ошибок':>10s} {'твёрдых':>7s} "
      f"{'кальки+пункт.':>13s} {'противор. коду':>14s} {'естеств.':>8s} {'рус.слов':>8s}")
for arm in a.arms:
    d = cell[arm]
    n = len(d)
    w = sum(x["words"] for x in d.values())
    h = sum(x["hard"] for x in d.values())
    nat = [x["nat"] for x in d.values() if x["nat"] is not None]
    print(f"{arm:18s} {n:7d} {PAGE*h/max(w,1):10.2f} {100*sum(x['hard']==0 for x in d.values())/n:9.0f}% "
          f"{h:7d} {sum(x['soft'] for x in d.values()):13d} {sum(x['tech'] for x in d.values()):14d} "
          f"{sum(nat)/len(nat):8.2f} {w:8d}")

for pair in a.pairs:
    x, y = pair.split(",")
    ids = sorted(set(cell[x]) & set(cell[y]))
    hx = [cell[x][i]["hard"] for i in ids]
    hy = [cell[y][i]["hard"] for i in ids]
    better = sum(1 for p, q in zip(hx, hy) if q < p)
    worse = sum(1 for p, q in zip(hx, hy) if q > p)
    n = better + worse
    pv = min(1.0, 2 * sum(math.comb(n, j) for j in range(min(better, worse) + 1)) / 2 ** n) if n else 1.0
    rng = random.Random(9)
    boots = []
    for _ in range(10000):
        s = [rng.randrange(len(ids)) for _ in ids]
        tx = sum(hx[i] for i in s)
        ty = sum(hy[i] for i in s)
        if tx:
            boots.append((ty - tx) / tx)
    boots.sort()
    lo, hi = boots[int(.025 * len(boots))], boots[int(.975 * len(boots))]
    print(f"\n{y} против {x} на {len(ids)} общих промптах: твёрдых {sum(hx)} → {sum(hy)} "
          f"({100*(sum(hy)-sum(hx))/max(sum(hx),1):+.0f}%)")
    print(f"  промптов, где у {y} меньше ошибок: {better}, больше: {worse}, поровну: {len(ids)-n};"
          f"  p = {pv:.3f};  95% интервал {100*lo:+.0f}% … {100*hi:+.0f}%")
