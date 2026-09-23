#!/usr/bin/env python3
"""Unblind confirmed errors and compare arms prompt by prompt.

Arms answer the same prompts, so each prompt is its own control: a hard prompt
produces more errors in every arm. Comparisons are paired -- per-prompt
differences, a sign test on them, and a bootstrap over prompts for the interval
of the total change.

Errors are split in two, because they mean different things:
  hard   agreement, government, non-words, wrong words, spelling, foreign
         script, sense -- violations of the language itself;
  soft   calques and punctuation -- real, but closer to register than grammar.
Naturalness (1-5, the mean of both annotators) is reported separately: it is
where "reads like machine output" lives, and no error count captures it.

Counts are also normalised per 1000 Russian words, since training can change
how much Russian gets written at all.

Usage: unblind_stats.py confirmed.json key.json ARM_A ARM_B gen_files...
"""
import json, math, random, re, sys, collections

conf_path, key_path, arm_a, arm_b = sys.argv[1:5]
gens = sys.argv[5:]
HARD = {"согласование", "управление", "несуществующее_слово", "неверное_слово",
        "орфография", "чужой_алфавит", "смысл"}
SOFT = {"калька", "пунктуация"}

conf = json.load(open(conf_path, encoding="utf-8"))
key = json.load(open(key_path, encoding="utf-8"))
CYR = re.compile(r"[А-Яа-яЁё]+")
words = collections.Counter()
for f in gens:
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if r["arm"] in (arm_a, arm_b):
            words[r["arm"]] += len(CYR.findall(r["text"]))

per = collections.defaultdict(lambda: collections.defaultdict(lambda: {"hard": 0, "soft": 0, "nat": None}))
cats = {arm_a: collections.Counter(), arm_b: collections.Counter()}
examples = {arm_a: [], arm_b: []}
for tag, v in conf.items():
    k = key[tag]
    arm = k["arm"]
    if arm not in (arm_a, arm_b):
        continue
    cell = per[k["id"]][arm]
    for e in v["confirmed"]:
        c = e["category"]
        cell["hard" if c in HARD else "soft"] += 1
        cats[arm][c] += 1
        examples[arm].append((c, e["quote"], e.get("correction", "")))
    nat = [x for x in v.get("naturalness", []) if isinstance(x, (int, float))]
    cell["nat"] = sum(nat) / len(nat) if nat else None

ids = sorted(i for i in per if arm_a in per[i] and arm_b in per[i])


def paired(metric, lower_is_better=True):
    xa = [per[i][arm_a][metric] for i in ids]
    xb = [per[i][arm_b][metric] for i in ids]
    pairs = [(a, b) for a, b in zip(xa, xb) if a is not None and b is not None]
    sa, sb = sum(a for a, _ in pairs), sum(b for _, b in pairs)
    better = sum(1 for a, b in pairs if (b < a if lower_is_better else b > a))
    worse = sum(1 for a, b in pairs if (b > a if lower_is_better else b < a))
    n = better + worse
    k = min(better, worse)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    rng = random.Random(7)
    boots = []
    for _ in range(10000):
        s = [rng.choice(pairs) for _ in pairs]
        ta, tb = sum(a for a, _ in s), sum(b for _, b in s)
        if ta:
            boots.append((tb - ta) / ta)
    boots.sort()
    ci = (boots[int(.025 * len(boots))], boots[int(.975 * len(boots))]) if boots else (float("nan"),) * 2
    return sa, sb, better, worse, len(pairs) - n, p, ci


print(f"пар (промптов): {len(ids)}   русских слов: {arm_a} {words[arm_a]}, {arm_b} {words[arm_b]}\n")
for m, title in (("hard", "ТВЁРДЫЕ ошибки"), ("soft", "кальки + пунктуация")):
    sa, sb, bt, wr, tie, p, ci = paired(m)
    ra = 1000 * sa / max(words[arm_a], 1)
    rb = 1000 * sb / max(words[arm_b], 1)
    print(f"{title}:")
    print(f"  всего: {arm_a} {sa}  →  {arm_b} {sb}   ({100*(sb-sa)/max(sa,1):+.1f}%)")
    print(f"  на 1000 рус. слов: {ra:.2f} → {rb:.2f}   ({100*(rb-ra)/max(ra,1e-9):+.1f}%)")
    print(f"  по промптам: лучше {bt}, хуже {wr}, поровну {tie};  знаковый тест p = {p:.3f}")
    print(f"  95% бутстреп-интервал изменения: {100*ci[0]:+.1f}% … {100*ci[1]:+.1f}%\n")

xa = [per[i][arm_a]["nat"] for i in ids if per[i][arm_a]["nat"] is not None and per[i][arm_b]["nat"] is not None]
xb = [per[i][arm_b]["nat"] for i in ids if per[i][arm_a]["nat"] is not None and per[i][arm_b]["nat"] is not None]
if xa:
    d = [b - a for a, b in zip(xa, xb)]
    up, down = sum(x > 0 for x in d), sum(x < 0 for x in d)
    n = up + down
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(min(up, down) + 1)) / 2 ** n) if n else 1.0
    print(f"ЕСТЕСТВЕННОСТЬ (1–5): {arm_a} {sum(xa)/len(xa):.2f} → {arm_b} {sum(xb)/len(xb):.2f};"
          f"  лучше {up}, хуже {down};  p = {p:.3f}\n")

print("по категориям:")
for c in sorted(set(cats[arm_a]) | set(cats[arm_b])):
    print(f"  {c:22s} {arm_a} {cats[arm_a][c]:4d}   {arm_b} {cats[arm_b][c]:4d}")

kind_of = {v["id"]: v["kind"] for v in key.values()}
bk = collections.defaultdict(lambda: [0, 0])
for i in ids:
    bk[kind_of[i]][0] += per[i][arm_a]["hard"]
    bk[kind_of[i]][1] += per[i][arm_b]["hard"]
print("\nтвёрдые по типам задач:")
for k, (x, y) in sorted(bk.items()):
    print(f"  {k:16s} {x:4d} → {y:4d}")

json.dump({a: examples[a] for a in examples}, open(conf_path.replace(".json", f".examples_{arm_a}_{arm_b}.json"), "w",
                                                  encoding="utf-8"), ensure_ascii=False, indent=0)
