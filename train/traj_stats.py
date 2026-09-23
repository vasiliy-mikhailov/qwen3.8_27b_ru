#!/usr/bin/env python3
"""Error trajectory across training checkpoints, in terms a user would read.

Combines annotation rounds. Round one annotated the base and a first five-minute
run; round two annotated the 5/20/60-minute checkpoints of one long run plus 40
base texts again. Those 40 are the drift anchor: the same texts annotated in two
separate rounds by fresh agents. If their counts differ a lot, cross-round
differences cannot be trusted; if they agree, checkpoints can be compared with
the round-one base directly.

Reported per arm: errors per page (250 Russian words), share of answers with no
hard error, and a paired comparison with the base over the same prompts.

Usage: traj_stats.py r1_confirmed r1_key r2_confirmed r2_key gen_files...
"""
import json, math, random, re, sys, collections

r1c, r1k, r2c, r2k = sys.argv[1:5]
gens = sys.argv[5:]
HARD = {"согласование", "управление", "несуществующее_слово", "неверное_слово",
        "орфография", "чужой_алфавит", "смысл"}
CYR = re.compile(r"[А-Яа-яЁё]+")
PAGE = 250

words = {}
for f in gens:
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        words[(r["arm"], r["id"])] = len(CYR.findall(r["text"]))

cells = {}  # (round, arm) -> {prompt id: hard count}
for rnd, cp, kp in (("r1", r1c, r1k), ("r2", r2c, r2k)):
    c = json.load(open(cp, encoding="utf-8"))
    k = json.load(open(kp, encoding="utf-8"))
    for tag, v in c.items():
        arm, pid = k[tag]["arm"], k[tag]["id"]
        h = sum(1 for e in v["confirmed"] if e["category"] in HARD)
        cells.setdefault((rnd, arm), {})[pid] = h

base = cells[("r1", "base")]

# Drift on the anchor: the same 40 base texts, round one versus round two.
anchor = cells.get(("r2", "base_rep"), {})
if anchor:
    a1 = sum(base[i] for i in anchor)
    a2 = sum(anchor.values())
    same = sum(1 for i in anchor if base[i] == anchor[i])
    diff = [anchor[i] - base[i] for i in anchor]
    print(f"ДРЕЙФ РАЗМЕТКИ на {len(anchor)} базовых текстах, размеченных дважды:")
    print(f"  раунд 1: {a1} ошибок   раунд 2: {a2} ошибок   ({100*(a2-a1)/max(a1,1):+.0f}%)")
    print(f"  совпало число ошибок в {same} текстах из {len(anchor)};  "
          f"средняя |разница| на текст {sum(abs(x) for x in diff)/len(diff):.2f}\n")


def arm_summary(rnd, arm):
    d = cells[(rnd, arm)]
    ids = sorted(i for i in d if i in base)
    w_arm = sum(words.get((arm if arm != "base_rep" else "base", i), 0) for i in ids)
    w_base = sum(words[("base", i)] for i in ids)
    e_arm = sum(d[i] for i in ids)
    e_base = sum(base[i] for i in ids)
    clean = sum(1 for i in ids if d[i] == 0)
    better = sum(1 for i in ids if d[i] < base[i])
    worse = sum(1 for i in ids if d[i] > base[i])
    n = better + worse
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(min(better, worse) + 1)) / 2 ** n) if n else 1.0
    rng = random.Random(5)
    boots = []
    for _ in range(10000):
        s = [rng.choice(ids) for _ in ids]
        tb = sum(base[i] for i in s)
        ta = sum(d[i] for i in s)
        if tb:
            boots.append((ta - tb) / tb)
    boots.sort()
    lo, hi = boots[int(.025 * len(boots))], boots[int(.975 * len(boots))]
    return {"ids": len(ids), "per_page": PAGE * e_arm / max(w_arm, 1),
            "base_per_page": PAGE * e_base / max(w_base, 1),
            "clean": clean / len(ids), "errors": e_arm, "base_errors": e_base,
            "better": better, "worse": worse, "p": p, "ci": (lo, hi)}


print(f"{'точка':22s} {'ошибок/стр':>10s} {'без ошибок':>11s} {'ошибок':>7s} "
      f"{'лучше/хуже':>11s} {'p':>6s}  {'95% интервал к базе':>22s}")
rows = [("база", "r1", "base"), ("5 мин (прогон 1)", "r1", "trained"),
        ("5 мин (прогон 2)", "r2", "t300"), ("20 мин", "r2", "t1200"), ("60 мин", "r2", "t3600")]
for name, rnd, arm in rows:
    if (rnd, arm) not in cells:
        continue
    s = arm_summary(rnd, arm)
    if arm == "base" and rnd == "r1":
        print(f"{name:22s} {s['per_page']:10.2f} {100*s['clean']:10.0f}% {s['errors']:7d} "
              f"{'—':>11s} {'—':>6s}  {'—':>22s}")
        continue
    print(f"{name:22s} {s['per_page']:10.2f} {100*s['clean']:10.0f}% {s['errors']:7d} "
          f"{s['better']:>5d}/{s['worse']:<5d} {s['p']:6.3f}  "
          f"{100*s['ci'][0]:+9.0f}% … {100*s['ci'][1]:+.0f}%")
print("\nошибок/стр — твёрдых ошибок на 250 русских слов; «лучше/хуже» — число промптов,"
      " где ошибок стало меньше/больше, чем у базы.")
