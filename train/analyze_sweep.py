#!/usr/bin/env python3
"""Read a sweep directory and judge convergence the way the raw numbers cannot.

Two corrections over reading summary.json directly:

1. Per-step training loss depends heavily on which text the step drew, so a
   trailing moving average is used instead of first-versus-last steps.

2. The base is a post-trained chat model. Any training on raw text shifts it
   toward "continue the document", which makes raw text cheaper in EVERY
   language. So an absolute drop on a Russian target proves little. What matters
   is how much more the target drops than the guards do: the Russian-specific
   gain.
"""
import json, os, sys, math

S = sys.argv[1]
summ = json.load(open(os.path.join(S, "summary.json")))
TARGETS = ["russian_code_register", "russian_code_comments"]
GUARDS = ["ninfer_code", "english_reference", "english_long_form", "chinese_reference"]


def smooth(xs, k=10):
    out, acc = [], []
    for x in xs:
        acc.append(x)
        if len(acc) > k:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


print("база PPL:", "  ".join(f"{d}={v:.4f}" for d, v in sorted(summ["base_ppl"].items())))
print()
print(f"{'конфиг':15s} {'шагов':>5s} {'млн ток':>7s} {'loss МА10':>15s} "
      f"{'Δ регистр':>10s} {'Δ коммент':>10s} {'Δ стражи':>9s} {'рус.выигрыш':>11s} {'Δ код':>8s}")
for r in summ["runs"]:
    rows = [json.loads(l) for l in open(os.path.join(S, r["tag"] + ".loss.jsonl"))]
    ma = smooth([x["loss"] for x in rows])
    d = r["eval_delta_nll"]
    gs = [g for g in GUARDS if g in d]
    guard = sum(d[g] for g in gs) / len(gs)
    ts = [t for t in TARGETS if t in d]
    tgt = sum(d[t] for t in ts) / len(ts)
    print(f"{r['tag']:15s} {r['steps']:5d} {r['tokens_seen']/1e6:7.2f} "
          f"{ma[min(9, len(ma)-1)]:6.3f}→{ma[-1]:<6.3f}  "
          f"{d.get(TARGETS[0], float('nan')):+10.4f} {d.get(TARGETS[1], float('nan')):+10.4f} "
          f"{guard:+9.4f} {tgt - guard:+11.4f} {d.get('ninfer_code', 0):+8.4f}")

print("\nТраектории целевых доменов (PPL по ходу обучения):")
for r in summ["runs"]:
    p = os.path.join(S, r["tag"] + ".trajectory.jsonl")
    if not os.path.exists(p):
        continue
    pts = [json.loads(l) for l in open(p)]
    end = {t: math.log(r["eval_ppl"][t]) for t in TARGETS if t in r["eval_ppl"]}
    pts.append({"train_seconds": r["train_seconds"], "tokens": r["tokens_seen"], "nll": end})
    for t in TARGETS:
        if t not in pts[0]["nll"]:
            continue
        seq = "  ".join(f"{p['train_seconds']:>4.0f}с:{math.exp(p['nll'][t]):.4f}" for p in pts)
        print(f"  {r['tag']:15s} {t:22s} {seq}")
print("\nрус.выигрыш = средний Δnll целей минус средний Δnll стражей; отрицательный — хорошо.")
print("Δ код > 0 — регрессия на коде.")
