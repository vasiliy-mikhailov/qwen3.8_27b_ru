#!/usr/bin/env python3
"""Automatic Russian-defect metrics for tracking a fine-tune across checkpoints.

Three signals, all mechanical, chosen because they match defects that were
actually found by reading generated text:

  oov        Cyrillic words no Russian morphology knows. Catches the corrupted
             words that dominate the lexical category: Библиотеарь, подхата,
             запажем, скольздя, изборозженное, фиксиционные.
  homoglyph  words mixing Cyrillic with a look-alike Latin letter (аврo, темy).
             Invisible when read, but breaks search and spellcheck downstream.
  foreign    CJK characters sitting inside or against a Russian word (某种, 琥珀).

Reading found roughly one error per 150 Russian words, but reading costs hours
per data point and resolves no better than +-50%, which cannot show a trajectory
over N training cycles. These three resolve to a single word and run in seconds.

They cover the mechanical part of what reading finds, not the semantic part:
agreement across a long noun phrase and contradictions across paragraphs stay
invisible here. RuBLiMP covers the first; nothing automatic covers the second.

Report a floor alongside every measurement: the same metrics on human reference
Russian. Technical jargon and loanwords are out-of-dictionary too, so only the
distance from that floor means anything.
"""
import json, re, sys, collections

CYR_RE = re.compile(r"[А-Яа-яЁё][А-Яа-яЁё-]*")
MIXED_RE = re.compile(r"[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z0-9_]*")
CJK_RE = re.compile(r"[一-鿿]+")
HOMOGLYPH = set("oyeacpxOAEHKMPTXBCn")


def load_analyzer():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()


def measure(text, morph):
    words = CYR_RE.findall(text or "")
    oov, oov_caps = [], []
    for w in words:
        if len(w) < 4:
            continue
        if any(p.is_known for p in morph.parse(w.lower())):
            continue
        # Capitalised unknowns are overwhelmingly proper names: on human
        # reference Russian they alone produce 53 per 1000 words, which would
        # bury the signal. They are counted separately, not mixed in.
        (oov_caps if w[0].isupper() else oov).append(w)
    homo = []
    for m in MIXED_RE.finditer(text or ""):
        w = m.group(0)
        cyr = sum(1 for c in w if "А" <= c <= "я" or c in "Ёё")
        lat = [c for c in w if c.isascii() and c.isalpha()]
        if cyr >= 3 and len(lat) == 1 and lat[0] in HOMOGLYPH:
            homo.append(w)
    foreign = [m.group(0) for m in CJK_RE.finditer(text or "")
               if re.search(r"[А-Яа-яЁё]", (text[max(0, m.start() - 12):m.end() + 12]))]
    return {"ru_words": len(words), "oov": oov, "oov_caps": oov_caps,
            "homoglyph": homo, "foreign": foreign}


def report(name, texts, morph):
    agg = collections.Counter()
    samples = {k: collections.Counter() for k in
               ("oov", "oov_caps", "homoglyph", "foreign")}
    for t in texts:
        r = measure(t, morph)
        agg["ru_words"] += r["ru_words"]
        for k in ("oov", "oov_caps", "homoglyph", "foreign"):
            agg[k] += len(r[k])
            samples[k].update(r[k])
    w = max(agg["ru_words"], 1)
    print(f"{name:26s} рус.слов {agg['ru_words']:7d} | "
          f"oov(строчные) {agg['oov']:4d} ({1000*agg['oov']/w:5.2f}‰) | "
          f"oov(с большой) {agg['oov_caps']:5d} | "
          f"гомоглифы {agg['homoglyph']:3d} | CJK {agg['foreign']:3d}")
    return agg, samples


if __name__ == "__main__":
    morph = load_analyzer()
    for spec in sys.argv[1:]:
        name, path = spec.split(":", 1)
        if path.endswith(".jsonl"):
            texts = []
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                texts.append(r.get("content") or "")
        elif path.endswith(".json"):
            texts = json.load(open(path, encoding="utf-8"))
        else:
            texts = [open(path, encoding="utf-8").read()]
        agg, samples = report(name, texts, morph)
        top = samples["oov"].most_common(8)
        if top:
            print("       частые oov:", ", ".join(f"{w}×{n}" for w, n in top))
