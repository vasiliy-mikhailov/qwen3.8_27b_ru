#!/usr/bin/env python3
"""Find words that mix Cyrillic and Latin letters.

In Russian technical text a homoglyph substitution (Latin o for Cyrillic о,
Latin y for у, Latin e for е, a for а, c for с, p for р, x for х) is invisible
to the eye but breaks search, spellcheck and any downstream tokenisation. It is
also mechanically detectable with high precision, unlike grammar errors.

Words that legitimately mix scripts are excluded: anything attached to a known
technical token (identifiers, paths, file extensions, Kafka topics) and any word
where the Latin part is itself a standalone English word.
"""
import re, sys, json, collections

CYR = "Ѐ-ӿ"
WORD = re.compile(rf"[{CYR}A-Za-z][{CYR}A-Za-z0-9_]*")
HOMOGLYPH = set("oyeacpxABCEHKMOPTXaeopcyxn")


def mixed_words(text):
    out = []
    for m in WORD.finditer(text or ""):
        w = m.group(0)
        cyr = sum(1 for ch in w if "Ѐ" <= ch <= "ӿ")
        lat = sum(1 for ch in w if ch.isascii() and ch.isalpha())
        if not (cyr and lat):
            continue
        # A single Latin letter inside an otherwise Cyrillic word, and that
        # letter has a Cyrillic look-alike: that is the homoglyph case.
        if lat == 1 and cyr >= 3:
            ch = next(c for c in w if c.isascii() and c.isalpha())
            if ch in HOMOGLYPH:
                out.append((w, ch, m.start()))
    return out


if __name__ == "__main__":
    texts = json.load(open(sys.argv[1], encoding="utf-8"))
    hits = collections.Counter()
    per_text = 0
    for t in texts:
        found = mixed_words(t)
        if found:
            per_text += 1
        for w, ch, _ in found:
            hits[(w, ch)] += 1
    print(f"текстов: {len(texts)}, с подменой букв: {per_text}")
    print(f"всего случаев: {sum(hits.values())}, уникальных написаний: {len(hits)}")
    for (w, ch), n in hits.most_common(25):
        print(f"  {n:3d}×  {w!r}   латинская {ch!r}")
