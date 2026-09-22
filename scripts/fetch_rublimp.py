#!/usr/bin/env python3
"""Download RuBLiMP (45 paradigms of Russian minimal pairs) into data/rublimp/.

RuBLiMP is Apache-2.0 but is fetched rather than vendored, so the copy here always
matches upstream. https://huggingface.co/datasets/RussianNLP/rublimp
"""
import json, os, sys, urllib.request

DEST = sys.argv[1] if len(sys.argv) > 1 else "data/rublimp"
API = "https://huggingface.co/api/datasets/RussianNLP/rublimp"
RESOLVE = "https://huggingface.co/datasets/RussianNLP/rublimp/resolve/main/"

os.makedirs(DEST, exist_ok=True)
meta = json.load(urllib.request.urlopen(API, timeout=60))
files = [s["rfilename"] for s in meta["siblings"] if s["rfilename"].endswith(".parquet")]
for f in files:
    paradigm = f.split("/")[0]
    out = os.path.join(DEST, paradigm + ".parquet")
    if os.path.exists(out):
        continue
    urllib.request.urlretrieve(RESOLVE + f, out)
    print("fetched", paradigm, flush=True)
print(f"{len(files)} paradigms in {DEST}")
