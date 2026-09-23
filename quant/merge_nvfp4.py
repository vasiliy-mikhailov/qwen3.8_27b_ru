#!/usr/bin/env python3
"""Put re-quantised NVFP4 MLP matrices into the production checkpoint layout.

ninfer's qwen3_8_27b_nvfp4 recipe imports every projection pre-encoded from one
compressed-tensors store, FP8 and NVFP4 alike. This copies the production store
(unsloth/Qwen3.8-27B-NVFP4) and swaps in only the NVFP4 tensors that
autoround_nvfp4.py produced, after checking that names, shapes and dtypes agree.
Everything else, FP8 matrices, norms, the vision tower, MTP, is byte-identical
to production, so a difference between the two artifacts is the MLP rounding
and its activation scales and nothing else.

Prints how much of each matrix changed: the share of FP4 codes that differ and
the ratio of the new static activation scale to the production one.

Static activation scales. NVFP4 inputs carry one global divisor per matrix,
448 * 6 / amax over the calibration set; an input larger than that amax is
clipped at inference. AutoRound's amax comes from the Russian calibration set
only and is often smaller than production's, so by default (--act-scale safe)
each matrix keeps whichever divisor is smaller, i.e. the larger amax. The gain
from a tighter divisor is small anyway: it only refines FP8 group scales that
would otherwise fall into E4M3 subnormals.

Clipping is real in ninfer: each 16-value group gets the E4M3 scale
divisor * max_abs / 6 with saturation at 448, so inputs above the static amax
are cut to it. --weights prod keeps production's FP4 weights and takes only the
activation scales, which isolates what the scales alone are worth.

Usage: merge_nvfp4.py --base UNSLOTH_NVFP4_DIR --ar AUTOROUND_DIR [--ar ...] --out DIR
"""
import argparse, glob, json, os, shutil, statistics

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SUFFIX = ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--ar", required=True, action="append")
ap.add_argument("--out", required=True)
ap.add_argument("--act-scale", choices=("safe", "new", "prod"), default="safe")
ap.add_argument("--weights", choices=("new", "prod"), default="new")
a = ap.parse_args()

new = {}
for d in a.ar:
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if ".mlp." in k and k.rsplit(".", 1)[1] in SUFFIX:
                    new[k] = s.get_tensor(k)
if not new:
    raise SystemExit("no NVFP4 MLP tensors found in " + ", ".join(a.ar))

src = os.path.join(a.base, "model.safetensors")
out, report = {}, {}
with safe_open(src, "pt") as s:
    meta = s.metadata()
    keys = list(s.keys())
    missing = sorted(set(new) - set(keys))
    if missing:
        raise SystemExit(f"{len(missing)} new tensors have no place in the base, e.g. {missing[:3]}")
    for k in keys:
        t = s.get_tensor(k)
        if k in new:
            n = new[k]
            if n.shape != t.shape or n.dtype != t.dtype:
                raise SystemExit(f"{k}: base {t.dtype}{tuple(t.shape)} vs new {n.dtype}{tuple(n.shape)}")
            prefix, suffix = k.rsplit(".", 1)
            r = report.setdefault(prefix, {})
            if suffix == "weight_packed":
                diff = ((n & 15) != (t & 15)).sum() + ((n >> 4) != (t >> 4)).sum()
                r["codes_changed"] = round(diff.item() / (2 * n.numel()), 4)
            elif suffix in ("weight_global_scale", "input_global_scale"):
                r[suffix + "_ratio"] = round(n.item() / t.item(), 4)
            if suffix == "input_global_scale":
                if a.act_scale != "new":
                    n = t if a.act_scale == "prod" else torch.minimum(n, t)
            elif a.weights == "prod":
                n = t
            t = n
        out[k] = t

os.makedirs(a.out, exist_ok=True)
save_file(out, os.path.join(a.out, "model.safetensors"), metadata=meta)
for f in os.listdir(a.base):
    p = os.path.join(a.base, f)
    if f == "model.safetensors" or f.endswith(".part") or os.path.isdir(p):
        continue
    shutil.copy2(p, os.path.join(a.out, f))

layers = sorted(report.items(), key=lambda kv: [int(x) if x.isdigit() else x for x in kv[0].split(".")])
with open(os.path.join(a.out, "merge_report.json"), "w") as f:
    json.dump({"base": os.path.abspath(a.base), "autoround": [os.path.abspath(d) for d in a.ar],
               "replaced_tensors": len(new), "act_scale": a.act_scale, "weights": a.weights, "matrices": dict(layers)}, f, indent=1)
cc = [r["codes_changed"] for _, r in layers if "codes_changed" in r]
ig = [r["input_global_scale_ratio"] for _, r in layers if "input_global_scale_ratio" in r]
print(f"replaced {len(new)} tensors in {len(layers)} matrices")
if cc:
    print(f"FP4 codes changed: median {statistics.median(cc):.1%}, min {min(cc):.1%}, max {max(cc):.1%}")
if ig:
    print(f"activation scale new/prod: median {statistics.median(ig):.3f}, min {min(ig):.3f}, max {max(ig):.3f}; "
          f"kept: {a.act_scale}, {sum(x > 1 for x in ig)} of {len(ig)} matrices would clip harder than production")
