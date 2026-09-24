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

FP8 projections (--take fp8 or both). An AutoRound run with --fp8 also writes
FP8 rows (weight F8_E4M3, one scale per row). They replace the base's FP8 rows
of the same name; ninfer requires the row scale in BF16, so a scale in another
dtype is rounded to BF16 and its row re-encoded against it, and the report says
how far that moved the dequantised weights. --base can be a previous merge (e.g.
r1's) to keep its NVFP4 MLP and swap only the FP8 part.

Usage: merge_nvfp4.py --base UNSLOTH_NVFP4_DIR --ar AUTOROUND_DIR [--ar ...] --out DIR [--take nvfp4|fp8|both]
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
ap.add_argument("--take", choices=("nvfp4", "fp8", "both"), default="nvfp4")
a = ap.parse_args()

new, fp8 = {}, {}
for d in a.ar:
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as s:
            keys = set(s.keys())
            for k in keys:
                # an NVFP4 matrix is one with packed codes; FP8 rows of MLP 56-63
                # also carry a weight_scale and must not be picked up here
                if a.take in ("nvfp4", "both") and ".mlp." in k and k.rsplit(".", 1)[1] in SUFFIX \
                        and k.rsplit(".", 1)[0] + ".weight_packed" in keys:
                    new[k] = s.get_tensor(k)
                if a.take in ("fp8", "both") and k.endswith(".weight") and k[:-7] + ".weight_scale" in keys:
                    w = s.get_tensor(k)
                    if w.dtype == torch.float8_e4m3fn:
                        fp8[k] = (w, s.get_tensor(k[:-7] + ".weight_scale"))
if not new and not fp8:
    raise SystemExit(f"nothing to take ({a.take}) in " + ", ".join(a.ar))


def fp8_rows(w, scale):
    """Encode (weight, per-row scale) as ninfer wants it: F8_E4M3 rows, BF16 scale [N, 1]."""
    scale = scale.reshape(-1, 1)
    if scale.dtype == torch.bfloat16:
        return w, scale, 0.0
    exact = w.float() * scale.float()
    s16 = scale.to(torch.bfloat16)
    q = (exact / s16.float()).clamp(-448, 448).to(torch.float8_e4m3fn)
    moved = (torch.linalg.norm(q.float() * s16.float() - exact) / torch.linalg.norm(exact)).item()
    return q, s16, moved

src = os.path.join(a.base, "model.safetensors")
out, report = {}, {}
with safe_open(src, "pt") as s:
    meta = s.metadata()
    keys = list(s.keys())
    missing = sorted(set(new) - set(keys)) + sorted(k for k in fp8 if k not in keys)
    if missing:
        raise SystemExit(f"{len(missing)} new tensors have no place in the base, e.g. {missing[:3]}")
    if fp8:
        # every FP8 projection of the layers the run covered must be replaced;
        # lm_head is outside the decoder blocks and never tuned
        layer_of = lambda k: int(k.split(".layers.")[1].split(".")[0])
        top = max(layer_of(k) for k in fp8)
        expected = {k for k in keys if k.endswith(".weight") and ".layers." in k and layer_of(k) <= top
                    and s.get_slice(k).get_dtype() == "F8_E4M3"}
        if expected != set(fp8):
            gap = sorted(expected - set(fp8))[:5], sorted(set(fp8) - expected)[:5]
            raise SystemExit(f"FP8 rows: expected {len(expected)}, got {len(fp8)}; missing {gap[0]}, unexpected {gap[1]}")
    fp8_done = set()
    for k in keys:
        t = s.get_tensor(k)
        if k in fp8 or (k.endswith(".weight_scale") and k[:-13] + ".weight" in fp8):
            wk = k if k in fp8 else k[:-13] + ".weight"
            if wk not in fp8_done:
                bw, bs = s.get_tensor(wk), s.get_tensor(wk[:-7] + ".weight_scale")
                w, sc, moved = fp8_rows(*fp8[wk])
                if w.shape != bw.shape or bw.dtype != torch.float8_e4m3fn or sc.numel() != bs.numel():
                    raise SystemExit(f"{wk}: base {bw.dtype}{tuple(bw.shape)}/{tuple(bs.shape)} "
                                     f"vs new {w.dtype}{tuple(w.shape)}/{tuple(sc.shape)}")
                sc = sc.reshape(bs.shape).to(bs.dtype)
                base_deq = bw.float() * bs.float().reshape(-1, 1)
                new_deq = w.float() * sc.float().reshape(-1, 1)
                r = report.setdefault(wk[:-7], {})
                r["fp8_codes_changed"] = round(((w.view(torch.uint8) != bw.view(torch.uint8)).sum() / w.numel()).item(), 4)
                r["fp8_rel_change"] = round((torch.linalg.norm(new_deq - base_deq) / torch.linalg.norm(base_deq)).item(), 5)
                r["fp8_scale_reencode"] = round(moved, 6)
                out[wk], out[wk[:-7] + ".weight_scale"] = w, sc
                fp8_done.add(wk)
            continue
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
               "replaced_tensors": len(new) + 2 * len(fp8), "take": a.take, "act_scale": a.act_scale, "weights": a.weights, "matrices": dict(layers)}, f, indent=1)
cc = [r["codes_changed"] for _, r in layers if "codes_changed" in r]
ig = [r["input_global_scale_ratio"] for _, r in layers if "input_global_scale_ratio" in r]
print(f"replaced {len(new)} NVFP4 tensors and {len(fp8)} FP8 matrices ({len(layers)} matrices in all)")
fc = [r["fp8_codes_changed"] for _, r in layers if "fp8_codes_changed" in r]
fr = [r["fp8_rel_change"] for _, r in layers if "fp8_rel_change" in r]
fm = [r["fp8_scale_reencode"] for _, r in layers if "fp8_scale_reencode" in r]
if fc:
    print(f"FP8 codes changed: median {statistics.median(fc):.1%}; weights moved: median {statistics.median(fr):.3%}, "
          f"max {max(fr):.3%}; BF16 scale re-encoding moved them by at most {max(fm):.4%}")
if cc:
    print(f"FP4 codes changed: median {statistics.median(cc):.1%}, min {min(cc):.1%}, max {max(cc):.1%}")
if ig:
    print(f"activation scale new/prod: median {statistics.median(ig):.3f}, min {min(ig):.3f}, max {max(ig):.3f}; "
          f"kept: {a.act_scale}, {sum(x > 1 for x in ig)} of {len(ig)} matrices would clip harder than production")
