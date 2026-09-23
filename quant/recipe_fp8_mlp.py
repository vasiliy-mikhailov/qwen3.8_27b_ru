"""ninfer recipe: production NVFP4 layout, with chosen MLP layers promoted to FP8.

Production keeps the MLP of layers 0-55 in NVFP4 (W4A4) and of layers 56-63 in
FP8. This recipe starts from the official qwen3_8_27b_nvfp4 assignment and
re-encodes the MLP of the layers listed in the environment variable
FP8_MLP_LAYERS (e.g. "0-55" or "0-3,20,48-55") as FP8 rows straight from bf16,
exactly like production's own FP8 layers (per-row max-abs, 8-bit activations).

Two uses:
- "0-55": every MLP in 8 bits, the ceiling of what bit allocation can buy on
  this engine; it does not fit the 262k-context serving budget, but it fits a
  perplexity run or a short-context server;
- a handful of layers: the affordable variant, about +0.11 GiB per layer.

Run: python -m tools.convert --recipe /w/quant/recipe_fp8_mlp.py ... with
FP8_MLP_LAYERS set, from the ninfer checkout.
"""
import os

from tools.convert.methods import fp8_row_maxabs
from tools.convert.official_recipes import FP8, qwen3_8_27b_nvfp4


def _layers(spec):
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def configure(model, recipe, sources):
    qwen3_8_27b_nvfp4(model, recipe, sources)
    layers = _layers(os.environ.get("FP8_MLP_LAYERS", ""))
    if not layers:
        raise ValueError("set FP8_MLP_LAYERS, e.g. 0-55")
    base = sources["base"]
    promoted = 0
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or "/mlp/" not in name or not parameter.projection:
            continue
        if int(name.split("/")[2]) not in layers:
            continue
        recipe.assign(name, format=FP8, method=fp8_row_maxabs, source=model.source(name, base),
                      activation_policy="AllowA8")
        promoted += 1
    print(f"recipe_fp8_mlp: {promoted} MLP matrices promoted to FP8 in layers {sorted(layers)}", flush=True)
