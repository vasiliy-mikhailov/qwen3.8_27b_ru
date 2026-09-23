#!/usr/bin/env python3
"""Re-quantise the NVFP4 part of Qwen3.8-27B with AutoRound on Russian data.

The production checkpoint (unsloth/Qwen3.8-27B-NVFP4) keeps attention, the
linear-attention projections, lm_head and the MLP of layers 56-63 in FP8, and
puts the MLP of layers 0-55 in NVFP4 (GPTQ-style, per its actorder: static) with
static activation scales from its own calibration. Only those 168 NVFP4 matrices lose
real precision, so only they are redone here:

- weights: AutoRound learns the rounding and clipping of every block so that
  the block's output with W4A4 matches the bf16 block on the calibration set;
- activations: AutoRound measures a static global scale for each NVFP4 input on
  the same set; merge_nvfp4.py decides whether to keep it (by default only
  where it clips less than production's).

The FP8 tensors are left to be copied from the production checkpoint by
merge_nvfp4.py, which keeps everything but the MLP comparable to production.

Two local patches make a 27B block fit a 32 GB card: auto_round's FP4 cast gets
an identity-gradient autograd function (bit-identical forward and gradients,
checked), and AutoRound's gradient accumulation is switched on, which
llm-compressor does not expose, so two sequences per step still give the
default effective batch of 8.

Usage (inside qwen-quant, GPU free):
  autoround_nvfp4.py --model BF16_DIR --calib calib_ru.jsonl --out OUT_DIR
  --truncate N builds only the first N decoder layers, for a pilot.
"""
import argparse, json, os, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--calib", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="0-55", help="decoder layers whose MLP is NVFP4")
ap.add_argument("--nsamples", type=int, default=256)
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--grad-acc", type=int, default=4, help="effective batch = batch * grad-acc (AutoRound's default is 8)")
ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--truncate", type=int, default=None, help="keep only the first N layers (pilot)")
ap.add_argument("--context", choices=("bf16", "nvfp4", "mxfp4"), default="bf16",
                help="how the non-target Linears of a block are simulated while the MLP is tuned: "
                     "bf16 (close to production's FP8; run r2), nvfp4 (AutoRound's default; run r1), "
                     "mxfp4 (coarser still; run r3)")
a = ap.parse_args()

lo, hi = map(int, a.layers.split("-"))
layers = list(range(lo, hi + 1))
if a.truncate:
    layers = [i for i in layers if i < a.truncate]
target = r"re:.*language_model\.layers\.(%s)\.mlp\.(gate|up|down)_proj$" % "|".join(map(str, layers))

from datasets import Dataset
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
import auto_round.data_type.nvfp as nvfp

# auto_round's cast_to_fp4 builds the E2M1 grid from round_ste pieces and masks,
# so autograd keeps some fifteen full-size fp32 intermediates per call: for one
# block's three 17408x5120 matrices plus their W4A4 inputs that alone overflows
# 32 GB. Every piece is a straight-through estimator with slope 1 and the input
# is already clipped to [-6, 6], so the whole function's gradient is the
# identity. Same forward, identity backward, nothing saved.
_cast_to_fp4 = nvfp.cast_to_fp4


class _CastFP4STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        with torch.no_grad():
            return _cast_to_fp4(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


nvfp.cast_to_fp4 = _CastFP4STE.apply

_apply_autoround = AutoRoundModifier.apply_autoround


def _apply_with_memory_log(self, state, modules):
    """Logs GPU memory around every block, and on request what occupies it."""
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated() / 2**30
    if os.environ.get("AR_MEMDEBUG"):
        import gc
        big = sorted(((o.numel() * o.element_size(), tuple(o.shape), o.dtype)
                      for o in gc.get_objects() if torch.is_tensor(o) and o.is_cuda), reverse=True)
        print(f"[mem] {len(big)} cuda tensors, largest:", [(round(b / 2**20), s, str(d)) for b, s, d in big[:12]], flush=True)
    t = time.time()
    _apply_autoround(self, state, modules)
    print(f"[mem] block: before {before:.1f} GiB, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
          f"{time.time() - t:.0f} s", flush=True)


AutoRoundModifier.apply_autoround = _apply_with_memory_log

# On 32 GB a block with W4A4 simulation fits two 2k-token sequences at a time;
# llm-compressor does not expose AutoRound's gradient accumulation, so it is
# added to the constructor call here to keep the effective batch at 8.
import llmcompressor.modifiers.autoround.base as _ar_base

_AutoRound = _ar_base.AutoRound


def _auto_round_with_accumulation(*args, **kwargs):
    kwargs.setdefault("gradient_accumulate_steps", a.grad_acc)
    return _AutoRound(*args, **kwargs)


_ar_base.AutoRound = _auto_round_with_accumulation

# The context the MLP rounding is tuned in. llm-compressor tells AutoRound to
# skip only target-matching layers that ended up without a scheme; every other
# Linear in the block falls under AutoRound's default scheme. With
# scheme="NVFP4" that makes attention and the linear-attention projections W4A4
# during tuning (8/8 and 7/7 layers per block), although production keeps them
# in FP8 -- that was run r1, and its MLP turned out to make fewer errors, not
# more. --context picks the context deliberately:
#   bf16  - every Linear without a scheme is ignored and stays bf16, close to
#           production's FP8 (run r2);
#   nvfp4 - AutoRound's default, everything W4A4 (run r1);
#   mxfp4 - the non-target Linears are simulated as MXFP4 (groups of 32,
#           power-of-two scales) at round-to-nearest, noisier than r1's
#           tuned NVFP4 context (run r3).
# Only the NVFP4 MLP tensors are kept by merge_nvfp4.py in every case. AutoRound
# matches ignore names as substrings of the wrapped block's names.


def _unquantized_linears(self, block):
    return [n for n, m in block.named_modules()
            if isinstance(m, torch.nn.Linear) and getattr(m, "quantization_scheme", None) is None]


if a.context == "bf16":
    AutoRoundModifier.get_unquantized_layer_names = _unquantized_linears
elif a.context == "mxfp4":
    from dataclasses import asdict
    from auto_round.schemes import PRESET_SCHEMES as _AR_PRESETS
    from auto_round.data_type.register import QUANT_FUNC_WITH_DTYPE as _AR_DTYPES

    # The context layers are noise, not something to tune: their MXFP4
    # quantiser runs without autograd and passes gradients straight through to
    # its input, so they stay at round-to-nearest (coarser than r1, whose context
    # layers AutoRound also tuned) and cost no memory for the backward pass.
    def _frozen(fn):
        def run(tensor, *args, **kwargs):
            with torch.no_grad():
                out = fn(tensor, *args, **kwargs)
            q = out[0]
            q = tensor + (q - tensor).detach() if tensor.requires_grad else q
            return (q, *out[1:])
        return run

    for _name in [n for n in _AR_DTYPES if n.startswith("mx_fp")]:
        _AR_DTYPES[_name] = _frozen(_AR_DTYPES[_name])

    _build_layer_config = AutoRoundModifier._build_layer_config_for_autoround

    def _layer_config_with_coarse_context(self, wrapped_model):
        config = _build_layer_config(self, wrapped_model) or {}
        default = self._quant_scheme_to_autoround_config(self._get_default_quant_scheme())
        coarse = {k: v for k, v in asdict(_AR_PRESETS["MXFP4"]).items() if k in default}
        # llm-compressor's own target matcher compares the full-path regex with
        # names relative to the block and finds nothing, so targets are
        # recognised by the quantization scheme llm-compressor attached to them.
        linears = [(n, m) for n, m in wrapped_model.named_modules() if isinstance(m, torch.nn.Linear)]
        has_target = any(getattr(m, "quantization_scheme", None) is not None for _, m in linears)
        coarse_names = [n for n, m in linears
                        if getattr(m, "quantization_scheme", None) is None] if has_target else []
        for name in coarse_names:
            config[name] = dict(coarse)
        print(f"[ctx] {len(coarse_names)} non-target Linears simulated as MXFP4", flush=True)
        return config

    AutoRoundModifier._build_layer_config_for_autoround = _layer_config_with_coarse_context

    # A block with no NVFP4 target (56-63) has nothing left to tune once its
    # context is frozen, so it is skipped whole, as in the bf16 context.
    def _skip_blocks_without_targets(self, block):
        if any(isinstance(m, torch.nn.Linear) and getattr(m, "quantization_scheme", None) is not None
               for m in block.modules()):
            return []
        return _unquantized_linears(self, block)

    AutoRoundModifier.get_unquantized_layer_names = _skip_blocks_without_targets

rows = [json.loads(l) for l in open(a.calib, encoding="utf-8")][: a.nsamples]
seqlen = len(rows[0]["input_ids"])
ds = Dataset.from_dict({
    "input_ids": [r["input_ids"] for r in rows],
    "attention_mask": [[1] * len(r["input_ids"]) for r in rows],
})

config = AutoConfig.from_pretrained(a.model)
if a.truncate:
    config.text_config.num_hidden_layers = a.truncate
    config.text_config.layer_types = config.text_config.layer_types[: a.truncate]
model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, config=config, dtype=torch.bfloat16)

recipe = AutoRoundModifier(
    targets=[target],
    scheme="NVFP4",
    ignore=[],
    iters=a.iters,
    batch_size=a.batch,
    lr=a.lr,
    enable_torch_compile=False,
)
t0 = time.time()
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=seqlen,
    num_calibration_samples=len(rows),
    shuffle_calibration_samples=False,
    sequential_targets=["Qwen3_5DecoderLayer"],
)
took = time.time() - t0
os.makedirs(a.out, exist_ok=True)
model.save_pretrained(a.out, save_compressed=True)
with open(os.path.join(a.out, "autoround_run.json"), "w") as f:
    json.dump({"layers": [layers[0], layers[-1]], "nsamples": len(rows), "seqlen": seqlen,
               "iters": a.iters, "batch": a.batch, "grad_acc": a.grad_acc, "context": a.context, "lr": a.lr, "truncate": a.truncate,
               "calib": os.path.abspath(a.calib), "seconds": round(took)}, f, indent=1)
print(f"done in {took/60:.1f} min -> {a.out}")
