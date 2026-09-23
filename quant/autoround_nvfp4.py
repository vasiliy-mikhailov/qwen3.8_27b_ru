#!/usr/bin/env python3
"""Re-quantise the NVFP4 part of Qwen3.8-27B with AutoRound on Russian data.

The production checkpoint (unsloth/Qwen3.8-27B-NVFP4) keeps attention, the
linear-attention projections, lm_head and the MLP of layers 56-63 in FP8, and
puts the MLP of layers 0-55 in NVFP4 with round-to-nearest weights and static
activation scales from its own calibration. Only those 168 NVFP4 matrices lose
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

# llm-compressor tells AutoRound to skip only target-matching layers that ended
# up without a scheme; every other Linear in the block falls under AutoRound's
# default scheme. With scheme="NVFP4" that made it tune attention and the
# linear-attention projections as W4A4 too (8/8 and 7/7 layers per block in run
# r1), although production keeps them in FP8, so the MLP rounding was fitted
# against a context far noisier than the one it runs in. Here every Linear
# without a quantization scheme is ignored and stays bf16, which is close to FP8.
# AutoRound matches these names as substrings of the wrapped block's names.


def _unquantized_linears(self, block):
    return [n for n, m in block.named_modules()
            if isinstance(m, torch.nn.Linear) and getattr(m, "quantization_scheme", None) is None]


AutoRoundModifier.get_unquantized_layer_names = _unquantized_linears

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
               "iters": a.iters, "batch": a.batch, "grad_acc": a.grad_acc, "lr": a.lr, "truncate": a.truncate,
               "calib": os.path.abspath(a.calib), "seconds": round(took)}, f, indent=1)
print(f"done in {took/60:.1f} min -> {a.out}")
