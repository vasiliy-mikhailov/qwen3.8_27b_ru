#!/usr/bin/env python3
"""Short timed QLoRA runs to find settings that converge, before any long run.

Each configuration trains for a fixed wall-clock budget (5 minutes by default)
and is judged on three things: whether the training loss falls steadily,
whether held-out Russian code-register text gets cheaper to predict, and
whether the guards (code, English, Chinese) hold still.

The 27B model is loaded ONCE. Loading takes minutes and would otherwise eat
most of a five-minute slot. Between configurations the LoRA weights are reset
to their initial state -- A re-drawn, B zeroed -- which makes the adapted model
exactly the base model again. That is verified, not assumed: right after every
reset the target domain is scored and must match the base score.

Rank is fixed within one process, since changing it means rebuilding adapters.
Learning rate and effective batch are what this sweeps.

Packing is done here rather than by the trainer. unsloth refuses to pack for
hybrid linear-attention models, and without packing a step is one random
paragraph -- the first sweep spent five minutes on ~500 short documents.
Documents are joined with <|endoftext|> and cut into full seq_len blocks, as in
ordinary pretraining. The linear-attention state does carry across a document
boundary inside a block; for continued pretraining that is the usual trade.
Every block is exactly seq_len tokens, so tokens seen is exact, not estimated.
"""
import argparse, gc, json, math, os, time

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--train-files", nargs="+", required=True)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lrs", type=float, nargs="+", default=[5e-5, 1e-4, 2e-4, 4e-4])
    p.add_argument("--accum", type=int, nargs="+", default=[4])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--train-seconds", type=int, default=300)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--eval-tokens", type=int, default=16384)
    p.add_argument("--probe-every", type=int, default=100,
                   help="training seconds between target-only probes")
    p.add_argument("--no-offload-embedding", action="store_true",
                   help="keep embeddings on the GPU; faster if it fits")
    p.add_argument("--targets", nargs="+",
                   default=["russian_code_register", "russian_code_comments"])
    return p.parse_args()


def load_eval_sets(eval_dir, tok, seq_len, budget):
    sets = {}
    for fn in sorted(os.listdir(eval_dir)):
        if not fn.endswith(".txt"):
            continue
        ids = tok(open(os.path.join(eval_dir, fn), encoding="utf-8").read(),
                  add_special_tokens=False)["input_ids"]
        wins = [torch.tensor(ids[i:i + seq_len]) for i in range(0, min(len(ids), budget), seq_len)
                if len(ids[i:i + seq_len]) >= 128]
        sets[fn[:-4]] = wins
    return sets


@torch.no_grad()
def score(model, sets, only=None):
    was_training = model.training
    model.eval()
    out = {}
    for d, wins in sets.items():
        if only and d not in only:
            continue
        nll = n = 0
        for w in wins:
            ids = w.unsqueeze(0).to(model.device)
            loss = model(input_ids=ids, labels=ids).loss.item()
            k = ids.shape[1] - 1
            nll += loss * k
            n += k
        out[d] = nll / max(n, 1)
    if was_training:
        model.train()
    return out


def lora_modules(model):
    from peft.tuners.lora import LoraLayer
    return [m for m in model.modules() if isinstance(m, LoraLayer)]


def reset_lora(mods, seed):
    """Put every adapter back to peft's initial state: A kaiming, B zero."""
    torch.manual_seed(seed)
    for m in mods:
        for name in m.lora_A.keys():
            torch.nn.init.kaiming_uniform_(m.lora_A[name].weight, a=math.sqrt(5))
            torch.nn.init.zeros_(m.lora_B[name].weight)


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    from unsloth import FastModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    from transformers import TrainerCallback

    t_load = time.time()
    model, tokenizer = FastModel.from_pretrained(
        model_name=a.base, max_seq_length=a.seq_len, load_in_4bit=True,
        full_finetuning=False, offload_embedding=not a.no_offload_embedding)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    print(f"модель загружена за {time.time() - t_load:.0f} с", flush=True)

    sets = load_eval_sets(a.eval_dir, tok, a.seq_len, a.eval_tokens)
    for d, w in sets.items():
        print(f"  eval[{d}]: {len(w)} окон", flush=True)

    t0 = time.time()
    base = score(model, sets)
    eval_seconds = time.time() - t0
    print(f"база ({eval_seconds:.0f} с): " +
          "  ".join(f"{d}={math.exp(v):.4f}" for d, v in sorted(base.items())), flush=True)

    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=a.rank, lora_alpha=a.rank, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=3407,
        use_rslora=False, loftq_config=None)
    mods = lora_modules(model)
    print(f"LoRA-модулей: {len(mods)}", flush=True)

    import random
    docs = []
    for path in a.train_files:
        for line in open(path, encoding="utf-8"):
            t = json.loads(line)["text"].strip()
            if len(t) > 200:
                docs.append(t)
    random.Random(3407).shuffle(docs)
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    if eot is None or eot == tok.unk_token_id:
        eot = tok.eos_token_id
    stream, blocks = [], []
    for d in docs:
        stream += tok(d, add_special_tokens=False)["input_ids"] + [eot]
        while len(stream) >= a.seq_len:
            blocks.append(stream[:a.seq_len])
            stream = stream[a.seq_len:]
    dataset = Dataset.from_dict({"input_ids": blocks})
    print(f"документов: {len(docs)}, блоков по {a.seq_len}: {len(blocks)} "
          f"({len(blocks) * a.seq_len / 1e6:.1f} млн токенов)", flush=True)
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    target = "russian_code_register"
    targets = [t for t in a.targets if t in sets]
    summary = []
    for accum in a.accum:
        for lr in a.lrs:
            tag = f"lr{lr:g}_acc{accum}"
            reset_lora(mods, 3407)
            check = score(model, sets, only=[target])[target]
            reset_ok = abs(check - base[target]) < 1e-3
            print(f"\n=== {tag}: сброс {'верен' if reset_ok else 'НЕВЕРЕН'} "
                  f"({math.exp(check):.4f} против базы {math.exp(base[target]):.4f})", flush=True)

            log_path = os.path.join(a.out, f"{tag}.loss.jsonl")
            open(log_path, "w").close()

            traj_path = os.path.join(a.out, f"{tag}.trajectory.jsonl")
            start = score(model, sets, only=targets)
            with open(traj_path, "w") as fh:
                fh.write(json.dumps({"train_seconds": 0, "step": 0, "tokens": 0,
                                     "nll": start}) + "\n")

            class Timed(TrainerCallback):
                """Budget counts training time only; probe time is excluded."""
                def on_train_begin(self, args, state, control, **kw):
                    self.t = time.time()
                    self.paused = 0.0
                    self.next_probe = a.probe_every

                def trained(self):
                    return time.time() - self.t - self.paused

                def on_step_end(self, args, state, control, **kw):
                    el = self.trained()
                    if el >= self.next_probe and el < a.train_seconds:
                        p0 = time.time()
                        r = score(model, sets, only=targets)
                        with open(traj_path, "a") as fh:
                            fh.write(json.dumps({"train_seconds": round(el, 1),
                                                 "step": state.global_step,
                                                 "tokens": state.global_step * accum * a.seq_len,
                                                 "nll": r}) + "\n")
                        print(f"    [{el:4.0f} с, шаг {state.global_step}] " +
                              "  ".join(f"{d}={math.exp(v):.4f}" for d, v in r.items()), flush=True)
                        self.paused += time.time() - p0
                        self.next_probe += a.probe_every
                    if self.trained() >= a.train_seconds:
                        control.should_training_stop = True

                def on_log(self, args, state, control, logs=None, **kw):
                    if logs and "loss" in logs:
                        with open(log_path, "a") as fh:
                            fh.write(json.dumps({"step": state.global_step,
                                                 "t": round(self.trained(), 1),
                                                 "loss": logs["loss"],
                                                 "grad_norm": logs.get("grad_norm"),
                                                 "lr": logs.get("learning_rate")}) + "\n")

            trainer = Trainer(
                model=model, train_dataset=dataset, data_collator=collator,
                args=TrainingArguments(
                    per_device_train_batch_size=1, gradient_accumulation_steps=accum,
                    warmup_steps=3, max_steps=100000, learning_rate=lr,
                    lr_scheduler_type="constant_with_warmup",
                    logging_steps=1, optim="adamw_8bit", seed=3407,
                    output_dir=os.path.join(a.out, "hf"), save_strategy="no",
                    report_to="none", remove_unused_columns=False),
                callbacks=[Timed()])
            t_train = time.time()
            trainer.train()
            train_seconds = time.time() - t_train
            steps = trainer.state.global_step

            after = score(model, sets)
            losses = [json.loads(l)["loss"] for l in open(log_path)]
            head = sum(losses[:5]) / max(len(losses[:5]), 1)
            tail = sum(losses[-5:]) / max(len(losses[-5:]), 1)
            row = {
                "tag": tag, "lr": lr, "accum": accum, "rank": a.rank,
                "offload_embedding": not a.no_offload_embedding,
                "tokens_per_second": round(steps * accum * a.seq_len / max(train_seconds, 1), 1),
                "reset_ok": reset_ok, "steps": steps,
                "train_seconds": round(train_seconds, 1),
                "tokens_seen": steps * accum * a.seq_len,
                "train_loss_first5": head, "train_loss_last5": tail,
                "eval_ppl": {d: math.exp(v) for d, v in after.items()},
                "eval_delta_nll": {d: after[d] - base[d] for d in after},
            }
            summary.append(row)
            with open(os.path.join(a.out, "summary.jsonl"), "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"--- {tag}: шагов {steps}, loss {head:.3f} -> {tail:.3f}; "
                  f"Δnll: " + "  ".join(f"{d}={row['eval_delta_nll'][d]:+.4f}"
                                        for d in sorted(after)), flush=True)

            del trainer
            gc.collect()
            torch.cuda.empty_cache()

    json.dump({"base_ppl": {d: math.exp(v) for d, v in base.items()},
               "eval_seconds": eval_seconds, "runs": summary},
              open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)
    print("\nготово", flush=True)


if __name__ == "__main__":
    main()
