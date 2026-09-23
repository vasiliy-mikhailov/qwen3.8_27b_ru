#!/usr/bin/env python3
"""Generate the same prompts before and after a short QLoRA run, in one process.

Perplexity can fall only because the model got used to raw text, while the
number of errors in what it writes stays the same. The question that matters is
the second one, so this produces two sets of generations that differ in exactly
one thing -- the adapter:

  base      the loaded model with no adapter
  trained   the same model after --train-seconds of QLoRA with the given config

Same prompts, same chat template, greedy decoding, same token limit. Both
generations happen in this harness (NF4 base via transformers), not on ninfer,
so quantisation and engine cannot differ between the arms.

The adapter is saved so the trained arm can be reproduced or extended.

--checkpoints turns one long run into a trajectory: training pauses at each
listed number of training seconds, generates an arm named t<seconds>, saves
the adapter, and resumes. Evaluation is far more expensive than five minutes of
training, so one long run with several generation points costs little more
than a short one. --skip-base reuses base generations, which are deterministic
under greedy decoding and need to be produced only once.
"""
import argparse, json, os, random, time

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--train-files", nargs="+", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--train-seconds", type=int, default=300)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=700)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--checkpoints", type=int, nargs="*", default=None,
                   help="training seconds at which to generate; replaces --train-seconds")
    p.add_argument("--skip-base", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    from unsloth import FastModel
    from datasets import Dataset
    from transformers import (Trainer, TrainingArguments, TrainerCallback,
                              DataCollatorForLanguageModeling)

    model, tokenizer = FastModel.from_pretrained(
        model_name=a.base, max_seq_length=a.seq_len, load_in_4bit=True,
        full_finetuning=False, offload_embedding=True)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    stop_ids = [i for i in {tok.convert_tokens_to_ids("<|im_end|>"),
                            tok.convert_tokens_to_ids("<|endoftext|>"), tok.eos_token_id}
                if i is not None and i != tok.unk_token_id]

    prompts = [json.loads(l) for l in open(a.prompts, encoding="utf-8")]

    def render(p):
        msgs = [{"role": "user", "content": p}]
        try:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def generate_all(arm):
        path = os.path.join(a.out, f"gen_{arm}.jsonl")
        try:
            FastModel.for_inference(model)
        except Exception as e:
            print("for_inference недоступен:", e, flush=True)
        model.eval()
        tok.padding_side = "left"
        t0 = time.time()
        new_total = 0
        with open(path, "w", encoding="utf-8") as fh:
            for i in range(0, len(prompts), a.batch):
                chunk = prompts[i:i + a.batch]
                enc = tok([render(p["prompt"]) for p in chunk], return_tensors="pt",
                          padding=True, add_special_tokens=False).to(model.device)
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     temperature=None, top_p=None, top_k=None,
                                     eos_token_id=stop_ids, pad_token_id=tok.pad_token_id)
                for p, row in zip(chunk, out):
                    gen = row[enc["input_ids"].shape[1]:]
                    n = int((gen != tok.pad_token_id).sum())
                    new_total += n
                    text = tok.decode(gen, skip_special_tokens=True).strip()
                    fh.write(json.dumps({"id": p["id"], "kind": p["kind"], "arm": arm,
                                         "prompt": p["prompt"], "text": text,
                                         "new_tokens": n}, ensure_ascii=False) + "\n")
                fh.flush()
                el = time.time() - t0
                print(f"  [{arm}] {min(i + a.batch, len(prompts))}/{len(prompts)}  "
                      f"{new_total} ток, {new_total / el:.0f} ток/с", flush=True)
        tok.padding_side = "right"
        try:
            FastModel.for_training(model)
        except Exception:
            pass
        model.train()
        return path

    if not a.skip_base:
        print("генерация базы", flush=True)
        generate_all("base")

    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=a.rank, lora_alpha=a.rank, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=3407,
        use_rslora=False, loftq_config=None)

    docs = []
    for path in a.train_files:
        for line in open(path, encoding="utf-8"):
            t = json.loads(line)["text"].strip()
            if len(t) > 200:
                docs.append(t)
    random.Random(3407).shuffle(docs)
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    stream, blocks = [], []
    for d in docs:
        stream += tok(d, add_special_tokens=False)["input_ids"] + [eot]
        while len(stream) >= a.seq_len:
            blocks.append(stream[:a.seq_len])
            stream = stream[a.seq_len:]
    dataset = Dataset.from_dict({"input_ids": blocks})

    points = sorted(a.checkpoints) if a.checkpoints else [a.train_seconds]
    log = []

    class Timed(TrainerCallback):
        """Budget counts training time only; generation pauses are excluded."""
        def on_train_begin(self, args, state, control, **kw):
            self.t = time.time()
            self.paused = 0.0
            self.left = list(points)

        def on_step_end(self, args, state, control, **kw):
            el = time.time() - self.t - self.paused
            if self.left and el >= self.left[0]:
                mark = self.left.pop(0)
                p0 = time.time()
                info = {"checkpoint_seconds": mark, "trained_seconds": round(el, 1),
                        "steps": state.global_step,
                        "tokens_seen": state.global_step * a.accum * a.seq_len}
                print("контрольная точка:", info, flush=True)
                model.save_pretrained(os.path.join(a.out, f"adapter-t{mark}"))
                generate_all(f"t{mark}")
                log.append(info)
                json.dump(log, open(os.path.join(a.out, "checkpoints.json"), "w"), indent=1)
                self.paused += time.time() - p0
            if not self.left:
                control.should_training_stop = True

    trainer = Trainer(
        model=model, train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
        args=TrainingArguments(
            per_device_train_batch_size=1, gradient_accumulation_steps=a.accum,
            warmup_steps=3, max_steps=100000, learning_rate=a.lr,
            lr_scheduler_type="constant_with_warmup", logging_steps=5,
            optim="adamw_8bit", seed=3407, output_dir=os.path.join(a.out, "hf"),
            save_strategy="no", report_to="none", remove_unused_columns=False),
        callbacks=[Timed()])
    t0 = time.time()
    trainer.train()
    steps = trainer.state.global_step
    meta = {"rank": a.rank, "accum": a.accum, "lr": a.lr, "steps": steps,
            "wall_seconds": round(time.time() - t0, 1), "checkpoints": points,
            "tokens_seen": steps * a.accum * a.seq_len}
    print("обучение:", meta, flush=True)
    if not a.checkpoints:
        model.save_pretrained(os.path.join(a.out, "adapter"))
    del trainer

    if not a.checkpoints:
        print("генерация после обучения", flush=True)
        generate_all("trained")
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("готово", flush=True)


if __name__ == "__main__":
    main()
