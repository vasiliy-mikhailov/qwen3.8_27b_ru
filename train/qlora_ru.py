#!/usr/bin/env python3
"""QLoRA on Russian code-register text, measuring the trend as it trains.

Follows unsloth's documented Qwen3.8-27B recipe (r=16, alpha=16, dropout=0,
adamw_8bit, lr 2e-4, batch 1 x accum 4, max_seq_length 2048, offload_embedding)
because deviating from it costs GPU hours to rediscover.

This is CONTINUED PRETRAINING on raw text, not instruction tuning: the goal is
to learn how Russians write about code, and the model already follows
instructions. So no chat template is applied — raw documents are packed into
blocks.

Evaluation runs inside the same process at fixed intervals. Reloading a 27B
model per checkpoint would cost more than the training itself, and the point is
the trajectory, not any single number. Every pass scores the target domains and
the guards together, so a Russian gain bought with a code loss is visible in the
same row.

Absolute perplexities here are NOT comparable to the ninfer baseline: the base is
NF4 here and NVFP4 there. Step 0 is measured in this same harness so the trend
has its own zero.
"""
import argparse, json, os, time

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--train-files", nargs="+", required=True)
    p.add_argument("--eval-dir", required=True,
                   help="directory of <domain>.txt files scored at every checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--eval-tokens", type=int, default=120_000,
                   help="tokens scored per domain per pass; keeps a pass to about a minute")
    return p.parse_args()


def load_eval_sets(eval_dir, tokenizer, seq_len, budget):
    """Fixed windows per domain, tokenised once and reused for every pass."""
    sets = {}
    for fn in sorted(os.listdir(eval_dir)):
        if not fn.endswith(".txt"):
            continue
        domain = fn[:-4]
        text = open(os.path.join(eval_dir, fn), encoding="utf-8").read()
        ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
        windows = []
        step = seq_len
        for i in range(0, min(len(ids), budget) - 1, step):
            chunk = ids[i:i + seq_len]
            if len(chunk) >= 128:
                windows.append(torch.tensor(chunk, dtype=torch.long))
        sets[domain] = windows
        print(f"  eval[{domain}]: {len(windows)} окон, {sum(len(w) for w in windows)} токенов",
              flush=True)
    return sets


@torch.no_grad()
def score(model, sets):
    """Mean NLL per domain over the fixed windows."""
    model.eval()
    out = {}
    for domain, windows in sets.items():
        nll = 0.0
        n = 0
        for w in windows:
            ids = w.unsqueeze(0).to(model.device)
            loss = model(input_ids=ids, labels=ids).loss
            # HF averages over the window's scored positions.
            k = ids.shape[1] - 1
            nll += loss.item() * k
            n += k
        out[domain] = {"mean_nll": nll / max(n, 1), "ppl": float(torch.exp(torch.tensor(nll / max(n, 1)))),
                       "tokens": n}
    model.train()
    return out


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    from unsloth import FastModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    from transformers import TrainerCallback

    model, tokenizer = FastModel.from_pretrained(
        model_name=a.base,
        max_seq_length=a.seq_len,
        load_in_4bit=True,
        full_finetuning=False,
        offload_embedding=True,
    )

    # Qwen3.8 is a vision-language model, so from_pretrained hands back a
    # processor. Calling it on plain text makes it hunt for images, so the inner
    # text tokenizer is used for everything here.
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)

    print("подготовка наборов оценки", flush=True)
    eval_sets = load_eval_sets(a.eval_dir, text_tok, a.seq_len, a.eval_tokens)

    trace_path = os.path.join(a.out, "trend.jsonl")

    def record(step, extra=None):
        t0 = time.time()
        res = score(model, eval_sets)
        row = {"step": step, "seconds": round(time.time() - t0, 1), "domains": res}
        if extra:
            row.update(extra)
        with open(trace_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        line = "  ".join(f"{d}={res[d]['ppl']:.4f}" for d in sorted(res))
        print(f"[шаг {step}] {line}", flush=True)

    # Zero point: the base model, before any adapter exists.
    record(0)

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    docs = []
    for path in a.train_files:
        raw = open(path, encoding="utf-8").read()
        docs += [d.strip() for d in raw.split("\n\n") if len(d.strip()) > 200]
    print(f"обучающих документов: {len(docs)}", flush=True)
    dataset = Dataset.from_dict({"text": docs}).shuffle(seed=3407)

    class Probe(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if state.global_step and state.global_step % a.eval_every == 0:
                loss = None
                if state.log_history:
                    loss = state.log_history[-1].get("loss")
                record(state.global_step, {"train_loss": loss})
                model.save_pretrained(os.path.join(a.out, f"adapter-{state.global_step}"))

    trainer = SFTTrainer(
        model=model,
        processing_class=text_tok,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=a.seq_len,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=a.max_steps,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            output_dir=os.path.join(a.out, "hf"),
            seed=3407,
            dataset_num_proc=1,
            report_to="none",
            save_strategy="no",
        ),
        callbacks=[Probe()],
    )
    trainer.train()
    record(a.max_steps, {"final": True})
    model.save_pretrained(os.path.join(a.out, "adapter-final"))
    print("готово ->", trace_path, flush=True)


if __name__ == "__main__":
    main()
