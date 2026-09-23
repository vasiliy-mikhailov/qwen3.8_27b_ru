#!/usr/bin/env python3
"""Calibration set for re-quantising Qwen3.8-27B to NVFP4 with Russian in mind.

AutoRound tunes each decoder block so that its quantised output matches the
bf16 output on these sequences, and the static activation scales of the NVFP4
layers are taken from them too. The set therefore describes where precision is
spent: here it goes to Russian technical prose, code with Russian comments and
agentic tool calls that carry such code, with a smaller share of plain code so
that code itself is not starved. English prose gets nothing on purpose.

The text does not have to be clean. Calibration matches the bf16 model's own
activations on it; it does not teach the model to write like it.

Every sequence is packed to exactly SEQLEN tokens so AutoRound can batch them.
None of it comes from the evaluation prompts (train/prompts_comments.jsonl
uses hand-written snippets).

Usage (inside qwen-quant): build_calib.py MODEL_DIR DATA_DIR OUT.jsonl
"""
import json, os, random, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from build_register_corpus import CYR, LOCALE_PATH, PROSE_FILE, is_russian, russian_comments  # noqa: E402

SEQLEN = 2048
PLAN = {                # sequences of SEQLEN tokens per kind
    "so_chat": 96,      # ru.stackoverflow question as user turn, answers as assistant turn
    "code_ru": 56,      # whole source files carrying Russian comments
    "tools": 48,        # agent turns: read a file, write it back with Russian comments
    "comments_ru": 24,  # Russian comment blocks extracted from code
    "code_en": 32,      # plain code without Cyrillic, to keep code itself calibrated
}
CODE_EXT = re.compile(r"\.(py|java|kt|cs|cpp|cc|c|h|hpp|go|rs|js|ts|tsx|php|rb|swift|scala|sql|sh)$", re.I)

SYSTEM = ("Ты — агент, который работает с кодом в репозитории пользователя. "
          "Отвечай по-русски, комментарии в коде пиши по-русски.")
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Overwrite a file in the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run", "description": "Run a shell command and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
ASKS = [
    "Добавь в {p} комментарии на русском: что делает каждая функция и почему так.",
    "Прокомментируй {p} по-русски, коллеги не понимают, как это работает.",
    "В {p} нет ни одного комментария. Опиши логику на русском прямо в коде.",
    "Разберись, что делает {p}, и допиши русские комментарии к неочевидным местам.",
]
THINK_READ = "I should read {p} first to see what the code does before commenting it."
THINK_WRITE = "Now I understand the logic. I'll write the file back with Russian comments."


def so_docs(path):
    for line in open(path, encoding="utf-8"):
        text = json.loads(line)["text"]
        head, _, rest = text.partition("\n\n---\n")
        if not rest:
            continue
        answers = [a.strip() for a in rest.split("\n\n---\n") if a.strip()]
        yield [{"role": "user", "content": head.strip()},
               {"role": "assistant", "content": answers[0]}]


def github_files(paths):
    import pyarrow.parquet as pq
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=["path", "content"]):
            for r in batch.to_pylist():
                fp, code = r["path"] or "", r["content"] or ""
                if LOCALE_PATH.search(fp) or PROSE_FILE.search(fp) or not CODE_EXT.search(fp):
                    continue
                if not 400 <= len(code) <= 6000:
                    continue
                yield fp, code


def strip_comments(code, comments):
    for c in comments:
        code = code.replace(c, "")
    return re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", code)


def tool_dialog(fp, code, comments, rng):
    bare = strip_comments(code, comments)
    name = fp.rsplit("/", 1)[-1]
    call = lambda n, **a: {"type": "function", "function": {"name": n, "arguments": a}}
    summary = "Готово. В {p} добавлены комментарии, например:\n\n{c}".format(
        p=name, c="\n".join("- " + c.splitlines()[0].strip("/*# ").strip() for c in comments[:3]))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": rng.choice(ASKS).format(p=name)},
        {"role": "assistant", "content": "", "reasoning_content": THINK_READ.format(p=name),
         "tool_calls": [call("read_file", path=name)]},
        {"role": "tool", "content": bare},
        {"role": "assistant", "content": "", "reasoning_content": THINK_WRITE,
         "tool_calls": [call("write_file", path=name, content=code)]},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": summary},
    ]


class Packer:
    """Concatenates token streams of one kind and cuts them into SEQLEN pieces."""

    def __init__(self, tok, eos):
        self.tok, self.eos, self.buf, self.out = tok, eos, [], []

    def add(self, ids):
        self.buf += ids + [self.eos]
        while len(self.buf) >= SEQLEN:
            self.out.append(self.buf[:SEQLEN])
            self.buf = self.buf[SEQLEN:]


def main():
    model_dir, data_dir, out = sys.argv[1:4]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    rng = random.Random(20260923)
    pack = {k: Packer(tok, eos) for k in PLAN}
    full = lambda k: len(pack[k].out) >= PLAN[k]
    chat = lambda msgs, **kw: tok.apply_chat_template(msgs, tokenize=False, **kw)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]

    so = list(so_docs(os.path.join(data_dir, "corpora/register/stackoverflow-train.jsonl")))
    rng.shuffle(so)
    for msgs in so:
        if full("so_chat"):
            break
        pack["so_chat"].add(enc(chat(msgs)))

    for line in open(os.path.join(data_dir, "corpora/register/comments-train.jsonl"), encoding="utf-8"):
        if full("comments_ru"):
            break
        pack["comments_ru"].add(enc(json.loads(line)["text"]))

    # The last shards were not used to build the training corpus.
    shards = sorted(os.path.join(data_dir, "raw", f) for f in os.listdir(os.path.join(data_dir, "raw"))
                    if f.startswith("gh_") and f.endswith(".parquet"))[::-1]
    for fp, code in github_files(shards):
        if all(full(k) for k in ("code_ru", "tools", "code_en")):
            break
        if not CYR.search(code):
            if not full("code_en") and rng.random() < 0.02:
                pack["code_en"].add(enc(f"// {fp}\n{code}"))
            continue
        comments = russian_comments(code, fp)
        if len(comments) < 3 or not is_russian("\n".join(comments)):
            continue
        if not full("tools") and rng.random() < 0.5:
            pack["tools"].add(enc(chat(tool_dialog(fp, code, comments, rng), tools=TOOLS)))
        elif not full("code_ru"):
            pack["code_ru"].add(enc(f"// {fp}\n{code}"))

    rows = []
    for k, p in pack.items():
        got = p.out[:PLAN[k]]
        if len(got) < PLAN[k]:
            print(f"warning: {k} has {len(got)} of {PLAN[k]}", file=sys.stderr)
        rows += [{"kind": k, "input_ids": ids} for ids in got]
    rng.shuffle(rows)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    counts = {k: sum(r["kind"] == k for r in rows) for k in PLAN}
    print(json.dumps({"sequences": len(rows), "tokens": len(rows) * SEQLEN, "by_kind": counts}, ensure_ascii=False))
    sample = next(r for r in rows if r["kind"] == "tools")
    print(tok.decode(sample["input_ids"][:900]))


if __name__ == "__main__":
    main()
