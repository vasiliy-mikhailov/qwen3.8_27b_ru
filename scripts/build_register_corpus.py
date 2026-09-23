#!/usr/bin/env python3
"""Extract Russian technical text in the register the model is bad at.

Two sources, because they teach different things:

  ru.stackoverflow   Russian prose interleaved with code blocks. 85% of posts
                     carry code. This is how Russians explain code.
  github comments    Russian comments and docstrings sitting in real source
                     files. This is how Russians annotate code.

Both need filtering that a Cyrillic test alone does not give:

  * Cyrillic is not Russian. Ukrainian (і ї є ґ) and Bulgarian (no ы э) files
    are common on GitHub, and the densest Cyrillic hits are Qt localisation
    files, which are translation tables rather than comments.
  * String literals are not comments. Only comment and docstring regions carry
    the register we want.

Output is split into a training pool and a held-out slice. The held-out slice
becomes a perplexity domain: if the model gets better at predicting how humans
actually comment Russian code, that is the register signal, and nothing in the
mechanical defect battery can see it.
"""
import io, json, os, re, subprocess, sys, random

CYR = re.compile("[А-Яа-яЁё]")
UKR = re.compile("[іїєґІЇЄҐ]")
RU_ONLY = re.compile("[ыэЫЭёЁъЪ]")

LOCALE_PATH = re.compile(r"(^|/)(locale|locales|i18n|translations?|lang)(/|$)|\.(ts|po|pot|resx|xlf|strings)$", re.I)

COMMENT_PATTERNS = [
    re.compile(r"/\*(.*?)\*/", re.S),          # C-family block
    re.compile(r"//[^\n]*", 0),                # C-family line
    re.compile(r'"""(.*?)"""', re.S),          # python docstring
    re.compile(r"<!--(.*?)-->", re.S),         # html
]
# A leading # is a comment in shell and Python but a heading in Markdown and a
# directive in reStructuredText, so it is only applied to files where it means
# a comment. Without this, Markdown documents contribute their headings.
HASH_COMMENT = re.compile(r"#[^\n]*")
HASH_OK = re.compile(r"\.(py|sh|bash|zsh|rb|pl|yml|yaml|toml|cfg|ini|conf|mk|tf|r)$", re.I)
PROSE_FILE = re.compile(r"\.(md|markdown|rst|txt|adoc|tex|html?|xml|json|csv)$", re.I)


CODE_OPEN = re.compile(r"\[code\]\s*", re.I)
CODE_CLOSE = re.compile(r"\s*\[/code\]", re.I)


def normalise_code_fences(text):
    """ru.stackoverflow marks code with [code] tags; the model has only ever
    seen fenced blocks, so the markup is converted rather than taught."""
    text = CODE_OPEN.sub("```\n", text)
    return CODE_CLOSE.sub("\n```", text)


def is_russian(text):
    if UKR.search(text):
        return False
    return bool(RU_ONLY.search(text)) or len(CYR.findall(text)) > 200


def russian_comments(code, path=""):
    if PROSE_FILE.search(path):
        return []
    pats = list(COMMENT_PATTERNS)
    if HASH_OK.search(path):
        pats.append(HASH_COMMENT)
    out = []
    for pat in pats:
        for m in pat.finditer(code):
            frag = m.group(0)
            if len(CYR.findall(frag)) >= 8 and is_russian(frag):
                out.append(frag.strip())
    return out


def from_stackoverflow(path, limit):
    p = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
    got = 0
    for line in io.TextIOWrapper(p.stdout, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        parts = []
        if r.get("title"):
            parts.append("# " + r["title"])
        if r.get("text_markdown"):
            parts.append(normalise_code_fences(r["text_markdown"]))
        for a in (r.get("answers") or [])[:2]:
            t = a.get("text_markdown") if isinstance(a, dict) else None
            if t:
                parts.append("---\n" + normalise_code_fences(t))
        doc = "\n\n".join(parts).strip()
        if len(CYR.findall(doc)) < 120 or not is_russian(doc):
            continue
        yield doc
        got += 1
        if got >= limit:
            break
    p.kill()


def from_github(paths, limit):
    import pyarrow.parquet as pq
    got = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000):
            for r in batch.to_pylist():
                fp = r.get("path") or ""
                if LOCALE_PATH.search(fp):
                    continue
                code = r.get("content") or ""
                if not CYR.search(code):
                    continue
                cs = russian_comments(code, fp)
                if len(cs) < 2:
                    continue
                yield "\n".join(cs)
                got += 1
                if got >= limit:
                    return


if __name__ == "__main__":
    dest = sys.argv[1]
    so_path = sys.argv[2]
    gh_paths = sys.argv[3:]
    os.makedirs(dest, exist_ok=True)
    rng = random.Random(20260923)

    so = list(from_stackoverflow(so_path, 12000))
    gh = list(from_github(gh_paths, 4000)) if gh_paths else []
    rng.shuffle(so)
    rng.shuffle(gh)
    # A tenth of each source is held out and never trained on.
    for name, docs in (("stackoverflow", so), ("comments", gh)):
        cut = max(1, len(docs) // 10)
        for split, part in (("heldout", docs[:cut]), ("train", docs[cut:])):
            fn = os.path.join(dest, f"{name}-{split}.txt")
            with open(fn, "w", encoding="utf-8") as fh:
                fh.write("\n\n".join(part))
            # Documents themselves contain blank lines, so the .txt form cannot
            # be split back into documents: a post falls apart into paragraphs
            # and its code block detaches from its explanation. Training reads
            # the JSONL, one whole document per line.
            with open(os.path.join(dest, f"{name}-{split}.jsonl"), "w", encoding="utf-8") as fh:
                for d in part:
                    fh.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
            w = sum(len(CYR.findall(d)) for d in part) // 6
            print(f"{name}-{split}: документов {len(part)}, ~{w} русских слов -> {fn}")
