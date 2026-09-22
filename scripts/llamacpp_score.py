#!/usr/bin/env python3
"""Exact sentence NLL from llama.cpp, under the model's own tokenisation.

llama.cpp exposes no prompt logprobs, so each token is scored with its own
request: the prefix is sent as the prompt and `logit_bias` forces the one token
that actually follows. With `post_sampling_probs: false` the server reports the
RAW model logprob of that token, unaffected by the bias — verified to match the
unbiased top-k readout to within float noise. Reading a top-k list instead would
not work: the correct token frequently falls outside the top 20.

The first token is context and left unscored, matching what ninfer-perplexity
scores for a stream, so totals are directly comparable across the two engines.

An earlier version forced whole sentences through a GBNF grammar in one request.
That was wrong: the grammar emits far more, shorter tokens than the natural
tokenisation, and the resulting NLL tracked token count (r = 0.88) rather than
grammaticality.

Run this single-threaded against a single-slot server. Scoring a sentence walks
a prefix that grows by one token per request, which is exactly the pattern the
prompt cache is built for: each step reuses the previous state and processes one
new token. Concurrency breaks that -- Qwen3.8 is mostly linear-attention layers,
so one cached prompt state is 1-2 GiB, and several workers on distinct prefixes
make the server thrash evictions until a single request takes over a minute.
"""
import json, sys, time, urllib.request, concurrent.futures as cf

URL = sys.argv[1].rstrip("/")
IN, OUT = sys.argv[2], sys.argv[3]
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 16


def post(path, body, timeout=300):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def score(item):
    err = "unknown"
    for _ in range(3):
        try:
            toks = post("/tokenize", {"content": item["text"], "add_special": False})["tokens"]
            if len(toks) < 2:
                return {"id": item["id"], "error": "too short"}
            total = 0.0
            for i in range(1, len(toks)):
                d = post("/completion", {"prompt": toks[:i], "n_predict": 1, "temperature": 0,
                                         "n_probs": 1, "post_sampling_probs": False,
                                         "cache_prompt": True,
                                         "logit_bias": [[toks[i], 100.0]]})
                p = d["completion_probabilities"][0]
                if p["id"] != toks[i]:
                    return {"id": item["id"], "error": f"bias failed at {i}"}
                total -= p["logprob"]
            return {"id": item["id"], "total_nll": total, "n": len(toks) - 1}
        except Exception as e:
            err = str(e)
            time.sleep(1)
    return {"id": item["id"], "error": err}


items = [json.loads(l) for l in open(IN, encoding="utf-8")]
done = set()
try:
    for l in open(OUT, encoding="utf-8"):
        r = json.loads(l)
        if not r.get("error"):
            done.add(r["id"])
except FileNotFoundError:
    pass
todo = [i for i in items if i["id"] not in done]
print(f"{len(done)} done, {len(todo)} to score", flush=True)

t0 = time.time()
with open(OUT, "a", encoding="utf-8") as fh, cf.ThreadPoolExecutor(WORKERS) as ex:
    for n, res in enumerate(ex.map(score, todo), 1):
        fh.write(json.dumps(res, ensure_ascii=False) + "\n")
        if n % 100 == 0:
            fh.flush()
            print(f"[{time.time()-t0:5.0f}s] {n}/{len(todo)}", flush=True)
print("done", round(time.time() - t0, 1), "s")
