#!/usr/bin/env python3
"""Generate the prompt set through an OpenAI-compatible endpoint.

Used to put other quantisations and engines on the same footing as the NF4
harness: same prompts, greedy decoding, thinking off, same token limit. Only
the model behind the URL differs -- Q8_0 on llama.cpp, the production NVFP4 on
ninfer.

Output rows match gen_compare.py, so blinding and annotation treat every arm
alike.
"""
import argparse, json, time, urllib.request, concurrent.futures as cf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--key", default="")
    p.add_argument("--arm", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--extra", default="{}", help="JSON merged into the request body")
    a = p.parse_args()

    prompts = [json.loads(l) for l in open(a.prompts, encoding="utf-8")]
    extra = json.loads(a.extra)

    def call(pr):
        body = {"messages": [{"role": "user", "content": pr["prompt"]}],
                "max_tokens": a.max_tokens, "temperature": 0.0, **extra}
        if a.model:
            body["model"] = a.model
        hdr = {"Content-Type": "application/json"}
        if a.key:
            hdr["Authorization"] = "Bearer " + a.key
        req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers=hdr)
        err = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    d = json.load(r)
                msg = d["choices"][0]["message"]
                return {"id": pr["id"], "kind": pr["kind"], "arm": a.arm, "prompt": pr["prompt"],
                        "text": (msg.get("content") or "").strip(),
                        "new_tokens": (d.get("usage") or {}).get("completion_tokens")}
            except Exception as e:
                err = str(e)
                time.sleep(3)
        return {"id": pr["id"], "kind": pr["kind"], "arm": a.arm, "prompt": pr["prompt"],
                "text": "", "error": err}

    t0 = time.time()
    rows = []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for n, r in enumerate(ex.map(call, prompts), 1):
            rows.append(r)
            if n % 8 == 0:
                print(f"[{a.arm}] {n}/{len(prompts)}  {time.time()-t0:.0f} с", flush=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    bad = sum(1 for r in rows if not r["text"])
    print(f"готово: {len(rows)} ответов, пустых {bad}, {time.time()-t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
