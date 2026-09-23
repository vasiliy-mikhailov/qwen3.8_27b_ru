#!/usr/bin/env python3
"""Does the model still call tools correctly? A guard for re-quantisation.

Forty agent turns built from the ten snippets of make_comment_prompts.py, four
situations each, all in Russian and all expecting a tool call:

  read   - the user asks to comment a file; the model must call read_file;
  write  - the file has been read; the model must call write_file with the
           whole file, now carrying Russian comments, as one long argument;
  run    - the user asks to run the tests; the model must call run;
  fix    - a test has failed with a traceback; the model must read or write.

Greedy decoding, thinking off, same flags for every artifact. A turn counts as
a clean call when the server returns structured tool_calls whose arguments are
JSON with every required parameter. A leak is tool-call markup left in the
text, the failure seen in production. Everything else is prose.

Usage: toolcall_probe.py --url URL [--model M] [--key K] --arm NAME --out OUT.jsonl
"""
import argparse, json, os, re, sys, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
_src = open(os.path.join(os.path.dirname(__file__), "..", "train", "make_comment_prompts.py"), encoding="utf-8").read()
_ns = {}
exec(_src.split("if __name__")[0], _ns)
SNIPPETS = _ns["SNIPPETS"]

EXT = {"python": "py", "java": "java", "csharp": "cs", "go": "go", "typescript": "ts", "sql": "sql",
       "bash": "sh", "kotlin": "kt", "elixir": "ex", "rust": "rs"}
TEST = {"python": "pytest -q", "java": "mvn -q test", "csharp": "dotnet test", "go": "go test ./...",
        "typescript": "npm test", "sql": "psql -f test.sql", "bash": "bats test", "kotlin": "gradle test",
        "elixir": "mix test", "rust": "cargo test"}
SYSTEM = ("Ты — агент, который работает с кодом в репозитории пользователя. Пользуйся "
          "инструментами, отвечай по-русски, комментарии в коде пиши по-русски.")
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Overwrite a file in the workspace with new content.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run", "description": "Run a shell command in the workspace and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
REQUIRED = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}
LEAK = re.compile(r"<tool_call>|<function=|<parameter=|</tool_call>")


def call(name, args, i):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def tasks():
    for lang, code in SNIPPETS:
        path = f"src/{lang}_module.{EXT[lang]}"
        ask = f"Добавь в {path} подробные комментарии на русском: что делает код и почему именно так."
        read = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask}]
        yield f"{lang}-read", read, {"read_file"}
        wrote = read + [{"role": "assistant", "content": "", "tool_calls": [call("read_file", {"path": path}, 1)]},
                        {"role": "tool", "tool_call_id": "call_1", "content": code}]
        yield f"{lang}-write", wrote, {"write_file"}
        run = [{"role": "system", "content": SYSTEM},
               {"role": "user", "content": "Запусти тесты проекта и, если что-то упадёт, разберись и почини."}]
        yield f"{lang}-run", run, {"run"}
        fail = (f"FAILED test_{lang}_module - AssertionError: expected 3 elements, got 2\n"
                f"  at {path}:7\n1 failed, 14 passed")
        fix = run + [{"role": "assistant", "content": "", "tool_calls": [call("run", {"command": TEST[lang]}, 1)]},
                     {"role": "tool", "tool_call_id": "call_1", "content": fail}]
        yield f"{lang}-fix", fix, {"read_file", "write_file", "run"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--key", default="")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--extra", default='{"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": false}}')
    a = ap.parse_args()
    extra = json.loads(a.extra)
    rows, t0 = [], time.time()
    for tid, messages, expect in tasks():
        body = {"messages": messages, "tools": TOOLS, "max_tokens": a.max_tokens, "temperature": 0.0, **extra}
        if a.model:
            body["model"] = a.model
        hdr = {"Content-Type": "application/json"}
        if a.key:
            hdr["Authorization"] = "Bearer " + a.key
        req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                msg = json.load(r)["choices"][0]["message"]
        except Exception as e:
            rows.append({"id": tid, "arm": a.arm, "outcome": "error", "error": str(e)})
            continue
        text = msg.get("content") or ""
        calls = msg.get("tool_calls") or []
        outcome, detail = "prose", ""
        if calls:
            outcome = "clean"
            for c in calls:
                fn = c.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    outcome, detail = "bad_args", fn.get("arguments", "")[:200]
                    break
                if fn.get("name") not in REQUIRED or any(k not in args for k in REQUIRED[fn["name"]]):
                    outcome, detail = "bad_args", json.dumps(fn, ensure_ascii=False)[:200]
                    break
            if outcome == "clean" and not {c["function"]["name"] for c in calls} & expect:
                outcome = "wrong_tool"
        if LEAK.search(text):
            outcome = "leak" if outcome != "clean" else "clean+markup"
        rows.append({"id": tid, "arm": a.arm, "outcome": outcome, "detail": detail, "text": text[:2000],
                     "tool_calls": calls})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    count = {}
    for r in rows:
        count[r["outcome"]] = count.get(r["outcome"], 0) + 1
    print(json.dumps({"arm": a.arm, "turns": len(rows), "outcomes": count, "seconds": round(time.time() - t0)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
