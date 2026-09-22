#!/usr/bin/env python3
"""Recover tool calls that a model emitted as text instead of as structured calls.

Some harnesses describe their tools in the system prompt using the XML-ish form

    <tool_call>
    <function=bash>
    <parameter=command>
    ...
    </parameter>
    </function>
    </tool_call>

and parse the model's reply themselves. When the reply arrives as plain content
with no `tool_calls`, the call is lost. This recovers it.

It tolerates the malformations actually observed in production traffic:
an opening tag left unclosed with no value (`<parameter=timeoutMs` followed
straight by `</parameter>`), a missing `</function>` or `</tool_call>` at the
end of a truncated reply, and prose before the first block.
"""
import re

_BLOCK = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.S)
_FUNC = re.compile(r"<function=([A-Za-z0-9_.-]+)\s*>(.*?)(?:</function>|\Z)", re.S)
# A parameter whose opening tag is properly closed, so it carries a value.
_PARAM = re.compile(r"<parameter=([A-Za-z0-9_.-]+)\s*>(.*?)(?:</parameter>|\Z)", re.S)
# A parameter whose opening tag was never closed: name only, no value.
_PARAM_BROKEN = re.compile(r"<parameter=([A-Za-z0-9_.-]+)\s*(?=</parameter>|<parameter=|</function>|</tool_call>|\Z)")


def parse_xml_tool_calls(content):
    """Return (calls, leftover_text).

    calls is a list of {"name": str, "arguments": dict, "malformed": [str]}.
    leftover_text is the content with every recovered block removed, so a caller
    can keep any prose the model wrote alongside the call.
    """
    calls, spans = [], []
    for block in _BLOCK.finditer(content or ""):
        spans.append(block.span())
        for func in _FUNC.finditer(block.group(1)):
            name, body = func.group(1), func.group(2)
            args, malformed = {}, []
            for p in _PARAM.finditer(body):
                args[p.group(1)] = p.group(2).strip("\n")
            # Names that only appear in a broken opening tag carry no value.
            for p in _PARAM_BROKEN.finditer(body):
                if p.group(1) not in args:
                    malformed.append(p.group(1))
            calls.append({"name": name, "arguments": args, "malformed": malformed})
    leftover = content or ""
    for start, end in reversed(spans):
        leftover = leftover[:start] + leftover[end:]
    return calls, leftover.strip()


if __name__ == "__main__":
    import json, sys
    texts = json.load(open(sys.argv[1], encoding="utf-8"))
    ok = bad = 0
    mal = {}
    for t in texts:
        calls, _ = parse_xml_tool_calls(t)
        if calls and all(c["arguments"] for c in calls):
            ok += 1
        else:
            bad += 1
        for c in calls:
            for m in c["malformed"]:
                mal[m] = mal.get(m, 0) + 1
    print(f"texts: {len(texts)}   recovered: {ok}   failed: {bad}")
    print("parameter names seen only as broken opening tags:", mal)
    calls, leftover = parse_xml_tool_calls(texts[0])
    print("\nexample ->", json.dumps({"name": calls[0]["name"],
                                      "argument_keys": sorted(calls[0]["arguments"]),
                                      "malformed": calls[0]["malformed"]}, ensure_ascii=False))
    print("leftover prose:", repr(leftover[:120]))
