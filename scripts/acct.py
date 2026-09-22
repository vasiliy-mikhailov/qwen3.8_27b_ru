import json, subprocess, collections, re, sys
def acct(path, label):
    raw = subprocess.run(["python3","-m","tools.artifact.inspect",path,"--bindings"],
                         capture_output=True, text=True).stdout
    i = raw.index("\n{\n")
    b = json.loads(raw[i+1:])["bindings"]
    txt = subprocess.run(["python3","-m","tools.artifact.inspect",path,"--objects"],
                         capture_output=True, text=True).stdout
    size = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+tensor\s+\S+\s+\[[^\]]*\]\s+(\S+)", line)
        if m: size[m.group(3)] = int(m.group(2))
    by = collections.Counter(); seen = set()
    for param, v in b.items():
        o = v.get("object") if isinstance(v, dict) else None
        if not o or o in seen: continue
        seen.add(o); by[param.split("/")[0]] += size.get(o, 0)
    print("=== " + label)
    for k, v in by.most_common():
        print("    %-10s %6.2f GiB" % (k, v / 2**30))
    print("    %-10s %6.2f GiB" % ("[все тензоры]", sum(size.values()) / 2**30))
    return by
a = acct(sys.argv[1], "nvfp4 (актуальный)")
g = acct(sys.argv[2], "groupwise")
print()
print("%-10s %12s %12s %8s" % ("компонент", "nvfp4", "groupwise", "дельта"))
for k in ("text", "mtp", "dflash2", "vision", "proposal"):
    print("%-10s %8.2f GiB %8.2f GiB %+7.2f" % (k, a.get(k,0)/2**30, g.get(k,0)/2**30, (a.get(k,0)-g.get(k,0))/2**30))
