import json, random, sys
# Three arms sharing one reference: q8 answers "do more bits help?",
# dflash answers "did the always-W4A4 switch cost anything?", nvfp4 is the anchor.
ARMS={"q8":"/tmp/bulk_q8.jsonl","nvfp4":"/tmp/bulk_nvfp4.jsonl","dflash":"/tmp/bulk_dflash.jsonl"}
N=int(sys.argv[1]) if len(sys.argv)>1 else 60
def load(f):
    d={}
    for l in open(f,encoding="utf-8"):
        r=json.loads(l)
        if r.get("content"): d[r["prompt_id"]]=r["content"]
    return d
data={k:load(v) for k,v in ARMS.items()}
common=sorted(set.intersection(*[set(v) for v in data.values()]))
rng=random.Random(20260922)
pick=sorted(rng.sample(common,min(N,len(common))))
items=[{"arm":a,"prompt_id":p,"text":data[a][p]} for p in pick for a in ARMS]
rng.shuffle(items)
key={}
with open("/tmp/blind_corpus.jsonl","w",encoding="utf-8") as fh:
    for i,it in enumerate(items):
        tag=f"T{i:03d}"
        key[tag]={"arm":it["arm"],"prompt_id":it["prompt_id"]}
        fh.write(json.dumps({"tag":tag,"text":it["text"]},ensure_ascii=False)+"\n")
json.dump(key,open("/tmp/blind_key.json","w"))
words=sum(len(it["text"].split()) for it in items)
print(f"prompts={len(pick)} texts={len(items)} words={words} est_tokens={int(words*1.83)}")
print("per arm:", {a:sum(1 for it in items if it["arm"]==a) for a in ARMS})
