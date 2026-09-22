import json, sys, time, urllib.request, re, os, concurrent.futures as cf
API=sys.argv[1]          # 'ninfer' or 'llamacpp'
URL=sys.argv[2]; OUT=sys.argv[3]; KEY=sys.argv[4] if len(sys.argv)>4 else ""
prompts=json.load(open('/tmp/prompts_1m.json'))
done={}
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r=json.loads(l); done[r["prompt_id"]]=1
        except Exception: pass
print(f"{len(done)} already done", flush=True)
def call(p):
    if API=="ninfer":
        body={"model":"qwen-3.8-27b-int","messages":[{"role":"user","content":p}],"max_tokens":2500,
              "temperature":0.0,"presence_penalty":0.0,"reasoning_effort":"none"}
        hdr={"Authorization":"Bearer "+KEY,"Content-Type":"application/json"}
    else:
        body={"messages":[{"role":"user","content":p}],"max_tokens":2500,"temperature":0.0,
              "presence_penalty":0.0,"chat_template_kwargs":{"enable_thinking":False}}
        hdr={"Content-Type":"application/json"}
    req=urllib.request.Request(URL+"/v1/chat/completions", data=json.dumps(body).encode(), headers=hdr)
    err=None
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=2400) as r: d=json.load(r)
            return {"content":d["choices"][0]["message"].get("content",""),"usage":d.get("usage"),"timings":d.get("timings")}
        except Exception as e:
            err=str(e); time.sleep(5)
    return {"error":err}
todo=[(i,p) for i,p in enumerate(prompts) if i not in done]
t0=time.time(); tok=0; n=0
W=2 if API=="ninfer" else 1
fh=open(OUT,"a")
with cf.ThreadPoolExecutor(max_workers=W) as ex:
    futs={ex.submit(call,p):(i,p) for i,p in todo}
    for f in cf.as_completed(futs):
        i,p=futs[f]; r=f.result(); r.update({"prompt_id":i,"prompt":p})
        fh.write(json.dumps(r, ensure_ascii=False)+"\n"); fh.flush()
        n+=1; tok+=(r.get("usage") or {}).get("completion_tokens",0)
        if n%10==0: print(f"[{time.time()-t0:6.0f}s] {n}/{len(todo)} tokens={tok} rate={tok/(time.time()-t0):.0f} tok/s", flush=True)
print("done", n, tok, time.time()-t0)
