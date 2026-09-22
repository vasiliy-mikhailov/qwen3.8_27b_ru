import json, sys, urllib.request, math
URL=sys.argv[1]; OUT=sys.argv[2]; ITEMS=sys.argv[3] if len(sys.argv)>3 else 'items2.json'
items=json.load(open(ITEMS))
INTRO="В небольшом городе у реки жила семья: отец, мать и двое детей. Каждый день у них происходило что-то новое. "
def frame_chat(t): return "<|im_start|>user\nНапиши короткий рассказ.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"+INTRO+t
def frame_raw(t): return INTRO+t
def topk(prompt):
    body={"prompt":prompt,"n_predict":1,"n_probs":20,"temperature":0,"cache_prompt":False}
    req=urllib.request.Request(URL+"/completion", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=300) as r: d=json.load(r)
    return [(t["token"], math.exp(t["logprob"])) for t in d["completion_probabilities"][0]["top_logprobs"]]
res=[]; agg={"chat":[], "raw":[]}
for it in items:
    rec={"word":it["word"],"ending":it["ending"]}
    for name,fr in [("chat",frame_chat),("raw",frame_raw)]:
        tp=topk(fr(it["prefix"])); probs=dict(tp)
        pc=probs.get(it["ending"],0.0)
        rank=next((i+1 for i,(t,p) in enumerate(tp) if t==it["ending"]), None)
        perr={c:probs.get(c,0.0) for c in it["competitors"]}
        rec[name]={"top":tp[:5],"p_correct":pc,"rank":rank,"p_err":perr,"err_mass":sum(perr.values())}
        agg[name].append((pc, rank, sum(perr.values())))
    res.append(rec); c=rec["chat"]
    print(f"{it['word']:14} {it['ending']!r:8} chat: rank={c['rank']} p={c['p_correct']:.3f} err={c['err_mass']:.3f} {[(k,round(v,3)) for k,v in c['p_err'].items() if v>0.001]} top1={c['top'][0][0]!r} | raw: rank={rec['raw']['rank']} p={rec['raw']['p_correct']:.3f} err={rec['raw']['err_mass']:.3f}", flush=True)
for name in agg:
    a=agg[name]; n=len(a)
    print(f"== {name}: rank1={sum(1 for x in a if x[1]==1)}/{n}  mean p_correct={sum(x[0] for x in a)/n:.3f}  mean err_mass={sum(x[2] for x in a)/n:.4f}  max err_mass={max(x[2] for x in a):.3f}")
json.dump(res, open(OUT,'w'), ensure_ascii=False, indent=1)
