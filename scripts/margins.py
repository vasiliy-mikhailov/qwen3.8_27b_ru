import json, math, sys
items=json.load(open('items2.json'))
def ninfer_margins(path):
    d=json.load(open(path)); nll={s["id"]:s["total_nll"] for s in d["streams"]}
    out=[]
    for i,it in enumerate(items):
        ok=nll[f"i{i:02d}_ok"]
        errs=[nll[f"i{i:02d}_err{j}"] for j in range(len(it["competitors"]))]
        # margin_j = log p(ok) - log p(err_j) = NLL(err_j) - NLL(ok)   (shared prefix)
        m=[e-ok for e in errs]
        err_ratio=sum(math.exp(-x) for x in m)   # sum p(err)/p(ok)
        out.append((it["word"], min(m), err_ratio, m))
    return out
def llama_margins(path, frame="raw"):
    d=json.load(open(path)); out=[]
    for it,rec in zip(items,d):
        r=rec[frame]; pc=r["p_correct"]; floor=min(p for t,p in r["top"]) if r["top"] else 1e-6
        ms=[math.log(pc)-math.log(max(r["p_err"][c],1e-9)) for c in it["competitors"]]
        out.append((it["word"], min(ms), r["err_mass"]/max(pc,1e-9), ms))
    return out
gw=ninfer_margins('report_gw.json'); q4=llama_margins('topk2_q4kxl.json')
print(f"{'word':14} {'ninfer-GW min-margin':>20} {'err/ok':>8} | {'Q4_K_XL min-margin':>18} {'err/ok':>8}")
for a,b in zip(gw,q4):
    print(f"{a[0]:14} {a[1]:20.2f} {a[2]:8.4f} | {b[1]:18.2f} {b[2]:8.4f}")
import statistics as st
print("\nninfer-GW : mean min-margin %.2f nats, min %.2f, mean err/ok %.4f, #margin<3: %d/%d" % (st.mean(x[1] for x in gw), min(x[1] for x in gw), st.mean(x[2] for x in gw), sum(1 for x in gw if x[1]<3), len(gw)))
print("Q4_K_XL   : mean min-margin %.2f nats, min %.2f, mean err/ok %.4f, #margin<3: %d/%d (competitors outside top-20 floored at 1e-9)" % (st.mean(x[1] for x in q4), min(x[1] for x in q4), st.mean(x[2] for x in q4), sum(1 for x in q4 if x[1]<3), len(q4)))
