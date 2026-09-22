"""High-precision lexical defect detector for Russian LLM output.
Categories:
  mixed_script : one word contains both Cyrillic and Latin letters  (unambiguous defect)
  latin_word   : a Latin-script word inside a Russian sentence, not a known term/name
  oov          : Cyrillic word unknown to the morphological dictionary (pymorphy3), lowercase, not a name
"""
import json, re, sys, collections
import pymorphy3
morph=pymorphy3.MorphAnalyzer()
CYR=re.compile(r'[а-яёА-ЯЁ]'); LAT=re.compile(r'[A-Za-z]')
WORD=re.compile(r'[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё-]*')
def strip_code(t):
    t=re.sub(r"```.*?```","\n",t,flags=re.S); t=re.sub(r"`[^`\n]*`"," ",t)
    t=re.sub(r"https?://\S+"," ",t); t=re.sub(r"\S*[/\\@]\S+"," ",t)
    return t
LATIN_OK={"ok","id","pdf","html","css","http","https","api","it","ai","usb","gps","led","tv","pc","sms","wi","fi","email","internet","online","the","de","la","van","von"}
def analyse(text):
    text=strip_code(text)
    out=collections.Counter(); hits=collections.defaultdict(list)
    sents=re.split(r'(?<=[.!?…])\s+', text)
    words_total=0
    for s in sents:
        ws=WORD.findall(s)
        cyr_ws=[w for w in ws if CYR.search(w)]
        words_total+=len(cyr_ws)
        russian_sentence=len(cyr_ws)>=3 and len(cyr_ws)>=0.6*len(ws)
        for w in ws:
            has_c, has_l = bool(CYR.search(w)), bool(LAT.search(w))
            if has_c and has_l:
                # skip legitimate hyphenated hybrids like SIM-карты, IT-специалист, PDF-файл
                if not re.match(r'^[A-Z0-9]{2,6}-[а-яё]', w):
                    out["mixed_script"]+=1; hits["mixed_script"].append((w,s[:160]))
                continue
            if has_l and russian_sentence and w.lower() not in LATIN_OK and len(w)>2 and not w[0].isupper():
                out["latin_word"]+=1; hits["latin_word"].append((w,s[:160])); continue
            if has_c and w.islower() and len(w)>2 and "-" not in w:
                if not morph.word_is_known(w):
                    out["oov"]+=1; hits["oov"].append((w,s[:160]))
    return words_total, out, hits
if __name__=="__main__":
    IN,OUT=sys.argv[1],sys.argv[2]
    tot_words=0; tot=collections.Counter(); allhits=collections.defaultdict(list); ndoc=0
    for line in open(IN):
        try: r=json.loads(line)
        except Exception: continue
        c=r.get("content") or ""
        if not c: continue
        ndoc+=1
        w,cnt,h=analyse(c); tot_words+=w; tot.update(cnt)
        for k,v in h.items(): allhits[k]+= [(x[0],x[1],r.get("prompt_id")) for x in v]
    n=sum(tot.values())
    print(f"{IN}: docs={ndoc} words={tot_words} flags={n} per10k_words={10000*n/max(tot_words,1):.2f} {dict(tot)}")
    json.dump({"docs":ndoc,"words":tot_words,"counts":dict(tot),"hits":{k:v[:4000] for k,v in allhits.items()}}, open(OUT,"w"), ensure_ascii=False)
