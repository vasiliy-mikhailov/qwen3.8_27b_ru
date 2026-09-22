"""Agreement error detector v2: spaCy for sentence split/POS/dep, pymorphy3 for agreement consistency.
A pair is flagged only if NO combination of pymorphy parses of the two words agrees."""
import json, re, sys, collections
import spacy, pymorphy3
IN=sys.argv[1]; OUT=sys.argv[2]
nlp=spacy.load("ru_core_news_lg", disable=["ner"]); morph=pymorphy3.MorphAnalyzer()
CYR=re.compile(r'^[а-яёА-ЯЁ-]+$')
ADJ_POS={"ADJF","ADJS","PRTF","PRTS","NUMR"}; NOUN_POS={"NOUN","NPRO"}
COMMON_GENDER={"врач","судья","коллега","сирота","староста","глава","доктор","профессор","директор","автор","педагог","бухгалтер","инженер","юрист","повар","президент","мастер","гость","человек","ребёнок","ребенок","малыш"}
def parses(w, pos_set=None):
    ps=[p for p in morph.parse(w) if p.score>=0.02 or True]
    if pos_set: ps=[p for p in ps if p.tag.POS in pos_set]
    return ps
DET_LEMMAS={"свой","этот","тот","каждый","какой","весь","мой","твой","наш","ваш","один","сам","самый","такой","который","чей","любой","другой","иной","некоторый","этакий","всякий","данный"}
COMPLEMENT_ADJ={"полный","полон","лишённый","лишенный","достойный","чуждый","исполненный","наполненный","насыщенный","богатый","довольный","занятый","покрытый","залитый","освещённый","освещенный","окружённый","окруженный","увлечённый","увлеченный","известный","знаменитый","видимый","незаметный","заметный","знакомый","подверженный","свойственный","верный","благодарный","равный","подобный","доступный","понятный","нужный","обязанный","готовый","способный","похожий","близкий","чужой","родной","верен","готов","способен","похож","известен","достоин","равен","подобен","доступен","понятен","нужен","обязан","занят","полна","полно","полны","чуждый","открытый","закрытый","обращённый","обращенный","повёрнутый","повернутый","привыкший","связанный","обусловленный","вызванный","пропитанный","пахнущий","пахнувший","отмеченный","охваченный","объятый","защищённый","защищенный","отделённый","отделенный","свободный","далёкий","далекий","чреватый","обременённый","обремененный","наделённый","наделенный","одержимый","озабоченный","предан","преданный","удивлённый","удивленный","огорчённый","огорченный","обрадованный","испуганный","встревоженный","озадаченный","измученный","уставший","усталый","опьянённый","опьяненный","заворожённый","завороженный","очарованный","ослеплённый","ослепленный","оглушённый","оглушенный","тронутый","растроганный","поражённый","пораженный"}
def _case(c):
    return {"loc2":"loct","gen2":"gent","acc2":"accs"}.get(c,c)
def agree_adj_noun(adj, noun):
    """Return None if agreement is possible, else 'gender' or 'case'."""
    ap=parses(adj, ADJ_POS); np_=parses(noun, NOUN_POS)
    if not ap or not np_: return None
    adj_lemma=ap[0].normal_form
    same_slot=[]   # pairs agreeing in number and case
    for a in ap:
        for n in np_:
            if a.tag.number and n.tag.number and a.tag.number!=n.tag.number: continue
            ac,nc=_case(a.tag.case),_case(n.tag.case)
            if ac and nc and ac!=nc and not ({ac,nc}=={"accs","gent"} and ("anim" in n.tag or "anim" in a.tag)): continue
            same_slot.append((a,n))
    if same_slot:
        for a,n in same_slot:
            if n.tag.number=="plur" or a.tag.number=="plur": return None
            if not a.tag.gender or not n.tag.gender or a.tag.gender==n.tag.gender: return None
        return "gender"
    # no same-slot pair: case/number mismatch. Flag only for determiners or when noun cannot be an oblique complement
    if adj_lemma in COMPLEMENT_ADJ: return None
    noun_cases={_case(n.tag.case) for n in np_}
    if adj_lemma in DET_LEMMAS or noun_cases<= {"nomn","accs"}: return "case"
    return None
def agree_subj_pred(subj, pred):
    """None if agreement possible; 'gender' if only gender blocks it; 'number' if number blocks it (subject nominative-only)."""
    sp=[p for p in parses(subj) if p.tag.POS=="NOUN"]; pp=parses(pred)
    if not sp or not pp: return None
    nom_only=all(_case(p.tag.case)=="nomn" for p in sp)
    num_ok=False
    for s_ in sp:
        for p in pp:
            if p.tag.POS not in ("VERB","ADJS","PRTS"): continue
            if s_.tag.number and p.tag.number and s_.tag.number!=p.tag.number: continue
            num_ok=True
            g_p = p.tag.gender if (p.tag.POS in ("ADJS","PRTS") or p.tag.tense=="past") else None
            if s_.tag.number=="sing" and g_p and s_.tag.gender and g_p!=s_.tag.gender: continue
            if p.tag.POS=="VERB" and p.tag.person and s_.tag.person and p.tag.person!=s_.tag.person: continue
            return None
    if num_ok: return "gender" if nom_only or True else None
    return "number" if nom_only else None
def check_sent(sent):
    flags=[]; checked=collections.Counter()
    toks=list(sent)
    for k,t in enumerate(toks):
        if not CYR.match(t.text): continue
        # (1) adjacent adjective-like word(s) before a noun: adj [adj|,|и]* noun
        if t.pos_ in ("ADJ","DET","NUM") or (t.pos_=="PRON" and t.dep_ in ("det","amod")):
            j=k+1
            while j<len(toks) and j-k<=3 and (toks[j].pos_ in ("ADJ","ADV") or toks[j].text in (",","и")): j+=1
            if j<len(toks) and toks[j].pos_ in ("NOUN","PROPN") and CYR.match(toks[j].text) and (t.head.i==toks[j].i or (t.head.pos_=="ADJ" and t.head.head.i==toks[j].i)):
                n=toks[j]
                if n.lemma_.lower() in COMMON_GENDER: continue
                if t.pos_=="NUM" or t.lemma_.lower() in ("два","две","три","четыре","оба","обе","полтора"): continue
                if k>0 and toks[k-1].lemma_.lower() in ("что","всё","все") and t.lemma_.lower() in ("такой","больший","меньший"): continue
                # skip adjectives after numerals 2-4 ("два больших дома")
                if k>0 and toks[k-1].pos_=="NUM": continue
                if k>0 and toks[k-1].text=="-" or (k+1<len(toks) and toks[k+1].text=="-"): continue
                if j>0 and toks[j-1].text=="-": continue
                checked["adj_noun"]+=1
                r=agree_adj_noun(t.text, n.text)
                if r: flags.append(("adj_noun_"+r,t.text,n.text))
        # (2) subject - predicate
        if t.dep_=="nsubj" and t.pos_ in ("NOUN","PROPN") and "-" not in t.text and t.head.pos_ in ("VERB","AUX","ADJ"):
            h=t.head
            if not CYR.match(h.text) or abs(h.i-t.i)>6: continue
            if any(c.dep_ in ("conj","nummod","nummod:gov") for c in t.children) or any(c.dep_=="conj" and c.pos_=="VERB" for c in h.children): continue
            if t.lemma_.lower() in COMMON_GENDER or t.lemma_.lower() in ("кто","что","это","всё","все","вы","они","мы","я","ты"): continue
            if h.pos_=="ADJ" and "Short" not in str(h.morph): continue
            if h.lemma_.lower() in ("нет","нужно","надо","можно","нельзя","должно","видно","слышно","свойственно","стать","оказаться","казаться","хотеться","мочь","звать","становиться","хватать","быть") : continue
            sp_=[p for p in morph.parse(t.text) if p.tag.POS=="NOUN"]
            if sp_ and all(p.tag.case in ("gent","datv","accs","ablt","loct") for p in sp_): continue  # no nominative parse -> not a real subject
            if h.pos_ in ("VERB","AUX") and "Fin" not in str(h.morph): continue
            checked["subj_pred"]+=1
            r=agree_subj_pred(t.text, h.text)
            if r: flags.append(("subj_pred_"+r,t.text,h.text))
    return flags, checked
rows=[json.loads(l) for l in open(IN)]
per=collections.defaultdict(lambda: {"sents":0,"words":0,"checked":collections.Counter(),"flags":collections.Counter(),"examples":[]})
seen=set(); texts=[]
def clean(text):
    text=re.sub(r"```.*?```","\n",text,flags=re.S); text=re.sub(r"`[^`\n]*`"," ",text); text=re.sub(r"https?://\S+"," ",text)
    return re.sub(r"[*_#>|]+"," ",text)
for r in rows:
    try: o=json.loads(r["output"])
    except Exception: continue
    c=o.get("content") if isinstance(o,dict) else None
    if isinstance(c,list): c=" ".join(p.get("text","") for p in c if isinstance(p,dict))
    if not isinstance(c,str) or not re.search(r"[а-яё]",c,re.I): continue
    texts.append((r["model"], clean(c)))
for (model,text),doc in zip(texts, nlp.pipe((t for m,t in texts), batch_size=16, n_process=6)):
    for sent in doc.sents:
        cyr=[t for t in sent if CYR.match(t.text)]
        if len(cyr)<4: continue
        key=(model,sent.text.strip())
        if key in seen: continue
        seen.add(key); pm=per[model]; pm["sents"]+=1; pm["words"]+=len(cyr)
        flags,checked=check_sent(sent); pm["checked"].update(checked)
        for f in flags:
            pm["flags"][f[0]]+=1; pm["examples"].append({"rule":f[0],"a":f[1],"b":f[2],"sent":sent.text.strip()[:300]})
res={}
for m,pm in per.items():
    tf=sum(pm["flags"].values())
    res[m]={"sents":pm["sents"],"words":pm["words"],"checked":dict(pm["checked"]),"flags":dict(pm["flags"]),"per10k_words":10000*tf/max(pm["words"],1),"examples":pm["examples"]}
    print(f"{m:28} words={pm['words']:7d} checked={sum(pm['checked'].values()):6d} flags={tf:4d} per10k_words={res[m]['per10k_words']:.2f} {dict(pm['flags'])}")
json.dump(res, open(OUT,"w"), ensure_ascii=False, indent=1)
