import json, sys, time, urllib.request, statistics as st, collections
KEY=sys.argv[1]; OUT=sys.argv[2]
URL=(sys.argv[3] if len(sys.argv)>3 else "http://localhost:8080")+"/v1/chat/completions"
CASES=[
 ("code","Напиши на Python класс LRUCache с методами get и put за O(1), на основе двусвязного списка и словаря. Полная реализация с докстрингами и разбором крайних случаев."),
 ("code","Write a Go HTTP middleware that rate-limits per client IP using a token bucket, with cleanup of stale buckets. Full working code with comments."),
 ("code","Напиши на C++ потокобезопасную очередь с ограниченной ёмкостью на condition_variable. Полный код с шаблонами и обработкой остановки."),
 ("structured","Верни JSON-массив из 25 объектов: id, name, email, department, salary, hired_at. Только валидный JSON, без пояснений."),
 ("structured","Output a JSON object describing a REST API with 12 endpoints: path, method, params (array of {name,type,required}), responses. Valid JSON only."),
 ("structured","Сформируй YAML-конфиг docker-compose для стека из 6 сервисов: api, worker, postgres, redis, nginx, prometheus. Только YAML."),
 ("translation","Переведи на английский, сохраняя абзацы: «Осенний вечер опускался на город медленно, как будто кто-то невидимый гасил свет по одной лампе. Прохожие спешили домой, поднимая воротники. В окнах зажигались первые огни, и в каждом из них была своя история — чья-то радость, чья-то тревога, чьё-то ожидание. Город дышал ровно и устало, как человек после долгого рабочего дня.» Затем переведи обратно на русский и сравни варианты."),
 ("translation","Translate to Russian and then to German, preserving tone: 'The engineer checked the pressure gauge twice before opening the valve. Steam hissed through the pipes, and the old turbine began to turn, slowly at first, then with confidence. He had rebuilt it from scratch over four months, and now it was alive again.'"),
 ("translation","Переведи технический текст на английский: «Система кеширования префиксов позволяет повторно использовать вычисленные состояния ключей и значений для совпадающих начал запросов, что существенно снижает время до первого токена при работе агентов с длинным контекстом.»"),
 ("story","Напиши рассказ примерно на 600 слов: старый смотритель маяка в последнюю ночь перед закрытием станции. Связный литературный текст, без списков."),
 ("story","Напиши рассказ примерно на 600 слов: девочка и её дед чинят старые часы. Связный литературный текст, без списков."),
 ("story","Напиши рассказ примерно на 600 слов: два альпиниста пережидают буран в палатке. Связный литературный текст, без списков."),
]
def call(p):
    body={"model":"qwen-3.8-27b-int","messages":[{"role":"user","content":p}],"max_tokens":1500,
          "temperature":0.0,"presence_penalty":0.0,"reasoning_effort":"none"}
    req=urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Authorization":"Bearer "+KEY,"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r: return json.load(r)
res=[]
for cat,p in CASES:
    for rep in range(3):
        d=call(p); t=d.get("timings",{}); u=d["usage"]
        res.append({"cat":cat,"rep":rep,"tok":u["completion_tokens"],
                    "tps":t.get("predicted_per_second"),
                    "acc":t.get("draft_n_accepted",0)/max(t.get("draft_n",1),1),
                    "rounds":t.get("draft_n",0)})
        print(f"{cat:12} #{rep} {u['completion_tokens']:5d} ток  {t.get('predicted_per_second',0):6.1f} tok/s  accept {100*res[-1]['acc']:5.1f}%", flush=True)
json.dump(res, open(OUT,"w"))
print()
by=collections.defaultdict(list)
for r in res: by[r["cat"]].append(r)
for c in ("code","structured","translation","story"):
    v=by[c]
    print(f"{c:12} {st.mean(x['tps'] for x in v):6.1f} ± {st.pstdev([x['tps'] for x in v]):4.1f} tok/s   accept {100*st.mean(x['acc'] for x in v):5.1f}%")
