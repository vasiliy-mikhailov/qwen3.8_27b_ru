# Русскоязычные наборы: бенчмарки и данные для файн-тюна

Дополнение к [PLAN.md](PLAN.md). Колонка «Роль» — где набор используется в плане:
**A** генеративная частота ошибок, **B/C** logprob-диагностика и minimal pairs, **E** guard-метрики,
**Calib** калибровка (M2), **FT** обучение (M3/M4/M5).

## 1. Грамматика и морфология (прямо под задачу)

| Набор | Что это | Размер | Роль | Ссылка |
|---|---|---|---|---|
| **RuBLiMP** | minimal pairs «правильно/неправильно», 45 парадигм, 12 явлений: согласование прил.–сущ., подл.–сказ., падежное управление, число/род, вид/время, рефлексивы. Построен из реальных корпусных предложений (86k уникальных токенов, а не шаблоны). | 45k пар | **C** (основной logprob-бенчмарк), B (разметка позиций окончаний) | https://github.com/RussianNLP/RuBLiMP , https://aclanthology.org/2024.emnlp-main.522/ |
| **RuCoLA** | бинарная приемлемость предложений; неприемлемые размечены по типу нарушения: морфология / синтаксис / семантика / галлюцинации. 9.8k in-domain (лингвистические публикации) + 3.6k out-of-domain (сгенерированные моделями). | 13.4k | C (доп. набор), калибровка детекторов | https://github.com/RussianNLP/RuCoLA , https://huggingface.co/datasets/RussianNLP/rucola |
| **RULEC-GEC** | тексты изучающих русский с разметкой ошибок (согласование, падеж, управление). 206k токенов, 13k исправлений. | 12k предл. | калибровка/обучение детектора ошибок для **A** | https://github.com/arozovskaya/RULEC-GEC |
| **ai-forever/spellcheck_benchmark** (RUSpellRU, MultidomainGold, MedSpellchecker, GitHubTypoCorpusRu) | орфография/пунктуация, не морфология; полезно только чтобы отделить опечатки от грамматики. | ~10k | вспомогательный | https://huggingface.co/datasets/ai-forever/spellcheck_benchmark |
| **MERA → USE** (ЕГЭ по русскому) | задания ЕГЭ, включая грамматические нормы. Проверяет *знание* грамматики, а не корректность генерации — слабая корреляция с нашей проблемой, но дёшево. | часть MERA | E | https://mera.a-ai.ru |

Инструменты-детекторы для метрики A (не датасеты, но нужны рядом):
- **SAGE** (ai-forever) — генеративная коррекция орфографии/пунктуации, модели `sage-fredt5-large`, `sage-m2m100`; можно использовать как «сколько правок делает корректор» → грубая метрика. https://github.com/ai-forever/sage
- **LanguageTool-ru** — правила согласования; низкий recall, высокий precision.
- **pymorphy3 + spaCy `ru_core_news_lg` / Natasha** — для парсерного детектора согласования (стиль L'AMBRE).
- **POLLUX LM-judges** (7B на T-lite, 32B на T-pro) — критерии «Грамотность» и «Отсутствие речевых ошибок» с текстовым обоснованием. https://arxiv.org/abs/2505.24616

## 2. Генеративные бенчмарки с судьёй (качество русского текста)

| Набор | Что это | Размер | Роль | Ссылка |
|---|---|---|---|---|
| **POLLUX** | 2115 экспертных промптов, 35 типов задач, 66 критериев, среди них «Грамотность». Reference-free оценка LM-судьёй. Готовый пул промптов для **A** + готовый судья. | 2.1k | **A** | https://arxiv.org/abs/2505.24616 , HF: ai-forever |
| **ru-arena-hard / ru_llm_arena** (Vikhr, t-tech) | Arena-Hard-Auto на русском, парные сравнения с судьёй. | 500 промптов | A (парное сравнение артефактов), E | https://github.com/VikhrModels/ru_llm_arena , https://huggingface.co/datasets/t-tech/ru-arena-hard |
| **WildChat Hard Ru** | реальные пользовательские русские запросы (kuk/wildchat-hard-ru). Источник «естественных» промптов. | — | A (промпты) | GitHub: kuk/wildchat-hard-ru |
| **RuSimulBench, RusConText** | симуляционные/контекстные задачи, вторичны. | — | — | https://aclanthology.org/2025.acl-srw.91.pdf |

## 3. Знания и reasoning (guard: чтобы фиксы не портили модель)

| Набор | Что это | Роль | Ссылка |
|---|---|---|---|
| **MERA** (ai-forever) | 23 текстовых задания: ruMMLU (14k), MaMuRAMu, RWSD, CheGeKa, ruHumanEval/ruCodeEval, SimpleAr, LCS, BPS, USE, ruEthics/ruHHH/ruDetox и др.; есть lm-evaluation-harness. | **E** | https://github.com/MERA-Evaluation/MERA , https://huggingface.co/datasets/ai-forever/MERA |
| **ruMMLU-Pro** (t-tech) | перевод MMLU-Pro | E | https://huggingface.co/datasets/t-tech/ruMMLU-pro |
| **ruAIME-2024/2025, ruMATH-500, T-Math** (t-tech) | математика на русском; T-Math — 331 олимпиадная задача | E (аналог AIME из README ninfer, но по-русски) | https://huggingface.co/datasets/t-tech/ru-reasoning-benchmarks , https://huggingface.co/datasets/t-tech/T-math |
| **DOoM** (Vikhr) | математика + физика на русском | E | https://huggingface.co/Vikhrmodels |
| **LIBRA** | 21 задача long-context 4k–128k на русском | E (если трогаем KV-формат) | https://arxiv.org/abs/2408.02439 |
| **Russian SuperGLUE, TAPE** | классификационные, устаревшие для LLM | не используем | https://arxiv.org/abs/2010.15925 |

## 4. SFT / instruct-данные (для M3/M4/M5)

| Набор | Что это | Размер | Заметки |
|---|---|---|---|
| **t-tech/T-Wix** | SFT-корпус T-pro 2.0: ~468k общих (math/code/science/instruction/knowledge/writing) + ~30k reasoning с трассами; ODC-By. Самый крупный и свежий (дек. 2025). | 500k | лучший кандидат на основу ru-части смеси для **M4**; reasoning-трассы важны для thinking-режима Qwen3.8 |
| **Vikhrmodels/GrandMaster-PRO-MAX** | агрегированные инструкции, код + общие знания | 150k | FT |
| **IlyaGusev/rulm — Saiga**: ru_turbo_saiga, ru_turbo_alpaca, ru_sharegpt_cleaned, oasst1_ru_main_branch, gpt_roleplay_realm, ru_turbo_alpaca_evol_instruct | классические self-instruct наборы 2023 г.; качество русского языка среднее (переводы) | ~100–200k | FT, но фильтровать: переводные тексты сами содержат корявые окончания |
| **RuAdapt** (Tikhomirov, Chernyshev) | смесь переводных и нативных сэмплов | — | FT |
| **oasst1/oasst2 (ru-ветки)** | человеческие диалоги | небольшой | FT, высокое качество языка |

Для дистилляции (M4) метки — логиты BF16-учителя, так что качество *ответов* в датасете вторично;
важны только промпты и покрытие морфологии. NVIDIA QAD показывает, что достаточно даже
self-generated ответов. Значит практичный рецепт: промпты из T-Wix + WildChat-Ru + POLLUX-like,
ответы генерирует сам Qwen3.8 BF16 (или квантованный на mh — быстро), учитель даёт логиты.

## 5. Сырые корпуса (Calib для M2, RU-corpus для PPL/B, pretraining-mix для M4)

| Корпус | Что это | Размер (ru) | Заметки |
|---|---|---|---|
| **FineWeb-2 `rus_Cyrl`** | фильтрованный веб, есть quality-скоры | сотни млрд токенов | основной источник для Calib и M4-смеси; брать high-quality шарды |
| **HPLT 3.0 `rus_Cyrl`** | веб, шарды отсортированы по качеству | очень большой | альтернатива FineWeb-2 |
| **CulturaX ru** | mC4+OSCAR очищенный | большой | использован в Bielik-Q2-Sharp как калибровка (для польского) — прямой аналог |
| **Taiga** | новости, худлит (Proza.ru, ЛитРес), научпоп (N+1), соцсети, субтитры, стихи | ~5 млрд токенов | жанровое разнообразие → хорош для RU-corpus (метрика B) и калибровки: худлит и научпоп дают редкие парадигмы. Лицензию подкорпуса ЛитРес проверить перед использованием |
| **ruWiki** | энциклопедия | ~1 млрд | стандарт для PPL |
| **Russian National Corpus / GICR** | размеченные корпуса; лицензия ограничена | — | только для ручного анализа |
| **UD_Russian-SynTagRus / Taiga treebanks** | деревья зависимостей с морфологией | 1.2M / 0.2M токенов | обучение/проверка парсерного детектора для **A**, извлечение правил согласования (как в L'AMBRE) |

Рекомендуемый калибровочный набор для M2: 1024 фрагмента × 2048 токенов, состав
ru 60 % (FineWeb-2 hq 30 %, Taiga худлит+новости 20 %, T-Wix ответы 10 %) / en 20 % / zh+code 20 %.
RU-corpus для PPL/B (1M токенов): те же источники, другие документы, без пересечения с калибровкой.

## 6. Чего НЕ делать (частые ошибочные советы)
- **GEC-пары (RULEC-GEC и т.п.) как данные для QAD/дистилляции.** QAD — это KL к логитам BF16-учителя на
  обычном тексте; пары «ошибка → исправление» туда не вставляются. Как plain text учебные тексты
  иностранцев вредны (модель выучит ошибки). Единственное место GEC-данных — детекторы для метрики A
  и, при необходимости, SFT-задача «исправь текст» (M5).
- **«Сложный синтаксис» в калибровке.** Калибровка AWQ/GPTQ подбирает скейлы по статистике активаций;
  важны языковое соответствие и разнообразие, а не длина синтаксических связей. У ninfer groupwise-int
  калибровки нет вовсе, NVFP4 импортируется готовым — калибровочный набор нужен только в M2.
- **LiDiRus / Russian SuperGLUE** — NLI-диагностика логики и entailment, к морфологии окончаний не относится.
- **SynTagRus как «эталонный текст» для обучения** — ~1 M токенов, в основном новости 2000-х; ценность
  только в разметке (правила согласования для детектора, золотые признаки для метрики B).

## 7. Что не нашлось / непроверено
- «RusGEC от Russian-NLP» — не найден; из открытых GEC для русского подтверждён только RULEC-GEC.
- «Shlepa» (Vikhr) — упоминается в сообществе, в поиске не подтверждён; не закладываем.
- ruIFEval — есть у Vikhr/t-tech внутренне, публичный релиз не подтверждён; вместо него IFBench из README ninfer.
- Отдельного открытого датасета ошибок *в генерациях LLM* по морфологии на русском нет — придётся
  делать свою ручную разметку (100–200 текстов), см. PLAN.md §1.3.

## Источники
- T-pro 2.0 — https://arxiv.org/html/2512.10430
- Vikhr — https://arxiv.org/abs/2405.13929 ; rulm — https://github.com/IlyaGusev/rulm
- MERA — https://arxiv.org/abs/2401.04531 ; https://github.com/MERA-Evaluation/MERA
- RuBLiMP — https://aclanthology.org/2024.emnlp-main.522/
- RuCoLA — https://arxiv.org/abs/2210.12814
- RULEC-GEC — https://github.com/arozovskaya/RULEC-GEC
- POLLUX — https://arxiv.org/abs/2505.24616
- LIBRA — https://arxiv.org/abs/2408.02439
- SAGE — https://github.com/ai-forever/sage ; spellcheck_benchmark — https://huggingface.co/datasets/ai-forever/spellcheck_benchmark
- FineWeb-2 — https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 ; HPLT 3.0 — https://huggingface.co/datasets/HPLT/HPLT3.0 ; CulturaX — https://huggingface.co/datasets/uonlp/CulturaX
- Taiga / список корпусов — https://tatianashavrina.github.io/2018/08/30/datasets/
- Russian SuperGLUE — https://arxiv.org/abs/2010.15925
