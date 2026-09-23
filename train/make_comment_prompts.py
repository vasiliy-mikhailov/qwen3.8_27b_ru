#!/usr/bin/env python3
"""Prompts that make the model write Russian in the register it is bad at.

Eight task types drawn from how the model is actually used here -- writing code
with Russian comments, commenting existing code, docstrings, reviews, specs,
READMEs, tests with scenario comments, explaining failures. Ten variants each,
spread over the languages that appear in the real traffic.

Every prompt demands Russian explicitly, so a reply in English is itself a
failure rather than a way around the measurement.
"""
import json, sys

SNIPPETS = [
    ("python", '''def merge(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    return out + a[i:] + b[j:]'''),
    ("java", '''public Optional<User> findActive(String login) {
    return users.stream()
        .filter(u -> u.getLogin().equalsIgnoreCase(login))
        .filter(u -> !u.isBlocked() && u.getExpiresAt().isAfter(Instant.now()))
        .findFirst();
}'''),
    ("csharp", '''public async Task<int> RetryAsync(Func<Task<int>> action, int attempts)
{
    for (var i = 0; i < attempts; i++)
    {
        try { return await action(); }
        catch (HttpRequestException) when (i < attempts - 1)
        {
            await Task.Delay(TimeSpan.FromMilliseconds(200 * Math.Pow(2, i)));
        }
    }
    throw new InvalidOperationException("unreachable");
}'''),
    ("go", '''func worker(jobs <-chan Job, results chan<- Result, wg *sync.WaitGroup) {
    defer wg.Done()
    for j := range jobs {
        r, err := process(j)
        if err != nil {
            log.Printf("job %d: %v", j.ID, err)
            continue
        }
        results <- r
    }
}'''),
    ("typescript", '''export function debounce<T extends (...a: any[]) => void>(fn: T, ms: number) {
  let t: ReturnType<typeof setTimeout> | undefined;
  return (...args: Parameters<T>) => {
    if (t) clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}'''),
    ("sql", '''SELECT c.id, c.name, SUM(o.total) AS revenue
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.id AND o.status = 'paid'
WHERE c.created_at >= NOW() - INTERVAL '1 year'
GROUP BY c.id, c.name
HAVING SUM(o.total) > 1000
ORDER BY revenue DESC;'''),
    ("bash", '''for f in "$SRC"/*.log; do
  [ -s "$f" ] || continue
  gzip -c "$f" > "$DST/$(basename "$f").gz" && rm -f "$f"
done
find "$DST" -name '*.gz' -mtime +30 -delete'''),
    ("kotlin", '''fun <K, V> MutableMap<K, V>.getOrPutSafe(key: K, factory: () -> V): V =
    synchronized(this) { this[key] ?: factory().also { this[key] = it } }'''),
    ("elixir", '''def handle_info(:tick, state) do
  {due, rest} = Enum.split_with(state.jobs, &(&1.at <= System.monotonic_time(:millisecond)))
  Enum.each(due, &send(&1.pid, {:run, &1.id}))
  Process.send_after(self(), :tick, 100)
  {:noreply, %{state | jobs: rest}}
end'''),
    ("rust", '''pub fn parse_kv(s: &str) -> HashMap<String, String> {
    s.split(';')
        .filter_map(|p| p.split_once('='))
        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
        .collect()
}'''),
]

THINGS = [
    ("python", "класс LRU-кэша с ограничением по времени жизни записей"),
    ("java", "сервис, который читает сообщения из Kafka и сохраняет их в PostgreSQL с идемпотентностью"),
    ("csharp", "фоновую службу, которая раз в минуту проверяет доступность HTTP-эндпоинтов"),
    ("go", "HTTP-обработчик загрузки файла с проверкой размера и типа"),
    ("typescript", "хук React для постраничной загрузки данных с отменой запросов"),
    ("kotlin", "репозиторий для работы с пользователями поверх Spring Data"),
    ("python", "парсер CSV-выписки банка с проверкой контрольных сумм"),
    ("java", "ограничитель частоты запросов (rate limiter) по алгоритму token bucket"),
    ("elixir", "GenServer, который накапливает события и сбрасывает их пачками"),
    ("rust", "итератор по строкам большого файла без загрузки его целиком в память"),
]

CONCEPTS = [
    "мутационное тестирование и что такое выжившие мутанты",
    "идемпотентность обработки сообщений в очередях",
    "разница между оптимистичной и пессимистичной блокировкой в базе данных",
    "как работает сборка мусора с поколениями",
    "что такое backpressure в потоковой обработке",
    "паттерн Outbox для надёжной публикации событий",
    "зачем нужны индексы в базе данных и когда они вредят",
    "как устроена конкурентность в Go через каналы",
    "что такое покрытие кода по строкам и по ветвям и почему одного мало",
    "как работает кэширование HTTP-ответов через ETag",
]

MODULES = [
    "модуля приёма платёжных поручений из очереди сообщений",
    "сервиса выдачи одноразовых кодов подтверждения по SMS",
    "компонента, отправляющего события в систему мониторинга",
    "модуля импорта справочника контрагентов из XML",
    "шлюза, пересылающего REST-запросы во внутреннюю шину без изменений",
    "планировщика отложенных задач с повторными попытками",
    "модуля проверки подписи входящих документов",
    "сервиса расчёта комиссии по тарифной сетке",
    "компонента архивирования логов старше заданного срока",
    "модуля синхронизации статусов заказов между двумя системами",
]

TOOLS = [
    "утилиты командной строки для сравнения двух JSON-файлов",
    "библиотеки для генерации тестовых данных (фейков) на Java",
    "сервиса, собирающего метрики покрытия кода по всем репозиториям",
    "скрипта резервного копирования базы PostgreSQL в S3",
    "плагина Gradle, проверяющего запрещённые зависимости",
    "бота, публикующего результаты сборок в рабочий чат",
    "инструмента для массового переименования тестовых методов по конвенции",
    "обёртки над HTTP-клиентом с повторными попытками и таймаутами",
    "локального стенда из Kafka и PostgreSQL в Docker Compose",
    "генератора отчётов о мутационном тестировании в HTML",
]

FAILS = [
    ("java", '''@Test
void shouldReturnUser() {
    when(repo.findByLogin("ivan")).thenReturn(Optional.of(user));
    User u = service.find("Ivan");
    assertEquals("ivan", u.getLogin());
}''', "NoSuchElementException: No value present"),
    ("python", '''def test_total():
    cart = Cart()
    cart.add(Item("book", 0.1))
    cart.add(Item("pen", 0.2))
    assert cart.total() == 0.3''', "AssertionError: assert 0.30000000000000004 == 0.3"),
    ("csharp", '''[Fact]
public async Task Saves_Order()
{
    var svc = new OrderService(_db);
    svc.SaveAsync(new Order { Id = 1 });
    Assert.Equal(1, _db.Orders.Count());
}''', "Assert.Equal() Failure: Expected 1, Actual 0"),
    ("go", '''func TestCounter(t *testing.T) {
    c := Counter{}
    for i := 0; i < 1000; i++ { go c.Inc() }
    if c.Value() != 1000 { t.Fatalf("got %d", c.Value()) }
}''', "counter_test.go:9: got 973"),
    ("typescript", '''it('loads data', () => {
  const { result } = renderHook(() => useData('/api/items'));
  expect(result.current.items).toHaveLength(3);
});''', "Expected length: 3, Received length: 0"),
    ("kotlin", '''@Test
fun `parses date`() {
    val d = parse("01/02/2024")
    assertEquals(Month.FEBRUARY, d.month)
}''', "expected:<FEBRUARY> but was:<JANUARY>"),
    ("python", '''def test_cache_expires():
    c = TTLCache(ttl=1)
    c.set("k", 1)
    time.sleep(1)
    assert c.get("k") is None''', "AssertionError: assert 1 is None (intermittent)"),
    ("java", '''@Test
void sendsEventOnce() {
    handler.handle(message);
    handler.handle(message);
    verify(publisher, times(1)).publish(any());
}''', "TooManyActualInvocations: wanted 1 time, but was 2 times"),
    ("sql", '''-- тест ожидает 2 строки
SELECT * FROM orders WHERE status = NULL;''', "expected 2 rows, got 0"),
    ("elixir", '''test "stores event" do
  Store.put(%{id: 1})
  assert Store.all() == [%{id: 1}]
end''', "left: [%{id: 1}, %{id: 1}] (другой тест оставил данные)"),
]


def build():
    out = []
    for i, (lang, thing) in enumerate(THINGS):
        out.append(("write_code", f"Напиши на {lang} {thing}. Снабди код подробными комментариями "
                    f"на русском языке: к каждому публичному методу — документирующий комментарий, "
                    f"внутри — пояснения к неочевидным местам."))
    for lang, code in SNIPPETS:
        out.append(("comment_code", f"Вот код на {lang}:\n```{lang}\n{code}\n```\nДобавь к нему комментарии "
                    f"на русском языке, объясняющие, что и зачем делает каждый фрагмент. Верни код целиком."))
    for lang, code in SNIPPETS:
        out.append(("docstring", f"Напиши подробный документирующий комментарий (docstring / javadoc / xmldoc — "
                    f"как принято в языке) на русском языке для этой функции на {lang}. Опиши назначение, "
                    f"параметры, возвращаемое значение, исключения и граничные случаи.\n```{lang}\n{code}\n```"))
    for lang, code in SNIPPETS:
        out.append(("review", f"Проведи код-ревью этого фрагмента на {lang}. Напиши замечания на русском "
                    f"языке: что может сломаться, что неочевидно, что стоит улучшить.\n```{lang}\n{code}\n```"))
    for c in CONCEPTS:
        out.append(("explain", f"Объясни по-русски, {c}, для разработчика, который раньше с этим не сталкивался. "
                    f"Приведи небольшой пример кода с комментариями на русском языке."))
    for m in MODULES:
        out.append(("spec", f"Напиши на русском языке раздел технической спецификации: поведение {m}. "
                    f"Опиши входные данные, результат, обработку ошибок и граничные случаи. Связный текст, "
                    f"можно с таблицами."))
    for t in TOOLS:
        out.append(("readme", f"Напиши на русском языке README для {t}: назначение, установка, пример "
                    f"использования, ограничения."))
    for lang, code, err in FAILS:
        out.append(("explain_failure", f"Объясни по-русски, почему этот тест падает с ошибкой «{err}», "
                    f"и как его исправить. Исправленный код снабди комментариями на русском.\n```{lang}\n{code}\n```"))
    return [{"id": f"p{i:03d}", "kind": k, "prompt": p} for i, (k, p) in enumerate(out)]


if __name__ == "__main__":
    rows = build()
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print(len(rows), "промптов:", dict(collections.Counter(r["kind"] for r in rows)))
