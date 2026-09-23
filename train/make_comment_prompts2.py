#!/usr/bin/env python3
"""A second, independent set of 80 prompts for replicating a comparison.

Same eight task types and the same wording as make_comment_prompts.py, so the
two sets measure the same thing, but every snippet, program, concept, module,
tool and failing test is new. A difference between two quantisations seen on
the first set and not on this one was luck of the draw: greedy decoding turns
any change of rounding into a different text within the first few words, so
80 answers are one sample, not a property of the model.

Ids are q000..q079 so the sets never mix.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
import make_comment_prompts as base  # noqa: E402

base.SNIPPETS = [
    ("python", '''def chunked(items, size):
    if size <= 0:
        raise ValueError("size must be positive")
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch'''),
    ("java", '''public BigDecimal applyDiscount(BigDecimal price, int percent) {
    if (percent < 0 || percent > 100) throw new IllegalArgumentException("percent");
    BigDecimal k = BigDecimal.valueOf(100 - percent).divide(BigDecimal.valueOf(100));
    return price.multiply(k).setScale(2, RoundingMode.HALF_EVEN);
}'''),
    ("csharp", '''public static IEnumerable<T> Distinct<T, K>(this IEnumerable<T> src, Func<T, K> key)
{
    var seen = new HashSet<K>();
    foreach (var x in src)
        if (seen.Add(key(x)))
            yield return x;
}'''),
    ("go", '''func (c *Cache) Get(key string) (string, bool) {
    c.mu.RLock()
    e, ok := c.items[key]
    c.mu.RUnlock()
    if !ok || time.Now().After(e.expires) {
        return "", false
    }
    return e.value, true
}'''),
    ("typescript", '''export function debounce<T extends (...a: any[]) => void>(fn: T, ms: number) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return (...args: Parameters<T>) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}'''),
    ("sql", '''SELECT d.name, COUNT(e.id) AS staff, AVG(e.salary) AS avg_salary
FROM departments d
LEFT JOIN employees e ON e.department_id = d.id AND e.fired_at IS NULL
GROUP BY d.name
HAVING COUNT(e.id) > 0
ORDER BY avg_salary DESC;'''),
    ("bash", '''for f in "$SRC"/*.log; do
  [ -e "$f" ] || continue
  if [ "$(find "$f" -mtime +7)" ]; then
    gzip -9 "$f" && mv "$f.gz" "$ARCHIVE"/
  fi
done'''),
    ("kotlin", '''fun <T> retry(times: Int, block: () -> T): T {
    var last: Throwable? = null
    repeat(times) {
        try { return block() } catch (e: IOException) { last = e }
    }
    throw last ?: IllegalStateException("times must be > 0")
}'''),
    ("elixir", '''def handle_info(:tick, state) do
  expired = for {id, t} <- state.sessions, t < now(), do: id
  sessions = Map.drop(state.sessions, expired)
  Process.send_after(self(), :tick, @interval)
  {:noreply, %{state | sessions: sessions}}
end'''),
    ("rust", '''pub fn median(v: &mut Vec<f64>) -> Option<f64> {
    if v.is_empty() { return None; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    Some(if v.len() % 2 == 0 { (v[m - 1] + v[m]) / 2.0 } else { v[m] })
}'''),
]

base.THINGS = [
    ("python", "загрузчик курсов валют с сайта ЦБ с кэшированием на диске"),
    ("java", "обработчик вебхуков платёжной системы с проверкой подписи"),
    ("csharp", "middleware ASP.NET Core, которое логирует медленные запросы"),
    ("go", "пул воркеров, обрабатывающий задачи из канала с ограничением параллелизма"),
    ("typescript", "клиент WebSocket с автоматическим переподключением и очередью исходящих сообщений"),
    ("kotlin", "валидатор ИНН и ОГРН с понятными сообщениями об ошибках"),
    ("python", "скрипт миграции данных между двумя схемами PostgreSQL пачками"),
    ("java", "кэш с вытеснением по давности использования (LRU) и ограничением по памяти"),
    ("elixir", "супервизор, перезапускающий упавшие соединения с внешним API с задержкой"),
    ("rust", "парсер аргументов командной строки для утилиты сжатия файлов"),
]

base.CONCEPTS = [
    "что такое событийная согласованность (eventual consistency) и чем она отличается от строгой",
    "как работает шардирование базы данных и как выбирают ключ шардирования",
    "зачем нужны feature flags и какие у них подводные камни",
    "что такое deadlock и как его избегать в многопоточном коде",
    "как устроен двухфазный коммит и почему его избегают в микросервисах",
    "что такое CQRS и когда он оправдан",
    "как работают миграции схемы базы данных без простоя",
    "чем отличаются модульные, интеграционные и сквозные тесты",
    "что такое circuit breaker и как он защищает систему",
    "как устроено логирование со сквозным идентификатором запроса",
]

base.MODULES = [
    "модуля расчёта графика платежей по кредиту",
    "сервиса бронирования мест с защитой от двойной продажи",
    "компонента выгрузки отчётности в налоговую в формате XML",
    "модуля начисления бонусных баллов за покупки",
    "сервиса отправки уведомлений по электронной почте с шаблонами",
    "модуля блокировки учётной записи после неудачных попыток входа",
    "сервиса конвертации загруженных документов в PDF",
    "компонента сверки остатков между складом и учётной системой",
    "модуля ограничения доступа к API по ключам и квотам",
    "сервиса поиска по каталогу товаров с опечатками",
]

base.TOOLS = [
    "утилиты для поиска неиспользуемых зависимостей в Maven-проекте",
    "линтера SQL-миграций, запрещающего опасные операции",
    "генератора changelog по истории коммитов",
    "сервиса, собирающего время сборок CI и строящего графики",
    "скрипта очистки старых Docker-образов в реестре",
    "библиотеки для сравнения скриншотов в UI-тестах",
    "командной утилиты для шифрования секретов в конфигурации",
    "планировщика нагрузочных тестов по расписанию",
    "прокси для записи и воспроизведения HTTP-запросов в тестах",
    "инструмента проверки битых ссылок в документации",
]

base.FAILS = [
    ("java", '''@Test
void parsesAmount() {
    BigDecimal a = Parser.amount("1 234,50");
    assertEquals(new BigDecimal("1234.5"), a);
}''', "expected: <1234.5> but was: <1234.50>"),
    ("python", '''def test_sorted_users():
    users = load_users()
    names = [u.name for u in users]
    assert names == sorted(names)''', "AssertionError: ['Ёлкин', 'Абрамов'] != ['Абрамов', 'Ёлкин']"),
    ("csharp", '''[Fact]
public void Formats_Date()
{
    var s = Formatter.Short(new DateTime(2024, 3, 5));
    Assert.Equal("05.03.2024", s);
}''', "Expected: 05.03.2024, Actual: 3/5/2024"),
    ("go", '''func TestReadConfig(t *testing.T) {
    cfg, err := ReadConfig("testdata/app.yaml")
    if err != nil { t.Fatal(err) }
    if cfg.Port != 8080 { t.Fatalf("port %d", cfg.Port) }
}''', "open testdata/app.yaml: no such file or directory"),
    ("typescript", '''it('rounds price', () => {
  expect(formatPrice(10.005)).toBe('10.01');
});''', "Expected: \"10.01\", Received: \"10.00\""),
    ("kotlin", '''@Test
fun `user is adult`() {
    val u = User(birthDate = LocalDate.of(2006, 9, 24))
    assertTrue(u.isAdult())
}''', "expected: <true> but was: <false> (тест падает только в определённые дни)"),
    ("python", '''def test_retry_calls_three_times(mocker):
    api = mocker.Mock(side_effect=TimeoutError)
    with pytest.raises(TimeoutError):
        fetch_with_retry(api)
    assert api.call_count == 3''', "AssertionError: assert 4 == 3"),
    ("java", '''@Test
void savesInTransaction() {
    service.transfer(a, b, 100);
    assertEquals(900, accounts.balance(a));
}''', "LazyInitializationException: could not initialize proxy - no Session"),
    ("sql", '''-- ожидается 1 строка на каждого клиента
SELECT c.id, o.total FROM clients c JOIN orders o ON o.client_id = c.id;''', "expected 50 rows, got 137"),
    ("elixir", '''test "sends welcome email" do
  Accounts.register(%{email: "a@b.ru"})
  assert_received {:email, "a@b.ru"}
end''', "No message matching {:email, \"a@b.ru\"} was received (письмо отправляется из Task)"),
]


if __name__ == "__main__":
    rows = base.build()
    for r in rows:
        r["id"] = "q" + r["id"][1:]
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print(len(rows), "промптов:", dict(collections.Counter(r["kind"] for r in rows)))
