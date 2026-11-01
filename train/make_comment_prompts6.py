#!/usr/bin/env python3
"""A sixth independent set of 80 prompts, on data and enterprise systems.

Same eight task types and the same wording as make_comment_prompts.py, but
every snippet, program, concept, module, tool and failing test comes from data
and ML, analytics, ETL, search, recommender systems, ERP and CRM integration,
document workflow, logistics and warehousing.

Ids are u000..u079 so the sets never mix.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
import make_comment_prompts as base  # noqa: E402

base.SNIPPETS = [
    ("python", '''def ndcg_at_k(ranked_ids, relevance, k):
    if k <= 0:
        raise ValueError("k must be positive")
    gains = [relevance.get(doc_id, 0) for doc_id in ranked_ids[:k]]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg'''),
    ("java", '''public List<Pick> allocate(String sku, int qty, List<Bin> bins) {
    if (qty <= 0) throw new IllegalArgumentException("qty must be positive");
    List<Bin> candidates = bins.stream()
        .filter(b -> b.getSku().equals(sku) && b.getAvailable() > 0)
        .sorted(Comparator.comparing(Bin::getExpiresAt))
        .collect(Collectors.toList());
    List<Pick> picks = new ArrayList<>();
    int left = qty;
    for (Bin b : candidates) {
        if (left == 0) break;
        int take = Math.min(left, b.getAvailable());
        picks.add(new Pick(b.getId(), take));
        left -= take;
    }
    if (left > 0) throw new InsufficientStockException(sku, qty - left, qty);
    return picks;
}'''),
    ("csharp", '''public async Task<SyncResult> SyncAccountsAsync(DateTime since, CancellationToken ct)
{
    var result = new SyncResult();
    var accounts = await _crm.GetAccountsModifiedSinceAsync(since, ct);
    foreach (var account in accounts)
    {
        if (string.IsNullOrWhiteSpace(account.Inn))
        {
            result.Skipped.Add(account.Id);
            continue;
        }
        var existing = await _erp.FindCounterpartyByInnAsync(account.Inn, ct);
        if (existing == null)
            await _erp.CreateCounterpartyAsync(Map(account), ct);
        else if (existing.ModifiedAt < account.ModifiedAt)
            await _erp.UpdateCounterpartyAsync(existing.Id, Map(account), ct);
        result.Processed++;
    }
    return result;
}'''),
    ("go", '''func (idx *Index) Search(query string, limit int) []int {
    terms := tokenize(query)
    if len(terms) == 0 || limit <= 0 {
        return nil
    }
    result, ok := idx.postings[terms[0]]
    if !ok {
        return nil
    }
    for _, term := range terms[1:] {
        result = intersect(result, idx.postings[term])
        if len(result) == 0 {
            return nil
        }
    }
    if len(result) > limit {
        result = result[:limit]
    }
    return result
}'''),
    ("typescript", '''export function rerank(
  candidates: Candidate[],
  purchased: Set<string>,
  maxPerCategory: number,
  limit: number,
): Candidate[] {
  const perCategory = new Map<string, number>();
  const result: Candidate[] = [];
  for (const c of [...candidates].sort((a, b) => b.score - a.score)) {
    if (purchased.has(c.itemId)) continue;
    const used = perCategory.get(c.category) ?? 0;
    if (used >= maxPerCategory) continue;
    perCategory.set(c.category, used + 1);
    result.push(c);
    if (result.length === limit) break;
  }
  return result;
}'''),
    ("sql", '''WITH first_orders AS (
    SELECT customer_id, DATE_TRUNC('month', MIN(created_at)) AS cohort
    FROM orders
    GROUP BY customer_id
),
activity AS (
    SELECT DISTINCT o.customer_id, f.cohort,
           DATE_TRUNC('month', o.created_at) AS active_month
    FROM orders o
    JOIN first_orders f ON f.customer_id = o.customer_id
)
SELECT cohort,
       EXTRACT(YEAR FROM AGE(active_month, cohort)) * 12
         + EXTRACT(MONTH FROM AGE(active_month, cohort)) AS month_offset,
       COUNT(*) AS active_customers,
       ROUND(100.0 * COUNT(*) / FIRST_VALUE(COUNT(*)) OVER (
           PARTITION BY cohort ORDER BY active_month), 1) AS retention_pct
FROM activity
GROUP BY cohort, active_month
ORDER BY cohort, active_month;'''),
    ("bash", r'''set -euo pipefail
shopt -s nullglob
for f in "$INBOX"/orders_*.csv; do
  if [ "$(head -n 1 "$f")" != "$EXPECTED_HEADER" ]; then
    mv "$f" "$FAILED"/
    continue
  fi
  if psql "$DB_URL" -v ON_ERROR_STOP=1 -q \
      -c "\copy staging.orders FROM '$f' WITH (FORMAT csv, HEADER true)"; then
    mv "$f" "$DONE"/
  else
    mv "$f" "$FAILED"/
    echo "load failed: $f" >&2
  fi
done'''),
    ("kotlin", '''fun planLoads(orders: List<Order>, capacityKg: Double): List<List<Order>> {
    require(capacityKg > 0) { "capacityKg must be positive" }
    val oversized = orders.filter { it.weightKg > capacityKg }
    if (oversized.isNotEmpty()) {
        throw IllegalArgumentException("orders exceed capacity: ${oversized.map { it.id }}")
    }
    val loads = mutableListOf<MutableList<Order>>()
    val free = mutableListOf<Double>()
    for (order in orders.sortedByDescending { it.weightKg }) {
        val i = free.indexOfFirst { it >= order.weightKg }
        if (i >= 0) {
            loads[i].add(order)
            free[i] -= order.weightKg
        } else {
            loads.add(mutableListOf(order))
            free.add(capacityKg - order.weightKg)
        }
    }
    return loads
}'''),
    ("elixir", '''def transition(%Document{status: status, submitted_by: author} = doc, event, user_id) do
  case {status, event} do
    {:draft, :submit} ->
      {:ok, %{doc | status: :on_review, submitted_by: user_id}}
    {:on_review, :approve} when author == user_id ->
      {:error, :self_approval}
    {:on_review, :approve} ->
      {:ok, %{doc | status: :approved, approved_by: user_id}}
    {:on_review, :reject} ->
      {:ok, %{doc | status: :draft}}
    {:approved, :archive} ->
      {:ok, %{doc | status: :archived}}
    _ ->
      {:error, {:invalid_transition, status, event}}
  end
end'''),
    ("rust", '''pub fn bm25(tf: u32, doc_len: u32, avg_len: f64, df: u32, n_docs: u32) -> f64 {
    const K1: f64 = 1.2;
    const B: f64 = 0.75;
    if tf == 0 || n_docs == 0 {
        return 0.0;
    }
    let idf = ((n_docs as f64 - df as f64 + 0.5) / (df as f64 + 0.5)).ln();
    let norm = 1.0 - B + B * doc_len as f64 / avg_len;
    let tf = tf as f64;
    idf * tf * (K1 + 1.0) / (tf + K1 * norm)
}'''),
]

base.THINGS = [
    ("python", "функцию разбиения временного ряда продаж на обучающую и тестовую выборки без заглядывания в будущее"),
    ("java", "клиент для постраничной выгрузки заказов из ERP-системы с повторными попытками при сбоях"),
    ("csharp", "службу, которая забирает входящие документы из почтового ящика и регистрирует их в системе документооборота"),
    ("go", "сервис автодополнения поисковых запросов на основе префиксного дерева с учётом популярности"),
    ("typescript", "компонент React для отслеживания статуса доставки посылки по трек-номеру с периодическим опросом сервера"),
    ("kotlin", "калькулятор стоимости доставки с учётом зоны, фактического и объёмного веса груза"),
    ("python", "ETL-процесс инкрементальной загрузки сделок из CRM в аналитическое хранилище по отметке времени"),
    ("java", "сервис трёхстороннего сопоставления счетов поставщиков с заказами и приходными накладными"),
    ("elixir", "конвейер Broadway, который читает события просмотров из RabbitMQ и обновляет счётчики популярности товаров"),
    ("rust", "парсер EDI-уведомлений об отгрузке (DESADV) с проверкой обязательных сегментов"),
]

base.CONCEPTS = [
    "как устроен инвертированный индекс в полнотекстовом поиске",
    "что такое коллаборативная фильтрация и чем она отличается от рекомендаций по содержанию",
    "в чём разница между ETL и ELT и когда какой подход выбирать",
    "что такое утечка данных (data leakage) при обучении моделей и как её обнаружить",
    "как работают медленно меняющиеся измерения второго типа (SCD Type 2) в хранилище данных",
    "зачем нужно A/B-тестирование рекомендательных алгоритмов и как не обмануться его результатами",
    "как устроено версионирование документов в системе электронного документооборота",
    "что такое захват изменений данных (CDC) и как с его помощью синхронизируют корпоративные системы",
    "как работает волновой отбор заказов на складе и по каким правилам заказы объединяют в волны",
    "как работает векторный поиск по эмбеддингам и чем он отличается от поиска по ключевым словам",
]

base.MODULES = [
    "модуля дедупликации записей о клиентах при загрузке из нескольких CRM",
    "сервиса ранжирования результатов поиска с учётом персональных предпочтений",
    "компонента маршрутизации входящих документов по цепочке согласования",
    "модуля резервирования товара под заказ на нескольких складах",
    "сервиса построения маршрутов курьеров с учётом временных окон доставки",
    "модуля ночного переобучения модели прогноза спроса с проверкой качества перед развёртыванием",
    "шлюза, принимающего прайс-листы поставщиков в Excel и обновляющего цены в ERP",
    "сервиса формирования персональных подборок товаров для главной страницы магазина",
    "модуля приёмки товара на склад по электронной накладной с контролем расхождений",
    "компонента ежедневного обновления витрины продаж для BI-отчётности",
]

base.TOOLS = [
    "утилиты, оценивающей качество данных в CSV-файлах",
    "библиотеки для офлайн-оценки качества поиска по размеченным запросам",
    "сервиса хранения и раздачи признаков для ML-моделей (feature store)",
    "коннектора для выгрузки данных из 1С в ClickHouse",
    "скрипта сверки числа строк и контрольных сумм между источником и хранилищем после загрузки",
    "набора операторов Airflow для загрузки данных из SAP в хранилище",
    "командной утилиты для переиндексации каталога в Elasticsearch без простоя",
    "библиотеки, формирующей паллетные этикетки со штрихкодом GS1-128",
    "сервиса распознавания и классификации сканов входящих документов",
    "инструмента для отслеживания экспериментов с рекомендательными моделями",
]

base.FAILS = [
    ("java", '''@Test
void allocatesOldestBatchFirst() {
    List<Pick> picks = allocator.allocate("SKU-1", 5, bins);
    assertEquals(List.of(new Pick("A-01", 5)), picks);
}''', "expected: <[Pick@6d86b085]> but was: <[Pick@75828a0f]>"),
    ("python", r'''def test_amount_column_is_integer():
    df = pd.read_csv(io.StringIO("id,amount\n1,10\n2,\n3,7\n"))
    assert df["amount"].dtype == "int64"''', "AssertionError: assert dtype('float64') == 'int64'"),
    ("csharp", '''[Fact]
public void Maps_Contact_Phone()
{
    var c = Mapper.ToCounterparty(new CrmContact { Phone = "8 (912) 345-67-89" });
    Assert.Equal("+79123456789", c.Phone);
}''', 'Assert.Equal() Failure: Strings differ. Expected: "+79123456789", Actual: "89123456789"'),
    ("go", '''func TestSnippet(t *testing.T) {
    s := Snippet("Поставка на склад", 8)
    if s != "Поставка" {
        t.Fatalf("got %q", s)
    }
}''', 'snippet_test.go:14: got "Пост"'),
    ("typescript", '''it('maps CRM lead to deal', () => {
  const deal = toDeal({ id: 7, amount: '1500.00', stage: 'won' });
  expect(deal).toBe({ id: 7, amount: 1500, stage: 'closed_won' });
});''', "expect(received).toBe(expected) // Object.is equality. Received: serializes to the same string"),
    ("kotlin", '''@Test
fun `computes volumetric weight`() {
    val w = volumetricWeight(lengthCm = 45, widthCm = 35, heightCm = 25)
    assertEquals(7.875, w)
}''', "expected: <7.875> but was: <7.0>"),
    ("python", '''def test_split_is_reproducible():
    df = load_events()
    train_a, _ = train_test_split(df, test_size=0.2)
    train_b, _ = train_test_split(df, test_size=0.2)
    assert train_a.index.equals(train_b.index)''', "AssertionError: assert False"),
    ("java", r'''@Test
void parsesErpOrder() throws Exception {
    String json = "{\"id\":42,\"status\":\"SHIPPED\",\"warehouse\":\"MSK-1\"}";
    ErpOrder order = mapper.readValue(json, ErpOrder.class);
    assertEquals(OrderStatus.SHIPPED, order.getStatus());
}''', 'UnrecognizedPropertyException: Unrecognized field "warehouse" (class ErpOrder), not marked as ignorable'),
    ("sql", '''-- ожидается 12 товаров, которые ни разу не продавались
SELECT p.id FROM products p
WHERE p.id NOT IN (SELECT s.product_id FROM sales s);''', "expected 12 rows, got 0 (тест сломался, когда в sales появились строки с пустым product_id)"),
    ("elixir", '''test "approves document on review" do
  doc = %Document{status: :on_review, submitted_by: 1}
  assert {:ok, %Document{status: :approved}} = Workflow.transition(doc, :approve, 1)
end''', "match (=) failed, right: {:error, :self_approval}"),
]


if __name__ == "__main__":
    rows = base.build()
    for r in rows:
        r["id"] = "u" + r["id"][1:]
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print(len(rows), "промптов:", dict(collections.Counter(r["kind"] for r in rows)))
