#!/usr/bin/env python3
"""A third, independent set of 80 prompts: banking and fintech.

Same eight task types and the same wording as make_comment_prompts.py, but
every snippet, program, concept, module, tool and failing test is new and comes
from one domain -- banks, accounting, payments, settlements, reporting,
balances, taxes and fees, money arithmetic -- so it overlaps neither the first
two sets nor the sets written on other topics.

Ids are r000..r079 so the sets never mix.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
import make_comment_prompts as base  # noqa: E402

base.SNIPPETS = [
    ("python", '''def split_amount(total_kop, weights):
    if total_kop < 0:
        raise ValueError("total must be non-negative")
    s = sum(weights)
    if s <= 0:
        raise ValueError("weights must sum to a positive number")
    parts = [total_kop * w // s for w in weights]
    rest = total_kop - sum(parts)
    order = sorted(range(len(weights)), key=lambda i: total_kop * weights[i] % s, reverse=True)
    for i in order[:rest]:
        parts[i] += 1
    return parts'''),
    ("java", '''public BigDecimal accruedInterest(BigDecimal principal, BigDecimal annualRate,
                                  LocalDate from, LocalDate to) {
    if (principal.signum() <= 0 || !to.isAfter(from)) {
        return BigDecimal.ZERO;
    }
    BigDecimal total = BigDecimal.ZERO;
    for (LocalDate d = from; d.isBefore(to); d = d.plusDays(1)) {
        int daysInYear = d.isLeapYear() ? 366 : 365;
        BigDecimal daily = principal.multiply(annualRate)
            .divide(BigDecimal.valueOf(daysInYear), 10, RoundingMode.HALF_UP);
        total = total.add(daily);
    }
    return total.setScale(2, RoundingMode.HALF_UP);
}'''),
    ("csharp", '''public static void ValidateEntry(IReadOnlyList<Posting> postings)
{
    if (postings.Count < 2)
        throw new ArgumentException("Entry must have at least two postings");
    decimal debit = 0m, credit = 0m;
    foreach (var p in postings)
    {
        if (p.Amount <= 0m)
            throw new ArgumentException($"Non-positive amount on account {p.Account}");
        if (p.Side == Side.Debit) debit += p.Amount;
        else credit += p.Amount;
    }
    if (debit != credit)
        throw new InvalidOperationException($"Unbalanced entry: debit {debit}, credit {credit}");
    if (postings.Select(p => p.Currency).Distinct().Count() > 1)
        throw new InvalidOperationException("Mixed currencies in one entry");
}'''),
    ("go", '''func (l *Ledger) Transfer(from, to *Account, amount int64) error {
    if amount <= 0 || from.ID == to.ID {
        return ErrInvalidTransfer
    }
    first, second := from, to
    if first.ID > second.ID {
        first, second = second, first
    }
    first.mu.Lock()
    defer first.mu.Unlock()
    second.mu.Lock()
    defer second.mu.Unlock()
    if from.Balance < amount {
        return fmt.Errorf("account %d: insufficient funds: have %d, need %d", from.ID, from.Balance, amount)
    }
    from.Balance -= amount
    to.Balance += amount
    return nil
}'''),
    ("typescript", '''export function invoiceTotals(lines: InvoiceLine[], vatRate: number) {
  if (vatRate < 0 || vatRate > 1) throw new RangeError(`vatRate out of range: ${vatRate}`);
  let net = 0;
  let vat = 0;
  for (const line of lines) {
    if (line.qty <= 0) continue;
    const lineNet = Math.round(line.priceKop * line.qty);
    const lineVat = Math.round(lineNet * vatRate);
    net += lineNet;
    vat += lineVat;
  }
  return { net, vat, gross: net + vat };
}'''),
    ("sql", '''SELECT t.account_id,
       date_trunc('month', t.posted_at) AS month,
       SUM(CASE WHEN t.direction = 'D' THEN t.amount ELSE 0 END) AS debit_turnover,
       SUM(CASE WHEN t.direction = 'C' THEN t.amount ELSE 0 END) AS credit_turnover,
       SUM(SUM(CASE WHEN t.direction = 'D' THEN t.amount ELSE -t.amount END))
           OVER (PARTITION BY t.account_id ORDER BY date_trunc('month', t.posted_at)) AS closing_balance
FROM transactions t
WHERE t.posted_at >= DATE '2025-01-01'
  AND t.status <> 'reversed'
GROUP BY t.account_id, date_trunc('month', t.posted_at)
ORDER BY t.account_id, month;'''),
    ("bash", '''set -euo pipefail
day=${1:-$(date -d yesterday +%F)}
file="$EXPORT_DIR/statement_${day}.xml"
if [ ! -s "$file" ]; then
  echo "no statement for $day" >&2
  exit 1
fi
sha256sum "$file" > "$file.sha256"
for attempt in 1 2 3; do
  if printf 'put %s\\nput %s\\n' "$file" "$file.sha256" | sftp -b - "$BANK_HOST"; then
    mv "$file" "$file.sha256" "$ARCHIVE_DIR"/
    exit 0
  fi
  echo "attempt $attempt failed" >&2
  sleep $((attempt * 30))
done
echo "upload of $day failed after 3 attempts" >&2
exit 2'''),
    ("kotlin", '''data class Bracket(val upTo: BigDecimal?, val rate: BigDecimal)

fun progressiveTax(income: BigDecimal, brackets: List<Bracket>): BigDecimal {
    require(income >= BigDecimal.ZERO) { "income must be non-negative" }
    var tax = BigDecimal.ZERO
    var lower = BigDecimal.ZERO
    for (b in brackets) {
        if (income <= lower) break
        val upper = b.upTo?.min(income) ?: income
        tax += (upper - lower) * b.rate
        lower = b.upTo ?: break
    }
    return tax.setScale(0, RoundingMode.HALF_UP)
}'''),
    ("elixir", '''def reconcile(payments, invoices) do
  by_ref = Map.new(invoices, &{&1.ref, &1})
  acc0 = %{matched: [], unmatched: [], mismatched: []}

  Enum.reduce(payments, acc0, fn %{ref: ref, amount: paid} = p, acc ->
    case Map.fetch(by_ref, ref) do
      {:ok, %{amount: ^paid} = inv} ->
        Map.update!(acc, :matched, &[{p, inv} | &1])

      {:ok, inv} ->
        Map.update!(acc, :mismatched, &[{p, inv, paid - inv.amount} | &1])

      :error ->
        Map.update!(acc, :unmatched, &[p | &1])
    end
  end)
end'''),
    ("rust", '''pub fn parse_kopecks(s: &str) -> Result<i64, String> {
    let s = s.trim();
    let (neg, digits) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    let (rub, kop) = digits.split_once('.').unwrap_or((digits, "0"));
    if rub.is_empty() || kop.len() > 2 || !rub.chars().chain(kop.chars()).all(|c| c.is_ascii_digit()) {
        return Err(format!("bad amount: {s}"));
    }
    let kop: i64 = format!("{kop:0<2}").parse().map_err(|e| format!("{e}"))?;
    let rub: i64 = rub.parse().map_err(|e| format!("{e}"))?;
    let total = rub.checked_mul(100).and_then(|v| v.checked_add(kop)).ok_or("overflow")?;
    Ok(if neg { -total } else { total })
}'''),
]

base.THINGS = [
    ("python", "функции, которые проверяют номер банковской карты по алгоритму Луна и маскируют его для логов"),
    ("java", "класс денежной суммы (Money) с валютой, арифметическими операциями и защитой от смешения валют"),
    ("csharp", "класс, формирующий платёжную строку для QR-кода по ГОСТ Р 56042-2014 с проверкой обязательных реквизитов"),
    ("go", "обработчик gRPC-запросов на перевод с проверкой суточного и месячного лимитов клиента"),
    ("typescript", "React-компонент для ввода денежной суммы с группировкой разрядов и двумя знаками после запятой"),
    ("kotlin", "конвертер сумм между валютами по кросс-курсу через рубль с банковским округлением"),
    ("python", "скрипт, который строит оборотно-сальдовую ведомость по счетам из CSV-выгрузки проводок"),
    ("java", "планировщик закрытия операционного дня с переносом незавершённых платежей на следующий рабочий день"),
    ("elixir", "GenServer, который хранит баланс кошелька, резервирует средства и снимает резерв по таймауту"),
    ("rust", "потокобезопасный генератор номеров платёжных документов, уникальных в пределах операционного дня"),
]

base.CONCEPTS = [
    "почему денежные суммы не хранят в типах с плавающей точкой и чем их заменяют",
    "как устроена двойная запись в бухгалтерском учёте и как её отражают в схеме базы данных",
    "что такое банковское округление и чем оно отличается от обычного",
    "как работает двухстадийная оплата картой с удержанием суммы и последующим списанием",
    "что такое сага и как с её помощью проводят перевод между счетами в разных сервисах",
    "что такое PCI DSS и какие ограничения он накладывает на код, работающий с номерами карт",
    "чем отличаются дата операции, дата проводки и дата валютирования",
    "как работает сторнирование и почему проводки в учёте не удаляют",
    "что такое токенизация банковских карт и чем она отличается от шифрования",
    "как считают НДС «сверху» и «в том числе» и где при этом теряются копейки",
]

base.MODULES = [
    "модуля расчёта пеней за просрочку платежа по ключевой ставке Банка России",
    "сервиса приёма переводов через Систему быстрых платежей по номеру телефона",
    "компонента формирования ежедневной выписки по счёту клиента",
    "модуля автоматического списания абонентской платы с привязанной карты",
    "сервиса полных и частичных возвратов платежей покупателям",
    "модуля расчёта амортизации основных средств линейным способом",
    "компонента выявления подозрительных операций по правилам противодействия отмыванию денег",
    "модуля переоценки валютных остатков на конец дня",
    "сервиса управления сменами онлайн-кассы с отчётами об открытии и закрытии смены",
    "модуля разнесения поступившей оплаты по нескольким неоплаченным счетам",
]

base.TOOLS = [
    "утилиты, конвертирующей банковские выписки из формата 1CClientBankExchange в CSV",
    "библиотеки для работы с денежными суммами и валютами на TypeScript",
    "сервиса-заглушки платёжного шлюза для интеграционных тестов",
    "скрипта ежемесячной выгрузки проводок из учётной системы в хранилище данных",
    "линтера, находящего в коде денежные расчёты на float и double",
    "генератора синтетических банковских транзакций для нагрузочного тестирования",
    "утилиты командной строки, проверяющей контрольный ключ расчётного счёта по БИК",
    "библиотеки расчёта НДС для интернет-магазина с поддержкой нескольких ставок",
    "набора дашбордов Grafana для мониторинга платёжного шлюза",
    "Python-клиента для API онлайн-касс с поддержкой чеков коррекции",
]

base.FAILS = [
    ("java", '''@Test
void splitsBillEqually() {
    BigDecimal total = new BigDecimal("100.00");
    BigDecimal share = total.divide(BigDecimal.valueOf(3));
    assertEquals(new BigDecimal("33.33"), share);
}''', "ArithmeticException: Non-terminating decimal expansion; no exact representable decimal result."),
    ("python", '''from decimal import Decimal

def format_amount(value):
    return f"{value:,.2f}".replace(",", " ")

def test_format_amount():
    assert format_amount(Decimal("1234567.5")) == "1 234 567,50"''',
     "AssertionError: assert '1 234 567.50' == '1 234 567,50'"),
    ("csharp", '''[Fact]
public void Third_Payment_Falls_On_31st()
{
    var first = new DateTime(2024, 1, 31);
    var third = first.AddMonths(1).AddMonths(1);
    Assert.Equal(new DateTime(2024, 3, 31), third);
}''', "Assert.Equal() Failure: Expected: 2024-03-31T00:00:00.0000000, Actual: 2024-03-29T00:00:00.0000000"),
    ("go", '''func TestBusinessDateLateEvening(t *testing.T) {
    ts := time.Date(2025, 3, 10, 22, 30, 0, 0, time.UTC)
    got := BusinessDate(ts).Format("2006-01-02")
    if got != "2025-03-11" {
        t.Errorf("BusinessDate(%v) = %s, want 2025-03-11", ts, got)
    }
}''', "date_test.go:14: BusinessDate(2025-03-10 22:30:00 +0000 UTC) = 2025-03-10, want 2025-03-11 "
          "(операционный день банка определяется по московскому времени)"),
    ("typescript", '''it('sums payments from the API', () => {
  const body = '[{"amount":"100.50"},{"amount":"200"}]';
  const payments: { amount: number }[] = JSON.parse(body);
  const total = payments.reduce((s, p) => s + p.amount, 0);
  expect(total).toBe(300.5);
});''', "Expected: 300.5, Received: \"0100.50200\""),
    ("kotlin", '''fun toKopecks(rubles: Int): Long = (rubles * 100).toLong()

@Test
fun `converts large amount to kopecks`() {
    assertEquals(3_000_000_000L, toKopecks(30_000_000))
}''', "expected: <3000000000> but was: <-1294967296>"),
    ("python", '''from decimal import Decimal

def test_fee_is_one_and_a_half_percent():
    amount = Decimal("100.00")
    fee = amount * 0.015
    assert fee == Decimal("1.50")''',
     "TypeError: unsupported operand type(s) for *: 'decimal.Decimal' and 'float'"),
    ("java", '''@Test
void dailyLimitReachedExactly() {
    Long limit = 5_000L;
    Long spent = List.of(2_000L, 3_000L).stream().reduce(0L, Long::sum);
    assertTrue(spent == limit, "limit must be reached");
}''', "limit must be reached ==> expected: <true> but was: <false>"),
    ("sql", '''-- ожидается строка на каждый счёт; у счетов без операций за март — 0
SELECT a.id, COALESCE(SUM(t.amount), 0) AS march_total
FROM accounts a
LEFT JOIN transactions t ON t.account_id = a.id
WHERE t.posted_at >= DATE '2025-03-01' AND t.posted_at < DATE '2025-04-01'
GROUP BY a.id;''', "expected 120 rows, got 87"),
    ("elixir", '''test "extracts amount from webhook payload" do
  body = ~s({"amount": 1500, "currency": "RUB"})
  payload = Jason.decode!(body)
  %{amount: amount, currency: currency} = payload
  assert {amount, currency} == {1500, "RUB"}
end''', "** (MatchError) no match of right hand side value: %{\"amount\" => 1500, \"currency\" => \"RUB\"}"),
]


if __name__ == "__main__":
    rows = base.build()
    for r in rows:
        r["id"] = "r" + r["id"][1:]
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print(len(rows), "промптов:", dict(collections.Counter(r["kind"] for r in rows)))
