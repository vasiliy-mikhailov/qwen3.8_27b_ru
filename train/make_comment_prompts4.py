#!/usr/bin/env python3
"""A fourth, independent set of 80 prompts: web, mobile and DevOps.

Same eight task types and the same wording as make_comment_prompts.py, but
every snippet, program, concept, module, tool and failing test is new and
comes from frontend, UI, browser APIs, mobile apps, CI/CD, containers, clouds
and infrastructure as code, so it does not overlap the other sets.

Ids are s000..s079 so the sets never mix.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
import make_comment_prompts as base  # noqa: E402

base.SNIPPETS = [
    ("python", '''UNITS = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40,
         "k": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}


def parse_memory(quantity):
    if isinstance(quantity, (int, float)):
        return int(quantity)
    quantity = quantity.strip()
    if not quantity:
        raise ValueError("empty memory quantity")
    for suffix in sorted(UNITS, key=len, reverse=True):
        if quantity.endswith(suffix):
            return int(float(quantity[:-len(suffix)]) * UNITS[suffix])
    if quantity.endswith("m"):
        return int(quantity[:-1]) // 1000
    return int(float(quantity))'''),
    ("java", '''public void applyCors(HttpServletRequest req, HttpServletResponse resp) {
    String origin = req.getHeader("Origin");
    if (origin == null) {
        return;
    }
    boolean allowed = allowedOrigins.contains(origin)
        || (allowSubdomains && origin.endsWith("." + baseDomain));
    if (!allowed) {
        resp.setStatus(HttpServletResponse.SC_FORBIDDEN);
        return;
    }
    resp.setHeader("Access-Control-Allow-Origin", origin);
    resp.setHeader("Access-Control-Allow-Credentials", "true");
    resp.addHeader("Vary", "Origin");
    if ("OPTIONS".equalsIgnoreCase(req.getMethod())) {
        resp.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE");
        resp.setHeader("Access-Control-Max-Age", "600");
        resp.setStatus(HttpServletResponse.SC_NO_CONTENT);
    }
}'''),
    ("csharp", '''public static string NextVersion(string current, IReadOnlyList<string> commitTitles)
{
    var parts = current.TrimStart('v').Split('.').Select(int.Parse).ToArray();
    if (parts.Length != 3)
        throw new FormatException($"Bad version: {current}");
    var (major, minor, patch) = (parts[0], parts[1], parts[2]);
    if (commitTitles.Any(t => t.Contains("BREAKING CHANGE") || t.Split(':')[0].EndsWith("!")))
        return major == 0 ? $"v0.{minor + 1}.0" : $"v{major + 1}.0.0";
    if (commitTitles.Any(t => t.StartsWith("feat")))
        return $"v{major}.{minor + 1}.0";
    if (commitTitles.Count == 0)
        return current;
    return $"v{major}.{minor}.{patch + 1}";
}'''),
    ("go", '''func run(srv *http.Server, timeout time.Duration) error {
    errCh := make(chan error, 1)
    go func() {
        if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
            errCh <- err
        }
    }()
    stop := make(chan os.Signal, 1)
    signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
    select {
    case err := <-errCh:
        return err
    case <-stop:
    }
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()
    return srv.Shutdown(ctx)
}'''),
    ("typescript", '''export function lazyLoadImages(root: ParentNode = document, margin = "200px"): () => void {
  const images = Array.from(root.querySelectorAll<HTMLImageElement>("img[data-src]"));
  const load = (img: HTMLImageElement) => {
    img.src = img.dataset.src!;
    img.removeAttribute("data-src");
  };
  if (!("IntersectionObserver" in window)) {
    images.forEach(load);
    return () => {};
  }
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      load(entry.target as HTMLImageElement);
      observer.unobserve(entry.target);
    }
  }, { rootMargin: margin });
  images.forEach((img) => observer.observe(img));
  return () => observer.disconnect();
}'''),
    ("sql", '''SELECT d.service,
       COUNT(*) AS deploys,
       COUNT(*) FILTER (WHERE d.status = 'failed') AS failed,
       ROUND(100.0 * COUNT(*) FILTER (WHERE d.status = 'failed') / COUNT(*), 1) AS failure_rate,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY d.finished_at - d.started_at) AS median_duration
FROM deployments d
WHERE d.environment = 'production'
  AND d.started_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY d.service
HAVING COUNT(*) >= 5
ORDER BY failure_rate DESC, deploys DESC;'''),
    ("bash", '''deploy_and_verify() {
  local image="$1" url="$2" attempts="${3:-30}"
  kubectl set image deployment/web web="$image" || return 1
  if ! kubectl rollout status deployment/web --timeout=180s; then
    echo "rollout failed, rolling back" >&2
    kubectl rollout undo deployment/web
    return 1
  fi
  for ((i = 1; i <= attempts; i++)); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "$url/healthz")
    if [ "$code" = "200" ]; then
      echo "healthy after $i attempt(s)"
      return 0
    fi
    sleep 2
  done
  echo "health check failed: last status $code" >&2
  kubectl rollout undo deployment/web
  return 1
}'''),
    ("kotlin", '''sealed interface Route {
    data class Product(val id: Long) : Route
    data class Search(val query: String, val page: Int) : Route
    object Home : Route
}

fun parseDeepLink(uri: Uri): Route {
    if (uri.scheme != "shop" && uri.host != "shop.example.com") return Route.Home
    val segments = uri.pathSegments
    return when (segments.firstOrNull()) {
        "product" -> segments.getOrNull(1)?.toLongOrNull()?.let { Route.Product(it) } ?: Route.Home
        "search" -> {
            val q = uri.getQueryParameter("q").orEmpty().trim()
            val page = uri.getQueryParameter("page")?.toIntOrNull()?.coerceAtLeast(1) ?: 1
            if (q.isEmpty()) Route.Home else Route.Search(q, page)
        }
        else -> Route.Home
    }
}'''),
    ("elixir", '''def call(conn, opts) do
  with ["Bearer " <> token] <- get_req_header(conn, "authorization"),
       {:ok, claims} <- Token.verify(token, opts[:secret]),
       true <- claims["exp"] > System.system_time(:second) do
    assign(conn, :current_user_id, claims["sub"])
  else
    false ->
      conn
      |> put_resp_header("www-authenticate", ~s(Bearer error="invalid_token"))
      |> send_resp(401, "token expired")
      |> halt()

    _ ->
      conn
      |> send_resp(401, "unauthorized")
      |> halt()
  end
end'''),
    ("rust", '''pub fn parse_image(s: &str) -> Option<(&str, &str, Option<&str>, Option<&str>)> {
    let (rest, digest) = match s.split_once('@') {
        Some((r, d)) if d.starts_with("sha256:") => (r, Some(d)),
        Some(_) => return None,
        None => (s, None),
    };
    let (name, tag) = match rest.rsplit_once(':') {
        Some((n, t)) if !t.contains('/') => (n, Some(t)),
        _ => (rest, None),
    };
    let (registry, repository) = match name.split_once('/') {
        Some((r, p)) if r.contains('.') || r.contains(':') || r == "localhost" => (r, p),
        _ => ("docker.io", name),
    };
    if repository.is_empty() || tag == Some("") {
        return None;
    }
    Some((registry, repository, tag, digest))
}'''),
]

base.THINGS = [
    ("python", "скрипт, который проверяет TLS-сертификаты списка доменов и заранее предупреждает об истечении срока их действия"),
    ("java", "фильтр Spring Security, который проверяет JWT-токен и заполняет контекст безопасности"),
    ("csharp", "обработчик Azure Functions, который по таймеру удаляет устаревшие снимки дисков"),
    ("go", "контроллер Kubernetes, который перезапускает поды при изменении ConfigMap, на который они ссылаются"),
    ("typescript", "сервис-воркер, который кэширует статику и показывает офлайн-страницу при отсутствии сети"),
    ("kotlin", "экран Jetpack Compose с формой входа, проверкой полей и индикатором загрузки"),
    ("python", "сервис FastAPI, рассылающий push-уведомления в браузеры и удаляющий недействительные подписки"),
    ("java", "задачу Gradle, которая собирает Docker-образ и присваивает ему тег с хешем текущего коммита"),
    ("elixir", "канал Phoenix для чата с отслеживанием присутствия пользователей в комнате"),
    ("rust", "HTTP-сервер, который раздаёт одностраничное приложение и на запросы к неизвестным маршрутам отвечает файлом index.html"),
]

base.CONCEPTS = [
    "как работает критический путь рендеринга в браузере",
    "как устроен цикл событий (event loop) в браузере и чем микрозадачи отличаются от макрозадач",
    "в чём разница между сине-зелёным и канареечным развёртыванием",
    "что такое инфраструктура как код и чем декларативный подход отличается от императивного",
    "зачем нужны пробы liveness и readiness в Kubernetes и чем они отличаются",
    "как устроены слои Docker-образа и почему порядок инструкций в Dockerfile влияет на скорость сборки",
    "что такое гидратация при серверном рендеринге и откуда берутся ошибки несовпадения разметки",
    "что такое состояние (state) в Terraform и зачем его блокировать при совместной работе",
    "как устроен жизненный цикл Activity в Android и почему при повороте экрана теряется состояние",
    "как устроен конвейер CI/CD и чем непрерывная доставка отличается от непрерывного развёртывания",
]

base.MODULES = [
    "модуля аутентификации мобильного приложения с обновлением токенов доступа",
    "сервиса выдачи временных учётных данных для доступа к облачному хранилищу",
    "компонента автодополнения адреса в форме оформления заказа",
    "модуля офлайн-синхронизации заметок между мобильным приложением и сервером",
    "конвейера развёртывания, который откатывает релиз при росте доли ошибок",
    "компонента загрузки изображений с предпросмотром и сжатием на стороне браузера",
    "модуля автомасштабирования воркеров по длине очереди задач",
    "модуля проверки обязательного обновления мобильного приложения при запуске",
    "контроллера GitOps, синхронизирующего состояние кластера с Git-репозиторием",
    "компонента переключения между светлой и тёмной темами с учётом системных настроек",
]

base.TOOLS = [
    "утилиты, проверяющей Dockerfile на соответствие внутренним правилам",
    "библиотеки UI-компонентов на React с поддержкой программ чтения с экрана и навигации с клавиатуры",
    "модуля Terraform, разворачивающего статический сайт в S3 с раздачей через CloudFront",
    "Helm-чарта, разворачивающего веб-приложение вместе с Redis и PostgreSQL",
    "набора повторно используемых workflow GitHub Actions, собирающих и публикующих Docker-образы",
    "утилиты, запускающей мобильное приложение сразу на нескольких эмуляторах",
    "плагина Vite, оптимизирующего SVG-иконки при сборке",
    "сервиса, создающего временное окружение для каждого запроса на слияние",
    "консольной утилиты, показывающей логи сразу нескольких подов Kubernetes",
    "библиотеки, собирающей метрики Web Vitals и отправляющей их в систему аналитики",
]

base.FAILS = [
    ("java", '''@Test
void serializesCreatedAt() throws Exception {
    Event e = new Event("deploy", LocalDate.of(2024, 5, 1));
    String json = new ObjectMapper().writeValueAsString(e);
    assertTrue(json.contains("\\"createdAt\\":\\"2024-05-01\\""));
}''', "InvalidDefinitionException: Java 8 date/time type `java.time.LocalDate` not supported by default"),
    ("python", '''def test_create_item(client):
    resp = client.post("/api/items", data={"name": "lamp"})
    assert resp.status_code == 201
    assert resp.get_json()["name"] == "lamp"''',
     "AssertionError: assert 415 == 201 (обработчик читает тело через request.get_json())"),
    ("csharp", '''[Fact]
public void Deserializes_Settings()
{
    var json = "{\\"theme\\":\\"dark\\",\\"fontSize\\":14}";
    var s = JsonSerializer.Deserialize<UiSettings>(json);
    Assert.Equal("dark", s!.Theme);
}''', "Assert.Equal() Failure: Strings differ, Expected: \"dark\", Actual: null"),
    ("go", '''func TestStatusIsJSON(t *testing.T) {
    rec := httptest.NewRecorder()
    statusHandler(rec, httptest.NewRequest(http.MethodGet, "/api/status", nil))
    ct := rec.Result().Header.Get("Content-Type")
    if ct != "application/json" {
        t.Errorf("Content-Type = %q, want %q", ct, "application/json")
    }
}''', "status_test.go:14: Content-Type = \"text/plain; charset=utf-8\", want \"application/json\" "
          "(в обработчике w.Header().Set вызывается после w.Write)"),
    ("typescript", '''it('shows the menu button on small screens', () => {
  Object.defineProperty(window, 'innerWidth', { value: 375, configurable: true });
  render(<Header />);
  expect(screen.getByRole('button', { name: 'Открыть меню' })).toBeInTheDocument();
});''', "TypeError: window.matchMedia is not a function "
          "(шапка выбирает раскладку через хук useMediaQuery)"),
    ("kotlin", '''@Test
fun `loading flag is reset after refresh`() = runTest {
    val vm = OrdersViewModel(FakeOrdersRepository())
    vm.refresh()
    assertFalse(vm.state.value.isLoading)
}''', "IllegalStateException: Module with the Main dispatcher had failed to initialize. "
      "For tests Dispatchers.setMain from kotlinx-coroutines-test module can be used"),
    ("python", '''def test_port_from_env(monkeypatch):
    monkeypatch.setenv("PORT", "9000")
    from app.config import settings
    assert settings.port == 9000''',
     "AssertionError: assert 8000 == 9000 (если запустить этот тест отдельно, он проходит)"),
    ("java", '''@Test
void createsOrder() throws Exception {
    mockMvc.perform(post("/api/orders")
            .contentType(MediaType.APPLICATION_JSON)
            .content("{\\"sku\\":\\"A-1\\",\\"qty\\":2}"))
        .andExpect(status().isCreated());
}''', "AssertionError: Status expected:<201> but was:<403> (в проекте подключён Spring Security)"),
    ("sql", '''-- ожидаются все узлы, на которых версия агента ниже 1.10.0
SELECT hostname, agent_version FROM nodes
WHERE agent_version < '1.10.0';''',
     "expected 7 rows, got 2 (столбец agent_version имеет тип text)"),
    ("elixir", '''setup do
  user = insert(:user, email: "anna@example.com")
  {:ok, conn: build_conn(), user: user}
end

test "GET /api/profile returns current user", %{conn: conn} do
  conn = get(conn, "/api/profile")
  assert %{"email" => "anna@example.com"} = json_response(conn, 200)
end''', "** (RuntimeError) expected response with status 200, got: 401, with body: \"unauthorized\""),
]


if __name__ == "__main__":
    rows = base.build()
    for r in rows:
        r["id"] = "s" + r["id"][1:]
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print(len(rows), "промптов:", dict(collections.Counter(r["kind"] for r in rows)))
