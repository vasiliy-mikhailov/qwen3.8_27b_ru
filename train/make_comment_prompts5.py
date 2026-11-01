#!/usr/bin/env python3
"""A fifth, independent set of 80 prompts in the format of make_comment_prompts.py.

Same eight task types and the same wording, but every snippet, program,
concept, module, tool and failing test is new and comes from systems
programming, networks and protocols, file systems, embedded devices, games,
graphics, multimedia and telemetry.

Ids are t000..t079 so the sets never mix.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
import make_comment_prompts as base  # noqa: E402

base.SNIPPETS = [
    ("python", '''def read_frames(sock, max_len=65536):
    buf = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            if buf:
                raise ConnectionError("connection closed mid-frame")
            return
        buf += chunk
        while len(buf) >= 4:
            (n,) = struct.unpack(">I", buf[:4])
            if n > max_len:
                raise ValueError(f"frame too large: {n}")
            if len(buf) < 4 + n:
                break
            yield buf[4:4 + n]
            buf = buf[4 + n:]'''),
    ("java", '''public long updateRto(long sampleMs) {
    if (sampleMs < 0) {
        throw new IllegalArgumentException("negative RTT sample");
    }
    if (srtt < 0) {
        srtt = sampleMs;
        rttvar = sampleMs / 2;
    } else {
        long delta = Math.abs(srtt - sampleMs);
        rttvar = (3 * rttvar + delta) / 4;
        srtt = (7 * srtt + sampleMs) / 8;
    }
    long rto = srtt + Math.max(granularityMs, 4 * rttvar);
    return Math.min(Math.max(rto, 1000L), 60000L);
}'''),
    ("csharp", '''public void Tick(double frameSeconds)
{
    if (frameSeconds > MaxFrame)
        frameSeconds = MaxFrame;
    _accumulator += frameSeconds;
    var steps = 0;
    while (_accumulator >= Step)
    {
        _world.Update(Step);
        _accumulator -= Step;
        if (++steps >= MaxStepsPerFrame)
        {
            _accumulator = 0;
            break;
        }
    }
    _renderer.Draw(_world, _accumulator / Step);
}'''),
    ("go", '''func WriteFileAtomic(path string, data []byte, perm os.FileMode) error {
    tmp, err := os.CreateTemp(filepath.Dir(path), ".tmp-*")
    if err != nil {
        return err
    }
    defer os.Remove(tmp.Name())
    if _, err = tmp.Write(data); err == nil {
        err = tmp.Sync()
    }
    if cerr := tmp.Close(); err == nil {
        err = cerr
    }
    if err == nil {
        err = os.Chmod(tmp.Name(), perm)
    }
    if err != nil {
        return err
    }
    return os.Rename(tmp.Name(), path)
}'''),
    ("typescript", '''export function parseRtpHeader(buf: Uint8Array) {
  if (buf.length < 12) throw new RangeError("packet shorter than RTP header");
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const b0 = view.getUint8(0);
  if (b0 >> 6 !== 2) throw new Error(`unsupported RTP version ${b0 >> 6}`);
  const csrcCount = b0 & 0x0f;
  let offset = 12 + csrcCount * 4;
  if (b0 & 0x10) {
    if (buf.length < offset + 4) throw new RangeError("truncated header extension");
    offset += 4 + view.getUint16(offset + 2) * 4;
  }
  return {
    marker: (view.getUint8(1) & 0x80) !== 0,
    payloadType: view.getUint8(1) & 0x7f,
    sequence: view.getUint16(2),
    timestamp: view.getUint32(4),
    ssrc: view.getUint32(8),
    payloadOffset: offset,
  };
}'''),
    ("sql", '''SELECT device_id,
       ts AS gap_start,
       next_ts AS gap_end,
       next_ts - ts AS gap
FROM (
    SELECT device_id,
           ts,
           LEAD(ts) OVER (PARTITION BY device_id ORDER BY ts) AS next_ts
    FROM telemetry
    WHERE ts >= NOW() - INTERVAL '1 day'
) t
WHERE next_ts - ts > INTERVAL '5 minutes'
ORDER BY gap DESC
LIMIT 100;'''),
    ("bash", '''img="$1"
dev="$2"
[ -f "$img" ] || { echo "no image: $img" >&2; exit 1; }
[ -b "$dev" ] || { echo "not a block device: $dev" >&2; exit 1; }
if grep -q "^$dev" /proc/mounts; then
  echo "$dev is mounted" >&2
  exit 1
fi
size=$(stat -c %s "$img")
dd if="$img" of="$dev" bs=4M conv=fsync status=progress
expected=$(sha256sum < "$img" | cut -d' ' -f1)
actual=$(head -c "$size" "$dev" | sha256sum | cut -d' ' -f1)
if [ "$expected" != "$actual" ]; then
  echo "verification failed" >&2
  exit 2
fi
echo "ok"'''),
    ("kotlin", '''fun blend(dst: Int, src: Int): Int {
    val a = src ushr 24
    if (a == 0) return dst
    if (a == 255) return src
    val inv = 255 - a
    val r = (((src shr 16) and 0xFF) * a + ((dst shr 16) and 0xFF) * inv) / 255
    val g = (((src shr 8) and 0xFF) * a + ((dst shr 8) and 0xFF) * inv) / 255
    val b = ((src and 0xFF) * a + (dst and 0xFF) * inv) / 255
    val outA = a + (dst ushr 24) * inv / 255
    return (outA shl 24) or (r shl 16) or (g shl 8) or b
}'''),
    ("elixir", '''def decode_tlv(<<>>, acc), do: {:ok, Enum.reverse(acc)}

def decode_tlv(<<type::8, len::16, rest::binary>>, acc) do
  case rest do
    <<value::binary-size(len), tail::binary>> ->
      decode_tlv(tail, [{type, value} | acc])

    _ ->
      {:error, {:truncated, type, len, byte_size(rest)}}
  end
end

def decode_tlv(other, _acc), do: {:error, {:bad_header, byte_size(other)}}'''),
    ("rust", '''pub fn parse_nmea(line: &str) -> Result<Vec<&str>, NmeaError> {
    let line = line.trim_end();
    let body = line.strip_prefix('$').ok_or(NmeaError::NoStart)?;
    let (payload, checksum) = body.rsplit_once('*').ok_or(NmeaError::NoChecksum)?;
    let expected = u8::from_str_radix(checksum, 16).map_err(|_| NmeaError::BadChecksum)?;
    let actual = payload.bytes().fold(0u8, |acc, b| acc ^ b);
    if actual != expected {
        return Err(NmeaError::Mismatch { expected, actual });
    }
    Ok(payload.split(',').collect())
}'''),
]

base.THINGS = [
    ("python", "асинхронный сканер открытых TCP-портов с ограничением числа одновременных подключений"),
    ("java", "сервер, принимающий телеметрию от датчиков по UDP и отбрасывающий дубликаты пакетов"),
    ("csharp", "систему частиц для 2D-игры с повторным использованием объектов из пула"),
    ("go", "утилиту, которая следит за каталогом и пересчитывает контрольные суммы изменённых файлов"),
    ("typescript", "компонент, который записывает звук с микрофона в браузере и показывает индикатор громкости"),
    ("kotlin", "модуль Android-приложения, который читает показания пульсометра по Bluetooth Low Energy"),
    ("python", "программный растеризатор треугольников с буфером глубины и сохранением кадра в PNG"),
    ("java", "пул прямых буферов ByteBuffer с ограничением общего объёма и обнаружением утечек"),
    ("elixir", "MQTT-клиент на основе GenServer с повторной подпиской на топики после переподключения"),
    ("rust", "драйвер датчика давления с интерфейсом I2C для микроконтроллера, работающий без стандартной библиотеки (no_std)"),
]

base.CONCEPTS = [
    "как работает управление перегрузкой в TCP и что такое окно перегрузки",
    "что такое mmap и чем отображение файла в память отличается от обычного чтения",
    "зачем файловым системам нужен журнал и от чего он защищает при внезапном отключении питания",
    "чем прерывания отличаются от опроса (polling) во встраиваемых системах",
    "что такое архитектура ECS (Entity Component System) и почему её используют в игровых движках",
    "как работает буфер глубины и откуда берётся z-fighting",
    "что такое джиттер-буфер и как он сглаживает неравномерную доставку аудиопакетов",
    "зачем при хранении телеметрии прореживают данные (downsampling) и почему при усреднении теряются пики",
    "как работает epoll и чем он лучше select при тысячах соединений",
    "что такое порядок байтов (endianness) и где он важен при разборе сетевых протоколов",
]

base.MODULES = [
    "модуля буферизации телеметрии на устройстве при потере связи с сервером",
    "сервиса обновления прошивки устройств по воздуху (OTA) с откатом на предыдущую версию при сбое",
    "компонента подбора соперников в многопользовательской игре по рейтингу и сетевой задержке",
    "модуля нарезки видео на сегменты для потоковой передачи по HLS",
    "сервиса генерации миниатюр изображений с учётом ориентации из EXIF",
    "компонента, воспроизводящего журнал файлового хранилища после аварийного отключения",
    "модуля формирования и разбора кадров Modbus RTU с проверкой контрольной суммы",
    "сервиса агрегации метрик устройств в поминутные временные ряды",
    "модуля ограничения пропускной способности сетевого интерфейса по классам трафика",
    "сервиса распределения заданий рендеринга между узлами с учётом памяти видеокарт",
]

base.TOOLS = [
    "утилиты командной строки для захвата сетевых пакетов с фильтрацией по протоколу",
    "библиотеки для разбора бинарных протоколов по декларативному описанию",
    "эмулятора последовательного порта для тестирования прошивок",
    "сервиса сбора дампов памяти при падении игровых клиентов",
    "конвертера текстур в сжатые GPU-форматы (ASTC, ETC2)",
    "библиотеки для чтения и записи метаданных аудиофайлов",
    "агента, собирающего метрики хоста и отправляющего их по протоколу OTLP (OpenTelemetry)",
    "утилиты для монтирования образов дисков в пространстве пользователя через FUSE",
    "инструмента покадрового профилирования отрисовки в игровом движке",
    "библиотеки очередей без блокировок (lock-free) для обмена данными между потоками",
]

base.FAILS = [
    ("java", '''@Test
void readsLittleEndianLength() {
    byte[] header = {0x01, 0x00, 0x00, 0x00};
    ByteBuffer buf = ByteBuffer.wrap(header);
    buf.order(ByteOrder.LITTLE_ENDIAN);
    assertEquals(1, PacketReader.readLength(header));
}''', "expected: <1> but was: <16777216>"),
    ("python", '''def test_detects_modified_file(tmp_path):
    f = tmp_path / "device.conf"
    f.write_text("rate=10")
    watcher = Watcher(tmp_path)
    watcher.snapshot()
    f.write_text("rate=20")
    assert watcher.changed() == [f]''',
     "AssertionError: assert [] == [PosixPath('/tmp/pytest-of-ci/pytest-3/test_detects_modified_file0/device.conf')] "
     "(падает не при каждом запуске)"),
    ("csharp", '''[Fact]
public void Enemy_Stays_Put_When_On_Target()
{
    var enemy = new Enemy(new Vector2(5, 5));
    enemy.MoveTowards(new Vector2(5, 5), speed: 2f, dt: 0.016f);
    Assert.Equal(new Vector2(5, 5), enemy.Position);
}''', "Assert.Equal() Failure: Expected <5, 5>, Actual <NaN, NaN> (MoveTowards нормализует вектор направления)"),
    ("go", '''func TestHealthEndpoint(t *testing.T) {
    srv := NewServer("127.0.0.1:8080")
    go srv.ListenAndServe()
    resp, err := http.Get("http://127.0.0.1:8080/health")
    if err != nil {
        t.Fatal(err)
    }
    defer resp.Body.Close()
}''', 'server_test.go:14: Get "http://127.0.0.1:8080/health": dial tcp 127.0.0.1:8080: '
      'connect: connection refused (падает не всегда)'),
    ("typescript", '''it('converts float samples to 16-bit PCM', () => {
  const pcm = toInt16(new Float32Array([1.0, -1.0, 0.5]));
  expect(Array.from(pcm)).toEqual([32767, -32768, 16384]);
});''', "Expected: [32767, -32768, 16384], Received: [-32768, -32768, 16384]"),
    ("kotlin", '''@Test
fun `decodes temperature from sensor bytes`() {
    val raw = byteArrayOf(0x01, 0xF4.toByte())
    assertEquals(50.0, Sensor.decodeTemperature(raw), 0.01)
}''', "expected: <50.0> but was: <-1.2> (значение собирается из двух байтов и делится на 10)"),
    ("python", '''def test_telemetry_record_size():
    record = struct.pack("BHIf", 1, 512, 1700000000, 21.5)
    assert len(record) == 11''', "AssertionError: assert 12 == 11"),
    ("java", '''@Test
void receivesWholeDatagram() throws Exception {
    byte[] payload = new byte[2000];
    sender.send(new DatagramPacket(payload, payload.length, receiverAddress));
    DatagramPacket packet = new DatagramPacket(new byte[1500], 1500);
    receiver.receive(packet);
    assertEquals(payload.length, packet.getLength());
}''', "expected: <2000> but was: <1500>"),
    ("sql", '''-- ожидается 24 строки: по одной на каждый час суток
SELECT date_trunc('hour', ts) AS hour_start, COUNT(*) AS samples
FROM telemetry
WHERE ts >= '2024-03-31' AND ts < '2024-04-01'
GROUP BY 1
ORDER BY 1;''', "expected 24 rows, got 23 (сессия работает в часовом поясе Europe/Berlin)"),
    ("elixir", '''test "reports device status" do
  {:ok, pid} = DeviceConn.start_link(host: "10.0.0.5", port: 502)
  assert {:ok, :online} = DeviceConn.status(pid)
end''', "** (exit) exited in: GenServer.call(#PID<0.215.0>, :status, 5000) ** (EXIT) time out "
        "(подключение к устройству выполняется в handle_continue)"),
]


if __name__ == "__main__":
    rows = base.build()
    for r in rows:
        r["id"] = "t" + r["id"][1:]
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print(len(rows), "промптов:", dict(collections.Counter(r["kind"] for r in rows)))
