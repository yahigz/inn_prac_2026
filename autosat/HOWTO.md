# HOWTO: запуск AutoSAT в этом репозитории


Этот файл описывает рабочий путь запуска с учетом текущих изменений в проекте:
- запуск на Linux;
- работа через OpenAI-compatible API (DeepInfra);
- поддержка нескольких датасетов:
  - мини-датасет `mini_v2`;
  - SAT20: `sat20_cnfs`;
  - Zamkeller;
  - cryptography-ascon;
- логирование расхода токенов;
- checkpoint/resume через конфиг;
- ретраи при временных ошибках API;
- запуск в фоне через `nohup`.
- каждый запуск получает свой `run_id`, поэтому артефакты не перетираются.


## 1. Что уже подготовлено

- Датасеты распакованы в папку `datasets/`.
- Поддерживаемые датасеты:
  - Мини-датасет: `datasets/mini_v2/`
    - `datasets/mini_v2/train`
    - `datasets/mini_v2/eval`
    - Включает 20 самых маленьких инстансов из Zamkeller, которых не было в оригинале.
  - SAT20: `datasets/sat20_cnfs/`
    - Оригинальные CNF-файлы SAT20 (используются для крупных тестов и бенчмарков).
  - Zamkeller: `datasets/Zamkeller/`
    - `datasets/Zamkeller/train`
    - `datasets/Zamkeller/eval`
    - Альтернативный набор для тестирования и расширения.
  - cryptography-ascon: `datasets/cryptography-ascon/`
    - `datasets/cryptography-ascon/train`
    - `datasets/cryptography-ascon/eval`
    - Используется для экспериментов с криптографическими задачами.
- Мини-конфиг: `AutoSAT_v1/AutoSAT/config.mini.yaml`
- Примеры конфигов для SAT20 и других датасетов: `AutoSAT_v1/AutoSAT/config.sat20_combined.train4func.yaml`, `config.sat20_small.train4func.yaml` и др.
- Git игнорирует тяжелые файлы через `.gitignore`:
  - `datasets/`
  - `AutoSAT_v1/AutoSAT/temp/`

## 2. Требования окружения

Рекомендуется использовать виртуальное окружение `../.venv` (оно уже используется в проекте).

Из корня репозитория:

```bash
source .venv/bin/activate
```

Установить/обновить зависимости AutoSAT:

```bash
cd AutoSAT_v1/AutoSAT
python -m pip install -r requirements.txt
python -m pip install -e .
python setup.py develop
```

Важно для `ray`:

```bash
python -m pip install "setuptools<81"
```

Это нужно из-за `pkg_resources`, который использует текущая версия `ray`.


## 3. Настройка `.env`

Файл `.env` должен лежать в корне репозитория (`autosat/.env`) и автоматически подхватывается при запуске.

**Обязательно пропишите свой API-ключ!**

Пример содержимого `.env`:

```env
# Модель для LLM (обязательно)
DEEPINFRA_MODEL="openai/gpt-oss-120b"
# Базовый URL для DeepInfra (обязательно)
DEEPINFRA_API_BASE="https://api.deepinfra.com/v1/openai"
# Ваш API-ключ (обязательно!)
DEEPINFRA_API_KEY="<YOUR_KEY>"
```

**Важно:**
- `DEEPINFRA_API_KEY` — сюда нужно вписать ваш персональный ключ доступа к DeepInfra API. Без него ничего не заработает!
- `DEEPINFRA_API_BASE` должен быть **base URL**, а не полный endpoint.
  - Неправильно: `.../chat/completions`
  - Правильно: `https://api.deepinfra.com/v1/openai`
- Модель должна быть совместима с OpenAI API (например, `openai/gpt-oss-120b`).

Почему так: библиотека OpenAI сама добавляет нужный путь API.


## 4. Примеры запуска обучения и оценки

Из папки `AutoSAT_v1/AutoSAT`:

### Мини-датасет (mini_v2)
```bash
python3 main.py --config ./config.mini.yaml
```

### SAT20 (пример)
```bash
python3 main.py --config ./config.sat20_combined.train4func.yaml
```

### Zamkeller (пример)
```bash
python3 main.py --config ./config.zamkeller.yaml
```

### cryptography-ascon (пример)
```bash
python3 main.py --config ./config.cryptography_ascon.yaml
```

#### Пример путей в конфиге для разных датасетов:

```yaml
data_dir: ../../datasets/sat20_cnfs/
eval_data_dir: ../../datasets/sat20_cnfs/
# или
data_dir: ../../datasets/Zamkeller/train
eval_data_dir: ../../datasets/Zamkeller/eval
# или
data_dir: ../../datasets/cryptography-ascon/train
eval_data_dir: ../../datasets/cryptography-ascon/eval
```

Если `run_id` пустой, он генерируется автоматически при старте.
Если нужно продолжить старый запуск, укажи тот же `run_id` или конкретный `checkpoint_path`.

Чтобы явно продолжить с конкретного состояния, укажи:

```yaml
checkpoint_path: "./results/checkpoints/iter_12_checkpoint.json"
```

Если `checkpoint_path` пустой, используется `latest_checkpoint.json` из `checkpoint_dir`.

## 5. Как работает checkpoint/resume

После каждой итерации сохраняются:
- `results/checkpoints/latest_checkpoint.json`
- `results/checkpoints/iter_<N>_checkpoint.json`
- `results/iter_<N>_result.json`
- `results/snapshots/iter_<N>_best.json`

В чекпоинте лежат:
- `next_iter` — с какой итерации продолжать;
- `results` — накопленные метрики;
- `answers` — ответы модели;
- `extra_params` — дополнительные параметры;
- `best_result` — лучший найденный результат.

На старте `main.py`:
1. читает `resume_from_checkpoint`;
2. ищет чекпоинт по `checkpoint_path` или в `checkpoint_dir`;
3. если чекпоинт есть, возобновляет работу с `next_iter`;
4. если чекпоинта нет, стартует с нуля.

## 6. Как читать вывод


### Базовые метрики
- `Backbone(original) result -- time: X seconds ; PAR-2: Y`

### Метрики кандидатов
- В конце печатается словарь `final`:
  - `time`
  - `PAR-2`
  - `prompt` (сгенерированный код)

Кандидат считается лучше baseline, если:
- `PAR-2(candidate) < PAR-2(baseline)`

Именно такие кандидаты идут в дополнительный eval.

### Проверка размера датасетов

```bash
# mini_v2
find datasets/mini_v2/train -type f -name '*.cnf' | wc -l
find datasets/mini_v2/eval -type f -name '*.cnf' | wc -l
# SAT20
find datasets/sat20_cnfs/ -type f -name '*.cnf' | wc -l
# Zamkeller
find datasets/Zamkeller/train -type f -name '*.cnf' | wc -l
find datasets/Zamkeller/eval -type f -name '*.cnf' | wc -l
# cryptography-ascon
find datasets/cryptography-ascon/train -type f -name '*.cnf' | wc -l
find datasets/cryptography-ascon/eval -type f -name '*.cnf' | wc -l
```

### Расход токенов
В логах есть строки:

```text
[TokenUsage] model=... call_prompt=... call_completion=... call_total=... cum_total=...
```

- `call_*` — токены конкретного запроса.
- `cum_*` — накопительный итог внутри процесса.

Практически: для оценки общего расхода запуска суммируйте `call_total` по всем строкам `TokenUsage` (включая worker-логи Ray).

## 7. Ретраи при падении API

Если API временно падает, запрос повторяется через паузу.

Поддерживаются переменные окружения:

```env
AUTOSAT_API_RETRY_SECONDS=10
AUTOSAT_API_MAX_RETRIES=0
```

Значения по умолчанию:
- `AUTOSAT_API_RETRY_SECONDS=10` — ждать 10 секунд между попытками;
- `AUTOSAT_API_MAX_RETRIES=0` — безлимитные повторы.

Если нужно ограничить число попыток, поставь, например:

```env
AUTOSAT_API_MAX_RETRIES=5
```

## 8. Запуск в фоне

Для долгого прогона удобно использовать `nohup`:

```bash
cd AutoSAT_v1/AutoSAT
nohup env PYTHONUNBUFFERED=1 /home/bibaboba/inn_prac/inn_prac_2026/.venv/bin/python main.py --config config.mini.yaml > nohup.mini_v2.log 2>&1 &
```

Смотреть лог:

```bash
tail -f nohup.mini_v2.log
```

Остановить прогон:

```bash
pkill -f 'main.py --config config.mini.yaml'
```

## 9. Graceful shutdown

Если нажать Ctrl+C во время train/eval:
- срабатывает `KeyboardInterrupt` handler в `main_MultiAgent.py`;
- вызывается `ExecutionWorker.shutdown_all()`;
- останавливаются solver-процессы и Ray.

Это защищает от зависших фоновых процессов после прерывания.

## 10. Где искать артефакты

- Промежуточные и финальные результаты промптов:
  - `AutoSAT_v1/AutoSAT/temp/prompts/`
- Результаты eval:
  - `AutoSAT_v1/AutoSAT/temp/eval_results/`
- Чекпоинты:
  - `AutoSAT_v1/AutoSAT/results/checkpoints/`
- Снимки лучших итераций:
  - `AutoSAT_v1/AutoSAT/results/snapshots/`

## 11. Типовые проблемы

### `Unsupported llm_model without external API endpoint`
Причина: пустой `api_base` и модель не попала в локальные ветки.

Проверить:
- `.env` загружен
- `DEEPINFRA_API_BASE` заполнен
- модель задана как `openai/gpt-oss-120b`

### `404 Not Found` от API
Почти всегда неверный `DEEPINFRA_API_BASE`.

Проверь, что он равен:
- `https://api.deepinfra.com/v1/openai`

а не:
- `.../chat/completions`

### Предупреждение про `total_time` в C++
Это compile warning (`unused variable`), запуск не ломает.

## 12. Полезные команды

Проверить размер мини-сплита:

```bash
find datasets/mini_v2/train -type f -name '*.cnf' | wc -l
find datasets/mini_v2/eval -type f -name '*.cnf' | wc -l
```


Запустить eval отдельно:

```bash
cd AutoSAT_v1/AutoSAT
python evaluate.py --config ./examples/EasySAT/eval_config.yaml
# или для кастомного датасета/конфига:
python evaluate.py --config ./config.sat20_combined.train4func.yaml
```

(Для отдельного eval проверьте пути к solver/data в конфиге — они должны указывать на нужный датасет и solver.)
