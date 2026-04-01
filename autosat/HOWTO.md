# HOWTO: запуск AutoSAT в этом репозитории

Этот файл описывает рабочий путь запуска с учетом текущих изменений в проекте:
- запуск на macOS;
- работа через OpenAI-compatible API (DeepInfra);
- мини-датасет из 10 самых маленьких CNF;
- логирование расхода токенов;
- graceful shutdown по Ctrl+C.

## 1. Что уже подготовлено

- Датасеты распакованы в `datasets/`.
- В `datasets/mini/` оставлены только 10 самых маленьких CNF:
  - `datasets/mini/train` -> 8 файлов
  - `datasets/mini/eval` -> 2 файла
- Мини-конфиг: `AutoSAT/config.mini.yaml`
- Git игнорирует тяжелые файлы через `.gitignore`:
  - `datasets/`
  - `AutoSAT/temp/`

## 2. Требования окружения

Рекомендуется использовать виртуальное окружение `../.venv` (оно уже используется в проекте).

Из корня репозитория:

```bash
source .venv/bin/activate
```

Установить/обновить зависимости AutoSAT:

```bash
cd AutoSAT
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

Файл `.env` лежит в корне репозитория (`autosat/.env`) и автоматически подхватывается из `main_MultiAgent.py`.

Нужные переменные:

```env
DEEPINFRA_MODEL="openai/gpt-oss-120b"
DEEPINFRA_API_BASE="https://api.deepinfra.com/v1/openai"
DEEPINFRA_API_KEY="<YOUR_KEY>"
```

Важно:
- `DEEPINFRA_API_BASE` должен быть **base URL**, а не полный endpoint.
- Неправильно: `.../chat/completions`
- Правильно: `https://api.deepinfra.com/v1/openai`

Почему так: библиотека OpenAI сама добавляет нужный путь API.

## 4. Мини-запуск обучения

Из папки `AutoSAT`:

```bash
python main_MultiAgent.py --config ./config.mini.yaml
```

Текущие параметры мини-конфига (`AutoSAT/config.mini.yaml`):
- `iteration_num: 1`
- `batch_size: 1`
- `data_parallel_size: 4`
- `data_dir: ../datasets/mini/train`
- `eval_data_dir: ../datasets/mini/eval`
- `llm_model: openai/gpt-oss-120b`

## 5. Как читать вывод

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

### Расход токенов
В логах есть строки:

```text
[TokenUsage] model=... call_prompt=... call_completion=... call_total=... cum_total=...
```

- `call_*` — токены конкретного запроса.
- `cum_*` — накопительный итог внутри процесса.

Практически: для оценки общего расхода запуска суммируйте `call_total` по всем строкам `TokenUsage` (включая worker-логи Ray).

## 6. Graceful shutdown

Если нажать Ctrl+C во время train/eval:
- срабатывает `KeyboardInterrupt` handler в `main_MultiAgent.py`;
- вызывается `ExecutionWorker.shutdown_all()`;
- останавливаются solver-процессы и Ray.

Это защищает от зависших фоновых процессов после прерывания.

## 7. Где искать артефакты

- Промежуточные и финальные результаты промптов:
  - `AutoSAT/temp/prompts/`
- Результаты eval:
  - `AutoSAT/temp/eval_results/`

## 8. Типовые проблемы

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

## 9. Полезные команды

Проверить размер мини-сплита:

```bash
find datasets/mini/train -type f -name '*.cnf' | wc -l
find datasets/mini/eval -type f -name '*.cnf' | wc -l
```

Запустить eval отдельно:

```bash
cd AutoSAT
python evaluate.py --config ./examples/EasySAT/eval_config.yaml
```

(Для отдельного eval проверь пути к solver/data в конфиге.)
