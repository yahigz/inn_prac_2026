# AutoSAT HOWTO

This document describes how to run [AutoSAT](autosat/AutoSAT_v1/AutoSAT) after the LLM setup was moved fully to environment variables.

## 1. Principles

- YAML configs in [AutoSAT](autosat/AutoSAT_v1/AutoSAT) define experiment logic, datasets, solver paths, iteration counts, evaluation settings, and write-back strategy.
- YAML configs must **not** define model API settings.
- Model provider, endpoint, model name, and credentials are taken only from environment variables read by [_apply_env_overrides()](autosat/AutoSAT_v1/AutoSAT/main.py:800) and resolved in [get_llm_api()](autosat/AutoSAT_v1/AutoSAT/autosat/llm_api/base_api.py:498).
- Each run stores its source config path and sanitized config payload in [run_metadata.json](autosat/AutoSAT_v1/AutoSAT/main.py:694) under `results/runs/<run_id>/`.

## 2. Installation

From [AutoSAT root](autosat/AutoSAT_v1/AutoSAT):

```bash
pip install -e .
python setup.py develop
```

Requirements:
- Python 3.10+
- `g++` with C++17 support
- datasets prepared under paths referenced by your YAML configs

## 3. Environment variables

### Common variables

Recognized LLM variables:

- `AUTOSAT_LLM_MODEL`
- `AUTOSAT_API_BASE`
- `AUTOSAT_API_KEY`
- `AUTOSAT_API_TYPE`

They are applied in [_apply_env_overrides()](autosat/AutoSAT_v1/AutoSAT/main.py:800).

### 3.1 Eliza

Use Eliza by setting:

```env
AUTOSAT_API_TYPE=eliza
AUTOSAT_API_KEY=<your_soy_oauth_token>
AUTOSAT_LLM_MODEL=claude-sonnet-4-6
```

Notes:
- `AUTOSAT_API_TYPE=eliza` switches [get_llm_api()](autosat/AutoSAT_v1/AutoSAT/autosat/llm_api/base_api.py:505) to [ElizaCallAPI](autosat/AutoSAT_v1/AutoSAT/autosat/llm_api/base_api.py:282).
- `AUTOSAT_LLM_MODEL` should be the raw Eliza model name, for example `claude-sonnet-4-6`.
- `AUTOSAT_API_BASE` is not needed for Eliza.

### 3.2 OpenAI-compatible providers

Use any OpenAI-compatible endpoint by setting:

```env
AUTOSAT_LLM_MODEL=openai/gpt-4.1
AUTOSAT_API_BASE=https://api.openai.com/v1
AUTOSAT_API_KEY=<your_api_key>
```

Examples:
- OpenAI-compatible local proxy
- DeepInfra-compatible endpoint
- other OpenAI-compatible vendor endpoints

Notes:
- leave `AUTOSAT_API_TYPE` unset
- if `AUTOSAT_API_BASE` is set, [GPTCallAPI](autosat/AutoSAT_v1/AutoSAT/autosat/llm_api/base_api.py:154) is used

### 3.3 Local built-in aliases

Some legacy aliases are still supported in [get_llm_api()](autosat/AutoSAT_v1/AutoSAT/autosat/llm_api/base_api.py:523):
- `Qwen`
- `llama`
- `deepseek`

These use hardcoded local endpoints from code, not YAML.

## 4. Config files

Configs no longer carry model/API settings. They should contain only run logic.

Examples:
- training configs in [config.sat20_combined.train4func.yaml](autosat/AutoSAT_v1/AutoSAT/config.sat20_combined.train4func.yaml), [config.train.4func.yaml](autosat/AutoSAT_v1/AutoSAT/config.train.4func.yaml), [config.mini.yaml](autosat/AutoSAT_v1/AutoSAT/config.mini.yaml)
- eval configs in [config.eval_crypto.yaml](autosat/AutoSAT_v1/AutoSAT/config.eval_crypto.yaml), [config.eval_run_20260417.yaml](autosat/AutoSAT_v1/AutoSAT/config.eval_run_20260417.yaml)
- example configs in [examples/EasySAT/config.yaml](autosat/AutoSAT_v1/AutoSAT/examples/EasySAT/config.yaml)

Important runtime fields commonly used in configs:
- `iteration_num`
- `batch_size`
- `data_parallel_size`
- `data_dir`
- `eval_data_dir`
- `project`
- `task` or `optimize_tasks`
- `task_selection_mode`
- `timeout`
- `eval_timeout`
- `eval_parallel_size`
- `SAT_solver_file_path`
- `run_id`
- `resume_from_checkpoint`
- `checkpoint_dir`
- `checkpoint_path`
- `template_update_strategy`
- `writeback_to_template`

## 5. Recommended local env files

Keep local env files near [main.py](autosat/AutoSAT_v1/AutoSAT/main.py) or export variables in your shell.

Example Eliza file:

```env
AUTOSAT_API_TYPE=eliza
AUTOSAT_API_KEY=<your_soy_oauth_token>
AUTOSAT_LLM_MODEL=claude-sonnet-4-6
```

Example OpenAI-compatible file:

```env
AUTOSAT_LLM_MODEL=openai/gpt-4.1
AUTOSAT_API_BASE=https://api.openai.com/v1
AUTOSAT_API_KEY=<your_api_key>
```

Load them before running:

```bash
export $(grep -v '^#' .env.eliza.local | xargs)
```

or

```bash
export $(grep -v '^#' .env.openai.local | xargs)
```

## 6. Training runs

Run from [AutoSAT root](autosat/AutoSAT_v1/AutoSAT):

```bash
python3 main.py --config config.sat20_combined.train4func.yaml
```

Example with Eliza:

```bash
export $(grep -v '^#' .env.eliza.local | xargs)
python3 main.py --config config.eliza.yaml
```

Example with OpenAI-compatible API:

```bash
export $(grep -v '^#' .env.openai.local | xargs)
python3 main.py --config config.train.4func.yaml
```

What happens during a run:
- a run id is created in [_make_run_id()](autosat/AutoSAT_v1/AutoSAT/main.py:612)
- directories are created by [_run_paths()](autosat/AutoSAT_v1/AutoSAT/main.py:619)
- run metadata is saved by [_save_run_metadata()](autosat/AutoSAT_v1/AutoSAT/main.py:694)
- training artifacts are written into `results/runs/<run_id>/`

## 7. Eval from an existing run

Use [config.eval_crypto.yaml](autosat/AutoSAT_v1/AutoSAT/config.eval_crypto.yaml) or another eval config with `eval_only_from_run: true`:

```bash
python3 main.py --config config.eval_crypto.yaml
```

This path loads the existing run and evaluates from artifacts already stored in `results/runs/<run_id>/`.

## 8. Sequential eval with write-back replay

Use [tools/eval_sequential_writeback.py](autosat/AutoSAT_v1/AutoSAT/tools/eval_sequential_writeback.py):

```bash
python3 tools/eval_sequential_writeback.py --config config.eval_eliza_run_20260518.yaml
```

This config defines:
- source `run_id`
- eval dataset
- baseline solver path
- candidate selection mode
- whether template write-back is enabled during replay

It no longer needs model settings in YAML.

## 9. Checkpoint evaluation

Use [tools/eval_iter_checkpoints.py](autosat/AutoSAT_v1/AutoSAT/tools/eval_iter_checkpoints.py):

```bash
python3 tools/eval_iter_checkpoints.py --config config.eval_cryptography_ascon_checkpoints.yaml
```

This evaluates a list of checkpoint result files against one eval dataset.

## 10. Smoke test for Eliza connectivity

Use [tools/test_eliza.py](autosat/AutoSAT_v1/AutoSAT/tools/test_eliza.py):

```bash
export $(grep -v '^#' .env.eliza.local | xargs)
python3 tools/test_eliza.py
```

It uses:
- `AUTOSAT_API_KEY`
- `AUTOSAT_LLM_MODEL`

## 11. Run outputs

For a run id `run_...`, primary outputs live under:

- `results/runs/<run_id>/final_result.json`
- `results/runs/<run_id>/iter_<n>_result.json`
- `results/runs/<run_id>/snapshots/`
- `results/runs/<run_id>/checkpoints/`
- `results/runs/<run_id>/eval_results/`
- `results/runs/<run_id>/run_metadata.json`

The [run_metadata.json](autosat/AutoSAT_v1/AutoSAT/main.py:694) file includes:
- the launch `run_id`
- the original `config_path`
- the sanitized config payload used for the run
- runtime LLM environment snapshot with secret key redacted

That is the place to check which config launched a stored run.

## 12. What must not be put into YAML anymore

Do not define these fields in config YAML files:
- `llm_model`
- `api_base`
- `api_key`
- `model_name`

These are runtime LLM settings and must come only from environment variables.

## 13. Troubleshooting

### Unsupported model without endpoint

If [get_llm_api()](autosat/AutoSAT_v1/AutoSAT/autosat/llm_api/base_api.py:498) raises unsupported-model errors:
- set `AUTOSAT_API_BASE`
- set `AUTOSAT_API_KEY`
- set `AUTOSAT_LLM_MODEL`
- or set `AUTOSAT_API_TYPE=eliza` for Eliza

### Eliza token errors

If [ElizaCallAPI](autosat/AutoSAT_v1/AutoSAT/autosat/llm_api/base_api.py:314) says token is missing:
- export `AUTOSAT_API_KEY`
- verify `AUTOSAT_API_TYPE=eliza`

### Wrong config behavior

If a run behaves differently than expected:
- open `results/runs/<run_id>/run_metadata.json`
- verify `config_path`
- verify saved config payload
- verify runtime env snapshot
