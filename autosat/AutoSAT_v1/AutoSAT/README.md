# AutoSAT

Welcome to the official repository for the paper ["AutoSAT: Automatically Optimize SAT Solvers via Large Language Models"](https://arxiv.org/abs/2402.10705).
This repository is dedicated to automatically optimize heuristics in SAT solvers through Large Language Models (LLMs).

## Clone this repo

`git clone https://github.com/YiwenAI/AutoSAT`

## Installation

We support both **Linux** and **Windows**.

1. Python 3.10
2. G++ 17 or higher to support `filesystem`

Install requirements:

`pip install -e .`

Install this package:

`python setup.py develop`

## Use Docker

`docker build -t autosat .`

## Preparation

- Prepare the dataset from [cnf_data_link](https://drive.google.com/drive/folders/1-au8hBbx4YAdJDlct9glCODpL0TQcYnA?usp=drive_link)
- Copy [`.env.example`](./.env.example) to `.env` and fill the API settings
- Keep the prompt templates under `./examples/`
- For a small end-to-end smoke run with StepFun, use [examples/EasySAT/config.stepfun.mini.yaml](./examples/EasySAT/config.stepfun.mini.yaml)

## Train & Test

### Environment

Create `./.env` from [`.env.example`](./.env.example) with one of the following sets:

```env
DEEPINFRA_MODEL="stepfun-ai/Step-3.5-Flash"
DEEPINFRA_API_BASE="https://api.deepinfra.com/v1/openai"
DEEPINFRA_API_KEY="<YOUR_KEY>"
```

or:

```env
DEEPINFRA_MODEL="stepfun-ai/Step-3.5-Flash"
DEEPINFRA_API_BASE="https://api.deepinfra.com/v1/openai"
DEEPINFRA_API_KEY="<YOUR_KEY>"
```

### Train

For a quick launch of AutoSAT_v1 with StepFun, start with the mini config:

```bash
python3 main.py --config ./examples/EasySAT/config.stepfun.mini.yaml
```

If you want your own settings, edit [examples/EasySAT/config.yaml](./examples/EasySAT/config.yaml) or pass another YAML through `--config`.

> Tips
>
> - Refer to [configs explanation](./examples/EasySAT/config_explanation.txt) for the main parameters.
> - `llm_model` can be any OpenAI-compatible model id when `api_base` is set.
> - The heuristic functions generated and the metrics are saved in `./temp/prompts/`.

### Test

```bash
python3 -m autosat.evaluation.evaluate \
    --SAT_solver_file_path SAT_solver_file_path \
    --results_save_path your_final_eval_results_savePath \
    --batch_size 4 \
    --eval_parallel_size 6 \
    --eval_timeout 500 \
    --eval_data_dir your_test_set_path \
    --rand_seed 42 \
    --keep_intermediate_results False \
    --method_name your_solver_name
```

> Tips
>
> - Use [examples/EasySAT/eval_config.yaml](./examples/EasySAT/eval_config.yaml) for a YAML-based eval run.
> - The final evaluation results are saved in the folder you set via `results_save_path`.

## Dataset

- All datasets can be obtained from [cnf_data_link](https://drive.google.com/drive/folders/1-au8hBbx4YAdJDlct9glCODpL0TQcYnA?usp=drive_link)
- The code used to generate the specific problems is in [`./data/`](./data/)
- More SAT Competition questions are available at [SAT Competition](https://satcompetition.github.io/)

## Metrics

We use the following metrics to evaluate a solver:

- PAR-2: Penalized Average Runtime with factor 2
- #solved: Number of questions solved within timeout
- total time: Total runtime for a solver
- #satisfied: Number of feasible solutions found
- #unsatisfied: Number of infeasible questions solved
- #timeout: Number of timeout cases

## Acknowledgement

Our baseline is [EasySAT](https://github.com/shaowei-cai-group/EasySAT) and we only add data parallel and file saving modules.

## Citing us

If our work has been helpful to you, please feel free to cite us:

```latex
@article{sun2024autosat,
  title={AutoSAT: Automatically Optimize SAT Solvers via Large Language Models},
  author={Sun, Yiwen and Zhang, Xianyin and Huang, Shiyu and Cai, Shaowei and Zhang, Bing-Zhen and Wei, Ke},
  journal={arXiv preprint arXiv:2402.10705},
  year={2024}
}
```