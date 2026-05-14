#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autosat.evaluation.evaluate import evaluate  # noqa: E402
from autosat.utils import revise_file, fill_core_codes, copy_folder  # noqa: E402
from main import _infer_task_from_code, _validate_filled_solver  # noqa: E402


@dataclass
class EvalSummary:
    checkpoint_path: str
    task_name: str | None
    candidate_par2: float | None
    baseline_par2: float | None
    delta: float | None
    status: str


def _load_yaml_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _sanitize_label(text: str) -> str:
    text = str(text or '').strip()
    text = re.sub(r'[^A-Za-z0-9_.-]+', '_', text)
    return text.strip('_') or 'run'


def _latest_result_file(results_dir: Path, method_name: str) -> Path:
    pattern = f'results_{method_name}_*.txt'
    matches = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f'No eval result file found for {method_name} in {results_dir}')
    return matches[-1]


def _read_par2_from_eval_file(result_file: Path) -> float:
    last_line = ''
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if not last_line:
        raise ValueError(f'Empty eval result file: {result_file}')
    payload = ast.literal_eval(last_line)
    if 'PAR-2' not in payload:
        raise KeyError(f'PAR-2 not found in {result_file}')
    return float(payload['PAR-2'])


def _extract_candidate_code(checkpoint_json: dict) -> tuple[str | None, str | None]:
    if not checkpoint_json:
        return None, None

    candidates = []
    for key, prompt in checkpoint_json.get('prompt', {}).items():
        candidates.append((str(key), str(prompt or '')))

    if not candidates:
        return None, None

    candidate_id, code = candidates[0]
    return candidate_id, code.strip()


def _evaluate_solver(args, solver_cpp_path: Path, method_name: str, results_dir: Path) -> float:
    args.results_save_path = str(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluate(args, SAT_solver_file_path=str(solver_cpp_path), method_name=method_name)
    result_file = _latest_result_file(results_dir, method_name)
    return _read_par2_from_eval_file(result_file)


def _prepare_solver_source(checkpoint_path: Path, candidate_code: str, task_name: str, work_root: Path, eval_timeout: int, eval_data_dir: str) -> Path:
    template_dir = ROOT / 'examples' / 'EasySAT' / task_name
    if not template_dir.exists():
        raise FileNotFoundError(f'Missing task template directory: {template_dir}')

    solver_dir = work_root / checkpoint_path.stem / task_name
    if solver_dir.exists():
        import shutil
        shutil.rmtree(solver_dir)
    
    copy_folder(src_folder=str(template_dir), num=1, mode='eval', target_folder=str(solver_dir))
    
    headers_dir = ROOT / 'examples' / 'EasySAT' / 'original_EasySAT'
    for header_file in ['EasySAT.hpp', 'heap.hpp']:
        import shutil
        src_header = headers_dir / header_file
        if src_header.exists():
            shutil.copy(str(src_header), str(solver_dir / header_file))
    
    solver_path = solver_dir / 'EasySAT_modified.cpp'
    
    prev_cwd = os.getcwd()
    try:
        os.chdir(solver_dir)
        revise_file(
            file_name='EasySAT.cpp',
            save_dir='EasySAT_modified.cpp',
            timeout=eval_timeout,
                data_dir=f'"{eval_data_dir}"',
                replace_code=candidate_code,
                lbd_queue_size=50,
        )
    finally:
        os.chdir(prev_cwd)
    
    ok, compile_error = _validate_filled_solver(str(solver_path), checkpoint_path.stem)
    if not ok:
        raise RuntimeError(compile_error)
    return solver_path


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate a list of iter_*_result.json checkpoints on cryptography-ascon.')
    parser.add_argument('--config', required=True, help='Path to YAML config with checkpoint list and eval settings.')
    parser.add_argument('--work-dir', default='./temp/eval_checkpoint_work/', help='Temporary working directory for generated solver sources.')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline evaluation and only report candidate PAR-2.')
    args_ns = parser.parse_args()

    config_path = Path(args_ns.config)
    config = _load_yaml_config(config_path)

    checkpoints = [Path(p) for p in config.get('checkpoints', [])]
    if not checkpoints:
        raise ValueError('Config does not contain any checkpoints.')

    work_root = Path(args_ns.work_dir)
    results_root = Path(config.get('results_save_path', './temp/eval_checkpoint_results/'))
    eval_data_dir = config.get('eval_data_dir', '../../datasets/cryptography-ascon/eval')
    eval_parallel_size = int(config.get('eval_parallel_size', 6))
    eval_timeout = int(config.get('eval_timeout', 5000))
    rand_seed = int(config.get('rand_seed', 42))
    keep_intermediate_results = bool(config.get('keep_intermediate_results', False))
    baseline_solver_path = Path(config.get('baseline_solver_path', './examples/EasySAT/original_EasySAT/EasySAT.cpp'))
    fallback_task = str(config.get('fallback_task', 'rephase_function'))
    llm_model = str(config.get('llm_model', 'checkpoint_eval'))
    project = str(config.get('project', 'EasySAT/'))

    baseline_cache: dict[str, float] = {}
    summaries: list[EvalSummary] = []

    for checkpoint in checkpoints:
        checkpoint = checkpoint.resolve() if checkpoint.is_absolute() else (ROOT / checkpoint).resolve()
        if not checkpoint.exists():
            summaries.append(EvalSummary(str(checkpoint), None, None, None, None, 'missing_checkpoint'))
            continue

        checkpoint_json = _load_json(checkpoint)
        candidate_id, candidate_code = _extract_candidate_code(checkpoint_json)
        if not candidate_code:
            summaries.append(EvalSummary(str(checkpoint), None, None, None, None, 'empty_checkpoint'))
            continue

        task_name = _infer_task_from_code(candidate_code) or fallback_task
        try:
            solver_path = _prepare_solver_source(checkpoint, candidate_code, task_name, work_root, eval_timeout, eval_data_dir)
        except Exception as exc:
            summaries.append(EvalSummary(str(checkpoint), task_name, None, None, None, f'build_failed: {exc}'))
            continue

        task_results_dir = results_root / _sanitize_label(checkpoint.stem) / task_name
        task_results_dir.mkdir(parents=True, exist_ok=True)

        eval_args = argparse.Namespace(
            eval_data_dir=str(eval_data_dir),
            eval_parallel_size=eval_parallel_size,
            eval_timeout=eval_timeout,
            keep_intermediate_results=keep_intermediate_results,
            rand_seed=rand_seed,
            SAT_solver_file_path=str(baseline_solver_path),
            results_save_path=str(task_results_dir),
            project=project,
            task=task_name,
            llm_model=llm_model,
            original=True,
            eval_baseline=False,
        )

        candidate_method_name = f'{_sanitize_label(checkpoint.stem)}_{task_name}_{candidate_id or "candidate"}'
        try:
            candidate_par2 = _evaluate_solver(eval_args, solver_path, candidate_method_name, task_results_dir)
        except Exception as exc:
            summaries.append(EvalSummary(str(checkpoint), task_name, None, None, None, f'eval_failed: {exc}'))
            continue

        if task_name not in baseline_cache and not args_ns.skip_baseline:
            baseline_method_name = f'baseline_{task_name}_{llm_model}'
            try:
                baseline_par2 = _evaluate_solver(eval_args, baseline_solver_path, baseline_method_name, task_results_dir)
            except Exception as exc:
                summaries.append(EvalSummary(str(checkpoint), task_name, candidate_par2, None, None, f'baseline_failed: {exc}'))
                continue
            baseline_cache[task_name] = baseline_par2

        baseline_par2 = baseline_cache.get(task_name)
        delta = None if baseline_par2 is None else candidate_par2 - baseline_par2
        summaries.append(EvalSummary(str(checkpoint), task_name, candidate_par2, baseline_par2, delta, 'ok'))

    header = ['checkpoint', 'task', 'candidate_PAR-2', 'baseline_PAR-2', 'delta', 'status']
    print('\t'.join(header))
    for row in summaries:
        values = [row.checkpoint_path, row.task_name or '', '' if row.candidate_par2 is None else f'{row.candidate_par2:.2f}', '' if row.baseline_par2 is None else f'{row.baseline_par2:.2f}', '' if row.delta is None else f'{row.delta:.2f}', row.status]
        print('\t'.join(values))

    summary_path = results_root / 'eval_iter_checkpoints_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump([row.__dict__ for row in summaries], f, ensure_ascii=False, indent=2)
    print(f'Summary saved to {summary_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
