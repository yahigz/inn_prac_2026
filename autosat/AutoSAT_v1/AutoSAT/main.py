import os
import argparse
import ast
import json
import time
import uuid
import random
import yaml
import re
import sys
import glob
import signal
import subprocess
from pathlib import Path

from autosat.utils import get_code, revise_file, clean_files, collect_results, \
                            copy_folder, fill_core_codes, delete_InfiniteLoopInst, get_batch_id, train_init, check_reIteration
from autosat.llm_api.base_api import get_llm_api, _parse_structured_response
from autosat.execution.execution_worker import ExecutionWorker
from autosat.evaluation.evaluate import evaluate
from autosat.plotting import plot_training_curve
import warnings


_SHUTDOWN_IN_PROGRESS = False


def _graceful_shutdown(reason="", exit_code=None):
    global _SHUTDOWN_IN_PROGRESS
    if _SHUTDOWN_IN_PROGRESS:
        if exit_code is not None:
            os._exit(int(exit_code))
        return
    _SHUTDOWN_IN_PROGRESS = True
    try:
        if reason:
            print(f"[Shutdown] {reason}", flush=True)
        ExecutionWorker.shutdown_all(timeout=2)
    except Exception:
        pass
    try:
        if os.name == 'posix':
            import subprocess
            patterns = ['EasySAT', 'SAT_Solver_tmp']
            for pattern in patterns:
                subprocess.run(['pkill', '-TERM', '-f', pattern], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            for pattern in patterns:
                subprocess.run(['pkill', '-KILL', '-f', pattern], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
    except Exception:
        pass
    finally:
        if exit_code is not None:
            os._exit(int(exit_code))


TASK_ALIASES = {}

DEFAULT_TASKS = {
    "restart_condition",
    "restart_function",
    "bump_var_function",
    "rephase_function",
}


def _discover_available_tasks(project):
    project_name = str(project or "EasySAT/").strip().strip('/')
    root = Path("./examples") / project_name
    discovered = set()
    if root.exists() and root.is_dir():
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            if (entry / "original_prompt.txt").exists() and (entry / "feedback_prompt.txt").exists():
                discovered.add(entry.name)
    return discovered or set(DEFAULT_TASKS)


def _normalize_task_name(task_name, allowed_tasks):
    task_name = str(task_name or "").strip()
    task_name = task_name.strip('/')
    task_name = TASK_ALIASES.get(task_name, task_name)
    if task_name not in allowed_tasks:
        raise ValueError(f"Unsupported task: {task_name}. Supported: {sorted(allowed_tasks)}")
    return task_name


def _resolve_task_sequence(args, allowed_tasks):
    raw_tasks = getattr(args, "optimize_tasks", None)
    if not raw_tasks:
        fallback_task = getattr(args, "task", "bump_var_function")
        return [_normalize_task_name(fallback_task, allowed_tasks)]

    if isinstance(raw_tasks, str):
        candidates = [part.strip() for part in raw_tasks.split(',') if part.strip()]
    elif isinstance(raw_tasks, (list, tuple, set)):
        candidates = [str(part).strip() for part in raw_tasks if str(part).strip()]
    else:
        candidates = [str(raw_tasks).strip()]

    normalized = []
    seen = set()
    for candidate in candidates:
        task_name = _normalize_task_name(candidate, allowed_tasks)
        if task_name in seen:
            continue
        normalized.append(task_name)
        seen.add(task_name)
    return normalized


def _run_task_sequence(args):
    main(args)


def _enable_realtime_output():
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass


def _infer_task_from_code(answer_code):
    text = str(answer_code or "").strip()
    match = re.search(r"void\s+Solver::([A-Za-z_]\w*)\s*\(", text)
    if not match:
        if "restart();" in text:
            return "restart_condition"
        return None
    func_name = match.group(1)
    if func_name == "restart":
        return "restart_function"
    if func_name == "rephase":
        return "rephase_function"
    if func_name == "bump_var":
        return "bump_var_function"
    return None


def _load_run_eval_artifacts(results_root):
    results_root = Path(results_root)
    final_result_path = results_root / "final_result.json"
    if not final_result_path.exists():
        raise FileNotFoundError(f"Cannot find run results for eval: {final_result_path}")

    with open(final_result_path, "r", encoding="utf-8") as f:
        final = json.load(f)

    checkpoint_path = results_root / "checkpoints" / "latest_checkpoint.json"
    extra_params = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        extra_params = {str(k): v for k, v in state.get("extra_params", {}).items()}

    return final, extra_params


def _validate_filled_solver(source_cpp_path, candidate_label):
    compile_out = Path(source_cpp_path).with_suffix(".compile_check")
    cmd = ["g++", "-O3", "-Wall", "-std=c++17", source_cpp_path, "-o", str(compile_out)]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if compile_out.exists():
            compile_out.unlink()
    except Exception:
        pass
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout or "").strip() or f"Compilation failed for {candidate_label}"


def _maybe_run_baseline_eval_for_task(args, task_name):
    baseline_solver_path = str(getattr(args, "SAT_solver_file_path", "") or "").strip()
    if not baseline_solver_path:
        baseline_solver_path = './examples/EasySAT/original_EasySAT/EasySAT.cpp'
    baseline_method_name = f"baseline_{task_name}_{args.llm_model}".replace('/', '')
    existing_baseline_eval = glob.glob(os.path.join(args.results_save_path, f"results_{baseline_method_name}_*.txt"))
    if existing_baseline_eval:
        print(f"[Eval] Skip baseline for {task_name}: already evaluated ({len(existing_baseline_eval)} file(s)).", flush=True)
        return
    print(f"[Eval] Run baseline for {task_name}: {baseline_method_name}", flush=True)
    prev_task = getattr(args, "task", "")
    try:
        args.task = task_name
        evaluate(args, method_name=baseline_method_name, SAT_solver_file_path=baseline_solver_path)
    finally:
        args.task = prev_task


def _collect_eval_candidates(final, extra_params, baseline):
    record_info = []
    for global_id_str, item in final.items():
        if global_id_str == "0":
            continue
        par_2 = item.get("PAR-2")
        answer_code = item.get("prompt", "")
        if par_2 is None or par_2 >= baseline:
            continue
        record_info.append((
            int(global_id_str),
            par_2,
            answer_code,
            extra_params.get(str(global_id_str), {}),
        ))
    record_info.sort(key=lambda x: x[1])
    return record_info



def _collect_eval_candidates_chronological(final, extra_params, baseline):
    record_info = []
    for global_id_str, item in final.items():
        if global_id_str == "0":
            continue
        par_2 = item.get("PAR-2")
        answer_code = item.get("prompt", "")
        if par_2 is None or par_2 >= baseline:
            continue
        record_info.append((
            int(global_id_str),
            par_2,
            answer_code,
            extra_params.get(str(global_id_str), {}),
        ))
    record_info.sort(key=lambda x: x[0])
    return record_info



def _select_eval_candidates(record_info, mode, fallback_task):
    mode = str(mode or "all").strip().lower()
    if mode == "all":
        return record_info
    if mode == "best_per_task":
        best_by_task = {}
        for global_id, par_2, answer_code, params_dict in record_info:
            task_name = _infer_task_from_code(answer_code) or fallback_task
            if not task_name:
                continue
            prev = best_by_task.get(task_name)
            if prev is None or par_2 < prev[1]:
                best_by_task[task_name] = (global_id, par_2, answer_code, params_dict)
        selected = list(best_by_task.values())
        selected.sort(key=lambda x: x[1])
        return selected
    if mode == "replay_writeback":
        return _collect_eval_candidates_chronological(
            {str(gid): {"PAR-2": par2, "prompt": code} for gid, par2, code, _ in record_info},
            {str(gid): params for gid, _, _, params in record_info},
            float("inf")
        )
    raise ValueError("eval_candidate_mode must be one of: all, best_per_task, replay_writeback")



def _latest_result_file(results_dir, method_name):
    matches = sorted(glob.glob(os.path.join(results_dir, f"results_{method_name}_*.txt")))
    return matches[-1] if matches else None



def _read_eval_par2(result_file):
    last_line = ""
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if not last_line:
        raise ValueError(f"Empty eval result file: {result_file}")
    payload = ast.literal_eval(last_line)
    if "PAR-2" not in payload:
        raise KeyError(f"PAR-2 key missing in {result_file}: {payload}")
    return float(payload["PAR-2"])



def _evaluate_candidate(args, task_name, method_name, answer_code, params_dict):
    project_dir = os.path.join(args.project, task_name)
    source_template = os.path.join("./examples/", project_dir, "EasySAT.cpp")
    if not os.path.exists(source_template):
        raise FileNotFoundError(f"task template not found for {task_name} ({source_template})")
    SAT_folder = os.path.join(args.temp_root, f'EasySAT_{method_name}')
    copy_folder(src_folder=args.temp_easy_root, num=1, mode='eval', target_folder=SAT_folder)
    SAT_solver_file_path = os.path.join(SAT_folder, 'EasySAT_modified.cpp')
    fill_core_codes(
        origin_file=source_template,
        target_file=SAT_solver_file_path,
        answer_code=answer_code,
        timeout=args.eval_timeout,
        data_dir="\"{}\"".format(args.eval_data_dir),
        **params_dict,
    )
    ok, compile_error = _validate_filled_solver(SAT_solver_file_path, method_name)
    if not ok:
        raise RuntimeError(compile_error)
    prev_task = getattr(args, "task", "")
    try:
        args.task = task_name
        evaluate(args, method_name=method_name, SAT_solver_file_path=SAT_solver_file_path)
    finally:
        args.task = prev_task
    result_file = _latest_result_file(args.results_save_path, method_name)
    if not result_file:
        raise FileNotFoundError(f"No eval result file found for {method_name}")
    return _read_eval_par2(result_file), SAT_solver_file_path



def _evaluate_final_composition(args, selected_codes_by_task, tag="final_composition"):
    if not selected_codes_by_task:
        return {}

    original_solver = os.path.join("./examples/", args.project.strip("/"), "original_EasySAT", "EasySAT.cpp")
    if not os.path.exists(original_solver):
        raise FileNotFoundError(f"Original solver template not found: {original_solver}")

    composite_dir = os.path.join(args.temp_root, f"EasySAT_{tag}")
    os.makedirs(composite_dir, exist_ok=True)
    composite_cpp = os.path.join(composite_dir, "EasySAT_modified.cpp")

    # Copy headers required by the solver source so standalone compile/eval works
    original_solver_dir = os.path.join("./examples/", args.project.strip("/"), "original_EasySAT")
    for hdr in ("EasySAT.hpp", "heap.hpp"):
        src_hdr = os.path.join(original_solver_dir, hdr)
        dst_hdr = os.path.join(composite_dir, hdr)
        if os.path.exists(src_hdr):
            import shutil as _shutil
            _shutil.copy(src_hdr, dst_hdr)

    # First render base solver to resolve Jinja placeholders like {{ timeout }} / {{ data_dir }}
    rendered_base_cpp = os.path.join(composite_dir, "EasySAT_base_rendered.cpp")
    revise_file(
        file_name=original_solver,
        save_dir=rendered_base_cpp,
        timeout=args.eval_timeout,
        data_dir="\"{}\"".format(args.eval_data_dir),
    )
    with open(rendered_base_cpp, "r", encoding="utf-8") as f:
        composite_content = f.read()

    def _strip_markers(code_text):
        text = str(code_text or "").strip()
        m = re.search(r"// start\n([\s\S]*?)\n// end", text, re.MULTILINE)
        return m.group(1).strip() if m else text

    for task_name, payload in sorted(selected_codes_by_task.items()):
        code = _strip_markers(payload.get("code", ""))
        if not code:
            continue
        if task_name == "bump_var_function":
            pattern = re.compile(r"void\s+Solver::bump_var\s*\([^)]*\)\s*\{[\s\S]*?\n\}", re.MULTILINE)
            composite_content, replaced = pattern.subn(code, composite_content, count=1)
            if replaced == 0:
                raise ValueError("Could not replace bump_var function in final composition")
        elif task_name == "restart_function":
            pattern = re.compile(r"void\s+Solver::restart\s*\([^)]*\)\s*\{[\s\S]*?\n\}", re.MULTILINE)
            composite_content, replaced = pattern.subn(code, composite_content, count=1)
            if replaced == 0:
                raise ValueError("Could not replace restart function in final composition")
        elif task_name == "rephase_function":
            pattern = re.compile(r"void\s+Solver::rephase\s*\([^)]*\)\s*\{[\s\S]*?\n\}", re.MULTILINE)
            composite_content, replaced = pattern.subn(code, composite_content, count=1)
            if replaced == 0:
                raise ValueError("Could not replace rephase function in final composition")
        elif task_name == "restart_condition":
            pattern = re.compile(r"else if \(lbd_queue_size == 50[\s\S]*?restart\(\);")
            composite_content, replaced = pattern.subn(code, composite_content, count=1)
            if replaced == 0:
                raise ValueError("Could not replace restart condition in final composition")
        else:
            raise ValueError(f"Unsupported task for final composition: {task_name}")

    with open(composite_cpp, "w", encoding="utf-8") as f:
        f.write(composite_content)

    ok, compile_error = _validate_filled_solver(composite_cpp, tag)
    if not ok:
        raise RuntimeError(f"Final composition compile failed: {compile_error}")

    method_name = f"{tag}_{args.llm_model}".replace('/', '')
    prev_task = getattr(args, "task", "")
    try:
        args.task = "whole_algorithm"
        evaluate(args, method_name=method_name, SAT_solver_file_path=composite_cpp)
    finally:
        args.task = prev_task

    result_file = _latest_result_file(args.results_save_path, method_name)
    if not result_file:
        raise FileNotFoundError(f"No eval result file found for final composition {method_name}")

    return {
        "composition": {
            task: {
                "global_id": payload.get("global_id"),
                "train_PAR-2": payload.get("train_par2"),
            }
            for task, payload in sorted(selected_codes_by_task.items())
        },
        "eval_PAR-2": _read_eval_par2(result_file),
        "solver_path": composite_cpp,
    }


def _run_eval_stage(args, final, extra_params, fallback_task, paths):
    print('start evaluation ...', flush=True)
    if os.path.exists(args.temp_results_dir):
        clean_files(folder_path=args.temp_results_dir, mode="all")

    baseline = final["0"]["PAR-2"]
    if bool(getattr(args, "original", False)):
        print('EasySAT baseline : {}'.format(baseline), flush=True)
    else:
        print('[Eval] Baseline threshold from run result (PAR-2): {}'.format(baseline), flush=True)

    eval_candidate_mode = str(getattr(args, "eval_candidate_mode", "all") or "all").strip().lower()
    eval_final_composition = bool(getattr(args, "eval_final_composition", False))

    all_record_info = _collect_eval_candidates(final, extra_params, baseline)
    record_info = _select_eval_candidates(all_record_info, eval_candidate_mode, fallback_task)
    print(f"[Eval] candidate_mode={eval_candidate_mode}; selected {len(record_info)} candidate(s) out of {len(all_record_info)}", flush=True)
    if len(record_info) == 0:
        return

    if bool(getattr(args, "eval_baseline", True)):
        baseline_tasks = set()
        for _, _, answer_code, _ in record_info:
            task_name = _infer_task_from_code(answer_code) or fallback_task
            if task_name:
                baseline_tasks.add(task_name)
        for task_name in sorted(baseline_tasks):
            _maybe_run_baseline_eval_for_task(args, task_name)
    else:
        print("[Eval] Baseline evaluation disabled by config.", flush=True)

    accepted_by_task = {}
    current_best_eval = {}
    if eval_candidate_mode == "replay_writeback":
        for task_name in sorted({_infer_task_from_code(code) or fallback_task for _, _, code, _ in record_info if (_infer_task_from_code(code) or fallback_task)}):
            baseline_method_name = f"baseline_{task_name}_{args.llm_model}".replace('/', '')
            baseline_file = _latest_result_file(args.results_save_path, baseline_method_name)
            if baseline_file:
                current_best_eval[task_name] = _read_eval_par2(baseline_file)

    for idx, (global_id, par_2, answer_code, params_dict) in enumerate(record_info, start=1):
        task_name = _infer_task_from_code(answer_code) or fallback_task
        if not task_name:
            warnings.warn(
                f"[Eval] Skip candidate {global_id}: cannot infer heuristic target from generated code.",
                category=UserWarning,
                stacklevel=2,
            )
            continue

        method_name = f"{task_name}_{args.llm_model}_{global_id}".replace('/', '')
        existing_eval = glob.glob(os.path.join(args.results_save_path, f"results_{method_name}_*.txt"))
        if existing_eval:
            print(f"[Eval] Skip {idx}/{len(record_info)} {method_name}: already evaluated ({len(existing_eval)} file(s)).", flush=True)
            try:
                eval_par2 = _read_eval_par2(existing_eval[-1])
                if eval_candidate_mode == "replay_writeback":
                    prev_best = current_best_eval.get(task_name, float("inf"))
                    if eval_par2 < prev_best:
                        current_best_eval[task_name] = eval_par2
                        accepted_by_task[task_name] = {
                            "global_id": global_id,
                            "train_par2": par_2,
                            "eval_par2": eval_par2,
                            "code": answer_code,
                            "params": params_dict,
                        }
                elif eval_candidate_mode == "best_per_task":
                    accepted_by_task[task_name] = {
                        "global_id": global_id,
                        "train_par2": par_2,
                        "eval_par2": eval_par2,
                        "code": answer_code,
                        "params": params_dict,
                    }
            except Exception:
                pass
            continue

        print(f"[Eval] Run {idx}/{len(record_info)}: {method_name}", flush=True)
        try:
            eval_par2, _ = _evaluate_candidate(args, task_name, method_name, answer_code, params_dict)
            if eval_candidate_mode == "replay_writeback":
                prev_best = current_best_eval.get(task_name, float("inf"))
                if eval_par2 < prev_best:
                    current_best_eval[task_name] = eval_par2
                    accepted_by_task[task_name] = {
                        "global_id": global_id,
                        "train_par2": par_2,
                        "eval_par2": eval_par2,
                        "code": answer_code,
                        "params": params_dict,
                    }
                    print(f"[EvalReplay] ACCEPT task={task_name} gid={global_id} eval_PAR-2={eval_par2:.2f} < prev_best={prev_best:.2f}", flush=True)
                else:
                    print(f"[EvalReplay] SKIP task={task_name} gid={global_id} eval_PAR-2={eval_par2:.2f} >= prev_best={prev_best:.2f}", flush=True)
            elif eval_candidate_mode == "best_per_task":
                accepted_by_task[task_name] = {
                    "global_id": global_id,
                    "train_par2": par_2,
                    "eval_par2": eval_par2,
                    "code": answer_code,
                    "params": params_dict,
                }
        except Exception as exc:
            warnings.warn(
                f"[Eval] Skip candidate {global_id} ({task_name}): eval failed.\n{exc}",
                category=UserWarning,
                stacklevel=2,
            )
            continue

    if eval_final_composition:
        if eval_candidate_mode == "all":
            best_by_task = {}
            for global_id, par_2, answer_code, params_dict in all_record_info:
                task_name = _infer_task_from_code(answer_code) or fallback_task
                if not task_name:
                    continue
                prev = best_by_task.get(task_name)
                if prev is None or par_2 < prev["train_par2"]:
                    best_by_task[task_name] = {
                        "global_id": global_id,
                        "train_par2": par_2,
                        "code": answer_code,
                        "params": params_dict,
                    }
            accepted_by_task = best_by_task
        if accepted_by_task:
            print(f"[Eval] Running ONE final composite solver with {len(accepted_by_task)} selected best functions...", flush=True)
            summary = _evaluate_final_composition(args, accepted_by_task, tag=f"final_{eval_candidate_mode}")
            summary_path = os.path.join(args.results_root, f"eval_final_composition_{eval_candidate_mode}.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[Eval] Final composition summary saved to {summary_path}", flush=True)


def _maybe_run_baseline_eval(args):
    _maybe_run_baseline_eval_for_task(args, getattr(args, "task", ""))


def _writeback_template(task_name: str, code: str, project: str) -> None:
    """Replace {{ replace_code }} in the task EasySAT.cpp template with *code*.

    Saves a .bak backup on the first write per template file.
    Raises if the template is missing or has no {{ replace_code }} placeholder.
    """
    template_path = os.path.join("./examples/", project.strip("/"), task_name, "EasySAT.cpp")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    placeholder_re = re.compile(r"\{\{\s*replace_code\s*\}\}")
    if not placeholder_re.search(content):
        raise ValueError(f"No {{{{ replace_code }}}} placeholder in {template_path}")

    backup_path = template_path + ".bak"
    if not os.path.exists(backup_path):
        import shutil as _shutil
        _shutil.copy(template_path, backup_path)
        print(f"[WriteBack] Backup saved: {backup_path}", flush=True)

    # Use a lambda to avoid re.sub interpreting backslashes in the replacement string
    new_content = placeholder_re.sub(lambda _: code, content)
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"[WriteBack] Updated template: {template_path}", flush=True)


def _select_task_for_run(args):
    available_tasks = _discover_available_tasks(getattr(args, "project", "EasySAT/"))
    task_sequence = _resolve_task_sequence(args, available_tasks)
    if not task_sequence:
        raise ValueError("No optimize_tasks resolved for run")

    selection_mode = str(getattr(args, "task_selection_mode", "random_one") or "random_one").strip().lower()
    if selection_mode not in {"random_one", "cycle", "sequential_all"}:
        raise ValueError("task_selection_mode must be one of: random_one, cycle, sequential_all")

    if selection_mode == "random_one":
        rng = random.Random(int(getattr(args, "rand_seed", 42) or 42))
        selected_task = rng.choice(task_sequence)
    else:
        selected_task = task_sequence[0]

    args.task = selected_task
    print(f"[TaskSelect] selected task={selected_task} from {task_sequence}", flush=True)


def _task_for_iteration(task_sequence, selection_mode, iter_idx, rand_seed=42):
    if not task_sequence:
        raise ValueError("task_sequence must not be empty")

    selection_mode = str(selection_mode or "random_one").strip().lower()
    if selection_mode == "random_one":
        rng = random.Random(int(rand_seed or 42))
        return rng.choice(task_sequence)
    if selection_mode in {"cycle", "sequential_all"}:
        return task_sequence[iter_idx % len(task_sequence)]
    raise ValueError("task_selection_mode must be one of: random_one, cycle, sequential_all")


def _make_run_id(explicit_run_id=""):
    explicit_run_id = str(explicit_run_id or "").strip()
    if explicit_run_id:
        return explicit_run_id
    return time.strftime("run_%Y%m%d_%H%M%S") + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _run_paths(run_id, task_namespace=""):
    base_temp_root = Path("./temp/runs") / run_id
    base_results_root = Path("./results/runs") / run_id
    task_namespace = str(task_namespace or "").strip().strip('/')

    temp_root = base_temp_root / task_namespace if task_namespace else base_temp_root
    results_root = base_results_root / task_namespace if task_namespace else base_results_root
    temp_results_dir = temp_root / "results"
    temp_prompts_dir = temp_root / "prompts"
    temp_easy_root = temp_root / "EasySAT"
    checkpoint_dir = results_root / "checkpoints"
    snapshots_dir = results_root / "snapshots"
    eval_results_dir = results_root / "eval_results"
    for path in [temp_results_dir, temp_prompts_dir, temp_easy_root, checkpoint_dir, snapshots_dir, eval_results_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": run_id,
        "base_temp_root": base_temp_root,
        "base_results_root": base_results_root,
        "task_namespace": task_namespace,
        "temp_root": temp_root,
        "temp_results_dir": temp_results_dir,
        "temp_prompts_dir": temp_prompts_dir,
        "temp_easy_root": temp_easy_root,
        "results_root": results_root,
        "checkpoint_dir": checkpoint_dir,
        "snapshots_dir": snapshots_dir,
        "eval_results_dir": eval_results_dir,
    }


def _ensure_parent_dir(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def _checkpoint_paths(base_dir="./results/checkpoints"):
    checkpoint_dir = Path(base_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, checkpoint_dir / "latest_checkpoint.json"


def _atomic_write_json(path_obj, payload):
    path_obj = Path(path_obj)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path_obj)


LLM_CONFIG_KEYS = {
    "llm_model",
    "api_base",
    "api_key",
    "model_name",
}


def _sanitize_config_for_persistence(config_dict):
    sanitized = {}
    for key, value in (config_dict or {}).items():
        if key in LLM_CONFIG_KEYS:
            continue
        sanitized[key] = value
    return sanitized


def _collect_runtime_llm_env():
    snapshot = {}
    for key in ["AUTOSAT_API_TYPE", "AUTOSAT_LLM_MODEL", "AUTOSAT_API_BASE"]:
        value = (os.getenv(key) or "").strip()
        if value:
            snapshot[key] = value
    if (os.getenv("AUTOSAT_API_KEY") or "").strip():
        snapshot["AUTOSAT_API_KEY"] = "***redacted***"
    return snapshot


def _save_run_metadata(paths, args, config_payload):
    metadata = {
        "run_id": str(getattr(args, "run_id", "") or ""),
        "task_namespace": str(getattr(args, "task_namespace", "") or ""),
        "config_path": str(getattr(args, "config", "") or ""),
        "saved_at": time.time(),
        "runtime_llm_env": _collect_runtime_llm_env(),
        "config": _sanitize_config_for_persistence(config_payload),
    }
    _atomic_write_json(Path(paths["results_root"]) / "run_metadata.json", metadata)


def _save_checkpoint(next_iter, results, answers, extra_params, best_result, checkpoint_dir="./results/checkpoints", run_id=""):
    checkpoint_dir, latest_path = _checkpoint_paths(checkpoint_dir)
    state = {
        "version": 1,
        "run_id": str(run_id or ""),
        "next_iter": int(next_iter),
        "results": results,
        "answers": {str(k): v for k, v in answers.items()},
        "extra_params": {str(k): v for k, v in extra_params.items()},
        "best_result": best_result,
        "saved_at": time.time(),
    }
    _atomic_write_json(latest_path, state)
    _atomic_write_json(checkpoint_dir / f"iter_{next_iter - 1}_checkpoint.json", state)


def _load_checkpoint(checkpoint_dir="./results/checkpoints", checkpoint_path=""):
    if checkpoint_path:
        candidate_path = Path(checkpoint_path)
        if candidate_path.is_dir():
            candidate_path = candidate_path / "latest_checkpoint.json"
        latest_path = candidate_path
    else:
        _, latest_path = _checkpoint_paths(checkpoint_dir)

    if not latest_path.exists():
        return None
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except json.JSONDecodeError as exc:
        warnings.warn(
            f"Checkpoint file is corrupted or partially written: {latest_path}. Error: {exc}",
            category=UserWarning,
            stacklevel=2,
        )
        return None

    state["answers"] = {int(k): v for k, v in state.get("answers", {}).items()}
    state["extra_params"] = {int(k): v for k, v in state.get("extra_params", {}).items()}
    return state


def _save_iteration_artifacts(iter_idx, result, best_result, temp_prompts_dir, results_root, snapshots_dir):
    results_root.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    with open(temp_prompts_dir / f'iter_{iter_idx}_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(results_root / f'iter_{iter_idx}_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if best_result and len(best_result) > 0:
        best_id = next(iter(best_result.keys()))
        best_data = best_result[best_id]
        snapshot = {
            "iter": iter_idx,
            "best_id": best_id,
            "time": best_data[0],
            "code": best_data[1],
            "PAR-2": best_data[2],
        }
        with open(snapshots_dir / f'iter_{iter_idx}_best.json', 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)


def _extract_reference_code(prompt_file_dir):
    with open(prompt_file_dir, 'r', encoding='utf-8') as file:
        prompt_text = file.read()

    scoped_pattern = re.compile(
        r"To replace the original code:\s*'''[\s\S]*?// start\n([\s\S]*?)\n// end",
        re.MULTILINE,
    )
    scoped_match = scoped_pattern.search(prompt_text)
    if scoped_match:
        candidate = scoped_match.group(1).strip()
        if _looks_like_solver_code(candidate):
            return candidate

    for match in re.finditer(r"// start\n([\s\S]*?)\n// end", prompt_text, re.MULTILINE):
        candidate = match.group(1).strip()
        if _looks_like_solver_code(candidate):
            return candidate

    return ''


def _looks_like_solver_code(text):
    if not text or len(text) < 20:
        return False
    banned_fragments = [
        "must start with",
        "Tips:",
        "execution time",
        "''' and end with '''",
    ]
    lowered = text.lower()
    if any(fragment.lower() in lowered for fragment in banned_fragments):
        return False
    code_markers = ["void Solver::", "else if", "restart();", "{", ";"]
    return any(marker in text for marker in code_markers)


def _load_env_file_candidates():
    candidates = [
        '.env',
        os.path.join('..', '.env'),
        os.path.join('..', '..', '.env'),
        os.path.join(os.path.dirname(__file__), '.env'),
        os.path.join(os.path.dirname(__file__), '..', '.env'),
        os.path.join(os.path.dirname(__file__), '..', '..', '.env'),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if (not line) or line.startswith('#') or ('=' not in line):
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        break


def _apply_env_overrides(args):
    _load_env_file_candidates()
    args.llm_model = os.getenv('AUTOSAT_LLM_MODEL', os.getenv('DEEPINFRA_MODEL', args.llm_model))
    args.api_base = os.getenv('AUTOSAT_API_BASE', os.getenv('DEEPINFRA_API_BASE', args.api_base))
    args.api_key = os.getenv('AUTOSAT_API_KEY', os.getenv('DEEPINFRA_API_KEY', args.api_key))
    return args


def synchronized_asked(prompt_file_dir, count, args):

    llm_api = get_llm_api(args)
    temperature = getattr(args, 'temperature', 1.0)
    use_structured = getattr(llm_api, '_structured_output', False)

    if use_structured:
        # Structured output path: model returns JSON with "code" and "lbd_queue_size"
        answer_code, lbd_queue_size = llm_api.call_api_structured(
            prompt_file=prompt_file_dir, temperature=temperature
        )
        print(f"[StructuredOutput] iter_count={count} code_len={len(answer_code)} lbd={lbd_queue_size!r}", flush=True)
    else:
        # Legacy text-parsing path
        answer = llm_api.call_api(prompt_file=prompt_file_dir, temperature=temperature)
        answer_code = get_code(answer, seperator=['// start\n', '\n// end'])
        lbd_queue_size = get_code(answer, seperator=['// start lbd_queue_size\n', '\n// end lbd_queue_size'])

    # If structured output returned a raw JSON string (from _call_structured_raw via call_api),
    # try to parse it here as a safety net.
    if answer_code.strip().startswith('{'):
        parsed_code, parsed_lbd = _parse_structured_response(answer_code)
        if parsed_code:
            answer_code = parsed_code
            if parsed_lbd:
                lbd_queue_size = parsed_lbd

    if len(answer_code.strip()) < 20:
        reference_code = _extract_reference_code(prompt_file_dir)
        if len(reference_code.strip()) >= 20:
            answer_code = reference_code

    return count, answer_code, lbd_queue_size.strip()


def synchronized_executed(count, results, arguments, answer_code, *args, **kwargs):
    project_dir = os.path.join(arguments.project, arguments.task)
    execution_worker = ExecutionWorker()

    if arguments.devoid_duplication and (answer_code in results["prompt"].values()):
        return count, 0, answer_code
    else:
        save_path = os.path.join(arguments.temp_root, f"EasySAT_{format((count - 1) % arguments.batch_size)}", "EasySAT.cpp")
        _ensure_parent_dir(save_path)
        revise_file(file_name=os.path.join("./examples/", project_dir, "EasySAT.cpp"),
                    save_dir=save_path,
                    replace_code=answer_code,
                    timeout=arguments.timeout,
                    data_dir="\"{}\"".format(arguments.data_dir),
                    *args, **kwargs
                    )
        success = execution_worker.execute(count, arguments.batch_size, arguments.data_parallel_size)
        return count, success, answer_code


def main(args):
    _enable_realtime_output()
    eval_only_from_run = bool(getattr(args, "eval_only_from_run", False))
    available_tasks = _discover_available_tasks(getattr(args, "project", "EasySAT/"))
    task_sequence = _resolve_task_sequence(args, available_tasks)
    if not task_sequence:
        raise ValueError("No optimize_tasks resolved for run")
    selection_mode = str(getattr(args, "task_selection_mode", "random_one") or "random_one").strip().lower()
    if selection_mode not in {"random_one", "cycle", "sequential_all"}:
        raise ValueError("task_selection_mode must be one of: random_one, cycle, sequential_all")

    selected_task = _task_for_iteration(task_sequence, selection_mode, 0, rand_seed=getattr(args, "rand_seed", 42))
    args.task = selected_task
    print(f"[TaskSelect] selected task={selected_task} from {task_sequence} mode={selection_mode}", flush=True)
    run_id = _make_run_id(getattr(args, "run_id", ""))
    os.environ["AUTOSAT_RUN_ID"] = run_id
    task_namespace = str(getattr(args, "task_namespace", "") or "").strip().strip('/')
    if task_namespace:
        os.environ["AUTOSAT_TASK_NAMESPACE"] = task_namespace
    else:
        os.environ.pop("AUTOSAT_TASK_NAMESPACE", None)
    paths = _run_paths(run_id, task_namespace=task_namespace)
    args.run_id = run_id
    args.temp_root = str(paths["temp_root"])
    args.temp_results_dir = './temp/results/'
    args.temp_prompts_dir = str(paths["temp_prompts_dir"])
    args.temp_easy_root = str(paths["temp_easy_root"])
    args.results_root = str(paths["results_root"])
    args.checkpoint_dir = str(paths["checkpoint_dir"])
    args.snapshots_dir = str(paths["snapshots_dir"])
    args.eval_results_dir = str(paths["eval_results_dir"])
    args.results_save_path = args.eval_results_dir
    os.makedirs(args.temp_results_dir, exist_ok=True)
    _save_run_metadata(paths, args, getattr(args, "_loaded_config_payload", {}))

    if eval_only_from_run:
        if not os.path.exists(args.temp_easy_root):
            train_init(args)
        final, extra_params = _load_run_eval_artifacts(args.results_root)
        if "0" not in final:
            raise ValueError(f"Baseline result with key '0' is missing in {args.results_root}/final_result.json")
        _run_eval_stage(args, final, extra_params, fallback_task=selected_task, paths=paths)
        return

    data_dir = args.data_dir
    data_num = len([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))])
    if args.data_parallel_size > data_num:
        warnings.warn(f"The parallel num for training is too large: {args.data_parallel_size} > {data_num}. "
                      f"It will be replaced with the train set total num: {data_num}",
                      category=UserWarning, stacklevel=2)
        setattr(args, 'data_parallel_size', data_num)
    execution_worker = ExecutionWorker()

    answers = {}  # record answers from llm.
    extra_params = {}  # e.g. lbd_queue_size
    count = 0
    best_result = {}
    start_iter = 0
    results = {
        "time": {},
        "prompt": {},
        "PAR-2": {}
    }

    project_dir = os.path.join(args.project, args.task)
    print("project_dir: {}".format(project_dir), flush=True)

    resume_enabled = bool(getattr(args, "resume_from_checkpoint", True))
    checkpoint_dir = str(getattr(args, "checkpoint_dir", str(paths["checkpoint_dir"])) or str(paths["checkpoint_dir"]))
    checkpoint_path = str(getattr(args, "checkpoint_path", "") or "").strip()
    resumed = False
    if resume_enabled:
        state = _load_checkpoint(checkpoint_dir=checkpoint_dir, checkpoint_path=checkpoint_path)
        if state is not None:
            loaded_run_id = str(state.get("run_id", run_id) or run_id)
            if loaded_run_id and loaded_run_id != run_id:
                run_id = loaded_run_id
                os.environ["AUTOSAT_RUN_ID"] = run_id
                paths = _run_paths(run_id, task_namespace=task_namespace)
                args.run_id = run_id
                args.temp_root = str(paths["temp_root"])
                args.temp_results_dir = str(paths["temp_results_dir"])
                args.temp_prompts_dir = str(paths["temp_prompts_dir"])
                args.temp_easy_root = str(paths["temp_easy_root"])
                args.results_root = str(paths["results_root"])
                args.checkpoint_dir = str(paths["checkpoint_dir"])
                args.snapshots_dir = str(paths["snapshots_dir"])
                args.eval_results_dir = str(paths["eval_results_dir"])
                args.results_save_path = args.eval_results_dir
                checkpoint_dir = args.checkpoint_dir
            start_iter = int(state.get("next_iter", 0))
            results = state.get("results", results)
            answers = state.get("answers", answers)
            extra_params = state.get("extra_params", extra_params)
            best_result = state.get("best_result", best_result)
            resumed = True
            print(f"[Checkpoint] Resumed from iteration {start_iter}.", flush=True)

    baseline_initialized_now = False
    baseline_executed_now = False
    if (not resumed) or ("0" not in results.get("time", {})):
        train_init(args)

        baseline_cpp = os.path.join(args.temp_root, f"EasySAT_{count}", "EasySAT.cpp")
        _ensure_parent_dir(baseline_cpp)
        revise_file(
            file_name=os.path.join("./examples/", project_dir, "EasySAT_original.cpp"),
            save_dir=baseline_cpp,
            timeout=args.timeout,
            data_dir="\"{}\"".format(args.data_dir),
        )

        if args.original:
            success = execution_worker.execute_original(count, args.data_parallel_size)
            assert (count == 0)
            filenames = [str(count) + "_" + str(num) + ".txt" for num in range(args.data_parallel_size)]
            start_time = time.time()
            while True:
                end_time = time.time()
                if end_time-start_time > args.timeout * (2*data_num/args.data_parallel_size):
                    raise ValueError("Infinite loop error!!!")
                all_exist = all(os.path.exists(os.path.join(args.temp_results_dir, 'finished'+filename)) for filename in filenames)
                if all_exist:
                    results, best_result = collect_results(answers={0: ''},
                                                           repetition_dict={},
                                                           results={},
                                                           args=args)
                    baseline_executed_now = True
                    break
        else:
            results["time"]["0"] = args.original_result['time']
            results["prompt"]["0"] = " "
            results["PAR-2"]["0"] = args.original_result['PAR-2']
        baseline_initialized_now = True
    else:
        if not os.path.exists(os.path.join(args.temp_root, "EasySAT_0")):
            train_init(args)
    if baseline_initialized_now and baseline_executed_now:
        print("EasySAT(baseline) result-- time: {} seconds ; PAR-2: {}".format(results["time"]["0"], results["PAR-2"]["0"]), flush=True)

    # --- Save baseline result to run directory ---
    if baseline_initialized_now:
        baseline_payload = {
            "time": results["time"].get("0"),
            "PAR-2": results["PAR-2"].get("0"),
            "prompt": results["prompt"].get("0", ""),
            "executed": baseline_executed_now,
        }
        _atomic_write_json(paths["results_root"] / "baseline_result.json", baseline_payload)
        print(f"[Baseline] Saved baseline_result.json → {paths['results_root'] / 'baseline_result.json'}", flush=True)

    start_iter_override = getattr(args, "start_iter_override", None)
    end_iter_override = getattr(args, "end_iter_override", None)
    loop_start = start_iter if start_iter_override is None else int(start_iter_override)
    loop_end = int(args.iteration_num) if end_iter_override is None else int(end_iter_override)
    if loop_end < loop_start:
        loop_end = loop_start

    result = {}  # Initialize result to avoid UnboundLocalError
    # template_update_strategy controls whether/how the EasySAT.cpp template is
    # updated during training when a better implementation is found.
    #   "none"         – no write-back (default, original behaviour)
    #   "greedy_train" – write back only when training PAR-2 strictly improves over prev written
    #   "annealing"    – write back using SA Metropolis criterion with log-annealed temperature
    #                    T(i) = 1.0*(0.1/1.0)^(i/(N-1)); objective = δPAR2/(0.25*PAR2_baseline)
    _template_strategy = str(
        getattr(args, "template_update_strategy", "none") or "none"
    ).strip().lower()
    # Track last written-back PAR-2 per task to avoid redundant writes
    _writeback_par2_per_task: dict = {}
    # Track iteration when template was last written per task (for current_code_age)
    _writeback_age_per_task: dict = {}

    for i in range(loop_start, loop_end):

        current_task = _task_for_iteration(task_sequence, selection_mode, i, rand_seed=getattr(args, "rand_seed", 42))
        args.task = current_task
        project_dir = os.path.join(args.project, current_task)
        # clean temp results
        clean_files(folder_path=args.temp_results_dir, mode="all")
        id_list = []

        # --- DEBUG: выводим task и prompt ---
        print(f"[DEBUG] ITER {i} TASK {args.task} PROJECT_DIR {project_dir}")

        if i == 0:
            prompt_file_dir = os.path.join("./examples/", project_dir, "original_prompt.txt")
        elif check_reIteration(round=i,best_result_dict=best_result,
                               baseline={'time': results["time"]["0"],'PAR-2': results["PAR-2"]["0"]}):
            # restart at iteration-1 if necessary..
            prompt_file_dir = os.path.join("./examples/", project_dir, "original_prompt.txt")
        else:
            if result and "time" in result and len(result["time"]) > 0:
                result_prompt = [
                    f"Experiment {num}, Your provided {args.project}--{args.task}: \n'''{result['prompt'][value[0]]}''', \n execution time is {result['time'][value[0]]} seconds. PAR-2 ( Penalized Average Runtime with factor 2) is {result['PAR-2'][value[0]]} seconds. "
                    for num, value in enumerate(result["time"].items())]
                result_prompt = '\n '.join(result_prompt)
                print("iteration: ", i, "\n results_prompt: \n", result_prompt, flush=True)

                # Add the result matrix into next round prompt.
                _feedback_extra = {}

                # Always pass training progress info so all feedback_prompt.txt templates can use it
                _feedback_extra["current_iteration"] = i
                _feedback_extra["total_iterations"] = loop_end

                if _template_strategy in ("greedy_train", "annealing"):
                    # original_code: baseline function from original_prompt.txt (never changes)
                    _orig_prompt = os.path.join("./examples/", project_dir, "original_prompt.txt")
                    _feedback_extra["original_code"] = _extract_reference_code(_orig_prompt)

                    # current_template_code: current function from EasySAT.cpp template
                    # (may have been updated by write-back; reflects what the LLM will build upon)
                    _current_template = os.path.join("./examples/", project_dir, "EasySAT.cpp")
                    _current_code = _extract_reference_code(_current_template)
                    # Only show if different from baseline (i.e. write-back has occurred)
                    _feedback_extra["current_template_code"] = _current_code

                    # current_code_age: how many iterations ago the template was last updated
                    _last_wb_iter = _writeback_age_per_task.get(current_task, None)
                    if _last_wb_iter is None:
                        _feedback_extra["current_code_age"] = "original (never updated during this run)"
                    else:
                        _age = i - _last_wb_iter
                        _feedback_extra["current_code_age"] = (
                            f"Updated {_age} iteration(s) ago (at iteration {_last_wb_iter})"
                        )

                    # training_history: PAR-2 per iteration, sorted by key
                    _hist_lines = []
                    _baseline_par2 = results["PAR-2"].get("0")
                    if _baseline_par2 is not None:
                        _hist_lines.append(f"iter 0 (baseline): {_baseline_par2:.0f}")
                    for _hk in sorted(results["PAR-2"].keys(), key=lambda k: int(k) if k.isdigit() else float("inf")):
                        if _hk == "0":
                            continue
                        _hv = results["PAR-2"][_hk]
                        _hist_lines.append(f"iter {_hk}: {_hv:.0f}")
                    _feedback_extra["training_history"] = "\n".join(_hist_lines)

                revise_file(file_name= os.path.join("./examples/", project_dir, "feedback_prompt.txt"),
                            save_dir=os.path.join(args.temp_prompts_dir, 'feedback_prompt.txt'),
                            replace_code=result_prompt,
                            original_time=int(results["time"]["0"]),
                            best_code=list(best_result.values())[0][1],
                            **_feedback_extra
                            )
                prompt_file_dir = os.path.join(args.temp_prompts_dir, 'feedback_prompt.txt')
            else:
                # No valid result from previous iteration, restart from original prompt
                print("iteration: ", i, " (no valid results from previous iteration, using original prompt)", flush=True)
                prompt_file_dir = os.path.join("./examples/", project_dir, "original_prompt.txt")

        print(f"[DEBUG] PROMPT FILE: {prompt_file_dir}")
        try:
            with open(prompt_file_dir, 'r') as f:
                print("[DEBUG] PROMPT CONTENT:\n" + f.read())
        except Exception as e:
            print(f"[DEBUG] PROMPT READ ERROR: {e}")

        # --- Simulated annealing acceptance temperature (for write-back decision, NOT LLM temperature) ---
        # T(i) = 1.0 * (0.1 / 1.0) ^ (i / max(N-1, 1))  — log-annealing from 1.0 to 0.1
        # Used in Metropolis acceptance criterion: accept worse solution with prob exp(-objective / T)
        # where objective = δPAR2 / (0.25 * PAR2_baseline)
        # Only active when template_update_strategy == "annealing"
        if _template_strategy == "annealing":
            _T_max_sa, _T_min_sa = 1.0, 0.1
            _N_sa = max(loop_end - loop_start, 1)
            _sa_accept_temp = _T_max_sa * (_T_min_sa / _T_max_sa) ** (i / max(_N_sa - 1, 1))
            print(f"[SA] iter={i} acceptance_temperature={_sa_accept_temp:.4f} "
                  f"(log-annealing {_T_max_sa}→{_T_min_sa} over {_N_sa} iters)", flush=True)
        else:
            _sa_accept_temp = None

        start_time = time.time()
        answer_code_cur_round = {}
        lbd_queue_size_cur_round = {}
        tasks = [synchronized_asked(prompt_file_dir, i * args.batch_size + batch_id + 1, args)
                 for batch_id in range(args.batch_size)]
        print(f"[Iteration {i}] LLM responses collected from {len(tasks)} tasks.", flush=True)
        for future in tasks:
            count, answer_code, lbd_queue_size = future
            batch_id = get_batch_id(count, args.batch_size)
            answer_code_cur_round[batch_id] = answer_code
            lbd_queue_size_cur_round[batch_id] = lbd_queue_size if lbd_queue_size.isdigit() else '50' # original lbd_queue_size = 50
        end_time = time.time()
        print("querying consuming: {} seconds".format(end_time-start_time), flush=True)

        start_time = time.time()
        if args.task == "restart_condition":
            tasks = [synchronized_executed(
                     count=i * args.batch_size + batch_id + 1,
                     results=results, arguments=args,
                     answer_code=answer_code_cur_round[batch_id],
                     lbd_queue_size=lbd_queue_size_cur_round[batch_id]) for batch_id in range(args.batch_size)]

        else:
            tasks = [synchronized_executed(
                     count=i * args.batch_size + batch_id + 1,
                     results=results, arguments=args,
                     answer_code=answer_code_cur_round[batch_id]) for batch_id in range(args.batch_size)]

        repetition_dict = {}
        print(f"[Iteration {i}] Execution results collected from {len(tasks)} tasks.", flush=True)
        for future in tasks:
            count, success, answer_code = future
            answers[count] = answer_code

            extra_params[count] = {'lbd_queue_size': lbd_queue_size_cur_round[get_batch_id(count, args.batch_size)]}

            if success:
                id_list.append(count)
            elif args.devoid_duplication and success == 0:
                repetition_dict[count] = answer_code
        end_time = time.time()
        print("sending execution time consuming: {} seconds.".format(end_time-start_time), flush=True)
        start_time = time.time()
        filenames = [str(global_id) + "_" + str(num) + ".txt" for global_id in id_list for num in range(args.data_parallel_size)]
        print(f"[Iteration {i}] Waiting for {len(filenames)} result files...", flush=True)
        while True:
            end_time = time.time()
            elapsed = end_time - start_time
            timeout_limit = args.timeout * (2*data_num/args.data_parallel_size)
            if elapsed > timeout_limit:
                warnings.warn(f"Solver timeout after {elapsed:.1f}s (limit {timeout_limit:.1f}s). Collecting partial results.",
                              category=UserWarning, stacklevel=2)
                result, best_result = collect_results(answers=answers,
                                                      repetition_dict=repetition_dict,
                                                      results=results,
                                                      args=args)
                delete_InfiniteLoopInst(candidates=['finished'+fname for fname in filenames], result_dict=result)
                break
            all_exist = all(os.path.exists(os.path.join(args.temp_results_dir, 'finished'+filename)) for filename in filenames)
            if all_exist:
                result, best_result = collect_results(answers=answers,
                                                      repetition_dict=repetition_dict,
                                                      results=results,
                                                      args=args)
                break
            if elapsed % 30 == 0 and elapsed > 0:
                missing = [f for f in filenames if not os.path.exists(os.path.join(args.temp_results_dir, 'finished'+f))]
                print(f"[Iteration {i}] Still waiting for {len(missing)} files after {elapsed:.1f}s...", flush=True)

        print("collecting execution time consuming: {} seconds.".format(end_time-start_time), flush=True)
        if len(id_list) == 0:
            warnings.warn(
                "No generated candidate compiled and executed successfully in this iteration. "
                "Keeping the previous best_result and continuing.",
                category=UserWarning,
                stacklevel=2,
            )
            result = {
                "time": {},
                "prompt": {},
                "PAR-2": {},
            }
            if 'best_result' not in locals() or len(best_result) == 0:
                best_result = {"0": [results["time"]["0"], results["prompt"]["0"], results["PAR-2"]["0"]]}
        results["time"].update(result["time"])
        results["PAR-2"].update(result["PAR-2"])
        results["prompt"].update(result["prompt"])
        _save_iteration_artifacts(i, result, best_result, paths["temp_prompts_dir"], paths["results_root"], paths["snapshots_dir"])
        _save_checkpoint(i + 1, results, answers, extra_params, best_result, checkpoint_dir=checkpoint_dir, run_id=run_id)

        # --- Training write-back: update template when strategy requires it ---
        # Controlled by template_update_strategy in config:
        #   "none"         – no write-back (default)
        #   "greedy_train" – write back only when PAR-2 strictly improves over last written
        #   "annealing"    – write back using SA Metropolis criterion (see _sa_accept_temp above)
        if _template_strategy in ("greedy_train", "annealing") and best_result:
            try:
                import math as _math
                import random as _random
                best_key = list(best_result.keys())[0]
                best_par2 = best_result[best_key][2]
                best_code = best_result[best_key][1]
                baseline_par2 = results["PAR-2"].get("0", float("inf"))
                if best_par2 < baseline_par2 and best_code and len(best_code.strip()) >= 20:
                    task_for_writeback = _infer_task_from_code(best_code) or current_task
                    prev_written = _writeback_par2_per_task.get(task_for_writeback, float("inf"))
                    if _template_strategy == "greedy_train":
                        # Strict improvement only
                        _accept = best_par2 < prev_written
                        _accept_reason = (
                            f"strict improvement (PAR-2={best_par2:.0f} < prev={prev_written:.0f})"
                            if _accept else
                            f"no improvement (PAR-2={best_par2:.0f} >= prev={prev_written:.0f}), skipped"
                        )
                    else:
                        # SA Metropolis acceptance criterion
                        # objective = δPAR2 / (0.25 * PAR2_baseline)
                        # δPAR2 = best_par2 - prev_written  (negative = improvement)
                        _delta_par2 = best_par2 - prev_written
                        _sa_objective = _delta_par2 / (0.25 * baseline_par2) if baseline_par2 > 0 else _delta_par2
                        _T = _sa_accept_temp if _sa_accept_temp is not None else 1.0
                        if _sa_objective < 0:
                            _accept = True
                            _accept_reason = f"SA improvement (objective={_sa_objective:.4f})"
                        else:
                            _accept_prob = _math.exp(-_sa_objective / _T) if _T > 0 else 0.0
                            _accept = _random.random() < _accept_prob
                            _accept_reason = (
                                f"SA Metropolis: prob={_accept_prob:.4f} "
                                f"(objective={_sa_objective:.4f}, T={_T:.4f}), "
                                f"{'accepted' if _accept else 'rejected'}"
                            )
                    print(f"[WriteBack] iter={i} task={task_for_writeback} "
                          f"PAR-2={best_par2:.0f} (prev={prev_written:.0f}, "
                          f"baseline={baseline_par2:.0f}) → {_accept_reason}", flush=True)
                    if _accept:
                        try:
                            _writeback_template(task_for_writeback, best_code, args.project)
                            _writeback_par2_per_task[task_for_writeback] = best_par2
                            _writeback_age_per_task[task_for_writeback] = i
                            print(f"[WriteBack] iter={i} → template updated", flush=True)
                        except Exception as _wb_exc:
                            warnings.warn(
                                f"[WriteBack] iter={i}: failed to update template: {_wb_exc}",
                                category=UserWarning,
                                stacklevel=2,
                            )
            except Exception as _wb_outer:
                warnings.warn(
                    f"[WriteBack] iter={i}: unexpected error: {_wb_outer}",
                    category=UserWarning,
                    stacklevel=2,
                )

    final = {}
    for key in results["time"]:
        final[key] = {
            "time": results["time"][key],
            "PAR-2": results["PAR-2"][key],
            "prompt": results["prompt"][key],
        }
    with open(paths["temp_prompts_dir"] / 'final_result.json', 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    with open(paths["results_root"] / 'final_result.json', 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # --- Auto-generate training curve plot ---
    try:
        baseline_par2 = final.get("0", {}).get("PAR-2")
        plot_training_curve(
            results_root=paths["results_root"],
            baseline_par2=baseline_par2,
            run_id=run_id,
        )
    except Exception as _plot_exc:
        warnings.warn(f"[Plotting] Failed to generate training curve: {_plot_exc}", category=UserWarning, stacklevel=2)

    if not bool(getattr(args, "run_eval", True)):
        print("skip evaluation for this step (run_eval=False)", flush=True)
        return
    _run_eval_stage(args, final, {str(k): v for k, v in extra_params.items()}, fallback_task=args.task, paths=paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./examples/EasySAT/config.yaml', help='Path to the config file')

    parser.add_argument('--iteration_num', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--data_parallel_size', type=int, default=3)
    parser.add_argument('--devoid_duplication', type=bool, default=False)
    parser.add_argument('--llm_model',
                        type=str,
                        default="gpt-4-1106-preview")
    parser.add_argument('--timeout', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default="data_test")
    parser.add_argument('--temperature', type=float, default=1.0, help='LLM sampling temperature (0.0-2.0)')
    parser.add_argument('--project', type=str, default="EasySAT/")
    parser.add_argument('--task',
                        type=str,
                        default="bump_var_function")
    parser.add_argument('--optimize_tasks', nargs='*', default=None)
    parser.add_argument('--task_selection_mode', type=str, default='random_one')

    parser.add_argument('--original', type=bool, default=False)

    parser.add_argument('--api_base', type=str, default='')
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--resume_from_checkpoint', type=bool, default=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./results/checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--run_id', type=str, default='')
    parser.add_argument('--run_eval', type=bool, default=True)
    parser.add_argument('--eval_baseline', type=bool, default=True)
    parser.add_argument('--eval_only_from_run', type=bool, default=False)
    parser.add_argument('--start_iter_override', type=int, default=None)
    parser.add_argument('--end_iter_override', type=int, default=None)
    parser.add_argument(
        '--template_update_strategy',
        type=str,
        default='none',
        help=(
            'Controls whether/how EasySAT.cpp templates are updated during training. '
            '"none" = no write-back (default); '
            '"greedy_train" = write back only when PAR-2 strictly improves; '
            '"annealing" = SA Metropolis write-back with log-annealed temperature '
            '(T: 1.0→0.1), objective = δPAR2/(0.25*PAR2_baseline).'
        ),
    )

    args = parser.parse_args()

    loaded_config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as file:
            loaded_config = yaml.safe_load(file) or {}
            for key, value in loaded_config.items():
                setattr(args, key, value)

    args = _apply_env_overrides(args)
    args._loaded_config_payload = _sanitize_config_for_persistence(loaded_config)

    def _signal_handler(signum, frame):
        _graceful_shutdown(f"Signal received: {signum}", exit_code=128 + int(signum))

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        main(args)
    except KeyboardInterrupt:
        _graceful_shutdown("KeyboardInterrupt", exit_code=130)
        raise
    finally:
        _graceful_shutdown("Process exit")
