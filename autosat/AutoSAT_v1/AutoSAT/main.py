import os
import argparse
import json
import time
import uuid
import yaml
import ray
import re
import sys
import glob
from pathlib import Path

from jinja2 import FileSystemLoader, Environment

from autosat.utils import get_code, revise_file, clean_files, collect_results, \
                            copy_folder, fill_core_codes, delete_InfiniteLoopInst, get_batch_id, train_init, check_reIteration
from autosat.llm_api.base_api import get_llm_api
from autosat.execution.execution_worker import ExecutionWorker
from autosat.evaluation.evaluate import evaluate
import warnings


def _enable_realtime_output():
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass


def _make_run_id(explicit_run_id=""):
    explicit_run_id = str(explicit_run_id or "").strip()
    if explicit_run_id:
        return explicit_run_id
    return time.strftime("run_%Y%m%d_%H%M%S") + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _run_paths(run_id):
    temp_root = Path("./temp/runs") / run_id
    results_root = Path("./results/runs") / run_id
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


@ray.remote
def synchronized_asked(prompt_file_dir, count, args):
    llm_api = get_llm_api(args)

    answer = llm_api.call_api(prompt_file=prompt_file_dir)

    answer_code = get_code(answer, seperator=['// start\n', '\n// end'])
    lbd_queue_size = get_code(answer, seperator=['// start lbd_queue_size\n', '\n// end lbd_queue_size'])

    if len(answer_code.strip()) < 20:
        reference_code = _extract_reference_code(prompt_file_dir)
        if len(reference_code.strip()) >= 20:
            answer_code = reference_code

    return count, answer_code, lbd_queue_size.strip()


@ray.remote
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
    run_id = _make_run_id(getattr(args, "run_id", ""))
    os.environ["AUTOSAT_RUN_ID"] = run_id
    paths = _run_paths(run_id)
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
                paths = _run_paths(run_id)
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

    if (not resumed) or ("0" not in results.get("time", {})):
        train_init(args)

        env = Environment(loader=FileSystemLoader(os.path.join("./examples/", project_dir)))
        template = env.get_template("EasySAT_original.cpp") # we set `EasySAT` as baseline here, you can replace with yours
        output = template.render(timeout=args.timeout, data_dir="\"{}\"".format(args.data_dir))
        baseline_cpp = os.path.join(args.temp_root, f"EasySAT_{count}", "EasySAT.cpp")
        _ensure_parent_dir(baseline_cpp)
        with open(baseline_cpp, 'w') as f:
            f.write(output)

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
                    break
        else:
            results["time"]["0"] = args.original_result['time']
            results["prompt"]["0"] = " "
            results["PAR-2"]["0"] = args.original_result['PAR-2']
    else:
        if not os.path.exists(os.path.join(args.temp_root, "EasySAT_0")):
            train_init(args)
    print("EasySAT(baseline) result-- time: {} seconds ; PAR-2: {}".format(results["time"]["0"], results["PAR-2"]["0"]), flush=True)

    for i in range(start_iter, args.iteration_num):
        # clean temp results
        clean_files(folder_path=args.temp_results_dir, mode="all")
        id_list = []

        if i == 0:
            prompt_file_dir = os.path.join("./examples/", project_dir, "original_prompt.txt")
        elif check_reIteration(round=i,best_result_dict=best_result,
                               baseline={'time': results["time"]["0"],'PAR-2': results["PAR-2"]["0"]}):
            # restart at iteration-1 if necessary..
            prompt_file_dir = os.path.join("./examples/", project_dir, "original_prompt.txt")
        else:
            result_prompt = [
                f"Experiment {num}, Your provided {args.project}--{args.task}: \n'''{result['prompt'][value[0]]}''', \n execution time is {result['time'][value[0]]} seconds. PAR-2 ( Penalized Average Runtime with factor 2) is {result['PAR-2'][value[0]]} seconds. "
                for num, value in enumerate(result["time"].items())]
            result_prompt = '\n '.join(result_prompt)
            print("iteration: ", i, "\n results_prompt: \n", result_prompt, flush=True)

            # Add the result matrix into next round prompt.
            revise_file(file_name= os.path.join("./examples/", project_dir, "feedback_prompt.txt"),
                        save_dir=os.path.join(args.temp_prompts_dir, 'feedback_prompt.txt'),
                        replace_code=result_prompt,
                        original_time=int(results["time"]["0"]),
                        best_code=list(best_result.values())[0][1]
                        )
            prompt_file_dir = os.path.join(args.temp_prompts_dir, 'feedback_prompt.txt')

        start_time = time.time()
        answer_code_cur_round = {}
        lbd_queue_size_cur_round = {}
        tasks = [synchronized_asked.remote(prompt_file_dir, i * args.batch_size + batch_id + 1, args)
                 for batch_id in range(args.batch_size)]
        print(f"[Iteration {i}] Waiting for LLM responses from {len(tasks)} tasks...", flush=True)
        try:
            futures = ray.get(tasks, timeout=args.timeout * 10)
        except Exception as e:
            print(f"[ERROR] Ray task timeout or failure: {e}", flush=True)
            raise
        for future in futures:
            count, answer_code, lbd_queue_size = future
            batch_id = get_batch_id(count, args.batch_size)
            answer_code_cur_round[batch_id] = answer_code
            lbd_queue_size_cur_round[batch_id] = lbd_queue_size if lbd_queue_size.isdigit() else '50' # original lbd_queue_size = 50
        end_time = time.time()
        print("querying consuming: {} seconds".format(end_time-start_time), flush=True)

        start_time = time.time()
        if args.task == "restart_condition":
            tasks = [synchronized_executed.remote(
                     count=i * args.batch_size + batch_id + 1,
                     results=results, arguments=args,
                     answer_code=answer_code_cur_round[batch_id],
                     lbd_queue_size=lbd_queue_size_cur_round[batch_id]) for batch_id in range(args.batch_size)]

        else:
            tasks = [synchronized_executed.remote(
                     count=i * args.batch_size + batch_id + 1,
                     results=results, arguments=args,
                     answer_code=answer_code_cur_round[batch_id]) for batch_id in range(args.batch_size)]

        repetition_dict = {}
        print(f"[Iteration {i}] Waiting for execution results from {len(tasks)} tasks...", flush=True)
        try:
            execute_futures = ray.get(tasks, timeout=args.timeout * 10)
        except Exception as e:
            print(f"[ERROR] Execution task timeout or failure: {e}", flush=True)
            raise
        for future in execute_futures:
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

    ray.shutdown()

    # Add evaluation 3.1
    print('start evaluation ...', flush=True)
    if os.path.exists(args.temp_results_dir):
        clean_files(folder_path=args.temp_results_dir, mode="all")
    baseline = results["PAR-2"]["0"]
    record_info = []
    print('EasySAT baseline : {}'.format(baseline), flush=True)

    baseline_solver_path = str(getattr(args, "SAT_solver_file_path", "") or "").strip()
    if not baseline_solver_path:
        baseline_solver_path = './examples/EasySAT/original_EasySAT/EasySAT.cpp'
    baseline_method_name = f"baseline_{args.task}_{args.llm_model}".replace('/', '')
    existing_baseline_eval = glob.glob(os.path.join(args.results_save_path, f"results_{baseline_method_name}_*.txt"))
    if existing_baseline_eval:
        print(f"[Eval] Skip baseline: already evaluated ({len(existing_baseline_eval)} file(s)).", flush=True)
    else:
        print(f"[Eval] Run baseline: {baseline_method_name}", flush=True)
        evaluate(args, method_name=baseline_method_name, SAT_solver_file_path=baseline_solver_path)

    for global_id_str in final.keys():
        global_id = int(global_id_str)
        if global_id_str != "0":
            if final[global_id_str]["PAR-2"] < baseline:
                record_info.append((global_id, final[global_id_str]["PAR-2"], final[global_id_str]["prompt"], extra_params[global_id]))
    record_info.sort(key=lambda x: x[1])

    print("{} Files to evaluate...".format(len(record_info)), flush=True)
    if len(record_info) == 0:
        return
    for idx, (global_id, par_2, answer_code, params_dict) in enumerate(record_info, start=1):
        method_name = f"{args.task}_{args.llm_model}_{global_id}".replace('/', '')  # algorithm identity
        existing_eval = glob.glob(os.path.join(args.results_save_path, f"results_{method_name}_*.txt"))
        if existing_eval:
            print(f"[Eval] Skip {idx}/{len(record_info)} {method_name}: already evaluated ({len(existing_eval)} file(s)).", flush=True)
            continue
        print(f"[Eval] Run {idx}/{len(record_info)}: {method_name}", flush=True)
        SAT_folder = os.path.join(args.temp_root, f'EasySAT_{method_name}')
        copy_folder(src_folder=args.temp_easy_root, num=1, mode='eval', target_folder=SAT_folder)
        SAT_solver_file_path = os.path.join(SAT_folder, 'EasySAT_modified.cpp')

        # replace the answer code for the specific function and other auxiliary params such as `lbd size`
        fill_core_codes(origin_file=os.path.join("./examples/", project_dir, "EasySAT.cpp"),
                        target_file=SAT_solver_file_path,
                        answer_code=answer_code,
                        **params_dict)
        evaluate(args, method_name=method_name, SAT_solver_file_path=SAT_solver_file_path)  # TODO check .. somtime can not delete dir..


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
    parser.add_argument('--project', type=str, default="EasySAT/")
    parser.add_argument('--task',
                        type=str,
                        default="bump_var_function",
                        choices=["restart_condition", "restart_function", "bump_var_function", "rephase_function"])

    parser.add_argument('--original', type=bool, default=False)

    parser.add_argument('--api_base', type=str, default='')
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--resume_from_checkpoint', type=bool, default=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./results/checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--run_id', type=str, default='')

    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
            for key, value in config.items():
                setattr(args, key, value)

    args = _apply_env_overrides(args)

    main(args)

