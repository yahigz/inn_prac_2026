import os
import argparse
import json
import time
import yaml
import ray

from jinja2 import FileSystemLoader, Environment

from autosat.utils import get_code, revise_file, clean_files, collect_results, \
                            copy_folder, fill_core_codes, delete_InfiniteLoopInst, get_batch_id, train_init, check_reIteration
from autosat.llm_api.base_api import get_llm_api
from autosat.execution.execution_worker import ExecutionWorker
from autosat.evaluation.evaluate import evaluate
import warnings


def _extract_reference_code(prompt_file_dir):
    with open(prompt_file_dir, 'r', encoding='utf-8') as file:
        prompt_text = file.read()

    start = prompt_text.rfind('// start')
    if start == -1:
        return ''
    end = prompt_text.find('// end', start)
    if end == -1:
        return ''

    reference_code = prompt_text[start + len('// start'):end]
    return reference_code.strip('\n')


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
        revise_file(file_name=os.path.join("./examples/", project_dir, "EasySAT.cpp"),
                    save_dir='./temp/EasySAT_{}/EasySAT.cpp'.format(format((count - 1) % arguments.batch_size)),
                    replace_code=answer_code,
                    timeout=arguments.timeout,
                    data_dir="\"{}\"".format(arguments.data_dir),
                    *args, **kwargs
                    )
        success = execution_worker.execute(count, arguments.batch_size, arguments.data_parallel_size)
        return count, success, answer_code


def main(args):
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
    results = {
        "time": {},
        "prompt": {},
        "PAR-2": {}
    }

    project_dir = os.path.join(args.project, args.task)
    print("project_dir: {}".format(project_dir))

    train_init(args)

    # replace file
    env = Environment(loader=FileSystemLoader(os.path.join("./examples/", project_dir)))

    template = env.get_template("EasySAT_original.cpp") # we set `EasySAT` as baseline here, you can replace with yours
    output = template.render(timeout=args.timeout, data_dir="\"{}\"".format(args.data_dir))
    with open("./temp/EasySAT_{}/EasySAT.cpp".format(count), 'w') as f:
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
            all_exist = all(os.path.exists(os.path.join('./temp/results/', 'finished'+filename)) for filename in filenames)
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
    print("EasySAT(baseline) result-- time: {} seconds ; PAR-2: {}".format(results["time"]["0"], results["PAR-2"]["0"]))

    for i in range(args.iteration_num):
        # clean temp results
        clean_files(folder_path="./temp/results/", mode="all")
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
            print("iteration: ", i, "\n results_prompt: \n", result_prompt)

            # Add the result matrix into next round prompt.
            revise_file(file_name= os.path.join("./examples/", project_dir, "feedback_prompt.txt"),
                        save_dir='./temp/prompts/feedback_prompt.txt',
                        replace_code=result_prompt,
                        original_time=int(results["time"]["0"]),
                        best_code=list(best_result.values())[0][1]
                        )
            prompt_file_dir = './temp/prompts/feedback_prompt.txt'

        start_time = time.time()
        answer_code_cur_round = {}
        lbd_queue_size_cur_round = {}
        tasks = [synchronized_asked.remote(prompt_file_dir, i * args.batch_size + batch_id + 1, args)
                 for batch_id in range(args.batch_size)]

        for future in ray.get(tasks):
            count, answer_code, lbd_queue_size = future
            batch_id = get_batch_id(count, args.batch_size)
            answer_code_cur_round[batch_id] = answer_code
            lbd_queue_size_cur_round[batch_id] = lbd_queue_size if lbd_queue_size.isdigit() else '50' # original lbd_queue_size = 50
        end_time = time.time()
        print("querying consuming: {} seconds".format(end_time-start_time))

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
        for future in ray.get(tasks):
            count, success, answer_code = future
            answers[count] = answer_code

            extra_params[count] = {'lbd_queue_size': lbd_queue_size_cur_round[get_batch_id(count, args.batch_size)]}

            if success:
                id_list.append(count)
            elif args.devoid_duplication and success == 0:
                repetition_dict[count] = answer_code
        end_time = time.time()
        print("sending execution time consuming: {} seconds.".format(end_time-start_time))
        start_time = time.time()
        filenames = [str(global_id) + "_" + str(num) + ".txt" for global_id in id_list for num in range(args.data_parallel_size)]
        print("filenames: ", filenames)
        while True:
            end_time = time.time()
            if end_time-start_time > args.timeout * (2*data_num/args.data_parallel_size):
                # raise ValueError("Infinite loop error!!!")
                warnings.warn(f": Infinite loop for some Solver Programs... please check later",
                              category=UserWarning, stacklevel=2)
                result, best_result = collect_results(answers=answers,
                                                      repetition_dict=repetition_dict,
                                                      results=results,
                                                      args=args)
                delete_InfiniteLoopInst(candidates=['finished'+fname for fname in filenames], result_dict=result)
                break
            all_exist = all(os.path.exists(os.path.join('./temp/results/', 'finished'+filename)) for filename in filenames)
            if all_exist:
                result, best_result = collect_results(answers=answers,
                                                      repetition_dict=repetition_dict,
                                                      results=results,
                                                      args=args)
                break

        print("collecting execution time consuming: {} seconds.".format(end_time-start_time))
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
        with open('./temp/prompts/iter_{}_result.json'.format(i), 'w') as f:
            json.dump(result, f)

    final = {}
    for key in results["time"]:
        final[key] = {
            "time": results["time"][key],
            "PAR-2": results["PAR-2"][key],
            "prompt": results["prompt"][key],
        }
    with open('./temp/prompts/final_result.json', 'w') as f:
        json.dump(final, f)

    ray.shutdown()

    # Add evaluation 3.1
    print('start evaluation ...')
    if os.path.exists("./temp/results/"):
        clean_files(folder_path="./temp/results/", mode="all")
    baseline = results["PAR-2"]["0"]
    record_info = []
    print('EasySAT baseline : {}'.format(baseline))
    for global_id_str in final.keys():
        global_id = int(global_id_str)
        if global_id_str != "0":
            if final[global_id_str]["PAR-2"] < baseline:
                record_info.append((global_id, final[global_id_str]["PAR-2"], final[global_id_str]["prompt"], extra_params[global_id]))
    record_info.sort(key=lambda x: x[1])

    print("{} Files to evaluate...".format(len(record_info)))
    if len(record_info) == 0:
        return
    for global_id, par_2, answer_code, params_dict in record_info:
        method_name = f"{args.task}_{args.llm_model}_{global_id}".replace('/', '')  # algorithm identity
        SAT_folder = f'./temp/EasySAT_{method_name}/'
        copy_folder(src_folder="./temp/EasySAT/", num=1, mode='eval', target_folder=SAT_folder)
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

    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
            for key, value in config.items():
                setattr(args, key, value)

    args = _apply_env_overrides(args)

    main(args)

