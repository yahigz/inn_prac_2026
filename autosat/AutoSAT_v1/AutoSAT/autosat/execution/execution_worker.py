import os
import subprocess
import platform


def _temp_root():
    run_id = (os.getenv("AUTOSAT_RUN_ID") or "").strip()
    if run_id:
        return os.path.join(".", "temp", "runs", run_id)
    return "./temp"


class ExecutionWorker():
    def __init__(self):
        pass

    def execute(self, id, batch_size, data_parallel_size):
        temp_root = _temp_root()
        if platform.system() == 'Windows':
            exec_code = os.system(
                "g++ -O3 -Wall -std=c++17 {root}/EasySAT_{idx}/EasySAT.cpp -o {root}/EasySAT_{idx}/EasySAT".format(root=temp_root, idx=(id-1) % batch_size))
            if exec_code != 0:
                return False

            for i in range(data_parallel_size):
                # argv id ���ĸ��ļ��� ���ݲ��е�index ���ݲ��д�С
                exec_code = os.system("start {root}/EasySAT_{idx}/EasySAT.exe {id} {parallel} {slot}".format(root=temp_root, idx=(id-1) % batch_size, id=id, parallel=data_parallel_size, slot=i))
                # subprocess.call("start ./temp/EasySAT_{}/EasySAT.exe {} ./temp/EasySAT_{}/".format((id-1) % batch_size, id, (id-1) % batch_size))
                if exec_code != 0:
                    return False
            return True

        elif platform.system() in ('Linux', 'Darwin'):
            exec_code = os.system(
                "g++ -O3 -Wall -std=c++17 {root}/EasySAT_{idx}/EasySAT.cpp -o {root}/EasySAT_{idx}/EasySAT".format(
                    root=temp_root, idx=(id - 1) % batch_size))
            if exec_code != 0:
                return False

            for i in range(data_parallel_size):
                exec_code = os.system(
                    "{root}/EasySAT_{idx}/EasySAT {id} {parallel} {slot} &".format(root=temp_root, idx=(id - 1) % batch_size, id=id, parallel=data_parallel_size, slot=i))
                # subprocess.call("start ./temp/EasySAT_{}/EasySAT.exe {} ./temp/EasySAT_{}/".format((id-1) % batch_size, id, (id-1) % batch_size))
                if exec_code != 0:
                    return False
            return True

        else:
            raise ValueError("Unsupported this kind of system!")

    def execute_original(self, id, data_parallel_size):
        return self.execute(id=id, batch_size=1, data_parallel_size=data_parallel_size)

    def execute_eval(self,source_cpp_path, executable_file_path, data_parallel_size):
        id = 1 # only to occupy the position for parameters in EasySAT.cpp
        temp_root = _temp_root()
        if platform.system() == 'Windows':
            exec_code = os.system(
                "g++ -O3 -Wall -std=c++17 {} -o {}".format( source_cpp_path, executable_file_path ) )
            if exec_code != 0:
                return False

            for i in range(data_parallel_size):
                exec_code = os.system("start {}.exe {} {} {}".format(executable_file_path, id, data_parallel_size, i))
                if exec_code != 0:
                    return False
            return True

        elif platform.system() in ('Linux', 'Darwin'):
            exec_code = os.system(
                "g++ -O3 -Wall -std=c++17 {} -o {}".format( source_cpp_path, executable_file_path ) )
            if exec_code != 0:
                return False

            for i in range(data_parallel_size):
                exec_code = os.system(
                    "{} {} {} {} &".format(executable_file_path, id, data_parallel_size, i)  )
                # subprocess.call("start ./temp/EasySAT_{}/EasySAT.exe {} ./temp/EasySAT_{}/".format((id-1) % batch_size, id, (id-1) % batch_size))
                if exec_code != 0:
                    return False
            return True

        else:
            raise ValueError("Unsupported this kind of system!")
