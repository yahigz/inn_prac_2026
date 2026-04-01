import os
import subprocess
import platform


class ExecutionWorker():
    _active_processes = []

    def __init__(self):
        pass

    @classmethod
    def _register_process(cls, process):
        cls._active_processes.append(process)

    @classmethod
    def _cleanup_local_processes(cls):
        still_active = []
        for process in cls._active_processes:
            if process.poll() is not None:
                continue
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    still_active.append(process)
        cls._active_processes = [p for p in still_active if p.poll() is None]

    @staticmethod
    def _cleanup_orphan_processes():
        system = platform.system()
        if system in ('Linux', 'Darwin'):
            # Kill detached solver executables started during train/eval.
            subprocess.run(["pkill", "-f", r"/temp/EasySAT_.*/EasySAT"], check=False)
            subprocess.run(["pkill", "-f", r"/temp/.*/SAT_Solver_tmp"], check=False)
        elif system == 'Windows':
            subprocess.run(["taskkill", "/IM", "EasySAT.exe", "/F"], check=False)
            subprocess.run(["taskkill", "/IM", "SAT_Solver_tmp.exe", "/F"], check=False)

    @classmethod
    def shutdown_all(cls):
        cls._cleanup_local_processes()
        cls._cleanup_orphan_processes()

    def execute(self, id, batch_size, data_parallel_size):
        if platform.system() == 'Windows':
            compile_result = subprocess.run(
                ["g++", "-O3", "-Wall", "-std=c++17",
                 "./temp/EasySAT_{}/EasySAT.cpp".format((id - 1) % batch_size),
                 "-o", "./temp/EasySAT_{}/EasySAT".format((id - 1) % batch_size)],
                check=False,
            )
            if compile_result.returncode != 0:
                return False

            for i in range(data_parallel_size):
                try:
                    proc = subprocess.Popen(
                        ["./temp/EasySAT_{}/EasySAT.exe".format((id - 1) % batch_size),
                         str(id), str(data_parallel_size), str(i)],
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                    self._register_process(proc)
                except Exception:
                    return False
            return True

        elif platform.system() in ('Linux', 'Darwin'):
            compile_result = subprocess.run(
                ["g++", "-O3", "-Wall", "-std=c++17",
                 "./temp/EasySAT_{}/EasySAT.cpp".format((id - 1) % batch_size),
                 "-o", "./temp/EasySAT_{}/EasySAT".format((id - 1) % batch_size)],
                check=False,
            )
            if compile_result.returncode != 0:
                return False

            for i in range(data_parallel_size):
                try:
                    proc = subprocess.Popen(
                        ["./temp/EasySAT_{}/EasySAT".format((id - 1) % batch_size),
                         str(id), str(data_parallel_size), str(i)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    self._register_process(proc)
                except Exception:
                    return False
            return True

        else:
            raise ValueError("Unsupported this kind of system!")

    def execute_original(self, id, data_parallel_size):
        return self.execute(id=id, batch_size=1, data_parallel_size=data_parallel_size)

    def execute_eval(self,source_cpp_path, executable_file_path, data_parallel_size):
        id = 1 # only to occupy the position for parameters in EasySAT.cpp
        if platform.system() == 'Windows':
            compile_result = subprocess.run(
                ["g++", "-O3", "-Wall", "-std=c++17", source_cpp_path, "-o", executable_file_path],
                check=False,
            )
            if compile_result.returncode != 0:
                return False

            for i in range(data_parallel_size):
                try:
                    proc = subprocess.Popen(
                        [f"{executable_file_path}.exe", str(id), str(data_parallel_size), str(i)],
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                    self._register_process(proc)
                except Exception:
                    return False
            return True

        elif platform.system() in ('Linux', 'Darwin'):
            compile_result = subprocess.run(
                ["g++", "-O3", "-Wall", "-std=c++17", source_cpp_path, "-o", executable_file_path],
                check=False,
            )
            if compile_result.returncode != 0:
                return False

            for i in range(data_parallel_size):
                try:
                    proc = subprocess.Popen(
                        [executable_file_path, str(id), str(data_parallel_size), str(i)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    self._register_process(proc)
                except Exception:
                    return False
            return True

        else:
            raise ValueError("Unsupported this kind of system!")
