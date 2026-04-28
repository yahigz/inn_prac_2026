import os
import subprocess
import platform
import signal


def _temp_root():
    run_id = (os.getenv("AUTOSAT_RUN_ID") or "").strip()
    if run_id:
        root = os.path.join(".", "temp", "runs", run_id)
        task_namespace = (os.getenv("AUTOSAT_TASK_NAMESPACE") or "").strip().strip('/')
        if task_namespace:
            root = os.path.join(root, task_namespace)
        return root
    return "./temp"


class ExecutionWorker():
    _active_processes = []

    def __init__(self):
        pass

    @classmethod
    def _cleanup_finished(cls):
        cls._active_processes = [proc for proc in cls._active_processes if proc.poll() is None]

    @classmethod
    def _register_process(cls, proc):
        cls._cleanup_finished()
        cls._active_processes.append(proc)

    @classmethod
    def shutdown_all(cls, timeout=5):
        cls._cleanup_finished()
        if platform.system() in ('Linux', 'Darwin'):
            for proc in cls._active_processes:
                if proc.poll() is not None:
                    continue
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    pass
            for proc in cls._active_processes:
                if proc.poll() is not None:
                    continue
                try:
                    proc.wait(timeout=timeout)
                except Exception:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
            try:
                subprocess.run(['pkill', '-f', 'EasySAT'], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            except Exception:
                pass
        elif platform.system() == 'Windows':
            for proc in cls._active_processes:
                if proc.poll() is not None:
                    continue
                try:
                    proc.terminate()
                except Exception:
                    pass
            for proc in cls._active_processes:
                if proc.poll() is not None:
                    continue
                try:
                    proc.wait(timeout=timeout)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        cls._cleanup_finished()

    @staticmethod
    def _compile_cpp(source_cpp_path, output_path):
        cmd = ["g++", "-O3", "-Wall", "-std=c++17", source_cpp_path, "-o", output_path]
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0

    @staticmethod
    def _spawn_solver(cmd):
        if platform.system() in ('Linux', 'Darwin'):
            return subprocess.Popen(cmd, preexec_fn=os.setsid)
        return subprocess.Popen(cmd)

    def execute(self, id, batch_size, data_parallel_size):
        temp_root = _temp_root()
        if platform.system() == 'Windows':
            source_cpp = "{root}/EasySAT_{idx}/EasySAT.cpp".format(root=temp_root, idx=(id-1) % batch_size)
            executable = "{root}/EasySAT_{idx}/EasySAT".format(root=temp_root, idx=(id-1) % batch_size)
            if not self._compile_cpp(source_cpp, executable):
                return False

            for i in range(data_parallel_size):
                cmd = [
                    "{root}/EasySAT_{idx}/EasySAT.exe".format(root=temp_root, idx=(id-1) % batch_size),
                    str(id), str(data_parallel_size), str(i),
                ]
                proc = self._spawn_solver(cmd)
                self._register_process(proc)
            return True

        elif platform.system() in ('Linux', 'Darwin'):
            source_cpp = "{root}/EasySAT_{idx}/EasySAT.cpp".format(root=temp_root, idx=(id - 1) % batch_size)
            executable = "{root}/EasySAT_{idx}/EasySAT".format(root=temp_root, idx=(id - 1) % batch_size)
            if not self._compile_cpp(source_cpp, executable):
                return False

            for i in range(data_parallel_size):
                cmd = [
                    executable,
                    str(id), str(data_parallel_size), str(i),
                ]
                proc = self._spawn_solver(cmd)
                self._register_process(proc)
            return True

        else:
            raise ValueError("Unsupported this kind of system!")

    def execute_original(self, id, data_parallel_size):
        return self.execute(id=id, batch_size=1, data_parallel_size=data_parallel_size)

    def execute_eval(self,source_cpp_path, executable_file_path, data_parallel_size):
        id = 1 # only to occupy the position for parameters in EasySAT.cpp
        if platform.system() == 'Windows':
            if not self._compile_cpp(source_cpp_path, executable_file_path):
                return False

            for i in range(data_parallel_size):
                cmd = [
                    executable_file_path + '.exe',
                    str(id), str(data_parallel_size), str(i),
                ]
                proc = self._spawn_solver(cmd)
                self._register_process(proc)
            return True

        elif platform.system() in ('Linux', 'Darwin'):
            if not self._compile_cpp(source_cpp_path, executable_file_path):
                return False

            for i in range(data_parallel_size):
                cmd = [
                    executable_file_path,
                    str(id), str(data_parallel_size), str(i),
                ]
                proc = self._spawn_solver(cmd)
                self._register_process(proc)
            return True

        else:
            raise ValueError("Unsupported this kind of system!")
