import multiprocessing
import time 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pynvml

from .utils import synchronize
matplotlib.use('Agg')

def get_memory_for_index(index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used // (1024 * 1024)  # Convert to MB

def get_max_memory_for_index(index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.total // (1024 * 1024)  # Convert to MB

def monitor_gpu_memory(index, stop_event, conn):
    mems = []
    times = []
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    start_time = time.perf_counter()

    while not stop_event.is_set():
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mems.append(mem_info.used // (1024 * 1024))  # Convert to MB
        times.append(time.perf_counter() - start_time)

    conn.send(tuple(mems))
    conn.send(tuple(times))
    conn.close()

class CUDAProfiler:
    """CUDA memory profiler for tracking GPU memory usage.

    This class provides functionality to monitor and profile GPU memory usage over time.
    It uses NVML (NVIDIA Management Library) to collect memory statistics.

    Example usage:

    ```python
    # Profile a code block
    with CUDAProfiler(index=0) as profiler:
        # Run GPU code here
        model.train()
        
    # Get peak memory usage
    peak_mem = profiler.max()
    print(f"Peak GPU memory: {peak_mem} MB")

    # Plot memory over time
    profiler.plot("memory_profile.png")
    ```

    The profiler will track memory usage from when the context is entered until it exits.
    Memory statistics can be accessed through the results and times attributes after profiling.
    """
    def __init__(self, index=0):
        """
        Parameters
        ----------
        index : int, optional
            GPU index, by default 0
        """
        self.index = index
        self.scopes = {}
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            assert index < device_count, f"Cannot find GPU with index {self.index}"
        except pynvml.NVMLError as e:
            raise Exception(f"CUDAProfiler: Failed to initialize NVML: {str(e)}") from e

    def __enter__(self):
        self.stop_event = multiprocessing.Event()
        parent_conn, child_conn = multiprocessing.Pipe()
        self.conn = parent_conn
        self.monitoring_process = multiprocessing.Process(
            target=monitor_gpu_memory, 
            args=(self.index, self.stop_event, child_conn))
        synchronize(self.index)
        self.start_mem = get_memory_for_index(self.index)
        self.start_time = time.perf_counter()
        self.monitoring_process.start()

        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        synchronize(self.index)
        if exc_type is KeyboardInterrupt:
            self.stop_event.set()
            self.monitoring_process.join()
            return True 
        elif exc_type is not None:
            return False 
        
        self.stop_event.set()
        results = self.conn.recv()
        times = self.conn.recv()
        self.monitoring_process.join()

        self.results = results
        self.times = times
        
        return True

    def max(self):
        return max(self.results)
    
    def min(self):
        return min(self.results)

    def mean(self):
        return np.mean(self.results)
    
    def std(self):
        return np.std(self.results)
    
    def __array__(self):
        return np.array(self.results)

    def plot(self, save_path="cuda_mem.png"):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(self.times, self.results)
        ax.set_xlabel("Time in seconds")
        ax.set_ylabel(f"Used GPU Memory (MB) for CUDA device {self.index}")

        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
        max_mem = self.max()
        min_mem = self.min()
        one_third = (max_mem - min_mem) / 3 + min_mem
        two_third = (max_mem - min_mem) / 3 * 2 + min_mem
        for i,(name, (start, end)) in enumerate(self.scopes.items()):
            is_odd = i % 2 == 1
            ax.axvspan(start-self.start_time, end-self.start_time, alpha=0.3, color=colors[i], label=name)
            ax.text((start-self.start_time+end-self.start_time)/2, one_third if is_odd else two_third, name, ha="center", va="center", color="black", fontsize=12)

        fig.savefig(save_path)
        plt.close()

    def scope(self, name):
        assert name not in self.scopes, f"Scope {name} already exists"
        self.scopes[name] = None
        return CUDAProfilerScope(self, name)

class CUDAProfilerScope:
    """A context manager for profiling CUDA memory usage within a specific scope.

    This class is used to track CUDA memory usage during a specific section of code.
    It works in conjunction with CUDAProfiler to record memory usage between the
    start and end times of the scope.

    Example usage:
        with CUDAProfiler(index=0) as profiler:
            # Create a named scope to track memory
            with profiler.scope("matrix multiply"):
                # Code to profile...
                result = torch.mm(a, b)
            
            # Create another scope
            with profiler.scope("optimization"):
                # More code to profile...
                optimizer.step()

            # Plot the memory usage over time
            profiler.plot("memory_profile.png")

    The resulting plot will show memory usage over time with colored regions
    indicating the different scopes and their names.
    """
    def __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        synchronize(self.profiler.index)
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is KeyboardInterrupt:
            return True 
        elif exc_type is not None:
            return False 
        
        synchronize(self.profiler.index)
        self.end_time = time.perf_counter()
    
        self.profiler.scopes[self.name] = (self.start_time, self.end_time)
        return True