import time 
import os 
import psutil
import matplotlib.pyplot as plt
import torch
from .utils import synchronize

class TimeProfiler:
    """Time profiler for measuring execution time of code blocks.

    Parameters
    ----------
    only_cpu : bool, optional
        If True, skip CUDA synchronization. Default False.

    Attributes
    ----------
    time : float
        Total execution time in seconds
    start : float 
        Start time of profiling
    scopes : dict
        Dictionary of scope names to (start, end) times
    only_cpu : bool
        Whether to skip CUDA synchronization

    Examples
    --------
    .. code-block:: python

        # Basic usage
        with TimeProfiler() as prof:
            # Code to profile
            time.sleep(1)
        print(f"Execution time: {prof.time} seconds")

        # Using scopes
        with TimeProfiler() as prof:
            with prof.scope("scope1"):
                time.sleep(1)
            with prof.scope("scope2"):
                time.sleep(0.5)
        prof.plot("timing.png")

    # CPU-only profiling
    .. code-block:: python

        with TimeProfiler(only_cpu=True) as prof:
            # Code to profile without CUDA sync
            pass

    Notes
    -----
    The profiler will automatically synchronize CUDA operations unless only_cpu=True.
    Scopes can be used to measure timing of nested code blocks.
    The plot() method visualizes timing results for all scopes.
    """
    def __init__(self,  only_cpu=False):
        self.time = None
        self.start = None 
        self.scopes = {}
        self.only_cpu = only_cpu

    def __enter__(self):
        # synchronize()
        if not self.only_cpu:
            torch.cuda.synchronize()
        self.start = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # synchronize()
        if  not self.only_cpu:
            torch.cuda.synchronize()
        if exc_type is KeyboardInterrupt:
            return True 
        elif exc_type is not None:
            return False 
        self.time = time.perf_counter() - self.start
        return True

    def scope(self, name):
        assert name not in self.scopes, f"Scope {name} already exists"
        self.scopes[name] = None
        return TimeProfilerScope(self, name, self.only_cpu)
    
    def plot(self, save_path="time.png"):
        assert len(self.scopes) > 0, "No scopes to plot"
        fig, ax = plt.subplots(figsize=(8,6))
        ax.broken_barh([(0, self.time)], (0, len(self.scopes)), facecolors='blue', alpha=0.2)
        for i,(scope, (start, end)) in enumerate(self.scopes.items()):
            ax.broken_barh([(start-self.start, end-start)], (i,1), facecolors='orange')
            ax.text((start-self.start+end-self.start)/2, i+0.5, scope, ha="center", va="center", color="black", fontsize=12)
        ax.set_xlabel("Time in seconds")
        ax.set_ylabel("Scope")
        ax.set_yticks([])
        fig.savefig(save_path)


class TimeProfilerScope:
    """A context manager for profiling execution time within a specific scope.

    This class is used to track timing during a specific section of code.
    It works in conjunction with TimeProfiler to record start and end times
    of the scope.

    Example usage:
        with TimeProfiler() as profiler:
            # Create a named scope to track timing
            with profiler.scope("data loading"):
                # Code to profile...
                data = load_data()
            
            # Create another scope
            with profiler.scope("training"):
                # More code to profile...
                model.train()

            # Plot the timing results
            profiler.plot("timing_profile.png")

    The resulting plot will show timing bars for each scope with their names.
    """
    def  __init__(self, profiler, name, only_cpu=False):
        self.profiler = profiler
        self.name = name
        self.only_cpu = only_cpu
    def __enter__(self):
        # synchronize()
        if not self.only_cpu:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is KeyboardInterrupt:
            return True 
        elif exc_type is not None:
            return False 
        
        # synchronize()
        if not self.only_cpu:
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
    
        self.profiler.scopes[self.name] = (self.start_time, self.end_time)
        return True
