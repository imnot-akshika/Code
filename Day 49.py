import timeit
import cProfile
import pstats
import tracemalloc
import io
from typing import Callable
from rich.console import Console
from rich.table import Table

class Benchmarker:
    def __init__(self):
        self.results: dict[str, dict] = {}

    def time_it(self, name: str, func: Callable, *args, runs: int = 1000) -> float:
       if name not in self.results:
           self.results[name] = {}
       run_time = timeit.timeit(lambda: func(*args), number=runs)
       avg_time = run_time / runs
       self.results[name]["time"] = avg_time
       return avg_time

    def memory_it(self, name: str, func: Callable, *args) -> int:
       if name not in self.results:
           self.results[name] = {}
       tracemalloc.start()
       func(*args)
       _, peak = tracemalloc.get_traced_memory()
       tracemalloc.stop()
       self.results[name]["memory_bytes"] = peak
       return peak

    def compare(self, baseline: str, challenger: str) -> str:
        # compare the results of two functions and return a summary
        time_diff = self.results[challenger]["time"] - self.results[baseline]["time"]
        mem_diff = self.results[challenger].get("memory_bytes", 0) - self.results[baseline].get("memory_bytes", 0)
        return f"{challenger} is {'faster' if time_diff < 0 else 'slower'} than {baseline} by {abs(time_diff):.6f} seconds and uses {'less' if mem_diff < 0 else 'more'} memory by {abs(mem_diff)} bytes."

    def report(self) -> None:
        console = Console()
        table = Table(title="Benchmark Results")
        table.add_column("Name", style="cyan")
        table.add_column("Time (s)", justify="right", style="green")
        table.add_column("Memory (bytes)", justify="right", style="blue")

        for name, metrics in self.results.items():
            time_str = f"{metrics['time']:.6f}" if "time" in metrics else "N/A"
            mem_str = str(metrics.get("memory_bytes", "N/A"))
            table.add_row(name, time_str, mem_str)
        console.print(table)


#expamle usage
b = Benchmarker()

# 1. string concatenation vs join
def concat(n):
    s = ""
    for i in range(n):
        s += str(i)
    return s

def join_str(n):
    return "".join(str(i) for i in range(n))

b.time_it("concat_1000", concat, 1000, runs=500)
b.time_it("join_1000", join_str, 1000, runs=500)
b.memory_it("concat_1000", concat, 1000)
b.memory_it("join_1000", join_str, 1000)

# 2. list vs set membership
big_list = list(range(100000))
big_set = set(range(100000))

b.time_it("list_search", lambda: 99999 in big_list, runs=10000)
b.time_it("set_search", lambda: 99999 in big_set, runs=10000)

# 3. list comprehension vs loop append
def loop_append(n):
    result = []
    for i in range(n):
        result.append(i**2)
    return result

def list_comp(n):
    return [i**2 for i in range(n)]

b.time_it("loop_append", loop_append, 10000, runs=1000)
b.time_it("list_comp", list_comp, 10000, runs=1000)

b.report()
print(b.compare("concat_1000", "join_1000"))
print(b.compare("list_search", "set_search"))
print(b.compare("loop_append", "list_comp"))
