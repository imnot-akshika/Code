import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def validate_args(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, (int, float)) and arg <= 0:
                raise ValueError("All arguments must be positive numbers")
        return func(*args, **kwargs)
    return wrapper


def cache_result(func):
    cache = {}
    @wraps(func)
    def wrapper(*args, **kwargs):
        if args in cache:
            return cache[args]
        result = func(*args, **kwargs)
        cache[args] = result
        return result
    return wrapper


# 1. timer — prints how long a function takes
@timer
def slow_function():
    import time
    time.sleep(0.1)
    return "done"

# 2. validate_args — checks all arguments are positive numbers
#    raises ValueError if any arg <= 0
@validate_args
def calculate_area(width, height):
    return width * height

calculate_area(5, 3)      # 15
calculate_area(-1, 3)     # ValueError: All arguments must be positive

# 3. cache_result — simple manual cache (dict-based, no lru_cache)
@cache_result
def expensive(n):
    print(f"computing {n}...")
    return n ** 2

expensive(5)    # computing 5... → 25
expensive(5)    # 25 (no print — cached)
expensive(3)    # computing 3... → 9