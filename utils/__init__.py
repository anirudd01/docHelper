# Utility modules for docHelper
import functools
import time
from typing import Any, Callable


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure execution time of functions.

    Usage:
        @timeit
        def my_function():
            pass
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrapper
