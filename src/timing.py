import time
from typing import Callable, Any


def measure_total_time(function: Callable, *args, **kwargs) -> tuple[Any, float]:
    start = time.perf_counter()
    result = function(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start