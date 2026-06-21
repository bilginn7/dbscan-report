import time


def measure_total_time(
    function,
    *args,
    **kwargs
):

    start = time.perf_counter()

    labels, stats = function(
        *args,
        **kwargs
    )

    total_time = (
        time.perf_counter()
        - start
    )

    stats["total_time"] = total_time

    return labels, stats