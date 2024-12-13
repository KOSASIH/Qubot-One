# src/utils/performance_utils.py

import time

class PerformanceTracker:
    @staticmethod
    def time_function(func, *args, **kwargs):
        """Time a function's execution.

        Args:
            func (callable): The function to time.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            tuple: The result of the function and the time taken.
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function {func.__name__} executed in {elapsed_time:.4f} seconds.")
        return result, elapsed_time
