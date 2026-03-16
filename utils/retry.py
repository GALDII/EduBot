import sys
import os
import time
from functools import wraps
from typing import Callable, Any
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MAX_RETRIES = 3
INITIAL_DELAY = 1  # seconds
BACKOFF_FACTOR = 2


def retry_with_backoff(max_retries: int = MAX_RETRIES, 
                       initial_delay: float = INITIAL_DELAY,
                       backoff_factor: float = BACKOFF_FACTOR,
                       exceptions: tuple = (Exception,)):
    """
    Decorator for retrying functions with exponential backoff.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        raise
        
        return wrapper
    return decorator


def rate_limit_check(last_request_time: float, min_interval: float = 1.0) -> bool:
    """Check if enough time has passed since last request."""
    current_time = time.time()
    return (current_time - last_request_time) >= min_interval

