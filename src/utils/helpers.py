"""
Helper utility functions for Agentic RAG System.

This module provides common utility functions used across the system:
- Text processing and cleaning
- Timing and performance measurement
- Dictionary and data structure manipulation
- Validation helpers
- String formatting utilities

These are general-purpose functions that don't belong to any specific component.
"""

import time
import hashlib
import re
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from functools import wraps


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Removes extra whitespace, fixes line breaks, and normalizes special characters.
    Useful for preprocessing document chunks and queries.
    
    Args:
        text: Raw text to clean
    
    Returns:
        Cleaned text string
    
    Example:
        >>> clean_text("Hello   World\\n\\n\\nTest")
        'Hello World\\nTest'
    """
    if not text:
        return ""
    
    # Fix line breaks (max 2 consecutive) BEFORE removing spaces
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove multiple spaces (but not newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Useful for logging and displaying long text snippets.
    Tries to break at word boundaries when possible.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: String to append when truncated (default: "...")
    
    Returns:
        Truncated text
    
    Example:
        >>> truncate_text("This is a very long sentence", max_length=15)
        'This is a...'
    """
    if len(text) <= max_length:
        return text
    
    # Account for suffix length
    actual_length = max_length - len(suffix)
    
    # Try to break at last space before max_length
    truncated = text[:actual_length]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated + suffix


def generate_hash(text: str, length: int = 8) -> str:
    """
    Generate short hash from text.
    
    Useful for creating unique IDs for chunks, queries, or cache keys.
    Uses MD5 for speed (not cryptographic security).
    
    Args:
        text: Text to hash
        length: Length of hash string (default: 8)
    
    Returns:
        Hash string
    
    Example:
        >>> generate_hash("Hello World")
        'b10a8db1'
    """
    hash_obj = hashlib.md5(text.encode('utf-8'))
    return hash_obj.hexdigest()[:length]


def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Wraps a function and prints its execution time.
    Useful for performance profiling.
    
    Args:
        func: Function to time
    
    Returns:
        Wrapped function
    
    Example:
        >>> @timer
        ... def slow_function():
        ...     time.sleep(1)
        >>> slow_function()
        Function 'slow_function' took 1.00s
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"Function '{func.__name__}' took {elapsed:.2f}s")
        return result
    return wrapper


def measure_time(func: Callable) -> tuple:
    """
    Measure function execution time and return result with timing.
    
    Unlike @timer decorator, this returns the timing info for programmatic use.
    
    Args:
        func: Function to execute and time
    
    Returns:
        Tuple of (result, elapsed_time)
    
    Example:
        >>> result, time_taken = measure_time(lambda: expensive_operation())
        >>> print(f"Took {time_taken:.2f}s")
    """
    start_time = time.time()
    result = func()
    elapsed = time.time() - start_time
    return result, elapsed


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default on division by zero.
    
    Useful for calculating ratios and averages without error handling boilerplate.
    
    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Value to return if denominator is zero
    
    Returns:
        Division result or default
    
    Example:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0, default=0.0)
        0.0
    """
    if denominator == 0:
        return default
    return numerator / denominator


def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary into single-level dictionary.
    
    Combines nested keys with separator.
    Useful for serialization and logging.
    
    Args:
        nested_dict: Dictionary to flatten
        parent_key: Prefix for keys (used in recursion)
        sep: Separator between nested keys
    
    Returns:
        Flattened dictionary
    
    Example:
        >>> nested = {"a": {"b": {"c": 1}}, "d": 2}
        >>> flatten_dict(nested)
        {'a.b.c': 1, 'd': 2}
    """
    items = []
    
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries into one.
    
    Later dictionaries override earlier ones for duplicate keys.
    
    Args:
        *dicts: Variable number of dictionaries to merge
    
    Returns:
        Merged dictionary
    
    Example:
        >>> merge_dicts({"a": 1}, {"b": 2}, {"a": 3})
        {'a': 3, 'b': 2}
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Useful for batch processing and pagination.
    
    Args:
        items: List to split
        chunk_size: Maximum size of each chunk
    
    Returns:
        List of chunks
    
    Example:
        >>> chunk_list([1, 2, 3, 4, 5], chunk_size=2)
        [[1, 2], [3, 4], [5]]
    """
    chunks = []
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i:i + chunk_size])
    return chunks


def remove_duplicates(items: List[Any], key: Optional[Callable] = None) -> List[Any]:
    """
    Remove duplicates from list while preserving order.
    
    Can use custom key function for determining uniqueness.
    
    Args:
        items: List with potential duplicates
        key: Optional function to extract comparison key from items
    
    Returns:
        List with duplicates removed
    
    Example:
        >>> remove_duplicates([1, 2, 2, 3, 1])
        [1, 2, 3]
        >>> remove_duplicates([{"id": 1}, {"id": 2}, {"id": 1}], key=lambda x: x["id"])
        [{'id': 1}, {'id': 2}]
    """
    seen = set()
    result = []
    
    for item in items:
        # Use key function if provided, otherwise use item itself
        check_value = key(item) if key else item
        
        # Handle unhashable types
        try:
            if check_value not in seen:
                seen.add(check_value)
                result.append(item)
        except TypeError:
            # For unhashable types, do linear search
            if check_value not in [key(x) if key else x for x in result]:
                result.append(item)
    
    return result


def format_timestamp(dt: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime object as string.
    
    Uses current time if no datetime provided.
    
    Args:
        dt: Datetime object to format (default: now)
        format_str: strftime format string
    
    Returns:
        Formatted timestamp string
    
    Example:
        >>> format_timestamp()
        '2024-12-23 21:30:00'
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime(format_str)


def parse_bool(value: Any) -> bool:
    """
    Parse various types into boolean.
    
    Handles strings like "true", "yes", "1" as True.
    Useful for configuration parsing.
    
    Args:
        value: Value to parse as boolean
    
    Returns:
        Boolean value
    
    Example:
        >>> parse_bool("true")
        True
        >>> parse_bool("no")
        False
        >>> parse_bool(1)
        True
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        return value.lower() in ('true', 'yes', '1', 'on')
    
    return bool(value)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between minimum and maximum.
    
    Ensures value stays within valid range.
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value
    
    Returns:
        Clamped value
    
    Example:
        >>> clamp(5, 0, 10)
        5
        >>> clamp(-5, 0, 10)
        0
        >>> clamp(15, 0, 10)
        10
    """
    return max(min_value, min(value, max_value))


def percentage(part: float, total: float, decimals: int = 2) -> float:
    """
    Calculate percentage with safe division.
    
    Returns 0.0 if total is 0.
    
    Args:
        part: Partial value
        total: Total value
        decimals: Number of decimal places to round
    
    Returns:
        Percentage (0-100)
    
    Example:
        >>> percentage(25, 100)
        25.0
        >>> percentage(1, 3)
        33.33
    """
    if total == 0:
        return 0.0
    return round((part / total) * 100, decimals)


def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Decorator to retry function on exception.
    
    Useful for handling transient failures (network, API rate limits, etc).
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        exceptions: Tuple of exceptions to catch
    
    Returns:
        Decorated function
    
    Example:
        >>> @retry(max_attempts=3, delay=1.0)
        ... def unstable_api_call():
        ...     # might fail
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                        continue
                    else:
                        raise last_exception
        
        return wrapper
    return decorator


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from text.
    
    Finds integers and floats in text string.
    
    Args:
        text: Text to extract numbers from
    
    Returns:
        List of numbers found
    
    Example:
        >>> extract_numbers("I have 3 apples and 2.5 oranges")
        [3.0, 2.5]
    """
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches]


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Simple word count by splitting on whitespace.
    
    Args:
        text: Text to count words in
    
    Returns:
        Number of words
    
    Example:
        >>> count_words("Hello world")
        2
    """
    return len(text.split())


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
    
    Returns:
        File size in MB
    
    Example:
        >>> get_file_size_mb("document.pdf")
        2.5
    """
    import os
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)