"""
Module for data preprocessing functions.
Includes functions for data cleaning, numerical transformations,
text processing, and structural operations.
"""

import math
import re
import random

# ============================================================
# DATA CLEANING FUNCTIONS
# ============================================================


def remove_missing_values(values):
    """
    Remove missing values from a list.
    Missing values include: None, empty string '', and float('nan').

    Args:
        values: List of values that may contain missing values.

    Returns:
        List of values with missing values removed.

    Example:
        >>> remove_missing_values([1, None, 3, '', float('nan'), 5])
        [1, 3, 5]
    """
    result = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        result.append(value)
    return result


def fill_missing_values(values, fill_value=0):
    """
    Replace missing values in a list with a given fill value.
    Missing values include: None, empty string '', and float('nan').

    Args:
        values: List of values that may contain missing values.
        fill_value: Value to replace missing values with. Default is 0.

    Returns:
        List of values where missing values are replaced with fill_value.

    Example:
        >>> fill_missing_values([1, None, 3, '', float('nan')], fill_value=0)
        [1, 0, 3, 0, 0]
    """
    result = []
    for value in values:
        if value is None:
            result.append(fill_value)
        elif isinstance(value, str) and value == "":
            result.append(fill_value)
        elif isinstance(value, float) and math.isnan(value):
            result.append(fill_value)
        else:
            result.append(value)
    return result


# ============================================================
# REMOVAL OF DUPLICATES
# ============================================================


def remove_duplicates(values):
    """
    Remove duplicated values from a list, preserving order.

    Args:
        values: List of values.

    Returns:
        List of unique values in original order.

    Example:
        >>> remove_duplicates([1, 2, 2, 3, 1, 4])
        [1, 2, 3, 4]
    """
    seen = []
    result = []
    for value in values:
        if value not in seen:
            seen.append(value)
            result.append(value)
    return result


# ============================================================
# NUMERICAL TRANSFORMATION FUNCTIONS
# ============================================================


def normalize(values, new_min=0.0, new_max=1.0):
    """
    Normalize numerical values using the min-max method.

    Args:
        values: List of numerical values.
        new_min: New minimum value. Default is 0.0.
        new_max: New maximum value. Default is 1.0.

    Returns:
        List of normalized values.

    Example:
        >>> normalize([1, 2, 3, 4, 5])
        [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    old_min = min(values)
    old_max = max(values)

    if old_max == old_min:
        return [new_min] * len(values)

    result = []
    for value in values:
        normalized = (value - old_min) / (old_max - old_min)
        scaled = normalized * (new_max - new_min) + new_min
        result.append(scaled)
    return result


def standardize(values):
    """
    Standardize numerical values using the z-score method.
    z = (x - mean) / std

    Args:
        values: List of numerical values.

    Returns:
        List of standardized values.

    Example:
        >>> standardize([10, 20, 30])
        [-1.2247448713915890, 0.0, 1.2247448713915890]
    """
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)

    if std == 0:
        return [0.0] * n

    return [(x - mean) / std for x in values]


def clip(values, min_value=0, max_value=1):
    """
    Clip numerical values to a certain range.

    Args:
        values: List of numerical values.
        min_value: Minimum value to clip. Default is 0.
        max_value: Maximum value to clip. Default is 1.

    Returns:
        List of clipped values.

    Example:
        >>> clip([-2, 0.5, 3], min_value=0, max_value=1)
        [0, 0.5, 1]
    """
    result = []
    for value in values:
        if value < min_value:
            result.append(min_value)
        elif value > max_value:
            result.append(max_value)
        else:
            result.append(value)
    return result


def convert_to_integers(values):
    """
    Convert a list of string values to integers.
    Non-numerical values are excluded from the output.

    Args:
        values: List of strings (can include both numerical and non-numerical).

    Returns:
        List of values converted to integers.

    Example:
        >>> convert_to_integers(["1", "hello", "3", "world", "5"])
        [1, 3, 5]
    """
    result = []
    for value in values:
        try:
            result.append(int(value))
        except (ValueError, TypeError):
            continue
    return result


def log_transform(values):
    """
    Apply logarithmic scale transformation.
    Only positive numbers are transformed; non-positive are excluded.

    Args:
        values: List of numerical values.

    Returns:
        List of values converted to logarithmic scale (only positive numbers).

    Example:
        >>> log_transform([1, 10, -5, 100, 0])
        [0.0, 2.302585092994046, 4.605170185988092]
    """
    result = []
    for value in values:
        if value > 0:
            result.append(math.log(value))
    return result


# ============================================================
# TEXT PROCESSING FUNCTIONS
# ============================================================


def tokenize(text):
    """
    Tokenize text into words, selecting only alphanumeric characters
    and lower-casing words.

    Args:
        text: Text to be processed.

    Returns:
        List of lowercase words containing only alphanumeric characters.

    Example:
        >>> tokenize("Hello, World! This is a Test 123.")
        ['hello', 'world', 'this', 'is', 'a', 'test', '123']
    """
    words = re.findall(r"[a-zA-Z0-9]+", text)
    return [word.lower() for word in words]


def select_alphanumeric(text):
    """
    Select only alphanumerical characters and spaces from text.

    Args:
        text: Text to be processed.

    Returns:
        Processed text with only alphanumeric characters and spaces.

    Example:
        >>> select_alphanumeric("Hello, World! #2026")
        'Hello World 2026'
    """
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def remove_stop_words(text, stop_words):
    """
    Remove stop words from text. Text should be lower-cased.

    Args:
        text: Text to be processed (should be lower-cased).
        stop_words: List of stop words to be removed.

    Returns:
        Processed text with stop words removed.

    Example:
        >>> remove_stop_words("this is a test", ["this", "is", "a"])
        'test'
    """
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return " ".join(filtered)


# ============================================================
# STRUCTURAL FUNCTIONS
# ============================================================


def flatten(list_of_lists):
    """
    Flatten a list of lists into a single list.

    Args:
        list_of_lists: A list of lists.

    Returns:
        A flattened list.

    Example:
        >>> flatten([[1, 2], [3, 4], [5]])
        [1, 2, 3, 4, 5]
    """
    result = []
    for sublist in list_of_lists:
        for item in sublist:
            result.append(item)
    return result


def shuffle_values(values, seed=None):
    """
    Randomly shuffle a list of values.

    Args:
        values: List of values.
        seed: Seed for reproducibility. Default is None.

    Returns:
        List of shuffled values.

    Example:
        >>> shuffle_values([1, 2, 3, 4, 5], seed=42)
        [4, 1, 5, 2, 3]
    """
    shuffled = values.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(shuffled)
    return shuffled
