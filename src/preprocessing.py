"""Data preprocessing functions."""

import math
import re
import random


def remove_missing_values(values):
    """Remove None, empty strings and NaN from a list."""
    result = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and v == "":
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        result.append(v)
    return result


def fill_missing_values(values, fill_value=0):
    """Replace None, empty strings and NaN with fill_value."""
    result = []
    for v in values:
        if v is None or (isinstance(v, str) and v == "") or (isinstance(v, float) and math.isnan(v)):
            result.append(fill_value)
        else:
            result.append(v)
    return result


def remove_duplicates(values):
    """Remove duplicates preserving order."""
    seen = []
    result = []
    for v in values:
        if v not in seen:
            seen.append(v)
            result.append(v)
    return result


def normalize(values, new_min=0.0, new_max=1.0):
    """Min-max normalization."""
    old_min = min(values)
    old_max = max(values)
    if old_max == old_min:
        return [new_min] * len(values)
    return [
        (v - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        for v in values
    ]


def standardize(values):
    """Z-score standardization."""
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
    if std == 0:
        return [0.0] * n
    return [(x - mean) / std for x in values]


def clip(values, min_value=0, max_value=1):
    """Clip values to [min_value, max_value]."""
    return [max(min_value, min(max_value, v)) for v in values]


def convert_to_integers(values):
    """Convert strings to ints, skip non-numeric."""
    result = []
    for v in values:
        try:
            result.append(int(v))
        except (ValueError, TypeError):
            continue
    return result


def log_transform(values):
    """Natural log of positive values only."""
    return [math.log(v) for v in values if v > 0]


def tokenize(text):
    """Split text into lowercase alphanumeric tokens."""
    return [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", text)]


def select_alphanumeric(text):
    """Keep only alphanumeric chars and spaces."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def remove_stop_words(text, stop_words):
    """Remove stop words from text."""
    return " ".join(w for w in text.split() if w not in stop_words)


def flatten(list_of_lists):
    """Flatten nested list."""
    result = []
    for sublist in list_of_lists:
        result.extend(sublist)
    return result


def shuffle_values(values, seed=None):
    """Shuffle a list. Use seed for reproducibility."""
    shuffled = values.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(shuffled)
    return shuffled
