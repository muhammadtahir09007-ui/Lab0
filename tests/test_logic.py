"""
Unit tests for preprocessing functions.
Uses pytest fixtures and parametrize decorator as required by Lab0.
"""

import math
import pytest
from src import preprocessing


# ============================================================
# FIXTURES — Reusable test data
# ============================================================

@pytest.fixture
def sample_list_with_missing():
    """List containing None, empty string, and NaN."""
    return [1, None, 3, '', float('nan'), 5]


@pytest.fixture
def sample_numbers():
    """Simple list of numbers."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def sample_duplicates():
    """List with duplicate values."""
    return [1, 2, 2, 3, 1, 4]


@pytest.fixture
def sample_text():
    """Sample text for text processing tests."""
    return "Hello, World! This is a Test 123."


@pytest.fixture
def sample_nested():
    """Nested list for flatten tests."""
    return [[1, 2], [3, 4], [5]]


@pytest.fixture
def sample_strings():
    """List of strings for convert_to_integers tests."""
    return ["1", "hello", "3", "world", "5"]


# ============================================================
# TESTS FOR DATA CLEANING
# ============================================================

class TestRemoveMissingValues:
    """Tests for remove_missing_values function."""

    def test_removes_none(self):
        assert preprocessing.remove_missing_values([1, None, 3]) == [1, 3]

    def test_removes_empty_string(self):
        assert preprocessing.remove_missing_values([1, '', 3]) == [1, 3]

    def test_removes_nan(self):
        result = preprocessing.remove_missing_values([1, float('nan'), 3])
        assert result == [1, 3]

    def test_with_fixture(self, sample_list_with_missing):
        result = preprocessing.remove_missing_values(sample_list_with_missing)
        assert result == [1, 3, 5]

    def test_empty_list(self):
        assert preprocessing.remove_missing_values([]) == []

    def test_all_missing(self):
        assert preprocessing.remove_missing_values([None, '', float('nan')]) == []


class TestFillMissingValues:
    """Tests for fill_missing_values function."""

    def test_fill_with_zero(self):
        result = preprocessing.fill_missing_values([1, None, 3])
        assert result == [1, 0, 3]

    def test_fill_with_custom_value(self):
        result = preprocessing.fill_missing_values([1, None, 3], fill_value=99)
        assert result == [1, 99, 3]

    def test_with_fixture(self, sample_list_with_missing):
        result = preprocessing.fill_missing_values(sample_list_with_missing, fill_value=-1)
        assert result == [1, -1, 3, -1, -1, 5]

    @pytest.mark.parametrize("fill_value", [0, -1, 99, 0.5])
    def test_various_fill_values(self, fill_value):
        result = preprocessing.fill_missing_values([None], fill_value=fill_value)
        assert result == [fill_value]


class TestRemoveDuplicates:
    """Tests for remove_duplicates function."""

    def test_basic(self, sample_duplicates):
        assert preprocessing.remove_duplicates(sample_duplicates) == [1, 2, 3, 4]

    def test_no_duplicates(self):
        assert preprocessing.remove_duplicates([1, 2, 3]) == [1, 2, 3]

    def test_all_same(self):
        assert preprocessing.remove_duplicates([5, 5, 5]) == [5]

    def test_empty(self):
        assert preprocessing.remove_duplicates([]) == []


# ============================================================
# TESTS FOR NUMERICAL TRANSFORMATIONS
# ============================================================

class TestNormalize:
    """Tests for normalize function."""

    def test_default_range(self, sample_numbers):
        result = preprocessing.normalize(sample_numbers)
        assert result == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_custom_range(self):
        result = preprocessing.normalize([1, 2, 3, 4, 5], new_min=0, new_max=10)
        assert result == [0.0, 2.5, 5.0, 7.5, 10.0]

    def test_same_values(self):
        result = preprocessing.normalize([5, 5, 5])
        assert result == [0.0, 0.0, 0.0]

    @pytest.mark.parametrize("new_min, new_max", [(0, 1), (0, 100), (-1, 1)])
    def test_various_ranges(self, new_min, new_max):
        result = preprocessing.normalize([1, 5], new_min=new_min, new_max=new_max)
        assert result[0] == new_min
        assert result[-1] == new_max


class TestStandardize:
    """Tests for standardize function."""

    def test_basic(self):
        result = preprocessing.standardize([10, 20, 30])
        assert len(result) == 3
        assert abs(result[1]) < 1e-10  # middle value should be ~0

    def test_same_values(self):
        result = preprocessing.standardize([5, 5, 5])
        assert result == [0.0, 0.0, 0.0]


class TestClip:
    """Tests for clip function."""

    @pytest.mark.parametrize("value, min_v, max_v, expected", [
        ([-2, 0.5, 3], 0, 1, [0, 0.5, 1]),
        ([5, 10, 15], 0, 20, [5, 10, 15]),
        ([-10, 0, 10], -5, 5, [-5, 0, 5]),
    ])
    def test_clip_ranges(self, value, min_v, max_v, expected):
        assert preprocessing.clip(value, min_value=min_v, max_value=max_v) == expected


class TestConvertToIntegers:
    """Tests for convert_to_integers function."""

    def test_basic(self, sample_strings):
        assert preprocessing.convert_to_integers(sample_strings) == [1, 3, 5]

    def test_all_numeric(self):
        assert preprocessing.convert_to_integers(["1", "2", "3"]) == [1, 2, 3]

    def test_none_numeric(self):
        assert preprocessing.convert_to_integers(["hello", "world"]) == []


class TestLogTransform:
    """Tests for log_transform function."""

    def test_basic(self):
        result = preprocessing.log_transform([1, 10, 100])
        assert abs(result[0] - 0.0) < 1e-10
        assert len(result) == 3

    def test_skips_non_positive(self):
        result = preprocessing.log_transform([1, -5, 0, 10])
        assert len(result) == 2  # only 1 and 10

    def test_empty(self):
        assert preprocessing.log_transform([]) == []


# ============================================================
# TESTS FOR TEXT PROCESSING
# ============================================================

class TestTokenize:
    """Tests for tokenize function."""

    def test_basic(self, sample_text):
        result = preprocessing.tokenize(sample_text)
        assert result == ['hello', 'world', 'this', 'is', 'a', 'test', '123']

    def test_empty_string(self):
        assert preprocessing.tokenize("") == []

    @pytest.mark.parametrize("text, expected", [
        ("Hello World", ['hello', 'world']),
        ("test123", ['test123']),
        ("!@#$%", []),
    ])
    def test_various_inputs(self, text, expected):
        assert preprocessing.tokenize(text) == expected


class TestSelectAlphanumeric:
    """Tests for select_alphanumeric function."""

    def test_basic(self):
        assert preprocessing.select_alphanumeric("Hello, World! #2026") == "Hello World 2026"

    def test_no_special_chars(self):
        assert preprocessing.select_alphanumeric("Hello World") == "Hello World"

    def test_all_special(self):
        assert preprocessing.select_alphanumeric("!@#$%^&*()") == ""


class TestRemoveStopWords:
    """Tests for remove_stop_words function."""

    def test_basic(self):
        result = preprocessing.remove_stop_words("this is a test", ["this", "is", "a"])
        assert result == "test"

    def test_no_stop_words(self):
        result = preprocessing.remove_stop_words("hello world", ["the", "is"])
        assert result == "hello world"

    def test_empty_stop_words(self):
        result = preprocessing.remove_stop_words("hello world", [])
        assert result == "hello world"


# ============================================================
# TESTS FOR STRUCTURAL OPERATIONS
# ============================================================

class TestFlatten:
    """Tests for flatten function."""

    def test_basic(self, sample_nested):
        assert preprocessing.flatten(sample_nested) == [1, 2, 3, 4, 5]

    def test_single_list(self):
        assert preprocessing.flatten([[1, 2, 3]]) == [1, 2, 3]

    def test_empty_sublists(self):
        assert preprocessing.flatten([[], [1], []]) == [1]

    def test_empty(self):
        assert preprocessing.flatten([]) == []


class TestShuffleValues:
    """Tests for shuffle_values function."""

    def test_with_seed(self, sample_numbers):
        result = preprocessing.shuffle_values(sample_numbers, seed=42)
        assert sorted(result) == sorted(sample_numbers)  # same elements
        assert len(result) == len(sample_numbers)

    def test_reproducible(self):
        a = preprocessing.shuffle_values([1, 2, 3, 4, 5], seed=42)
        b = preprocessing.shuffle_values([1, 2, 3, 4, 5], seed=42)
        assert a == b  # same seed = same result

    def test_does_not_modify_original(self):
        original = [1, 2, 3]
        preprocessing.shuffle_values(original, seed=1)
        assert original == [1, 2, 3]  # original unchanged
    