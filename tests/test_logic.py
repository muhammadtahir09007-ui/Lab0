"""Unit tests for preprocessing functions."""

import math
import pytest
from src import preprocessing


# fixtures

@pytest.fixture
def missing_list():
    return [1, None, 3, "", float("nan"), 5]


@pytest.fixture
def numbers():
    return [1, 2, 3, 4, 5]


@pytest.fixture
def with_dupes():
    return [1, 2, 2, 3, 1, 4]


@pytest.fixture
def nested():
    return [[1, 2], [3, 4], [5]]


# remove_missing_values

class TestRemoveMissing:
    def test_basic(self, missing_list):
        assert preprocessing.remove_missing_values(missing_list) == [1, 3, 5]

    def test_empty(self):
        assert preprocessing.remove_missing_values([]) == []

    def test_all_missing(self):
        assert preprocessing.remove_missing_values([None, "", float("nan")]) == []


# fill_missing_values

class TestFillMissing:
    def test_default(self):
        assert preprocessing.fill_missing_values([1, None, 3]) == [1, 0, 3]

    def test_custom(self):
        assert preprocessing.fill_missing_values([1, None, 3], fill_value=99) == [1, 99, 3]

    @pytest.mark.parametrize("fill", [0, -1, 99, 0.5])
    def test_different_fills(self, fill):
        assert preprocessing.fill_missing_values([None], fill_value=fill) == [fill]


# remove_duplicates

class TestRemoveDuplicates:
    def test_basic(self, with_dupes):
        assert preprocessing.remove_duplicates(with_dupes) == [1, 2, 3, 4]

    def test_no_dupes(self):
        assert preprocessing.remove_duplicates([1, 2, 3]) == [1, 2, 3]

    def test_all_same(self):
        assert preprocessing.remove_duplicates([5, 5, 5]) == [5]


# normalize

class TestNormalize:
    def test_default(self, numbers):
        assert preprocessing.normalize(numbers) == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_custom_range(self):
        assert preprocessing.normalize([1, 2, 3, 4, 5], new_min=0, new_max=10) == [0.0, 2.5, 5.0, 7.5, 10.0]

    def test_same_values(self):
        assert preprocessing.normalize([5, 5, 5]) == [0.0, 0.0, 0.0]

    @pytest.mark.parametrize("lo,hi", [(0, 1), (0, 100), (-1, 1)])
    def test_ranges(self, lo, hi):
        result = preprocessing.normalize([1, 5], new_min=lo, new_max=hi)
        assert result[0] == lo
        assert result[-1] == hi


# standardize

class TestStandardize:
    def test_basic(self):
        result = preprocessing.standardize([10, 20, 30])
        assert abs(result[1]) < 1e-10

    def test_same_values(self):
        assert preprocessing.standardize([5, 5, 5]) == [0.0, 0.0, 0.0]


# clip

class TestClip:
    @pytest.mark.parametrize("vals,lo,hi,expected", [
        ([-2, 0.5, 3], 0, 1, [0, 0.5, 1]),
        ([5, 10, 15], 0, 20, [5, 10, 15]),
        ([-10, 0, 10], -5, 5, [-5, 0, 5]),
    ])
    def test_ranges(self, vals, lo, hi, expected):
        assert preprocessing.clip(vals, min_value=lo, max_value=hi) == expected


# convert_to_integers

class TestConvertToIntegers:
    def test_mixed(self):
        assert preprocessing.convert_to_integers(["1", "hello", "3", "world", "5"]) == [1, 3, 5]

    def test_all_numeric(self):
        assert preprocessing.convert_to_integers(["1", "2", "3"]) == [1, 2, 3]

    def test_none_numeric(self):
        assert preprocessing.convert_to_integers(["hello", "world"]) == []


# log_transform

class TestLogTransform:
    def test_basic(self):
        result = preprocessing.log_transform([1, 10, 100])
        assert abs(result[0]) < 1e-10
        assert len(result) == 3

    def test_skips_non_positive(self):
        assert len(preprocessing.log_transform([1, -5, 0, 10])) == 2


# tokenize

class TestTokenize:
    def test_basic(self):
        assert preprocessing.tokenize("Hello, World! This is a Test 123.") == ["hello", "world", "this", "is", "a", "test", "123"]

    def test_empty(self):
        assert preprocessing.tokenize("") == []

    @pytest.mark.parametrize("text,expected", [
        ("Hello World", ["hello", "world"]),
        ("test123", ["test123"]),
        ("!@#$%", []),
    ])
    def test_various(self, text, expected):
        assert preprocessing.tokenize(text) == expected


# select_alphanumeric

class TestSelectAlphanumeric:
    def test_basic(self):
        assert preprocessing.select_alphanumeric("Hello, World! #2026") == "Hello World 2026"

    def test_no_special(self):
        assert preprocessing.select_alphanumeric("Hello World") == "Hello World"


# remove_stop_words

class TestRemoveStopWords:
    def test_basic(self):
        assert preprocessing.remove_stop_words("this is a test", ["this", "is", "a"]) == "test"

    def test_no_matches(self):
        assert preprocessing.remove_stop_words("hello world", ["the", "is"]) == "hello world"


# flatten

class TestFlatten:
    def test_basic(self, nested):
        assert preprocessing.flatten(nested) == [1, 2, 3, 4, 5]

    def test_empty(self):
        assert preprocessing.flatten([]) == []

    def test_empty_sublists(self):
        assert preprocessing.flatten([[], [1], []]) == [1]


# shuffle_values

class TestShuffle:
    def test_same_elements(self, numbers):
        result = preprocessing.shuffle_values(numbers, seed=42)
        assert sorted(result) == sorted(numbers)

    def test_reproducible(self):
        a = preprocessing.shuffle_values([1, 2, 3, 4, 5], seed=42)
        b = preprocessing.shuffle_values([1, 2, 3, 4, 5], seed=42)
        assert a == b

    def test_original_unchanged(self):
        original = [1, 2, 3]
        preprocessing.shuffle_values(original, seed=1)
        assert original == [1, 2, 3]
