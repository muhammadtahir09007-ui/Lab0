"""Integration tests for the CLI using CliRunner."""

import pytest
from click.testing import CliRunner
from src.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


# clean commands

class TestClean:
    def test_remove_missing(self, runner):
        result = runner.invoke(cli, ["clean", "remove-missing", '[1, null, 3, "", 5]'])
        assert result.exit_code == 0
        assert "1" in result.output

    def test_fill_missing(self, runner):
        result = runner.invoke(cli, ["clean", "fill-missing", "[1, null, 3]", "--fill", "99"])
        assert result.exit_code == 0
        assert "99" in result.output

    def test_deduplicate(self, runner):
        result = runner.invoke(cli, ["clean", "deduplicate", "[1, 2, 2, 3, 1, 4]"])
        assert result.exit_code == 0
        assert "Result:" in result.output


# numeric commands

class TestNumeric:
    def test_norm(self, runner):
        result = runner.invoke(cli, ["numeric", "norm", "[1, 2, 3, 4, 5]"])
        assert result.exit_code == 0
        assert "0.0" in result.output

    def test_norm_custom(self, runner):
        result = runner.invoke(cli, ["numeric", "norm", "[1, 5]", "--min", "0", "--max", "10"])
        assert result.exit_code == 0
        assert "10.0" in result.output

    def test_stand(self, runner):
        result = runner.invoke(cli, ["numeric", "stand", "[10, 20, 30]"])
        assert result.exit_code == 0

    def test_clp(self, runner):
        result = runner.invoke(cli, ["numeric", "clp", "[-2, 0.5, 3]", "--min", "0", "--max", "1"])
        assert result.exit_code == 0
        assert "0.5" in result.output

    def test_to_int(self, runner):
        result = runner.invoke(cli, ["numeric", "to-int", '["1", "hello", "3"]'])
        assert result.exit_code == 0
        assert "1" in result.output

    def test_log(self, runner):
        result = runner.invoke(cli, ["numeric", "log", "[1, 10, 100]"])
        assert result.exit_code == 0
        assert "0.0" in result.output


# text commands

class TestText:
    def test_tokenize(self, runner):
        result = runner.invoke(cli, ["text", "tokenize", "Hello, World! Test 123"])
        assert result.exit_code == 0
        assert "hello" in result.output

    def test_alphanumeric(self, runner):
        result = runner.invoke(cli, ["text", "alphanumeric", "Hello, World! #2026"])
        assert result.exit_code == 0
        assert "Hello World 2026" in result.output

    def test_remove_stops(self, runner):
        result = runner.invoke(cli, ["text", "remove-stops", "this is a test", "--stops", "this,is,a"])
        assert result.exit_code == 0
        assert "test" in result.output


# struct commands

class TestStruct:
    def test_flat(self, runner):
        result = runner.invoke(cli, ["struct", "flat", "[[1, 2], [3, 4], [5]]"])
        assert result.exit_code == 0
        assert "1" in result.output

    def test_shuffle(self, runner):
        result = runner.invoke(cli, ["struct", "shuffle", "[1, 2, 3, 4, 5]", "--seed", "42"])
        assert result.exit_code == 0
        assert "Result:" in result.output


# error handling

class TestErrors:
    def test_no_args(self, runner):
        result = runner.invoke(cli, [])
        assert result.exit_code == 2

    def test_bad_group(self, runner):
        result = runner.invoke(cli, ["invalid"])
        assert result.exit_code != 0

    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    @pytest.mark.parametrize("group", ["clean", "numeric", "text", "struct"])
    def test_group_help(self, runner, group):
        result = runner.invoke(cli, [group, "--help"])
        assert result.exit_code == 0
