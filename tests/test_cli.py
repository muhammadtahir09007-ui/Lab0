"""
Integration tests for the CLI using Click's CliRunner.
Tests that the CLI commands work end-to-end.
"""

import pytest
from click.testing import CliRunner
from src.cli import cli


# ============================================================
# FIXTURE — CliRunner instance
# ============================================================

@pytest.fixture
def runner():
    """Create a CliRunner instance for testing CLI commands."""
    return CliRunner()


# ============================================================
# TESTS FOR CLEAN GROUP
# ============================================================

class TestCleanCommands:
    """Integration tests for the 'clean' command group."""

    def test_remove_missing(self, runner):
        result = runner.invoke(cli, ['clean', 'remove-missing', '[1, null, 3, "", 5]'])
        assert result.exit_code == 0
        assert "1" in result.output
        assert "3" in result.output
        assert "5" in result.output

    def test_fill_missing_default(self, runner):
        result = runner.invoke(cli, ['clean', 'fill-missing', '[1, null, 3]'])
        assert result.exit_code == 0
        assert "0" in result.output

    def test_fill_missing_custom(self, runner):
        result = runner.invoke(cli, ['clean', 'fill-missing', '[1, null, 3]', '--fill', '99'])
        assert result.exit_code == 0
        assert "99" in result.output

    def test_deduplicate(self, runner):
        result = runner.invoke(cli, ['clean', 'deduplicate', '[1, 2, 2, 3, 1, 4]'])
        assert result.exit_code == 0
        assert "Result:" in result.output


# ============================================================
# TESTS FOR NUMERIC GROUP
# ============================================================

class TestNumericCommands:
    """Integration tests for the 'numeric' command group."""

    def test_normalize(self, runner):
        result = runner.invoke(cli, ['numeric', 'norm', '[1, 2, 3, 4, 5]'])
        assert result.exit_code == 0
        assert "0.0" in result.output
        assert "1.0" in result.output

    def test_normalize_custom_range(self, runner):
        result = runner.invoke(cli, ['numeric', 'norm', '[1, 5]', '--min', '0', '--max', '10'])
        assert result.exit_code == 0
        assert "10.0" in result.output

    def test_standardize(self, runner):
        result = runner.invoke(cli, ['numeric', 'stand', '[10, 20, 30]'])
        assert result.exit_code == 0
        assert "Result:" in result.output

    def test_clip(self, runner):
        result = runner.invoke(cli, ['numeric', 'clp', '[-2, 0.5, 3]', '--min', '0', '--max', '1'])
        assert result.exit_code == 0
        assert "0.5" in result.output

    def test_convert_to_integers(self, runner):
        result = runner.invoke(cli, ['numeric', 'to-int', '["1", "hello", "3"]'])
        assert result.exit_code == 0
        assert "1" in result.output
        assert "3" in result.output

    def test_log_transform(self, runner):
        result = runner.invoke(cli, ['numeric', 'log', '[1, 10, 100]'])
        assert result.exit_code == 0
        assert "0.0" in result.output


# ============================================================
# TESTS FOR TEXT GROUP
# ============================================================

class TestTextCommands:
    """Integration tests for the 'text' command group."""

    def test_tokenize(self, runner):
        result = runner.invoke(cli, ['text', 'tokenize', 'Hello, World! Test 123'])
        assert result.exit_code == 0
        assert "hello" in result.output
        assert "world" in result.output

    def test_alphanumeric(self, runner):
        result = runner.invoke(cli, ['text', 'alphanumeric', 'Hello, World! #2026'])
        assert result.exit_code == 0
        assert "Hello World 2026" in result.output

    def test_remove_stops(self, runner):
        result = runner.invoke(cli, ['text', 'remove-stops', 'this is a test', '--stops', 'this,is,a'])
        assert result.exit_code == 0
        assert "test" in result.output


# ============================================================
# TESTS FOR STRUCT GROUP
# ============================================================

class TestStructCommands:
    """Integration tests for the 'struct' command group."""

    def test_flatten(self, runner):
        result = runner.invoke(cli, ['struct', 'flat', '[[1, 2], [3, 4], [5]]'])
        assert result.exit_code == 0
        assert "1" in result.output
        assert "5" in result.output

    def test_shuffle_with_seed(self, runner):
        result = runner.invoke(cli, ['struct', 'shuffle', '[1, 2, 3, 4, 5]', '--seed', '42'])
        assert result.exit_code == 0
        assert "Result:" in result.output


# ============================================================
# TESTS FOR ERROR HANDLING
# ============================================================

class TestErrorHandling:
    """Tests that CLI handles bad input gracefully."""

    def test_no_arguments(self, runner):
        result = runner.invoke(cli, [])
        assert result.exit_code == 2  # shows help text

    def test_invalid_group(self, runner):
        result = runner.invoke(cli, ['invalid_group'])
        assert result.exit_code != 0  # should fail

    def test_help_flag(self, runner):
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "Data Preprocessing CLI" in result.output


# ============================================================
# TESTS FOR GROUP HELP
# ============================================================

class TestGroupHelp:
    """Tests that each group shows its help text."""

    @pytest.mark.parametrize("group", ['clean', 'numeric', 'text', 'struct'])
    def test_group_help(self, runner, group):
        result = runner.invoke(cli, [group, '--help'])
        assert result.exit_code == 0