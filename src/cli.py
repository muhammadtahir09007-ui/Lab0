"""
CLI for data preprocessing operations.
Uses Click to organize commands into four groups:
  - clean   : Data cleaning operations
  - numeric : Numerical transformations
  - text    : Text processing operations
  - struct  : Structural operations
"""

import click
import json
from src import preprocessing

# ============================================================
# MAIN CLI GROUP
# ============================================================


@click.group()
def cli():
    """Data Preprocessing CLI - A tool for common data preprocessing tasks."""
    pass


# ============================================================
# CLEAN GROUP — Data Cleaning
# ============================================================


@cli.group()
def clean():
    """Data cleaning operations (remove/fill missing values, remove duplicates)."""
    pass


@clean.command()
@click.argument("values")
def remove_missing(values):
    """Remove missing values from a JSON list.

    Example: uv run python -m src.cli clean remove-missing '[1, null, 3, "", 5]'
    """
    data = json.loads(values)
    result = preprocessing.remove_missing_values(data)
    click.echo(f"Result: {result}")


@clean.command()
@click.argument("values")
@click.option(
    "--fill",
    default=0,
    type=float,
    help="Value to fill missing entries with. Default: 0",
)
def fill_missing(values, fill):
    """Replace missing values in a JSON list with a fill value.

    Example: uv run python -m src.cli clean fill-missing '[1, null, 3, ""]' --fill 99
    """
    data = json.loads(values)
    result = preprocessing.fill_missing_values(data, fill_value=fill)
    click.echo(f"Result: {result}")


@clean.command()
@click.argument("values")
def deduplicate(values):
    """Remove duplicate values from a JSON list.

    Example: uv run python -m src.cli clean deduplicate '[1, 2, 2, 3, 1, 4]'
    """
    data = json.loads(values)
    result = preprocessing.remove_duplicates(data)
    click.echo(f"Result: {result}")


# ============================================================
# NUMERIC GROUP — Numerical Transformations
# ============================================================


@cli.group()
def numeric():
    """Numerical transformation operations (normalize, standardize, clip, etc.)."""
    pass


@numeric.command()
@click.argument("values")
@click.option(
    "--min", "new_min", default=0.0, type=float, help="New minimum value. Default: 0.0"
)
@click.option(
    "--max", "new_max", default=1.0, type=float, help="New maximum value. Default: 1.0"
)
def norm(values, new_min, new_max):
    """Normalize values using min-max scaling.

    Example: uv run python -m src.cli numeric norm '[1, 2, 3, 4, 5]'
    """
    data = json.loads(values)
    result = preprocessing.normalize(data, new_min=new_min, new_max=new_max)
    click.echo(f"Result: {result}")


@numeric.command()
@click.argument("values")
def stand(values):
    """Standardize values using z-score method.

    Example: uv run python -m src.cli numeric stand '[10, 20, 30]'
    """
    data = json.loads(values)
    result = preprocessing.standardize(data)
    click.echo(f"Result: {result}")


@numeric.command()
@click.argument("values")
@click.option(
    "--min", "min_val", default=0, type=float, help="Minimum clip value. Default: 0"
)
@click.option(
    "--max", "max_val", default=1, type=float, help="Maximum clip value. Default: 1"
)
def clp(values, min_val, max_val):
    """Clip values to a specified range.

    Example: uv run python -m src.cli numeric clp '[-2, 0.5, 3]' --min 0 --max 1
    """
    data = json.loads(values)
    result = preprocessing.clip(data, min_value=min_val, max_value=max_val)
    click.echo(f"Result: {result}")


@numeric.command()
@click.argument("values")
def to_int(values):
    """Convert string values to integers (skip non-numeric).

    Example: uv run python -m src.cli numeric to-int '["1", "hello", "3"]'
    """
    data = json.loads(values)
    result = preprocessing.convert_to_integers(data)
    click.echo(f"Result: {result}")


@numeric.command()
@click.argument("values")
def log(values):
    """Apply natural log transform (positive values only).

    Example: uv run python -m src.cli numeric log '[1, 10, 100]'
    """
    data = json.loads(values)
    result = preprocessing.log_transform(data)
    click.echo(f"Result: {result}")


# ============================================================
# TEXT GROUP — Text Processing
# ============================================================


@cli.group()
def text():
    """Text processing operations (tokenize, filter, remove stop words)."""
    pass


@text.command()
@click.argument("input_text")
def tokenize(input_text):
    """Tokenize text into lowercase alphanumeric words.

    Example: uv run python -m src.cli text tokenize "Hello, World! Test 123"
    """
    result = preprocessing.tokenize(input_text)
    click.echo(f"Result: {result}")


@text.command()
@click.argument("input_text")
def alphanumeric(input_text):
    """Keep only alphanumeric characters and spaces.

    Example: uv run python -m src.cli text alphanumeric "Hello, World! #2026"
    """
    result = preprocessing.select_alphanumeric(input_text)
    click.echo(f"Result: {result}")


@text.command()
@click.argument("input_text")
@click.option(
    "--stops",
    required=True,
    help='Comma-separated stop words. Example: --stops "the,is,a"',
)
def remove_stops(input_text, stops):
    """Remove stop words from text.

    Example: uv run python -m src.cli text remove-stops "this is a test" --stops "this,is,a"
    """
    stop_words = [word.strip() for word in stops.split(",")]
    result = preprocessing.remove_stop_words(input_text, stop_words)
    click.echo(f"Result: {result}")


# ============================================================
# STRUCT GROUP — Structural Operations
# ============================================================


@cli.group()
def struct():
    """Structural operations (flatten lists, shuffle values)."""
    pass


@struct.command()
@click.argument("values")
def flat(values):
    """Flatten a list of lists into a single list.

    Example: uv run python -m src.cli struct flat '[[1, 2], [3, 4], [5]]'
    """
    data = json.loads(values)
    result = preprocessing.flatten(data)
    click.echo(f"Result: {result}")


@struct.command()
@click.argument("values")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility.")
def shuffle(values, seed):
    """Shuffle values randomly.

    Example: uv run python -m src.cli struct shuffle '[1, 2, 3, 4, 5]' --seed 42
    """
    data = json.loads(values)
    result = preprocessing.shuffle_values(data, seed=seed)
    click.echo(f"Result: {result}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    cli()
