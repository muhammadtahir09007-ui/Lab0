"""CLI for preprocessing operations using Click."""

import click
import json
from src import preprocessing


@click.group()
def cli():
    """Data Preprocessing CLI."""
    pass


# clean group

@cli.group()
def clean():
    """Data cleaning commands."""
    pass


@clean.command()
@click.argument("values")
def remove_missing(values):
    """Remove missing values from a JSON list."""
    data = json.loads(values)
    result = preprocessing.remove_missing_values(data)
    click.echo(f"Result: {result}")


@clean.command()
@click.argument("values")
@click.option("--fill", default=0, type=float, help="Fill value (default 0)")
def fill_missing(values, fill):
    """Fill missing values in a JSON list."""
    data = json.loads(values)
    result = preprocessing.fill_missing_values(data, fill_value=fill)
    click.echo(f"Result: {result}")


@clean.command()
@click.argument("values")
def deduplicate(values):
    """Remove duplicates from a JSON list."""
    data = json.loads(values)
    result = preprocessing.remove_duplicates(data)
    click.echo(f"Result: {result}")


# numeric group

@cli.group()
def numeric():
    """Numerical transformation commands."""
    pass


@numeric.command()
@click.argument("values")
@click.option("--min", "new_min", default=0.0, type=float)
@click.option("--max", "new_max", default=1.0, type=float)
def norm(values, new_min, new_max):
    """Normalize values (min-max scaling)."""
    data = json.loads(values)
    result = preprocessing.normalize(data, new_min=new_min, new_max=new_max)
    click.echo(f"Result: {result}")


@numeric.command()
@click.argument("values")
def stand(values):
    """Standardize values (z-score)."""
    data = json.loads(values)
    result = preprocessing.standardize(data)
    click.echo(f"Result: {result}")


@numeric.command()
@click.argument("values")
@click.option("--min", "min_val", default=0, type=float)
@click.option("--max", "max_val", default=1, type=float)
def clp(values, min_val, max_val):
    """Clip values to a range."""
    data = json.loads(values)
    result = preprocessing.clip(data, min_value=min_val, max_value=max_val)
    click.echo(f"Result: {result}")


@numeric.command()
@click.argument("values")
def to_int(values):
    """Convert strings to integers."""
    data = json.loads(values)
    result = preprocessing.convert_to_integers(data)
    click.echo(f"Result: {result}")


@numeric.command()
@click.argument("values")
def log(values):
    """Log transform (positive values only)."""
    data = json.loads(values)
    result = preprocessing.log_transform(data)
    click.echo(f"Result: {result}")


# text group

@cli.group()
def text():
    """Text processing commands."""
    pass


@text.command()
@click.argument("input_text")
def tokenize(input_text):
    """Tokenize text into words."""
    result = preprocessing.tokenize(input_text)
    click.echo(f"Result: {result}")


@text.command()
@click.argument("input_text")
def alphanumeric(input_text):
    """Keep only alphanumeric characters and spaces."""
    result = preprocessing.select_alphanumeric(input_text)
    click.echo(f"Result: {result}")


@text.command()
@click.argument("input_text")
@click.option("--stops", required=True, help="Comma-separated stop words")
def remove_stops(input_text, stops):
    """Remove stop words from text."""
    stop_words = [w.strip() for w in stops.split(",")]
    result = preprocessing.remove_stop_words(input_text, stop_words)
    click.echo(f"Result: {result}")


# struct group

@cli.group()
def struct():
    """Structural operation commands."""
    pass


@struct.command()
@click.argument("values")
def flat(values):
    """Flatten a nested list."""
    data = json.loads(values)
    result = preprocessing.flatten(data)
    click.echo(f"Result: {result}")


@struct.command()
@click.argument("values")
@click.option("--seed", default=None, type=int, help="Random seed")
def shuffle(values, seed):
    """Shuffle values randomly."""
    data = json.loads(values)
    result = preprocessing.shuffle_values(data, seed=seed)
    click.echo(f"Result: {result}")


if __name__ == "__main__":
    cli()
