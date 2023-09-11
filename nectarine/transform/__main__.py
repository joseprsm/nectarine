import json

import click
import pandas as pd

from nectarine import Extractor


@click.command()
@click.option("--users-path", "--users", "users", required=True)
@click.option("--items-path", "--items", "items", required=True)
@click.option("--schema-path", "--schema", "schema", required=True)
@click.option("--interactions-path", "--interactions", "interactions", required=True)
@click.option("--output-path", "--outputs", "outputs", default="outputs")
def transform(
    interactions: str,
    users: str,
    items: str,
    schema: str,
    outputs: str = None,
):
    x: pd.DataFrame = pd.read_csv(interactions)

    with open(schema, "r") as f:
        schema = json.load(f)

    output = Extractor(schema, users, items)(x)
    output.save(outputs)


if __name__ == "__main__":
    transform()
