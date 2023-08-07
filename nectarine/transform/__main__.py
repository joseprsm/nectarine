import json

import click
import pandas as pd

from nectarine import Extractor


@click.command()
@click.option("--users-path", "--users", "users", required=True)
@click.option("--items-path", "--items", "items", required=True)
@click.option("--schema-path", "--schema", "schema", required=True)
@click.option("--interactions-path", "--interactions", "interactions", required=True)
@click.option("--transform-layer", "interactions")
@click.option("--model-config", "interactions")
def transform(
    interactions,
    users,
    items,
    schema,
    transform_layer: str = None,
    model_config: str = None,
):
    with open(schema, "r") as f:
        schema = json.load(f)
    interactions = pd.read_csv(interactions)
    extractor = Extractor(schema, users, items)
    features = extractor.fit(interactions).transform(interactions)  # noqa: F841


if __name__ == "__main__":
    transform()
