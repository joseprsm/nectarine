import json
import pickle

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
    interactions: str,
    users: str,
    items: str,
    schema: str,
    transform_layer: str = None,
    model_config: str = None,
):
    x = pd.read_csv(interactions)

    with open(schema, "r") as f:
        schema = json.load(f)

    x, lookup_layer, model_config = Extractor(schema, users, items)(x)

    with open(transform_layer, "wb") as lookup_fp:
        pickle.dump(lookup_layer, lookup_fp)

    with open(model_config, "w") as config_fp:
        json.dump(model_config, config_fp)


if __name__ == "__main__":
    transform()
