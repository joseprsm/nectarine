import click
import pandas as pd
import yaml

from nectarine import Transform


@click.command()
@click.option("--data", required=True)
@click.option("--schema", required=True)
@click.option("--target")
def transform(data: str, schema: str, target: str):
    with open(schema, "r") as s:
        schema = yaml.load(s)

    with Transform(schema, target) as trf:
        data: pd.DataFrame = trf.download(data)
        features = trf(data)
        features.upload(target)


if __name__ == "__main__":
    transform()
