import click
import pandas as pd
import yaml

from .._util import get_dataset
from ..transform import Transform


@click.command()
@click.option("--data", required=True)
@click.option("--config", required=True)
@click.option("--target")
def transform(data: str, config: str, target: str):
    with open(config, "r") as s:
        config = yaml.safe_load(s)

    with Transform(config, target) as transform:
        dataset: dict = get_dataset(config, "transform", target)
        read: callable = getattr(pd, f"read_{dataset['format']}")
        data: pd.DataFrame = read(data)
        ids, features = transform(data)


if __name__ == "__main__":
    transform()
