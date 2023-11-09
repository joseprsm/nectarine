import click
import pandas as pd

from ..transform import Transform


@click.command()
@click.option("--config", required=True)
@click.option("--target")
def transform(config: str, target: str):
    with Transform(config, target) as transform:
        dataset: dict = transform.get_dataset(target=target)
        read: callable = getattr(pd, f"read_{dataset['format']}")
        data: pd.DataFrame = read(dataset["location"])
        transform(data)


if __name__ == "__main__":
    transform()
