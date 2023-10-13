import click
import pandas as pd
import yaml

from nectarine import Recommender, Train


@click.command()
@click.option("--config", "config")
def train(config: str):
    with open(config, "r") as c:
        config = yaml.load(c)

    with Train(config) as trn:
        data: tuple[pd.DataFrame] = trn.download(split=True)
        model: Recommender = trn(*data)
        model.save(config["output"]["path"])
