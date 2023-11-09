import argparse

import pandas as pd

from ..transform import Transform


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--target", "-t", required=True)
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = get_args()
    config, target, output = args.values()

    with Transform(config, target) as transform:
        dataset: dict = transform.get_dataset()
        read: callable = getattr(pd, f"read_{dataset['format']}")
        data: pd.DataFrame = read(dataset["location"])
        transform(data)
