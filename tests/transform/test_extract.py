import os
import string
import tempfile

import numpy as np
import pandas as pd

from nectarine import Extractor


schema = {
    "user": {"user_id": "id", "age": "number", "gender": "category"},
    "item": {"item_id": "id"},
}


def create_data(dirname):
    users = pd.DataFrame(
        {
            "age": np.random.choice(100, size=1000),
            "user_id": np.random.choice(list(string.ascii_lowercase), size=1000),
            "gender": np.random.choice(["m", "f"], size=1000),
        }
    )
    users_path = os.path.join(dirname, "users.csv")
    users.to_csv(users_path, index=None)

    items = pd.DataFrame({"item_id": np.arange(1, 101)})
    items_path = os.path.join(dirname, "items.csv")
    items.to_csv(items_path, index=None)

    interactions = pd.DataFrame(
        {
            "user_id": np.random.choice(users.user_id, size=1000),
            "item_id": np.random.choice(items.item_id, size=1000),
        }
    )

    return interactions, users_path, items_path


def test_fit():
    with tempfile.TemporaryDirectory() as tmpdirname:
        interactions, users_path, items_path = create_data(tmpdirname)
        extractor = Extractor(schema, users_path, items_path)
        extractor.fit(interactions)


def test_transform():
    with tempfile.TemporaryDirectory() as tmpdirname:
        interactions, users_path, items_path = create_data(tmpdirname)
        extractor = Extractor(schema, users_path, items_path)
        y_hat = extractor.fit(interactions).transform(interactions)
    assert interactions.shape[0] == y_hat.shape[0]
