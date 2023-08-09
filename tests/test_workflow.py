import os
import string

import jax
import numpy as np
import pandas as pd

from nectarine import Extractor, Recommender
from nectarine.train.__main__ import (
    RNG,
    TEST_SIZE,
    create_train_state,
    train,
    train_and_evaluate,
)
from nectarine.transform.__main__ import transform


transform = transform.callback
train = train.callback

schema = {
    "user": {"user_id": "id", "age": "number", "gender": "category"},
    "item": {"item_id": "id"},
}


def create_data(dirname: str = "data"):
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


def test_workflow():
    interactions, users_path, items_path = create_data("data")
    df, transform_layer, model_config = Extractor(schema, users_path, items_path)(
        interactions
    )

    df = df.sample(frac=1, random_state=1).astype(int)
    cutoff = np.floor(df.shape[0] * TEST_SIZE).astype(int)
    _, train_data = df.iloc[:cutoff], df.iloc[cutoff:]

    model = Recommender(config=model_config, transform=transform_layer)

    rng, init_rng = jax.random.split(RNG)
    state = create_train_state(model, init_rng)
    state = train_and_evaluate(model, state, train_data, rng, epochs=1)
