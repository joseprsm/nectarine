import json
import os
import string
import tempfile

import numpy as np
import pandas as pd

from nectarine.transform.__main__ import transform


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
    interactions_path = os.path.join(dirname, "interactions.csv")
    interactions.to_csv(interactions_path, index=None)

    schema = {
        "user": {"user_id": "id", "age": "number", "gender": "category"},
        "item": {"item_id": "id"},
    }
    schema_path = os.path.join(dirname, "schema.json")
    with open(schema_path, "w") as fp:
        json.dump(schema, fp)

    return interactions_path, users_path, items_path, schema_path


def test_transform():
    with tempfile.TemporaryDirectory() as tmpdirname:
        interactions_path, users_path, items_path, schema_path = create_data(tmpdirname)
        output_dir = os.path.join(tmpdirname, "outputs")
        os.makedirs(output_dir)
        transform.callback(
            interactions_path,
            users_path,
            items_path,
            schema_path,
            encoded=os.path.join(output_dir, "encoded.csv"),
            model_config=os.path.join(output_dir, "config.json"),
            transform_layer=os.path.join(output_dir, "transform"),
        )
        outputs = os.listdir(output_dir)
        assert "encoded.csv" in outputs
        assert "config.json" in outputs
        assert "transform" in outputs
