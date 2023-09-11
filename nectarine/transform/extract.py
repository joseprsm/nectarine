import json
import os
import pickle

import pandas as pd

from .output import Transform
from .transform import FeatureTransformer


class Extractor:
    def __init__(self, schema: dict, users: str, items: str):
        self._schema = schema
        self._user_path = users
        self._item_path = items

    def __call__(self, X: pd.DataFrame) -> tuple[pd.DataFrame, Transform, dict]:
        def fit_transformer(target: str):
            input_path = getattr(self, f"_{target}_path")
            transformer = FeatureTransformer(self._schema[target])
            inputs = pd.read_csv(input_path)
            ids, features = transformer.fit(inputs).transform(inputs)
            outputs = pd.DataFrame(features, ids.reshape(-1).astype(int))
            return transformer, outputs.sort_index().values

        user_transformer, users = fit_transformer("user")
        item_transformer, items = fit_transformer("item")
        transform_output = Transform(users, items)

        def encode(x_):
            x_ = user_transformer.encode(x_)
            x_ = item_transformer.encode(x_)
            return x_

        x = X.copy(deep=True)
        x = encode(x)

        return ExtractOutput(x, transform_output)


class ExtractOutput:
    def __init__(self, transformed_data: pd.DataFrame, transform_module: Transform):
        self._transformed_data = transformed_data
        self._transform_module = transform_module
        self._model_config = self._get_model_config()

    def save(self, path: str):
        output_data_path = os.path.join(path, "output_data.csv")
        self._transformed_data.to_csv(output_data_path, index=None)

        transform_module_path = os.path.join(path, "transform_module")
        with open(transform_module_path, "wb") as lookup_fp:
            pickle.dump(self._transform_module, lookup_fp)

        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as config_fp:
            json.dump(self._model_config, config_fp)

    def _get_model_config(self) -> dict:
        def get_dims(target: str):
            return getattr(self._transform_module, target).shape[0]

        return {
            "query": {"n_dims": get_dims("users")},
            "candidate": {"n_dims": get_dims("items")},
        }

    @property
    def transformed_data(self):
        return self._transformed_data

    @property
    def transform_module(self):
        return self._transform_module
