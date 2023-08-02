import pandas as pd

from .transform import FeatureTransformer


class Extractor:
    def __init__(self, schema: dict, users: str, items: str):
        self._schema = schema
        self._users = users
        self._items = items

        self._user_transformer = FeatureTransformer(self._schema["user"])
        self._item_transformer = FeatureTransformer(self._schema["item"])

    def fit(self, X):
        def fit_transformer(target: str):
            input_path = getattr(self, f"_{target}")
            transformer = getattr(self, f"_{target[:-1]}_transformer")
            x = pd.read_csv(input_path)
            _ = transformer.fit(x)

        fit_transformer("users")
        fit_transformer("items")

        return self

    def transform(self, X):
        x = self._user_transformer.encode(X)
        x = self._item_transformer.encode(x)
        return x
