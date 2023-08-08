import pandas as pd
from flax import linen as nn

from .output import TransformOutput
from .transform import FeatureTransformer


class Extractor:
    def __init__(self, schema: dict, users: str, items: str):
        self._schema = schema
        self._user_path = users
        self._item_path = items

    def __call__(self, X: pd.DataFrame) -> nn.Module:
        def fit_transformer(target: str):
            input_path = getattr(self, f"_{target}_path")
            transformer = FeatureTransformer(self._schema[target])
            inputs = pd.read_csv(input_path)
            ids, features = transformer.fit(inputs).transform(inputs)
            outputs = pd.DataFrame(features, ids.reshape(-1).astype(int))
            return transformer, outputs

        user_transformer, users = fit_transformer("user")
        item_transformer, items = fit_transformer("item")
        transform_output = TransformOutput(users, items)

        def encode(x_):
            x_ = user_transformer.encode(x_)
            x_ = item_transformer.encode(x_)
            return x_

        x = X.copy(deep=True)
        x = encode(x)

        model_config = {}

        return x, transform_output, model_config