import pandas as pd

from .output import TransformOutput
from .transform import FeatureTransformer


class Extractor:
    def __init__(self, schema: dict, users: str, items: str):
        self._schema = schema
        self._user_path = users
        self._item_path = items

    def __call__(self, X: pd.DataFrame) -> tuple[pd.DataFrame, TransformOutput, dict]:
        def fit_transformer(target: str):
            input_path = getattr(self, f"_{target}_path")
            transformer = FeatureTransformer(self._schema[target])
            inputs = pd.read_csv(input_path)
            ids, features = transformer.fit(inputs).transform(inputs)
            outputs = pd.DataFrame(features, ids.reshape(-1).astype(int))
            return transformer, outputs.sort_index().values

        user_transformer, users = fit_transformer("user")
        item_transformer, items = fit_transformer("item")
        transform_output = TransformOutput(users, items)

        def encode(x_):
            x_ = user_transformer.encode(x_)
            x_ = item_transformer.encode(x_)
            return x_

        x = X.copy(deep=True)
        x = encode(x)

        model_config = {
            "query": {"n_dims": users.shape[0]},
            "candidate": {"n_dims": items.shape[0]},
        }

        return x, transform_output, model_config
