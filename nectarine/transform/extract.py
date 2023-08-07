import jax.numpy as jnp
import pandas as pd
from flax import linen as nn

from .transform import FeatureTransformer


class _LookupModule(nn.Module):
    data: jnp.ndarray

    @nn.compact
    def __call__(self, inputs: int | jnp.ndarray):
        return jnp.take(self.data, inputs, axis=0)


class Extractor:
    def __init__(self, schema: dict, users: str, items: str):
        self._schema = schema
        self._user_path = users
        self._item_path = items

    def __call__(self, X: pd.DataFrame) -> nn.Module:
        user_transformer, user_lookup = self._get_fitted_transformer("user")
        item_transformer, item_lookup = self._get_fitted_transformer("item")

        def encode(x_):
            x_ = user_transformer.encode(x_)
            x_ = item_transformer.encode(x_)
            return x_

        x = X.copy(deep=True)
        x = encode(x)

        return x, user_lookup, item_lookup

    def _get_fitted_transformer(self, target: str):
        input_path = getattr(self, f"_{target}_path")
        transformer = FeatureTransformer(self._schema[target])
        inputs = pd.read_csv(input_path)
        ids, features = transformer.fit(inputs).transform(inputs)
        outputs = pd.DataFrame(features, ids.reshape(-1).astype(int))
        lookup = _LookupModule(outputs.sort_index().values)
        return transformer, lookup
