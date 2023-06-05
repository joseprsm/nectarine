from sklearn.base import BaseEstimator, TransformerMixin

from xtrax.data import Dataset, Users, Items
from xtrax.transform import FeatureTransformer, EventTransformer


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, schema: dict, users: str, items: str):
        super().__init__()
        self._schema = schema
        self._users = users
        self._items = items

        self._user_transformer = FeatureTransformer(schema, "user")
        self._item_transformer = FeatureTransformer(schema, "item")
        self._event_transformer = EventTransformer(
            schema, self._timestamp, self._window_size
        )

    def fit(self, X, **fit_params):
        def fit_transformer(inputs: Dataset, transformer: FeatureTransformer):
            input_attr: str = getattr(self, f"_{inputs.__name__.lower()}")
            x = inputs.load(input_attr, schema=self._schema)
            transformer.fit(x).transform(x)

        fit_transformer(Users)
        fit_transformer(Items)

        X = self._encode(self._user_transformer)(X)
        _ = self._encode(self._item_transformer)(X)

        return self

    def transform(self, X):
        interactions = self._encode(self._user_transformer)(X)
        interactions = self._encode(self._item_transformer, interactions)
        interactions = self._event_transformer.transform(interactions)
        return interactions

    def _encode(transformer):
        def _encode(data):
            encoder, index = transformer.encoder
            data[:, index] = encoder.transform(data[:, index])
            return data

        return _encode
