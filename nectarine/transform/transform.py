import numpy as np
import pandas as pd

from .base import BaseTransformer
from .features import CategoryEncoder, IDEncoder, NumberEncoder


_ENCODER_MAPPING = {
    "category": CategoryEncoder,
    "number": NumberEncoder,
    "id": IDEncoder,
}


class FeatureTransformer(BaseTransformer):
    def encode(self, X, header: list[str]):
        def check_id_encoder(x):
            return x[0] == IDEncoder.__name__

        def get_idx():
            return [self._get_feature_indexes(self._schema, header)["id"][0]]

        X = X.values if isinstance(X, pd.DataFrame) else X
        transformer = list(filter(check_id_encoder, self.transformers_))[0][1]
        return transformer.transform(X[:, get_idx()])

    def fit(self, X):
        x = X.values if isinstance(X, pd.DataFrame) else X
        self.transformers = (
            self._get_transformers(self._schema, self._header)
            if len(self.transformers) == 0
            else self.transformers
        )
        return super().fit(x)

    def transform(self, X):
        x = X.values if isinstance(X, pd.DataFrame) else X
        x = super().transform(x)
        return x[:, [0]], x[:, 1:]

    @classmethod
    def _get_transformers(cls, schema: dict[str, str], header: list[str] = None):
        if header:
            feature_indexes = cls._get_feature_indexes(schema, header)
            return [
                tuple(_ENCODER_MAPPING[feature_type](idx))
                for feature_type, idx in feature_indexes.items()
                if len(idx) > 0
            ]
        return []

    @staticmethod
    def _get_feature_indexes(schema: dict[str, str], header: list[str]):
        feature_indexes = {feature_type: [] for feature_type in _ENCODER_MAPPING.keys()}
        for feature, feature_type in schema.items():
            mask = np.array(header) == feature
            feature_indexes[feature_type] += np.where(mask)[0].tolist()
        return feature_indexes
