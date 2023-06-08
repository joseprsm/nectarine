import numpy as np

from sklearn.compose import ColumnTransformer

from .features import CategoryEncoder, IDEncoder, NumberEncoder

_ENCODER_MAPPING = {
    "category": CategoryEncoder,
    "number": NumberEncoder,
    "id": IDEncoder,
}


class FeatureTransformer(ColumnTransformer):
    def __init__(self, schema: dict, header: list[str] = None, remainder: str = "drop"):
        self._schema = schema
        self._header = header
        transformers = self._get_transformers(self._schema, self._header)
        super().__init__(transformers=transformers, remainder=remainder)

    def fit(self, X, header: list[str] = None):
        self.transformers = (
            self._get_transformers(self._schema, self._header)
            if len(self.transformers) == 0
            else self.transformers
        )
        return super().fit(X)

    def encode(self, X):
        def check_id_encoder(x):
            return x[0] == IDEncoder.__name__

        return ColumnTransformer(
            transformers=list(filter(check_id_encoder, self.transformers))[0][1]
        ).transform(X)

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

    @property
    def header(self):
        return self._header

    @property
    def schema(self):
        return self._schema
