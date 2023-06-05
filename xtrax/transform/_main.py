import numpy as np

from sklearn.compose import ColumnTransformer

from .features import CategoryEncoder, IDEncoder, NumberEncoder

_ENCODER_MAPPING = {
    "category": CategoryEncoder,
    "number": NumberEncoder,
    "id": IDEncoder,
}


class FeatureTransformer(ColumnTransformer):
    def __init__(self, schema: dict[str, dict], target: str, header: list[str] = None):
        self._schema: dict[str, str] = schema[target]
        self._header = header
        transformers = self._get_transformers(schema, header) if header else []
        super().__init__(transformers=transformers)

    def fit(self, X):
        self.transformers = (
            self._get_transformers(self._schema, self._header)
            if len(self.transformers) == 0
            else self.transformers
        )
        return super().fit(X)

    @classmethod
    def _get_transformers(cls, schema: dict[str, str], header: list[str]):
        feature_indexes = cls._get_feature_indexes(schema, header)
        return [
            _ENCODER_MAPPING[feature_type](idx)
            for feature_type, idx in feature_indexes.items()
        ]

    @staticmethod
    def _get_feature_indexes(schema: dict[str, str], header: list[str]):
        feature_indexes = {feature_type: [] for feature_type in _ENCODER_MAPPING.keys()}
        for feature, feature_type in schema.items():
            mask = np.array(header) == feature
            feature_indexes[feature_type] += np.where(mask)[0].tolist()
        return feature_indexes
