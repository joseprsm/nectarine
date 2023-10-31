import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from .features import CategoryEncoder, IDEncoder, NumberEncoder


_ENCODER_MAPPING = {
    "category": CategoryEncoder,
    "number": NumberEncoder,
    "id": IDEncoder,
}


class Transform(ColumnTransformer):
    _feature_index: dict[str, list[int]]

    def __init__(self, config: dict, target: str):
        self.target = target
        self.config = config
        self.schema = self.config["datasets"][target]["schema"]
        super().__init__(transformers=[], remainder="drop")

    def __enter__(self):
        # todo: add connection to feature store
        return self

    def __call__(self, X: pd.DataFrame):
        return self.fit(X).transform(X)

    def __exit__(self, *_):
        pass

    def fit(self, X: pd.DataFrame):
        def get_feature_indices(schema: dict[str, str]) -> dict[str, list[int]]:
            header = X.columns.to_list()
            idx = {feature_type: [] for feature_type in _ENCODER_MAPPING.keys()}
            for feature, feature_type in schema.items():
                mask = np.array(header) == feature
                idx[feature_type] += np.where(mask)[0].tolist()
            return idx

        def get_transformers(index_dict: dict[str, list]):
            return [
                tuple(_ENCODER_MAPPING[feature_type](index))
                for feature_type, index in index_dict.items()
                if len(index) > 0
            ]

        self._feature_index = get_feature_indices(self.schema)
        self.transformers = get_transformers(self._feature_index)
        return super().fit(X.values)

    def transform(self, X):
        transformed: np.ndarray = super().transform(X.values)
        encoded_id = transformed[:, [-1]]  # assumes it's the last transformation made
        transformed = np.delete(transformed, -1, axis=1)
        return encoded_id, transformed
