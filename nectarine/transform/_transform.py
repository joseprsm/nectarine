import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from .._config import Config
from .features import CategoryEncoder, IDEncoder, NumberEncoder


_ENCODER_MAPPING = {
    "category": CategoryEncoder,
    "number": NumberEncoder,
    "id": IDEncoder,
}


class Transform(ColumnTransformer):
    _feature_index: dict[str, list[int]]

    def __init__(self, config: str | dict, target: str):
        self.target = target
        self.config = Config(config) if isinstance(config, str) else config
        self.dataset = self.get_dataset(target)
        self.schema = self.dataset.pop("schema")
        super().__init__(transformers=[], remainder="drop")

    def __enter__(self):
        # todo: add connection to feature store
        return self

    def __call__(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(X).transform(X)

    def __exit__(self, *_):
        pass

    def fit(self, X: pd.DataFrame):
        feature_index = self._get_feature_indices(self.schema, X.columns.values)

        if self._has_references():
            _ = feature_index.pop("id")
            feature_index["id"] = []
            self.remainder = "passthrough"

        self.transformers = self._get_transformers(feature_index)
        return super().fit(X.values)

    def transform(self, X) -> tuple[np.ndarray, np.ndarray]:
        transformed: np.ndarray = super().transform(X.values)

        if self._has_references():
            # assumes it's the last transformation made
            encoded_id = transformed[:, [-1]]
            transformed = np.delete(transformed, -1, axis=1)
            return encoded_id, transformed

        return transformed[:, -2:], transformed[:, :-2]

    @staticmethod
    def _get_transformers(index_dict: dict[str, list]) -> list[tuple]:
        return [
            tuple(_ENCODER_MAPPING[feature_type](index))
            for feature_type, index in index_dict.items()
            if len(index) > 0
        ]

    @staticmethod
    def _get_feature_indices(
        schema: dict[str, str], header: np.ndarray
    ) -> dict[str, list[int]]:
        idx = {feature_type: [] for feature_type in _ENCODER_MAPPING.keys()}
        for feature, feature_type in schema.items():
            idx[feature_type] += np.where(header == feature)[0].tolist()
        return idx

    def _has_references(self):
        return "references" in self.dataset.keys()

    def get_dataset(self, target: str = None) -> dict:
        def check_name(x):
            return x["name"] == target

        return next(filter(check_name, self.config["transform"]["datasets"])).copy()
