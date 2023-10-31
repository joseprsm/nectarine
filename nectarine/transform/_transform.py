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
    def __init__(self, config: dict, target: str):
        self.target = target
        self.config = config
        self.schema = self.config["datasets"][target]["schema"]
        super().__init__(transformers=[], remainder="drop")

    def __enter__(self):
        return self

    def __call__(self, X: pd.DataFrame):
        return self.fit(X).transform(X)

    def __exit__(self, *_):
        pass

    def fit(self, X: pd.DataFrame):
        header = X.columns.to_list()

        def get_feature_indices(schema: dict[str, str], target: str):
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

        feature_index = get_feature_indices(self.schema, self.target)
        self.transformers = get_transformers(feature_index)
        return super().fit(X)
