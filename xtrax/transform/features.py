from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline

from .min_max import MinMaxScaler
from .one_hot import OneHotEncoder


class BaseEncoder:
    def __init__(self, transformer, feature_index: list[int], name: str = None):
        self._name = name or self.__class__.__name__
        self._transformer = transformer
        self._feature_index = feature_index

    def __iter__(self):
        for x in [self._name, self._transformer, self._feature_index]:
            yield x


class CategoryEncoder(BaseEncoder):
    def __init__(self, feature_index: list[int], name: str = None):
        pipeline = make_pipeline(OrdinalEncoder(), OneHotEncoder())
        return super().__init__(pipeline, feature_index, name)


class NumberEncoder(BaseEncoder):
    def __init__(self, feature_index: list[int], name: str = None):
        transformer = MinMaxScaler(feature_range=(-1, 1))
        return super().__init__(transformer, feature_index, name)


class IDEncoder(BaseEncoder):
    def __init__(cls, feature_index: list[int], name: str = None):
        transformer = OrdinalEncoder()
        return super().__init__(transformer, feature_index, name)
